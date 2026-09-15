"""
EdgeBot — a forecasting bot for the Metaculus FutureEval bot tournaments.

Built on top of the official Summer 2026 template, but replacing the single
news-then-guess pass with a four-stage chain that targets the specific ways
LLM forecasters are known to lose points:

  1. RESOLUTION READ   Pin down what literally must be true by the deadline.
                       A large share of Metaculus questions are lost on wording,
                       not on world-modelling.
  2. OUTSIDE VIEW      Establish a reference class and base rate BEFORE the news
                       is allowed to touch the estimate. Going straight to
                       headlines is what anchors a model on recency.
  3. INSIDE VIEW       Update the base rate with evidence, explicitly, in a
                       stated direction and magnitude.
  4. RED TEAM          Attack the forecast, then reconcile. Cheap, and it catches
                       the confident-and-wrong cases that a log score punishes
                       hardest.

Scoring note that drives the calibration language in the prompts: Metaculus
uses a log score, so a confident miss costs far more than a hedged one gains.
The prompts push against both classic failures — false precision at the tails
and reflexive 50% hedging.
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

import dotenv

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    AskNewsSearcher,
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

from main import SummerTemplateBot2026  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


CALIBRATION_RULES = """
    Calibration rules you must follow:
    - You are scored with a log score. A confident miss costs far more than a
      hedged hit gains. Never state 0% or 100%.
    - Do not use the tails (below 3% or above 97%) unless the outcome is
      effectively already determined by facts that have happened.
    - Equally, do not hide at 50%. 50% means you genuinely have no information.
      If you have any, move off it and say by how much.
    - Avoid round numbers as a reflex. If you land on exactly 50, 75 or 90,
      check whether that is the evidence talking or your own tidiness.
    - Most things that have not happened yet do not happen in a short window.
      The shorter the time to resolution, the harder the status quo pulls.
"""


# --------------------------------------------------------------------- #
# MODEL TIERS
#
# "free"     — OpenRouter zero-cost models. Used to prove the four-stage
#              chain end to end before the season starts, and whenever no
#              paid credits are available. Free models are heavily rate
#              limited, so the chain also runs at reduced depth here.
# "frontier" — the real configuration. Set EDGEBOT_TIER=frontier once LLM
#              credits land; nothing else needs to change.
# --------------------------------------------------------------------- #

TIER = os.getenv("EDGEBOT_TIER", "free").strip().lower()

FREE_REASONER = "openrouter/nex-agi/nex-n2.5-pro:free"
FREE_SMALL = "openrouter/nex-agi/nex-n2.5-mini:free"


def build_llm_config(tier: str) -> tuple[dict, int, int]:
    """Return (llms, research_reports_per_question, predictions_per_report)."""
    if tier == "frontier":
        return (
            {
                "default": GeneralLlm(
                    model=os.getenv(
                        "EDGEBOT_MODEL", "openrouter/anthropic/claude-opus-4.5"
                    ),
                    temperature=0.3,
                    timeout=120,
                    allowed_tries=2,
                ),
                "summarizer": os.getenv(
                    "EDGEBOT_SMALL_MODEL", "openrouter/openai/gpt-5-mini"
                ),
                "researcher": os.getenv(
                    "EDGEBOT_RESEARCH_MODEL",
                    "openrouter/perplexity/sonar-reasoning",
                ),
                "parser": os.getenv(
                    "EDGEBOT_SMALL_MODEL", "openrouter/openai/gpt-5-mini"
                ),
            },
            3,
            2,
        )
    # free tier: one research report, one forecast, still the full four stages
    return (
        {
            "default": GeneralLlm(
                model=FREE_REASONER,
                temperature=0.3,
                timeout=180,
                allowed_tries=3,
            ),
            "summarizer": FREE_SMALL,
            "researcher": FREE_REASONER,
            "parser": FREE_SMALL,
        },
        1,
        1,
    )


class EdgeBot(SummerTemplateBot2026):
    """Four-stage forecaster. Overrides research and the three main question types."""

    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    # ------------------------------------------------------------------ #
    # RESEARCH: two passes — current evidence, then historical frequency  #
    # ------------------------------------------------------------------ #

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            news_prompt = clean_indents(
                f"""
                You are a research assistant to a superforecaster. You do not forecast.

                Gather the most decision-relevant current information on this question.
                Prioritise: (a) facts that have already happened and cannot be undone,
                (b) scheduled events with dates before the resolution deadline,
                (c) statements by people with the actual power to cause or block the outcome.
                Explicitly flag anything that is speculation, rumour or opinion rather than fact.
                If the question would already resolve one way on today's information, say so plainly.

                Question:
                {question.question_text}

                Resolution criteria:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            base_rate_prompt = clean_indents(
                f"""
                You are a research assistant to a superforecaster. You do not forecast.

                Do NOT report current news. Instead establish the historical record:
                - What reference class does this event belong to?
                - How often has this kind of thing happened historically, per unit of time?
                - What is the longest and shortest this kind of process has taken?
                - What normally has to happen first, and does that usually happen on time?

                Give concrete numbers and dates where they exist. If the reference class is
                thin or ambiguous, say so and give the closest analogues instead of inventing
                a rate.

                Question:
                {question.question_text}

                Resolution criteria:
                {question.resolution_criteria}
                """
            )

            news = await self._invoke_researcher(news_prompt)
            base_rates = await self._invoke_researcher(base_rate_prompt)

            research = clean_indents(
                f"""
                === CURRENT EVIDENCE ===
                {news}

                === HISTORICAL RECORD AND BASE RATES ===
                {base_rates}
                """
            )
            logger.info(f"Research for {question.page_url}:\n{research}")
            return research

    async def _invoke_researcher(self, prompt: str) -> str:
        """Route a research prompt through whichever researcher is configured."""
        researcher = self.get_llm("researcher")
        try:
            if isinstance(researcher, GeneralLlm):
                return await researcher.invoke(prompt)
            if isinstance(researcher, str) and researcher.startswith("asknews/"):
                return await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            if isinstance(researcher, str) and researcher.startswith("smart-searcher"):
                searcher = SmartSearcher(
                    model=researcher.removeprefix("smart-searcher/"),
                    temperature=0,
                    num_searches_to_run=3,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                return await searcher.invoke(prompt)
            if not researcher or researcher in ("None", "no_research"):
                return ""
            return await self.get_llm("researcher", "llm").invoke(prompt)
        except Exception as exc:  # research must never kill a forecast
            logger.warning(f"Researcher failed, continuing without it: {exc}")
            return ""

    # ------------------------------------------------------------------ #
    # BINARY                                                             #
    # ------------------------------------------------------------------ #

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        today = datetime.now().strftime("%Y-%m-%d")
        llm = self.get_llm("default", "llm")

        # --- Stage 1 + 2: resolution read and outside view, news withheld ---
        outside_prompt = clean_indents(
            f"""
            You are a superforecaster. Today is {today}.
            You have NOT yet been shown any news. That is deliberate: establish the
            outside view first so that headlines cannot anchor you.

            Question: {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}

            {question.fine_print}

            Write, in order:
            (1) RESOLUTION READ. State in one sentence exactly what must be true, and by
                when, for this to resolve Yes. Name any wording in the criteria that is
                stricter or looser than the question title suggests.
            (2) HORIZON. How much time remains until resolution, and is that long or short
                relative to how long this kind of change normally takes?
            (3) REFERENCE CLASS. What class of events is this? How often do they occur?
            (4) BASE RATE. A single number: the probability you would give knowing only
                the reference class and the horizon, and nothing about current events.

            End with exactly: "Base rate: ZZ%"
            """
        )
        outside_view = await llm.invoke(outside_prompt)

        # --- Stage 3: inside view, now with evidence ---
        inside_prompt = clean_indents(
            f"""
            You are a superforecaster. Today is {today}.

            Question: {question.question_text}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}

            {question.fine_print}

            Your own prior analysis, written before you saw any evidence:
            {outside_view}

            Your research assistant reports:
            {research}

            Now update. Write:
            (a) STATUS QUO. What happens if nothing changes from today? Good forecasters
                weight this heavily, because the world changes slowly most of the time.
            (b) EVIDENCE THAT MOVES IT. Only facts, not speculation. For each, say which
                direction it pushes and roughly how hard.
            (c) NO SCENARIO. The most plausible path to No, in one or two sentences.
            (d) YES SCENARIO. The most plausible path to Yes, in one or two sentences.
            (e) UPDATE. State your base rate, then the updated probability, and justify
                the size of the move. A large move needs decisive, already-realised facts.

            {CALIBRATION_RULES}
            {self._get_conditional_disclaimer_if_necessary(question)}

            End with exactly: "Probability: ZZ%"
            """
        )
        inside_view = await llm.invoke(inside_prompt)

        # --- Stage 4: red team and reconcile ---
        final_reasoning = await self._red_team_and_reconcile(
            question_text=question.question_text,
            resolution_criteria=question.resolution_criteria,
            analysis=inside_view,
            today=today,
            answer_format='"Probability: ZZ%"',
            llm=llm,
        )

        full_reasoning = clean_indents(
            f"""
            ## Stage 1-2 — Resolution read and outside view
            {outside_view}

            ## Stage 3 — Inside view
            {inside_view}

            ## Stage 4 — Red team and final
            {final_reasoning}
            """
        )

        prediction: BinaryPrediction = await structure_output(
            final_reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.01, min(0.99, prediction.prediction_in_decimal))
        logger.info(f"Forecasted {question.page_url}: {decimal_pred}")
        return ReasonedPrediction(
            prediction_value=decimal_pred, reasoning=full_reasoning
        )

    # ------------------------------------------------------------------ #
    # MULTIPLE CHOICE                                                    #
    # ------------------------------------------------------------------ #

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        today = datetime.now().strftime("%Y-%m-%d")
        llm = self.get_llm("default", "llm")

        analysis_prompt = clean_indents(
            f"""
            You are a superforecaster. Today is {today}.

            Question: {question.question_text}
            Options: {question.options}

            Background:
            {question.background_info}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}

            {question.fine_print}

            Research:
            {research}

            Write:
            (a) RESOLUTION READ. What exactly decides which option is selected, and when.
            (b) STATUS QUO OPTION. Which option wins if nothing changes? Name it explicitly.
            (c) PER-OPTION CASE. For each option in turn, the strongest one-line case for it.
            (d) ELIMINATION. Which options are near-impossible given what has already happened,
                and why. Give them small but non-zero mass — surprises happen.
            (e) ALLOCATION. Assign probabilities. They must sum to 100%.

            {CALIBRATION_RULES}
            Additionally: leave real mass on unexpected outcomes. Multiple-choice questions
            are where overconfident bots lose the most, because the obvious option is already
            priced in and the surprise is not.
            {self._get_conditional_disclaimer_if_necessary(question)}

            End with the final probabilities, one option per line, as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            """
        )
        analysis = await llm.invoke(analysis_prompt)

        final_reasoning = await self._red_team_and_reconcile(
            question_text=question.question_text,
            resolution_criteria=question.resolution_criteria,
            analysis=analysis,
            today=today,
            answer_format=(
                "the full list of options with probabilities, one per line, "
                "summing to 100%"
            ),
            llm=llm,
        )

        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text may prepend options with some variation of "Option" which you should
            remove if it is not part of the option names given. Do not skip options with 0%
            probability — include them as an entry with 0%.
            """
        )
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=final_reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        logger.info(f"Forecasted {question.page_url}: {predicted_option_list}")
        return ReasonedPrediction(
            prediction_value=predicted_option_list,
            reasoning=f"## Analysis\n{analysis}\n\n## Red team and final\n{final_reasoning}",
        )

    # ------------------------------------------------------------------ #
    # NUMERIC                                                            #
    # ------------------------------------------------------------------ #

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        today = datetime.now().strftime("%Y-%m-%d")
        llm = self.get_llm("default", "llm")
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )

        analysis_prompt = clean_indents(
            f"""
            You are a superforecaster. Today is {today}.

            Question: {question.question_text}
            Units: {question.unit_of_measure}

            Background:
            {question.background_info}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}

            {question.fine_print}

            {lower_bound_message}
            {upper_bound_message}

            Research:
            {research}

            Write:
            (a) LAST KNOWN VALUE. The most recent actual measurement, with its date.
            (b) TREND. The recent rate of change, with numbers, and whether it is steady.
            (c) NAIVE EXTRAPOLATION. Where the trend alone lands by the resolution date.
            (d) LOW SCENARIO. What would have to happen for an unexpectedly low outcome.
            (e) HIGH SCENARIO. What would have to happen for an unexpectedly high outcome.
            (f) TAIL WIDTH. State explicitly that your 90% interval should be wide enough
                that you would be genuinely surprised to fall outside it. Under-wide tails
                are the most common and most expensive numeric error.

            {CALIBRATION_RULES}
            {self._get_conditional_disclaimer_if_necessary(question)}

            End with exactly these six lines, values strictly increasing, in {question.unit_of_measure}:
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            """
        )
        analysis = await llm.invoke(analysis_prompt)

        final_reasoning = await self._red_team_and_reconcile(
            question_text=question.question_text,
            resolution_criteria=question.resolution_criteria,
            analysis=analysis,
            today=today,
            answer_format=(
                "the six percentile lines (10/20/40/60/80/90) with values strictly "
                f"increasing, in {question.unit_of_measure}"
            ),
            llm=llm,
            extra_instruction=(
                "Pay particular attention to whether the 10th and 90th percentiles are "
                "too close together. Widen them if the analysis does not justify that "
                "level of confidence."
            ),
        )

        parsing_instructions = clean_indents(
            f"""
            The text is giving a forecast distribution for the numeric question:
            "{question.question_text}".
            - Give values in the correct units: {question.unit_of_measure}
            - Convert any scientific notation into regular numbers.
            - If percentiles are not explicitly given, indicate the answer is not present.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            final_reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(f"Forecasted {question.page_url}: {prediction.declared_percentiles}")
        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=f"## Analysis\n{analysis}\n\n## Red team and final\n{final_reasoning}",
        )

    # ------------------------------------------------------------------ #
    # SHARED: adversarial pass                                           #
    # ------------------------------------------------------------------ #

    async def _red_team_and_reconcile(
        self,
        question_text: str,
        resolution_criteria: str,
        analysis: str,
        today: str,
        answer_format: str,
        llm,
        extra_instruction: str = "",
    ) -> str:
        """Attack the forecast, then produce the reconciled final answer."""
        critique_prompt = clean_indents(
            f"""
            You are a red team reviewer. Today is {today}. Your job is to find what is
            wrong with the forecast below, not to agree with it.

            Question: {question_text}

            Resolution criteria:
            {resolution_criteria}

            The forecast to attack:
            {analysis}

            Check specifically, and say clearly whether each is a real problem here:
            1. MISREAD CRITERIA. Does the forecast answer the question actually asked,
               including every condition in the fine print and the exact deadline?
            2. RECENCY ANCHORING. Is a recent headline doing work that a base rate
               should be doing?
            3. STATUS QUO NEGLECT. Has it assumed change that requires several things
               to go right in a short window?
            4. UNSUPPORTED CONFIDENCE. Is the stated confidence earned by realised
               facts, or by a plausible-sounding story?
            5. ARITHMETIC AND DIRECTION. Any sums that do not add up, percentiles out
               of order, or a conclusion pointing the opposite way to its own evidence?
            {extra_instruction}

            Be concrete. If the forecast is sound, say so briefly rather than inventing
            objections — a forced critique is worse than none.
            """
        )
        critique = await llm.invoke(critique_prompt)

        reconcile_prompt = clean_indents(
            f"""
            You are the superforecaster. Today is {today}. You wrote an analysis and a
            reviewer attacked it. Decide the final answer.

            Question: {question_text}

            Your analysis:
            {analysis}

            The reviewer's critique:
            {critique}

            Say in two or three sentences which criticisms you accept and which you reject,
            and whether your number moves. Do not move just because you were criticised —
            move only if a specific objection is correct.

            {CALIBRATION_RULES}

            Then give your final answer as {answer_format}.
            """
        )
        reconciled = await llm.invoke(reconcile_prompt)
        return f"### Red team\n{critique}\n\n### Final\n{reconciled}"


# ---------------------------------------------------------------------- #
# ENTRY POINT                                                            #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run EdgeBot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "minibench", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full chain but do not publish to Metaculus",
    )
    args = parser.parse_args()
    run_mode: Literal[
        "tournament", "minibench", "metaculus_cup", "test_questions"
    ] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = not args.dry_run
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    llms, reports_per_question, predictions_per_report = build_llm_config(TIER)
    logger.info(
        f"Model tier: {TIER} | research reports/question: {reports_per_question} "
        f"| forecasts/report: {predictions_per_report}"
    )

    bot = EdgeBot(
        # On the frontier tier, three independent research reports give
        # genuinely different evidence bases and two forecasts each keeps the
        # median honest. The free tier drops to 1x1 to stay inside the
        # zero-cost rate limits while still exercising all four stages.
        research_reports_per_question=reports_per_question,
        predictions_per_research_report=predictions_per_report,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to="forecast_logs/",
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms=llms,
    )

    client = MetaculusClient()
    if run_mode == "tournament":
        reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        ) + asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
    elif run_mode == "minibench":
        reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    else:
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(
            bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)
        )

    bot.log_report_summary(reports)
    print_run_summary_banner(reports, will_publish=publish_to_metaculus)
