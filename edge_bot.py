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
import email.utils
import logging
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import dotenv
import requests

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
# LLM BACKENDS
#
# No single provider is trusted. Each run, before spending any real calls,
# the bot sends every backend in the catalogue a one-word prompt, keeps the
# ones that answer, and chains them best-first. If the best backend rate
# limits or dies mid-run, calls fall through to the next one instead of
# failing the question.
#
# This is also how upgrades happen with nobody touching anything: the paid
# backends (OpenRouter, the Metaculus proxy) are always in the catalogue.
# They fail the probe while there are no credits, and the day credits land
# they start answering and move to the front of the chain by themselves.
#
# EDGEBOT_TIER can still force a fixed setup: "frontier" (paid OpenRouter)
# or "free" (OpenRouter zero-cost models). The default is "auto".
# --------------------------------------------------------------------- #

def _env(name: str, default: str) -> str:
    """os.getenv, but an empty value counts as unset.

    GitHub Actions sets `VAR: ${{ vars.X }}` to an EMPTY STRING when the repo
    variable does not exist, rather than leaving it undefined. os.getenv would
    then hand back "" instead of the default, and litellm fails with
    "LLM Provider NOT provided. You passed model=". Hence this wrapper.
    """
    value = os.getenv(name, "")
    return value.strip() or default


def _has(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


TIER = _env("EDGEBOT_TIER", "auto").lower()

# OpenRouter's zero-cost models. Last resort only: in September 2026 every
# one tested timed out on real prompts. Kept so the chain has a floor.
FREE_REASONER = _env(
    "EDGEBOT_FREE_MODEL", "openrouter/nvidia/nemotron-3.5-lightning:free"
)
FREE_SMALL = _env("EDGEBOT_FREE_SMALL_MODEL", FREE_REASONER)

POLLINATIONS_URL = "https://text.pollinations.ai/openai"


@dataclass(frozen=True)
class Backend:
    """One way of reaching one model.

    `raw=True` backends are plain OpenAI-compatible HTTP endpoints called
    directly (RawChatLlm) instead of through litellm. That gives exact error
    messages, per-endpoint pacing and Retry-After handling, which the free
    endpoints need and litellm hides.
    """

    label: str
    model: str  # litellm model string (litellm backends) / placeholder (raw)
    quality: int  # higher = better forecaster; orders the main chain
    cheap: bool = False  # fine for parsing; preferred for the parser chain
    timeout: int = 120
    base_url: str | None = None
    api_key_env: str | None = None
    api_key_literal: str | None = None
    raw: bool = False
    wire_model: str | None = None  # model id sent over the wire (raw only)
    headers: tuple = ()  # extra HTTP headers (raw only)
    min_interval: float = 0.0  # seconds between requests (raw only)
    pace_key: str | None = None  # backends sharing one rate limit share a key
    rate_limit_wait: int | None = None  # wait on a 429 with no Retry-After

    def api_key(self) -> str | None:
        if self.api_key_literal is not None:
            return self.api_key_literal
        if self.api_key_env:
            return os.getenv(self.api_key_env, "").strip() or None
        return None

    def build(self, timeout: int | None = None) -> GeneralLlm:
        if self.raw:
            return RawChatLlm(self, timeout or self.timeout)
        kwargs: dict = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        key = self.api_key()
        if key:
            kwargs["api_key"] = key
        # allowed_tries=1: retrying is the chain's job, not the backend's. A
        # backend retrying a 429 with exponential backoff just burns minutes.
        return GeneralLlm(
            model=self.model,
            temperature=0.3,
            timeout=timeout or self.timeout,
            allowed_tries=1,
            **kwargs,
        )


class HttpLlmError(RuntimeError):
    """An HTTP-level failure, carrying the status code for routing."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class EmptyAnswerError(RuntimeError):
    """The endpoint answered 200 but with no usable text."""


def _retry_after_seconds(response: requests.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class RawChatLlm(GeneralLlm):
    """Minimal OpenAI-compatible chat client, used instead of litellm for the
    free endpoints. Only `.invoke` matters to the rest of the framework."""

    _locks: dict[str, asyncio.Lock] = {}
    _last_call: dict[str, float] = {}

    def __init__(self, backend: Backend, timeout: int) -> None:
        super().__init__(
            model=backend.model, temperature=0.3, timeout=timeout, allowed_tries=1
        )
        self.backend = backend
        self.request_timeout = timeout

    @staticmethod
    def _messages(prompt, system_prompt: str | None) -> list[dict]:
        if isinstance(prompt, list):
            return prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": str(prompt)})
        return messages

    async def _pace(self) -> None:
        interval = self.backend.min_interval
        if not interval:
            return
        key = self.backend.pace_key or self.backend.label
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            wait = self._last_call.get(key, 0.0) + interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call[key] = time.monotonic()

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:
        b = self.backend
        headers = {"Content-Type": "application/json", **dict(b.headers)}
        key = b.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": b.wire_model,
            "messages": self._messages(prompt, system_prompt),
            "temperature": 0.3,
        }
        url = (b.base_url or "").rstrip("/") + "/chat/completions"
        for attempt in range(3):
            await self._pace()
            response = await asyncio.to_thread(
                requests.post,
                url,
                json=payload,
                headers=headers,
                timeout=self.request_timeout,
            )
            if response.status_code == 429 and attempt < 2:
                wait = _retry_after_seconds(response) or b.rate_limit_wait
                if wait is not None and wait <= 90:
                    logger.info(f"{b.label}: 429, waiting {wait:.0f}s")
                    await asyncio.sleep(wait + 1)
                    continue
            if response.status_code != 200:
                body = " ".join(response.text.split())[:300]
                raise HttpLlmError(
                    response.status_code, f"HTTP {response.status_code}: {body}"
                )
            try:
                data = response.json()
            except ValueError:
                body = " ".join(response.text.split())[:300]
                raise HttpLlmError(502, f"200 but not JSON: {body}")
            choices = data.get("choices") or []
            if not choices:
                raise HttpLlmError(502, f"200 but no choices: {str(data)[:300]}")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, list):  # some servers return content parts
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if not isinstance(content, str) or not content.strip():
                raise EmptyAnswerError(
                    f"{b.label}: empty answer "
                    f"(finish_reason={choices[0].get('finish_reason')})"
                )
            return content
        raise HttpLlmError(429, f"{b.label}: still rate limited after retries")


def backend_catalogue() -> list[Backend]:
    """Every backend worth trying. Order within equal quality is preference."""
    cat: list[Backend] = []
    if _has("OPENROUTER_API_KEY"):
        cat += [
            Backend(
                "openrouter/claude-opus-4.5",
                _env("EDGEBOT_MODEL", "openrouter/anthropic/claude-opus-4.5"),
                quality=10,
            ),
            Backend(
                "openrouter/gpt-5-mini",
                _env("EDGEBOT_SMALL_MODEL", "openrouter/openai/gpt-5-mini"),
                quality=6,
                cheap=True,
            ),
        ]
    if _has("METACULUS_TOKEN"):
        # Metaculus sponsors LLM credits for tournament bots. Until an
        # allowance is granted these answer "You don't have an allowance";
        # the day it is, they win the probe and lead the chain.
        cat += [
            Backend(
                "metaculus-proxy/claude-sonnet-4.5",
                "metaculus/anthropic/claude-sonnet-4-5-20250929",
                quality=9,
            ),
            Backend("metaculus-proxy/gpt-4.1", "metaculus/gpt-4.1", quality=7),
            Backend("metaculus-proxy/gpt-4o", "metaculus/gpt-4o", quality=6),
            Backend(
                "metaculus-proxy/gpt-4o-mini",
                "metaculus/gpt-4o-mini",
                quality=4,
                cheap=True,
            ),
        ]
    # Pollinations: an established open-source platform with a free public
    # endpoint, no key. Modest models and strict pacing, but it is the one
    # free option that has answered reliably. Two model ids share one rate
    # limit, hence one pace key.
    def pollinations(wire, quality):
        return Backend(
            f"pollinations/{wire}",
            f"openai/{wire}",
            quality=quality,
            cheap=True,
            timeout=120,
            base_url=POLLINATIONS_URL,
            raw=True,
            wire_model=wire,
            min_interval=16,
            pace_key="pollinations",
            rate_limit_wait=20,
        )

    cat += [pollinations("openai", 3), pollinations("openai-fast", 2)]
    if _has("OPENROUTER_API_KEY"):
        cat.append(
            Backend("openrouter/free", FREE_REASONER, quality=1, timeout=60)
        )
    return cat


def list_endpoint_models() -> str:
    """Probe-mode diagnostics: which model ids each free endpoint offers."""
    lines = []
    targets = [
        ("pollinations", "https://text.pollinations.ai/models", {}),
    ]
    for name, url, headers in targets:
        try:
            response = requests.get(url, headers=headers, timeout=20)
            try:
                data = response.json()
            except ValueError:
                data = None
            if isinstance(data, dict):
                data = data.get("data") or data.get("models") or data
            ids = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        ids.append(str(item.get("id") or item.get("name")))
                    else:
                        ids.append(str(item))
            summary = ", ".join(ids[:60]) if ids else " ".join(response.text.split())[:300]
            lines.append(f"- **{name}** (HTTP {response.status_code}): {summary}")
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            lines.append(f"- **{name}**: {type(exc).__name__}: {exc}")
    return "\n".join(lines)


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _short_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    code = _status_code(exc)
    prefix = f"{type(exc).__name__}" + (f" [{code}]" if code else "")
    return f"{prefix}: {text[:220]}"


_HARD_FAIL_CODES = {401, 402, 403, 404, 429}
_HARD_FAIL_TEXT = (
    "insufficient",
    "credits",
    "quota",
    "rate limit",
    "ratelimit",
    "unauthorized",
    "not authorized",
    "permission",
    "no access",
    "not found",
    "does not exist",
    "unknown model",
    "unavailable",
    "invalid api key",
    "invalid_api_key",
)


def _is_hard_failure(exc: BaseException) -> bool:
    """Failures that will not fix themselves within this run."""
    if _status_code(exc) in _HARD_FAIL_CODES:
        return True
    text = str(exc).lower()
    if "too large" in text or "tokens_limit" in text or "context length" in text:
        return False  # this prompt is too big for it; the next one may fit
    return any(marker in text for marker in _HARD_FAIL_TEXT)


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, asyncio.TimeoutError) or "timeout" in type(
        exc
    ).__name__.lower()


class FallbackLlm(GeneralLlm):
    """A GeneralLlm that walks a chain of backends until one answers.

    Everything in the framework (the bot, structure_output, the researcher)
    only ever calls `.invoke`, so overriding that one method is enough.
    Disabled backends are shared across every chain in the run: once a
    backend has hit its daily quota for the parser, the forecaster does not
    waste a call rediscovering that.
    """

    disabled: dict[str, str] = {}
    timeouts: dict[str, int] = {}
    usage: dict[str, int] = {}

    def __init__(self, chain: list[Backend], role: str) -> None:
        if not chain:
            raise ValueError(f"Empty backend chain for {role}")
        super().__init__(model=chain[0].model, temperature=0.3, allowed_tries=1)
        self.role = role
        self.chain = [(backend, backend.build()) for backend in chain]

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:
        errors: list[str] = []
        for sweep in range(2):
            live = [
                (b, llm) for b, llm in self.chain if b.label not in self.disabled
            ]
            if not live:
                break
            if sweep:
                await asyncio.sleep(15)  # brief pause before a second pass
            for backend, llm in live:
                if backend.label in self.disabled:
                    continue  # disabled by a concurrent call meanwhile
                try:
                    answer = await llm.invoke(prompt, system_prompt)
                except Exception as exc:  # noqa: BLE001 - we route on any failure
                    reason = _short_error(exc)
                    errors.append(f"{backend.label}: {reason}")
                    if _is_hard_failure(exc):
                        self.disabled[backend.label] = reason
                        logger.warning(
                            f"[{self.role}] {backend.label} disabled for this run: {reason}"
                        )
                    elif _is_timeout(exc):
                        count = self.timeouts.get(backend.label, 0) + 1
                        self.timeouts[backend.label] = count
                        if count >= 2:
                            self.disabled[backend.label] = "timed out twice"
                            logger.warning(
                                f"[{self.role}] {backend.label} disabled: timed out twice"
                            )
                        else:
                            logger.warning(f"[{self.role}] {backend.label} timed out")
                    else:
                        logger.warning(
                            f"[{self.role}] {backend.label} failed, trying next: {reason}"
                        )
                    continue
                self.usage[backend.label] = self.usage.get(backend.label, 0) + 1
                self.timeouts[backend.label] = 0
                return answer
        raise RuntimeError(
            f"All LLM backends failed for role '{self.role}': "
            + " | ".join(errors[-6:])
        )


@dataclass
class ProbeResult:
    backend: Backend
    ok: bool
    seconds: float
    detail: str


async def probe_backends(
    backends: list[Backend], timeout: int = 45
) -> list[ProbeResult]:
    """Send each backend a trivial prompt, concurrently. Costs one call each."""

    async def one(backend: Backend) -> ProbeResult:
        start = time.monotonic()
        try:
            reply = await asyncio.wait_for(
                backend.build(timeout=timeout).invoke(
                    "Reply with the single word OK."
                ),
                timeout + 60,  # room for shared-rate-limit pacing
            )
            return ProbeResult(
                backend, True, time.monotonic() - start, " ".join(reply.split())[:40]
            )
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(
                backend, False, time.monotonic() - start, _short_error(exc)
            )

    return list(await asyncio.gather(*(one(b) for b in backends)))


def llms_from_working(working: list[Backend]) -> dict:
    """Main chain best-first; parser chain cheap-first (it only extracts)."""
    by_quality = sorted(working, key=lambda b: -b.quality)
    cheap_first = [b for b in by_quality if b.cheap] + [
        b for b in by_quality if not b.cheap
    ]
    main_chain = FallbackLlm(by_quality, "default")
    parser_chain = FallbackLlm(cheap_first, "parser")
    return {
        "default": main_chain,
        "researcher": main_chain,
        "summarizer": parser_chain,
        "parser": parser_chain,
    }


def build_fixed_llm_config(tier: str) -> dict:
    """The old fixed tiers, kept for EDGEBOT_TIER=frontier|free overrides."""
    if tier == "frontier":
        return {
            "default": GeneralLlm(
                model=_env("EDGEBOT_MODEL", "openrouter/anthropic/claude-opus-4.5"),
                temperature=0.3,
                timeout=120,
                allowed_tries=2,
            ),
            "summarizer": _env("EDGEBOT_SMALL_MODEL", "openrouter/openai/gpt-5-mini"),
            "researcher": _env(
                "EDGEBOT_RESEARCH_MODEL", "openrouter/perplexity/sonar-reasoning"
            ),
            "parser": _env("EDGEBOT_SMALL_MODEL", "openrouter/openai/gpt-5-mini"),
        }
    return {
        "default": GeneralLlm(
            model=FREE_REASONER, temperature=0.3, timeout=60, allowed_tries=2
        ),
        "summarizer": FREE_SMALL,
        "researcher": FREE_REASONER,
        "parser": FREE_SMALL,
    }


# --------------------------------------------------------------------- #
# NEWS: real headlines, so the model is not asked to recall "current news"
# it cannot know. Google News RSS is public and needs no key.
# --------------------------------------------------------------------- #

_STOPWORDS = set(
    "will the a an of in on at by to for from with and or be is are was were "
    "before after than more less least most any this that these those its it "
    "as between during within until what which who whom whose how when where "
    "does do did has have had not no yes than end".split()
)


def _keyword_query(text: str, max_words: int = 6) -> str:
    words = [
        w.strip("?,.:;()\"'")
        for w in text.split()
        if w.strip("?,.:;()\"'").lower() not in _STOPWORDS
    ]
    words = [w for w in words if w and not w.isdigit()]
    return " ".join(words[:max_words])


def _fetch_headlines_sync(query: str, max_items: int = 12) -> list[tuple]:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": f"{query} when:30d", "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    response = requests.get(
        url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (EdgeBot research)"}
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    items = []
    for item in root.iter("item"):
        title = " ".join((item.findtext("title") or "").split())
        raw_date = (item.findtext("pubDate") or "").strip()
        try:
            published = email.utils.parsedate_to_datetime(raw_date)
        except (TypeError, ValueError):
            published = None
        if title:
            items.append((published, title))
        if len(items) >= max_items:
            break
    return items


async def fetch_headlines(queries: list[str], limit: int = 15) -> str:
    """Headlines for several queries, deduplicated, newest first."""
    seen: set[str] = set()
    collected: list[tuple] = []
    for query in queries:
        if not query.strip():
            continue
        try:
            items = await asyncio.to_thread(_fetch_headlines_sync, query)
        except Exception as exc:  # noqa: BLE001 - news is best effort
            logger.warning(f"Headline fetch failed for '{query}': {exc}")
            continue
        for published, title in items:
            key = title.lower()[:90]
            if key not in seen:
                seen.add(key)
                collected.append((published, title))
    collected.sort(
        key=lambda pair: pair[0].timestamp() if pair[0] else 0, reverse=True
    )
    lines = [
        f"- {published.strftime('%Y-%m-%d') if published else 'undated'}: {title}"
        for published, title in collected[:limit]
    ]
    return "\n".join(lines)



class EdgeBot(SummerTemplateBot2026):
    """Four-stage forecaster. Overrides research and the three main question types."""

    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    # ------------------------------------------------------------------ #
    # RESEARCH: two passes — current evidence, then historical frequency  #
    # ------------------------------------------------------------------ #

    async def _search_queries(self, question: MetaculusQuestion) -> list[str]:
        """Two to three short news queries; keyword fallback if the LLM fails."""
        fallback = [_keyword_query(question.question_text)]
        prompt = clean_indents(
            f"""
            Write 3 short Google News search queries (2 to 5 words each) that would
            surface the latest news relevant to this forecasting question.
            One query per line. No numbering, no quotes, nothing else.

            Question: {question.question_text}
            """
        )
        try:
            raw = await self.get_llm("parser", "llm").invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - fall back to keywords
            logger.warning(f"Query generation failed, using keywords: {exc}")
            return fallback
        queries = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip().strip("\"'")
            if 1 <= len(line.split()) <= 8:
                queries.append(line)
        return queries[:3] or fallback

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            queries = await self._search_queries(question)
            headlines = await fetch_headlines(queries)
            logger.info(
                f"Headlines for {question.page_url} (queries: {queries}):\n"
                f"{headlines or '(none)'}"
            )
            today = datetime.now().strftime("%Y-%m-%d")
            news_prompt = clean_indents(
                f"""
                You are a research assistant to a superforecaster. You do not forecast.
                Today is {today}.

                Gather the most decision-relevant current information on this question.
                Prioritise: (a) facts that have already happened and cannot be undone,
                (b) scheduled events with dates before the resolution deadline,
                (c) statements by people with the actual power to cause or block the outcome.
                Explicitly flag anything that is speculation, rumour or opinion rather than fact.
                If the question would already resolve one way on today's information, say so plainly.

                Your own knowledge stops at your training date. Anything more recent is known
                ONLY from the headlines below, which were retrieved today and are real. Say which
                points come from the headlines and which from background knowledge. Never invent
                news; if the headlines are silent on something, say that it is unknown.

                Headlines, newest first:
                {headlines or "(no headlines were found)"}

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
#
# Exit code policy: a run exits 0 unless the code itself is broken. "No new
# questions", "no LLM backend answered" and "some questions failed" are all
# normal operating states for a bot that runs around the clock, and a
# non-zero exit makes GitHub email the repo owner every time. Problems are
# reported instead as annotations and in the run's job summary.
#
# Why a polling loop: MiniBench questions open one at a time and each is
# open for only about three hours. GitHub's cron fires unreliably (in
# practice a handful of times a day), so a cron-only bot misses most
# questions. Instead each run stays alive for several hours, checking for
# new questions every few minutes, and the workflow hands over to a fresh
# run when this one ends.
# ---------------------------------------------------------------------- #

RunMode = Literal[
    "tournament", "minibench", "metaculus_cup", "test_questions", "single", "probe"
]
LOOPABLE_MODES = ("tournament", "minibench")

# Questions that failed this many times in this process are left alone, so
# one question the bot cannot handle does not eat every cycle's LLM budget.
MAX_ATTEMPTS_PER_QUESTION = 2
_attempts: dict[str, int] = {}
# Forecast in this process already; never re-forecast even if the API's
# "already forecasted" flag lags behind.
_done: set[str] = set()


def write_job_summary(markdown: str) -> None:
    """Append to the GitHub Actions run page summary (no-op locally)."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(markdown.rstrip() + "\n\n")
    except OSError as exc:
        logger.warning(f"Could not write job summary: {exc}")


def annotate(level: str, message: str) -> None:
    """GitHub Actions annotation: shows on the run page, fails nothing."""
    if os.getenv("GITHUB_ACTIONS"):
        print(f"::{level}::{message}", flush=True)
    else:
        logger.info(f"[{level}] {message}")


def probe_table(results: list[ProbeResult]) -> str:
    rows = ["| Backend | Result | Seconds | Detail |", "|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (not r.ok, -r.backend.quality)):
        detail = r.detail.replace("|", "/")[:160]
        rows.append(
            f"| {r.backend.label} | {'OK' if r.ok else 'fail'} "
            f"| {r.seconds:.1f} | {detail} |"
        )
    return "\n".join(rows)


def collect_questions(
    client: MetaculusClient, run_mode: RunMode
) -> list[MetaculusQuestion]:
    """Open questions for this mode, not yet forecasted, soonest-closing first."""
    if run_mode == "single":
        # Smoke test: exactly one question, preferably binary (simplest type
        # that still exercises the whole chain). Re-forecasting is allowed.
        pool = client.get_all_open_questions_from_tournament("bot-testing-area")
        binaries = [q for q in pool if isinstance(q, BinaryQuestion)]
        return (binaries or pool)[:1]

    tournament_ids: list[int | str] = {
        "tournament": [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID],
        "minibench": [client.CURRENT_MINIBENCH_ID],
        "metaculus_cup": [client.CURRENT_METACULUS_CUP_ID],
        "test_questions": ["bot-testing-area"],
    }[run_mode]
    reforecast = run_mode in ("metaculus_cup", "test_questions")

    questions: list[MetaculusQuestion] = []
    for tournament_id in tournament_ids:
        try:
            found = client.get_all_open_questions_from_tournament(tournament_id)
        except Exception as exc:  # noqa: BLE001 - one bad feed must not stop the other
            logger.warning(f"Could not list tournament {tournament_id}: {exc}")
            continue
        fresh = [
            q
            for q in found
            if (reforecast or (not q.already_forecasted and q.page_url not in _done))
            and _attempts.get(q.page_url, 0) < MAX_ATTEMPTS_PER_QUESTION
        ]
        logger.info(
            f"Tournament {tournament_id}: {len(found)} open, {len(fresh)} to forecast"
        )
        questions += fresh

    far_future = datetime.max.replace(tzinfo=timezone.utc)

    def closes(q: MetaculusQuestion) -> datetime:
        close = q.close_time
        if close is None:
            return far_future
        return close if close.tzinfo else close.replace(tzinfo=timezone.utc)

    return sorted(questions, key=closes)


async def run_cycle(
    run_mode: RunMode, publish: bool, budget_minutes: float
) -> tuple[int, int]:
    """One pass: find new questions, pick LLMs, forecast. Returns (done, failed)."""
    client = MetaculusClient()

    # 1. Anything to do? Most cycles find nothing new, and those must cost
    #    zero LLM calls: free tiers have small daily quotas.
    if run_mode == "probe":
        questions: list[MetaculusQuestion] = []
    else:
        questions = collect_questions(client, run_mode)
        if not questions:
            logger.info("No new questions to forecast.")
            return 0, 0
        max_per_cycle = int(_env("EDGEBOT_MAX_QUESTIONS_PER_RUN", "12"))
        if len(questions) > max_per_cycle:
            logger.info(
                f"{len(questions)} questions pending; doing the {max_per_cycle} "
                "closing soonest now, the rest next cycle."
            )
            questions = questions[:max_per_cycle]

    # 2. Pick the LLMs. Probed fresh every cycle that has work, because
    #    quotas reset and credits can land at any time.
    FallbackLlm.disabled.clear()
    FallbackLlm.timeouts.clear()
    FallbackLlm.usage.clear()
    if TIER in ("frontier", "free"):
        llms = build_fixed_llm_config(TIER)
        reports_per_question, predictions_per_report = (
            (3, 2) if TIER == "frontier" else (1, 1)
        )
        backend_note = f"fixed tier '{TIER}'"
    else:
        results = await probe_backends(backend_catalogue())
        working = [r.backend for r in results if r.ok]
        table = probe_table(results)
        logger.info("Backend probe:\n" + table)
        if run_mode == "probe":
            models = await asyncio.to_thread(list_endpoint_models)
            logger.info("Endpoint model lists:\n" + models)
            write_job_summary(
                f"### EdgeBot backend probe\n{table}\n\n#### Models offered\n{models}"
            )
            return 0, 0
        if not working:
            write_job_summary(f"### EdgeBot backend probe\n{table}")
            annotate(
                "error",
                f"No LLM backend answered; {len(questions)} question(s) left for the "
                "next cycle. See the probe table in the job summary.",
            )
            return 0, 0
        llms = llms_from_working(working)
        best = max(working, key=lambda b: b.quality)
        # Depth scales with what we have. Frontier-class backends get an
        # ensemble; quota-limited free backends get one careful pass per
        # question so the daily budget covers every question.
        if best.quality >= 9:
            reports_per_question, predictions_per_report = 2, 2
        else:
            reports_per_question, predictions_per_report = 1, 1
        backend_note = "chain: " + " > ".join(
            b.label for b in sorted(working, key=lambda b: -b.quality)
        )

    logger.info(
        f"{backend_note} | research reports/question: {reports_per_question} "
        f"| forecasts/report: {predictions_per_report}"
    )

    bot = EdgeBot(
        research_reports_per_question=reports_per_question,
        predictions_per_research_report=predictions_per_report,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to="forecast_logs/",
        # Already filtered in collect_questions; single/test modes re-forecast.
        skip_previously_forecasted_questions=False,
        extra_metadata_in_explanation=True,
        llms=llms,
    )
    if reports_per_question == 1:
        # One parse, not two: with a single cheap parser, two samples that
        # disagree on formatting would throw away a perfectly good forecast.
        bot._structure_output_validation_samples = 1

    # 3. Forecast inside a hard time budget. Each question is published as
    #    soon as it is done, so work finished before the cut is kept.
    for q in questions:
        logger.info(f"Queued: {q.page_url}")
    try:
        reports = await asyncio.wait_for(
            bot.forecast_questions(questions, return_exceptions=True),
            timeout=budget_minutes * 60,
        )
    except asyncio.TimeoutError:
        annotate(
            "warning",
            f"Cycle stopped at its {budget_minutes:.0f}-minute budget; unfinished "
            "questions will be retried.",
        )
        reports = []

    ok, failed = [], []
    for question, report in zip(questions, reports):
        if isinstance(report, BaseException):
            failed.append((question, report))
            _attempts[question.page_url] = _attempts.get(question.page_url, 0) + 1
        else:
            ok.append(question)
            _done.add(question.page_url)
    bot.log_report_summary(reports, raise_errors=False)
    print_run_summary_banner(reports, will_publish=publish)

    usage = ", ".join(f"{k}: {v}" for k, v in sorted(FallbackLlm.usage.items()))
    disabled = "; ".join(f"{k} ({v[:80]})" for k, v in FallbackLlm.disabled.items())
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"### EdgeBot cycle at {stamp} ({run_mode})",
        f"- Forecasts {'published' if publish else 'made (dry run)'}: {len(ok)}"
        f" of {len(questions)}",
        f"- LLM setup: {backend_note}",
    ]
    if usage:
        lines.append(f"- Calls answered per backend: {usage}")
    if disabled:
        lines.append(f"- Disabled during cycle: {disabled}")
    for question in ok:
        lines.append(f"  - done: {question.page_url}")
    for question, exc in failed:
        lines.append(
            f"  - failed: {question.page_url} — {' '.join(str(exc).split())[:200]}"
        )
    write_job_summary("\n".join(lines))
    if failed:
        annotate("warning", f"{len(failed)} of {len(questions)} question(s) failed.")
    return len(ok), len(failed)


async def run(
    run_mode: RunMode, publish: bool, loop_minutes: float, poll_minutes: float
) -> None:
    looping = loop_minutes > 0 and run_mode in LOOPABLE_MODES
    deadline = time.monotonic() + loop_minutes * 60
    default_budget = float(_env("EDGEBOT_RUN_BUDGET_MINUTES", "40"))
    cycles = done = failed = 0
    while True:
        cycles += 1
        remaining = (deadline - time.monotonic()) / 60 if looping else default_budget
        budget = max(5.0, min(default_budget, remaining - 1))
        try:
            cycle_done, cycle_failed = await run_cycle(run_mode, publish, budget)
            done += cycle_done
            failed += cycle_failed
        except Exception as exc:  # noqa: BLE001 - a bad cycle must not end the loop
            logger.exception("Cycle crashed")
            annotate("warning", f"Cycle {cycles} crashed: {type(exc).__name__}: {exc}")
        if not looping or time.monotonic() + poll_minutes * 60 > deadline:
            break
        await asyncio.sleep(poll_minutes * 60)
    if looping:
        write_job_summary(
            f"### EdgeBot run finished\n- Cycles: {cycles}\n"
            f"- Forecasts published: {done}\n- Failed: {failed}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run EdgeBot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(RunMode.__args__),
        default="tournament",
        help="What to forecast on (default: tournament). 'probe' only tests "
        "which LLM backends answer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full chain but do not publish to Metaculus",
    )
    parser.add_argument(
        "--loop-minutes",
        type=float,
        default=0,
        help="Keep checking for new questions for this long (tournament and "
        "minibench modes). 0 = a single pass.",
    )
    parser.add_argument(
        "--poll-minutes",
        type=float,
        default=10,
        help="Minutes between checks when looping.",
    )
    args = parser.parse_args()
    run_mode: RunMode = args.mode

    check_environment(strict=True)
    publish_to_metaculus = not args.dry_run
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)
    logger.info(
        f"Model tier: {TIER} | loop: {args.loop_minutes} min, "
        f"poll every {args.poll_minutes} min"
    )

    asyncio.run(
        run(run_mode, publish_to_metaculus, args.loop_minutes, args.poll_minutes)
    )
