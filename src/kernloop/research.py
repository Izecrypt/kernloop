"""Tavily research step for the KernLoop agent loop.

One web-search call per trigger point per run, injected into the Nemotron
prompt as clearly delimited reference context (never instructions):

- "initial": once at run start, a query built from the workload spec.
- "recovery": after 3 consecutive rejected iterations, a query built from
  the dominant (latest) rejection reason.

The client is a plain REST call (POST https://api.tavily.com/search) on
stdlib urllib, so no extra dependency; the transport is injectable for
tests. Every failure mode (missing key, HTTP error, timeout, malformed
response) degrades to a logged warning and empty context: research must
never break the loop.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("kernloop.research")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 3
HTTP_TIMEOUT_S = 30.0

TRIGGER_INITIAL = "initial"
TRIGGER_RECOVERY = "recovery"


@dataclass
class ResearchNote:
    """Cached result of one research call."""

    trigger: str
    query: str
    snippets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"trigger": self.trigger, "query": self.query, "snippets": self.snippets}


def build_initial_query(spec) -> str:
    """Run-start query from the workload spec."""

    return (
        f"Triton {spec.dtype} GEMM optimization best practices "
        f"split-K two-phase block size tuning num_warps num_stages H100"
    )


def build_recovery_query(reason: str) -> str:
    """Query from the dominant rejection reason, after repeated rejections."""

    cleaned = " ".join(str(reason).split())
    return f"Triton kernel {cleaned}"


def _default_transport(url: str, payload: dict, timeout_s: float) -> tuple[int, dict]:
    import json
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode() or "{}"
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode() or "{}"
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            detail = {"raw": body}
        return exc.code, detail
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Tavily unreachable: {exc.reason}") from exc


class TavilyClient:
    """Minimal REST client for the Tavily search API."""

    def __init__(self, api_key: str | None = None, transport=None, timeout_s: float = HTTP_TIMEOUT_S):
        self.api_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY")
        self._transport = transport or _default_transport
        self.timeout_s = timeout_s

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[str]:
        """Top snippets for a query. Raises RuntimeError on any failure."""

        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        try:
            status, body = self._transport(TAVILY_SEARCH_URL, payload, self.timeout_s)
        except Exception as exc:  # transport-level failure (timeout, DNS, TLS)
            raise RuntimeError(f"Tavily request failed: {exc}") from exc
        if status != 200:
            raise RuntimeError(f"Tavily API returned HTTP {status}: {body}")
        results = body.get("results") or []
        snippets = []
        for item in results[:max_results]:
            text = " ".join(str(item.get("content") or item.get("title") or "").split())
            if text:
                snippets.append(text)
        return snippets


class Researcher:
    """Caches one Tavily call per trigger per run; degrades to empty context.

    fetch() never raises: failures are logged as warnings and cached so a
    flaky API is not hammered once per iteration.
    """

    def __init__(self, client: TavilyClient | None = None):
        self.client = client or TavilyClient()
        self._cache: dict[str, ResearchNote] = {}

    def _fetch(self, trigger: str, query: str) -> ResearchNote | None:
        if trigger in self._cache:
            return self._cache[trigger]
        if not self.client.configured:
            logger.warning("research %s skipped: TAVILY_API_KEY is not set", trigger)
            self._cache[trigger] = None
            return None
        try:
            snippets = self.client.search(query)
        except Exception as exc:  # HTTP error, timeout, malformed response: never break the loop
            logger.warning("research %s failed (%s); continuing without it", trigger, exc)
            self._cache[trigger] = None
            return None
        note = ResearchNote(trigger=trigger, query=query, snippets=snippets)
        self._cache[trigger] = note
        return note

    def initial(self, spec) -> ResearchNote | None:
        return self._fetch(TRIGGER_INITIAL, build_initial_query(spec))

    def recovery(self, reason: str) -> ResearchNote | None:
        return self._fetch(TRIGGER_RECOVERY, build_recovery_query(reason))

    def notes_for_prompt(self) -> list[ResearchNote]:
        """All successful notes, in trigger order, for the prompt section."""

        return [n for n in self._cache.values() if n is not None and n.snippets]


def research_enabled(flag: bool | None = None) -> bool:
    """Resolve whether research is on: explicit flag wins, else KERNLOOP_RESEARCH env."""

    if flag is not None:
        return flag
    return os.environ.get("KERNLOOP_RESEARCH", "") == "1"
