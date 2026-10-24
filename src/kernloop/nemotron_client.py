"""LLM clients for kernel proposal.

Two implementations share one protocol:

- NemotronClient: OpenAI-compatible chat client pointed at the Nebius Token
  Factory endpoint, using Nemotron for reasoning plus code generation. Requires
  the optional ``openai`` package and the NEBIUS_API_KEY environment variable.
  Retries with exponential backoff and jitter on connection, timeout, and rate
  limit errors.
- HeuristicLLM: offline fallback that applies deterministic transformations to
  the current champion (block sizes, num_stages / num_warps, split-K hints) so
  the full loop runs with no network. Clearly labeled in logs and never
  presented as Nemotron output.

Model name is configurable via the NEBIUS_MODEL environment variable.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Protocol

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"

PROMPT_TEMPLATE = """\
You are an expert GPU kernel engineer optimizing a Triton kernel for an NVIDIA H100.

PROBLEM SPEC:
{spec}

CURRENT CHAMPION KERNEL (the best variant so far, benchmark profile below):
```python
{champion}
```

BENCHMARK PROFILE OF THE CURRENT CHAMPION:
{profile}

Propose ONE improved variant of the kernel for this exact problem.

STRICT OUTPUT REQUIREMENTS:
- Output ONLY a complete, self-contained Python file.
- The file must contain the Triton kernel(s) and a `kernel_fn(a, b)` entrypoint.
- No prose, no markdown fences, no explanation, no commentary of any kind.
- The file must pass a correctness check against a reference GEMM and then be
  benchmarked, so it must be runnable as-is.
"""


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    synthetic: bool = False  # True for HeuristicLLM output


class LLMClient(Protocol):
    model: str

    def propose(self, prompt: str) -> LLMResponse: ...


def build_prompt(spec, champion_source: str, profile: str) -> str:
    return PROMPT_TEMPLATE.format(
        spec=spec.problem_summary + "\n\n" + spec.target_notes,
        champion=champion_source,
        profile=profile,
    )


class LLMConfigError(RuntimeError):
    pass


class NemotronClient:
    """OpenAI-compatible client for Nemotron via Nebius Token Factory."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
        timeout_s: float = 120.0,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("NEBIUS_API_KEY")
        if not self.api_key:
            raise LLMConfigError(
                "NEBIUS_API_KEY is not set. Get a key from the Nebius Token "
                "Factory console, or run with the offline proposer "
                "(kernloop run --llm heuristic) for local simulation."
            )
        self.model = model or os.environ.get("NEBIUS_MODEL") or DEFAULT_MODEL
        self.base_url = base_url or os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        try:
            import openai
        except ImportError as exc:
            raise LLMConfigError(
                "the 'openai' package is required for the Nemotron client: "
                "pip install 'kernloop[llm]'"
            ) from exc
        self._openai = openai
        self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout_s)

    def propose(self, prompt: str) -> LLMResponse:
        transient = (self._openai.APIConnectionError, self._openai.APITimeoutError, self._openai.RateLimitError)
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                started = time.monotonic()
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You write production Triton kernels. You output code only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                )
                latency = time.monotonic() - started
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                return LLMResponse(
                    text=text,
                    model=self.model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_s=round(latency, 3),
                )
            except transient as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(delay + random.uniform(0, delay * 0.5))
                    delay = min(delay * 2, 30.0)
        raise RuntimeError(f"Nemotron request failed after {self.max_retries} retries: {last_exc}")


class HeuristicLLM:
    """Offline proposer: deterministic mutations of the champion source.

    Exists so the whole loop (compile, correctness, benchmark, promotion) is
    testable with no GPU, no Triton, and no network. Every response is marked
    synthetic=True.
    """

    model = "heuristic-offline"

    # Staged progression: each stage applies the next optimization the fake
    # backend scores, so a run shows real promotions followed by honest
    # rejections once the plateau is reached.
    _SPLITK_NOTE = (
        "\n\n# split-K two-phase structure: phase 1 writes float32 partial\n"
        "# products to a workspace buffer, phase 2 reduces partials to bf16.\n"
    )

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def _mutate(self, source: str) -> str:
        out = source
        if "num_stages" not in out:
            return out.replace("BLOCK_K=BLOCK_K,", "BLOCK_K=BLOCK_K, num_stages=4, num_warps=8,", 1)
        if "BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32" in out:
            return out.replace("BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32", "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64")
        if "split-K two-phase" not in out:
            return out + self._SPLITK_NOTE
        return out

    def propose(self, prompt: str) -> LLMResponse:
        # The prompt embeds the champion; mutate whatever kernel text is in it.
        idx = prompt.find("```python\n")
        if idx == -1:
            text = "def kernel_fn(a, b):\n    raise NotImplementedError\n"
        else:
            start = idx + len("```python\n")
            end = prompt.find("\n```", start)
            champion = prompt[start:end]
            text = self._mutate(champion)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            latency_s=0.0,
            synthetic=True,
        )


def get_llm(name: str = "auto", seed: int = 0) -> LLMClient:
    """Resolve the proposer. 'auto' uses Nemotron when a key is present, else the offline heuristic."""

    if name == "nemotron":
        return NemotronClient()
    if name == "heuristic":
        return HeuristicLLM(seed=seed)
    if name == "auto":
        if os.environ.get("NEBIUS_API_KEY"):
            return NemotronClient()
        return HeuristicLLM(seed=seed)
    raise LLMConfigError(f"unknown llm: {name!r} (choices: auto, nemotron, heuristic)")
