"""The KernLoop agent loop.

Flow per iteration:

1. Build the Nemotron prompt from the spec, the current champion kernel, and
   the champion's benchmark profile.
2. Parse the response strictly: it must be a complete Python file defining
   kernel_fn, nothing else.
3. Compile check (syntax + entrypoint), then correctness on the backend.
4. Benchmark: warmup reps discarded, then timed reps, mean and min recorded.
5. Promote to champion only if the mean improves by strictly more than the
   threshold (default 2 percent) relative to the current champion; otherwise
   reject and log the reason.

Every iteration appends one JSON line to the run log, including token usage.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .harness import PROMOTION_THRESHOLD, CompileResult, TimingStats, compile_check, make_stats, should_promote
from .nemotron_client import LLMClient, build_prompt
from .nebius_jobs import BenchmarkBackend
from .runlog import Decision, IterationRecord, RunLogger
from .workload import WorkloadSpec


@dataclass
class Champion:
    source: str
    mean_ms: float
    min_ms: float
    iteration: int  # 0 == baseline from the spec


@dataclass
class RunSummary:
    run_id: str
    budget: int
    iterations_run: int
    promotions: int
    champion: Champion
    log_path: str
    backend: str
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def speedup_vs_baseline(self) -> float | None:
        return None  # filled by caller when baseline timing is known


def extract_python(text: str) -> tuple[str | None, str | None]:
    """Strip prose. Accept bare code or a single fenced block; reject anything else."""

    stripped = text.strip()
    fences = re.findall(r"```(?:python)?\n(.*?)```", stripped, flags=re.DOTALL)
    if fences:
        if len(fences) > 1 or stripped.replace("```python\n", "").replace("```\n", "").replace("```", "").strip():
            # fenced block plus extra prose
            leftover = stripped
            for f in fences:
                leftover = leftover.replace(f"```python\n{f}```", "").replace(f"```\n{f}```", "")
            if leftover.strip(" `\n"):
                return None, "response contains prose around the code fence"
        return fences[0].strip(), None
    # No fences: require it to look like a Python file.
    if re.search(r"^\s*(import|from|@|def)\s", stripped, flags=re.MULTILINE) and "kernel_fn" in stripped:
        return stripped, None
    return None, "response is not a bare Python file (found prose or missing code)"


class Orchestrator:
    def __init__(
        self,
        spec: WorkloadSpec,
        llm: LLMClient,
        backend: BenchmarkBackend,
        logger: RunLogger,
        budget: int,
        reps: int = 5,
        warmups: int = 2,
        threshold: float = PROMOTION_THRESHOLD,
    ):
        if budget < 1:
            raise ValueError("budget must be >= 1")
        self.spec = spec
        self.llm = llm
        self.backend = backend
        self.logger = logger
        self.budget = budget
        self.reps = reps
        self.warmups = warmups
        self.threshold = threshold
        self.champion = Champion(
            source=spec.baseline_source,
            mean_ms=float("inf"),
            min_ms=float("inf"),
            iteration=0,
        )
        self._tokens = {"prompt": 0, "completion": 0}

    def _profile(self, stats: TimingStats | None) -> str:
        if stats is None or stats.mean_ms is None:
            return (
                "No valid benchmark yet. The current champion is the naive "
                "tiled baseline and has not been timed on this backend."
            )
        lines = [
            f"mean: {stats.mean_ms:.4f} ms over {len(stats.reps)} reps ({self.reps} timed, {self.warmups} warmup discarded)",
            f"min-of-reps: {stats.min_ms:.4f} ms",
        ]
        if stats.stdev_ms is not None:
            lines.append(f"stdev: {stats.stdev_ms:.4f} ms")
        lines.append(f"per-rep (ms): {['%.4f' % r for r in stats.reps]}")
        lines.append("A variant is promoted only if its mean beats the champion mean by more than 2 percent.")
        return "\n".join(lines)

    def _run_iteration(self, i: int) -> None:
        prompt = build_prompt(self.spec, self.champion.source, self._profile(self._champion_stats))
        resp = self.llm.propose(prompt)
        self._tokens["prompt"] += resp.prompt_tokens
        self._tokens["completion"] += resp.completion_tokens

        proposal = {
            "model": resp.model,
            "synthetic": resp.synthetic,
            "latency_s": resp.latency_s,
            "raw_chars": len(resp.text),
        }
        source, extract_err = extract_python(resp.text)
        compile_res = CompileResult(ok=False, error=extract_err or "empty proposal")
        correctness: dict = {"ok": False, "error": "not run (compile failed)"}
        timing: dict = {"mean_ms": None, "min_ms": None, "reps": [], "reps_discarded_reason": None}
        decision = Decision(promoted=False, reason="compile failed", threshold=self.threshold)

        stats: TimingStats | None = None
        if source is not None:
            compile_res = compile_check(source)
            proposal["source"] = source
            if compile_res.ok:
                outcome = self.backend.run_correctness(self.spec, source)
                correctness = {"ok": outcome.ok, "error": outcome.error, "detail": outcome.detail}
                if outcome.ok:
                    job = self.backend.run_benchmark(self.spec, source, self.reps, self.warmups)
                    stats = make_stats(job.reps_ms, job.warmups)
                    timing = {
                        "mean_ms": stats.mean_ms,
                        "min_ms": stats.min_ms,
                        "stdev_ms": stats.stdev_ms,
                        "reps": stats.reps,
                        "warmups": stats.warmups,
                        "meta": job.meta,
                    }
                    promoted, reason = should_promote(stats.mean_ms, self.champion.mean_ms, self.threshold)
                    decision = Decision(promoted=promoted, reason=reason, threshold=self.threshold)
                    if promoted:
                        self.champion = Champion(
                            source=source,
                            mean_ms=stats.mean_ms,
                            min_ms=stats.min_ms,
                            iteration=i,
                        )
                        self._champion_stats = stats
                else:
                    decision = Decision(promoted=False, reason=f"correctness failed: {outcome.error}", threshold=self.threshold)
            else:
                decision = Decision(promoted=False, reason=f"compile failed: {compile_res.error}", threshold=self.threshold)

        record = IterationRecord(
            iteration=i,
            run_id=self.logger.run_id,
            timestamp=time.time(),
            proposal=proposal,
            compile={"ok": compile_res.ok, "error": compile_res.error, "entrypoint_found": compile_res.entrypoint_found},
            correctness=correctness,
            timing=timing,
            decision={"promoted": decision.promoted, "reason": decision.reason, "threshold": decision.threshold},
            tokens={"prompt": resp.prompt_tokens, "completion": resp.completion_tokens},
            backend=self.backend.name,
        )
        self.logger.log_iteration(record)

    _champion_stats: TimingStats | None = None

    def run(self) -> RunSummary:
        # Time the baseline first so the promotion threshold is relative to a
        # real measurement, not infinity.
        baseline_job = self.backend.run_benchmark(self.spec, self.spec.baseline_source, self.reps, self.warmups)
        baseline_stats = make_stats(baseline_job.reps_ms, baseline_job.warmups)
        self.champion = Champion(
            source=self.spec.baseline_source,
            mean_ms=baseline_stats.mean_ms,
            min_ms=baseline_stats.min_ms,
            iteration=0,
        )
        self._champion_stats = baseline_stats

        for i in range(1, self.budget + 1):
            self._run_iteration(i)

        return RunSummary(
            run_id=self.logger.run_id,
            budget=self.budget,
            iterations_run=self.budget,
            promotions=sum(1 for r in self.logger.iterations() if r["decision"]["promoted"]),
            champion=self.champion,
            log_path=str(self.logger.path),
            backend=self.backend.name,
            tokens=dict(self._tokens),
        )
