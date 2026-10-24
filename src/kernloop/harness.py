"""Correctness checking and benchmark harness for kernel candidates.

Discipline (matches how the owner runs the Open Kernel Challenge loop):

- Compile check first: the candidate must be a parseable Python file that
  defines a ``kernel_fn`` entrypoint. No prose, no partial files.
- Correctness before timing: a candidate that fails correctness is never
  benchmarked.
- Benchmark: warmup reps are run but discarded; timed reps use cold-cache
  discipline where the backend supports it (on real GPUs the backend flushes
  L2 between reps; the fake backend models per-rep cold-cache noise).
- Stats: mean of reps is the promotion metric; min of reps is the secondary
  (best-case) metric. Promotion requires strictly more than ``threshold``
  relative improvement of the mean (default 2 percent).
"""

from __future__ import annotations

import ast
import math
import statistics
from dataclasses import dataclass, field

PROMOTION_THRESHOLD = 0.02  # new champion must beat old mean by > 2 percent


@dataclass
class CompileResult:
    ok: bool
    error: str | None = None
    entrypoint_found: bool = False
    n_functions: int = 0


@dataclass
class TimingStats:
    reps: list[float] = field(default_factory=list)
    warmups: int = 0

    @property
    def mean_ms(self) -> float | None:
        return statistics.fmean(self.reps) if self.reps else None

    @property
    def min_ms(self) -> float | None:
        return min(self.reps) if self.reps else None

    @property
    def stdev_ms(self) -> float | None:
        return statistics.stdev(self.reps) if len(self.reps) >= 2 else None


def compile_check(source: str) -> CompileResult:
    """Static compile check: valid Python, defines kernel_fn, no top-level prose."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return CompileResult(ok=False, error=f"syntax error: {exc}")
    functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if "kernel_fn" not in functions:
        return CompileResult(
            ok=False,
            error="missing top-level `kernel_fn` entrypoint",
            n_functions=len(functions),
        )
    return CompileResult(ok=True, entrypoint_found=True, n_functions=len(functions))


def should_promote(new_mean_ms: float, champion_mean_ms: float, threshold: float = PROMOTION_THRESHOLD) -> tuple[bool, str]:
    """Champion promotion rule.

    Promote only if the new mean improves by strictly more than ``threshold``
    relative to the current champion mean. Returns (promoted, reason).
    """

    if champion_mean_ms <= 0:
        return False, "champion mean is not positive, refusing to promote"
    improvement = (champion_mean_ms - new_mean_ms) / champion_mean_ms
    if improvement > threshold:
        return True, f"mean improved {improvement:.2%} (> {threshold:.0%} threshold)"
    return False, f"mean improved only {improvement:.2%} (needs > {threshold:.0%})"


def make_stats(reps: list[float], warmups: int = 0) -> TimingStats:
    if not reps:
        raise ValueError("no timed reps")
    if any((not math.isfinite(r)) or r <= 0 for r in reps):
        raise ValueError("reps must be positive finite milliseconds")
    return TimingStats(reps=list(reps), warmups=warmups)
