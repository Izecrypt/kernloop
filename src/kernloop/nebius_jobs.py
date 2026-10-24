"""Backends that execute and time kernel candidates.

Interface (v1, stable):

    backend.run_correctness(spec, source) -> CorrectnessOutcome
    backend.run_benchmark(spec, source, reps, warmups) -> JobTiming

Implementations:

- FakeJobBackend: deterministic local simulator. Parses the candidate source,
  scores it with a transparent feature model, and emits per-rep timings with
  seeded noise. Lets the whole loop run and be tested with no GPU and no
  network. Timings are synthetic and always labeled as such in the log.
- NebiusJobBackend: STUB. Interface is fixed, but submitting to Nebius
  Serverless Jobs is not wired up yet (needs Nebius AI Cloud credentials and
  the job packaging step). It fails fast with a clear message until then.

When the real backend lands, the job must: build a sandbox with the spec,
reference checker, and candidate file; run the correctness check; then run
``warmups`` untimed reps followed by ``reps`` timed reps with L2 flushed
between reps (cold-cache discipline); and return per-rep times in ms.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from .workload import WorkloadSpec


class BackendConfigError(RuntimeError):
    pass


@dataclass
class CorrectnessOutcome:
    ok: bool
    error: str | None = None
    detail: dict = field(default_factory=dict)


@dataclass
class JobTiming:
    reps_ms: list[float]
    warmups: int
    meta: dict = field(default_factory=dict)


class BenchmarkBackend(Protocol):
    name: str

    def run_correctness(self, spec: WorkloadSpec, source: str) -> CorrectnessOutcome: ...

    def run_benchmark(self, spec: WorkloadSpec, source: str, reps: int, warmups: int) -> JobTiming: ...


# ---------------------------------------------------------------------------
# Fake backend: local, deterministic, no GPU, no network.
# ---------------------------------------------------------------------------

_BASELINE_MS = 128.0  # synthetic naive-tiled time at 4096^3 bf16


def _score_candidate(source: str) -> tuple[float, list[str]]:
    """Feature model: (relative time, features found). 1.0 == baseline speed."""

    features: list[str] = []
    time_factor = 1.0

    blocks = {m: int(v) for m, v in re.findall(r"BLOCK_([MNK])\s*[:=]\s*(\d+)", source)}
    if blocks:
        features.append(f"blocks={blocks}")
        bm, bn = blocks.get("M", 32), blocks.get("N", 32)
        if bm >= 64 and bn >= 64:
            time_factor *= 0.72
        if blocks.get("K", 32) >= 64:
            time_factor *= 0.9

    stages = re.search(r"num_stages\s*=\s*(\d+)", source)
    if stages:
        s = int(stages.group(1))
        features.append(f"num_stages={s}")
        if s >= 3:
            time_factor *= 0.82
    warps = re.search(r"num_warps\s*=\s*(\d+)", source)
    if warps:
        w = int(warps.group(1))
        features.append(f"num_warps={w}")
        if w >= 4:
            time_factor *= 0.94

    lowered = source.lower()
    if "split" in lowered and ("workspace" in lowered or "partial" in lowered):
        time_factor *= 0.68
        features.append("split-k")
    if "atomic" in lowered:
        time_factor *= 1.06
        features.append("atomics")
    if "mask=" in source:
        time_factor *= 0.97
        features.append("masked-loads")
    if "tl.dot" in source:
        features.append("tl.dot")
    if "cdiv" in source and "grid" in lowered:
        features.append("grid-math")

    # Diminishing returns floor: no fake candidate goes below 18% of baseline.
    time_factor = max(time_factor, 0.18)
    return time_factor, features


def _correctness_heuristic(source: str) -> CorrectnessOutcome:
    """Deterministic stand-in for running the reference checker on a GPU."""

    if "tl.dot" not in source:
        return CorrectnessOutcome(ok=False, error="candidate never calls tl.dot", detail={"check": "fake"})
    if "tl.load" not in source or "tl.store" not in source:
        return CorrectnessOutcome(ok=False, error="candidate missing tl.load/tl.store", detail={"check": "fake"})
    lowered = source.lower()
    if "atomic" in lowered and "float32" not in lowered.replace("tl.float32", "float32"):
        return CorrectnessOutcome(
            ok=False,
            error="atomic split-K without a float32 workspace loses precision",
            detail={"check": "fake"},
        )
    return CorrectnessOutcome(ok=True, detail={"check": "fake", "note": "simulated, not executed on GPU"})


class FakeJobBackend:
    """Deterministic local simulator. Timings are synthetic, always labeled fake."""

    name = "fake"

    def __init__(self, seed: int = 0, baseline_ms: float = _BASELINE_MS):
        self.seed = seed
        self.baseline_ms = baseline_ms

    def _rng(self, source: str, salt: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}:{salt}:{source}".encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def run_correctness(self, spec: WorkloadSpec, source: str) -> CorrectnessOutcome:
        return _correctness_heuristic(source)

    def run_benchmark(self, spec: WorkloadSpec, source: str, reps: int, warmups: int) -> JobTiming:
        if reps < 1:
            raise ValueError("reps must be >= 1")
        factor, features = _score_candidate(source)
        center = self.baseline_ms * factor
        out: list[float] = []
        for i in range(warmups):
            self._rng(source, f"warmup:{i}").gauss(center, center * 0.01)  # discarded
        for i in range(reps):
            rng = self._rng(source, f"rep:{i}")
            # Cold-cache discipline: each timed rep is independently noisy,
            # with an occasional L2-resident outlier removed by taking the mean
            # over reps as the promotion metric.
            noise = rng.gauss(0.0, center * 0.015)
            out.append(round(max(center + noise, center * 0.5), 4))
        return JobTiming(
            reps_ms=out,
            warmups=warmups,
            meta={"synthetic": True, "features": features, "time_factor": round(factor, 4)},
        )


# ---------------------------------------------------------------------------
# Nebius Serverless Jobs backend: STUB, not yet configured.
# ---------------------------------------------------------------------------

class NebiusJobBackend:
    """Submits candidates to Nebius Serverless Jobs for real GPU execution.

    STUB (v0.1): the interface above is the contract; the submission path is
    not implemented. Needs: Nebius AI Cloud credentials, a job image with
    Triton + torch + CUDA, packaging of (spec, checker, candidate), and result
    retrieval. Until then, constructing this backend raises.
    """

    name = "nebius"

    def __init__(self, token: str | None = None, endpoint: str | None = None):
        if not token:
            raise BackendConfigError(
                "Nebius Serverless Jobs backend is not yet configured: set "
                "NEBIUS_JOBS_TOKEN (service account token for Nebius AI Cloud). "
                "The submission path (job packaging, correctness check, timed "
                "reps with cold-cache flushes) is a stub in v0.1."
            )
        raise BackendConfigError(
            "NebiusJobBackend is a stub: job submission is not implemented yet. "
            "Use --backend fake for local simulation."
        )

    def run_correctness(self, spec: WorkloadSpec, source: str) -> CorrectnessOutcome:  # pragma: no cover
        raise NotImplementedError

    def run_benchmark(self, spec: WorkloadSpec, source: str, reps: int, warmups: int) -> JobTiming:  # pragma: no cover
        raise NotImplementedError


def get_backend(name: str, seed: int = 0) -> BenchmarkBackend:
    if name == "fake":
        return FakeJobBackend(seed=seed)
    if name == "nebius":
        import os

        return NebiusJobBackend(token=os.environ.get("NEBIUS_JOBS_TOKEN"))
    raise BackendConfigError(f"unknown backend: {name!r} (choices: fake, nebius)")
