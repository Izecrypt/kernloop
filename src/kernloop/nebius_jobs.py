"""Backends that execute and time kernel candidates.

Interface (v1, stable):

    backend.run_correctness(spec, source) -> CorrectnessOutcome
    backend.run_benchmark(spec, source, reps, warmups) -> JobTiming

Implementations:

- FakeJobBackend: deterministic local simulator. Parses the candidate source,
  scores it with a transparent feature model, and emits per-rep timings with
  seeded noise. Lets the whole loop run and be tested with no GPU and no
  network. Timings are synthetic and always labeled as such in the log.
- NebiusJobBackend: submits candidates to Nebius Serverless Jobs over the
  REST API (docs.nebius.com/serverless, Jobs). The candidate kernel and the
  GPU checker are injected as files, the checker prints one KERNLOOP_RESULT
  JSON line, and the backend parses it from the job logs (via the `nebius`
  CLI, since the REST API has no logs endpoint). Not yet run against live
  credentials; the transport and log runner are injectable for offline tests.

When the real backend lands, the job must: build a sandbox with the spec,
reference checker, and candidate file; run the correctness check; then run
``warmups`` untimed reps followed by ``reps`` timed reps with L2 flushed
between reps (cold-cache discipline); and return per-rep times in ms.
"""

from __future__ import annotations

import hashlib
import json
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
# Nebius Serverless Jobs backend (REST).
#
# API shape per the Nebius docs (docs.nebius.com/serverless, Jobs, REST API
# tab): jobs live under https://api.nebius.cloud/ai/v1/jobs with protobuf-ish
# JSON envelopes (metadata/spec/status). Everything network-facing goes
# through an injectable transport, so the full submit/poll/parse flow is
# testable offline.
#
# One honest limitation: job LOGS are only exposed via the web console and
# the `nebius` CLI (the REST API has no logs endpoint), so result retrieval
# shells out to `nebius ai job logs <id>` (injectable runner). TODO(nebius):
# replace with a REST logs endpoint if Nebius adds one.
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "https://api.nebius.cloud/ai/v1"
DEFAULT_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
# TODO(nebius): pin a purpose-built image with torch + triton preinstalled
# (build and push to the project's Container Registry) once credentials
# exist. The CUDA devel image above is a placeholder that still needs
# `pip install torch triton` at job start (see _job_bootstrap_command).
DEFAULT_PLATFORM = "gpu-h100-sxm"
DEFAULT_PRESET = "1gpu-16vcpu-200gb"
JOB_TIMEOUT_S = "3600s"
CHECKER_CONTAINER_PATH = "/opt/kernloop/gpu_checker.py"
CANDIDATE_CONTAINER_PATH = "/opt/kernloop/candidate.py"
MAX_INJECTED_FILE_BYTES = 64 * 1024  # per Nebius docs

# Job states observed in status.state / status.phase. The docs confirm
# `COMPLETED` as the success status and canceling for the rest, but do not
# enumerate the full enum, so match defensively.
# TODO(nebius): confirm the exact state enum against a live job.
_TERMINAL_STATES = {"SUCCEEDED", "COMPLETED", "FAILED", "ERROR", "CANCELLED", "CANCELED"}
_FAILURE_STATES = {"FAILED", "ERROR", "CANCELLED", "CANCELED"}


class JobExecutionError(RuntimeError):
    """A Nebius job ran but did not produce a usable result."""


def _default_transport(method: str, url: str, token: str, payload: dict | None, timeout_s: float) -> tuple[int, dict]:
    """Minimal JSON REST transport on stdlib urllib (no extra deps)."""

    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
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
        raise BackendConfigError(f"Nebius API error {exc.code} for {method} {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BackendConfigError(f"cannot reach Nebius API at {url}: {exc.reason}") from exc


def _default_logs_runner(cmd: list[str], timeout_s: float) -> str:
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise JobExecutionError(
            f"`{' '.join(cmd)}` exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )
    return proc.stdout


class NebiusJobBackend:
    """Runs candidates as Nebius Serverless Jobs on real GPUs.

    Config via environment (or constructor kwargs):
    - NEBIUS_IAM_TOKEN: access token for the Nebius API (owner signup).
    - NEBIUS_PROJECT_ID: project the jobs are created in.

    Each correctness check or benchmark is one job: the candidate kernel file
    and the GPU checker are injected as files (spec.injectedFiles), and the
    checker prints one KERNLOOP_RESULT JSON line that we parse from the job
    logs.
    """

    name = "nebius"

    def __init__(
        self,
        token: str | None = None,
        project_id: str | None = None,
        api_url: str | None = None,
        image: str | None = None,
        platform: str | None = None,
        preset: str | None = None,
        poll_interval_s: float = 15.0,
        poll_timeout_s: float = 1800.0,
        transport=None,
        logs_runner=None,
    ):
        import os

        self.token = token if token is not None else os.environ.get("NEBIUS_IAM_TOKEN")
        self.project_id = project_id if project_id is not None else os.environ.get("NEBIUS_PROJECT_ID")
        missing = [n for n, v in (("NEBIUS_IAM_TOKEN", self.token), ("NEBIUS_PROJECT_ID", self.project_id)) if not v]
        if missing:
            raise BackendConfigError(
                "Nebius Serverless Jobs backend is not yet configured: set "
                f"{', '.join(missing)} (from the Nebius AI Cloud console). The "
                "backend code is complete but has not been run against live "
                "credentials yet; use --backend fake for local simulation."
            )
        self.api_url = (api_url or DEFAULT_API_URL).rstrip("/")
        self.image = image or DEFAULT_IMAGE
        self.platform = platform or DEFAULT_PLATFORM
        self.preset = preset or DEFAULT_PRESET
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s
        self._transport = transport or _default_transport
        self._logs_runner = logs_runner or _default_logs_runner

    # -- payload construction ------------------------------------------------

    def build_job_payload(self, job_name: str, files: dict[str, str], args: str) -> dict:
        """Assemble the REST body for POST /ai/v1/jobs.

        files maps container path -> file content (injectedFiles, base64).
        """

        import base64

        injected = []
        for path, content in files.items():
            raw = content.encode()
            if len(raw) > MAX_INJECTED_FILE_BYTES:
                raise ValueError(f"injected file {path} exceeds {MAX_INJECTED_FILE_BYTES} bytes")
            injected.append({"containerPath": path, "content": base64.b64encode(raw).decode()})
        return {
            "metadata": {"parentId": self.project_id, "name": job_name},
            "spec": {
                "image": self.image,
                "containerCommand": "bash",
                "args": args,
                "injectedFiles": injected,
                "platform": self.platform,
                "preset": self.preset,
                "timeout": JOB_TIMEOUT_S,
                "disk": {"type": "NETWORK_SSD", "sizeBytes": str(50 * 1024**3)},
            },
        }

    def _checker_args(self, spec: WorkloadSpec, mode: str, reps: int, warmups: int) -> str:
        cmd = (
            f"python {CHECKER_CONTAINER_PATH}"
            f" --candidate {CANDIDATE_CONTAINER_PATH}"
            f" --mode {mode}"
            f" --M {spec.M} --N {spec.N} --K {spec.K}"
            f" --dtype {spec.dtype}"
            f" --rtol {spec.correctness_rtol}"
            f" --reps {reps} --warmups {warmups}"
        )
        # args is a single string passed to `docker run` style entrypoints.
        return f"-c {_shell_quote(_bootstrap_prefix() + cmd)}"

    # -- REST plumbing ---------------------------------------------------------

    def _submit(self, payload: dict) -> str:
        _, body = self._transport("POST", f"{self.api_url}/jobs", self.token, payload, timeout_s=60)
        job_id = (body.get("metadata") or {}).get("id")
        if not job_id:
            raise JobExecutionError(f"Nebius did not return a job id: {body}")
        return job_id

    def _poll(self, job_id: str) -> dict:
        import time as _time

        deadline = _time.monotonic() + self.poll_timeout_s
        while True:
            _, body = self._transport("GET", f"{self.api_url}/jobs/{job_id}", self.token, None, timeout_s=60)
            status = body.get("status") or {}
            state = str(status.get("state") or status.get("phase") or "").upper()
            if state in _TERMINAL_STATES:
                return {"state": state, "status": status}
            if _time.monotonic() > deadline:
                raise JobExecutionError(f"job {job_id} did not finish within {self.poll_timeout_s}s (last state {state!r})")
            _time.sleep(self.poll_interval_s)

    def _logs(self, job_id: str) -> str:
        return self._logs_runner(["nebius", "ai", "job", "logs", job_id], timeout_s=120)

    @staticmethod
    def _parse_result(logs: str) -> dict:
        result = None
        for line in logs.splitlines():
            if line.startswith("KERNLOOP_RESULT "):
                result = line[len("KERNLOOP_RESULT "):]
        if result is None:
            raise JobExecutionError("job logs contain no KERNLOOP_RESULT line")
        return json.loads(result)

    # -- protocol --------------------------------------------------------------

    def _run_job(self, spec: WorkloadSpec, source: str, mode: str, reps: int, warmups: int) -> dict:
        checker = spec.checker_path().read_text(encoding="utf-8")
        payload = self.build_job_payload(
            job_name=f"kernloop-{spec.name}-{mode}",
            files={CHECKER_CONTAINER_PATH: checker, CANDIDATE_CONTAINER_PATH: source},
            args=self._checker_args(spec, mode, reps, warmups),
        )
        job_id = self._submit(payload)
        outcome = self._poll(job_id)
        logs = self._logs(job_id)
        if outcome["state"] in _FAILURE_STATES:
            tail = "\n".join(logs.splitlines()[-20:])
            raise JobExecutionError(f"Nebius job {job_id} failed (state {outcome['state']}). Log tail:\n{tail}")
        parsed = self._parse_result(logs)
        parsed["job_id"] = job_id
        return parsed

    def run_correctness(self, spec: WorkloadSpec, source: str) -> CorrectnessOutcome:
        try:
            result = self._run_job(spec, source, mode="check", reps=1, warmups=0)
        except JobExecutionError as exc:
            return CorrectnessOutcome(ok=False, error=str(exc), detail={"backend": "nebius"})
        ok = bool(result.get("correct"))
        return CorrectnessOutcome(
            ok=ok,
            error=None if ok else result.get("error", "correctness failed on GPU"),
            detail={"backend": "nebius", "rel_l2_error": result.get("rel_l2_error"), "job_id": result.get("job_id")},
        )

    def run_benchmark(self, spec: WorkloadSpec, source: str, reps: int, warmups: int) -> JobTiming:
        result = self._run_job(spec, source, mode="bench", reps=reps, warmups=warmups)
        if not result.get("correct"):
            raise JobExecutionError(f"benchmark job reported incorrect kernel: {result.get('error')}")
        return JobTiming(
            reps_ms=[float(r) for r in result["reps_ms"]],
            warmups=warmups,
            meta={"synthetic": False, "job_id": result.get("job_id"), "device": result.get("device")},
        )


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _bootstrap_prefix() -> str:
    # TODO(nebius): bake torch+triton into the job image instead of
    # installing at container start once a project image exists.
    return "pip install -q torch triton 2>/dev/null || pip install -q torch-triton; "


def get_backend(name: str, seed: int = 0) -> BenchmarkBackend:
    if name == "fake":
        return FakeJobBackend(seed=seed)
    if name == "nebius":
        return NebiusJobBackend()
    raise BackendConfigError(f"unknown backend: {name!r} (choices: fake, nebius)")
