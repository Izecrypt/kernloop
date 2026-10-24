import base64
import json

import pytest

from kernloop.nebius_jobs import (
    BackendConfigError,
    JobExecutionError,
    NebiusJobBackend,
)


RESULT_LINE = "KERNLOOP_RESULT " + json.dumps(
    {
        "correct": True,
        "rel_l2_error": 0.001,
        "reps_ms": [10.0, 11.0, 12.0],
        "mean_ms": 11.0,
        "min_ms": 10.0,
        "device": "NVIDIA H100 80GB HBM3",
    }
)


class FakeTransport:
    """Simulates the Nebius REST API: create returns a job id, GET cycles states."""

    def __init__(self, states=("RUNNING", "SUCCEEDED"), result_logs=None):
        self.states = list(states)
        self.calls = []
        self.result_logs = result_logs if result_logs is not None else RESULT_LINE

    def __call__(self, method, url, token, payload, timeout_s):
        self.calls.append((method, url, payload))
        assert token == "tok"
        if method == "POST":
            assert url.endswith("/jobs")
            return 200, {"metadata": {"id": "job-123"}}
        if method == "GET":
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return 200, {"status": {"state": state}}
        raise AssertionError(f"unexpected {method} {url}")


def make_backend(transport=None, logs=None, **kwargs):
    return NebiusJobBackend(
        token="tok",
        project_id="proj-1",
        transport=transport or FakeTransport(),
        logs_runner=(lambda cmd, timeout_s: logs) if logs is not None else (lambda cmd, timeout_s: RESULT_LINE),
        poll_interval_s=0,
        **kwargs,
    )


def test_construction_requires_credentials():
    with pytest.raises(BackendConfigError, match="NEBIUS_IAM_TOKEN"):
        NebiusJobBackend(token=None, project_id="p")
    with pytest.raises(BackendConfigError, match="NEBIUS_PROJECT_ID"):
        NebiusJobBackend(token="t", project_id=None)


def test_build_job_payload_shape(spec):
    backend = make_backend()
    payload = backend.build_job_payload(
        job_name="kernloop-test",
        files={"/opt/kernloop/candidate.py": "def kernel_fn(a, b):\n    pass\n"},
        args="-c 'echo hi'",
    )
    assert payload["metadata"] == {"parentId": "proj-1", "name": "kernloop-test"}
    spec_body = payload["spec"]
    assert spec_body["image"]
    assert spec_body["containerCommand"] == "bash"
    assert spec_body["platform"].startswith("gpu-")
    assert spec_body["preset"]
    assert spec_body["timeout"] == "3600s"
    injected = {f["containerPath"]: base64.b64decode(f["content"]).decode() for f in spec_body["injectedFiles"]}
    assert "def kernel_fn" in injected["/opt/kernloop/candidate.py"]


def test_build_job_payload_rejects_oversized_files(spec):
    backend = make_backend()
    with pytest.raises(ValueError, match="exceeds"):
        backend.build_job_payload("n", {"/big.py": "x" * (64 * 1024 + 1)}, "-c true")


def test_run_benchmark_happy_path(spec):
    transport = FakeTransport(states=("PROVISIONING", "RUNNING", "SUCCEEDED"))
    backend = make_backend(transport=transport)
    timing = backend.run_benchmark(spec, "def kernel_fn(a, b):\n    pass", reps=3, warmups=2)
    assert timing.reps_ms == [10.0, 11.0, 12.0]
    assert timing.warmups == 2
    assert timing.meta["synthetic"] is False
    assert timing.meta["job_id"] == "job-123"
    # one POST (submit) then GETs until terminal
    assert transport.calls[0][0] == "POST"
    assert all(m == "GET" for m, _, _ in transport.calls[1:])


def test_run_correctness_maps_result(spec):
    good = make_backend()
    outcome = good.run_correctness(spec, "def kernel_fn(a, b):\n    pass")
    assert outcome.ok
    assert outcome.detail["backend"] == "nebius"
    assert outcome.detail["rel_l2_error"] == 0.001

    bad_logs = "KERNLOOP_RESULT " + json.dumps({"correct": False, "error": "rel-L2 error 0.5 exceeds tolerance 0.01", "reps_ms": []})
    bad = make_backend(logs=bad_logs)
    outcome = bad.run_correctness(spec, "def kernel_fn(a, b):\n    pass")
    assert not outcome.ok
    assert "rel-L2" in outcome.error


def test_failed_job_maps_to_clean_error(spec):
    transport = FakeTransport(states=("FAILED",))
    backend = make_backend(transport=transport, logs="some log\nKERNEL PANIC\n")
    with pytest.raises(JobExecutionError, match="job-123.*failed"):
        backend.run_benchmark(spec, "def kernel_fn(a, b):\n    pass", reps=1, warmups=0)


def test_missing_result_line_maps_to_clean_error(spec):
    backend = make_backend(logs="nothing useful here\n")
    with pytest.raises(JobExecutionError, match="KERNLOOP_RESULT"):
        backend.run_benchmark(spec, "def kernel_fn(a, b):\n    pass", reps=1, warmups=0)


def test_incorrect_kernel_in_bench_mode_raises(spec):
    logs = "KERNLOOP_RESULT " + json.dumps({"correct": False, "error": "rel-L2 error 0.5", "reps_ms": []})
    backend = make_backend(logs=logs)
    with pytest.raises(JobExecutionError, match="incorrect kernel"):
        backend.run_benchmark(spec, "def kernel_fn(a, b):\n    pass", reps=1, warmups=0)


def test_poll_timeout_raises(spec):
    transport = FakeTransport(states=("RUNNING",))
    backend = make_backend(transport=transport, poll_timeout_s=0.01)
    with pytest.raises(JobExecutionError, match="did not finish"):
        backend.run_benchmark(spec, "def kernel_fn(a, b):\n    pass", reps=1, warmups=0)


def test_checker_args_and_packaging(spec):
    transport = FakeTransport()
    backend = make_backend(transport=transport)
    backend.run_benchmark(spec, "SOURCE", reps=5, warmups=2)
    payload = transport.calls[0][2]
    args = payload["spec"]["args"]
    assert "--mode bench" in args
    assert f"--M {spec.M}" in args and f"--N {spec.N}" in args and f"--K {spec.K}" in args
    assert "--reps 5" in args and "--warmups 2" in args
    assert f"--rtol {spec.correctness_rtol}" in args
    injected = {f["containerPath"]: base64.b64decode(f["content"]).decode() for f in payload["spec"]["injectedFiles"]}
    assert injected["/opt/kernloop/candidate.py"] == "SOURCE"
    assert "KERNLOOP_RESULT" in injected["/opt/kernloop/gpu_checker.py"]
