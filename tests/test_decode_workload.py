import json

import pytest

from kernloop.nebius_jobs import FakeJobBackend
from kernloop.orchestrator import Orchestrator
from kernloop.runlog import RunLogger, read_log
from kernloop.workload import WorkloadSpec, cpu_mirror_check, decode_spec
from tests.conftest import EXAMPLES_SPEC_PATH, ScriptedLLM

DECODE_SPEC_PATH = EXAMPLES_SPEC_PATH.parent / "gemm_spec_decode.json"


def improved_decode(source: str) -> str:
    return source.replace(
        "BLOCK_K=BLOCK_K,",
        "BLOCK_K=BLOCK_K, num_stages=4, num_warps=8,",
        1,
    )


def test_decode_spec_examples_json_roundtrip():
    assert DECODE_SPEC_PATH.exists()
    assert WorkloadSpec.load(DECODE_SPEC_PATH) == decode_spec()


def test_decode_spec_is_a_genuinely_different_shape():
    spec = decode_spec()
    proj = WorkloadSpec.load(EXAMPLES_SPEC_PATH)
    assert spec.M <= 16 and proj.M >= 4096
    assert spec.K > proj.K
    assert spec.name != proj.name
    assert spec.workload_dir != proj.workload_dir
    assert "decode" in spec.description.lower()


def test_decode_preflight_without_gpu():
    result = cpu_mirror_check(decode_spec())
    assert result["ok"]
    assert result["backend"] in ("pure-python", "torch-cpu")


def test_decode_embedded_baseline_matches_file():
    spec = decode_spec()
    assert spec.baseline_source in spec.workload_file("baseline_kernel.py").read_text()


def test_decode_uses_shared_gpu_checker():
    spec = decode_spec()
    with pytest.raises(FileNotFoundError):
        spec.workload_file("gpu_checker.py")
    shared = spec.checker_path()
    assert shared.parent.name == "gemm_splitk"
    assert "KERNLOOP_RESULT" in shared.read_text()


def test_decode_fake_backend_loop_end_to_end(tmp_path):
    spec = decode_spec()
    logger = RunLogger(tmp_path / "d.runlog.jsonl", run_id="decode", backend="fake")
    improved = improved_decode(spec.baseline_source)
    llm = ScriptedLLM([improved, improved])
    orch = Orchestrator(
        spec,
        llm=llm,
        backend=FakeJobBackend(seed=0),
        logger=logger,
        budget=3,
        reps=5,
        warmups=2,
    )
    summary = orch.run()

    assert summary.iterations_run == 3
    assert summary.promotions == 1
    assert summary.champion.iteration == 1
    assert summary.champion.mean_ms < 128.0

    lines = (tmp_path / "d.runlog.jsonl").read_text().strip().splitlines()
    assert json.loads(lines[0])["type"] == "run_header"
    records = [json.loads(l) for l in lines[1:]]
    assert len(records) == 3
    assert records[0]["decision"]["promoted"] is True
    assert all(r["backend"] == "fake" for r in records)

    # decode shapes appear in the prompts the proposer saw
    assert "K=14336" in llm.prompts[0]
    assert spec.name in llm.prompts[0]


def test_decode_spec_through_nebius_payload():
    import base64

    from kernloop.nebius_jobs import NebiusJobBackend

    class FakeTransport:
        def __init__(self):
            self.calls = []

        def __call__(self, method, url, token, payload, timeout_s):
            self.calls.append((method, payload))
            if method == "POST":
                return 200, {"metadata": {"id": "job-decode"}}
            return 200, {"status": {"state": "SUCCEEDED"}}

    result = json.dumps({"correct": True, "reps_ms": [1.0], "mean_ms": 1.0, "min_ms": 1.0})
    backend = NebiusJobBackend(
        token="t",
        project_id="p",
        transport=FakeTransport(),
        logs_runner=lambda cmd, timeout_s: "KERNLOOP_RESULT " + result,
        poll_interval_s=0,
    )
    timing = backend.run_benchmark(decode_spec(), "def kernel_fn(a, b):\n    pass", reps=1, warmups=0)
    assert timing.reps_ms == [1.0]
    payload = backend._transport.calls[0][1]
    assert "--K 14336" in payload["spec"]["args"]
    injected = {f["containerPath"]: base64.b64decode(f["content"]).decode() for f in payload["spec"]["injectedFiles"]}
    assert "KERNLOOP_RESULT" in injected["/opt/kernloop/gpu_checker.py"]
