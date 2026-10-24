import pytest

from kernloop import workload
from kernloop.workload import WorkloadSpec, cpu_mirror_check, default_spec


def test_default_spec_roundtrip():
    spec = default_spec()
    assert spec.M == spec.N == spec.K == 4096
    assert spec.dtype == "bfloat16"
    assert "kernel_fn" in spec.baseline_source
    assert "split" in spec.target_notes.lower()


def test_examples_json_matches_default_spec(spec_from_examples_json):
    assert spec_from_examples_json == default_spec()


def test_cpu_mirror_check_passes_without_gpu(spec):
    result = cpu_mirror_check(spec)
    assert result["ok"]
    assert result["max_abs_err"] <= spec.correctness_rtol
    # pure Python fallback or torch CPU: either is honest, both run on a Mac
    assert result["backend"] in ("pure-python", "torch-cpu")


def test_cpu_mirror_check_detects_bad_tolerance(spec):
    spec.correctness_rtol = -1.0  # impossible tolerance must fail
    assert not cpu_mirror_check(spec)["ok"]


def test_package_imports_without_triton():
    assert hasattr(workload, "HAS_TRITON")


def test_load_kernel_source_guards_triton(spec):
    if workload.HAS_TRITON:
        pytest.skip("Triton is installed on this host")
    with pytest.raises(RuntimeError, match="Triton is not installed"):
        workload.load_kernel_source(spec.baseline_source)


def test_spec_load_from_dict_roundtrip(spec, tmp_path):
    import dataclasses, json

    p = tmp_path / "spec.json"
    p.write_text(json.dumps(dataclasses.asdict(spec)))
    assert WorkloadSpec.load(p) == spec
