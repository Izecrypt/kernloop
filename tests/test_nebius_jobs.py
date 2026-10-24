import pytest

from kernloop.nebius_jobs import BackendConfigError, FakeJobBackend, NebiusJobBackend, get_backend
from tests.conftest import improved_kernel


def test_fake_backend_is_deterministic(spec):
    backend = FakeJobBackend(seed=0)
    a = backend.run_benchmark(spec, spec.baseline_source, reps=5, warmups=2)
    b = backend.run_benchmark(spec, spec.baseline_source, reps=5, warmups=2)
    assert a.reps_ms == b.reps_ms
    assert len(a.reps_ms) == 5
    assert a.warmups == 2
    assert a.meta["synthetic"] is True


def test_fake_backend_rewards_real_optimizations(spec):
    backend = FakeJobBackend(seed=0)
    base = backend.run_benchmark(spec, spec.baseline_source, reps=7, warmups=2)
    improved = backend.run_benchmark(spec, improved_kernel(spec.baseline_source), reps=7, warmups=2)
    assert sum(improved.reps_ms) < sum(base.reps_ms)
    assert "num_stages=4" in " ".join(improved.meta["features"])


def test_fake_backend_correctness(spec):
    backend = FakeJobBackend(seed=0)
    assert backend.run_correctness(spec, spec.baseline_source).ok
    bad = backend.run_correctness(spec, "def kernel_fn(a, b):\n    return a\n")
    assert not bad.ok


def test_nebius_backend_errors_not_yet_configured():
    with pytest.raises(BackendConfigError, match="not yet configured"):
        NebiusJobBackend()


def test_get_backend_unknown():
    with pytest.raises(BackendConfigError, match="unknown backend"):
        get_backend("nope")


def test_get_backend_nebius_graceful_without_token(monkeypatch):
    monkeypatch.delenv("NEBIUS_JOBS_TOKEN", raising=False)
    with pytest.raises(BackendConfigError, match="not yet configured"):
        get_backend("nebius")
