import pytest

from kernloop.harness import compile_check, make_stats, should_promote
from kernloop.workload import NAIVE_TILED_GEMM


def test_compile_check_accepts_valid_kernel():
    res = compile_check(NAIVE_TILED_GEMM)
    assert res.ok
    assert res.entrypoint_found
    assert res.n_functions >= 2


def test_compile_check_rejects_syntax_error():
    res = compile_check("def kernel_fn(:\n    pass")
    assert not res.ok
    assert "syntax error" in res.error


def test_compile_check_requires_kernel_fn():
    res = compile_check("import triton\n\ndef helper():\n    pass\n")
    assert not res.ok
    assert "kernel_fn" in res.error


def test_should_promote_requires_more_than_two_percent():
    promoted, reason = should_promote(97.9, 100.0)
    assert promoted
    promoted, _ = should_promote(98.0, 100.0)  # exactly 2 percent is not enough
    assert not promoted
    promoted, reason = should_promote(98.5, 100.0)
    assert not promoted
    assert "2%" in reason or "2 percent" in reason or "2.0%" in reason


def test_should_promote_rejects_regressions_and_sanity():
    promoted, _ = should_promote(120.0, 100.0)
    assert not promoted
    promoted, reason = should_promote(50.0, 0.0)
    assert not promoted
    assert "not positive" in reason


def test_make_stats():
    stats = make_stats([10.0, 12.0, 11.0], warmups=2)
    assert stats.mean_ms == pytest.approx(11.0)
    assert stats.min_ms == 10.0
    assert stats.stdev_ms is not None
    assert stats.warmups == 2


def test_make_stats_validates_input():
    with pytest.raises(ValueError):
        make_stats([])
    with pytest.raises(ValueError):
        make_stats([1.0, -2.0])
    with pytest.raises(ValueError):
        make_stats([1.0, float("nan")])
