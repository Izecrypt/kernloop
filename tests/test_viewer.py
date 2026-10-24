import json

from kernloop.runlog import RunLogger, IterationRecord
from kernloop.viewer import render_html, render_log_to_html


def record(i, mean, promoted, reason, compile_ok=True, correct_ok=True, model="m<script>"):
    return IterationRecord(
        iteration=i,
        run_id="r",
        timestamp=1.0,
        proposal={"model": model},
        compile={"ok": compile_ok, "error": None if compile_ok else "bad"},
        correctness={"ok": correct_ok, "error": None if correct_ok else "wrong"},
        timing={"mean_ms": mean, "min_ms": None if mean is None else mean - 1, "reps": [] if mean is None else [mean], "stdev_ms": 0.1, "warmups": 1, "meta": {}},
        decision={"promoted": promoted, "reason": reason, "threshold": 0.02},
        tokens={"prompt": 1, "completion": 2},
        backend="fake",
    )


def test_viewer_renders_single_self_contained_page(tmp_path):
    logger = RunLogger(tmp_path / "r.runlog.jsonl", run_id="r", backend="fake")
    logger.log_iteration(record(1, 100.0, True, "improved 5.0%"))
    logger.log_iteration(record(2, 99.5, False, "improved only 0.5% <script>alert(1)</script>"))
    logger.log_iteration(record(3, None, False, "compile failed", compile_ok=False))

    out = render_log_to_html(logger.path, tmp_path / "out.html")
    html = out.read_text()
    assert "<svg" in html and "</svg>" in html
    assert "champion" in html
    assert "promoted" in html and "rejected" in html
    # single file: no external assets, no scripts
    assert "<script" not in html
    assert 'src="http' not in html and 'href="http' not in html
    # everything is escaped
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_viewer_chart_geometry():
    def rec(i, mean, promoted):
        return {
            "iteration": i,
            "timing": {"mean_ms": mean, "min_ms": mean - 1},
            "decision": {"promoted": promoted, "reason": "x"},
            "compile": {"ok": True},
            "correctness": {"ok": True},
            "proposal": {"model": "m"},
        }

    html = render_html([rec(1, 100.0, True), rec(2, 90.0, True)], "r", "fake")
    assert "<circle" in html
    assert 'fill="#188038"' in html  # promoted markers are green
    assert "<svg" in html


def test_viewer_handles_empty_log(tmp_path):
    logger = RunLogger(tmp_path / "r.runlog.jsonl", run_id="r", backend="fake")
    out = render_log_to_html(logger.path, tmp_path / "out.html")
    assert "No timed iterations" in out.read_text()
