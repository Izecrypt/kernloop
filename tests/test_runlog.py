import json

import pytest

from kernloop.runlog import RunLogger, read_log, summarize


def make_record(i):
    return {
        "iteration": i,
        "run_id": "r",
        "timestamp": 1.0,
        "proposal": {"model": "m"},
        "compile": {"ok": True},
        "correctness": {"ok": True},
        "timing": {"mean_ms": 100.0 - i, "min_ms": 90.0 - i, "reps": [1.0], "stdev_ms": 0.1, "warmups": 1, "meta": {}},
        "decision": {"promoted": i % 2 == 1, "reason": "improved", "threshold": 0.02},
        "tokens": {"prompt": 1, "completion": 2},
        "backend": "fake",
    }


def test_logger_appends_and_reads(tmp_path):
    logger = RunLogger(tmp_path / "x.runlog.jsonl", run_id="r", backend="fake")
    from kernloop.runlog import IterationRecord

    for i in (1, 2, 3):
        logger.log_iteration(IterationRecord(**make_record(i)))
    assert logger.header()["run_id"] == "r"
    recs = logger.iterations()
    assert [r["iteration"] for r in recs] == [1, 2, 3]

    # appending to an existing log keeps earlier lines
    logger.log_iteration(IterationRecord(**make_record(4)))
    assert len(read_log(logger.path)) == 4


def test_read_log_rejects_bad_json(tmp_path):
    p = tmp_path / "bad.runlog.jsonl"
    p.write_text('{"type": "run_header"}\nnot json\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        read_log(p)


def test_record_run_id_mismatch_rejected(tmp_path):
    from kernloop.runlog import IterationRecord

    logger = RunLogger(tmp_path / "x.runlog.jsonl", run_id="r", backend="fake")
    rec = IterationRecord(**{**make_record(1), "run_id": "other"})
    with pytest.raises(ValueError):
        logger.log_iteration(rec)


def test_summarize():
    recs = [make_record(i) for i in (1, 2, 3)]
    s = summarize(recs)
    assert s["iterations"] == 3
    assert s["promotions"] == 2
    assert s["best_mean_ms"] == 97.0
    assert s["final_champion_iteration"] == 3
