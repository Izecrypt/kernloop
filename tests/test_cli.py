import json
from pathlib import Path

from kernloop.cli import main
from tests.conftest import EXAMPLES_SPEC_PATH


def test_cli_run_fake_end_to_end(tmp_path, capsys):
    out_dir = tmp_path / "runs"
    rc = main(
        [
            "run",
            "--spec", str(EXAMPLES_SPEC_PATH),
            "--budget", "3",
            "--backend", "fake",
            "--llm", "heuristic",
            "--seed", "0",
            "--out", str(out_dir),
            "--run-id", "clitest",
        ]
    )
    assert rc == 0
    log = out_dir / "clitest.runlog.jsonl"
    html = out_dir / "clitest.html"
    assert log.exists() and html.exists()

    lines = log.read_text().strip().splitlines()
    header = json.loads(lines[0])
    assert header["run_id"] == "clitest"
    records = [json.loads(l) for l in lines[1:]]
    assert len(records) == 3
    assert any(r["decision"]["promoted"] for r in records)

    page = html.read_text()
    assert "KernLoop trajectory" in page
    assert "clitest" in page

    captured = capsys.readouterr().out
    assert "promotions" in captured
    assert "run log:" in captured and "viewer:" in captured


def test_cli_nebius_backend_graceful_error(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("NEBIUS_JOBS_TOKEN", raising=False)
    rc = main(
        [
            "run",
            "--spec", str(EXAMPLES_SPEC_PATH),
            "--budget", "1",
            "--backend", "nebius",
            "--out", str(tmp_path),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "not yet configured" in err


def test_cli_preflight(capsys):
    rc = main(["preflight", "--spec", str(EXAMPLES_SPEC_PATH)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True


def test_cli_view(tmp_path):
    out_dir = tmp_path / "runs"
    main(
        [
            "run",
            "--spec", str(EXAMPLES_SPEC_PATH),
            "--budget", "1",
            "--backend", "fake",
            "--llm", "heuristic",
            "--out", str(out_dir),
            "--run-id", "viewtest",
        ]
    )
    dest = tmp_path / "page.html"
    assert main(["view", "--log", str(out_dir / "viewtest.runlog.jsonl"), "--out", str(dest)]) == 0
    assert dest.exists()
