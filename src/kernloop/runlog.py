"""Append-only JSONL run log for KernLoop.

Every iteration of the agent loop appends exactly one JSON object (one line).
The log is the source of truth for the trajectory: proposal, compile result,
correctness, timings, champion decision, and token usage per iteration.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1


@dataclass
class Decision:
    """Champion promotion decision for one iteration."""

    promoted: bool
    reason: str
    threshold: float  # relative improvement required, e.g. 0.02


@dataclass
class IterationRecord:
    """One line of the run log."""

    iteration: int
    run_id: str
    timestamp: float
    proposal: dict[str, Any]
    compile: dict[str, Any]
    correctness: dict[str, Any]
    timing: dict[str, Any]
    decision: dict[str, Any]
    tokens: dict[str, int]
    backend: str

    def to_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class RunLogger:
    """Append-only logger. Safe to open an existing log to continue appending."""

    def __init__(self, path: str | Path, run_id: str, backend: str):
        self.path = Path(path)
        self.run_id = run_id
        self.backend = backend
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            header = {
                "schema": SCHEMA_VERSION,
                "type": "run_header",
                "run_id": run_id,
                "created": time.time(),
                "backend": backend,
            }
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(header, sort_keys=True) + "\n")

    def log_iteration(self, record: IterationRecord) -> None:
        if record.run_id != self.run_id or record.backend != self.backend:
            raise ValueError("record run_id/backend does not match logger")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_line() + "\n")

    def log_research(self, note) -> None:
        """Append a research record (Tavily query + snippets) to the run log."""

        line = {
            "type": "research",
            "run_id": self.run_id,
            "timestamp": time.time(),
            "backend": self.backend,
            **note.to_dict(),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")

    def header(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        if first.get("type") != "run_header":
            raise ValueError(f"{self.path} is not a KernLoop run log")
        return first

    def iterations(self) -> list[dict[str, Any]]:
        out = []
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "iteration" not in rec:  # header, research, or future event lines
                    continue
                rec["_lineno"] = lineno
                out.append(rec)
        return out


def read_log(path: str | Path) -> list[dict[str, Any]]:
    """Read iterations from a run log, validating each line as JSON."""

    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {lineno} of {path}: {exc}") from exc
            if rec.get("type") == "run_header":
                continue
            out.append(rec)
    return out


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate stats over a trajectory for CLI output."""

    recs = list(records)
    promoted = [r for r in recs if r.get("decision", {}).get("promoted")]
    valid = [r for r in recs if r.get("timing", {}).get("mean_ms") is not None]
    best = min((r["timing"]["mean_ms"] for r in valid), default=None)
    return {
        "iterations": len(recs),
        "promotions": len(promoted),
        "best_mean_ms": best,
        "final_champion_iteration": promoted[-1]["iteration"] if promoted else None,
    }
