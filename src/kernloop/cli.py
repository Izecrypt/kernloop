"""Command line interface: kernloop run / kernloop view / kernloop preflight."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from .harness import PROMOTION_THRESHOLD
from .nemotron_client import LLMConfigError, get_llm
from .nebius_jobs import BackendConfigError, get_backend
from .orchestrator import Orchestrator
from .runlog import RunLogger, summarize
from .viewer import render_log_to_html
from .workload import cpu_mirror_check


def _cmd_run(args: argparse.Namespace) -> int:
    from .workload import WorkloadSpec

    spec = WorkloadSpec.load(args.spec)
    run_id = args.run_id or uuid.uuid4().hex[:12]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_id}.runlog.jsonl"

    try:
        backend = get_backend(args.backend, seed=args.seed)
    except BackendConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        llm = get_llm(args.llm, seed=args.seed)
    except LLMConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    preflight = cpu_mirror_check(spec)
    if not preflight["ok"]:
        print(f"error: CPU pre-flight mirror check failed: {preflight}", file=sys.stderr)
        return 2
    print(f"pre-flight CPU mirror check ok ({preflight['backend']}, max err {preflight['max_abs_err']:.3g})")

    logger = RunLogger(log_path, run_id=run_id, backend=args.backend)
    orchestrator = Orchestrator(
        spec,
        llm=llm,
        backend=backend,
        logger=logger,
        budget=args.budget,
        reps=args.reps,
        warmups=args.warmups,
        threshold=PROMOTION_THRESHOLD,
    )
    print(
        f"run {run_id}: budget={args.budget} backend={args.backend} llm={llm.model} "
        f"reps={args.reps} warmups={args.warmups}"
    )
    summary = orchestrator.run()

    html_path = render_log_to_html(log_path, out_dir / f"{run_id}.html")
    stats = summarize(logger.iterations())
    print(f"done: {summary.iterations_run} iterations, {summary.promotions} promotions")
    print(f"champion: iteration {summary.champion.iteration}, mean {summary.champion.mean_ms:.4f} ms, min {summary.champion.min_ms:.4f} ms")
    print(f"tokens: {summary.tokens['prompt']} prompt + {summary.tokens['completion']} completion")
    print(f"run log: {log_path}")
    print(f"viewer:  {html_path}")
    if stats["best_mean_ms"] is not None:
        print(f"best mean {stats['best_mean_ms']:.4f} ms")
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    out = render_log_to_html(args.log, args.out)
    print(f"wrote {out}")
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    from .workload import WorkloadSpec

    spec = WorkloadSpec.load(args.spec)
    result = cpu_mirror_check(spec)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernloop",
        description="Autonomous GPU kernel engineer (Nebius x NVIDIA Global AI Hackathon).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the agent loop on a kernel problem spec")
    run.add_argument("--spec", required=True, help="path to the workload spec JSON")
    run.add_argument("--budget", type=int, default=12, help="max proposals per run (default 12)")
    run.add_argument("--backend", default="fake", choices=["fake", "nebius"], help="execution backend")
    run.add_argument("--llm", default="auto", choices=["auto", "nemotron", "heuristic"], help="proposal source")
    run.add_argument("--reps", type=int, default=5, help="timed benchmark reps per candidate")
    run.add_argument("--warmups", type=int, default=2, help="untimed warmup reps, discarded")
    run.add_argument("--seed", type=int, default=0, help="rng seed for the fake backend and offline proposer")
    run.add_argument("--out", default="runs", help="output directory for run log and HTML viewer")
    run.add_argument("--run-id", default=None, help="run id (default: random)")
    run.set_defaults(func=_cmd_run)

    view = sub.add_parser("view", help="render a run log to a self-contained HTML trajectory page")
    view.add_argument("--log", required=True, help="path to a .runlog.jsonl file")
    view.add_argument("--out", required=True, help="output HTML path")
    view.set_defaults(func=_cmd_view)

    pre = sub.add_parser("preflight", help="CPU correctness mirror check (no GPU needed)")
    pre.add_argument("--spec", required=True, help="path to the workload spec JSON")
    pre.set_defaults(func=_cmd_preflight)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
