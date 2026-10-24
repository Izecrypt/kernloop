"""GPU-side checker and benchmark driver for KernLoop candidates.

Runs inside the Nebius Serverless Job container (Linux + CUDA + Triton +
torch). Loads a candidate kernel file, checks it against a torch reference
with a relative L2 tolerance, then, in bench mode, runs warmup reps followed
by timed reps with an L2 flush between every rep (cold-cache discipline) and
reports mean and min.

Usage:
    python gpu_checker.py --candidate /path/candidate.py \
        --M 4096 --N 4096 --K 4096 --rtol 0.01 --mode bench --reps 5 --warmups 2

Output: exactly one line on stdout:
    KERNLOOP_RESULT {"correct": true, "reps_ms": [...], "mean_ms": x, ...}
The NebiusJobBackend parses this line from the job logs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

RESULT_PREFIX = "KERNLOOP_RESULT "

DTYPES = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


def _load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("kernloop_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "kernel_fn", None)
    if not callable(fn):
        raise RuntimeError("candidate defines no callable kernel_fn")
    return fn


def _rel_l2(out, ref) -> float:
    diff = (out.float() - ref.float()).norm().item()
    denom = ref.float().norm().item()
    return diff / max(denom, 1e-12)


def _l2_flush_buffer(device):
    # H100 L2 is 50 MB; write a 256 MB buffer to evict resident cache lines.
    import torch

    buf = torch.empty(256 * 1024 * 1024 // 4, device=device, dtype=torch.float32)
    return buf


def run(args) -> dict:
    import torch

    torch.manual_seed(args.seed)
    dtype = getattr(torch, DTYPES[args.dtype])
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_checker requires a CUDA device (run on the Nebius job VM)")

    kernel_fn = _load_candidate(args.candidate)

    a = torch.randn(args.M, args.K, device=device).to(dtype)
    b = torch.randn(args.K, args.N, device=device).to(dtype)
    ref = (a.float() @ b.float()).to(dtype)

    out = kernel_fn(a, b)
    rel_err = _rel_l2(out, ref)
    correct = rel_err <= args.rtol

    result = {
        "correct": correct,
        "rel_l2_error": rel_err,
        "rtol": args.rtol,
        "shapes": {"M": args.M, "N": args.N, "K": args.K},
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0),
        "reps_ms": [],
        "mean_ms": None,
        "min_ms": None,
    }
    if not correct:
        result["error"] = f"rel-L2 error {rel_err:.6f} exceeds tolerance {args.rtol}"
        return result

    if args.mode == "bench":
        flush = _l2_flush_buffer(device)
        try:
            for _ in range(args.warmups):
                flush.zero_()
                torch.cuda.synchronize()
                kernel_fn(a, b)
                torch.cuda.synchronize()
            reps = []
            for _ in range(args.reps):
                flush.zero_()  # cold cache: evict L2 before every timed rep
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                kernel_fn(a, b)
                end.record()
                torch.cuda.synchronize()
                reps.append(start.elapsed_time(end))
        finally:
            del flush
        result["reps_ms"] = reps
        result["mean_ms"] = sum(reps) / len(reps)
        result["min_ms"] = min(reps)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="KernLoop GPU correctness checker and benchmark")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--mode", choices=["check", "bench"], default="check")
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        result = run(args)
        code = 0 if result["correct"] else 1
    except Exception as exc:  # never exit without a parseable result line
        result = {"correct": False, "error": f"{type(exc).__name__}: {exc}", "reps_ms": []}
        code = 1
    print(RESULT_PREFIX + json.dumps(result))
    return code


if __name__ == "__main__":
    sys.exit(main())
