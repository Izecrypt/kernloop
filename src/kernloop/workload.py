"""The v1 workload: a split-K two-phase Triton GEMM.

The workload defines the problem the agent loop optimizes:

- A naive tiled GEMM baseline (Triton source, kept as a string so the package
  imports fine without Triton installed).
- The optimization target: a split-K two-phase structure (phase 1: partial
  products into a workspace with atomics or a reduction buffer; phase 2:
  reduce), the bf16 projection pattern the owner knows cold.
- A CPU correctness mirror so pre-flight validation runs on a machine with no
  GPU. Uses torch on CPU when torch is available, otherwise a pure Python
  mirror on small shapes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

try:  # Triton is only needed on the GPU job runner, never locally.
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# The starting point the agent must beat: a naive tiled GEMM, one program per
# (BLOCK_M, BLOCK_N) tile, simple loop over K. No split-K, no pipelining hints.
NAIVE_TILED_GEMM = '''\
import torch
import triton
import triton.language as tl


@triton.jit
def _naive_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


def kernel_fn(a, b):
    """Naive tiled GEMM: C = A @ B, all bf16. a is (M, K), b is (K, N)."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _naive_gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
'''

SPLIT_K_TARGET_NOTES = """\
Optimization target: split-K two-phase GEMM in Triton.
Phase 1: split the K dimension across programs; each computes a partial
product into a float32 workspace (separate program per (m-tile, n-tile,
k-split)). Phase 2: a lightweight reduction kernel sums partials and emits
bf16. Use masked loads on the K tail, pick BLOCK sizes for the 4096-wide K,
and consider num_warps / num_stages for the H100. The entrypoint must remain
kernel_fn(a, b) -> c with a (M, K) bf16 and b (K, N) bf16.
"""


@dataclass
class WorkloadSpec:
    name: str
    description: str
    M: int
    N: int
    K: int
    dtype: str
    baseline_source: str
    target_notes: str
    correctness_rtol: float = 1e-2
    workload_dir: str = "workloads/gemm_splitk"

    @property
    def problem_summary(self) -> str:
        return (
            f"{self.name}: {self.dtype} GEMM C = A @ B with "
            f"M={self.M}, N={self.N}, K={self.K}. {self.description}"
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(data: dict) -> "WorkloadSpec":
        return WorkloadSpec(**data)

    @staticmethod
    def load(path: str | Path) -> "WorkloadSpec":
        with Path(path).open("r", encoding="utf-8") as fh:
            return WorkloadSpec.from_dict(json.load(fh))

    def workload_file(self, name: str) -> Path:
        """Resolve a file inside the workload dir (checker, kernels).

        Tries the path as given (repo checkout), then relative to the package
        repo root, so it works from the repo and from the installed CLI.
        """

        repo_root = Path(__file__).resolve().parents[2]
        for base in (Path.cwd(), repo_root):
            candidate = base / self.workload_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"workload file {name!r} not found under {self.workload_dir!r}")


def require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. Triton kernels can only run on the GPU "
            "job backend (Nebius Serverless Jobs); local pre-flight checks do "
            "not need it. Install with: pip install triton (Linux + GPU only)."
        )


def load_kernel_source(source: str):
    """Compile candidate Triton source into a module (requires Triton + GPU host)."""

    require_triton()
    namespace: dict = {}
    exec(compile(source, "<candidate>", "exec"), namespace)  # noqa: S102
    fn = namespace.get("kernel_fn")
    if not callable(fn):
        raise RuntimeError("candidate defines no callable kernel_fn")
    return fn


def _dot_f32(seq) -> float:
    # Plain left-to-right float accumulation on purpose: built-in sum() uses
    # Neumaier compensation since 3.12, which would hide summation-order
    # effects from the cross-check.
    total = 0.0
    for x in seq:
        total += x
    return total


def _mirror_pure_python(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    # Deliberately different summation order (reverse over K) than the
    # cross-check below, so float non-associativity is actually exercised.
    n = len(b[0])
    k = len(b)
    out = []
    for row in a:
        out_row = []
        for j in range(n):
            terms = [row[i] * b[i][j] for i in reversed(range(k))]
            out_row.append(_dot_f32(terms))
        out.append(out_row)
    return out


def cpu_mirror_check(spec: WorkloadSpec, m: int = 48, n: int = 32, k: int = 40, seed: int = 0) -> dict:
    """Pre-flight correctness mirror on CPU, no GPU required.

    Builds random bf16-rounded inputs at small shapes, computes the reference
    GEMM with torch (CPU) when available, else a pure Python mirror, and
    verifies the mirror against an independent recomputation. This validates
    the spec shapes, dtype rounding, and the tolerance the GPU correctness
    check will use, all before any GPU job is launched.
    """

    import random

    rng = random.Random(seed)

    def value() -> float:
        # Wide dynamic range so summation-order rounding differences between
        # the mirror and the cross-check are actually visible.
        return round(rng.uniform(-1, 1) * (10 ** rng.uniform(-2, 2)), 6)

    a = [[value() for _ in range(k)] for _ in range(m)]
    b = [[value() for _ in range(n)] for _ in range(k)]

    if HAS_TORCH:
        ta = torch.tensor(a, dtype=torch.bfloat16).float()
        tb = torch.tensor(b, dtype=torch.bfloat16).float()
        ref = (ta @ tb).to(torch.bfloat16).float()
        ref_list = ref.tolist()
        backend = "torch-cpu"
    else:
        ref_list = _mirror_pure_python(a, b)
        backend = "pure-python"

    # Independent recomputation with a different summation order (column-major
    # walk) to cross-check the reference itself.
    cross = [[0.0] * n for _ in range(m)]
    for j in range(n):
        for i in range(m):
            cross[i][j] = _dot_f32(a[i][kk] * b[kk][j] for kk in range(k))

    max_err = max(abs(ref_list[i][j] - cross[i][j]) for i in range(m) for j in range(n))
    ok = max_err <= spec.correctness_rtol
    return {
        "ok": ok,
        "backend": backend,
        "shapes": {"m": m, "n": n, "k": k},
        "max_abs_err": max_err,
        "rtol": spec.correctness_rtol,
        "workload": spec.name,
    }


def default_spec() -> WorkloadSpec:
    return WorkloadSpec(
        name="bf16_proj_gemm_splitk",
        description=(
            "Projection-layer GEMM (the bf16_proj pattern): square-ish activations "
            "times a wide K reduction. The naive tiled baseline underutilizes the "
            "GPU on K-heavy shapes; split-K two-phase recovers occupancy."
        ),
        M=4096,
        N=4096,
        K=4096,
        dtype="bfloat16",
        baseline_source=NAIVE_TILED_GEMM,
        target_notes=SPLIT_K_TARGET_NOTES,
    )
