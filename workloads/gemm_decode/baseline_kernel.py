"""Baseline for the decode-style workload: skinny-M Triton GEMM.

LLM decode projection shape: M is tiny (a few tokens), N and K are large.
The naive structure here is one program per N tile with a full K loop: it is
correct but leaves the GPU almost idle, because the grid has too few blocks
to fill the SMs. Split-K is the win on this shape.

Contract (enforced by the orchestrator and the GPU checker):
- Complete, self-contained Python file.
- Defines kernel_fn(a, b) -> c with a (M, K) bfloat16 and b (K, N) bfloat16,
  returning c (M, N) bfloat16 on the same device.
- No host-side syncs in the timed path, no data-dependent branching.

This file is what the agent must beat. It is also the embedded
baseline_source in examples/gemm_spec_decode.json (keep them in sync).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=offs_m[:, None] < M)


def kernel_fn(a, b):
    """Skinny-M GEMM: C = A @ B, all bf16. a is (M, K), b is (K, N)."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 128, 64
    grid = (triton.cdiv(N, BLOCK_N),)
    _decode_gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
