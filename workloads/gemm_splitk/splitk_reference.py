"""Target shape reference: split-K two-phase Triton GEMM (the bf16_proj pattern).

This is the optimization target the agent is steered toward (see
examples/gemm_spec.json, target_notes). It is a hand-written reference in the
owner's split-K style:

- Phase 1: split K into SPLIT_K chunks. Each program computes one
  (BLOCK_M, BLOCK_N) tile of one K-chunk as a float32 partial product and
  stores it to a (SPLIT_K, M, N) workspace. Masked loads on the K tail.
- Phase 2: one program per output tile sums the SPLIT_K partials and emits
  bf16.

Correctness and timing for this file are validated on the GPU checker
(workloads/gemm_splitk/gpu_checker.py); it cannot run on a Mac. The agent is
never shown this file: it is the hidden reference for what "good" looks like.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _splitk_phase1_kernel(
    a_ptr, b_ptr, w_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_wk, stride_wm, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_end)
        b_mask = (offs_k[:, None] < k_end) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)

    w_ptrs = (
        w_ptr
        + pid_k * stride_wk
        + offs_m[:, None] * stride_wm
        + offs_n[None, :] * stride_wn
    )
    w_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(w_ptrs, acc, mask=w_mask)


@triton.jit
def _splitk_phase2_kernel(
    w_ptr, c_ptr,
    M, N,
    stride_wk, stride_wm, stride_wn,
    stride_cm, stride_cn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        w_ptrs = (
            w_ptr
            + k * stride_wk
            + offs_m[:, None] * stride_wm
            + offs_n[None, :] * stride_wn
        )
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        acc += tl.load(w_ptrs, mask=mask, other=0.0)
    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def kernel_fn(a, b, split_k: int = 8):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    workspace = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32)
    grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)
    _splitk_phase1_kernel[grid1](
        a, b, workspace, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        workspace.stride(0), workspace.stride(1), workspace.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_K=split_k,
        num_stages=4, num_warps=8,
    )
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _splitk_phase2_kernel[grid2](
        workspace, c, M, N,
        workspace.stride(0), workspace.stride(1), workspace.stride(2),
        c.stride(0), c.stride(1),
        SPLIT_K=split_k, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return c
