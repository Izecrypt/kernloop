# Workload: decode-style GEMM (skinny M, large K)

Second workload shape for the v1 demo (generality close in the video). Same
harness and GPU checker as workloads/gemm_splitk (the checker is generic
over M/N/K and is resolved from the shared location when this directory has
no copy).

- `baseline_kernel.py`: naive skinny-M GEMM, one program per N tile, full K
  loop. Correct but low occupancy: with M=16 there are only N/BLOCK_N blocks
  in the grid, far fewer than the GPU has SMs.
- Optimization target: split-K two-phase. Splitting K across programs
  multiplies the grid by SPLIT_K and is the standard decode-GEMM fix.
