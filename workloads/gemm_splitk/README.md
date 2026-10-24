# Workload: bf16 projection GEMM, split-K two-phase (v1)

Files in this directory run on the GPU job backend (Nebius Serverless Jobs,
Linux + CUDA + Triton + torch). Nothing here imports on a machine without
Triton; local pre-flight validation uses `kernloop preflight` (CPU mirror)
and the `fake` backend.

- `baseline_kernel.py`: naive tiled Triton GEMM, `kernel_fn(a, b)` contract,
  the starting champion. Same source as `baseline_source` in
  examples/gemm_spec.json.
- `splitk_reference.py`: hand-written split-K two-phase reference (phase 1
  writes float32 partials to a workspace, phase 2 reduces to bf16). Hidden
  from the agent; used to validate the harness and to sanity-check job
  timing once the Nebius backend is live.
- `gpu_checker.py`: correctness (rel-L2 vs float32 torch reference) plus
  benchmark with L2 flush between timed reps (cold cache), mean and min
  reporting. Prints one `KERNLOOP_RESULT {json}` line that
  `NebiusJobBackend` parses from the job logs.
