# STATUS

Last updated: 2026-10-24 (night 1 of the Oct 24 to 30 build window).

## What works now

- The full agent loop, end to end, offline: `kernloop run --spec
  examples/gemm_spec.json --budget 12 --backend fake` runs 12 proposals through
  strict extraction, compile check, correctness, benchmark (5 timed reps,
  2 discarded warmups, mean + min + stdev), and the > 2 percent mean
  promotion rule, logging every iteration as JSONL plus a self-contained HTML
  trajectory page.
- Champion discipline: baseline is timed first; promotions are relative to a
  real measurement; rejections carry reasons ("improved only 0.29%", "compile
  failed", "correctness failed: ...").
- CPU pre-flight mirror (`kernloop preflight`): validates spec shapes, dtype
  rounding, and tolerance on a Mac with no GPU (torch CPU if installed, pure
  Python fallback with an order-independent cross-check).
- Nemotron client (OpenAI-compatible, Token Factory endpoint, model via
  `NEBIUS_MODEL`, retries with backoff, clean errors for missing key or
  package) and the prompt template (spec + champion + benchmark profile,
  code-only output contract).
- Test suite: 37 tests, all passing on a Mac with no GPU, no Triton, no
  network. Covers the loop end to end against the FakeJobBackend and a canned
  LLM stub, promotion boundaries, JSONL validity, HTML rendering, and the
  graceful `nebius` backend error.

## What is stubbed (be honest with judges)

- `NebiusJobBackend` (src/kernloop/nebius_jobs.py) is interface-only. It
  fails fast with "not yet configured". No job packaging, no credential
  handling, no result retrieval yet.
- FakeJobBackend timings are synthetic, produced by a transparent feature
  model of the candidate source. Every fake timing is labeled
  `synthetic: true` in the run log; fake correctness is a heuristic, not an
  executed reference check.
- HeuristicLLM (the offline proposer) applies three staged mutations then
  plateaus. It exists to exercise the loop mechanics, not to find real
  speedups.
- The v1 workload ships the naive tiled GEMM baseline and the split-K
  two-phase target notes; the real split-K kernel files and the GPU-side
  reference checker job are not written yet.

## What remains before Oct 30, 10am PDT

1. Nebius wiring (owner signup tonight): Token Factory account plus Nemotron
   access (`NEBIUS_API_KEY`), Nebius AI Cloud credits, service account token
   for Serverless Jobs. Then implement `NebiusJobBackend`: package (spec,
   checker, candidate) into a job, run correctness plus timed reps with L2
   flushes, return per-rep times.
2. Real Triton workload files: the split-K two-phase kernel, the GPU-side
   correctness checker (vs torch reference), and the benchmark driver with
   cold-cache discipline, all inside the job image.
3. A real Nemotron-driven run on a Nebius GPU, recorded: run log plus HTML
   viewer output that the demo can serve or screenshot.
4. Demo video (under 3 min) per the PRD narrative: spec, loop running,
   trajectory chart improving, final kernel and speedup, generality close.
5. Devpost submission: category, description, demo URL, video URL, repo URL
   (public, MIT), feedback on Token Factory.
6. Nice to have: seed a couple of example run logs in `runs/` for the demo.
