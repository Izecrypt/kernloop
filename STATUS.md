# STATUS

Last updated: 2026-10-24 (night 2 of the Oct 24 to 30 build window).

## What works now

- The full agent loop, end to end, offline: `kernloop run --spec
  examples/gemm_spec.json --budget 12 --backend fake` runs 12 proposals through
  strict extraction, compile check, correctness, benchmark (5 timed reps,
  2 discarded warmups, mean + min + stdev), and the > 2 percent mean
  promotion rule, logging every iteration as JSONL plus a self-contained HTML
  trajectory page.
- Champion discipline: baseline is timed first; promotions are relative to a
  real measurement; rejections carry reasons.
- CPU pre-flight mirror (`kernloop preflight`): validates spec shapes, dtype
  rounding, and tolerance with no GPU.
- Nemotron client (OpenAI-compatible, Token Factory endpoint, model via
  `NEBIUS_MODEL`, backoff retries, clean missing-key errors) with an
  engineering-grade prompt: spec + champion + profile + recent rejection
  history, the knobs the model may change (block sizes, num_warps/num_stages,
  split-K), forbidden patterns (host syncs, data-dependent branching,
  contract changes), and the code-only output contract.
- Real workload files (workloads/gemm_splitk/): naive tiled baseline, a
  hand-written split-K two-phase reference kernel (the hidden target shape),
  and a GPU checker (rel-L2 vs float32 torch reference, L2 flush before every
  timed rep, mean + min, one parseable KERNLOOP_RESULT line).
- Nebius Serverless Jobs backend (src/kernloop/nebius_jobs.py): full REST
  implementation against the documented Jobs API (submit, poll, injected
  files, result parsing, failure mapping), with injectable transport and log
  runner so the whole flow is unit tested offline.
- Test suite: 50 tests, all passing on a Mac with no GPU, no Triton, no
  network.

## Tested live vs structurally complete (be honest with judges)

- Tested live: nothing on Nebius. The owner has not finished signup, so no
  credential has touched this code.
- Structurally complete, unit tested offline: everything above, including
  the Nebius REST backend (payload shape per docs.nebius.com/serverless,
  submit/poll/parse against a fake transport, failure mapping).
- Known unknowns, marked TODO(nebius) in the code: the exact job state enum
  (we match defensively on status.state/status.phase), log retrieval shells
  out to `nebius ai job logs` because the REST API exposes no logs endpoint,
  and the job image is a CUDA devel placeholder that pip-installs torch and
  triton at start instead of a pinned project image.

## What is simulated

- FakeJobBackend timings are synthetic (transparent feature model of the
  candidate source), always labeled `synthetic: true`; its correctness check
  is a heuristic, not an executed reference.
- HeuristicLLM (offline proposer) applies three staged mutations then
  plateaus; it exists to exercise loop mechanics, not to find real speedups.

## What remains before Oct 30, 10am PDT

1. Owner Nebius signup (the blocker for everything below): Token Factory key
   (`NEBIUS_API_KEY`), AI Cloud project plus IAM token
   (`NEBIUS_IAM_TOKEN`, `NEBIUS_PROJECT_ID`).
2. First live run of the Nebius backend: build and push a pinned job image
   (torch + triton), confirm the state enum and the logs path, run one
   candidate through `run_correctness` and `run_benchmark` for real.
3. A full Nemotron-driven run on a Nebius GPU (`--backend nebius --llm
   nemotron`): recorded run log plus HTML viewer for the demo.
4. Demo video per SUBMISSION.md shot list (under 3 min).
5. Devpost submission per SUBMISSION.md checklist (repo public, video URL,
   demo URL, feedback paragraph).
6. Nice to have: seed example run logs in `runs/`, second workload shape for
   the generality close in the video.
