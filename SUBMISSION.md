# SUBMISSION (Devpost draft)

Working draft for the Nebius x NVIDIA Global AI Hackathon Devpost submission,
due Oct 30, 10am PDT. Fill the bracketed items as they land.

## Basics

- **Project name:** KernLoop
- **Tagline:** An autonomous GPU kernel engineer: Nemotron writes the Triton
  kernels, Nebius GPUs grade them, and only statistically real speedups
  become champion.
- **Track:** Coding and Agentic Engineering
- **Repo:** https://github.com/Izecrypt/kernloop (public, MIT)
- **Demo:** https://izecrypt.github.io/kernloop/ (hosted trajectory viewer for run 8d267696a355, the run in the video)
- **Video:** rendered at /Users/tbo/Desktop/kernloop-demo-nebius.mp4 (82s, under 3 min); needs a public YouTube upload, then paste the URL here

## Description (for judges)

**What it is.** KernLoop is an agent loop that autonomously optimizes GPU
kernels. You give it a kernel problem spec (shapes, dtype, a naive baseline,
and correctness tolerance). It asks a reasoning LLM, NVIDIA Nemotron served
through Nebius Token Factory, to propose a Triton kernel variant given the
current champion and its measured benchmark profile. The candidate is
validated for real: a GPU job on Nebius AI Cloud (Serverless Jobs) checks it
against a float32 torch reference with a relative-L2 tolerance, then times it
with cold-cache discipline (L2 flush before every timed rep, warmup reps
discarded, mean plus min over reps). A candidate becomes the new champion
only if its mean beats the champion mean by more than 2 percent, otherwise
it is rejected with the reason logged. The loop repeats within a fixed
budget and writes an append-only JSONL run log plus a self-contained HTML
trajectory page a human can inspect.
**Why.** Kernel optimization is the hardest real codebase class to hand an
agent: correctness is exacting, speedups are measured in microseconds, and a
wrong answer that is fast is worthless. The interesting agentic engineering
is not "write code" but "propose, falsify, and keep honest score": strict
output contracts, executed correctness checks, statistical promotion
criteria, and a full audit trail of every decision.

**How it works (architecture).** `orchestrator.py` runs the loop;
`nemotron_client.py` is an OpenAI-compatible client for the Token Factory
endpoint whose prompt carries the spec, the champion kernel, the benchmark
profile, and the recent rejection history, with an explicit code-only output
contract (knobs the model may change: block sizes, num_warps/num_stages,
split-K structure; forbidden: host syncs, data-dependent branching, contract
changes). `nebius_jobs.py` implements the Serverless Jobs backend: the
candidate file and the GPU checker are injected into the job, the checker
prints one parseable result line, and the backend maps every failure mode
(job failed, no result line, timeout) to a clean error the loop logs.
`research.py` optionally calls the Tavily search API at run start and after 3
consecutive rejections, injecting fresh optimization notes into the prompt
when the proposer is stuck. `runlog.py` and `viewer.py` turn the run into an
inspectable artifact.

**What is real vs simulated, as of submission.** Be precise with judges:
(1) The agent loop, promotion rule, run log, HTML viewer, prompt pipeline,
Tavily research step, and Nebius Serverless Jobs REST backend are implemented
and unit tested (70 tests, all passing, no GPU required for the suite). (2) The v1 workload is a
bf16 projection GEMM with a naive tiled baseline and a hand-written split-K
two-phase reference kernel; the GPU checker enforces rel-L2 correctness and
cold-cache timing. (3) The recorded demo run executes the full loop with the
fake backend (a deterministic feature-model simulator, clearly labeled
synthetic in every log line); the LLM proposals in that run are real, from
NVIDIA Nemotron via the NVIDIA API, and the Tavily research calls are real
API calls. The Nebius path (Token Factory endpoint plus AI Cloud Serverless
Jobs) is fully implemented in the repo and documented in the README, but no
live Nebius job had run at submission time because account billing could not
be activated from our region during the hackathon window. Every run log
records which backend and which model produced each number, so real and
simulated results are never mixed, and the Devpost can be updated with a
live Nebius run during judging.

**Origin story (two lines).** The founder competes on the OKC (Open Kernel
Challenge) leaderboard and has written split-K two-phase GEMMs by hand for
production inference work; KernLoop productizes that exact practitioner loop
(propose, validate on GPU, keep champions, never trust an unmeasured
speedup) into an open source agent anyone can point at a kernel problem.

## Feedback on Nebius (Token Factory and AI Cloud)

Nebius Token Factory's OpenAI-compatible endpoint meant the Nemotron
integration was a base URL plus a model name, and the `NEBIUS_MODEL`
environment variable lets us swap Nemotron variants without code changes;
that compatibility is the right call and should not move. On AI Cloud, the
Serverless Jobs model (one REST call, injected files, pay per job) is a
natural fit for agent-driven sandboxed benchmarking; the one gap we hit is
that job logs are only reachable via the console and CLI, so result
retrieval has to shell out to `nebius ai job logs` instead of a REST
endpoint (see TODO(nebius) markers in `src/kernloop/nebius_jobs.py`). A
logs endpoint (or an injected-files style "collect files on completion"
mechanism) would close that cleanly.

## Requirement checklist

| Devpost requirement | Where it is satisfied |
|---|---|
| Built during Aug 26 to Oct 30 window | Repo created Oct 24, 2026; git history shows it |
| Public repo, MIT license | https://github.com/Izecrypt/kernloop, LICENSE at root |
| README with setup and run instructions | README.md |
| Uses Nemotron via Nebius Token Factory | `nemotron_client.py`, README "Built for" section |
| Functional Tavily API call at runtime (Best Use of Tavily prize) | `research.py`, `--research` flag or `KERNLOOP_RESEARCH=1`, calls logged in the run log |
| Benchmarks on Nebius AI Cloud | `nebius_jobs.py` (Serverless Jobs REST); demo run used the labeled-synthetic backend, Nebius path implemented and documented, live run pending billing activation (stated plainly in the description) |
| Working demo | https://izecrypt.github.io/kernloop/ (trajectory viewer for the run in the video) |
| 3 min video | /Users/tbo/Desktop/kernloop-demo-nebius.mp4 (82s), YouTube upload pending |
| Category, description, demo/video/repo URLs | This file |
| Feedback on Token Factory | Paragraph above |

## Demo video shot list (under 3 minutes)

1. **0:00 to 0:20, the problem.** Terminal: show `examples/gemm_spec.json`
   (shapes, dtype, tolerance) and the naive tiled baseline. One sentence:
   "This GEMM is K-heavy and the naive kernel wastes the GPU."
2. **0:20 to 0:45, the loop starts.** `kernloop run --backend nebius
   --llm nemotron --budget 12`. Scroll the prompt assembly: champion kernel
   and benchmark profile going to Nemotron, code-only contract, failure
   history. Show the job being created on the Nebius console.
3. **0:45 to 1:30, one iteration, end to end.** Split screen: job logs
   showing the checker (rel-L2 pass, L2 flush lines, per-rep times) while the
   run log line appears (compile, correctness, timing, decision, tokens).
   Show one rejection with its reason, not only promotions.
4. **1:30 to 2:10, the trajectory.** Open the HTML viewer: champion line
   stepping down across iterations, green promotions, orange rejections.
   Zoom the decision table. State the final speedup vs the baseline mean,
   both measured on the same Nebius GPU.
5. **2:10 to 2:40, the final kernel.** Show the champion source (split-K
   two-phase structure) and run the baseline vs champion back to back on one
   job to confirm the number live.
6. **2:40 to 3:00, generality close.** Point the loop at a second shape
   (edit M/N/K in the spec), watch it start working, cut. Closing line:
   "Same loop, new problem, no human in the middle."
