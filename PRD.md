---
type: prd
status: draft
owner: Ize
created: 2026-10-24
tags:
  - project/kernloop
---

# KernLoop PRD (draft)

## What

An autonomous GPU kernel engineer: an agent that writes, validates, and benchmarks Triton kernels on real GPUs, improving them over iterations without human intervention. Built on Nebius Token Factory (Nemotron models for reasoning and code generation) and Nebius AI Cloud (GPU jobs for sandboxed execution and timing).

One sentence: the Open Kernel Challenge loop (research, write kernel, validate, benchmark, keep champion), productized as an open source agent anyone can point at a kernel problem.

## Why (for the Nebius x NVIDIA hackathon)

- Coding and Agentic Engineering Track: agents that write, run, and test code. This is that, on the hardest real codebase class (GPU kernels).
- Technological implementation judging: real GPU benchmarks via Nebius Serverless Jobs, not mocked results.
- Quality of idea: autonomous performance engineering is a non-obvious use; the founder is a practitioner (OKC leaderboard, Izecrypt), so the problem understanding is genuine.
- All submission requirements are satisfiable in the window (Oct 24 to 30): working demo (live run URL or recorded), 3 min video, public repo with MIT license.

## Track

Coding and Agentic Engineering Track.

## Core loop (v1 scope)

1. Problem spec in (kernel task description, reference implementation, correctness checker, benchmark harness).
2. Nemotron (Token Factory) proposes a kernel variant, with the current champion and benchmark profile in context.
3. Nebius Serverless Job runs: correctness check, then N timed benchmark reps, cold cache discipline.
4. Results appended to run log; champion promoted only on statistically real improvement (mean of reps, min-of-reps as secondary).
5. Repeat within budget; human inspects the trajectory.

## v1 scope (must fit Oct 24 to 30)

- ONE workload class: a Triton GEMM with split-K two-phase structure (the bf16_proj pattern Ize knows cold).
- Fixed iteration budget per run (e.g. 12 proposals), one GPU type.
- CLI plus JSON run log; simple HTML trajectory viewer for the demo video.
- Reference harness: torch CPU correctness mirror for pre-flight checks (no GPU needed locally), Nebius job for the real timing.
- Explicitly out of scope: multi-problem dispatch, CUDA graphs, network search, persistence across runs.

## Demo narrative (video, under 3 min)

1. Show the problem spec and naive kernel timing.
2. Show the agent loop running on Nebius: proposals, checks, benchmark numbers.
3. Show the trajectory chart improving over iterations.
4. Show the final kernel and its speedup vs baseline, measured on Nebius GPUs.
5. Close on the loop being general: point it at a new shape, watch it work.

## Submission checklist (due Oct 30 10am PDT)

- [ ] Nebius Token Factory account plus Nemotron model access (owner signup)
- [ ] Nebius AI Cloud credits or trial for GPU jobs (owner signup)
- [ ] Repo public on GitHub (Izecrypt org or new), MIT license, README with setup plus run instructions, clear Nemotron/Token Factory usage notes
- [ ] Working demo: run log plus HTML viewer served or recorded
- [ ] 3 min YouTube video
- [ ] Devpost submission: category, description, demo URL, video URL, repo URL, feedback on Token Factory
- [ ] Note: project built during submission period (Aug 26 to Oct 30); no prior-version explanation needed

## Risks

- Nebius account approval or credit provisioning could be slow: owner signs up TONIGHT.
- Token Factory API rate limits or model availability: fallback to Nemotron via an alternative Nebius endpoint is acceptable per rules (must stay on Nebius).
- GPU job latency could make live demos slow: record the video from a real completed run.
