# KernLoop

An autonomous GPU kernel engineer. You point it at a kernel problem spec; an
agent loop proposes Triton kernel variants with a reasoning LLM (NVIDIA
Nemotron via Nebius Token Factory), validates correctness, benchmarks on real
GPUs (Nebius Serverless Jobs), and promotes a new champion only when the
improvement is statistically real. Every iteration is written to a run log a
human can inspect, plus a self-contained HTML trajectory page for demos.

Built for the **Nebius x NVIDIA Global AI Hackathon** (Coding and Agentic
Engineering Track). See [STATUS.md](STATUS.md) for what works today and what
is still stubbed.

## The loop

```
problem spec (JSON: shapes, dtype, baseline kernel, target notes)
        |
        v
+-------------------------------------------------------------+
| 1. prompt Nemotron: spec + champion kernel + benchmark      |
|    profile; demand ONE complete Python file, kernel_fn      |
|    entrypoint, no prose                                     |
| 2. strict extraction: bare Python file or single fence,     |
|    nothing else                                             |
| 3. compile check (syntax + kernel_fn entrypoint)            |
| 4. correctness check on the backend (reference GEMM)        |
| 5. benchmark: warmups discarded, N timed reps,              |
|    cold-cache discipline, mean + min                        |
| 6. promote only if mean improves > 2% vs champion;          |
|    otherwise reject with reason                             |
| 7. append everything (proposal, compile, correctness,       |
|    timings, decision, tokens) to the JSONL run log          |
+-------------------------------------------------------------+
        | repeat within budget
        v
run log (.runlog.jsonl) + HTML trajectory viewer
```

## Architecture

```
                 +--------------------+
 kernloop run -->|  orchestrator.py   |<----------------------+
                 |  (the loop)        |                       |
                 +----+----------+----+                       |
          prompt |    | compile  | benchmark                 |
                 v    v          v                           |
        +--------+ +-------+ +--------------------+          |
        |nemotron| |harness| | nebius_jobs.py     |          |
        |_client | |       | |  FakeJobBackend    | local    |
        | (Token| |       | |  NebiusJobBackend  | GPU jobs |
        |Factory)| |       | |  (STUB, not wired) |          |
        +--------+ +-------+ +--------------------+          |
                 |    |          |                           |
                 v    v          v                           |
        +---------------------------------------+             |
        | runlog.py (append-only JSONL)         |-------------+
        | viewer.py  (self-contained HTML)      |
        +---------------------------------------+
```

## Setup

Python 3.11+. For local simulation (no GPU, no network):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
```

For the real Nemotron proposer (OpenAI-compatible client):

```bash
pip install -e '.[llm,test]'
export NEBIUS_API_KEY=...          # from the Nebius Token Factory console
export NEBIUS_MODEL=...            # optional, overrides the default model
```

## How to run

Local simulation, fully offline (deterministic fake backend + offline
proposer):

```bash
kernloop run --spec examples/gemm_spec.json --budget 12 --backend fake
# or: python3 -m kernloop run --spec examples/gemm_spec.json --budget 12 --backend fake
```

Real proposer once you have a Token Factory key (still simulated execution):

```bash
kernloop run --spec examples/gemm_spec.json --budget 12 --backend fake --llm nemotron
```

Real GPU execution is not wired up yet:

```bash
kernloop run --backend nebius   # errors with "not yet configured"
```

Outputs land in `runs/`: one `<run_id>.runlog.jsonl` and one `<run_id>.html`.
Render a log again later with `kernloop view --log runs/<id>.runlog.jsonl --out out.html`,
or validate a spec without any GPU with `kernloop preflight --spec examples/gemm_spec.json`.

## Testing

```bash
python3 -m pytest
```

The suite runs the full loop end to end against the FakeJobBackend and a
canned LLM stub: loop termination, champion promotion logic, JSONL validity,
and HTML rendering. No GPU, no Triton, no network required.

## Built for the Nebius x NVIDIA Global AI Hackathon

- Reasoning and code generation: **Nemotron models via Nebius Token Factory**
  (OpenAI-compatible endpoint, `NEBIUS_API_KEY`), proposing Triton kernel
  variants from the problem spec, the current champion, and its benchmark
  profile.
- Execution and timing: **benchmarks on Nebius AI Cloud** via Nebius
  Serverless Jobs (the `nebius` backend; see STATUS.md for wiring status).
  The `fake` backend is a deterministic local simulator used for development
  and tests, and every fake timing is labeled `synthetic` in the run log.
- Honesty note: run log entries record which backend and which model produced
  each number, so real GPU results and simulated ones are never confused.

## License

MIT, see [LICENSE](LICENSE).
