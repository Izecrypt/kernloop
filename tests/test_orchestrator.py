import json

from kernloop.orchestrator import Orchestrator, extract_python
from kernloop.runlog import RunLogger
from kernloop.nebius_jobs import FakeJobBackend
from tests.conftest import ScriptedLLM, improved_kernel

REQUIRED_KEYS = {"iteration", "proposal", "compile", "correctness", "timing", "decision", "tokens", "backend"}


def make_orchestrator(spec, tmp_path, responses, budget=3, seed=0):
    logger = RunLogger(tmp_path / "run.runlog.jsonl", run_id="testrun", backend="fake")
    llm = ScriptedLLM(responses)
    orch = Orchestrator(
        spec,
        llm=llm,
        backend=FakeJobBackend(seed=seed),
        logger=logger,
        budget=budget,
        reps=5,
        warmups=2,
    )
    return orch, llm


def test_loop_terminates_and_log_is_valid_jsonl(spec, tmp_path):
    improved = improved_kernel(spec.baseline_source)
    orch, llm = make_orchestrator(spec, tmp_path, [improved, "just prose, no code"], budget=3)
    summary = orch.run()

    assert summary.iterations_run == 3
    assert summary.promotions == 1
    assert summary.champion.iteration == 1
    assert summary.champion.source == improved.strip()
    assert summary.tokens == {"prompt": 30, "completion": 60}

    lines = (tmp_path / "run.runlog.jsonl").read_text().strip().splitlines()
    header = json.loads(lines[0])
    assert header["type"] == "run_header"
    records = [json.loads(l) for l in lines[1:]]
    assert len(records) == 3
    for rec in records:
        assert REQUIRED_KEYS <= set(rec)
        assert rec["run_id"] == "testrun"
        assert rec["backend"] == "fake"
        assert rec["tokens"]["prompt"] == 10


def test_promotion_only_on_real_improvement(spec, tmp_path):
    # Iteration 1: real improvement (> 2 percent), iteration 2: identical
    # champion (0 percent), iteration 3: garbage prose (compile failure).
    improved = improved_kernel(spec.baseline_source)
    orch, _ = make_orchestrator(
        spec, tmp_path, [improved, improved, "I cannot do that, here is my reasoning in prose."], budget=3
    )
    summary = orch.run()
    records = orch.logger.iterations()

    decisions = [(r["decision"]["promoted"], r["decision"]["reason"]) for r in records]
    assert decisions[0][0] is True
    assert "improved" in decisions[0][1]
    assert decisions[1][0] is False
    assert "2%" in decisions[1][1] or "2.0%" in decisions[1][1] or "2 percent" in decisions[1][1]
    assert decisions[2][0] is False
    assert "prose" in decisions[2][1] or "compile" in decisions[2][1].lower()
    assert summary.promotions == 1


def test_repeated_champion_is_never_repromoted(spec, tmp_path):
    improved = improved_kernel(spec.baseline_source)
    orch, _ = make_orchestrator(spec, tmp_path, [improved], budget=4)
    summary = orch.run()
    assert summary.promotions == 1
    assert all(r["decision"]["promoted"] is False for r in orch.logger.iterations()[1:])


def test_prompt_contains_required_context(spec, tmp_path):
    improved = improved_kernel(spec.baseline_source)
    orch, llm = make_orchestrator(spec, tmp_path, [improved], budget=1)
    orch.run()
    prompt = llm.prompts[0]
    assert spec.problem_summary.split(".")[0] in prompt  # problem spec
    assert "kernel_fn" in prompt  # champion kernel code
    assert "mean:" in prompt and "min-of-reps" in prompt  # benchmark profile
    assert "ONLY a complete, self-contained Python file" in prompt
    assert "No prose" in prompt


def test_extract_python_strictness():
    code = "import triton\n\ndef kernel_fn(a, b):\n    pass\n"
    assert extract_python(code)[0] == code.strip()
    assert extract_python(f"```python\n{code}```")[0] == code.strip()
    assert extract_python(f"Here you go:\n```python\n{code}```\nHope this helps!")[0] is None
    assert extract_python("Sure! The answer is to vectorize.")[0] is None


def test_correctness_failure_is_rejected_not_benchmarked(spec, tmp_path):
    no_dot = spec.baseline_source.replace(
        "acc = tl.dot(a, b, acc)",
        "acc += tl.sum(a, 0)[:, None] * tl.sum(b, 1)[None, :]",
    )
    orch, _ = make_orchestrator(spec, tmp_path, [no_dot], budget=2)
    summary = orch.run()
    assert summary.promotions == 0
    first = orch.logger.iterations()[0]
    assert first["correctness"]["ok"] is False
    assert first["timing"]["mean_ms"] is None  # never benchmarked
