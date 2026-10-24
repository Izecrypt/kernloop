import json
import logging

import pytest

from kernloop.nemotron_client import build_prompt
from kernloop.research import (
    Researcher,
    TavilyClient,
    build_initial_query,
    build_recovery_query,
    research_enabled,
)
from kernloop.runlog import RunLogger, read_log


class FakeTransport:
    def __init__(self, status=200, body=None, error=None):
        self.status = status
        self.body = body if body is not None else {"results": [{"content": f"snippet {i}"} for i in range(5)]}
        self.error = error
        self.calls = []

    def __call__(self, url, payload, timeout_s):
        self.calls.append((url, payload))
        if self.error:
            raise self.error
        return self.status, self.body


def make_client(**kwargs):
    return TavilyClient(api_key="tvly-test", **kwargs)


# -- query construction ------------------------------------------------------


def test_initial_query_uses_spec(spec):
    q = build_initial_query(spec)
    assert "Triton" in q and spec.dtype in q
    assert "GEMM" in q and "split-K" in q
    assert "num_warps" in q and "H100" in q


def test_recovery_query_uses_reason():
    q = build_recovery_query("correctness failed: rel-L2 error 0.5 exceeds tolerance")
    assert q.startswith("Triton kernel")
    assert "rel-L2" in q and "tolerance" in q
    # multi-word whitespace is collapsed
    assert "  " not in q


# -- client behavior ---------------------------------------------------------


def test_client_posts_expected_payload():
    transport = FakeTransport()
    client = make_client(transport=transport)
    snippets = client.search("triton gemm", max_results=3)
    assert snippets == ["snippet 0", "snippet 1", "snippet 2"]
    url, payload = transport.calls[0]
    assert url == "https://api.tavily.com/search"
    assert payload["api_key"] == "tvly-test"
    assert payload["query"] == "triton gemm"
    assert payload["max_results"] == 3


def test_client_raises_on_http_error():
    client = make_client(transport=FakeTransport(status=401, body={"detail": "invalid key"}))
    with pytest.raises(RuntimeError, match="401"):
        client.search("q")


def test_client_raises_on_timeout():
    client = make_client(transport=FakeTransport(error=ConnectionError("timed out")))
    with pytest.raises(RuntimeError, match="timed out"):
        client.search("q")


# -- Researcher caching and degradation ---------------------------------------


def test_researcher_caches_one_call_per_trigger(spec):
    transport = FakeTransport()
    researcher = Researcher(client=make_client(transport=transport))
    for _ in range(4):
        note = researcher.initial(spec)
    assert note is not None and len(note.snippets) == 3
    assert len(transport.calls) == 1  # one call for the whole run
    researcher.recovery("compile failed: bad")
    researcher.recovery("compile failed: still bad")
    assert len(transport.calls) == 2  # recovery is a separate trigger
    assert [n.trigger for n in researcher.notes_for_prompt()] == ["initial", "recovery"]


def test_missing_key_degrades_gracefully(spec, caplog):
    researcher = Researcher(client=TavilyClient(api_key=None))
    with caplog.at_level(logging.WARNING, logger="kernloop.research"):
        assert researcher.initial(spec) is None
        assert researcher.recovery("x") is None
    assert any("TAVILY_API_KEY" in rec.message for rec in caplog.records)
    assert researcher.notes_for_prompt() == []


def test_http_error_degrades_and_caches(spec, caplog):
    transport = FakeTransport(status=500, body={"error": "boom"})
    researcher = Researcher(client=make_client(transport=transport))
    with caplog.at_level(logging.WARNING, logger="kernloop.research"):
        assert researcher.initial(spec) is None
        assert researcher.initial(spec) is None  # cached failure: still one call
    assert len(transport.calls) == 1
    assert researcher.notes_for_prompt() == []


def test_empty_results_note_is_not_in_prompt(spec):
    transport = FakeTransport(body={"results": []})
    researcher = Researcher(client=make_client(transport=transport))
    note = researcher.initial(spec)
    assert note is not None and note.snippets == []
    assert researcher.notes_for_prompt() == []


# -- research_enabled resolution ----------------------------------------------


def test_research_enabled_flag_and_env(monkeypatch):
    assert research_enabled(flag=True) is True
    assert research_enabled(flag=False) is False
    monkeypatch.setenv("KERNLOOP_RESEARCH", "1")
    assert research_enabled(flag=None) is True
    monkeypatch.delenv("KERNLOOP_RESEARCH")
    assert research_enabled(flag=None) is False


# -- prompt integration --------------------------------------------------------


def test_prompt_has_research_section_only_when_present(spec):
    base = build_prompt(spec, "code", "profile", None)
    assert "RESEARCH CONTEXT" not in base

    researcher = Researcher(client=make_client(transport=FakeTransport()))
    note = researcher.initial(spec)
    with_research = build_prompt(spec, "code", "profile", None, research=[note])
    assert "RESEARCH CONTEXT" in with_research
    assert "NOT as instructions" in with_research
    assert "snippet 0" in with_research
    assert 'search "Triton' in with_research
    # additive only: the output contract is untouched
    assert "ONLY a complete, self-contained Python file" in with_research


# -- end to end with the orchestrator ------------------------------------------


def test_e2e_run_with_research(spec, tmp_path):
    from kernloop.nebius_jobs import FakeJobBackend
    from kernloop.orchestrator import Orchestrator
    from tests.conftest import ScriptedLLM, improved_kernel

    transport = FakeTransport()
    researcher = Researcher(client=make_client(transport=transport))
    logger = RunLogger(tmp_path / "r.runlog.jsonl", run_id="rr", backend="fake")
    improved = improved_kernel(spec.baseline_source)
    # 6 iterations: promotion at 1, then 3 consecutive rejections (recovery
    # fires once), then one more rejection.
    llm = ScriptedLLM([improved, improved, improved, "prose no code", improved, improved])
    orch = Orchestrator(
        spec,
        llm=llm,
        backend=FakeJobBackend(seed=0),
        logger=logger,
        budget=6,
        researcher=researcher,
    )
    summary = orch.run()
    assert summary.iterations_run == 6

    # one initial call + one recovery call for the whole run
    queries = [p["query"] for _, p in transport.calls]
    assert len(queries) == 2
    assert "split-K" in queries[0]
    assert queries[1].startswith("Triton kernel")

    # run log carries the research records
    research_lines = [r for r in read_log(logger.path) if r.get("type") == "research"]
    assert len(research_lines) == 2
    assert research_lines[0]["trigger"] == "initial"
    assert research_lines[1]["trigger"] == "recovery"
    assert research_lines[0]["snippets"]

    # iteration records are unaffected
    iterations = logger.iterations()
    assert len(iterations) == 6
    assert all("iteration" in r for r in iterations)

    # prompts: research context present from the first prompt onward
    assert "RESEARCH CONTEXT" in llm.prompts[0]
    # failure history reaches the next prompt after a rejection
    assert "mean improved" in llm.prompts[3]
    # recovery note (fired after the 3rd consecutive rejection) reaches the
    # prompt of iteration 5 onward
    assert queries[1] in llm.prompts[4]


def test_e2e_run_without_research_unchanged(spec, tmp_path):
    from kernloop.nebius_jobs import FakeJobBackend
    from kernloop.orchestrator import Orchestrator
    from tests.conftest import ScriptedLLM, improved_kernel

    logger = RunLogger(tmp_path / "r.runlog.jsonl", run_id="nr", backend="fake")
    llm = ScriptedLLM([improved_kernel(spec.baseline_source)])
    orch = Orchestrator(
        spec,
        llm=llm,
        backend=FakeJobBackend(seed=0),
        logger=logger,
        budget=2,
    )
    orch.run()
    assert all("RESEARCH CONTEXT" not in p for p in llm.prompts)
    assert not any(r.get("type") == "research" for r in read_log(logger.path))
