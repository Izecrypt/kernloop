import dataclasses
import json
from pathlib import Path

import pytest

from kernloop.nemotron_client import LLMResponse
from kernloop.workload import WorkloadSpec, default_spec

EXAMPLES_SPEC_PATH = Path(__file__).resolve().parent.parent / "examples" / "gemm_spec.json"


@pytest.fixture
def spec() -> WorkloadSpec:
    return default_spec()


@pytest.fixture
def spec_from_examples_json() -> WorkloadSpec:
    return WorkloadSpec.load(EXAMPLES_SPEC_PATH)


class ScriptedLLM:
    """Canned LLM stub: returns queued responses in order, then repeats the last."""

    model = "scripted-stub"

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def propose(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        text = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return LLMResponse(text=text, model=self.model, prompt_tokens=10, completion_tokens=20)


def improved_kernel(source: str) -> str:
    """A faster valid candidate, in the way the fake backend scores features."""
    out = source.replace(
        "BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32",
        "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64",
    )
    out = out.replace(
        "BLOCK_K=BLOCK_K,",
        "BLOCK_K=BLOCK_K, num_stages=4, num_warps=8,",
    )
    return out


def write_spec_json(path: Path, spec: WorkloadSpec) -> Path:
    path.write_text(json.dumps(dataclasses.asdict(spec), indent=2) + "\n")
    return path
