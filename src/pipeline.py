"""One moderation call, and a benchmark over the fixture chat."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.fixtures import FIXTURES, Fixture
from src.llm_client import LLMClient
from src.parse import ParseAnalysis, analyze
from src.policy import render_user, system_prompt
from src.schemas import ExecutionMode, response_schema, validation_model


@dataclass(frozen=True)
class ModerationOutcome:
    mode: ExecutionMode
    provider: str
    raw_text: str
    latency_ms: float
    parsed: ParseAnalysis

    @property
    def action(self) -> str | None:
        return self.parsed.action

    @property
    def violated_rule(self) -> str | None:
        return self.parsed.violated_rule

    @property
    def confidence(self) -> float | None:
        return self.parsed.confidence

    @property
    def scratchpad(self) -> str | None:
        return self.parsed.scratchpad

    @property
    def raw_compliant(self) -> bool:
        return self.parsed.raw_compliant

    @property
    def failure(self) -> str:
        return self.parsed.failure


@dataclass(frozen=True)
class BenchRow:
    fixture: Fixture
    outcomes: dict[ExecutionMode, ModerationOutcome]


def run_moderation(message: str, mode: ExecutionMode, client: LLMClient) -> ModerationOutcome:
    schema = response_schema(mode)
    started = time.perf_counter()
    raw = client.generate(system=system_prompt(mode), user=render_user(message), schema=schema)
    latency_ms = (time.perf_counter() - started) * 1000
    return ModerationOutcome(
        mode=mode,
        provider=client.name,
        raw_text=raw,
        latency_ms=latency_ms,
        parsed=analyze(raw, validation_model(mode)),
    )


def run_bench(client: LLMClient, fixtures: tuple[Fixture, ...] = FIXTURES) -> list[BenchRow]:
    rows = []
    for fixture in fixtures:
        outcomes = {
            mode: run_moderation(fixture.message, mode, client)
            for mode in ExecutionMode
        }
        rows.append(BenchRow(fixture, outcomes))
    return rows
