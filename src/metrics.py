"""Score benchmark rows against the hand labels."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from src.pipeline import BenchRow
from src.schemas import ExecutionMode


@dataclass(frozen=True)
class ModeScore:
    mode: ExecutionMode
    total: int
    raw_compliant: int
    parsed: int
    action_correct: int
    action_correct_and_raw: int
    rule_correct: int
    latency_p50_ms: float

    @property
    def raw_compliant_rate(self) -> float:
        return self.raw_compliant / self.total

    @property
    def action_accuracy(self) -> float:
        return self.action_correct / self.total


def score_mode(mode: ExecutionMode, rows: list[BenchRow]) -> ModeScore:
    outcomes = [row.outcomes[mode] for row in rows]
    latencies = [outcome.latency_ms for outcome in outcomes]
    action_correct = 0
    action_correct_and_raw = 0
    rule_correct = 0
    for row, outcome in zip(rows, outcomes, strict=True):
        action_ok = outcome.action == row.fixture.action
        if action_ok:
            action_correct += 1
            if outcome.raw_compliant:
                action_correct_and_raw += 1
        if outcome.violated_rule == row.fixture.violated_rule:
            rule_correct += 1
    return ModeScore(
        mode=mode,
        total=len(rows),
        raw_compliant=sum(outcome.raw_compliant for outcome in outcomes),
        parsed=sum(outcome.parsed.repaired for outcome in outcomes),
        action_correct=action_correct,
        action_correct_and_raw=action_correct_and_raw,
        rule_correct=rule_correct,
        latency_p50_ms=statistics.median(latencies) if latencies else 0.0,
    )
