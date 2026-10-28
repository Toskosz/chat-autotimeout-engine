from src.fixtures import FIXTURES, PIVOT_IDS
from src.grammar_inspector import compile_gbnf, grammar_accepts
from src.llm_client import MockClient
from src.metrics import score_mode
from src.mock_moderator import classify, explain_simulation, first_clause, slip_for
from src.parse import FAILURE_NONE
from src.pipeline import run_bench, run_moderation
from src.policy import system_prompt
from src.schemas import (
    ExecutionMode,
    PrematureModerationDecision,
    ScratchpadModerationDecision,
)


def test_gold_labels_match_the_full_message_policy():
    assert len(FIXTURES) == 20
    for fixture in FIXTURES:
        decision = classify(fixture.message)
        assert (decision.action, decision.violated_rule) == (
            fixture.action,
            fixture.violated_rule,
        ), fixture.id


def test_pivot_messages_are_exactly_the_prefix_disagreements():
    found = set()
    for fixture in FIXTURES:
        prefix = classify(first_clause(fixture.message))
        full = classify(fixture.message)
        if (prefix.action, prefix.violated_rule) != (full.action, full.violated_rule):
            found.add(fixture.id)
    assert found == set(PIVOT_IDS)


def test_grammar_modes_share_a_prompt_and_unconstrained_must_ask_for_json():
    premature = system_prompt(ExecutionMode.GRAMMAR_PREMATURE_CONSTRAINT)
    scratch = system_prompt(ExecutionMode.GRAMMAR_SCRATCHPAD_FIRST)
    loose = system_prompt(ExecutionMode.UNCONSTRAINED_PROMPT_ONLY)
    assert premature == scratch
    assert "scratchpad, action, violated_rule, confidence" in loose
    assert loose != premature


def test_mock_grammar_outputs_match_gold_or_the_prefix_and_parse_clean():
    client = MockClient()
    premature_gbnf = compile_gbnf(PrematureModerationDecision)
    scratch_gbnf = compile_gbnf(ScratchpadModerationDecision)
    slips = set()
    for fixture in FIXTURES:
        slips.add(slip_for(fixture.message))
        premature = run_moderation(fixture.message, ExecutionMode.GRAMMAR_PREMATURE_CONSTRAINT, client)
        scratch = run_moderation(fixture.message, ExecutionMode.GRAMMAR_SCRATCHPAD_FIRST, client)
        loose = run_moderation(fixture.message, ExecutionMode.UNCONSTRAINED_PROMPT_ONLY, client)

        assert premature.raw_compliant
        assert premature.failure == FAILURE_NONE
        assert grammar_accepts(premature_gbnf, premature.raw_text)
        prefix = classify(first_clause(fixture.message))
        assert (premature.action, premature.violated_rule) == (prefix.action, prefix.violated_rule)

        assert scratch.raw_compliant
        assert grammar_accepts(scratch_gbnf, scratch.raw_text)
        assert (scratch.action, scratch.violated_rule) == (fixture.action, fixture.violated_rule)
        assert scratch.scratchpad

        assert loose.action == fixture.action or loose.action is None
        if slip_for(fixture.message) == "clean":
            assert loose.raw_compliant
            assert grammar_accepts(scratch_gbnf, loose.raw_text)
        else:
            assert not loose.raw_compliant
            assert not grammar_accepts(scratch_gbnf, loose.raw_text)
    assert {"markdown_fence", "preamble", "invalid_enum", "trailing_commentary", "invalid_json", "clean"} <= slips


def test_benchmark_scores_separate_syntax_from_the_decision():
    rows = run_bench(MockClient())
    premature = score_mode(ExecutionMode.GRAMMAR_PREMATURE_CONSTRAINT, rows)
    scratch = score_mode(ExecutionMode.GRAMMAR_SCRATCHPAD_FIRST, rows)
    loose = score_mode(ExecutionMode.UNCONSTRAINED_PROMPT_ONLY, rows)
    assert premature.raw_compliant == premature.total
    assert scratch.raw_compliant == scratch.total
    assert scratch.action_correct == scratch.total
    assert premature.action_correct == premature.total - len(PIVOT_IDS)
    assert loose.raw_compliant < loose.total
    assert "Prefix the premature simulation" in explain_simulation(rows[0].fixture.message)
