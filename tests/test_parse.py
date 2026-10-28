from src.parse import (
    FAILURE_EMPTY,
    FAILURE_FENCE,
    FAILURE_INVALID_ENUM,
    FAILURE_INVALID_JSON,
    FAILURE_NONE,
    FAILURE_PREAMBLE,
    FAILURE_SCHEMA,
    FAILURE_TRAILING,
    analyze,
)
from src.schemas import PrematureModerationDecision, ScratchpadModerationDecision

CLEAN = (
    '{"scratchpad": "Nothing in the message matches a rule.", '
    '"action": "ALLOW", "violated_rule": "NONE", "confidence": 0.97}'
)


def test_raw_json_is_compliant():
    parsed = analyze(CLEAN, ScratchpadModerationDecision)
    assert parsed.raw_compliant
    assert parsed.failure == FAILURE_NONE
    assert parsed.action == "ALLOW"
    assert parsed.scratchpad.startswith("Nothing")


def test_fence_preamble_and_trailing_text_are_repaired_and_named():
    fenced = analyze(f"```json\n{CLEAN}\n```", ScratchpadModerationDecision)
    assert not fenced.raw_compliant
    assert fenced.repaired
    assert fenced.failure == FAILURE_FENCE
    assert fenced.action == "ALLOW"

    prefaced = analyze(f"Sure — here is the JSON decision:\n{CLEAN}", ScratchpadModerationDecision)
    assert prefaced.failure == FAILURE_PREAMBLE
    assert prefaced.action == "ALLOW"

    trailed = analyze(CLEAN + "\n\nHope that helps!", ScratchpadModerationDecision)
    assert trailed.failure == FAILURE_TRAILING
    assert trailed.action == "ALLOW"


def test_invented_action_and_broken_json_do_not_become_a_decision():
    warned = analyze(CLEAN.replace("ALLOW", "WARN"), ScratchpadModerationDecision)
    assert warned.failure == FAILURE_INVALID_ENUM
    assert warned.action is None

    broken = analyze(CLEAN[:-1], ScratchpadModerationDecision)
    assert broken.failure == FAILURE_INVALID_JSON
    assert broken.action is None

    empty = analyze("  ", ScratchpadModerationDecision)
    assert empty.failure == FAILURE_EMPTY


def test_extra_and_missing_fields_are_schema_mismatches():
    extra = analyze(
        '{"action": "ALLOW", "violated_rule": "NONE", "confidence": 0.5, "note": "hi"}',
        PrematureModerationDecision,
    )
    assert extra.failure == FAILURE_SCHEMA
    missing = analyze('{"action": "ALLOW"}', PrematureModerationDecision)
    assert missing.failure == FAILURE_SCHEMA
    assert missing.action is None
