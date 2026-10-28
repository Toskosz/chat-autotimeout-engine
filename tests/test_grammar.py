import json

import pytest

from src.grammar_inspector import (
    GrammarError,
    compile_gbnf,
    generation_steps,
    grammar_accepts,
    json_schema_text,
    to_openai_strict_schema,
)
from src.schemas import PrematureModerationDecision, ScratchpadModerationDecision


def test_matcher_covers_alternation_and_repetition():
    grammar = 'root ::= "a" | "bb"\n'
    assert grammar_accepts(grammar, "a")
    assert grammar_accepts(grammar, "bb")
    assert not grammar_accepts(grammar, "b")
    assert not grammar_accepts(grammar, "aa")
    assert grammar_accepts('root ::= "a"*\n', "")
    assert grammar_accepts('root ::= "a"*\n', "aaa")
    assert not grammar_accepts('root ::= "a"*\n', "b")


def test_field_order_is_the_schema_order():
    premature = json.loads(json_schema_text(PrematureModerationDecision))
    scratch = json.loads(json_schema_text(ScratchpadModerationDecision))
    assert list(premature["properties"]) == ["action", "violated_rule", "confidence"]
    assert list(scratch["properties"]) == ["scratchpad", "action", "violated_rule", "confidence"]


def test_generation_steps_put_the_mask_in_the_right_place():
    premature = generation_steps(PrematureModerationDecision)
    scratch = generation_steps(ScratchpadModerationDecision)
    assert premature[0].kind == "enum_mask"
    assert premature[0].field == "action"
    assert "WARN" not in "".join(premature[0].legal or ())
    assert scratch[0].kind == "free_text"
    assert scratch[0].field == "scratchpad"
    assert scratch[1].kind == "enum_mask"


def test_compiled_grammar_accepts_its_own_models_and_rejects_warn():
    premature_gbnf = compile_gbnf(PrematureModerationDecision)
    scratch_gbnf = compile_gbnf(ScratchpadModerationDecision)
    allowed = PrematureModerationDecision(
        action="TIMEOUT_60S", violated_rule="TOXICITY", confidence=0.91
    ).model_dump_json()
    reasoned = ScratchpadModerationDecision(
        scratchpad='He said "nice" and then posted 123 Fake Street.',
        action="BAN",
        violated_rule="DOXXING",
        confidence=0.5,
    ).model_dump_json()

    assert grammar_accepts(premature_gbnf, allowed)
    assert grammar_accepts(premature_gbnf, _compact(allowed))
    assert grammar_accepts(scratch_gbnf, reasoned)
    assert not grammar_accepts(premature_gbnf, allowed.replace("TIMEOUT_60S", "WARN"))
    assert not grammar_accepts(premature_gbnf, allowed.replace("0.91", "1.5"))
    assert not grammar_accepts(premature_gbnf, allowed.replace("0.91", "2"))
    assert not grammar_accepts(scratch_gbnf, allowed)
    assert not grammar_accepts(premature_gbnf, reasoned)


def test_gbnf_root_mentions_fields_in_generation_order():
    premature = compile_gbnf(PrematureModerationDecision)
    scratch = compile_gbnf(ScratchpadModerationDecision)
    premature_fields = " ".join(line for line in premature.splitlines() if line.startswith("fields"))
    scratch_fields = " ".join(line for line in scratch.splitlines() if line.startswith("fields"))
    assert premature_fields.index("action-kv") < premature_fields.index("violated-rule-kv")
    assert premature_fields.index("violated-rule-kv") < premature_fields.index("confidence-kv")
    assert scratch_fields.index("scratchpad-kv") < scratch_fields.index("action-kv")
    assert "scratchpad" not in premature_fields


def test_openai_strict_schema_drops_numeric_bounds_and_locks_keys():
    schema = to_openai_strict_schema(PrematureModerationDecision)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["action", "violated_rule", "confidence"]
    assert "minimum" not in schema["properties"]["confidence"]
    assert "maximum" not in schema["properties"]["confidence"]
    assert schema["properties"]["action"]["enum"] == ["ALLOW", "DELETE", "TIMEOUT_60S", "BAN"]


def test_broken_grammar_is_an_error():
    with pytest.raises(GrammarError):
        grammar_accepts("this is not gbnf", "a")


def _compact(payload: str) -> str:
    return json.dumps(json.loads(payload), separators=(",", ":"))
