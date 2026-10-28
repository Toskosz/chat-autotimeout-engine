"""Recover a decision from model text, and name how the text failed.

Grammar mode should make `raw_compliant` true: the completion is the JSON
object and nothing else. Prompt-only JSON often needs a repair step, and some
failures (an invented action, a missing brace) cannot be repaired into the schema.
"""

import json
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

FAILURE_NONE = "none"
FAILURE_FENCE = "markdown_fence"
FAILURE_PREAMBLE = "preamble"
FAILURE_TRAILING = "trailing_commentary"
FAILURE_INVALID_JSON = "invalid_json"
FAILURE_INVALID_ENUM = "invalid_enum"
FAILURE_SCHEMA = "schema_mismatch"
FAILURE_EMPTY = "empty"


@dataclass(frozen=True)
class ParseAnalysis:
    raw_compliant: bool
    repaired: bool
    failure: str
    action: str | None
    violated_rule: str | None
    confidence: float | None
    scratchpad: str | None
    detail: str | None


def analyze(raw: str, model: type[BaseModel]) -> ParseAnalysis:
    text = raw or ""
    stripped = text.strip()
    if not stripped:
        return _fail(FAILURE_EMPTY, "The completion was empty.")

    try:
        value = model.model_validate_json(stripped)
    except (ValidationError, json.JSONDecodeError):
        value = None
    else:
        return _from_model(value, raw_compliant=True, failure=FAILURE_NONE, detail=None)

    extracted = _extract_object(text)
    if extracted is None:
        return _fail(FAILURE_INVALID_JSON, "No JSON object could be recovered.")

    try:
        value = model.model_validate_json(extracted)
    except json.JSONDecodeError as exc:
        return _fail(FAILURE_INVALID_JSON, str(exc))
    except ValidationError as exc:
        failure = FAILURE_INVALID_ENUM if _is_enum_error(exc) else FAILURE_SCHEMA
        return _fail(failure, _error_detail(exc))

    if "```" in text:
        failure = FAILURE_FENCE
        detail = "A markdown fence surrounded the object."
    elif not stripped.startswith("{"):
        failure = FAILURE_PREAMBLE
        detail = "Prose preceded the object."
    elif stripped != extracted.strip():
        failure = FAILURE_TRAILING
        detail = "Commentary followed the object."
    else:
        failure = FAILURE_NONE
        detail = None
    return _from_model(value, raw_compliant=False, failure=failure, detail=detail)


def _from_model(
    value: BaseModel,
    *,
    raw_compliant: bool,
    failure: str,
    detail: str | None,
) -> ParseAnalysis:
    data = value.model_dump()
    return ParseAnalysis(
        raw_compliant=raw_compliant,
        repaired=True,
        failure=failure,
        action=data.get("action"),
        violated_rule=data.get("violated_rule"),
        confidence=data.get("confidence"),
        scratchpad=data.get("scratchpad"),
        detail=detail,
    )


def _fail(failure: str, detail: str) -> ParseAnalysis:
    return ParseAnalysis(
        raw_compliant=False,
        repaired=False,
        failure=failure,
        action=None,
        violated_rule=None,
        confidence=None,
        scratchpad=None,
        detail=detail,
    )


def _extract_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        return None
    return text[start : end + 1]


def _is_enum_error(exc: ValidationError) -> bool:
    return any(err.get("type") in {"literal_error", "enum"} for err in exc.errors())


def _error_detail(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg', 'invalid')}" if loc else err.get("msg", "invalid"))
    return "; ".join(parts) or str(exc)
