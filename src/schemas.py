"""The three execution modes, and the schemas that make field order visible.

Constrained decoders walk properties in schema order and mask every token that
would leave the grammar. That order is the order the model is allowed to think
in its own output. The two Pydantic models below describe the same decision;
they disagree about whether any free text is generated before the action.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

class ExecutionMode(str, Enum):
    UNCONSTRAINED_PROMPT_ONLY = "Unconstrained_Prompt_Only"
    GRAMMAR_PREMATURE_CONSTRAINT = "Grammar_Premature_Constraint"
    GRAMMAR_SCRATCHPAD_FIRST = "Grammar_Scratchpad_First"


class PrematureModerationDecision(BaseModel):
    """Decision-first schema. This is the premature-constraint anti-pattern.

    A constrained decoder emits keys in this order, so `action` is the first
    value the model is allowed to sample. A reasoning field placed under
    `confidence` would arrive after that token. It can justify the action.
    It cannot change it.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["ALLOW", "DELETE", "TIMEOUT_60S", "BAN"]
    violated_rule: Literal["NONE", "TOXICITY", "SPAM", "SELF_PROMO", "DOXXING"]
    confidence: float = Field(ge=0.0, le=1.0)


class ScratchpadModerationDecision(BaseModel):
    """Scratchpad-first schema.

    `scratchpad` is free text, so almost any token is legal while it is written.
    The action enum is masked only after that text is already in the context.
    The mask still guarantees a legal action. It applies later.
    """

    model_config = ConfigDict(extra="forbid")

    scratchpad: str = Field(
        description=(
            "Working notes written before the action. Quote the span that "
            "decides it, including a later retraction or a violation that "
            "follows a friendly opening."
        )
    )
    action: Literal["ALLOW", "DELETE", "TIMEOUT_60S", "BAN"]
    violated_rule: Literal["NONE", "TOXICITY", "SPAM", "SELF_PROMO", "DOXXING"]
    confidence: float = Field(ge=0.0, le=1.0)


def response_schema(mode: ExecutionMode) -> type[BaseModel] | None:
    """Schema passed to the provider. Unconstrained mode sends none."""

    if mode is ExecutionMode.UNCONSTRAINED_PROMPT_ONLY:
        return None
    if mode is ExecutionMode.GRAMMAR_PREMATURE_CONSTRAINT:
        return PrematureModerationDecision
    return ScratchpadModerationDecision


def validation_model(mode: ExecutionMode) -> type[BaseModel]:
    """Model used to accept or reject the completion text."""

    if mode is ExecutionMode.GRAMMAR_PREMATURE_CONSTRAINT:
        return PrematureModerationDecision
    return ScratchpadModerationDecision


def field_order(model: type[BaseModel]) -> list[str]:
    return list(model.model_fields)
