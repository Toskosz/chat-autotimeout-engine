"""Shared moderation policy.

The two grammar modes receive the same system prompt. The schema is the only
difference between them. Unconstrained mode adds a prose description of the
JSON shape, because nothing else tells the model what to emit.
"""

from src.schemas import ExecutionMode

POLICY = """\
You moderate a livestream chat. Judge the whole message.

A friendly opening does not excuse a later violation. A later retraction does
excuse an earlier joke. Posting someone else's private information still stands
when the writer walks it back.

Actions:
- ALLOW: good-faith chat, questions, emotes, criticism of the game, a joke that is clearly retracted
- DELETE: spam, or a message that should disappear without a penalty
- TIMEOUT_60S: a directed insult, harassment, or self-promotion
- BAN: someone else's phone, email, street address, or real-world identity, or an unretracted self-harm directive

violated_rule is one of NONE, TOXICITY, SPAM, SELF_PROMO, DOXXING.
confidence is a number from 0 to 1.
"""

_JSON_INSTRUCTIONS = """
Reply with one JSON object and no other text. Use these keys, in this order:
scratchpad, action, violated_rule, confidence.
scratchpad is a short string quoting the span that decided the action.
"""

_GRAMMAR_INSTRUCTIONS = """
A decoder grammar fixes the response shape. Follow that field order.
Base every closed-field choice on the whole message.
"""


def system_prompt(mode: ExecutionMode) -> str:
    if mode is ExecutionMode.UNCONSTRAINED_PROMPT_ONLY:
        return POLICY + _JSON_INSTRUCTIONS
    return POLICY + _GRAMMAR_INSTRUCTIONS


USER_PREFIX = "Chat message:\n"


def render_user(message: str) -> str:
    return USER_PREFIX + message


def message_from_user_prompt(user: str) -> str:
    if not user.startswith(USER_PREFIX):
        raise ValueError("User prompt is missing the chat message prefix.")
    return user[len(USER_PREFIX) :]
