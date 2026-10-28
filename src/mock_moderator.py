"""Deterministic stand-in for the three modes.

The scratchpad path and the premature path call the same policy. The premature
path sees only the first clause. That is a caricature of premature constraint,
drawn so the ordering effect is visible with no API key: a live model still
reads the whole input, and usually shows a smaller gap.

Unconstrained mode uses the full-message policy, then sometimes wraps, prefaces,
or corrupts the JSON the way prompt-only requests do.
"""

import json
import re
import zlib
from dataclasses import dataclass

from pydantic import BaseModel

from src.policy import message_from_user_prompt

_SEVERE = (r"\bkys\b", r"kill yourself", r"\bi will kill\b", r"\bi'll kill\b", r"\bswat you\b")
_INSULTS = (
    r"\btrash\b",
    r"\bidiot\b",
    r"\bstupid\b",
    r"\buninstall\b",
    r"\bclown\b",
    r"\bhate you\b",
    r"\byou suck\b",
    r"\bso bad\b",
)
_PROMO = (
    r"\bfollow my\b",
    r"\bmy channel\b",
    r"\bcheck out my\b",
    r"\buse code\b",
    r"\bdiscord\.gg/",
    r"\bonlyfans\b",
    r"\bsub to me\b",
)
_DOX = (
    r"\b\d{3}[-.]\d{3}[-.]\d{4}\b",
    r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b",
    r"\b\d{1,5}\s+[a-z0-9]+(?:\s+[a-z0-9]+){0,3}\s+(?:street|st|avenue|ave|road|rd|lane|ln)\b",
    r"\breal name is\b",
    r"\bhome address\b",
    r"\blives at\b",
)
_GAME = ("boss", "mob", "mobs", "hitbox", "hitboxes", "lag", "mechanics", "rng", "dps", "loot")
_RETRACTIONS = (
    "just kidding",
    "wait no",
    "never mind",
    "nevermind",
    "ignore that",
    "not you",
    "lol jk",
    "jk",
)


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    violated_rule: str
    confidence: float
    scratchpad: str


def first_clause(message: str) -> str:
    """The span a prefix-only policy commits on."""

    match = re.search(r"[.!?]+(?:\s+|$)", message.strip())
    if match is None:
        return message.strip()
    return message.strip()[: match.end()].strip()


def classify(message: str) -> PolicyDecision:
    text = message.lower()
    cut = _retraction_at(text)
    dox = _earliest(text, _DOX)
    severe = _kept(_earliest(text, _SEVERE), cut)
    insult = _kept(_directed_insult_at(text), cut)
    promo = _kept(_earliest(text, _PROMO), cut)
    spam = _spam_span(text, cut)

    if dox is not None:
        return _decision("BAN", "DOXXING", 0.96, _dox_note(text))
    if severe is not None:
        return _decision("BAN", "TOXICITY", 0.95, "Unretracted self-harm directive.")
    if insult is not None:
        return _decision("TIMEOUT_60S", "TOXICITY", 0.91, "Directed insult, with no later retraction.")
    if promo is not None:
        return _decision("TIMEOUT_60S", "SELF_PROMO", 0.9, "Self-promotion.")
    if spam is not None:
        return _decision("DELETE", "SPAM", 0.9, "Repeated links or a repeated token.")
    if cut is not None and _anything_before(text, cut):
        marker = _retraction_label(text)
        return _decision(
            "ALLOW",
            "NONE",
            0.78,
            f"An earlier violation is withdrawn by {marker}.",
        )
    return _decision("ALLOW", "NONE", 0.97, "Nothing in the message matches a rule.")


def explain_simulation(message: str) -> str:
    prefix = first_clause(message)
    early = classify(prefix)
    full = classify(message)
    return (
        f"Prefix the premature simulation commits on: {prefix} "
        f"→ {early.action} / {early.violated_rule}. "
        f"Full message → {full.action} / {full.violated_rule}."
    )


def structured_text(message: str, schema: type[BaseModel]) -> str:
    """JSON whose key order follows the schema. Decision uses the first field.

    A schema whose first field is `scratchpad` sees the whole message. Any other
    first field is treated as a premature commitment and sees the first clause.
    """

    first = next(iter(schema.model_fields))
    decision = classify(message if first == "scratchpad" else first_clause(message))
    payload = {
        "scratchpad": decision.scratchpad,
        "action": decision.action,
        "violated_rule": decision.violated_rule,
        "confidence": decision.confidence,
    }
    data = {name: payload[name] for name in schema.model_fields}
    return schema.model_validate(data).model_dump_json()


def unconstrained_text(message: str) -> str:
    decision = classify(message)
    payload = json.dumps(
        {
            "scratchpad": decision.scratchpad,
            "action": decision.action,
            "violated_rule": decision.violated_rule,
            "confidence": decision.confidence,
        },
        ensure_ascii=False,
    )
    kind = slip_for(message)
    if kind == "markdown_fence":
        return f"```json\n{payload}\n```"
    if kind == "preamble":
        return f"Sure — here is the JSON decision:\n{payload}"
    if kind == "trailing_commentary":
        return payload + "\n\nHope that helps!"
    if kind == "invalid_enum":
        return re.sub(r'"action": "([A-Z0-9_]+)"', '"action": "WARN"', payload, count=1)
    if kind == "invalid_json":
        return payload[:-1]
    return payload


def slip_for(message: str) -> str:
    """Stable syntax slip for this message. Clean is half the buckets."""

    slips = (
        "markdown_fence",
        "preamble",
        "invalid_enum",
        "trailing_commentary",
        "invalid_json",
        "clean",
        "clean",
        "clean",
        "clean",
        "clean",
    )
    # A fixed prefix keeps the labeled chat spread across every slip.
    # crc32 of the raw messages clusters on "clean".
    bucket = zlib.crc32(f"0:{message}".encode("utf-8")) % len(slips)
    return slips[bucket]


def generate(user: str, schema: type[BaseModel] | None) -> str:
    message = message_from_user_prompt(user)
    if schema is None:
        return unconstrained_text(message)
    return structured_text(message, schema)


def _decision(action: str, rule: str, confidence: float, scratchpad: str) -> PolicyDecision:
    return PolicyDecision(action, rule, confidence, scratchpad)


def _retraction_at(text: str) -> int | None:
    found: int | None = None
    for marker in _RETRACTIONS:
        match = re.search(rf"\b{re.escape(marker)}\b", text)
        if match and (found is None or match.start() < found):
            found = match.start()
    return found


def _retraction_label(text: str) -> str:
    index = _retraction_at(text)
    for marker in _RETRACTIONS:
        match = re.search(rf"\b{re.escape(marker)}\b", text)
        if match and match.start() == index:
            return f"{marker!r}"
    return "a later retraction"


def _earliest(text: str, patterns: tuple[str, ...]) -> int | None:
    found: int | None = None
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and (found is None or match.start() < found):
            found = match.start()
    return found


def _kept(span: int | None, cut: int | None) -> int | None:
    """Drop a hit that ends before a retraction. Keep hits that start after it."""

    if span is None:
        return None
    if cut is None or span >= cut:
        return span
    return None


def _directed_insult_at(text: str) -> int | None:
    span = _earliest(text, _INSULTS)
    if span is None:
        return None
    directed = re.search(r"\byou\b|\byour\b|\byou're\b|\byoure\b", text) is not None
    game = any(re.search(rf"\b{word}\b", text) for word in _GAME)
    if game and not directed:
        return None
    if not directed:
        return None
    return span


def _spam_span(text: str, cut: int | None) -> int | None:
    if cut is None:
        return 0 if _is_spam(text) else None
    if _is_spam(text[cut:]):
        return cut
    return None


def _is_spam(text: str) -> bool:
    if len(re.findall(r"https?://", text)) >= 3:
        return True
    words = re.findall(r"[a-z0-9']+", text)
    return len(words) >= 5 and len(set(words)) == 1


def _anything_before(text: str, cut: int) -> bool:
    head = text[:cut]
    return any(
        (
            _earliest(head, _SEVERE) is not None,
            _directed_insult_at(head) is not None,
            _earliest(head, _PROMO) is not None,
            _is_spam(head),
        )
    )


def _dox_note(text: str) -> str:
    bits = []
    if re.search(_DOX[0], text):
        bits.append("a phone number")
    if re.search(_DOX[1], text):
        bits.append("an email address")
    if re.search(_DOX[2], text) or re.search(_DOX[4], text) or re.search(_DOX[5], text):
        bits.append("a street address")
    if re.search(_DOX[3], text):
        bits.append("a real-world identity")
    joined = ", ".join(bits) or "private information"
    return f"Posts {joined}."
