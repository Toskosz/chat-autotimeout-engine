"""Compile a Pydantic model into JSON Schema and llama.cpp GBNF.

Production samplers (llama.cpp, Outlines, llguidance) turn this grammar into a
token mask: at each step, any vocabulary token that cannot continue a legal
derivation is given probability zero. This module emits the grammar a person
can read, and a small matcher so the lab can prove what that grammar accepts.

The string rule is the llama.cpp JSON string, shortened a little. Numeric
minimum and maximum are not context-free in general; a confidence between
0 and 1 is approximated with a regular numeric language, and Pydantic checks
the bound again after decoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Literal, get_args, get_origin

from pydantic import BaseModel

_STEP_BUDGET = 100_000


class GrammarError(ValueError):
    """The GBNF text itself could not be parsed, or a match ran away."""


@dataclass(frozen=True)
class GenerationStep:
    index: int
    field: str
    kind: str
    summary: str
    legal: tuple[str, ...] | None


@dataclass(frozen=True)
class _Lit:
    value: str


@dataclass(frozen=True)
class _Ref:
    name: str


@dataclass(frozen=True)
class _Class:
    negated: bool
    ranges: tuple[tuple[str, str], ...]
    chars: frozenset[str]

    def matches(self, char: str) -> bool:
        hit = any(start <= char <= end for start, end in self.ranges) or char in self.chars
        return not hit if self.negated else hit


@dataclass(frozen=True)
class _Seq:
    items: tuple[object, ...]


@dataclass(frozen=True)
class _Alt:
    options: tuple[object, ...]


@dataclass(frozen=True)
class _Rep:
    inner: object
    low: int
    high: int | None


_Expr = _Lit | _Ref | _Class | _Seq | _Alt | _Rep


@dataclass
class _Grammar:
    rules: dict[str, _Expr]

    def matches(self, text: str) -> bool:
        if "root" not in self.rules:
            raise GrammarError("Grammar has no root rule.")
        matcher = _Matcher(self.rules)
        return any(end == len(text) for end in matcher.alts(_Ref("root"), text, 0))


def json_schema_text(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), indent=2)


def to_openai_strict_schema(model: type[BaseModel]) -> dict:
    """Subset OpenAI strict structured outputs accept.

    Strict mode rejects numeric minimum and maximum. The lab still validates
    those bounds with Pydantic after the completion comes back.
    """

    return _strictify(model.model_json_schema())


def generation_steps(model: type[BaseModel]) -> list[GenerationStep]:
    steps = []
    for index, (name, field) in enumerate(model.model_fields.items(), start=1):
        values = _literal_values(field.annotation)
        if values is not None:
            kind = "enum_mask"
            legal = tuple(f'"{value}"' for value in values)
            summary = "Closed set. Every other token is masked, including invented labels such as WARN."
        elif _is_str(field.annotation):
            kind = "free_text"
            legal = None
            summary = "Free string. The action mask has not started, so these tokens can condition it."
        else:
            kind = "number"
            legal = ("0", "0.5", "1", "1.0")
            summary = "A JSON number restricted to the range 0 through 1."
        steps.append(GenerationStep(index, name, kind, summary, legal))
    return steps


def compile_gbnf(model: type[BaseModel]) -> str:
    """Render `model` as a GBNF grammar whose field chain is the generation order.

    Each field is its own rule. llama.cpp rejects a single sequence of more
    than 15 symbols, and one flat object rule crosses that once the keys,
    colons, and commas are spelled out.
    """

    aux: list[str] = []
    needs_string = False
    kv_names: list[str] = []
    kv_rules: list[str] = []
    for name, field in model.model_fields.items():
        symbol, extra, uses_string = _value_symbol(_rule_name(name), field)
        kv_name = f"{_rule_name(name)}-kv"
        kv_names.append(kv_name)
        kv_rules.append(
            f"{kv_name} ::= {_gbnf_literal(chr(34) + name + chr(34))} ws \":\" ws {symbol}"
        )
        aux.extend(extra)
        needs_string = needs_string or uses_string
    lines = [
        _header(model),
        'root ::= "{" ws fields ws "}" ws',
        *_chain_fields(kv_names),
        *kv_rules,
        *aux,
    ]
    if needs_string:
        lines.extend(
            [
                'string ::= "\\"" chars "\\""',
                "chars ::= char*",
                'char ::= [^"\\\\] | escape',
                'escape ::= "\\\\" (["\\\\/bfnrt] | "u" hex4)',
                "hex4 ::= [0-9a-fA-F]{4}",
            ]
        )
    lines.append("ws ::= [ \\t\\n]*")
    return "\n".join(lines) + "\n"


def grammar_accepts(gbnf: str, text: str) -> bool:
    return parse_gbnf(gbnf).matches(text)


def parse_gbnf(gbnf: str) -> _Grammar:
    rules: dict[str, _Expr] = {}
    for name, body in _split_rules(gbnf):
        parser = _Parser(body)
        expr = parser.parse_alt()
        parser.skip()
        if not parser.eof():
            raise GrammarError(f"Trailing syntax in rule {name!r}: {body[parser.i :]!r}")
        rules[name] = expr
    if not rules:
        raise GrammarError("No rules found.")
    return _Grammar(rules)


def _chain_fields(names: list[str]) -> list[str]:
    """Pair fields so every sequence stays short enough for llama.cpp."""

    rules: list[str] = []
    pending = list(names)
    rule = "fields"
    while pending:
        chunk = pending[:2]
        pending = pending[2:]
        pieces = chunk + ([f"{rule}-rest"] if pending else [])
        body = ' "," ws '.join(pieces)
        rules.append(f"{rule} ::= {body}")
        rule = f"{rule}-rest"
    return rules


def _rule_name(name: str) -> str:
    """llama.cpp rule names accept hyphens and reject underscores."""

    return name.replace("_", "-")


def _header(model: type[BaseModel]) -> str:
    first = next(iter(model.model_fields))
    if _literal_values(model.model_fields[first].annotation) is not None:
        note = (
            "# The first sampled value is a closed enum. "
            "The action is chosen before any scratchpad tokens exist."
        )
    else:
        note = (
            "# The first sampled value is free text. "
            "Action tokens are masked only after that text is in the context."
        )
    return f"# Compiled from {model.__name__}\n# Field order is generation order.\n{note}"


def _value_symbol(name: str, field: object) -> tuple[str, list[str], bool]:
    annotation = field.annotation  # type: ignore[attr-defined]
    values = _literal_values(annotation)
    if values is not None:
        alternatives = " | ".join(_gbnf_literal(f'"{value}"') for value in values)
        return name, [f"{name} ::= {alternatives}"], False
    if _is_str(annotation):
        return "string", [], True
    if _is_number(annotation) and _bounds_are_unit_interval(field):
        rule = (
            f'{name} ::= ("0" ("." [0-9]{{1,8}})?) | ("1" ("." [0]{{1,8}})?)'
        )
        return name, [rule], False
    return "number", ['number ::= "-"? ([0] | [1-9] [0-9]*) ("." [0-9]+)?'], False


def _bounds_are_unit_interval(field: object) -> bool:
    low = high = None
    for meta in getattr(field, "metadata", ()):
        if type(meta).__name__ == "Ge":
            low = meta.ge
        elif type(meta).__name__ == "Le":
            high = meta.le
    return low == 0 and high == 1 or low == 0.0 and high == 1.0


def _literal_values(annotation: object) -> tuple[str, ...] | None:
    if get_origin(annotation) is Literal:
        return tuple(str(value) for value in get_args(annotation))
    return None


def _is_str(annotation: object) -> bool:
    return annotation is str


def _is_number(annotation: object) -> bool:
    return annotation in {float, int}


def _is_name_char(char: str) -> bool:
    return char.isalnum() or char in "_-"


def _is_rule_name(name: str) -> bool:
    return bool(name) and (name[0].isalpha() or name[0] == "_") and all(_is_name_char(char) for char in name)


def _gbnf_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _strictify(node: object) -> object:
    if isinstance(node, list):
        return [_strictify(item) for item in node]
    if not isinstance(node, dict):
        return node
    cleaned = {
        key: _strictify(value)
        for key, value in node.items()
        if key not in {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "title", "default"}
    }
    if cleaned.get("type") == "object" or "properties" in cleaned:
        cleaned["additionalProperties"] = False
        if "properties" in cleaned:
            cleaned["required"] = list(cleaned["properties"].keys())
    return cleaned


def _split_rules(src: str) -> list[tuple[str, str]]:
    markers = _assign_markers(src)
    if not markers:
        raise GrammarError("No '::=' rules found.")
    chunks: list[tuple[str, str]] = []
    for index, at in enumerate(markers):
        name_end = at
        while name_end > 0 and src[name_end - 1].isspace():
            name_end -= 1
        name_start = name_end
        while name_start > 0 and _is_name_char(src[name_start - 1]):
            name_start -= 1
        name = src[name_start:name_end]
        if not _is_rule_name(name):
            raise GrammarError(f"Bad rule name before column {at}.")
        body_start = at + 3
        body_end = _name_start(src, markers[index + 1]) if index + 1 < len(markers) else len(src)
        chunks.append((name, src[body_start:body_end]))
    return chunks


def _name_start(src: str, assign_at: int) -> int:
    cursor = assign_at
    while cursor > 0 and src[cursor - 1].isspace():
        cursor -= 1
    while cursor > 0 and _is_name_char(src[cursor - 1]):
        cursor -= 1
    return cursor


def _assign_markers(src: str) -> list[int]:
    markers = []
    index = 0
    depth = 0
    length = len(src)
    while index < length:
        char = src[index]
        if char == "#":
            newline = src.find("\n", index)
            index = length if newline < 0 else newline
            continue
        if char == '"':
            index = _skip_string(src, index)
            continue
        if char == "[":
            index = _skip_class(src, index)
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and src.startswith("::=", index):
            markers.append(index)
            index += 3
            continue
        index += 1
    return markers


def _skip_string(src: str, index: int) -> int:
    index += 1
    while index < len(src):
        if src[index] == "\\":
            index += 2
            continue
        if src[index] == '"':
            return index + 1
        index += 1
    raise GrammarError("Unterminated string in grammar.")


def _skip_class(src: str, index: int) -> int:
    index += 1
    while index < len(src):
        if src[index] == "\\":
            index += 2
            continue
        if src[index] == "]":
            return index + 1
        index += 1
    raise GrammarError("Unterminated character class in grammar.")


class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.i = 0

    def eof(self) -> bool:
        self.skip()
        return self.i >= len(self.source)

    def skip(self) -> None:
        while self.i < len(self.source):
            if self.source[self.i].isspace():
                self.i += 1
                continue
            if self.source[self.i] == "#":
                newline = self.source.find("\n", self.i)
                self.i = len(self.source) if newline < 0 else newline
                continue
            break

    def peek(self) -> str:
        return self.source[self.i] if self.i < len(self.source) else ""

    def expect(self, char: str) -> None:
        self.skip()
        if self.peek() != char:
            raise GrammarError(f"Expected {char!r} at column {self.i}.")
        self.i += 1

    def parse_alt(self) -> _Expr:
        options = [self.parse_seq()]
        while True:
            self.skip()
            if self.peek() != "|":
                break
            self.i += 1
            options.append(self.parse_seq())
        if len(options) == 1:
            return options[0]
        return _Alt(tuple(options))

    def parse_seq(self) -> _Expr:
        items: list[_Expr] = []
        while True:
            self.skip()
            if self.i >= len(self.source) or self.peek() in "|)":
                break
            items.append(self.parse_atom())
        if not items:
            raise GrammarError("Empty expression.")
        if len(items) == 1:
            return items[0]
        return _Seq(tuple(items))

    def parse_atom(self) -> _Expr:
        self.skip()
        char = self.peek()
        if char == '"':
            atom: _Expr = _Lit(self._parse_string())
        elif char == "[":
            atom = self._parse_class()
        elif char == "(":
            self.i += 1
            atom = self.parse_alt()
            self.expect(")")
        elif char.isalpha() or char == "_":
            atom = _Ref(self._parse_ident())
        else:
            raise GrammarError(f"Unexpected {char!r} at column {self.i}.")
        return self._parse_postfix(atom)

    def _parse_postfix(self, atom: _Expr) -> _Expr:
        self.skip()
        char = self.peek()
        if char == "?":
            self.i += 1
            return _Rep(atom, 0, 1)
        if char == "*":
            self.i += 1
            return _Rep(atom, 0, None)
        if char == "+":
            self.i += 1
            return _Rep(atom, 1, None)
        if char == "{":
            self.i += 1
            low = self._parse_int()
            self.skip()
            if self.peek() == "}":
                self.i += 1
                return _Rep(atom, low, low)
            self.expect(",")
            high = self._parse_int()
            self.expect("}")
            return _Rep(atom, low, high)
        return atom

    def _parse_int(self) -> int:
        self.skip()
        start = self.i
        while self.peek().isdigit():
            self.i += 1
        if start == self.i:
            raise GrammarError(f"Expected a number at column {self.i}.")
        return int(self.source[start : self.i])

    def _parse_ident(self) -> str:
        start = self.i
        self.i += 1
        while _is_name_char(self.peek()):
            self.i += 1
        return self.source[start : self.i]

    def _parse_string(self) -> str:
        self.i += 1
        chars: list[str] = []
        while self.i < len(self.source):
            char = self.source[self.i]
            if char == '"':
                self.i += 1
                return "".join(chars)
            if char == "\\":
                self.i += 1
                if self.i >= len(self.source):
                    break
                escape = self.source[self.i]
                chars.append({"n": "\n", "t": "\t", "r": "\r"}.get(escape, escape))
                self.i += 1
                continue
            chars.append(char)
            self.i += 1
        raise GrammarError("Unterminated string literal.")

    def _parse_class(self) -> _Class:
        self.i += 1
        negated = False
        if self.peek() == "^":
            negated = True
            self.i += 1
        ranges: list[tuple[str, str]] = []
        chars: list[str] = []
        if self.peek() == "]":
            chars.append("]")
            self.i += 1
        while self.peek() and self.peek() != "]":
            start = self._class_char()
            if self.peek() == "-" and self.source[self.i + 1 : self.i + 2] not in {"", "]"}:
                self.i += 1
                ranges.append((start, self._class_char()))
            else:
                chars.append(start)
        self.expect("]")
        return _Class(negated, tuple(ranges), frozenset(chars))

    def _class_char(self) -> str:
        if self.peek() == "\\":
            self.i += 1
            escape = self.peek()
            self.i += 1
            return {"n": "\n", "t": "\t", "r": "\r"}.get(escape, escape)
        char = self.peek()
        if not char:
            raise GrammarError("Unterminated character class.")
        self.i += 1
        return char


class _Matcher:
    def __init__(self, rules: dict[str, _Expr]) -> None:
        self.rules = rules
        self.steps = 0

    def alts(self, expr: _Expr, text: str, index: int) -> Iterator[int]:
        self.steps += 1
        if self.steps > _STEP_BUDGET:
            raise GrammarError("Match exceeded its step budget.")
        if isinstance(expr, _Lit):
            if text.startswith(expr.value, index):
                yield index + len(expr.value)
            return
        if isinstance(expr, _Class):
            if index < len(text) and expr.matches(text[index]):
                yield index + 1
            return
        if isinstance(expr, _Ref):
            try:
                target = self.rules[expr.name]
            except KeyError as exc:
                raise GrammarError(f"Undefined rule {expr.name!r}.") from exc
            yield from self.alts(target, text, index)  # type: ignore[arg-type]
            return
        if isinstance(expr, _Alt):
            for option in expr.options:
                yield from self.alts(option, text, index)  # type: ignore[arg-type]
            return
        if isinstance(expr, _Seq):
            yield from self._sequence(expr.items, text, index)
            return
        if isinstance(expr, _Rep):
            yield from self._repeat(expr, text, index)
            return
        raise GrammarError(f"Unknown expression {type(expr).__name__}.")

    def _sequence(self, items: tuple[object, ...], text: str, index: int) -> Iterator[int]:
        if not items:
            yield index
            return
        head, *tail = items
        for mid in self.alts(head, text, index):  # type: ignore[arg-type]
            yield from self._sequence(tuple(tail), text, mid)

    def _repeat(self, expr: _Rep, text: str, index: int) -> Iterator[int]:
        ends = [index]
        cursor = index
        while expr.high is None or len(ends) - 1 < expr.high:
            nxt = next((end for end in self.alts(expr.inner, text, cursor) if end > cursor), None)  # type: ignore[arg-type]
            if nxt is None:
                break
            cursor = nxt
            ends.append(cursor)
        for count, end in reversed(list(enumerate(ends))):
            if count >= expr.low:
                yield end
