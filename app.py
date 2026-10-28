"""Grammar lab dashboard. Run with `streamlit run app.py`."""

from __future__ import annotations

import html
import os

import pandas as pd
import streamlit as st

from src.fixtures import FIXTURES, FIXTURES_BY_ID
from src.grammar_inspector import (
    compile_gbnf,
    generation_steps,
    json_schema_text,
    to_openai_strict_schema,
)
from src.llm_client import (
    DEFAULT_HF_FILE,
    DEFAULT_HF_REPO,
    GeminiClient,
    HuggingFaceClient,
    MockClient,
    OpenAIClient,
    load_env_file,
)
from src.metrics import score_mode
from src.mock_moderator import explain_simulation
from src.pipeline import BenchRow, ModerationOutcome, run_bench, run_moderation
from src.policy import system_prompt
from src.schemas import (
    ExecutionMode,
    PrematureModerationDecision,
    ScratchpadModerationDecision,
    field_order,
    response_schema,
)

load_env_file()

st.set_page_config(page_title="Grammar Lab · Chat Auto-Timeout", page_icon="▣", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; max-width: 1180px; }
    div[data-testid="stMetric"] {
        background: #121821;
        border: 1px solid #243041;
        padding: 0.55rem 0.75rem;
        border-radius: 10px;
    }
    .chip {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-weight: 650;
        letter-spacing: 0.02em;
        color: #0B0F14;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_COLORS = {
    "ALLOW": "#6EE7B7",
    "DELETE": "#F5C16C",
    "TIMEOUT_60S": "#FF9B6A",
    "BAN": "#FF6B81",
}
_MODE_HELP = {
    ExecutionMode.UNCONSTRAINED_PROMPT_ONLY: (
        "Every token stays unmasked. The model is asked, in prose, to emit JSON. "
        "It can open with a sentence, wrap the object in a markdown fence, or invent an action such as WARN."
    ),
    ExecutionMode.GRAMMAR_PREMATURE_CONSTRAINT: (
        "Illegal tokens are masked, so the JSON is valid. The first value the model may sample is the action enum. "
        "That choice is made before any scratchpad tokens exist."
    ),
    ExecutionMode.GRAMMAR_SCRATCHPAD_FIRST: (
        "The action enum is masked only after a free-text scratchpad. "
        "The mask still guarantees a legal action. The action distribution can condition on the scratchpad."
    ),
}


def _bootstrap() -> None:
    st.session_state.setdefault("message", FIXTURES_BY_ID["late_dox"].message)
    st.session_state.setdefault("provider_choice", "Hugging Face")
    st.session_state.setdefault("openai_model", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    st.session_state.setdefault("gemini_model", os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    st.session_state.setdefault("openai_key", "")
    st.session_state.setdefault("gemini_key", "")


def _busy_text() -> str:
    if st.session_state.get("provider_choice") == "Hugging Face":
        return "Running Qwen2.5-1.5B locally. The first call also downloads the weights."
    if st.session_state.get("provider_choice") == "Mock":
        return "Scoring locally…"
    return "Calling the model…"


def _chip(action: str | None) -> str:
    if action is None:
        label, color = "NO DECISION", "#8B95A7"
    else:
        label, color = action, _COLORS.get(action, "#8B95A7")
    return (
        f"<span class='chip' style='background:{color}'>{html.escape(label)}</span>"
    )


def _client():
    provider = st.session_state.get("provider_choice", "Hugging Face")
    if provider == "Hugging Face":
        return HuggingFaceClient()
    if provider == "Mock":
        return MockClient()
    if provider == "OpenAI":
        return OpenAIClient(
            model=st.session_state["openai_model"],
            api_key=st.session_state["openai_key"] or None,
        )
    return GeminiClient(
        model=st.session_state["gemini_model"],
        api_key=st.session_state["gemini_key"] or None,
    )


def _sidebar() -> None:
    st.sidebar.header("Provider")
    st.sidebar.radio(
        "Provider",
        ["Hugging Face", "Mock", "OpenAI", "Gemini"],
        key="provider_choice",
        label_visibility="collapsed",
    )
    provider = st.session_state["provider_choice"]
    if provider == "Hugging Face":
        st.sidebar.caption(
            f"Local {DEFAULT_HF_REPO.split('/')[1].removesuffix('-GGUF')}, file `{DEFAULT_HF_FILE}`. "
            "No API key. The first call downloads the weights from the Hub. "
            "Grammar modes sample under the lab's GBNF."
        )
    elif provider == "Mock":
        st.sidebar.caption(
            "Offline simulation. Premature mode scores the first clause only, "
            "so the ordering gap is visible without an API key. Temperature on live calls is 0."
        )
    elif provider == "OpenAI":
        st.sidebar.text_input("Model", key="openai_model")
        st.sidebar.text_input("API key", type="password", key="openai_key", placeholder="OPENAI_API_KEY")
        st.sidebar.caption(
            "Uses response_format json_schema. gpt-4.1-mini is a non-reasoning default, "
            "so the only place a scratchpad can exist is in the output tokens."
        )
    else:
        st.sidebar.text_input("Model", key="gemini_model")
        st.sidebar.text_input("API key", type="password", key="gemini_key", placeholder="GEMINI_API_KEY")
        st.sidebar.caption("Uses response_schema. Leave the key blank to read GEMINI_API_KEY from the environment.")
    if provider in {"OpenAI", "Gemini"}:
        st.sidebar.caption("Keys stay in this browser session. They are not written to disk.")


def _render_outcome(outcome: ModerationOutcome) -> None:
    st.markdown(_chip(outcome.action), unsafe_allow_html=True)
    rule = outcome.violated_rule or "—"
    confidence = "—" if outcome.confidence is None else f"{outcome.confidence:.2f}"
    syntax = "raw JSON" if outcome.raw_compliant else outcome.failure
    st.write(f"Rule `{rule}` · confidence `{confidence}`")
    st.write(f"Syntax `{syntax}` · {outcome.latency_ms:.2f} ms · {len(outcome.raw_text)} chars")
    if outcome.parsed.detail:
        st.caption(outcome.parsed.detail)
    if outcome.scratchpad:
        st.markdown("**Scratchpad**")
        st.write(outcome.scratchpad)
    st.markdown("**Raw completion**")
    st.code(outcome.raw_text, language="json")


def _simulate() -> None:
    mode_labels = [mode.value for mode in ExecutionMode]
    mode_value = st.selectbox("Mode", mode_labels, key="sim_mode")
    mode = ExecutionMode(mode_value)
    st.caption(_MODE_HELP[mode])
    columns = st.columns([1, 1, 2])
    run = columns[0].button("Run this message", type="primary", key="btn_run")
    burst = columns[1].button("Replay fixture chat", key="btn_burst")
    if run:
        _store_single(mode)
    if burst:
        _store_burst(mode)
    single = st.session_state.get("single")
    if single and single["mode"] == mode.value:
        _render_outcome(single["outcome"])
    burst_rows = st.session_state.get("burst")
    if burst_rows and burst_rows["mode"] == mode.value:
        _render_burst(burst_rows["rows"], burst_rows["elapsed_ms"])


def _store_single(mode: ExecutionMode) -> None:
    message = st.session_state["message"].strip()
    if not message:
        st.warning("Write a chat message first.")
        return
    try:
        with st.spinner(_busy_text()):
            outcome = run_moderation(message, mode, _client())
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        return
    st.session_state["single"] = {"mode": mode.value, "outcome": outcome}


def _store_burst(mode: ExecutionMode) -> None:
    try:
        client = _client()
        with st.spinner(_busy_text()):
            rows = [(fixture, run_moderation(fixture.message, mode, client)) for fixture in FIXTURES]
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        return
    elapsed = sum(outcome.latency_ms for _, outcome in rows)
    st.session_state["burst"] = {"mode": mode.value, "rows": rows, "elapsed_ms": elapsed}


def _render_burst(rows: list[tuple], elapsed_ms: float) -> None:
    rate = len(rows) / (elapsed_ms / 1000) if elapsed_ms else float(len(rows))
    hits = sum(outcome.action == fixture.action for fixture, outcome in rows)
    raw = sum(outcome.raw_compliant for _, outcome in rows)
    left, middle, right = st.columns(3)
    left.metric("Matches gold", f"{hits}/{len(rows)}")
    middle.metric("Raw JSON", f"{raw}/{len(rows)}")
    right.metric("Throughput", f"{rate:.0f} msg/s")
    for fixture, outcome in rows:
        notes = []
        if outcome.action != fixture.action:
            notes.append("differs from gold")
        if not outcome.raw_compliant:
            notes.append(outcome.failure)
        label, body = st.columns([1, 5])
        label.markdown(_chip(outcome.action), unsafe_allow_html=True)
        suffix = f"  ({', '.join(notes)})" if notes else ""
        body.write(fixture.message + suffix)


def _compare() -> None:
    st.caption(
        "The model reads the whole input in every mode. Field order changes which output tokens "
        "exist before the action is sampled. The mock makes that gap large on purpose. "
        "A live non-reasoning model usually shows a smaller one. A reasoning model's hidden chain "
        "runs before the grammar and can wash the gap out."
    )
    if st.button("Run all three modes", type="primary", key="btn_compare"):
        message = st.session_state["message"].strip()
        if not message:
            st.warning("Write a chat message first.")
        else:
            try:
                client = _client()
                with st.spinner(_busy_text()):
                    outcomes = {
                        mode: run_moderation(message, mode, client) for mode in ExecutionMode
                    }
                st.session_state["compare"] = {
                    "provider": client.name,
                    "message": message,
                    "outcomes": outcomes,
                }
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
    payload = st.session_state.get("compare")
    if not payload:
        return
    if payload["provider"] == "mock":
        st.info(explain_simulation(payload["message"]))
    columns = st.columns(3)
    for column, mode in zip(columns, ExecutionMode, strict=True):
        with column:
            st.markdown(f"**{mode.value}**")
            order = field_order(response_schema(mode) or ScratchpadModerationDecision)
            if mode is ExecutionMode.UNCONSTRAINED_PROMPT_ONLY:
                st.caption("Prompt order: " + " → ".join(order) + " (unenforced)")
            else:
                st.caption("Grammar order: " + " → ".join(order))
            _render_outcome(payload["outcomes"][mode])


def _benchmark() -> None:
    calls = len(FIXTURES) * len(ExecutionMode)
    st.caption(
        f"Scores {len(FIXTURES)} labeled messages in all three modes ({calls} calls). "
        "On the mock, scratchpad accuracy matches the labels because the mock implements that policy. "
        "The number that matters there is the premature gap, and the unconstrained syntax rate. "
        "A live model is scored against the same labels. "
        "On the local Hugging Face model, sixty generations can take several minutes."
    )
    if st.button("Run benchmark", type="primary", key="btn_bench"):
        try:
            with st.spinner("Classifying the fixture chat…"):
                st.session_state["bench"] = run_bench(_client())
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
    rows: list[BenchRow] | None = st.session_state.get("bench")
    if not rows:
        return
    scores = [score_mode(mode, rows) for mode in ExecutionMode]
    columns = st.columns(3)
    for column, mode_score in zip(columns, scores, strict=True):
        with column:
            st.markdown(f"**{mode_score.mode.value}**")
            st.metric("Raw JSON", f"{mode_score.raw_compliant_rate:.0%}")
            st.metric("Action accuracy", f"{mode_score.action_accuracy:.0%}")
            st.caption(
                f"Correct and raw: {mode_score.action_correct_and_raw}/{mode_score.total}"
                f" · rule {mode_score.rule_correct}/{mode_score.total}"
                f" · p50 {mode_score.latency_p50_ms:.2f} ms"
            )
    chart = pd.DataFrame(
        {
            "raw JSON %": [round(item.raw_compliant_rate * 100) for item in scores],
            "action accuracy %": [round(item.action_accuracy * 100) for item in scores],
        },
        index=[item.mode.value.replace("_", " ") for item in scores],
    )
    st.bar_chart(chart)
    table = []
    for row in rows:
        loose = row.outcomes[ExecutionMode.UNCONSTRAINED_PROMPT_ONLY]
        premature = row.outcomes[ExecutionMode.GRAMMAR_PREMATURE_CONSTRAINT]
        scratch = row.outcomes[ExecutionMode.GRAMMAR_SCRATCHPAD_FIRST]
        table.append(
            {
                "id": row.fixture.id,
                "gold": f"{row.fixture.action}/{row.fixture.violated_rule}",
                "unconstrained": loose.action or loose.failure,
                "syntax": "raw" if loose.raw_compliant else loose.failure,
                "premature": premature.action,
                "scratchpad": scratch.action,
                "prefix miss": premature.action != row.fixture.action,
            }
        )
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    inspected = st.selectbox("Raw completions for", [row.fixture.id for row in rows], key="bench_pick")
    chosen = next(row for row in rows if row.fixture.id == inspected)
    st.caption(chosen.fixture.why)
    detail = st.columns(3)
    for column, mode in zip(detail, ExecutionMode, strict=True):
        with column:
            st.markdown(f"**{mode.value}**")
            st.code(chosen.outcomes[mode].raw_text, language="json")


def _inspector() -> None:
    choice = st.radio(
        "Artifact",
        ["Unconstrained prompt", "Premature schema", "Scratchpad schema"],
        horizontal=True,
        key="artifact",
    )
    if choice == "Unconstrained prompt":
        st.write(
            "This mode does not compile a grammar. The prose below is the only contract, "
            "and a completion can ignore it."
        )
        st.code(system_prompt(ExecutionMode.UNCONSTRAINED_PROMPT_ONLY), language="text")
        return
    model = PrematureModerationDecision if choice == "Premature schema" else ScratchpadModerationDecision
    st.write("Scaffolding such as braces, keys, and commas is forced. The steps below are the values the model actually samples.")
    for step in generation_steps(model):
        legal = "open vocabulary" if step.legal is None else " · ".join(step.legal)
        st.markdown(f"**{step.index}. `{step.field}` · {step.kind}**")
        st.caption(f"{step.summary} Legal continuations: {legal}")
    left, right = st.columns(2)
    schema_text = json_schema_text(model)
    gbnf = compile_gbnf(model)
    with left:
        st.markdown("**JSON Schema**")
        st.code(schema_text, language="json")
    with right:
        st.markdown("**GBNF**")
        st.caption("Paste into a llama.cpp grammar sampler. The same schema is what Outlines and llguidance compile into a mask.")
        st.code(gbnf, language="text")
        st.download_button(
            "Download GBNF",
            data=gbnf,
            file_name=f"{model.__name__}.gbnf",
            mime="text/plain",
            key=f"download_{model.__name__}",
        )
    with st.expander("OpenAI strict schema"):
        st.write(
            "Strict structured outputs drop numeric minimum and maximum. "
            "The 0-to-1 bound is rechecked with Pydantic after the call. The GBNF approximates it directly."
        )
        st.json(to_openai_strict_schema(model))


def main() -> None:
    _bootstrap()
    _sidebar()
    st.markdown("## Grammar lab")
    st.markdown(
        "A livestream chat moderation pipeline, and a bench for grammar-constrained decoding. "
        "The same policy is classified three ways: prompt-only JSON, a grammar that samples the action first, "
        "and a grammar that writes a scratchpad before the action."
    )
    st.session_state.setdefault("example_id", "late_dox")
    example_ids = [fixture.id for fixture in FIXTURES]
    picked = st.selectbox("Example", example_ids, key="example_id")
    load_column, _rest = st.columns([1, 3])
    if load_column.button("Load example", key="btn_load"):
        st.session_state["message"] = FIXTURES_BY_ID[picked].message
    st.caption(FIXTURES_BY_ID[picked].why)
    st.text_area("Message", key="message", height=90)
    simulate, compare, benchmark, inspector = st.tabs(
        ["Simulate", "Compare", "Benchmark", "Grammar inspector"]
    )
    with simulate:
        _simulate()
    with compare:
        _compare()
    with benchmark:
        _benchmark()
    with inspector:
        _inspector()


main()
