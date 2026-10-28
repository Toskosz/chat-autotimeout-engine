# Usage

Classify a livestream chat line three ways, or read the grammars the lab compiles. The design of those modes is in [DESIGN.md](DESIGN.md).

You need Python 3.11 or newer.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -e .` is enough to run the app. The `[dev]` extra adds pytest.

The install pulls Streamlit, Pydantic, the OpenAI and Gemini clients, `huggingface_hub`, and `llama-cpp-python`. Cloud keys are optional. The dashboard runs with no key on the local model or the mock.

## Tests

```bash
pytest
```

The suite does not download model weights and does not call OpenAI or Gemini.

## Dashboard

```bash
streamlit run app.py
```

Streamlit opens a local page (usually `http://localhost:8501`). The theme is the dark palette in `.streamlit/config.toml`. Usage stats collection is off.

The page is one message box and four tabs. The sidebar picks who answers.

### Pick a provider

| Provider | What you need | What it does |
| --- | --- | --- |
| **Hugging Face** | Nothing. First call downloads the weights. | Local [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF), Q4, about 1 GB, cached by the Hub after that. Grammar modes sample under the lab's GBNF. Inference stays on this machine. |
| **Mock** | Nothing. | Offline caricature. Premature mode scores only the first clause, so the ordering gap is obvious. A live model still reads the whole message and usually shows a smaller gap. |
| **OpenAI** | `OPENAI_API_KEY` | `response_format: json_schema` (strict). Default model `gpt-4.1-mini`. |
| **Gemini** | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `response_schema`. Default model `gemini-3.5-flash`. |

For a cloud provider, copy the example env file and fill in a key:

```bash
cp .env.example .env
```

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash
HF_MODEL_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF
HF_MODEL_FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf
```

You can also type a key and a model name in the sidebar. A sidebar key is kept in that browser session and is not written to disk. Leave the key blank to use the environment. The sidebar model overrides `OPENAI_MODEL` / `GEMINI_MODEL` for that session.

`HF_MODEL_REPO` and `HF_MODEL_FILE` change which GGUF the local provider loads. They are read from the environment, not from the sidebar.

Use a non-reasoning model if you want the field-order effect to show up. A reasoning model's hidden chain runs before the grammar and can wash the gap out. `gpt-4.1-mini` is the non-reasoning default so a scratchpad has to live in the output tokens.

Temperature is 0 on every live call.

### Pick a message

The box starts on the `late_dox` example: praise in the first sentence, a home address and a phone number after it. The dropdown lists all twenty fixture ids. **Load example** copies that fixture into the box and the caption under the dropdown says why it was labeled that way. You can also type any line.

### Simulate

Choose one mode, then:

- **Run this message** classifies the box once.
- **Replay fixture chat** classifies all twenty fixtures in that mode.

The result shows the action, the violated rule, confidence, a syntax label, latency, and length. A scratchpad is printed when the completion has one. The raw completion is below that.

Syntax `raw JSON` means the completion was the object and nothing else. Other labels (`markdown_fence`, `preamble`, `trailing_commentary`, `invalid_json`, `invalid_enum`, `schema_mismatch`, `empty`) name how it failed. Fence, preamble, and trailing text still recover a decision. An invented action, broken JSON, or an empty completion does not: the chip reads `NO DECISION`.

The fixture replay adds three numbers: how many actions match the gold label, how many completions are raw JSON, and messages per second. A row marked `differs from gold` disagreed with the hand label.

The first Hugging Face call also downloads the weights. The button shows that in the spinner. Later calls reuse the process cache.

### Compare

**Run all three modes** classifies the current message as unconstrained, premature, and scratchpad-first.

Each column prints the field order. Unconstrained order is the order the prompt asked for, and nothing enforces it. Grammar order is the order the sampler is masked in. Under that is the same result card as Simulate.

On the mock, a note above the columns states the prefix the premature simulation committed on and the full-message decision. That sentence is the caricature, not a claim about a live model.

### Benchmark

**Run benchmark** scores all twenty labeled messages in all three modes (sixty calls).

On the local model that can take several minutes. On the mock it returns immediately. The caption on the tab says how to read the mock numbers: scratchpad accuracy matches the labels because the mock implements that policy. The figures that matter there are the premature gap and the unconstrained syntax rate. A live model is scored against the same labels.

Each mode shows:

- **Raw JSON**: share of completions that validated with no repair.
- **Action accuracy**: share whose action matches the gold label, including actions recovered from dirty text.
- A caption with correct-and-raw count, rule matches, and median latency.

The bar chart plots raw JSON and action accuracy. The table is one row per fixture: gold label, unconstrained action and syntax, premature action, scratchpad action, and whether premature missed the gold label (`prefix miss`). Pick a fixture id to read all three raw completions and the reason that line was labeled.

### Grammar inspector

Three artifacts:

- **Unconstrained prompt.** The system prompt for that mode. There is no grammar. A completion can ignore the prose.
- **Premature schema.** Generation steps, JSON Schema, GBNF, and the OpenAI strict schema. The first step is the action enum.
- **Scratchpad schema.** The same views. The first step is free text; the action mask is the second step.

**Download GBNF** saves `PrematureModerationDecision.gbnf` or `ScratchpadModerationDecision.gbnf`. Paste that file into a llama.cpp grammar sampler. Outlines and llguidance compile the same JSON Schema into an equivalent token mask.

The OpenAI strict schema drops numeric `minimum` / `maximum`. The lab rechecks `0 ≤ confidence ≤ 1` with Pydantic after a cloud call. The GBNF approximates that range directly.

## Terminal preview

A text view of the six pivot messages on the mock, then both compiled grammars:

```bash
python -m src
```

After an editable install, the same entry point is:

```bash
chat-moderate
```

Columns are fixture id, gold label, and the three mode decisions. A leading `*` means the action was recovered from text that was not raw JSON. Pivot ids are the lines where the first clause and the full message disagree: `retract_insult`, `redirected_insult`, `retract_promo`, `retract_severe`, `late_dox`, `late_promo`.

This command always uses the mock. It does not load Qwen and it does not call a cloud API.

## What the labels mean

Gold labels follow the full-message policy. They are the score key for the benchmark, not a second set of instructions sent to the model.

| Action | When the label uses it |
| --- | --- |
| `ALLOW` | Ordinary chat, questions, criticism of the game, or a violation that is clearly retracted. |
| `DELETE` | Spam: repeated links, or one token repeated. |
| `TIMEOUT_60S` | A directed insult, or self-promotion. |
| `BAN` | Someone else's phone, email, street address, or real-world identity, or an unretracted self-harm directive. |

Doxxing stays a ban even if the writer walks it back. A retracted joke does not.
