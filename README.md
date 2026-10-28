# Chat auto-timeout engine

A livestream chat moderation pipeline and a lab for the **Grammar** pattern: grammar-constrained decoding, and what field order does to an autoregressive model.

Three modes classify the same chat line into `ALLOW`, `DELETE`, `TIMEOUT_60S`, or `BAN`.

| Mode | What is enforced | What to watch |
| --- | --- | --- |
| `Unconstrained_Prompt_Only` | Nothing. The prompt asks for JSON. | Markdown fences, a preamble, invented actions such as `WARN`. |
| `Grammar_Premature_Constraint` | A grammar whose first sampled value is the action. | Valid JSON whose action was chosen before any scratchpad tokens existed. |
| `Grammar_Scratchpad_First` | The same action mask, after a free-text `scratchpad`. | Valid JSON whose action can condition on that scratchpad. |

The model reads the whole input in every mode. Field order changes the output tokens that exist before the action is sampled. A reasoning field placed under `confidence` can justify an action. It cannot change it.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
streamlit run app.py
```

A terminal preview of the pivot messages and both grammars:

```bash
python -m src
```

The dashboard defaults to a free local model from the Hugging Face Hub: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF), quantized to Q4. The first run downloads the weights (about 1 GB) into the Hub cache. After that, inference stays on this machine. No API key and no hosted bill. Grammar modes hand the lab's compiled GBNF to llama.cpp, so the token mask is the grammar in the inspector.

The mock is still there. It is a caricature: premature mode scores only the first clause, which makes the ordering gap obvious. A live model still reads the whole message and usually shows a smaller gap.

## Cloud providers

Copy `.env.example` to `.env` if you want a real model.

- **OpenAI** uses `response_format: json_schema` (strict). Default model: `gpt-4.1-mini`, a non-reasoning model, so a scratchpad has to live in the output tokens. The sidebar can override the model. Reasoning models think before the grammar and can wash the field-order effect out.
- **Gemini** uses `response_schema`. Default model: `gemini-3.5-flash`.

Strict OpenAI schemas reject numeric `minimum` / `maximum`. The lab drops those keywords for the request and rechecks `0 ≤ confidence ≤ 1` with Pydantic. The GBNF approximates the same range directly.

## Where the lesson lives

- `src/schemas.py` — the two Pydantic models. Field order is the exhibit.
- `src/grammar_inspector.py` — Pydantic → JSON Schema, OpenAI strict schema, and GBNF. A small matcher proves what the grammar accepts, including the rejection of `WARN` and of a confidence outside 0–1.
- `src/llm_client.py` — local Hugging Face (llama.cpp + GBNF), `Mock`, OpenAI, and Gemini behind one `generate()` method. `schema=None` leaves tokens unmasked.
- `src/fixtures.py` — twenty labeled chat lines. Six of them disagree between the first clause and the full message.
- `app.py` — simulate one mode, compare all three, score the fixture chat, and read the compiled grammar.

Paste the downloaded `.gbnf` into a llama.cpp grammar sampler. Outlines and llguidance compile this same JSON Schema into an equivalent token mask; the file in the inspector is the readable form.
