# Design

This project is a livestream chat moderation pipeline and a lab for the **Grammar** pattern: grammar-constrained decoding, and what field order does to an autoregressive model.

The same chat line is classified three ways into `ALLOW`, `DELETE`, `TIMEOUT_60S`, or `BAN`. The policy text is shared. What changes is whether a grammar masks the sampler, and which output tokens already exist when the action is sampled.

## The Grammar pattern

An autoregressive model samples one token at a time. Each new token is conditioned on the input and on every token already written. A constrained decoder (llama.cpp GBNF, Outlines, llguidance, OpenAI strict `json_schema`, Gemini `response_schema`) walks the schema in property order and sets the probability of any illegal continuation to zero. That order is the order the model is allowed to think in its own output.

The model reads the whole input in every mode. Field order changes the **output** tokens that exist before the action is sampled.

Two schemas describe the same decision and disagree about whether any free text is generated before the action.

**Premature constraint.** The first sampled value is the action enum. A reasoning field placed under `confidence` arrives after that token. It can justify the action. It cannot change it. The JSON is valid. The decision was locked before any scratchpad tokens existed.

**Scratchpad first.** `scratchpad` is free text, so almost any token is legal while it is written. The action enum is masked only after that text is already in the context. The mask still guarantees a legal action. It applies later, so the action distribution can condition on the scratchpad.

A hidden chain of thought on a reasoning model runs before the grammar and can wash this field-order effect out. The lab therefore defaults to non-reasoning models: local Qwen2.5-1.5B-Instruct, and `gpt-4.1-mini` on OpenAI. The scratchpad has to live in the output tokens for the ordering to be visible.

## The three modes

| Mode | Schema sent to the sampler | What is enforced | What to watch |
| --- | --- | --- | --- |
| `Unconstrained_Prompt_Only` | `None`. Every token stays unmasked. | The prompt asks for one JSON object, keys in scratchpad order. | Markdown fences, a preamble, trailing commentary, invented actions such as `WARN`, truncated JSON. |
| `Grammar_Premature_Constraint` | `PrematureModerationDecision` | A grammar whose first sampled value is `action`. | Valid JSON whose action was chosen before any scratchpad tokens existed. |
| `Grammar_Scratchpad_First` | `ScratchpadModerationDecision` | The same action mask, after a free-text `scratchpad`. | Valid JSON whose action can condition on that scratchpad. |

`response_schema()` in `src/schemas.py` is what the provider receives. Unconstrained mode sends no schema. `validation_model()` is what the parser uses after the call. Unconstrained completions are still checked against `ScratchpadModerationDecision`, because that is the shape the prompt asked for. A grammar completion is checked against the model that was compiled into the mask.

The two grammar modes receive the same system prompt (`POLICY` plus a one-line note that a decoder grammar fixes the shape). The schema is the only difference between them. Unconstrained mode appends a prose description of the JSON keys instead, because nothing else tells the model what to emit.

## Schemas

Both models set `extra = "forbid"`. Unknown keys are a schema mismatch.

`PrematureModerationDecision` fields, in generation order:

1. `action`: `ALLOW` | `DELETE` | `TIMEOUT_60S` | `BAN`
2. `violated_rule`: `NONE` | `TOXICITY` | `SPAM` | `SELF_PROMO` | `DOXXING`
3. `confidence`: float, `0 ≤ confidence ≤ 1`

`ScratchpadModerationDecision` inserts one field in front:

1. `scratchpad`: free string. The description asks the model to quote the span that decides the action, including a later retraction or a violation that follows a friendly opening.
2. `action`, `violated_rule`, `confidence` as above.

Pydantic field order is generation order. `field_order()` returns `model.model_fields` keys in that order, and the dashboard prints it above each compared completion.

## Policy

`src/policy.py` holds the shared moderation policy. The model is told to judge the whole message.

- A friendly opening does not excuse a later violation.
- A later retraction does excuse an earlier joke.
- Posting someone else's private information still stands when the writer walks it back.

Actions:

- `ALLOW`: good-faith chat, questions, emotes, criticism of the game, a joke that is clearly retracted.
- `DELETE`: spam, or a message that should disappear without a penalty.
- `TIMEOUT_60S`: a directed insult, harassment, or self-promotion.
- `BAN`: someone else's phone, email, street address, or real-world identity, or an unretracted self-harm directive.

`violated_rule` is one of `NONE`, `TOXICITY`, `SPAM`, `SELF_PROMO`, `DOXXING`. `confidence` is a number from 0 to 1.

The user turn is always `Chat message:\n` plus the raw line (`render_user`). `message_from_user_prompt` strips that prefix; the mock uses it to recover the line it is scoring.

## Pipeline

`run_moderation` in `src/pipeline.py` is one call:

1. Pick the response schema for the mode (`None` when unconstrained).
2. Call `client.generate(system=..., user=..., schema=...)`.
3. Record latency.
4. Run `analyze()` against the validation model.

The result is a `ModerationOutcome`: mode, provider name, raw completion, latency, and a `ParseAnalysis`. Action, rule, confidence, scratchpad, raw compliance, and the failure name are properties of that analysis.

`run_bench` runs every fixture through every mode and returns one `BenchRow` per fixture. The dashboard benchmark is that loop. A full bench is 20 messages × 3 modes = 60 generations.

## Parsing

`src/parse.py` separates a usable decision from a raw-compliant completion.

`raw_compliant` is true only when the stripped completion validates as the model with no extraction. Grammar modes should land here: the completion is the JSON object and nothing else.

When the raw text does not validate, the parser takes the span from the first `{` to the last `}` and tries again.

| Failure | Meaning | Decision recovered? |
| --- | --- | --- |
| `none` | Raw JSON matched the model. | Yes |
| `markdown_fence` | A fence surrounded a valid object. | Yes, `raw_compliant` false |
| `preamble` | Prose preceded the object. | Yes |
| `trailing_commentary` | Commentary followed the object. | Yes |
| `invalid_json` | No object, or the extracted span is not JSON. | No |
| `invalid_enum` | A closed field is outside its literal set (`WARN`, a bad rule). | No |
| `schema_mismatch` | Extra keys, missing keys, or a confidence outside 0–1. | No |
| `empty` | The completion was blank. | No |

`repaired` means a `ParseAnalysis` was built from a validated model. Fence, preamble, and trailing text set `repaired` true and `raw_compliant` false. Invented actions and broken JSON stay unrepaired: there is no decision to score.

Benchmark metrics keep these apart. Action accuracy counts a recovered action even when the syntax was dirty. `action_correct_and_raw` counts only decisions that were both correct and raw JSON. Raw JSON rate is the syntax score.

## Grammar compilation

`src/grammar_inspector.py` turns a Pydantic model into three artifacts and a matcher that proves what the grammar accepts.

**JSON Schema.** `model.model_json_schema()`, pretty-printed. Property order follows field declaration order.

**OpenAI strict schema.** `to_openai_strict_schema` walks that document and:

- drops `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `title`, and `default` (strict mode rejects numeric bounds);
- sets `additionalProperties: false` on every object;
- sets `required` to every property key.

The lab still checks `0 ≤ confidence ≤ 1` with Pydantic after the completion comes back.

**GBNF.** `compile_gbnf` renders a llama.cpp grammar whose field chain is the generation order. Each field is its own `-kv` rule (`"action" ws ":" ws action`) because llama.cpp rejects a single sequence of more than 15 symbols, and one flat object rule crosses that once keys, colons, and commas are spelled out. `fields` is then chained in pairs (`fields`, `fields-rest`, …) so every sequence stays short. Rule names replace `_` with `-`; llama.cpp accepts hyphens and rejects underscores.

Closed literals become an alternation of quoted strings. A string field reuses one JSON string rule (`"`, chars, escapes, `\u` plus four hex digits). A unit-interval number is approximated with a regular language, because numeric minimum and maximum are not context-free in general:

```text
("0" ("." [0-9]{1,8})?) | ("1" ("." [0]{1,8})?)
```

That accepts `0`, `0.5`, `1`, and `1.0`, and rejects `1.5` and `2`. Pydantic checks the bound again after decoding. Whitespace is `[ \t\n]*`, so compact and pretty JSON both match.

A header comment records which model was compiled and whether the first sampled value is a closed enum or free text.

**Matcher.** `parse_gbnf` / `grammar_accepts` is a small recursive matcher over the GBNF subset the compiler emits: literals, character classes, references, sequences, alternation, and `?` `*` `+` `{n}` `{n,m}`. It is how the tests prove the grammar accepts its own models, rejects `WARN`, rejects a confidence outside 0–1, and rejects a scratchpad object against the premature grammar (and the reverse). Match steps are capped so a runaway repetition fails as `GrammarError`.

**Generation steps.** `generation_steps` labels each field for the inspector:

- `enum_mask`: closed set; every other token, including `WARN`, is masked.
- `free_text`: open vocabulary; the action mask has not started.
- `number`: a JSON number restricted to 0 through 1.

Paste the downloaded `.gbnf` into a llama.cpp grammar sampler. Outlines and llguidance compile this same JSON Schema into an equivalent token mask. The file in the inspector is the readable form. Local Hugging Face runs already pass that compiled string to `LlamaGrammar.from_string`, so the token mask is the grammar in the inspector.

## Providers

`LLMClient` is a protocol with one method: `generate(*, system, user, schema) -> str`. The return value is the completion exactly as the model emitted it. `schema is None` leaves tokens unmasked. A schema turns constrained decoding on at that provider. Temperature is 0 everywhere, so repeated runs of the same prompt are as stable as the backend allows.

`load_env_file(".env")` fills missing environment variables. It does not override variables already set. The dashboard calls it at import.

### Hugging Face (default)

`HuggingFaceClient` loads a public GGUF with `huggingface_hub` and samples it with `llama-cpp-python`. No token and no hosted bill. Defaults:

- repo `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- file `qwen2.5-1.5b-instruct-q4_k_m.gguf` (about 1 GB, Q4)

Overrides: `HF_MODEL_REPO`, `HF_MODEL_FILE`. The sampler is cached on `(repo, filename)` for the process. `n_ctx` is 2048, `n_threads` is the CPU count, `seed` is 0, `verbose` is off. Chat completions use `max_tokens` 512. Grammar modes pass `grammar=LlamaGrammar.from_string(compile_gbnf(schema))`. Unconstrained mode omits `grammar`.

The first call downloads the weights into the Hub cache if `try_to_load_from_cache` misses. Later calls stay on this machine.

### Mock

`MockClient` never calls a model. It is a caricature of premature constraint, drawn so the ordering gap is visible offline.

`classify()` in `src/mock_moderator.py` implements the written policy as regular expressions, in this priority:

1. Doxxing (phone, email, street address, "real name is", "home address", "lives at") → `BAN` / `DOXXING`. A retraction does not clear it.
2. An unretracted severe phrase (`kys`, `kill yourself`, a kill threat, `swat you`) → `BAN` / `TOXICITY`.
3. A directed insult (`you` / `your` plus trash, idiot, stupid, uninstall, clown, …) → `TIMEOUT_60S` / `TOXICITY`. The same words aimed at the game (`boss`, `mobs`, `hitbox`, `lag`, `dps`, …) with nobody addressed are allowed.
4. Self-promotion (follow my, my channel, check out my, use code, discord.gg, onlyfans, sub to me) → `TIMEOUT_60S` / `SELF_PROMO`.
5. Spam (three or more `http` links, or five or more identical tokens) → `DELETE` / `SPAM`.
6. An earlier insult, promo, severe line, or spam that ends before a retraction marker (`just kidding`, `wait no`, `never mind`, `ignore that`, `not you`, `jk`, …) → `ALLOW` / `NONE`.
7. Otherwise → `ALLOW` / `NONE`.

`first_clause()` cuts at the first `.` `!` or `?`. `structured_text` looks at the schema's first field. If that field is `scratchpad`, the mock classifies the whole message. Any other first field is treated as a premature commitment and classifies the first clause only. Key order in the emitted JSON follows the schema, and the text is dumped through Pydantic so it is raw-compliant and accepted by the compiled GBNF.

A live model still reads the whole message and usually shows a smaller gap. The mock scores only the first clause on purpose.

Unconstrained mock output uses the full-message policy, then applies a stable syntax slip. `slip_for` hashes `crc32("0:" + message)` into ten buckets: fence, preamble, invalid enum (`WARN`), trailing commentary, truncated JSON, and five clean buckets. Clean is half the space. A star in the terminal preview means the action was recovered from text that was not raw JSON.

`explain_simulation` prints the prefix decision next to the full-message decision. The Compare tab shows that sentence when the provider is the mock.

### OpenAI

`OpenAIClient` calls `chat.completions.create` with `temperature=0` and `max_completion_tokens=600`, timeout 45 seconds. Grammar modes set:

```text
response_format: {
  type: json_schema,
  json_schema: { name, strict: true, schema: <strictified JSON Schema> }
}
```

The key is `OPENAI_API_KEY` or the sidebar field. The default model is `gpt-4.1-mini` (`OPENAI_MODEL`). The sidebar can override the model. A reasoning model thinks before the grammar and can wash the field-order effect out.

### Gemini

`GeminiClient` calls `models.generate_content` with `temperature=0` and `max_output_tokens=600`. Grammar modes set `response_mime_type=application/json` and `response_schema` to the Pydantic model itself. The key is `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or the sidebar field. The default model is `gemini-3.5-flash` (`GEMINI_MODEL`). An empty response is reported as a possible safety-filter block.

## Fixtures and scores

`src/fixtures.py` holds twenty hand-labeled lines. Gold labels are the full-message policy in the mock, and the tests assert that: `classify(message)` matches `(action, violated_rule)` for every fixture.

Six **pivot** ids disagree between the first clause and the full message. Those are the cases the premature simulation is built to miss:

| id | First clause | Full message |
| --- | --- | --- |
| `retract_insult` | directed insult | withdrawn → `ALLOW` |
| `redirected_insult` | "You're trash." | redirected at the boss → `ALLOW` |
| `retract_promo` | "follow my channel" | withdrawn → `ALLOW` |
| `retract_severe` | `kys` | marked as a joke → `ALLOW` |
| `late_dox` | "Nice play!" | address and phone → `BAN` |
| `late_promo` | a complaint about lag | "follow my channel" → `TIMEOUT_60S` |

`score_mode` in `src/metrics.py` reduces a bench to, per mode: raw-compliant count, parsed (repaired) count, action accuracy, actions that are both correct and raw, rule accuracy, and latency p50. Rates divide by the fixture count.

On the mock those numbers are determined by construction:

- both grammar modes are raw JSON on every row;
- scratchpad action accuracy is 20/20, because the mock implements the gold policy;
- premature action accuracy is 20 − 6, one miss per pivot;
- unconstrained raw JSON rate is below 100%, because some slips are dirty.

On a live model the same labels are an external score. Scratchpad accuracy is no longer tautological. The comparison that still matters is the gap between premature and scratchpad on the pivot lines, next to the unconstrained syntax rate.

## Dashboard

`app.py` is a Streamlit lab, not a second moderation policy. It calls `run_moderation` and `run_bench` and renders the artifacts from `grammar_inspector`.

Session state holds the message (default `late_dox`), the provider, cloud model names and keys, and the last single run, burst, comparison, and bench. API keys typed in the sidebar stay in that browser session. They are not written to disk. A blank key falls through to the environment.

Four tabs:

- **Simulate.** One mode on the current message, or a replay of all twenty fixtures in that mode. The burst reports gold matches, raw-JSON count, and throughput.
- **Compare.** All three modes on one message, with the grammar or prompt field order printed above each raw completion. The mock also prints the prefix-versus-full explanation.
- **Benchmark.** Sixty calls, three score columns (raw JSON rate, action accuracy, correct-and-raw, rule hits, p50), a bar chart, a per-fixture table with a `prefix miss` flag, and the raw completion for a picked fixture.
- **Grammar inspector.** The unconstrained system prompt, or either schema's generation steps, JSON Schema, GBNF (with a download), and the OpenAI strict schema.

Action chips use a fixed color per label. `NO DECISION` is the chip when parsing recovered no action.

## Module map

| Module | Role |
| --- | --- |
| `src/schemas.py` | The three modes and the two Pydantic models. Field order is the exhibit. |
| `src/policy.py` | Shared policy text and the user-turn wrapper. |
| `src/grammar_inspector.py` | Pydantic → JSON Schema, OpenAI strict schema, and GBNF, plus a matcher. |
| `src/llm_client.py` | Hugging Face (llama.cpp + GBNF), Mock, OpenAI, and Gemini behind one `generate()`. |
| `src/mock_moderator.py` | Deterministic policy, first-clause caricature, and unconstrained syntax slips. |
| `src/parse.py` | Raw compliance, repair, and named failures. |
| `src/pipeline.py` | One moderation call, and the fixture bench. |
| `src/metrics.py` | Per-mode scores against the hand labels. |
| `src/fixtures.py` | Twenty labeled lines. Six of them are pivots. |
| `src/__main__.py` | Terminal preview of the pivot rows and both grammars, on the mock. |
| `app.py` | Simulate, compare, score, and read the compiled grammar. |

## Tests

`tests/test_grammar.py` locks field order, generation-step kinds, GBNF acceptance and rejection (`WARN`, confidence outside 0–1, cross-schema objects), the llama.cpp field chain, and the strict-schema cleanup.

`tests/test_modes.py` locks gold labels, the pivot set, the shared grammar prompt, mock behavior (prefix decision, full-message decision, raw JSON, slips), and the bench arithmetic (premature misses exactly the pivots).

`tests/test_parse.py` locks the failure taxonomy: raw JSON, fence, preamble, trailing text, invented enum, broken JSON, empty, extra and missing fields.

`tests/test_huggingface.py` covers the local client with a stand-in sampler so the suite does not download weights.
