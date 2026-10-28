# Chat auto-timeout engine

A livestream chat moderation pipeline and a lab for the **Grammar** pattern: grammar-constrained decoding, and what field order does to an autoregressive model.

Three modes classify the same chat line into `ALLOW`, `DELETE`, `TIMEOUT_60S`, or `BAN`.

| Mode | What is enforced |
| --- | --- |
| `Unconstrained_Prompt_Only` | Nothing. The prompt asks for JSON. |
| `Grammar_Premature_Constraint` | A grammar whose first sampled value is the action. |
| `Grammar_Scratchpad_First` | The same action mask, after a free-text `scratchpad`. |

- **[Design](DESIGN.md)** — the pattern, the schemas, how the grammars are compiled, and how each provider samples.
- **[Usage](USAGE.md)** — install, the dashboard, providers, and the terminal preview.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
streamlit run app.py
```
