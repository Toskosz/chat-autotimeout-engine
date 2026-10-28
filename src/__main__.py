"""Terminal preview of the lab. The dashboard is `streamlit run app.py`."""

from src.fixtures import FIXTURES, PIVOT_IDS
from src.grammar_inspector import compile_gbnf
from src.llm_client import MockClient
from src.pipeline import run_moderation
from src.schemas import (
    ExecutionMode,
    PrematureModerationDecision,
    ScratchpadModerationDecision,
)


def main() -> None:
    client = MockClient()
    print("Messages where the first clause and the full message disagree\n")
    header = f"{'id':<20} {'gold':<22} {'unconstrained':<22} {'premature':<22} {'scratchpad':<22}"
    print(header)
    print("-" * len(header))
    for fixture in FIXTURES:
        if fixture.id not in PIVOT_IDS:
            continue
        cells = [fixture.id, f"{fixture.action}/{fixture.violated_rule}"]
        for mode in ExecutionMode:
            outcome = run_moderation(fixture.message, mode, client)
            if outcome.action is None:
                cells.append(outcome.failure)
            else:
                mark = "" if outcome.raw_compliant else "*"
                cells.append(f"{mark}{outcome.action}/{outcome.violated_rule}")
        print(f"{cells[0]:<20} {cells[1]:<22} {cells[2]:<22} {cells[3]:<22} {cells[4]:<22}")
    print("\n* means the action was recovered from text that was not raw JSON.\n")
    print(compile_gbnf(PrematureModerationDecision))
    print(compile_gbnf(ScratchpadModerationDecision))


if __name__ == "__main__":
    main()
