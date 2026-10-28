from src.llm_client import HuggingFaceClient, grammar_for
from src.schemas import PrematureModerationDecision, ScratchpadModerationDecision


class _FakeLlama:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("grammar") is None:
            return {"choices": [{"message": {"content": "Sure — ```json\n{\"action\": \"WARN\"}"}}]}
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"action":"ALLOW","violated_rule":"NONE","confidence":0.5}'
                    }
                }
            ]
        }


def test_grammar_modes_pass_gbnf_and_unconstrained_does_not():
    llama = _FakeLlama()
    client = HuggingFaceClient(llama=llama)
    client.generate(system="policy", user="Chat message:\nhi", schema=None)
    client.generate(
        system="policy",
        user="Chat message:\nhi",
        schema=PrematureModerationDecision,
    )
    assert "grammar" not in llama.calls[0]
    assert llama.calls[0]["temperature"] == 0.0
    assert llama.calls[1]["grammar"] is not None
    assert llama.calls[1]["messages"][0]["role"] == "system"


def test_compiled_schemas_are_grammars_llama_cpp_accepts():
    for model in (PrematureModerationDecision, ScratchpadModerationDecision):
        grammar = grammar_for(model)
        assert grammar is not None
