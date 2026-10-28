"""Provider adapters.

`generate` returns the raw completion text. Passing a schema turns on
grammar-constrained decoding at the provider (OpenAI `response_format`,
Gemini `response_schema`). Passing None leaves every token unmasked.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from src.grammar_inspector import compile_gbnf, to_openai_strict_schema
from src.mock_moderator import generate as mock_generate

# Public GGUF. No token and no hosted bill: the weights download once and run on this machine.
DEFAULT_HF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_HF_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
_LLAMAS: dict[tuple[str, str], object] = {}


class LLMClient(Protocol):
    name: str

    def generate(self, *, system: str, user: str, schema: type[BaseModel] | None) -> str:
        """Return the completion exactly as the model emitted it."""


class MockClient:
    """Offline caricature of the three modes. See `src/mock_moderator.py`."""

    name = "mock"

    def generate(self, *, system: str, user: str, schema: type[BaseModel] | None) -> str:
        del system
        return mock_generate(user, schema)


class HuggingFaceClient:
    """Local Qwen weights from the Hugging Face Hub, sampled with llama.cpp.

    Grammar modes pass the lab's compiled GBNF straight to the sampler, so the
    token mask is the grammar shown in the inspector. Unconstrained mode leaves
    the sampler open.
    """

    def __init__(
        self,
        repo: str | None = None,
        filename: str | None = None,
        llama: object | None = None,
    ) -> None:
        self.repo = repo or os.environ.get("HF_MODEL_REPO", DEFAULT_HF_REPO)
        self.filename = filename or os.environ.get("HF_MODEL_FILE", DEFAULT_HF_FILE)
        self._llama = llama
        self.name = f"huggingface:{self.repo}"

    def generate(self, *, system: str, user: str, schema: type[BaseModel] | None) -> str:
        llama = self._llama if self._llama is not None else load_huggingface_model(self.repo, self.filename)
        kwargs: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "seed": 0,
        }
        if schema is not None:
            kwargs["grammar"] = grammar_for(schema)
        response = llama.create_chat_completion(**kwargs)
        content = response["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not content:
            raise RuntimeError("The local model returned an empty completion.")
        return content


def grammar_for(schema: type[BaseModel]):
    """Turn a Pydantic model into the llama.cpp grammar object."""

    try:
        from llama_cpp import LlamaGrammar
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed, so the local Hugging Face model cannot run."
        ) from exc
    return LlamaGrammar.from_string(compile_gbnf(schema))


def load_huggingface_model(repo: str, filename: str):
    """Download the GGUF on first use and keep the sampler for the process."""

    key = (repo, filename)
    cached = _LLAMAS.get(key)
    if cached is not None:
        return cached
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "Install llama-cpp-python and huggingface_hub to run the local Hugging Face model."
        ) from exc
    cached = try_to_load_from_cache(repo_id=repo, filename=filename)
    path = cached if isinstance(cached, str) else hf_hub_download(repo_id=repo, filename=filename)
    model = Llama(
        model_path=path,
        n_ctx=2048,
        n_threads=os.cpu_count() or 4,
        verbose=False,
        seed=0,
    )
    _LLAMAS[key] = model
    return model


class OpenAIClient:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set OPENAI_API_KEY, or paste a key in the sidebar.")
        self.name = f"openai:{model}"

    def generate(self, *, system: str, user: str, schema: type[BaseModel] | None) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is not installed.") from exc

        kwargs: dict = {}
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": to_openai_strict_schema(schema),
                },
            }
        client = OpenAI(api_key=self.api_key, timeout=45.0)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_completion_tokens=600,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The model returned an empty completion.")
        return content


class GeminiClient:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY, or paste a key in the sidebar.")
        self.name = f"gemini:{model}"

    def generate(self, *, system: str, user: str, schema: type[BaseModel] | None) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("The google-genai package is not installed.") from exc

        config: dict = {
            "system_instruction": system,
            "temperature": 0.0,
            "max_output_tokens": 600,
        }
        if schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema
        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(**config),
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError(
                "The model returned no text. A safety filter may have blocked the completion."
            )
        return text


def load_env_file(path: str = ".env") -> None:
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
