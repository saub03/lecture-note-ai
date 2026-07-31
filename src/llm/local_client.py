"""Local LLM client (Ollama, vLLM, etc.)."""

from .base import BaseLLM


class LocalClient(BaseLLM):
    def generate(self, prompt: str, context: str | None = None) -> str:
        raise NotImplementedError
