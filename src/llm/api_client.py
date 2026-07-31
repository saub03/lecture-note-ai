"""API-based LLM client (OpenAI, Gemini, etc.)."""

from .base import BaseLLM


class APIClient(BaseLLM):
    def generate(self, prompt: str, context: str | None = None) -> str:
        raise NotImplementedError
