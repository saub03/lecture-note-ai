"""Common LLM interface shared by API and local clients."""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract LLM client."""

    @abstractmethod
    def generate(self, prompt: str, context: str | None = None) -> str:
        """Generate a response for *prompt*, optionally grounded on slide *context*."""
