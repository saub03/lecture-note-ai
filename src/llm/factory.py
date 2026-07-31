"""Instantiate the LLM client selected in config/settings."""

from config import settings

from .api_client import APIClient
from .base import BaseLLM
from .local_client import LocalClient


def create_llm() -> BaseLLM:
    if settings.LLM_PROVIDER == "local":
        return LocalClient()
    return APIClient()
