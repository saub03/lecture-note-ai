"""Environment variables and model path settings."""

import os

# LLM provider: "api" (OpenAI/Gemini) or "local" (Ollama/vLLM)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "api")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
