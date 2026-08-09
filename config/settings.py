"""Environment variables and model path settings."""

import os
# huggingface token
HF_TOKEN = os.getenv("HF_TOKEN", "")

# LLM provider: "api" (OpenAI/Gemini) or "local" (Ollama/vLLM)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "api")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# ASR provider: "mlx-whisper" (Apple Silicon MLX, 메인) or "faster-whisper" (CPU int8 교차 검증)
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "mlx-whisper")
ASR_MODEL = os.getenv("ASR_MODEL", "")  # 비워두면 엔진별 기본 모델 사용
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "ko")
