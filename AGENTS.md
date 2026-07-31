# AGENTS.md

## Project

Real-time ASR + AI lecture summarizer (Streamlit). Bilingual README (EN/KO).

## Commands

```bash
streamlit run app.py          # run the app
pip install -r requirements.txt
```

## Structure (per plan.md)

- `app.py` — Streamlit entrypoint
- `src/audio/` — microphone recording + local ASR (Faster-Whisper / Whisper.cpp)
- `src/document/` — lecture slide parsing (PDF/PPT) + pre-summarization
- `src/llm/` — LLM abstraction (API: OpenAI/Gemini, Local: Ollama/vLLM) via factory pattern
- `src/pipeline/` — real-time ASR text + slide context + LLM orchestration
- `ui/` — Streamlit components + custom CSS
- `config/settings.py` — env vars and model paths
- `data/` — gitignored temp storage (uploads, audio buffers)

## Conventions

- `.env` for secrets (gitignored); `.streamlit/secrets.toml` also gitignored
- `data/` contents are ephemeral — never commit uploaded files or audio buffers
- LLM clients follow a base abstract class (`src/llm/base.py`) with factory instantiation (`src/llm/factory.py`)
