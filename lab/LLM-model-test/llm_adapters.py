# -*- coding: utf-8 -*-
"""LLM 엔진 어댑터 패턴 (lab/LLM-model-test).

test-way.md 3.10(범용 계약)의 LLM 버전입니다. API(OpenAI/Gemini/Groq/OpenRouter)와
로컬(Ollama / MLX-LM / Transformers-MPS) 엔진을 하나의 인터페이스로 감쌉니다.

엔진마다 "자기 것 하나만" 고유 출력 → 계약 dict로 바꾸면, 다운스트림
(`llm_metrics.run_benchmark`, 가드 생성, 로깅, 파이프라인)은 절대 변하지 않습니다.
새 엔진 추가: `LLMAdapter`를 상속받아 `name`/`importable`/`_load`/`_generate_raw`만
구현하고 `ENGINE_REGISTRY`에 등록하면 끝입니다.

계약 dict (LLM_CONTRACT_KEYS):
    model           실제 모델 식별자
    text            생성된 텍스트
    latency_ms      요청 → 전체 완료 시간(ms)
    ttft_ms         첫 토큰까지 시간(ms, 스트리밍 시 측정, 아니면 None)
    ttfs_ms         첫 문장까지 시간(ms, 스트리밍 시 측정)
    hangul_ratio    출력 내 한글 비율 (코드 스위칭 지표)
    length_chars    출력 길이(문자)
    compliance_ok   정상성 프로브 통과 여부 (sanity_check)

API 키는 `.env`(gitignore)에서 로드하며, 키가 없으면 해당 엔진은
`adapter_available()` = False 가 되어 노트북이 조용히 건너뜁니다.
"""

import json
import os
import time
from abc import ABC, abstractmethod

import numpy as np

from llm_metrics import hangul_ratio, sanity_check, first_sentence
from llm_utils import load_env

# import 시 .env 자동 로드 (키 없이는 어댑터가 비활성화되도록).
load_env()

LLM_CONTRACT_KEYS = {
    "model", "text", "latency_ms", "ttft_ms", "ttfs_ms",
    "hangul_ratio", "length_chars", "compliance_ok",
}

DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0


def make_llm_contract(*, model, text, latency_s, ttft_s=None, ttfs_s=None) -> dict:
    """엔진 중립 결과 → 계약 dict."""
    text = (text or "").strip()
    return {
        "model": model,
        "text": text,
        "latency_ms": latency_s * 1000.0,
        "ttft_ms": ttft_s * 1000.0 if ttft_s is not None else None,
        "ttfs_ms": ttfs_s * 1000.0 if ttfs_s is not None else None,
        "hangul_ratio": hangul_ratio(text),
        "length_chars": len(text),
        "compliance_ok": sanity_check(text)[0],
    }


class LLMAdapter(ABC):
    """모든 LLM 엔진의 공통 인터페이스."""

    name = "base"
    default_model_id = None

    def __init__(self, model_id: str = None, system: str = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE, **options):
        self.model_id = model_id or self.default_model_id
        self.system = system          # 시스템 프롬프트 (없으면 모델 기본값)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.options = options
        self._model = None

    # -- 하위 클래스 구현 지점 ----------------------------------------------
    @classmethod
    @abstractmethod
    def importable(cls, **config) -> bool:
        """설치·키·서버 가용성 가드. False면 노트북이 건너뜁니다."""

    def _load(self):
        raise NotImplementedError

    def _generate_raw(self, prompt: str, stream: bool = False) -> dict:
        """(prompt) → {"text", "latency_s", "ttft_s", "ttfs_s", "events"}."""
        raise NotImplementedError

    # -- 공용 구현 ----------------------------------------------------------
    def load(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def generate(self, prompt: str, stream: bool = False) -> dict:
        """프롬프트 → 계약 dict. 엔진 예외는 그대로 전파(loud failure)."""
        raw = self._generate_raw(prompt, stream=stream)
        return make_llm_contract(
            model=self.model_id,
            text=raw["text"],
            latency_s=raw["latency_s"],
            ttft_s=raw.get("ttft_s"),
            ttfs_s=raw.get("ttfs_s"),
        )

    def stream_events(self, prompt: str) -> list:
        """스트리밍 (t_s, chunk) 이벤트 리스트. llm_metrics.analyze_stream 입력."""
        return self._generate_raw(prompt, stream=True).get("events", [])


# ---------------------------------------------------------------------------
# OpenAI 호환 API — OpenAI / Groq / OpenRouter / 로컬 vLLM (test-way.md P0/P4)
# ---------------------------------------------------------------------------

class OpenAICompatibleAdapter(LLMAdapter):
    """`openai` 패키지 기반, OpenAI 호환 API 공용 어댑터.

    `base_url`만 바꾸면 OpenAI / Groq / OpenRouter / 로컬 vLLM·Ollama
    OpenAI 엔드포인트를 모두 처리할 수 있습니다.
    """

    name = "openai-compatible"

    def __init__(self, model_id: str = None, system: str = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE,
                 api_key_env: str = None, base_url: str = None, **options):
        super().__init__(model_id=model_id, system=system,
                         max_tokens=max_tokens, temperature=temperature,
                         **options)
        self.api_key_env = api_key_env
        self.base_url = base_url

    @classmethod
    def importable(cls, **config) -> bool:
        try:
            import openai  # noqa: F401
        except Exception:
            return False
        env_var = config.get("api_key_env")
        return not env_var or bool(os.environ.get(env_var))

    def _load(self):
        from openai import OpenAI
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        return OpenAI(api_key=key, base_url=self.base_url or None)

    def _generate_raw(self, prompt: str, stream: bool = False) -> dict:
        client = self.load()
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        ttft_s = ttfs_s = None
        events = []

        if stream:
            resp = client.chat.completions.create(
                model=self.model_id, messages=messages,
                max_tokens=self.max_tokens, temperature=self.temperature,
                stream=True)
            buf = ""
            for chunk in resp:
                delta = (chunk.choices[0].delta.content
                         if chunk.choices else None)
                if not delta:
                    continue
                now = time.perf_counter() - t0
                if ttft_s is None:
                    ttft_s = now
                buf += delta
                if ttfs_s is None and first_sentence(buf):
                    ttfs_s = now
                events.append((now, delta))
            return {"text": buf, "latency_s": time.perf_counter() - t0,
                    "ttft_s": ttft_s, "ttfs_s": ttfs_s, "events": events}

        resp = client.chat.completions.create(
            model=self.model_id, messages=messages,
            max_tokens=self.max_tokens, temperature=self.temperature)
        text = resp.choices[0].message.content or ""
        return {"text": text, "latency_s": time.perf_counter() - t0,
                "ttft_s": ttft_s, "ttfs_s": ttfs_s, "events": events}


# ---------------------------------------------------------------------------
# Google Gemini (test-way.md P0 클라우드 단일 호출 경로)
# ---------------------------------------------------------------------------

class GeminiAdapter(LLMAdapter):
    """`google-genai` 패키지 기반 Gemini 어댑터 (generate_content)."""

    name = "gemini"

    def __init__(self, model_id: str = None, system: str = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE,
                 api_key_env: str = "GEMINI_API_KEY", **options):
        super().__init__(model_id=model_id, system=system,
                         max_tokens=max_tokens, temperature=temperature,
                         **options)
        self.api_key_env = api_key_env

    @classmethod
    def importable(cls, **config) -> bool:
        try:
            from google import genai  # noqa: F401
        except Exception:
            return False
        env_var = config.get("api_key_env", "GEMINI_API_KEY")
        return bool(os.environ.get(env_var))

    def _load(self):
        from google import genai
        return genai.Client(api_key=os.environ.get(self.api_key_env))

    def _generate_raw(self, prompt: str, stream: bool = False) -> dict:
        client = self.load()
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=self.system,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        t0 = time.perf_counter()
        ttft_s = ttfs_s = None
        events = []

        if stream:
            resp = client.models.generate_content_stream(
                model=self.model_id, contents=prompt, config=config)
            buf = ""
            for chunk in resp:
                try:
                    delta = chunk.text or ""
                except Exception:
                    delta = ""
                if not delta:
                    continue
                now = time.perf_counter() - t0
                if ttft_s is None:
                    ttft_s = now
                buf += delta
                if ttfs_s is None and first_sentence(buf):
                    ttfs_s = now
                events.append((now, delta))
            return {"text": buf, "latency_s": time.perf_counter() - t0,
                    "ttft_s": ttft_s, "ttfs_s": ttfs_s, "events": events}

        resp = client.models.generate_content(
            model=self.model_id, contents=prompt, config=config)
        text = (resp.text or "") if hasattr(resp, "text") else ""
        return {"text": text, "latency_s": time.perf_counter() - t0,
                "ttft_s": ttft_s, "ttfs_s": ttfs_s, "events": events}


# ---------------------------------------------------------------------------
# Ollama — 로컬 GGUF (llama.cpp) 추론, HTTP API
# ---------------------------------------------------------------------------

class OllamaAdapter(LLMAdapter):
    """Ollama 로컬 서버(기본 http://localhost:11434) HTTP API 어댑터.

    별도 Python 패키지가 필요 없고 `requests`로 호출합니다. 스트리밍은
    NDJSON 라인을 순회하며 TTFT/TTFS를 측정합니다.
    """

    name = "ollama"

    def __init__(self, model_id: str = None, system: str = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE,
                 base_url: str = "http://localhost:11434", **options):
        super().__init__(model_id=model_id, system=system,
                         max_tokens=max_tokens, temperature=temperature,
                         **options)
        self.base_url = base_url

    @classmethod
    def importable(cls, **config) -> bool:
        try:
            import requests  # noqa: F401
        except Exception:
            return False
        base = config.get("base_url", "http://localhost:11434")
        try:
            import requests
            return requests.get(f"{base}/api/tags", timeout=1).ok
        except Exception:
            return False

    def _load(self):
        return None   # HTTP 상태 비저장 — 로드 대상 없음.

    def _generate_raw(self, prompt: str, stream: bool = False) -> dict:
        import requests

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": self.temperature,
                        "num_predict": self.max_tokens},
        }
        if self.system:
            payload["system"] = self.system

        t0 = time.perf_counter()
        ttft_s = ttfs_s = None
        events = []

        with requests.post(f"{self.base_url}/api/generate", json=payload,
                           stream=stream, timeout=600) as r:
            r.raise_for_status()
            if not stream:
                obj = r.json()
                return {"text": obj.get("response", ""),
                        "latency_s": time.perf_counter() - t0,
                        "ttft_s": ttft_s, "ttfs_s": ttfs_s, "events": events}
            buf = ""
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                chunk = obj.get("response", "")
                if chunk:
                    now = time.perf_counter() - t0
                    if ttft_s is None:
                        ttft_s = now
                    buf += chunk
                    if ttfs_s is None and first_sentence(buf):
                        ttfs_s = now
                    events.append((now, chunk))
                if obj.get("done"):
                    break
        return {"text": buf, "latency_s": time.perf_counter() - t0,
                "ttft_s": ttft_s, "ttfs_s": ttfs_s, "events": events}


# ---------------------------------------------------------------------------
# MLX LM — Apple Silicon 네이티브 LLM (mlx-lm)
# ---------------------------------------------------------------------------

class MLXLLMAdapter(LLMAdapter):
    """`mlx-lm` 기반 어댑터. Hugging Face의 MLX 포맷 모델을 Metal로 추론합니다.

    모델 예:
        mlx-community/exaone-4.0-1.2b-4bit   (한국어, ~1GB, 4bit)
        mlx-community/Qwen3-8B-4bit          (다국어, ~5.4GB, 4bit)
        KYUNGYONG/DeepSeek-llama3.1-Bllossom-8B-Q4-mlx (한국어 파인튜닝)
    """

    name = "mlx-lm"

    @classmethod
    def importable(cls, **config) -> bool:
        try:
            import mlx_lm  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        from mlx_lm import load
        return load(self.model_id)   # (model, tokenizer) 튜플

    def _format_prompt(self, prompt: str):
        """시스템 프롬프트를 포함해 채팅 템플릿으로 감싼 프롬프트를 만든다.

        인스트럭트 모델은 채팅 템플릿(예: Qwen3의 <|im_start|>)을 따라야
        응답 품질과 JSON 준수율이 안정적입니다. 템플릿 적용이 불가능한
        모델은 "system\\n\\nuser" 평문으로 폴백합니다.

        `disable_thinking=True`(기본: Qwen3 계열)이면 thinking 단계를
        프리필로 건너뛰어(test-way.md 3.5 skip-prefill) JSON 등
        구조화 출력의 준수율을 높입니다.
        """
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        try:
            _, tokenizer = self.load()
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"{self.system}\n\n{prompt}" if self.system else prompt
        if self.options.get("disable_thinking"):
            text += "<think>\n\n</think>\n"
        return text

    def _generate_raw(self, prompt: str, stream: bool = False) -> dict:
        model, tokenizer = self.load()
        text_prompt = self._format_prompt(prompt)
        # mlx-lm 0.31은 temperature를 직접 받지 않으므로 sampler로 감싼다.
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=self.temperature)
        t0 = time.perf_counter()
        ttft_s = ttfs_s = None
        events = []

        if stream:
            from mlx_lm.generate import stream_generate
            buf = ""
            for resp in stream_generate(
                    model, tokenizer, prompt=text_prompt,
                    max_tokens=self.max_tokens, sampler=sampler):
                chunk = getattr(resp, "text", "") or ""
                if not chunk:
                    continue
                now = time.perf_counter() - t0
                if ttft_s is None:
                    ttft_s = now
                buf += chunk
                if ttfs_s is None and first_sentence(buf):
                    ttfs_s = now
                events.append((now, chunk))
            return {"text": buf, "latency_s": time.perf_counter() - t0,
                    "ttft_s": ttft_s, "ttfs_s": ttfs_s, "events": events}

        from mlx_lm import generate
        text = generate(model, tokenizer, prompt=text_prompt,
                        max_tokens=self.max_tokens, sampler=sampler)
        return {"text": text, "latency_s": time.perf_counter() - t0,
                "ttft_s": ttft_s, "ttfs_s": ttfs_s, "events": events}


# ---------------------------------------------------------------------------
# Transformers + MPS — 포괄 폴백 (MLX 변환이 없는 모델용)
# ---------------------------------------------------------------------------

class TransformersMPSAdapter(LLMAdapter):
    """`transformers` + torch MPS 어댑터. MLX 포맷이 없는 모델을 위한 폴백.

    스트리밍은 미지원(비스트리밍만)이며, TTFT/TTFS는 None으로 표시됩니다.
    """

    name = "transformers"

    @classmethod
    def importable(cls, **config) -> bool:
        try:
            import torch
            import transformers  # noqa: F401
            return torch.backends.mps.is_available()
        except Exception:
            return False

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16).to("mps")
        return model, tokenizer

    def _format_prompt(self, prompt: str) -> str:
        if self.system:
            return f"{self.system}\n\n{prompt}"
        return prompt

    def _generate_raw(self, prompt: str, stream: bool = False) -> dict:
        import torch
        model, tokenizer = self.load()
        inputs = tokenizer(self._format_prompt(prompt), return_tensors="pt")
        inputs = {k: v.to("mps") for k, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        return {"text": text, "latency_s": time.perf_counter() - t0,
                "ttft_s": None, "ttfs_s": None, "events": []}


# ---------------------------------------------------------------------------
# 팩토리 (test-way.md 3.10 / 4.3 비교 매트릭스의 "행" 단위)
# ---------------------------------------------------------------------------

ENGINE_REGISTRY = {
    "openai": OpenAICompatibleAdapter,
    "groq": OpenAICompatibleAdapter,
    "openrouter": OpenAICompatibleAdapter,
    "gemini": GeminiAdapter,
    "ollama": OllamaAdapter,
    "ollama-32b": OllamaAdapter,
    "mlx-qwen3-4b": MLXLLMAdapter,
    "mlx-qwen3-8b": MLXLLMAdapter,
    "mlx-bllossom": MLXLLMAdapter,
    "transformers": TransformersMPSAdapter,
}

ENGINE_DEFAULTS = {
    "openai": {"model_id": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    "groq": {"model_id": "llama-3.3-70b-versatile",
             "api_key_env": "GROQ_API_KEY",
             "base_url": "https://api.groq.com/openai/v1"},
    "openrouter": {"model_id": "openai/gpt-4o-mini",
                   "api_key_env": "OPENROUTER_API_KEY",
                   "base_url": "https://openrouter.ai/api/v1"},
    "gemini": {"model_id": "gemini-2.0-flash", "api_key_env": "GEMINI_API_KEY"},
    "ollama": {"model_id": "qwen2.5:7b", "base_url": "http://localhost:11434"},
    "ollama-32b": {"model_id": "qwen2.5:32b", "base_url": "http://localhost:11434"},
    "mlx-qwen3-4b": {"model_id": "mlx-community/Qwen3-4B-4bit",
                     "disable_thinking": True},
    "mlx-qwen3-8b": {"model_id": "mlx-community/Qwen3-8B-4bit",
                     "disable_thinking": True},
    "mlx-bllossom": {"model_id":
                     "KYUNGYONG/DeepSeek-llama3.1-Bllossom-8B-Q4-mlx"},
    "transformers": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
}


def _engine_config(engine: str, **overrides) -> dict:
    """엔진 기본 설정과 호출자 오버라이드를 합친다."""
    if engine not in ENGINE_REGISTRY:
        raise ValueError(
            f"미등록 엔진 '{engine}'. 등록됨: {sorted(ENGINE_REGISTRY)}")
    cfg = dict(ENGINE_DEFAULTS.get(engine, {}))
    cfg.update(overrides)
    return cfg


def adapter_available(engine: str) -> bool:
    """설치·키·서버 가용성 기준으로 벤치마크에 태울 수 있는 엔진인지."""
    if engine not in ENGINE_REGISTRY:
        return False
    cls = ENGINE_REGISTRY[engine]
    try:
        return cls.importable(**_engine_config(engine))
    except Exception:
        return False


def create_adapter(engine: str, **overrides) -> LLMAdapter:
    """엔진 이름 → 어댑터 인스턴스. 미등록/미가용 엔진은 명확히 실패합니다."""
    cfg = _engine_config(engine, **overrides)
    cls = ENGINE_REGISTRY[engine]
    if not cls.importable(**cfg):
        env = cfg.get("api_key_env")
        hint = f"  (키: {env})" if env else ""
        raise RuntimeError(
            f"'{engine}' 어댑터를 사용할 수 없습니다{hint}. "
            f"패키지 설치/키 설정을 확인하세요.")
    system = cfg.pop("system", None)
    max_tokens = cfg.pop("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = cfg.pop("temperature", DEFAULT_TEMPERATURE)
    return cls(model_id=cfg.pop("model_id", None), system=system,
               max_tokens=max_tokens, temperature=temperature, **cfg)


def make_generate_fn(adapter: LLMAdapter):
    """(prompt) -> 원시 텍스트 함수. llm_metrics.guarded_generate와 호환."""
    def fn(prompt: str) -> str:
        return adapter.generate(prompt, stream=False)["text"]
    return fn
