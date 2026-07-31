# -*- coding: utf-8 -*-
"""ASR 엔진 어댑터 패턴 (lab/ASR-model-test).

test-way.md 3.10(범용 계약)의 구현입니다. Colab/T4 의존 엔진(faster-whisper,
CUDA 전제 funasr 등) 대신 Apple Silicon에서 동작하는 엔진을 어댑터로 감쌌습니다.

엔진마다 "자기 것 하나만" 고유 출력 → 계약 dict로 바꾸면,
다운스트림(벤치마크 `asr_metrics.run_benchmark`, 로깅, 파이프라인)은 절대 변하지
않습니다. 새 엔진을 추가하는 법: `ASRAdapter`를 상속받아 `name`/`_load`/
`_transcribe_raw`만 구현하고 `ENGINE_REGISTRY`에 등록하면 끝입니다.

계약 dict (CONTRACT_KEYS):
    utt_id          발화 ID (없으면 "utt_000")
    engine          어댑터 이름 (예: "mlx-whisper")
    model           실제 모델 식별자 (예: "mlx-community/whisper-large-v3-turbo")
    text            전사 텍스트
    language        언어 코드 ("ko" 등)
    confidence_ok   judge_confidence 결과 (test-way.md 2.3)
    avg_logprob     Whisper 계열 평균 로그 확률 (없으면 None)
    no_speech_prob  Whisper 계열 무음 확률 (없으면 None)
    latency_ms      전사 소요 시간 (ms)
    rtf             처리 시간 / 오디오 길이
    segments        세그먼트 리스트 (엔진이 안 주면 [])

사용법:
    adapter = create_adapter("mlx-whisper")           # 기본 = large-v3-turbo
    result = adapter.transcribe("audio.wav")          # 경로 또는 np.array
    fn = make_transcribe_fn(adapter)                  # asr_metrics 벤치마크용
    row = asr_metrics.run_benchmark("mlx-whisper", fn, eval_set)

선택(optional) 엔진은 import 실패 시 adapter_available() = False 입니다.
install 없이 노트북이 깨지지 않도록 가드해 두었습니다.
"""

import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from asr_metrics import avg_logprob_from_segments, judge_confidence

SAMPLE_RATE = 16000
CONTRACT_KEYS = {
    "utt_id", "engine", "model", "text", "language",
    "confidence_ok", "avg_logprob", "no_speech_prob",
    "latency_ms", "rtf", "segments",
}

# SenseVoice 리치 전사 태그: <|zh|> <|en|> <|yue|> <|ja|> <|ko|> <|nospeech|>
# <|withitn|> <|nocache|> <|emotion_*|> <|event_*|> ... 전부 제거 대상.
_SENSEVOICE_TAG = re.compile(r"<\|[^|]*\|>")


def _audio_duration_s(audio_or_path, sr: int = SAMPLE_RATE) -> float:
    """경로면 librosa로, 배열이면 길이/샘플레이트로 오디오 길이(초)를 얻는다."""
    if isinstance(audio_or_path, (str, os.PathLike)):
        import librosa
        return float(librosa.get_duration(path=str(audio_or_path)))
    return len(np.asarray(audio_or_path)) / sr


def _to_waveform(audio_or_path) -> np.ndarray:
    """경로 또는 numpy 배열 → 16kHz float32 모노 웨이브폼."""
    if isinstance(audio_or_path, (str, os.PathLike)):
        import librosa
        y, _ = librosa.load(str(audio_or_path), sr=SAMPLE_RATE, mono=True)
        return y.astype(np.float32)
    return np.asarray(audio_or_path, dtype=np.float32)


def _as_path(audio_or_path, suffix=".wav") -> str:
    """배열이면 임시 wav로 저장해 경로를 돌려준다 (funasr 등 경로만 받는 엔진용)."""
    if isinstance(audio_or_path, (str, os.PathLike)):
        return str(audio_or_path)
    import tempfile
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    import soundfile as sf
    sf.write(path, np.asarray(audio_or_path, dtype=np.float32), SAMPLE_RATE)
    return path


def make_contract(*, utt_id, engine, model, text, language, avg_logprob,
                  no_speech_prob, latency_s, duration_s, segments) -> dict:
    """엔진 중립 결과 → 계약 dict (test-way.md 3.10)."""
    # RTF(실시간 계수) = 처리 시간 / 오디오 길이 (test-way.md 2.2).
    # 오디오 길이를 알 수 없으면(0 또는 None) 측정 불가 → None 처리.
    if duration_s is None or duration_s <= 0:
        rtf = None
    else:
        rtf = latency_s / duration_s
    return {
        "utt_id": utt_id or "utt_000",     # 발화 식별자 (없으면 기본값)
        "engine": engine,                  # 어댑터 이름 ("mlx-whisper" 등)
        "model": model,                    # 실제 사용 모델 식별자
        "text": (text or "").strip(),      # 전사 텍스트 (공백 제거)
        "language": language,              # 언어 코드 ("ko")
        "confidence_ok": judge_confidence(avg_logprob, text),  # 신뢰도 판정
        "avg_logprob": avg_logprob,        # Whisper 계열 신뢰도 신호
        "no_speech_prob": no_speech_prob,  # 무음 확률 (환각 위험 신호)
        "latency_ms": latency_s * 1000.0,  # 전사 소요 시간(ms)
        "rtf": rtf,                        # 실시간 계수
        "segments": segments,              # 세그먼트 상세 (없으면 [])
    }


class ASRAdapter(ABC):
    """모든 ASR 엔진의 공통 인터페이스. 이름 하나 + 구현 하나면 계약을 준수한다."""

    name = "base"
    default_model_id = None

    def __init__(self, model_id: str = None, language: str = "ko", **options):
        self.model_id = model_id or self.default_model_id
        self.language = language
        self.options = options
        self._model = None

    # -- 하위 클래스 구현 지점 ----------------------------------------------
    @classmethod
    def importable(cls) -> bool:
        """필수 패키지가 설치돼 있는지(가드). 미설치면 노트북이 건너뜁니다."""
        return True

    def _load(self):
        raise NotImplementedError

    def _transcribe_raw(self, audio):
        raise NotImplementedError

    # -- 공용 구현 ----------------------------------------------------------
    def load(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def transcribe(self, audio, utt_id: str = None,
                   audio_duration_s: float = None) -> dict:
        """오디오(경로 또는 16kHz 배열) → 계약 dict. 절대 예외를 던지지 않고
        텍스트가 없어도 계약 형태로 반환합니다 (test-way.md 3.3)."""
        # 지연/RTF 계산에 필요한 오디오 길이(초). 호출자가 주면 그대로 사용.
        duration_s = audio_duration_s if audio_duration_s is not None \
            else _audio_duration_s(audio)
        t0 = time.perf_counter()          # 전사 시간 측정 시작
        raw = self._transcribe_raw(audio) # 엔진별 실제 호출 (하위 클래스 구현)
        latency_s = time.perf_counter() - t0  # 순수 전사 소요 시간

        text = (raw.get("text") or "").strip()
        # 엔진이 no_speech_prob 수치를 주지 않았지만 no_speech 플래그만
        # 준 경우(SenseVoice의 <|nospeech|> 태그 등)는 1.0으로 매핑.
        no_speech = raw.get("no_speech_prob")
        if no_speech is None and raw.get("no_speech"):
            no_speech = 1.0
        return make_contract(
            utt_id=utt_id,
            engine=self.name,
            model=self.model_id,
            text=text,
            language=raw.get("language") or self.language,
            avg_logprob=raw.get("avg_logprob"),
            no_speech_prob=no_speech,
            latency_s=latency_s,
            duration_s=duration_s,
            segments=raw.get("segments") or [],
        )


# ---------------------------------------------------------------------------
# MLX Whisper — Apple Silicon 네이티브 (faster-whisper 대체)
# ---------------------------------------------------------------------------

class MLXWhisperAdapter(ASRAdapter):
    """mlx-whisper 래퍼. faster-whisper(large-v3-turbo)의 Apple Silicon 대체품.

    세그먼트에 avg_logprob / no_speech_prob / compression_ratio를 그대로 주므로
    asr_metrics의 신뢰도·환각 함수와 완전 호환됩니다. beam_size / initial_prompt
    / temperature 등은 options로 넘깁니다.
    """

    name = "mlx-whisper"
    default_model_id = "mlx-community/whisper-large-v3-turbo"

    @classmethod
    def importable(cls) -> bool:
        """필수 패키지가 설치·동작하는지(가드). 미설치/호환 불가면 False."""
        try:
            import mlx_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        import mlx_whisper
        return mlx_whisper

    def _transcribe_raw(self, audio):
        import mlx_whisper
        # mlx-whisper는 경로를 ffmpeg로 읽으므로, 16kHz 웨이브폼으로 미리
        # 변환해 넘긴다 (ffmpeg 시스템 의존성 제거).
        waveform = _to_waveform(audio)
        result = mlx_whisper.transcribe(
            waveform,
            path_or_hf_repo=self.model_id,
            language=self.language,
            verbose=False,
            **self.options,  # beam_size, initial_prompt, temperature ...
        )
        segments = result.get("segments") or []
        no_speech_vals = [s.get("no_speech_prob") for s in segments]
        no_speech = max(
            (v for v in no_speech_vals if v is not None), default=None)
        return {
            "text": result.get("text", ""),
            "language": result.get("language") or self.language,
            "avg_logprob": avg_logprob_from_segments(segments),
            "no_speech_prob": no_speech,
            "segments": segments,
        }


# ---------------------------------------------------------------------------
# OpenAI Whisper — torch MPS 기준선 (어댑터 스왑 시연용)
# ---------------------------------------------------------------------------

class OpenAIWhisperAdapter(ASRAdapter):
    """openai-whisper 래퍼. MPS(Apple Silicon)에서 동작하는 기존 기준선.

    Colab의 faster-whisper(CTranslate2/CUDA)와 같은 Whisper 계열이므로 CER 기준을
    맞춰볼 수 있습니다. fp16은 MPS에서 불안정해 fp16=False(안전)를 기본으로 합니다.
    """

    name = "openai-whisper"
    default_model_id = "turbo"

    def __init__(self, model_id: str = "turbo", language: str = "ko",
                 device: str = "mps", **options):
        super().__init__(model_id=model_id, language=language, **options)
        self.device = device
        self.options.setdefault("fp16", False)

    @classmethod
    def importable(cls) -> bool:
        try:
            import whisper  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        import whisper
        return whisper.load_model(self.model_id, device=self.device)

    def _transcribe_raw(self, audio):
        model = self.load()
        # openai-whisper는 경로를 ffmpeg로 읽으므로, 경로를 16kHz 웨이브폼으로
        # 미리 변환해 넘긴다 (ffmpeg 시스템 의존성 제거, Apple Silicon 친화).
        waveform = _to_waveform(audio)
        result = model.transcribe(waveform, language=self.language,
                                  **self.options)
        segments = result.get("segments") or []
        no_speech_vals = [s.get("no_speech_prob") for s in segments]
        no_speech = max(
            (v for v in no_speech_vals if v is not None), default=None)
        return {
            "text": result.get("text", ""),
            "language": result.get("language") or self.language,
            "avg_logprob": avg_logprob_from_segments(segments),
            "no_speech_prob": no_speech,
            "segments": segments,
        }


# ---------------------------------------------------------------------------
# SenseVoice-Small (funasr) — CTC 속도 + <|nospeech|> 태그 (test-way.md P4)
# ---------------------------------------------------------------------------

class SenseVoiceAdapter(ASRAdapter):
    """funasr 기반 SenseVoice-Small. CPU 추론으로 Apple Silicon에서 충분히 빠릅니다.

    출력 텍스트에 <|ko|>, <|nospeech|> 같은 리치 태그가 붙으므로 제거하고,
    <|nospeech|>가 있으면 계약의 no_speech_prob=1.0으로 매핑합니다.
    """

    name = "sensevoice"
    default_model_id = "FunAudioLLM/SenseVoiceSmall"

    @classmethod
    def importable(cls) -> bool:
        try:
            from funasr import AutoModel  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        import logging
        from funasr import AutoModel
        # funasr의 세부 로그/진행바를 조용히 만든다 (노트북 출력 정리).
        logging.getLogger("funasr").setLevel(logging.ERROR)
        logging.getLogger("modelscope").setLevel(logging.ERROR)
        return AutoModel(
            model=self.model_id,
            hub="hf",
            device=self.options.pop("device", "cpu"),
            disable_update=True,
        )

    def _transcribe_raw(self, audio):
        import contextlib
        import io
        model = self.load()
        path = _as_path(audio)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):  # tqdm 진행바 억제
                res = model.generate(input=path, cache={},
                                     language=self.language, use_itn=True)
        finally:
            if isinstance(audio, np.ndarray):
                os.unlink(path)
        raw_text = res[0]["text"]
        is_nospeech = "<|nospeech|>" in raw_text
        text = _SENSEVOICE_TAG.sub("", raw_text).strip()
        return {
            "text": "" if is_nospeech else text,
            "language": self.language,
            "no_speech_prob": 1.0 if is_nospeech else 0.0,
            "avg_logprob": None,
            "segments": [],
        }


# ---------------------------------------------------------------------------
# Qwen3-ASR-0.6B (qwen-asr) — LLM 기반, 코드 스위칭 강점 (test-way.md P4)
# ---------------------------------------------------------------------------

class Qwen3ASRAdapter(ASRAdapter):
    """qwen-asr 기반 Qwen3-ASR-0.6B. transformers device_map으로 MPS/CPU 추론.

    transcribe()가 (waveform, sr) 배치를 받으므로 내부에서 배열로 변환합니다.
    """

    name = "qwen3-asr"
    default_model_id = "Qwen/Qwen3-ASR-0.6B"

    def __init__(self, model_id: str = None, language: str = "ko", **options):
        super().__init__(model_id=model_id, language=language, **options)
        self.options.setdefault("device_map", "cpu")
        self.options.setdefault("max_new_tokens", 256)

    @classmethod
    def importable(cls) -> bool:
        try:
            from qwen_asr import Qwen3ASRModel  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        import torch
        from qwen_asr import Qwen3ASRModel
        options = dict(self.options)
        device_map = options.pop("device_map", "cpu")
        return Qwen3ASRModel.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map=device_map,
            **options,
        )

    def _transcribe_raw(self, audio):
        model = self.load()
        y = _to_waveform(audio)
        r = model.transcribe(audio=[(y, SAMPLE_RATE)],
                             language=["Korean"])[0]
        return {
            "text": getattr(r, "text", ""),
            "language": str(getattr(r, "language", self.language)),
            "avg_logprob": None,
            "segments": [],
        }


# ---------------------------------------------------------------------------
# 팩토리 (test-way.md 3.10 / 4.3 비교 매트릭스의 "행" 단위)
# ---------------------------------------------------------------------------

ENGINE_REGISTRY = {
    "mlx-whisper": MLXWhisperAdapter,
    "openai-whisper": OpenAIWhisperAdapter,
    "sensevoice": SenseVoiceAdapter,
    "qwen3-asr": Qwen3ASRAdapter,
}


def create_adapter(engine: str, model_id: str = None, language: str = "ko",
                   **options) -> ASRAdapter:
    """엔진 이름 → 어댑터 인스턴스. 미등록/미설치 엔진은 명확히 실패합니다."""
    # 1) 등록되지 않은 엔진명은 바로 오류.
    if engine not in ENGINE_REGISTRY:
        raise ValueError(
            f"미등록 엔진 '{engine}'. 등록됨: {sorted(ENGINE_REGISTRY)}")
    cls = ENGINE_REGISTRY[engine]
    # 2) 필수 패키지가 설치돼 있지 않으면(가드) 사용 불가를 명확히 안내.
    if not cls.importable():
        raise RuntimeError(
            f"'{engine}' 어댑터의 필수 패키지가 없어 사용할 수 없습니다. "
            f"lab/lab-requirements.txt 참고.")
    # 3) 모델/언어/디코딩 옵션을 넣어 어댑터 인스턴스 생성.
    return cls(model_id=model_id, language=language, **options)


def adapter_available(engine: str) -> bool:
    """설치돼 있어 벤치마크에 태울 수 있는 엔진인지."""
    cls = ENGINE_REGISTRY.get(engine)
    return bool(cls and cls.importable())


def make_transcribe_fn(adapter: ASRAdapter):
    """(오디오 경로/배열) -> 텍스트 함수. asr_metrics.evaluate_set과 호환."""
    def fn(audio):
        # 계약 dict에서 텍스트만 꺼내 벤치마크 하네스가 요구하는 형태로 맞춘다.
        return adapter.transcribe(audio)["text"]
    return fn
