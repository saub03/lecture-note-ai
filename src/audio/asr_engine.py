"""Local ASR engine (Faster-Whisper / mlx-whisper).

Adapter pattern from lab/my-lab/01-ASR-review.ipynb (2.8 / 2.11 / 2.12):
- `BaseASR` (ABC) defines a shared `transcribe()` that returns a *standard contract dict*.
  Each engine only implements `_raw_transcribe()`; everything else (preprocess → greedy →
  confidence gate → beam retry → hallucination filter → Korean postprocess) is shared.
- `create_engine()` picks the engine from `config/settings.py`.
- The lecture summary prompt (`parser.summarized_markdown`) is passed as Whisper's
  native `initial_prompt` (pre-transcription probabilistic hint — notebook 2-1/2-3).
"""
from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

# ═══ 2.0 audio normalization (aicc_env.to_16k_mono 규격) ═══


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a or 1


def to_16k_mono(audio: np.ndarray, sr: int | None = None) -> np.ndarray:
    """임의 샘플레이트·채널 오디오 → float32 16kHz 모노.
    audio: (모노 1D 또는 (n, c)) — 리샘플은 scipy.resample_poly, 없으면 선형 보간 폴백."""
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if sr is None or sr == 16000:
        return audio

    try:
        from scipy.signal import resample_poly
        g = 16000 // _gcd(sr, 16000)
        d = sr // _gcd(sr, 16000)
        return resample_poly(audio, g, d).astype(np.float32)
    except Exception:
        n = int(round(len(audio) * 16000 / sr))
        x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
        x_new = np.linspace(0, 1, num=n, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


# ═══ 2.1 데이터 계약 — 모든 엔진이 지키는 최소 형식 ═══

CONTRACT_KEYS = {"engine", "text", "language",
                 "confidence_ok", "avg_logprob", "latency_ms"}


def validate_contract(rec) -> bool:
    """계약 준수 여부: 필수 키만 존재하면 OK. 확장 키는 자유 ('계약은 최소 보장, 확장은 자유')."""
    return isinstance(rec, dict) and set(rec.keys()) >= CONTRACT_KEYS


# ═══ 2.2 신뢰도 판정·재시도 정책 ═══

LOGPROB_THRESHOLD = -0.8


def judge_confidence(avg_logprob, text, threshold=LOGPROB_THRESHOLD):
    """전사 결과를 신뢰할 수 있는가?
    ① 빈 텍스트 → False ② logprob 없음(API형 엔진) → 텍스트 존재만으로 True ③ logprob < 임계값 → False."""
    if not text or not text.strip():
        return False
    if avg_logprob is None:
        return True
    return avg_logprob >= threshold


def should_retry(attempt, confidence_ok, max_retries=1):
    """2단 전략: 평소 greedy(빠름), 신뢰도 의심될 때만 beam=5 재시도(확실한 길)."""
    return (not confidence_ok) and (attempt < max_retries)


# ═══ 2.3 환각 필터 + 한국어 후처리 ═══

HALLUCINATION_PATTERNS = [
    "시청해주셔서 감사합니다", "구독", "좋아요", "MBC 뉴스",
    "자막 제공", "다음 영상에서 만나요", "Thank you for watching",
]


def is_hallucination(text, audio_duration_s, speech_ratio):
    """환각 의심 판정 — 세 신호: ①말 비율 5% 미만인데 텍스트 존재 ②정형 문구 ③같은 어절 4회 연속 반복."""
    t = (text or "").strip()
    if not t:
        return False
    if speech_ratio < 0.05:
        return True
    for p in HALLUCINATION_PATTERNS:
        if p in t:
            return True
    words = t.split()
    run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        if run >= 4:
            return True
    return False


# 강의 도메인 오인식 교정 (원하면 외부에서 dict 주입 — 미니 lexicon)
_DOMAIN_LEXICON = {}
_NUM_UNIT_RE = re.compile(r"(\d[\d,]*)\s+(원|일|개|번|시|분|명|장|페이지)")


def postprocess_ko(text, lexicon=None):
    """도메인 교정 → 숫자-단위 붙여쓰기 → 공백 정리. initial_prompt와 달리 전사 후 결정적 교정."""
    lexicon = lexicon if lexicon is not None else _DOMAIN_LEXICON
    out = text or ""
    for wrong, right in lexicon.items():
        out = out.replace(wrong, right)
    out = _NUM_UNIT_RE.sub(r"\1\2", out)
    return re.sub(r"\s{2,}", " ", out).strip()


# ═══ 2.4 VAD 발화 분리 (환각 필터의 speech_ratio 입력·스트림용) ═══


def split_utterances(audio, sr, frame_ms=30, energy_thresh=0.02, min_silence_s=0.8):
    """연속 오디오 → (시작, 끝) 발화 구간 목록. 에너지 VAD + EoU(침묵 카운터)."""
    frame_len = int(sr * frame_ms / 1000)
    min_silence_frames = int(min_silence_s * 1000 / frame_ms)
    segments, start, silence = [], None, 0
    n_frames = len(audio) // frame_len
    for i in range(n_frames):
        frame = audio[i * frame_len:(i + 1) * frame_len]
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if rms > energy_thresh:
            if start is None:
                start = i * frame_len
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_silence_frames:
                segments.append((start, (i + 1 - silence) * frame_len))
                start, silence = None, 0
    if start is not None:
        segments.append((start, n_frames * frame_len))
    return segments


def speech_ratio(audio, sr, frame_ms=30, energy_thresh=0.02):
    """전체 중 말 프레임 비율 — 환각 필터의 ①번 신호 입력."""
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return 0.0
    speaking = sum(
        1 for i in range(n_frames)
        if float(np.sqrt(np.mean(audio[i * frame_len:(i + 1) * frame_len].astype(np.float64) ** 2)))
        > energy_thresh)
    return speaking / n_frames


# ═══ 2.8 어댑터 — 어떤 엔진이든 표준 계약으로 ═══

AudioLike = Union[str, Path, np.ndarray]


class BaseASR(ABC):
    """표준 계약을 지키는 ASR 어댑터의 공통 틀.
    엔진별로 `_raw_transcribe`만 구현하면 `transcribe`(표준 계약 반환)는 공유한다."""

    name = "base-engine"

    def __init__(self, language: str = "ko"):
        self.language = language

    # ── 엔진별 구현부 ──
    @abstractmethod
    def _raw_transcribe(self, audio: AudioLike, initial_prompt: str | None,
                        beam_size: int) -> tuple[str, float | None]:
        """(텍스트, avg_logprob) 반환 — avg_logprob는 없으면 None."""

    # ── 공통 파이프라인 (2.12: 전처리→전사→신뢰도→재시도→계약) ──
    def transcribe(self, audio_or_path: AudioLike, sr: int | None = None,
                   initial_prompt: str | None = None,
                   retry_beam_size: int = 5, max_retries: int = 1,
                   postprocess: bool = True, lexicon: dict | None = None,
                   filter_hallucination: bool = True) -> dict:
        """표준 계약 반환: {engine, text, language, confidence_ok, avg_logprob, latency_ms}.
        initial_prompt: 강의 요약(parser.summarized_markdown)을 Whisper 네이티브 프롬프트로 전달.
        filter_hallucination=False: 환각 필터를 건너뛰어 원문 그대로 반환 (일괄 전문용)."""
        if isinstance(audio_or_path, (str, Path)):
            path = Path(audio_or_path)
            audio = self._load_audio(path)
            sr = 16000
        else:
            audio = to_16k_mono(audio_or_path, sr)
            sr = 16000

        t0 = time.perf_counter()
        text, avg_logprob = self._raw_transcribe(audio, initial_prompt, beam_size=1)

        for attempt in range(max_retries):
            ok = judge_confidence(avg_logprob, text)
            if not should_retry(attempt, ok, max_retries=max_retries):
                break
            text, avg_logprob = self._raw_transcribe(audio, initial_prompt,
                                                     beam_size=retry_beam_size)

        if filter_hallucination and is_hallucination(text, len(audio) / sr,
                                                     speech_ratio(audio, sr)):
            text = ""
        if postprocess:
            text = postprocess_ko(text, lexicon=lexicon)

        return {
            "engine": self.name,
            "text": text,
            "language": self.language,
            "confidence_ok": judge_confidence(avg_logprob, text),
            "avg_logprob": None if avg_logprob is None else round(float(avg_logprob), 3),
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        }

    @staticmethod
    def _load_audio(path: Path) -> np.ndarray:
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path), dtype="float32")
            return to_16k_mono(audio, sr)
        except Exception:
            import wave
            with wave.open(str(path), "rb") as w:
                sr = w.getframerate()
                raw = w.readframes(w.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            return to_16k_mono(audio, sr)


class MlxWhisperEngine(BaseASR):
    """mlx-whisper (Apple MLX) — M시리즈 통합 메모리 최적화. 생성 시점에만 import+로드.
    beam search 미지원 → greedy 전용. 신뢰도는 mlx-whisper 내장 temperature 폴백에 의존."""

    def __init__(self, model="mlx-community/whisper-large-v3-turbo", language="ko"):
        super().__init__(language=language)
        import mlx_whisper
        self.model = model
        self._mlx_whisper = mlx_whisper
        self.name = f"mlx-whisper:{Path(model).name}"

    def _raw_transcribe(self, audio, initial_prompt, beam_size) -> tuple[str, float | None]:
        del beam_size  # mlx-whisper는 beam search 미지원 — greedy 고정
        try:
            result = self._mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model,
                language=self.language,
                initial_prompt=initial_prompt,
            )
        except Exception as e:
            if type(e).__name__ == "RepositoryNotFoundError":
                raise RuntimeError(
                    f"mlx-whisper 모델 리포를 찾을 수 없습니다: {self.model!r}\n"
                    f"ASR_MODEL 을 HF 리포명으로 지정하세요. 예) ASR_MODEL={MLX_DEFAULT_MODEL}"
                ) from e
            raise
        text = result.get("text", "")
        lps = [s.get("avg_logprob") for s in result.get("segments", []) if s.get("avg_logprob") is not None]
        return text, (float(np.mean(lps)) if lps else None)


class FasterWhisperEngine(BaseASR):
    """faster-whisper (CTranslate2) — CPU int8 macOS 표준. 생성 시점에만 import+로드."""

    def __init__(self, model="large-v3-turbo", device="cpu", compute_type="int8",
                 language="ko"):
        super().__init__(language=language)
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.name = f"faster-whisper:{model}-{compute_type}"

    def _raw_transcribe(self, audio, initial_prompt, beam_size) -> tuple[str, float | None]:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
        )
        segs = list(segments)
        text = "".join(s.text for s in segs)
        lps = [s.avg_logprob for s in segs if s.avg_logprob is not None]
        return text, (float(np.mean(lps)) if lps else None)


class MockEngine(BaseASR):
    """모델 없이 계약 파이프라인을 검증하는 목 엔진."""

    name = "mock"

    def __init__(self, text="배송 조회 도와드리겠습니다", lp=-0.3, language="ko"):
        super().__init__(language=language)
        self._text, self._lp = text, lp

    def _raw_transcribe(self, audio, initial_prompt, beam_size) -> tuple[str, float | None]:
        return self._text, self._lp


# ═══ 2.8 팩토리 — config 한 줄이 엔진을 결정 ═══

MLX_DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
FASTER_DEFAULT_MODEL = "large-v3-turbo"


def create_engine(provider=None, model=None, **kwargs) -> BaseASR:
    """config/settings.ASR_PROVIDER 로 엔진 선택. 교체 지점을 한 곳으로 모은다.
    ASR_MODEL을 안 채우면 엔진별 기본 모델을 쓴다 (faster-whisper: large-v3-turbo,
    mlx-whisper: HF 리포 mlx-community/whisper-large-v3-turbo)."""
    from config import settings

    provider = provider or settings.ASR_PROVIDER
    if provider == "mlx-whisper":
        return MlxWhisperEngine(model=model or settings.ASR_MODEL or MLX_DEFAULT_MODEL,
                                language=kwargs.get("language", settings.ASR_LANGUAGE))
    if provider == "faster-whisper":
        return FasterWhisperEngine(model=model or settings.ASR_MODEL or FASTER_DEFAULT_MODEL,
                                   device=kwargs.get("device", settings.ASR_DEVICE),
                                   compute_type=kwargs.get("compute_type", settings.ASR_COMPUTE_TYPE),
                                   language=kwargs.get("language", settings.ASR_LANGUAGE))
    raise ValueError(f"모르는 ASR 엔진: {provider!r}")


# ═══ 자가 점검 (모델·GPU 불필요) ═══
if __name__ == "__main__":
    t = np.arange(16000) / 16000
    burst = (np.sin(2 * np.pi * 300 * t) * 0.5).astype(np.float32)
    sil = np.zeros(16000, dtype=np.float32)
    stream = np.concatenate([burst, sil, burst, sil])
    assert len(split_utterances(stream, 16000)) == 2
    assert 0.4 < speech_ratio(stream, 16000) < 0.6

    assert is_hallucination("시청해주셔서 감사합니다", 3.0, 0.5) is True
    assert is_hallucination("배송이 어디까지 왔나요", 3.0, 0.5) is False
    assert postprocess_ko("3 일 이내 30,000 원 환불") == "3일 이내 30,000원 환불"

    dummy = np.zeros(16000, dtype=np.float32)

    rec = MockEngine(lp=-1.2).transcribe(dummy)
    assert rec["confidence_ok"] is False
    assert isinstance(rec["avg_logprob"], float)
    assert validate_contract(rec)

    rec_retry = MockEngine(lp=-1.2).transcribe(dummy, retry_beam_size=5)
    assert rec_retry["engine"] == "mock"

    rec_prompt = MockEngine(text="initial_prompt 전달 확인").transcribe(
        stream, initial_prompt="인공지능 강의 요약")
    assert rec_prompt["confidence_ok"] is True
    print("✅ asr_engine 자가 점검 통과 (계약·재시도·환각필터·VAD)")
