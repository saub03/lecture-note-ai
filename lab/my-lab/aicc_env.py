"""
aicc_env.py — my-lab 공용 실행 환경 헬퍼 (Apple Silicon)

* .env 로더 (추가 의존성 없음 — 수동 파싱)
* 엔진 가용성 탐지 (AVAIL dict — importlib.find_spec)
* 모델 설정 (환경변수로 교체 가능)
* 오디오 헬퍼 (16k mono 규격화 · wav 저장/로드 · 타이머)

사용법:
    import sys, os
    sys.path.insert(0, os.getcwd())   # my-lab/ 루트에서 실행 시
    import aicc_env as ae
    ae.AVAIL, ae.KEYS, ae.MODEL_CFG ...
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE / ".env"


# ── .env 로더 (python-dotenv 없이) ──────────────────────────────
def load_env(path: Path = ENV_PATH) -> None:
    """.env 를 읽어 os.environ 에 반영. 이미 있으면 덮어쓰지 않음."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("\"'")
        os.environ.setdefault(key, val)


load_env()

KEYS = {
    "OPENAI": os.getenv("OPENAI_API_KEY", ""),
    "OPENROUTER": os.getenv("OPENROUTER_API_KEY", ""),
    "ELEVENLABS": os.getenv("ELEVENLABS_API_KEY", ""),
}

MODEL_CFG = {
    # ASR — mlx-whisper (Apple Silicon)
    "asr_mlx": os.getenv("AICC_ASR_MLX", "mlx-community/whisper-large-v3-turbo"),
    # ASR — faster-whisper (CPU int8 교차 검증)
    "asr_faster": os.getenv("AICC_ASR_FASTER", "large-v3-turbo"),
    "asr_faster_compute": os.getenv("AICC_ASR_FASTER_COMPUTE", "int8"),
    # LLM — mlx-lm
    "llm_mlx": os.getenv("AICC_LLM_MLX", "mlx-community/Qwen3-4B-Instruct-2507-4bit"),
    "llm_max_tokens": int(os.getenv("AICC_LLM_MAX_TOKENS", "512")),
    # TTS — sherpa-onnx / MeloTTS
    "tts_supertonic_model": os.getenv("AICC_TTS_SUPERTONIC", "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"),
    "tts_kokoro_model": os.getenv("AICC_TTS_KOKORO", "kokoro-multi-lang-v1_0"),
}


# ── 가용성 탐지 ──────────────────────────────────────────────────
def _avail(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


AVAIL = {
    "mlx": _avail("mlx"),
    "mlx_whisper": _avail("mlx_whisper"),
    "faster_whisper": _avail("faster_whisper"),
    "mlx_lm": _avail("mlx_lm"),
    "melotts": _avail("melo"),
    "sherpa_onnx": _avail("sherpa_onnx"),
    "torch": _avail("torch"),
    "fastapi": _avail("fastapi"),
    "uvicorn": _avail("uvicorn"),
    "websockets": _avail("websockets"),
    "httpx": _avail("httpx"),
    "aiortc": _avail("aiortc"),
    "nest_asyncio": _avail("nest_asyncio"),
    "elevenlabs": _avail("elevenlabs"),
    "resemblyzer": _avail("resemblyzer"),
    "openai": _avail("openai"),
    "soundfile": _avail("soundfile"),
}

KEYS_PRESENT = {k: bool(v) for k, v in KEYS.items()}


def has(*names: str) -> bool:
    """여러 항목이 모두 준비됐는지. 없으면 무엇이 빠졌는지 안내 print."""
    missing = [n for n in names if not AVAIL.get(n, False)]
    if missing:
        print("⛔ 필요한 패키지가 없습니다:", ", ".join(missing))
        print("   → setup_apple_silicon.sh 실행 또는 pip install 하세요. (README_실행.md 참고)")
    return not missing


def key_present(name: str) -> bool:
    ok = bool(KEYS.get(name, ""))
    if not ok:
        print(f"⛔ {name}_API_KEY 가 .env 에 없습니다. (my-lab/.env 에 등록 후 커널 재시작)")
    return ok


def engine_report() -> str:
    """가용성 표 출력 — 설정 스크립트·노트 초반에서 확인용."""
    lines = ["[aicc_env] 가용성 리포트"]
    lines.append(f"  .env 경로: {ENV_PATH} (존재: {ENV_PATH.exists()})")
    for name, ok in AVAIL.items():
        lines.append(f"  {'✅' if ok else '— '} {name}")
    lines.append(f"  키 등록: { {k: ('있음' if v else '없음') for k, v in KEYS_PRESENT.items()} }")
    lines.append(f"  모델 설정: {json.dumps({k: v for k, v in MODEL_CFG.items() if isinstance(v, str)}, ensure_ascii=False)}")
    return "\n".join(lines)


# ── 오디오 헬퍼 ──────────────────────────────────────────────────
def to_16k_mono(audio, sr: int):
    """임의 샘플레이트·채널 오디오 → float32 16kHz 모노 (TTS 계약 규격).
    audio: np.ndarray (모노 1D 또는 (n, c)) — scipy/soundfile 이 있으면 리샘플."""
    import numpy as np

    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if sr == 16000:
        return audio

    # scipy.signal.resample_poly — 가용 시 사용, 아니면 선형 보간 폴백
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


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a or 1


def save_wav(path, audio, sr: int) -> Path:
    """float32 모노 (임의 sr) → wav 저장. soundfile 없으면 wave 모듈로 16k PCM16."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = to_16k_mono(audio, sr)
    try:
        import soundfile as sf
        sf.write(path, audio, 16000)
    except Exception:
        import wave
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def load_wav(path) -> tuple:
    """wav → (float32 16k mono, 16000). """
    import numpy as np

    path = Path(path)
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")
        return to_16k_mono(audio, sr), 16000
    except Exception:
        import wave
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            raw = w.readframes(n)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        return to_16k_mono(audio, sr), 16000


def timed(fn, *args, **kw):
    """(출력, 경과 ms) — 계약·계측 노트에서 공용."""
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, (time.perf_counter() - t0) * 1000


# ── 테스트 오디오 자산 ───────────────────────────────────────────
def _supertonic_dir() -> Path:
    """Supertonic-3(31언어, 한국어 포함) 모델 디렉토리 보장 — 1회 다운로드(~130MB)."""
    import urllib.request

    name = MODEL_CFG["tts_supertonic_model"]
    d = HERE / "models" / name
    if (d / "tts.json").exists():
        return d
    models = HERE / "models"
    models.mkdir(exist_ok=True)
    tar = models / (name + ".tar.bz2")
    if not tar.exists():
        url = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
               + name + ".tar.bz2")
        print("Supertonic 모델 다운로드:", url)
        urllib.request.urlretrieve(url, tar)
    import tarfile
    with tarfile.open(tar, "r:bz2") as tf:
        tf.extractall(models)
    tar.unlink(missing_ok=True)
    return d


def synth_korean(text: str):
    """한국어 TTS — (float32 모노, sr) 반환. 우선순위: sherpa-onnx Supertonic(ko, cp314 지원)
    → MeloTTS(KR, Python 3.12/3.13 필요) → None."""
    if AVAIL.get("sherpa_onnx"):
        try:
            import numpy as np
            import sherpa_onnx as so
            d = _supertonic_dir()
            st = so.OfflineTtsSupertonicModelConfig(
                duration_predictor=str(d / "duration_predictor.int8.onnx"),
                text_encoder=str(d / "text_encoder.int8.onnx"),
                vector_estimator=str(d / "vector_estimator.int8.onnx"),
                vocoder=str(d / "vocoder.int8.onnx"),
                tts_json=str(d / "tts.json"),
                unicode_indexer=str(d / "unicode_indexer.bin"),
                voice_style=str(d / "voice.bin"))
            model_cfg = so.OfflineTtsModelConfig(supertonic=st, debug=False,
                                                 num_threads=2, provider="cpu")
            tts = so.OfflineTts(so.OfflineTtsConfig(model=model_cfg))
            g = so.GenerationConfig()
            g.sid, g.speed, g.extra["lang"] = 0, 1.0, "ko"
            out = tts.generate(text, g)
            return np.asarray(out.samples, dtype=np.float32), int(out.sample_rate)
        except Exception as e:
            print("Supertonic(ko) 합성 실패:", type(e).__name__, e)
    if AVAIL.get("melotts"):
        try:
            import numpy as np
            from melo.api import TTS as MeloTTS
            m = MeloTTS(language="KR", device="cpu")
            path = str(HERE / "assets" / "_melo_tmp.wav")
            m.tts_to_file(text, m.hps.data.spk2id["KR"], path)
            return load_wav(path)
        except Exception as e:
            print("MeloTTS 합성 실패:", type(e).__name__, e)
    return None


def ensure_test_audio(texts: list | None = None) -> Path:
    """assets/test_ko.wav 보장. TTS(Supertonic ko/MeloTTS)로 생성하되,
    없으면 오디오 계약과 무관한 ASR 목적용으로 진동 합성 폴백."""
    import numpy as np

    texts = texts or ["안녕하세요, 지난달 요금이 평소보다 많이 나온 것 같아서 확인 부탁드려요.",
                      "인터넷이 어제 저녁부터 자꾸 끊기는데 기사님 방문 예약할 수 있을까요?"]
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    out = assets / "test_ko.wav"
    if out.exists():
        return out

    made = synth_korean(" ".join(texts))
    if made is not None:
        save_wav(out, made[0], made[1])
    else:
        # 폴백: 220Hz 1.5초 진동 (ASR 목업 대본용 — 음소는 무의미)
        sr = 16000
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        save_wav(out, audio, sr)
    return out


if __name__ == "__main__":
    print(engine_report())
