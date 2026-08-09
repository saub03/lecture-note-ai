"""Shared fixtures and live-test skip helpers."""
import importlib.util

import numpy as np
import pytest

from src.audio.asr_engine import create_engine


def _available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.fixture(scope="session")
def engine():
    """실물 ASR 엔진 (config/settings의 ASR_PROVIDER). 첫 호출 시 모델 다운로드 포함."""
    return create_engine()


@pytest.fixture(scope="session")
def synthetic_audio():
    """16k mono float32 합성 음성 (burst 1s + silence 1s) × 2."""
    t = np.arange(16000) / 16000
    burst = (np.sin(2 * np.pi * 300 * t) * 0.5).astype(np.float32)
    sil = np.zeros(16000, dtype=np.float32)
    return np.concatenate([burst, sil, burst, sil])


@pytest.fixture(scope="session")
def sample_wav():
    """my-lab 테스트 한국어 음성 wav (없으면 None)."""
    path = __import__("pathlib").Path(
        __file__).resolve().parents[1] / "lab" / "my-lab" / "assets" / "test_ko.wav"
    return path if path.exists() else None


@pytest.fixture(scope="session")
def summarized_markdown():
    """parser.summarized_markdown을 대신할 샘플 ASR 프롬프트 (강의 요약 한 문장)."""
    return "인공지능과 딥러닝 강의에서 신경망의 역전파 알고리즘과 학습률 최적화 기법을 설명합니다."


def skip_if_no_sounddevice():
    if not _available("sounddevice"):
        pytest.skip("sounddevice 미설치 — pip install -r requirements.txt")
