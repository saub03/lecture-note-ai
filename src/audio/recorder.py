"""Microphone streaming and audio buffer management.

Produces the 16k mono float32 numpy format consumed by `asr_engine.BaseASR.transcribe`.
`sounddevice` is imported lazily — the buffer can be fed from any source.
"""
from __future__ import annotations

import numpy as np

from .asr_engine import to_16k_mono

SAMPLE_RATE = 16000
CHANNELS = 1


class AudioBuffer:
    """float32 mono 16k 청크 누적 버퍼. push() → numpy 배열로 반환."""

    def __init__(self):
        self._chunks: list[np.ndarray] = []

    def push(self, audio: np.ndarray, sr: int | None = None) -> None:
        """스테레오/임의 샘플레이트 오디오를 16k mono로 정규화해 누적."""
        self._chunks.append(to_16k_mono(audio, sr))

    def __len__(self) -> int:
        return int(sum(len(c) for c in self._chunks))

    @property
    def duration_s(self) -> float:
        return len(self) / SAMPLE_RATE

    def audio(self) -> np.ndarray:
        """누적 청크 전체를 단일 float32 16k mono 배열로 반환."""
        chunks = list(self._chunks)  # 백그라운드 push와의 경쟁 회피 (스냅샷)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def clear(self) -> None:
        self._chunks.clear()


class Recorder:
    """마이크 스트리밍 — sounddevice 기반."""

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.buffer = AudioBuffer()
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=CHANNELS,
            dtype="float32",
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _on_audio(self, indata, frames, time_info, status) -> None:
        self.buffer.push(indata, self.sr)
