"""Live ASR tests — real model + real mic (`pytest -m live`).

Two flavors:
1. File-based: transcribe every wav in `test/audio/samples/` (drop recordings there).
2. Mic-based: record via `Recorder` (sounddevice) and transcribe the buffer.

The lecture summary (`summarized_markdown`) is passed as `initial_prompt`.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

from src.audio.recorder import Recorder

SAMPLES_DIR = Path(__file__).parent / "samples"

pytestmark = pytest.mark.live


def _skip_if_no_sounddevice():
    if importlib.util.find_spec("sounddevice") is None:
        pytest.skip("sounddevice 미설치 — pip install -r requirements.txt")


def _collect_wavs():
    return sorted(SAMPLES_DIR.glob("*.wav")) if SAMPLES_DIR.exists() else []


@pytest.fixture()
def wavs():
    wavs = _collect_wavs()
    if not wavs:
        pytest.skip(f"{SAMPLES_DIR} 에 *.wav 가 없음 — wav를 넣어 주세요")
    return wavs


class TestFileBased:
    def test_transcribe_all_samples(self, engine, wavs, summarized_markdown):
        for path in wavs:
            rec = engine.transcribe(str(path), initial_prompt=summarized_markdown)
            assert rec["engine"]
            assert rec["language"] == "ko"
            assert rec["latency_ms"] >= 0
            print(f"\n[{path.name}] → {rec['text']}")
            print(f"  confidence_ok={rec['confidence_ok']} "
                  f"avg_logprob={rec['avg_logprob']} latency={rec['latency_ms']}ms")
            if rec["text"].strip():
                assert rec["confidence_ok"] is True


class TestMic:
    def test_record_and_transcribe(self, engine, summarized_markdown, capsys):
        _skip_if_no_sounddevice()
        rec = Recorder()
        print("\n🟢 3초간 말하세요...")
        rec.start()
        with capsys.disabled():
            try:
                time.sleep(3.0)
            finally:
                rec.stop()
        audio = rec.buffer.audio()
        print(f"녹음 길이: {rec.buffer.duration_s:.1f}s")
        assert len(audio) > 0

        result = engine.transcribe(audio, initial_prompt=summarized_markdown)
        print(f"\n→ {result['text']}")
        print(f"  confidence_ok={result['confidence_ok']} "
              f"avg_logprob={result['avg_logprob']} latency={result['latency_ms']}ms")
        if result["text"].strip():
            assert result["confidence_ok"] is True


def test_sample_wav_if_present(engine, sample_wav, summarized_markdown):
    if sample_wav is None:
        pytest.skip("lab/my-lab/assets/test_ko.wav 없음")
    rec = engine.transcribe(str(sample_wav), initial_prompt=summarized_markdown)
    print(f"\n[test_ko.wav] → {rec['text']}  (confidence_ok={rec['confidence_ok']})")
