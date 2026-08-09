"""Interactive live ASR — record your voice with the mic and transcribe it.

Usage:
    python test/audio/run_live.py                # mlx-whisper (기본, Apple Silicon)
    ASR_PROVIDER=faster-whisper python test/audio/run_live.py   # CPU int8 교차 검증

Press Enter to start recording, Enter again to stop, then the transcript is shown.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.audio.asr_engine import create_engine
from src.audio.recorder import Recorder

# parser.summarized_markdown 을 대신할 샘플 강의 요약 (initial_prompt)
SUMMARIZED_MARKDOWN = (
    "인공지능과 딥러닝 강의에서 신경망의 역전파 알고리즘과 학습률 최적화 기법을 설명합니다."
)


def main():
    engine = create_engine()
    print(f"🦻 엔진: {engine.name} — 첫 실행은 모델 다운로드 포함")

    recorder = Recorder()
    input("⏺  [Enter] 눌러 녹음 시작... (말하고 싶은 문장을 준비하세요)")
    print("🟢 녹음 중... [Enter] 눌러 정지")
    recorder.start()
    try:
        input()
    finally:
        recorder.stop()

    audio = recorder.buffer.audio()
    dur = recorder.buffer.duration_s
    print(f"⏱  녹음 {dur:.1f}s / {len(audio)} samples\n")

    t0 = time.perf_counter()
    rec = engine.transcribe(audio, initial_prompt=SUMMARIZED_MARKDOWN)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"📝 {rec['text']}")
    print(f"   confidence_ok = {rec['confidence_ok']}")
    print(f"   avg_logprob   = {rec['avg_logprob']}")
    print(f"   latency       = {rec['latency_ms']}ms (wall {elapsed_ms:.0f}ms)")
    print(f"   RTF           = {rec['latency_ms'] / 1000 / dur:.2f}x" if dur > 0 else "")


if __name__ == "__main__":
    main()
