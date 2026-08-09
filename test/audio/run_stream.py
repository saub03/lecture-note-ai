"""Realtime streaming ASR — transcribe the mic incrementally while you speak.

The mic buffer keeps growing in a background thread (Recorder); every `--step`
seconds the last `--window` seconds are re-transcribed and the newly stabilized
text is printed as it appears. `[⟲]` marks a re-recognition of the overlap.

After `--duration`, the FULL buffer is transcribed once more — that batch result
is the authoritative final transcript (the streaming pass is best-effort: a
mid-sentence window boundary can drop or revise text).

Usage:
    python test/audio/run_stream.py                       # 60s · window 8 · step 2
    python test/audio/run_stream.py --window 4 --step 2   # 짧은 청크 — 경계 절단 관찰
    python test/audio/run_stream.py --window 15 --step 3 --duration 90  # 긴 청크 — 지연 증가
    ASR_PROVIDER=faster-whisper python test/audio/run_stream.py   # CPU int8 교차 검증

Press Enter to start; the script stops itself after `--duration` seconds.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.audio.asr_engine import create_engine
from src.audio.recorder import SAMPLE_RATE, Recorder

# parser.summarized_markdown 을 대신할 샘플 강의 요약 (initial_prompt)
SUMMARIZED_MARKDOWN = (
    "인공지능과 딥러닝 강의에서 신경망의 역전파 알고리즘과 학습률 최적화 기법을 설명합니다."
)


def delta_from_tail(tail: str, new_text: str) -> str | None:
    """이전 window 텍스트(tail)와 단어 단위 prefix 비교.
    tail이 new_text의 prefix면 새로 확인된 델타만, 아니면(재인식) None."""
    tw, nw = tail.split(), new_text.split()
    i = 0
    while i < min(len(tw), len(nw)) and tw[i] == nw[i]:
        i += 1
    if i < len(tw):
        return None  # 이전에 출력한 부분이 재인식됨
    return " ".join(nw[i:])


def main():
    ap = argparse.ArgumentParser(description="실시간 스트리밍 ASR (슬라이딩 윈도우)")
    ap.add_argument("--duration", type=float, default=60.0, help="녹음 길이 (초)")
    ap.add_argument("--window", type=float, default=8.0,
                    help="슬라이딩 윈도우 크기 (초) — 재전사할 마지막 구간")
    ap.add_argument("--step", type=float, default=2.0,
                    help="재전사 간격 (초) — window 이하 권장")
    args = ap.parse_args()
    if args.step > args.window:
        sys.exit("--step 은 --window 보다 클 수 없습니다")

    engine = create_engine()
    print(f"🦻 엔진: {engine.name} — 첫 실행은 모델 다운로드 포함")
    print(f"⚙  duration={args.duration:g}s window={args.window:g}s step={args.step:g}s")

    recorder = Recorder()
    input("⏺  [Enter] 눌러 녹음 시작... (말하세요)")
    recorder.start()

    tail = ""
    revisions = 0
    stats = []
    t_start = time.perf_counter()
    next_pass = t_start + args.step

    try:
        while time.perf_counter() - t_start < args.duration:
            while time.perf_counter() < next_pass:
                time.sleep(0.05)
            next_pass += args.step
            elapsed = time.perf_counter() - t_start

            window = recorder.buffer.audio()[-int(args.window * SAMPLE_RATE):]
            wdur = len(window) / SAMPLE_RATE
            t0 = time.perf_counter()
            rec = engine.transcribe(window, sr=16000, initial_prompt=SUMMARIZED_MARKDOWN)
            decode_ms = (time.perf_counter() - t0) * 1000
            stats.append((elapsed, wdur, decode_ms))

            text = rec["text"]
            delta = delta_from_tail(tail, text)
            if delta is None:
                revisions += 1
                print(f"\n  [⟲ 수정] {text}", flush=True)
            elif delta:
                print(delta, end="", flush=True)
            tail = text

            sys.stderr.write(
                f"\r[{elapsed:5.1f}s/{args.duration:g}s · window {wdur:4.1f}s · "
                f"decode {decode_ms:6.0f}ms · RTF {decode_ms / 1000 / max(wdur, 1e-6):.2f}] ")
            sys.stderr.flush()
    finally:
        recorder.stop()

    t0 = time.perf_counter()
    final = engine.transcribe(recorder.buffer.audio(), sr=16000,
                              initial_prompt=SUMMARIZED_MARKDOWN,
                              filter_hallucination=False)
    final_ms = (time.perf_counter() - t0) * 1000

    final_text = final["text"]
    if not final_text.strip():
        print("\n  ⚠ 최종 일괄 전사가 비어 있음 — 스트리밍에서 확정된 텍스트로 대체합니다.")
        final_text = tail

    print("\n")
    print("─" * 60)
    print(f"📝 최종 전문 (일괄 전사 — 전체 {recorder.buffer.duration_s:.1f}s, {final_ms:.0f}ms)")
    print(final_text)
    print("─" * 60)
    decode_avg = sum(s[2] for s in stats) / max(len(stats), 1)
    rtf = [s[2] / 1000 / max(s[1], 1e-6) for s in stats]
    print(f"pass {len(stats)} 회 | decode 평균 {decode_avg:.0f}ms "
          f"최대 {max(s[2] for s in stats):.0f}ms")
    print(f"RTF 평균 {sum(rtf) / max(len(rtf), 1):.2f} 최대 {max(rtf):.2f} | 재인식 {revisions} 회")


if __name__ == "__main__":
    main()
