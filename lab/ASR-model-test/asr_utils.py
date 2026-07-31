# -*- coding: utf-8 -*-
"""ASR 모델 테스트 유틸리티 모듈 (lab/ASR-model-test).

`asr_test.ipynb`가 사용하는 "부수 작업"을 모아 두는 모듈입니다.
노트북을 간결하고 읽기 쉽게 유지하기 위해 아래 기능을 담당합니다 (test-way.md 참조):

- `env_check_mac`      : Apple Silicon 실험 환경(파이썬/MLX/MPS/ASR 엔진/패키지) 점검
- 오디오 I/O           : `load_audio` / `save_wav` / `audio_duration_s` /
                          `synth_utterance`(gTTS) / `make_silence` / `make_quiet_noise`
- 평가 세트 생성       : `DATASET_V1` / `DATASET_V2` + `build_eval_set` +
                          `summarize_eval_set` — test-way.md 4.1의 테스트 트랙
                          (clean / code_switched / noisy / silence / streaming) 제작
- 스트리밍 시뮬레이션  : `simulate_stream` — 성장 버퍼 재전사 방식 (test-way.md 3.8)
- 결과 저장            : `save_results_json`

모든 오디오는 **16kHz 모노 float32** 표준을 따릅니다 (test-way.md 3.1).
"""

import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

# 평가용 오디오의 공통 샘플링 레이트 (모든 ASR 엔진의 표준 입력).
SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# 환경 체크
# ---------------------------------------------------------------------------

def env_check_mac():
    checks = []

    # 1. 파이썬 버전 체크 (3.10+)
    checks.append(
        ("파이썬 3.10+", sys.version_info >= (3, 10), sys.version.split()[0])
    )

    # 2. PyTorch MPS (Metal Performance Shaders) 가속 체크
    try:
        import torch

        mps_ok = torch.backends.mps.is_available()
        gpu_detail = "Apple Silicon (MPS)" if mps_ok else "CPU 모드"
        checks.append(("PyTorch (MPS)", mps_ok, gpu_detail))
    except Exception as e:
        checks.append(("PyTorch (MPS)", False, f"미설치 또는 오류: {e}"))

    # 3. Apple MLX 프레임워크 체크
    mlx_spec = importlib.util.find_spec("mlx")
    checks.append(("Apple MLX", mlx_spec is not None, "Apple native ML"))

    # 4. ASR 엔진 체크 (어댑터 등록 여부와 연결)
    engines = [
        ("mlx-whisper", "mlx_whisper", "Apple Silicon 네이티브 Whisper"),
        ("openai-whisper", "whisper", "torch MPS 기준선"),
        ("sensevoice", "funasr", "SenseVoice-Small (CTC)"),
        ("qwen3-asr", "qwen_asr", "Qwen3-ASR-0.6B (LLM)"),
    ]
    for name, mod, detail in engines:
        checks.append(
            (f"엔진 {name}", importlib.util.find_spec(mod) is not None, detail)
        )

    # 5. 평가/오디오 패키지 체크
    packages = [
        "gtts",
        "librosa",
        "soundfile",
        "jiwer",
        "numpy",
        "matplotlib",
        "koreanize_matplotlib",
        "seaborn",
        "pandas"
    ]
    for pkg in packages:
        checks.append(
            (f"패키지 {pkg}", importlib.util.find_spec(pkg) is not None, "")
        )

    # 결과 출력
    print("=" * 56)
    for name, ok, detail in checks:
        print(f" {'✅' if ok else '❌'} {name:<20} {detail}")

    n = sum(1 for _, ok, _ in checks if ok)
    print("-" * 56)
    print(
        f" 통과 {n}/{len(checks)}",
        "→ 🎉 이륙 준비 완료!"
        if n == len(checks)
        else "→ ❌ 미설치 항목 조치 후 재실행",
    )


# ---------------------------------------------------------------------------
# 오디오 I/O — 16kHz 모노 float32 표준 (test-way.md 3.1)
# ---------------------------------------------------------------------------

def load_audio(path, sr=SAMPLE_RATE) -> np.ndarray:
    """오디오 파일 → 16kHz 모노 float32 배열 (librosa)."""
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def save_wav(audio, path, sr=SAMPLE_RATE) -> None:
    """float32 배열을 wav로 저장 (soundfile)."""
    import soundfile as sf
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)


def audio_duration_s(path) -> float:
    """오디오 파일의 길이(초)를 반환 (librosa, RTF/지연 지표 계산용)."""
    import librosa
    return float(librosa.get_duration(path=str(path)))


def synth_utterance(text, out_wav, lang="ko", sr=SAMPLE_RATE) -> np.ndarray:
    """gTTS로 문장 합성 → 16kHz wav 저장, 웨이브폼 반환.

    gTTS mp3 → librosa 로드 → wav 저장의 표준 패턴입니다.
    네트워크 필요(Google TTS).
    """
    from gtts import gTTS
    import tempfile
    fd, mp3 = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        gTTS(text=text, lang=lang).save(mp3)
        y = load_audio(mp3, sr=sr)
    finally:
        os.unlink(mp3)
    save_wav(y, out_wav, sr=sr)
    return y


def make_silence(dur_s, sr=SAMPLE_RATE) -> np.ndarray:
    """순수 디지털 침묵 (test-way.md 4.1 침묵 트랙, 0dBFS)."""
    return np.zeros(int(sr * dur_s), dtype=np.float32)


def make_quiet_noise(dur_s, amp=0.001, seed=0, sr=SAMPLE_RATE) -> np.ndarray:
    """아주 작은 백색소음(≈-60dBFS). 순수 침묵과 달리 Whisper 계열이
    환각을 내는 것으로 알려진 "조용하지만 완전히 0은 아닌" 신호입니다."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(sr * dur_s)) * amp).astype(np.float32)


# ---------------------------------------------------------------------------
# 평가 세트 생성 (test-way.md 4.1 테스트 트랙)
# ---------------------------------------------------------------------------
# 트랙 별 발화: clean / code_switched는 TTS, noisy는 clean에 SNR 5dB 주입,
# silence는 순수 침묵 + 미세 소음. 평가 세트 항목 형식은 asr_metrics와 동일:
#   (path_or_audio, reference_text, condition, duration_s)

SILENCE_DUR_S = 4.0  # 침묵 트랙 발화 길이(초)

# 평가용 발화 모음. v1/v2는 서로 다른 문장으로, "데이터셋을 바꿔도 같은
# 결론이 나오는지"(재현성 + 일반화)를 검증하기 위한 두 세트입니다.
#   - clean / code_switched : {발화 ID: gTTS로 합성할 한국어(또는 혼용) 문장}
#   - silence               : {발화 ID: "zeros"(순수 침묵) | "quiet"(미세 소음)}
DATASET_V1 = {
    "clean": {
        "utt_001": "오늘 강의에서는 데이터베이스 트랜잭션의 네 가지 성질에 대해 다룹니다",
        "utt_002": "트랜잭션은 원자성 일관성 격리성 그리고 지속성을 보장해야 합니다",
        "utt_003": "합성곱 신경망에서 풀링 레이어는 특징 맵의 크기를 줄이는 역할을 합니다",
        "utt_004": "정렬 알고리즘의 시간 복잡도는 입력 크기에 따라 결정됩니다",
        "utt_005": "버스 어레이 접근 시간이 중요한 이유는 캐시 지역성 때문입니다",
    },
    "code_switched": {
        "utt_cs_001": "payment API의 response를 확인해보면 timeout이 발생했습니다",
        "utt_cs_002": "이 모델은 fine-tuning 하기 전에 tokenizer를 먼저 설정해야 합니다",
        "utt_cs_003": "데이터가 sparse하면 embedding layer에서 memory가 낭비됩니다",
        "utt_cs_004": "loss function의 gradient가 explode하지 않도록 gradient clipping을 적용합니다",
    },
    "silence": {
        "utt_sil_001": "zeros",
        "utt_sil_002": "zeros",
        "utt_sil_003": "quiet",
        "utt_sil_004": "quiet",
    },
}

DATASET_V2 = {
    "clean": {
        "utt_101": "자연어 처리에서 단어는 고정 길이 벡터로 임베딩하여 표현합니다",
        "utt_102": "그래프 순회는 너비 우선 탐색과 깊이 우선 탐색으로 나뉩니다",
        "utt_103": "운영체제는 프로세스 스케줄링을 통해 CPU 자원을 효율적으로 배분합니다",
        "utt_104": "암호화폐 거래소는 콜드 월렛에 대부분의 자산을 보관합니다",
        "utt_105": "병렬 컴퓨팅에서 스레드 간 동기화는 경쟁 상태를 막는 핵심입니다",
    },
    "code_switched": {
        "utt_cs_101": "cloud storage에 저장된 backup은 async 방식으로 복제됩니다",
        "utt_cs_102": "이 query는 index를 활용하면 execution time이 크게 줄어듭니다",
        "utt_cs_103": "fraud detection model의 recall과 precision은 trade-off 관계입니다",
        "utt_cs_104": "docker container는 isolation을 보장하면서도 overhead가 낮습니다",
    },
    "silence": {
        "utt_sil_101": "zeros",
        "utt_sil_102": "zeros",
        "utt_sil_103": "quiet",
        "utt_sil_104": "quiet",
    },
}

DATASETS = {"v1": DATASET_V1, "v2": DATASET_V2}


def build_eval_set(tracks, out_dir, snr_db=5.0, seed=42, silence_dur_s=SILENCE_DUR_S):
    """트랙 정의 → 평가 세트 항목 리스트 생성.

    Args:
        tracks: DATASET_V1과 같은 dict. {condition: {utt_id: text_or_kind}}
                clean/code_switched는 TTS 문장, silence는 "zeros"/"quiet".
        out_dir: 생성된 wav를 저장할 디렉터리 (data/ 하위 권장, gitignore 대상).
        snr_db: noisy 트랙의 목표 SNR(dB). 기본 5 (test-way.md 4.1).
        seed: 노이즈 시드 (재현성).
    Returns:
        [(path, ref_text, condition, duration_s), ...] — asr_metrics.evaluate_set 호환.
    """
    from asr_metrics import add_noise

    # 저장 디렉터리 준비 (data/ 아래 권장 → gitignore 대상, 실험 산출물 아님)
    os.makedirs(out_dir, exist_ok=True)
    # 최종 반환값: asr_metrics.evaluate_set()이 그대로 받아 쓰는 항목 목록.
    eval_set = []

    # (1) clean / code_switched / silence 트랙 생성
    for condition, items in tracks.items():
        if condition == "silence":
            # 침묵 트랙: "zeros"(순수 침묵) 또는 "quiet"(미세 소음) wav 생성.
            for utt_id, kind in items.items():
                if kind == "quiet":
                    y = make_quiet_noise(silence_dur_s, seed=int(utt_id[-3:]))
                else:
                    y = make_silence(silence_dur_s)
                wav = os.path.join(out_dir, f"{utt_id}.wav")
                save_wav(y, wav)
                # 침묵 트랙의 정답 텍스트는 "" — 아무 텍스트나 나오면 환각으로 간주.
                eval_set.append((wav, "", condition, float(silence_dur_s)))
        else:
            # 일반 트랙: 문장을 gTTS(한국어)로 합성해 wav로 저장.
            for utt_id, text in items.items():
                wav = os.path.join(out_dir, f"{utt_id}.wav")
                synth_utterance(text, wav)
                eval_set.append(
                    (wav, text, condition, audio_duration_s(wav)))

    # (2) noisy 트랙: clean 트랙의 wav에 SNR 5dB 백색소음을 주입해 생성
    #     (test-way.md 3.1 — 노이즈 스트레스 테스트, SNR 5dB = 콜센터 험지 수준).
    clean = tracks.get("clean") or {}
    for utt_id, text in clean.items():
        clean_wav = os.path.join(out_dir, f"{utt_id}.wav")
        y = load_audio(clean_wav)
        y_noisy = add_noise(y, snr_db, seed=seed)
        noisy_wav = os.path.join(out_dir, f"{utt_id}_noisy{int(snr_db)}.wav")
        save_wav(y_noisy, noisy_wav)
        eval_set.append((noisy_wav, text, "noisy", len(y_noisy) / SAMPLE_RATE))

    return eval_set


def summarize_eval_set(eval_set) -> None:
    """생성된 평가 세트 요약 출력 (트랙별 항목 수, 길이 범위)."""
    from collections import defaultdict
    # 트랙(condition)별로 항목을 모아 길이를 집계한다.
    by_cond = defaultdict(list)
    for _, ref, cond, dur in eval_set:
        by_cond[cond].append((dur, ref))
    for cond, rows in sorted(by_cond.items()):
        durs = [d for d, _ in rows]
        print(f"  {cond:<14} {len(rows)}건  길이 "
              f"{min(durs):.1f}s ~ {max(durs):.1f}s")


# ---------------------------------------------------------------------------
# 스트리밍 시뮬레이션 (test-way.md 3.8 / 4.1 스트리밍 트랙)
# ---------------------------------------------------------------------------

def simulate_stream(audio, adapter, chunk_s=1.0, eou_silence_chunks=2,
                    sr=SAMPLE_RATE, min_speech_ratio=0.05):
    """청크를 늘려가며 재전사하는 "성장 버퍼" 방식의 스트리밍 시뮬레이션.

    오디오를 chunk_s 간격으로 잘라 매번 지금까지의 버퍼를 전사합니다(재전사 방식).
    침묵 청크가 eou_silence_chunks 연속 이어지면 final 이벤트를 내보냅니다.

    Args:
        audio: 전체 발화 웨이브폼(float32, 16kHz).
        adapter: ASRAdapter 인스턴스.
        chunk_s: 청크 간격(초). 기본 1.0.
        eou_silence_chunks: EoU 판정용 연속 침묵 청크 수. 기본 2 (≈2s).
        sr: 샘플링 레이트.
        min_speech_ratio: 발화 판정용 speech_ratio 임계값.
    Returns:
        (events, speech_ms, wall_s) — asr_metrics.streaming_metrics() 입력 호환.
    """
    from asr_metrics import speech_ratio

    n = len(audio)
    chunk_len = int(chunk_s * sr)  # 청크당 샘플 수
    events = []
    t0 = time.perf_counter()
    silence_run = 0   # 연속 침묵 청크 수 (발화 종료 판정용)
    started = False   # 발화가 아직 진행 중인지
    last_text = ""    # 직전 전사 결과 (final 이벤트에 재사용)
    speech_ms = 0.0   # 마지막 발화 구간의 종료 시각(ms)

    pos = 0
    while pos < n:
        # (1) 다음 청크 경계까지 "성장 버퍼" 확장 후 전체를 재전사
        pos = min(pos + chunk_len, n)
        t_ms = pos / sr * 1000.0
        chunk = audio[:pos]                    # 지금까지 쌓인 오디오
        result = adapter.transcribe(chunk)     # 버퍼 전체 재전사 (test-way.md 3.8)
        text = result["text"]
        ratio = speech_ratio(chunk, sr)        # RMS 기반 발화 비율

        if ratio >= min_speech_ratio and text:
            # 발화 중 → 중간 결과(partial) 이벤트 발생
            started = True
            silence_run = 0
            speech_ms = t_ms
            events.append({"event": "partial", "t_ms": t_ms, "text": text})
        else:
            # 침묵 청크 → 연속 침묵이 임계값을 넘으면 발화 종료(final)
            silence_run += 1
            if started and silence_run >= eou_silence_chunks:
                events.append({"event": "final", "t_ms": t_ms, "text": last_text})
                started = False
                silence_run = 0
        last_text = text

    # (2) 오디오가 끝났는데도 미종료 상태면 마지막 시점에 final 확정
    if started:
        events.append({"event": "final", "t_ms": n / sr * 1000.0, "text": last_text})
    wall_s = time.perf_counter() - t0  # 전체 벽시계 시간(초)
    return events, speech_ms, wall_s


# ---------------------------------------------------------------------------
# 결과 저장 헬퍼
# ---------------------------------------------------------------------------

def save_results_json(data, path) -> None:
    """결과 dict를 JSON 파일로 저장 (UTF-8, 한글 그대로 유지)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

