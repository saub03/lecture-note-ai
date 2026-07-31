# -*- coding: utf-8 -*-
"""ASR 모델 평가 지표 모듈 (lab/ASR-model-test).

이 파일은 `test-way.md`의 2장(핵심 평가 지표)과 `lab/reference/ASR/`의
레퍼런스 노트북 12개 중 지표 계산에 해당하는 코드(2-1, 2-2, 2-3, 2-D, 2-R,
2-S, 2-V, 2-W)를 종합해 만든 평가용 모듈입니다.

테스트 목적은 크게 네 가지로 나뉩니다 (test-way.md 2장):

- 2.1 정확도(Accuracy) : `cer`/`wer`(문자/어절 오류율), `der`(화자 분리 오류율)
- 2.2 속도(Speed)     : `rtf`, `latency_ms`, `streaming_metrics`,
                          `eou_wait_ms`, `budget_verdict`
- 2.3 견고성(Robustness) : `speech_ratio`, `judge_confidence`,
                          `is_hallucination`, `is_no_speech`
- 2.4 배포(Deployment) : `batch_throughput`, `quantization_gap`,
                          `model_load_time_s`, `measure_vram_mb`

그리고 4.1절의 테스트 트랙(클린 / SNR 5dB 노이즈 / 코드 스위칭 / 침묵 /
스트리밍)을 만들고 4.3절의 비교 매트릭스 한 행을 생산하기 위한 도우미
`add_noise`, `measure_snr`, `run_benchmark`, `evaluate_set`도 포함합니다.

사용법 예시:
    # 전사 함수(엔진의 transcribe) 하나만 준비하면 비교 매트릭스 행이 나옵니다.
    result = run_benchmark("faster-whisper", my_transcribe_fn, eval_set)

의존성: numpy (나머지는 표준 라이브러리 re / time / itertools).
jiwer는 설치돼 있으면 교차 검증에만 사용합니다(선택 사항).

모든 함수는 순수 계산 함수라서 모델·GPU 없이 단독으로 테스트할 수 있습니다.
`python3 asr_metrics.py`로 실행하면 `__main__` 블록의 자체 검증(self-test)이
돌아갑니다.
"""

import itertools
import re
import time

import numpy as np


# ---------------------------------------------------------------------------
# 공통 상수 (test-way.md에서 정한 임계값/기준선 모음)
# ---------------------------------------------------------------------------

# 지연 예산: 1.5초 왕복(latency budget) 중 ASR이 차지하는 몫.
# test-way.md 2.2 — 발화 종료 → 최종 전사까지 500ms 이하여야 실시간 서비스가 성립.
ASR_LATENCY_BUDGET_MS = 500.0

# Whisper 고유 신뢰도 신호 avg_logprob의 기준선.
# test-way.md 2.3 — avg_logprob >= -1.0 이면 confidence_ok = True.
LOGPROB_THRESHOLD = -1.0

# "발화 비율" 최소 임계값. 이 값 아래로 텍스트가 나오면 침묵 역설(환각)로 의심.
# test-way.md 3.4 — speech_ratio < 5% 인데 텍스트가 있으면 환각 판정.
MIN_SPEECH_RATIO = 0.05

# 토큰 반복 환각 기준. 같은 토큰이 이 횟수 이상 연속되면 루프 환각으로 판정.
# test-way.md 2.3 — 동일 토큰 >= 4회 연속 → 환각 루프.
REPETITION_RUN = 4

# 정형구 환각 패턴. Whisper 계열 모델이 무음/잡음 구간에서 "아무것도 들었는데"
# 만들어내는 유튜브식·뉴스식 문구들. test-way.md 3.4의 신호 ②.
# 참고(2-3): Cohere, Whisper, Qwen3 모두 침묵에서 이렇게 환각을 냅니다.
HALLUCINATION_PATTERNS = [
    "시청해주셔서 감사합니다",
    "구독",
    "좋아요",
    "MBC 뉴스",
    "자막 제공",
    "다음 영상에서 만나요",
    "Thank you for watching",
]


# ---------------------------------------------------------------------------
# 2.1 정확도 — CER / WER (문자 오류율 / 어절 오류율)
# ---------------------------------------------------------------------------

def normalize_ko(text: str, keep_spaces: bool = False) -> str:
    """한국어 전사 정규화 (2-S / 2-V / 2-W 세션의 공통 기준선).

    평가 전에 원문(ref)과 인식 결과(hyp)를 같은 형태로 맞춰야 합니다.
    그렇지 않으면 "카드 결제가 안 돼요." 와 "카드결제가안돼요"가
    같은 의미인데도 띄어쓰기·문장부호 차이만으로 오류로 집계됩니다.

    하는 일:
    1. 문장부호·특수문자 제거 — 정규식 `[^\\w가-힣\\s]` 를 제거한다.
       (`\\w`는 영숫자와 밑줄(`_`)을 뜻하므로, 밑줄은 다음 줄에서 별도 제거)
    2. 소문자화 — 영어 대문자/소문자 차이(코드 스위칭 트랙)를 무시.
    3. 띄어쓰기 처리:
       - CER(기본): 공백까지 제거 → 음절(문자) 단위 비교.
         한국어는 띄어쓰기가 일관되지 않아 어절 단위보다 음절 단위가 안정적.
       - WER(keep_spaces=True): 띄어쓰기(어절 경계)를 보존하고 연속 공백을
         1개로 압축 → 어절 단위 비교가 가능해진다.

    Args:
        text: 정규화할 전사 문자열.
        keep_spaces: True면 띄어쓰기를 보존(WER용), False면 제거(CER용).
    Returns:
        정규화된 문자열.
    """
    # ① 문장부호/특수문자 제거. `\\w` = [a-zA-Z0-9_], `가-힣` = 한글 음절.
    t = re.sub(r"[^\w가-힣\s]", "", text or "")
    # ② `\\w`가 밑줄(_)을 포함하므로 명시적으로 제거.
    t = t.replace("_", "")
    # ③ 띄어쓰기 처리: WER은 보존+압축, CER은 완전 제거.
    if keep_spaces:
        t = re.sub(r"\s+", " ", t).strip()
    else:
        t = re.sub(r"\s+", "", t)
    # ④ 소문자화 (영어 코드 스위칭 대비).
    return t.lower()


def levenshtein(a, b) -> int:
    """최소 편집 거리(삽입 + 삭제 + 치환 횟수).

    "문자열 a를 문자열 b로 바꾸는 데 필요한 최소 편집 연산 수"를 구하는
    전형적인 동적 프로그래밍(DP) 문제입니다. CER에서는 문자 시퀀스,
    WER에서는 어절(단어) 리스트를 그대로 넣을 수 있어 str/list 모두 동작합니다.

    알고리즘(DP 재귀식):
        dp[i][j] = min(
            dp[i-1][j] + 1,       # a[i] 삭제
            dp[i][j-1] + 1,       # b[j] 삽입
            dp[i-1][j-1] + cost,  # 같은 문자면 유지(0), 다르면 치환(1)
        )
    i는 a의 길이, j는 b의 길이.

    메모리 최적화: 전체 (n+1) x (m+1) 표를 만들지 않고, 바로 윗행 `prev`
    한 줄만 유지합니다. 어떤 셀도 직전 행/열만 참조하기 때문에
    한 줄 DP로 충분합니다(2-W에서 쓴 방식).

    Args:
        a: 시퀀스(문자열 또는 리스트).
        b: 시퀀스(문자열 또는 리스트).
    Returns:
        최소 편집 거리(정수).
    """
    # 편의상 항상 a가 더 긴 쪽이 되도록 스왑 (행 수를 줄여 반복량 최소화).
    if len(a) < len(b):
        a, b = b, a
    # 0행 초기화: a가 ""일 때 b를 만들려면 b의 길이만큼 삽입이 필요.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):           # i = a의 인덱스(1부터)
        cur = [i]                            # 0열: b가 ""일 때 a[i]개 삭제 필요.
        for j, cb in enumerate(b, 1):        # j = b의 인덱스(1부터)
            # 삭제 / 삽입 / (유지 또는 치환) 중 최솟값 선택.
            # `ca != cb`는 True(1)/False(0)로 정수처럼 계산된다.
            cur.append(min(prev[j] + 1,      # 삭제
                           cur[j - 1] + 1,   # 삽입
                           prev[j - 1] + (ca != cb)))  # 유지(0)/치환(1)
        prev = cur                            # 한 줄 이동.
    return prev[-1]                           # 마지막 셀이 답.


def cer(ref: str, hyp: str) -> float:
    """문자 오류율(CER, Character Error Rate).

    공식 (test-way.md 2.1):  CER = 편집 거리 / 정답 문자 수(N)
        = (치환 + 삭제 + 삽입) / N

    한국어에서 WER 대신 CER을 쓰는 이유: 한국어는 띄어쓰기가 일관되지 않아
    음절(문자) 단위 CER이 더 안정적입니다.

    정규화 정책(test-way.md 2.1): 공백·구두점을 제거한 뒤 비교하며,
    ITN(Inverse Text Normalization) 규약은 정답(GT)과 동일하게 맞춥니다.

    엣지 케이스: 정답이 빈 문자열이면 — 인식 결과도 비어 있으면 0.0(완벽),
    텍스트가 있으면 1.0(100% 오류). 0으로 나누기를 방지하기 위함.

    Args:
        ref: 정답 텍스트(ground truth).
        hyp: 인식 결과 텍스트(hypothesis).
    Returns:
        CER (0.0 ~ 1.0, 1.0 = 100% 오류).
    """
    # 공백·구두점을 제거한 "정규화된" 문자열로 비교.
    ref_n, hyp_n = normalize_ko(ref), normalize_ko(hyp)
    if not ref_n:
        # 0으로 나누기 방지: 정답이 없으면 정의상 빈 결과만 정답.
        return 0.0 if not hyp_n else 1.0
    return levenshtein(ref_n, hyp_n) / len(ref_n)


def wer(ref: str, hyp: str) -> float:
    """어절 오류율(WER, Word Error Rate).

    CER과 동일한 편집 거리 기반이지만, 단위가 "문자"가 아니라 "어절(띄어쓰기
    단위의 단어)"입니다. `normalize_ko(..., keep_spaces=True)`로 띄어쓰기를
    보존한 뒤 공백으로 분리해 토큰 리스트를 만들고, 그 위에 편집 거리를 잽니다.

    코드 스위칭 트랙(한국어 + 영어 기술 용어)이나 영어 비중이 높은 데이터를
    평가할 때 유용합니다. 한국어 강의에는 주로 CER을 기본 지표로 사용합니다.

    Args:
        ref: 정답 텍스트.
        hyp: 인식 결과 텍스트.
    Returns:
        WER (0.0 ~ 1.0).
    """
    # keep_spaces=True로 띄어쓰기(어절 경계)를 보존한 뒤 토큰화.
    ref_w = normalize_ko(ref, keep_spaces=True).split()
    hyp_w = normalize_ko(hyp, keep_spaces=True).split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return levenshtein(ref_w, hyp_w) / len(ref_w)


def cer_batch(refs, hyps) -> list:
    """여러 (정답, 인식) 쌍의 CER을 한 번에 계산.

    Args:
        refs: 정답 텍스트 리스트.
        hyps: 인식 결과 텍스트 리스트.
    Returns:
        항목별 CER 리스트. 두 리스트는 같은 길이여야 합니다.
    """
    return [cer(r, h) for r, h in zip(refs, hyps)]


# ---------------------------------------------------------------------------
# 2.1 정확도 — DER (화자 분리 오류율, Diarization Error Rate)
# ---------------------------------------------------------------------------

# 비발화(non-speech) 프레임 라벨. 화자 ID는 0, 1, 2, ... 를 쓰고,
# 음성이 없는 프레임은 항상 -1(NOSPK)로 표시합니다. (2-D 세션 규약)
NOSPK = -1


def labels_from_segments(segments, total_ms: int, frame_ms: int = 10) -> np.ndarray:
    """(시작ms, 끝ms, 화자ID) 세그먼트들을 프레임 단위 라벨 배열로 변환.

    DER은 "프레임별"로 계산하므로, 먼저 시간 구간 목록을 10ms 단위의
    라벨 배열로 펼칩니다. 전체 시간을 frame_ms(기본 10ms)로 나눈 만큼의
    프레임을 만들고, 각 세그먼트가 덮는 프레임 구간에 해당 화자 ID를 채웁니다.

    Args:
        segments: (start_ms, end_ms, speaker_id) 튜플의 리스트.
        total_ms: 전체 오디오 길이(ms). 프레임 수 결정에 사용.
        frame_ms: 프레임 해상도(ms). DER 평가 해상도(2-D: 10ms).
    Returns:
        길이 = ceil(total_ms / frame_ms)인 int 배열.
        음성 없는 구간은 NOSPK(-1), 음성 구간은 화자 ID.
    """
    n = int(np.ceil(total_ms / frame_ms))
    labels = np.full(n, NOSPK, dtype=int)
    for start, end, spk in segments:
        # ms → 프레임 인덱스로 변환 후 슬라이스에 화자 ID 할당.
        labels[int(round(start / frame_ms)):int(round(end / frame_ms))] = spk
    return labels


def der(ref_segments, hyp_segments, total_ms: int, frame_ms: int = 10) -> dict:
    """화자 분리 오류율(DER) — 정답 구간 vs 예측 구간의 프레임 비교.

    공식 (test-way.md 2.1):
        DER = (누락 Miss + 오탐 False Alarm + 혼동 Confusion) / 정답 발화 수

    각 항목은 파이프라인 어느 단계가 잘못됐는지 가리킵니다:
        Miss(누락)      → VAD가 너무 보수적(음성을 놓침)
        False Alarm(오탐) → VAD가 너무 민감(침묵을 음성으로 오인)
        Confusion(혼동)  → 임베딩/클러스터링 실패(화자를 잘못 구분)

    화자 ID는 "최적 매핑"으로 정렬됩니다. 예를 들어 정답이 [화자0, 화자1]인데
    모델이 [화자1, 화자0]으로 뒤집어서 출력했다면, 이는 실제로는 완벽한 결과이므로
    화자 번호를 순열(permutation)로 모두 시도해 가장 낮은 Confusion을 채택합니다.
    따라서 화자 번호만 뒤집힌 결과는 DER = 0.0이 됩니다.

    Args:
        ref_segments: 정답 세그먼트 리스트 [(start_ms, end_ms, speaker_id), ...].
        hyp_segments: 모델(파이프라인)이 낸 세그먼트 리스트(같은 형식).
        total_ms: 전체 오디오 길이(ms).
        frame_ms: 프레임 해상도(ms). 기본 10ms.
    Returns:
        dict — der(총괄), miss, false_alarm, confusion (각각 0.0~1.0),
        mapping(최적 화자 매핑 {hyp_id: ref_id}).
    """
    # ① 세그먼트 → 프레임 라벨 배열로 변환.
    ref = labels_from_segments(ref_segments, total_ms, frame_ms)
    hyp = labels_from_segments(hyp_segments, total_ms, frame_ms)
    ref_speech, hyp_speech = ref != NOSPK, hyp != NOSPK
    n_ref = int(ref_speech.sum())  # 분모: 정답 발화 프레임 수.

    # 엣지 케이스: 정답에 발화가 하나도 없다면 분모가 0 → 0으로 나누기 방지.
    # 정의상 오류 없음으로 처리한다.
    if n_ref == 0:
        return {"der": 0.0, "miss": 0.0, "false_alarm": 0.0,
                "confusion": 0.0, "mapping": {}}

    # ② Miss와 False Alarm은 VAD 레벨에서 바로 계산 가능.
    miss = int((ref_speech & ~hyp_speech).sum())   # 정답 발화인데 예측은 비발화.
    false_alarm = int((~ref_speech & hyp_speech).sum())  # 정답 비발화인데 예측은 발화.

    # ③ Confusion: 발화라고 둘 다 인정한 프레임에서 화자 라벨이 틀린 개수.
    #    최적 매핑을 찾기 위해 화자 ID 순열을 모두 시도한다.
    ref_ids = sorted(set(ref[ref_speech].tolist()))   # 정답에 등장한 화자들.
    hyp_ids = sorted(set(hyp[hyp_speech].tolist()))   # 예측에 등장한 화자들.
    both = ref_speech & hyp_speech
    # 예측 화자가 정답보다 많으면 None(무시 대상)으로 채워 순열 크기를 맞춘다.
    pad = max(0, len(hyp_ids) - len(ref_ids))
    best_conf, best_map = None, {}
    for perm in set(itertools.permutations(ref_ids + [None] * pad, len(hyp_ids))):
        mapping = dict(zip(hyp_ids, perm))   # 예측 화자 → 정답 화자 매핑.
        # 매핑을 적용했을 때 라벨이 어긋난 프레임 수(confusion 후보).
        conf = sum(1 for i in np.where(both)[0]
                   if mapping.get(int(hyp[i])) != ref[i])
        if best_conf is None or conf < best_conf:
            best_conf, best_map = conf, mapping

    # ④ 각 오류 유형을 정답 발화 프레임 수로 나눠 비율로 표준화.
    return {
        "der": round((miss + false_alarm + best_conf) / n_ref, 4),
        "miss": round(miss / n_ref, 4),
        "false_alarm": round(false_alarm / n_ref, 4),
        "confusion": round(best_conf / n_ref, 4),
        "mapping": best_map,
    }


# ---------------------------------------------------------------------------
# 2.2 속도 지표 (test-way.md 2.2)
# ---------------------------------------------------------------------------

def rtf(processing_time_s: float, audio_duration_s: float) -> float:
    """실시간 계수(RTF, Real-Time Factor) = 처리 시간 / 오디오 길이.

    RTF < 1.0 이면 "오디오 재생 속도보다 처리 속도가 빠르다"는 뜻이라
    스트리밍(실시간) 처리 가능 여부의 기준이 됩니다. test-way.md 2.2의 예산은
    < 1.0 입니다.

    Args:
        processing_time_s: 전사 처리에 걸린 시간(초).
        audio_duration_s: 오디오 길이(초).
    Returns:
        RTF. 오디오 길이가 0이면 inf(정의 불가).
    """
    if audio_duration_s <= 0:
        return float("inf")
    return processing_time_s / audio_duration_s


def latency_ms(t_start_s: float, t_end_s: float) -> float:
    """구간 소요 시간을 밀리초로 환산.

    test-way.md 2.2의 Latency 정의인 "발화 종료 → 최종 전사" 지연 측정에
    사용합니다. 예산은 ASR 몫 500ms.

    Args:
        t_start_s: 시작 시각(초).
        t_end_s: 종료 시각(초).
    Returns:
        (t_end - t_start) * 1000, 단위 ms.
    """
    return (t_end_s - t_start_s) * 1000.0


def budget_verdict(value_ms: float, budget_ms: float = ASR_LATENCY_BUDGET_MS) -> str:
    """지연 예산 통과 여부 판정 (test-way.md 4.3).

    T4 환경에서 발화당 500ms 예산을 지켰는지 확인합니다.

    Args:
        value_ms: 측정된 지연(ms).
        budget_ms: 예산(ms). 기본 500.
    Returns:
        "pass" (예산 내) 또는 "exceed" (초과).
    """
    return "pass" if value_ms <= budget_ms else "exceed"


def eou_wait_ms(silence_chunks: int, chunk_ms: float = 320.0) -> float:
    """발화 종료(End-of-Utterance) 대기 시간.

    스트리밍 파이프라인은 "침묵이 몇 청크 연속 이어지면 발화가 끝났다고
    판정"하는 방식으로 발화 경계를 찾습니다. 그 대기 시간은
    `침묵 청크 수 x 청크 길이` 입니다 (test-way.md 3.2).

    기준값: 침묵 3청크 x 320ms = 960ms (프로덕션 표준).
    취향: 300ms(공격적, 응답 빠름) ~ 1200ms(안전, 잘 끊기지 않음).
    이 값은 최종 지연의 주범이므로(fixed latency floor) 신중히 정합니다.

    Args:
        silence_chunks: 발화 종료 판정에 필요한 연속 침묵 청크 수.
        chunk_ms: 청크 길이(ms). 기본 320ms.
    Returns:
        대기 시간(ms).
    """
    return silence_chunks * chunk_ms


def streaming_metrics(events: list, speech_ms: float, wall_s: float) -> dict:
    """스트리밍 지연 요약 — 시뮬레이션 이벤트 스트림에서 계산 (2-R).

    스트리밍 전사는 오디오가 끝나기 전부터 partial(중간) 결과를 내보내고,
    발화가 끝나면 final(최종) 결과를 확정합니다. 이 함수는 그 이벤트들에서
    체감 지연 지표를 뽑아냅니다 (test-way.md 2.2):

        first_partial_ms : 스트림 시작 → 첫 partial 가설 (체감 반응성)
        final_latency_ms : 마지막 final이 나온 시각 - 발화 종료 시각
                           즉 "발화가 끝난 뒤 최종 결과까지 걸린 시간"
        rtf              : 전체 벽시계 시간 / 실제 발화 시간

    Args:
        events: {"event": "partial"|"final", "t_ms": ...} 딕셔너리 리스트.
                t_ms는 "오디오 시작을 0으로 한" 벽시계 시각(ms)입니다.
        speech_ms: 실제 발화(음성) 길이(ms).
        wall_s: 전체 스트리밍에 걸린 벽시계 시간(초).
    Returns:
        dict — first_partial_ms / final_latency_ms / rtf.
        partial 또는 final 이벤트가 없으면 해당 항목은 None.
    """
    partials = [e for e in events if e.get("event") == "partial"]
    finals = [e for e in events if e.get("event") == "final"]
    # 첫 partial까지의 시각이 체감 반응성.
    first_partial_ms = partials[0]["t_ms"] if partials else None
    # 마지막 final이 나온 시각에서 발화 종료 시각을 뺀 값 = 최종 지연.
    final_latency_ms = finals[-1]["t_ms"] - speech_ms if finals else None
    # 벽시계 시간 / 발화 시간 = 실시간 계수.
    rtf_value = wall_s / (speech_ms / 1000.0) if speech_ms > 0 else float("inf")
    return {"first_partial_ms": first_partial_ms,
            "final_latency_ms": final_latency_ms,
            "rtf": rtf_value}


# ---------------------------------------------------------------------------
# 2.3 견고성 — 신뢰도 & 환각 탐지 (test-way.md 2.3, 3.4)
# ---------------------------------------------------------------------------

def speech_ratio(audio: np.ndarray, sr: int, frame_ms: int = 30,
                 energy_thresh: float = 0.02) -> float:
    """발화 비율 = RMS 임계값을 넘는 프레임의 비율.

    오디오를 frame_ms(기본 30ms) 프레임으로 나누고, 각 프레임의 RMS(제곱 평균
    제곱근)가 energy_thresh(기본 0.02)를 넘으면 "말하는 프레임"으로 셉니다.
    전체 프레임 중 말하는 프레임의 비율을 반환합니다 (2-3 세션).

    쓰임새 (환각 필터의 입력): speech_ratio가 거의 0인데(== 거의 침묵) 텍스트가
    돌아왔다면 모델이 침묵에서 환각을 만든 것 (test-way.md 2.3의 Speech ratio).

    Args:
        audio: 1차원 오디오 배열(float32 권장).
        sr: 샘플링 레이트(Hz).
        frame_ms: 프레임 길이(ms).
        energy_thresh: 발화 판정 RMS 임계값.
    Returns:
        0.0 ~ 1.0. 프레임이 없으면 0.0.
    """
    frame_len = int(sr * frame_ms / 1000)   # 프레임당 샘플 수.
    n_frames = len(audio) // frame_len      # 전체 프레임 수.
    if n_frames == 0:
        return 0.0
    # 프레임별 RMS 계산이 편하도록 float64로 올려 계산(정밀도 확보).
    # 주의: 큰 오디오면 복사 비용이 있지만, 평가용 유틸이라 가독성을 택함.
    audio = audio.astype(np.float64)
    speaking = 0
    for i in range(n_frames):
        frame = audio[i * frame_len:(i + 1) * frame_len]
        if float(np.sqrt(np.mean(frame ** 2))) > energy_thresh:
            speaking += 1
    return speaking / n_frames


def judge_confidence(avg_logprob, text: str,
                     threshold: float = LOGPROB_THRESHOLD) -> bool:
    """전사 결과를 신뢰할 수 있는지 판정 (2-3 세션).

    엔진마다 신뢰도 신호가 다르므로 정책이 그 차이를 흡수해야 합니다
    (어댑터 패턴의 핵심 — test-way.md 3.10).

    판정 규칙:
      1. 빈 텍스트 → False (결과가 없으면 신뢰 불가).
      2. avg_logprob이 None(API 엔진: Cohere/Qwen3/Omnilingual은 logprob을
         안 줌) → 텍스트가 존재하기만 하면 True (대리 신호).
      3. avg_logprob이 임계값 미만 → False. Whisper 기준 -1.0 (test-way.md 2.3).

    Args:
        avg_logprob: Whisper 세그먼트 평균 로그 확률. None 가능.
        text: 전사 텍스트.
        threshold: 신뢰도 기준선. 기본 -1.0.
    Returns:
        신뢰 가능 여부(bool).
    """
    if not text or not text.strip():
        return False
    if avg_logprob is None:
        return True
    return avg_logprob >= threshold


def avg_logprob_from_segments(segments) -> float:
    """faster-whisper 세그먼트들에서 평균 avg_logprob 집계.

    faster-whisper는 발화 단위로 avg_logprob(각 토큰 로그 확률의 평균)를
    주므로, 한 발화의 신뢰도를 얻으려면 세그먼트들을 평균해야 합니다.
    세그먼트가 dict든 객체든 모두 받을 수 있습니다.

    Args:
        segments: 세그먼트 리스트. 각 항목은 avg_logprob 속성을 가진 객체
                  또는 {"avg_logprob": float} 형태의 dict.
    Returns:
        세그먼트 avg_logprob의 평균. 값이 하나도 없으면 None.
    """
    lps = []
    for s in segments:
        # dict면 .get, 객체면 getattr 로 값을 꺼낸다.
        lp = s.get("avg_logprob") if isinstance(s, dict) else getattr(s, "avg_logprob", None)
        if lp is not None:
            lps.append(lp)
    return float(np.mean(lps)) if lps else None


def is_no_speech(no_speech_prob: float, threshold: float = 0.9) -> bool:
    """Whisper의 no_speech_prob 신호로 침묵(무음) 여부 판정.

    Whisper는 입력이 침묵일 가능성(0~1)을 no_speech_prob로 줍니다.
    값이 높으면 "무음 → 환각 위험" 신호로 사용합니다 (test-way.md 2.3).
    VAD 게이트가 ASR보다 먼저 침묵을 차단하는 것이 본질적으로 더 안전하지만
    (test-way.md 3.2), 이 함수는 그 게이트의 2차 확인용으로 씁니다.

    Args:
        no_speech_prob: Whisper no_speech 확률(0~1). None이면 0으로 취급.
        threshold: 무음 판정 임계값. 기본 0.9.
    Returns:
        무음 판정 여부(bool).
    """
    return (no_speech_prob or 0.0) >= threshold


def is_hallucination(text: str, audio_duration_s: float, speech_ratio_val: float,
                     patterns: list = HALLUCINATION_PATTERNS,
                     min_speech_ratio: float = MIN_SPEECH_RATIO,
                     repetition_run: int = REPETITION_RUN) -> bool:
    """3가지 신호로 환각(hallucination) 여부 판정 (2-3 / test-way.md 3.4).

    모든 ASR 엔진(Cohere, Whisper, Qwen3)이 침묵에서 환각을 만드는 것이
    확인되어 있어, VAD 게이트에 더해 텍스트 레벨 필터가 필요합니다.
    다음 세 신호 중 하나라도 걸리면 환각으로 판정합니다:

      신호 ① 침묵 역설: speech_ratio가 거의 0(침묵)인데 텍스트가 있다.
              → min_speech_ratio(기본 0.05) 미만이면 즉시 환각.
      신호 ② 정형구 패턴: 유튜브식 문구("구독과 좋아요", "자막 제공" 등),
              필러 반복이 텍스트에 포함됨 → HALLUCINATION_PATTERNS 매칭.
      신호 ③ 토큰 반복: 같은 어절이 4회 이상 연속 반복(루프 환각)
              → 텍스트를 공백 분리해 연속 동일 토큰의 런(run) 길이를 셈.

    Args:
        text: 전사 텍스트.
        audio_duration_s: 오디오 길이(초). 파이프라인 호출 계약에 포함된 인자로,
                          현재 규칙은 길이를 직접 쓰지 않지만 향후
                          "짧은 길이 + 고신뢰도" 같은 규칙을 위해 보관합니다.
                          (레퍼런스 KoreanASR.recognize 와 시그니처 일치)
        speech_ratio_val: speech_ratio() 결과 (0.0~1.0).
        patterns: 정형구 패턴 목록. 기본 HALLUCINATION_PATTERNS.
        min_speech_ratio: 침묵 역설 판정 기준. 기본 0.05.
        repetition_run: 연속 반복 판정 기준. 기본 4.
    Returns:
        환각 여부(bool). 빈 텍스트는 항상 False(환각 아님).
    """
    t = (text or "").strip()
    if not t:
        return False

    # 신호 ① 침묵 역설: 말 비율이 기준 미만인데 텍스트가 존재 → 환각.
    if speech_ratio_val < min_speech_ratio:
        return True

    # 신호 ② 정형구 패턴.
    for p in patterns:
        if p in t:
            return True

    # 신호 ③ 토큰 반복: 같은 어절이 repetition_run회 이상 연속.
    words = t.split()
    run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1  # 연속이면 +1, 끊기면 1로 리셋.
        if run >= repetition_run:
            return True
    return False


def hallucination_rate(texts, speech_ratios, durations=None) -> float:
    """환각률 = 환각 판정된 발화의 비율 (침묵 트랙용 지표).

    test-way.md 4.1의 침묵 트랙(순수 침묵 0dBFS 입력)에서는 "텍스트가 조금이라도
    나오면 모두 오탐(환각)"입니다. 따라서 이 지표는 VAD 게이트가 얼마나 잘
    침묵을 차단하는지 검증하는 척도입니다. 목표: 오탐 텍스트 허용치 0.

    Args:
        texts: 발화별 전사 텍스트 리스트.
        speech_ratios: 발화별 speech_ratio() 결과 리스트.
        durations: 발화별 오디오 길이(초) 리스트(선택). 없으면 0으로 대체.
    Returns:
        0.0 ~ 1.0. 리스트가 비어 있으면 0.0.
    """
    n = len(texts)
    if n == 0:
        return 0.0
    # 길이 불일치는 조용히 데이터를 버리는(zip이 잘라내는) 것보다
    # 명확히 오류로 드러내는 편이 안전하다.
    if len(speech_ratios) != n or (durations is not None and len(durations) != n):
        raise ValueError("texts / speech_ratios / durations의 길이가 일치해야 합니다.")
    durs = durations or [0.0] * n
    flagged = sum(1 for t, d, r in zip(texts, durs, speech_ratios)
                  if is_hallucination(t, d, r))
    return flagged / n


# ---------------------------------------------------------------------------
# 2.4 배포 지표 (test-way.md 2.4)
# ---------------------------------------------------------------------------

def batch_throughput(num_utterances: int, total_time_s: float) -> float:
    """배치 처리량 = 발화 수 / 총 처리 시간 (발화/초).

    야간 배치 QA(전체 녹취록 재전사)는 처리량이 중요하고, 실시간 단건 처리는
    지연이 중요합니다. test-way.md 2.4 — 배치 처리량(utterances/s).

    Args:
        num_utterances: 처리한 발화 수.
        total_time_s: 총 소요 시간(초).
    Returns:
        발화/초. 시간이 0이면 inf.
    """
    if total_time_s <= 0:
        return float("inf")
    return num_utterances / total_time_s


def quantization_gap(fp16_cer: float, int8_cer: float) -> dict:
    """양자화로 인한 CER 차이 — "측정하라, 가정하지 마라" (test-way.md 3.6).

    int8/float16 혼합 양자화는 VRAM을 절반(~1.6GB)으로 줄이고 속도를 올리지만,
    CER이 얼마나 나빠지는지는 반드시 직접 재야 합니다. VRAM ≤ 6GB 경로를
    검증할 때 사용합니다.

    Args:
        fp16_cer: float16 전사 CER.
        int8_cer: int8 전사 CER.
    Returns:
        dict — absolute(절대 차이 = int8 - fp16), relative(상대 비율).
        fp16 CER이 0이면 relative는 inf.
    """
    gap = int8_cer - fp16_cer
    relative = gap / fp16_cer if fp16_cer else float("inf")
    return {"absolute": gap, "relative": relative}


def model_load_time_s(load_fn) -> float:
    """모델 로드(콜드 스타트) 시간 측정(초).

    사용자가 앱을 켰을 때 첫 반응까지 걸리는 시간 = 콜드 스타트 UX
    (test-way.md 2.4 — 모델 로드 시간).

    Args:
        load_fn: 모델을 로드하는 호출 가능 객체(인자 없음).
    Returns:
        로드에 걸린 시간(초).
    """
    t0 = time.perf_counter()
    load_fn()
    return time.perf_counter() - t0


def measure_vram_mb() -> float:
    """현재 프로세스가 점유한 GPU VRAM(MB) 측정.

    T4 Colab의 VRAM 상한이 16GB이므로, 선택한 모델+양자화 조합이 예산 안에
    들어오는지 확인할 때 씁니다 (test-way.md 2.4 — VRAM).

    torch가 설치돼 있고 CUDA를 사용할 수 있을 때만 측정하며,
    그 외 환경(CPU, torch 미설치)에서는 None을 반환합니다.
    참고: 이 값은 호출 시점의 할당량이므로, 모델 로드 직후에 불러야 의미가 있습니다.

    Returns:
        VRAM 사용량(MB, float). 측정 불가 환경이면 None.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated()) / (1024 ** 2)
    except Exception:
        pass  # torch 미설치/CUDA 불가 등 — None 반환.
    return None


# ---------------------------------------------------------------------------
# 노이즈 주입 — 클린 / SNR 5dB 평가 트랙 제작용 (test-way.md 4.1)
# ---------------------------------------------------------------------------

def add_noise(clean: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """목표 SNR(dB)이 되도록 백색소음을 섞는다 (2-W 세션 방식).

    노이즈 스트레스 테스트(test-way.md 3.1)에서 클린 오디오에 특정 SNR의
    잡음을 합성합니다. 기본 SNR 5dB는 "콜센터 험지"를 재현한 조건입니다.

    수식 배경:
        SNR(dB) = 10 * log10(P_signal / P_noise)
        → 목표 노이즈 파워 P_noise = P_signal / 10^(snr_db / 10)
    따라서 (1) 클린 신호의 파워를 재고, (2) 노이즈를 그 파워로 스케일링해
    목표 SNR을 만족시킵니다. `seed`로 노이즈를 고정해 재현성을 보장합니다.

    Args:
        clean: 클린 오디오 배열(float32 권장).
        snr_db: 목표 SNR(dB).
        seed: 노이즈 생성 시드(재현성용, 기본 42).
    Returns:
        노이즈가 섞인 float32 배열. 클리핑 방지를 위해 피크가 1.0을 넘으면
        정규화합니다.
    """
    clean = np.asarray(clean, dtype=np.float32)
    # 엣지 케이스: 빈 입력 → 평균 계산이 NaN이 되므로 즉시 반환.
    if clean.size == 0:
        return clean.copy()

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(clean)).astype(np.float32)  # 표준정규 백색소음.

    p_signal = float(np.mean(clean.astype(np.float64) ** 2))  # 신호 파워.
    p_noise = float(np.mean(noise.astype(np.float64) ** 2))   # 원본 노이즈 파워.

    # 목표 노이즈 파워로 스케일링. (10^(snr_db/10): dB→선형 배율 변환)
    target_p_noise = p_signal / (10 ** (snr_db / 10))
    noise = (noise * np.sqrt(target_p_noise / p_noise)).astype(np.float32)

    noisy = clean + noise
    # 클리핑 방지: 피크가 1.0을 넘으면 전체를 피크로 나눠 정규화.
    peak = float(np.max(np.abs(noisy)))
    if peak > 1.0:
        noisy = noisy / peak
    return noisy.astype(np.float32)


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """합성 결과의 "실측" SNR(dB)을 계산해 검증한다 (2-W).

    add_noise()가 의도한 SNR을 실제로 만들었는지 교차 검증하는 용도입니다.
    (클리핑 정규화 때문에 목표치와 미세한 차이가 날 수 있습니다.)

    수식: SNR(dB) = 10 * log10(P_signal / P_noise), P_noise = mean((noisy-clean)^2)

    Args:
        clean: 원본 클린 오디오.
        noisy: 노이즈가 섞인 오디오.
    Returns:
        실측 SNR(dB). 잡음이 전혀 없으면 inf.
    """
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noisy, dtype=np.float64) - clean
    p_signal = float(np.mean(clean ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise == 0:
        return float("inf")   # 0으로 나누기 방지: 완전 동일 → 잡음 없음.
    return float(10 * np.log10(p_signal / p_noise))


# ---------------------------------------------------------------------------
# 벤치마크 하네스 — 비교 매트릭스(4.3)의 한 행을 만드는 심판 함수
# ---------------------------------------------------------------------------

def evaluate_set(transcribe_fn, eval_set: list) -> list:
    """평가 세트의 모든 항목을 전사하고 항목별 지표를 계산 (2-2).

    "어떤 엔진이든 transcribe_fn 하나만 맞추면 심사대에 올린다"는
    어댑터 패턴(3.10)의 발상이 벤치마크에도 그대로 적용됩니다.

    Args:
        transcribe_fn: (오디오 경로 또는 배열) -> (인식 텍스트) 함수.
        eval_set: 항목 리스트. 각 항목은
                  (path_or_audio, reference_text, condition, duration_s)
                  튜플입니다.
                  condition 예: "clean" / "noisy" / "code-switched" / "silence".
                  침묵 트랙은 reference_text를 ""로 주면 됩니다.
    Returns:
        항목별 딕셔너리 리스트 — condition, ref, hyp, cer, latency_ms, rtf.
    """
    rows = []
    for path, ref, cond, dur in eval_set:
        t0 = time.perf_counter()
        hyp = transcribe_fn(path)          # 엔진별 전사 (text 반환 가정).
        elapsed = time.perf_counter() - t0
        rows.append({
            "condition": cond,
            "ref": ref,
            "hyp": hyp,
            "cer": cer(ref, hyp),
            "latency_ms": elapsed * 1000.0,
            "rtf": rtf(elapsed, dur),
        })
    return rows


def run_benchmark(engine_name: str, transcribe_fn, eval_set: list,
                  conditions: list = None, verbose: bool = True) -> dict:
    """평가 세트 전체를 돌려 조건별 CER + 평균 지연/RTF를 집계 (2-2).

    결과는 test-way.md 4.3 비교 매트릭스의 한 행 형태입니다:
    engine / cer_<condition> / latency_avg_ms / rtf_avg.

    Args:
        engine_name: 엔진 이름(표 출력용).
        transcribe_fn: (오디오 경로 또는 배열) -> (인식 텍스트) 함수.
        eval_set: evaluate_set()과 같은 형식의 항목 리스트.
        conditions: 집계할 조건 리스트. None이면 eval_set에서 자동 추출.
        verbose: True면 콘솔에 요약 출력.
    Returns:
        dict — engine, n(항목 수), 조건별 cer_<cond>, latency_avg_ms, rtf_avg.
        조건에 해당하는 항목이 없으면 해당 CER은 NaN.
    """
    rows = evaluate_set(transcribe_fn, eval_set)
    # 엣지 케이스: 항목이 하나도 없으면 평균이 NaN이 되므로 명시적으로 거부.
    if not rows:
        raise ValueError("eval_set이 비어 있습니다 — 최소 1개 항목이 필요합니다.")

    conditions = conditions or sorted({r["condition"] for r in rows})
    result = {"engine": engine_name, "n": len(rows)}
    # 조건별 CER 평균 (클린/노이즈 등 트랙별 기본 정확도).
    for cond in conditions:
        vals = [r["cer"] for r in rows if r["condition"] == cond]
        result[f"cer_{cond}"] = float(np.mean(vals)) if vals else float("nan")

    # 전체 평균 지연(ms)과 RTF.
    result["latency_avg_ms"] = float(np.mean([r["latency_ms"] for r in rows]))
    result["rtf_avg"] = float(np.mean([r["rtf"] for r in rows]))

    if verbose:
        summary = " | ".join(
            f"{cond} CER {result[f'cer_{cond}']:.1%}" for cond in conditions)
        print(f"[{engine_name}] {summary} | "
              f"평균 지연 {result['latency_avg_ms']:.0f}ms | "
              f"RTF {result['rtf_avg']:.2f}")
    return result


# ---------------------------------------------------------------------------
# 자체 검증(self-test) — 모델·GPU 없이 오프라인으로 실행되는 테스트
# ---------------------------------------------------------------------------
# 레퍼런스 노트북의 assert 블록을 그대로 옮겨 온 것입니다.
# `python3 asr_metrics.py`로 실행하면 아래 모든 검증이 통과해야 합니다.

if __name__ == "__main__":
    # --- 2.1 CER 검증 (2-1 / 2-S 케이스) ---
    assert cer("안녕하세요", "안녕하세요") == 0.0                          # 완벽 일치
    assert abs(cer("안녕하세요", "안녕하세유") - 0.2) < 1e-9              # 치환 1/5
    assert cer("카드 결제가 안 돼요.", "카드결제가 안돼요") == 0.0         # 공백/부호 무시
    assert cer("배송 조회", "배숭 조회") == 0.25                          # 치환 1/4
    assert cer("배송 조회", "배송 조회요") == 0.25                        # 삽입 1/4
    assert cer("배송 조회", "배송") == 0.5                                # 삭제 2/4
    assert cer("", "") == 0.0 and cer("", "x") == 1.0                    # 빈 정답 엣지

    # --- 2.1 WER 검증 ---
    assert wer("배송 조회 부탁해요", "배송 조회 부탁해") == 1 / 3          # 어절 1개 삭제
    assert wer("배송 조회 부탁해요", "배송 조회 부탁해요") == 0.0

    # --- 2.1 DER 검증 (2-D): 화자 번호가 뒤집혀도 DER = 0이어야 함 ---
    _r = [(0, 1000, 0), (1500, 2500, 1)]
    assert der(_r, [(0, 1000, 1), (1500, 2500, 0)], 3000)["der"] == 0.0
    assert der(_r, [(0, 1000, 0)], 3000)["miss"] > 0                     # 둘째 발화 누락

    # --- 2.3 신뢰도 정책 검증 (2-3) ---
    assert judge_confidence(-0.3, "안녕하세요") is True                   # 임계값 이상
    assert judge_confidence(-1.2, "안녕하세요") is False                  # 임계값 미만
    assert judge_confidence(None, "안녕하세요") is True                   # API 엔진 대리 신호
    assert judge_confidence(-0.3, "") is False                           # 빈 텍스트

    # --- 2.3 환각 필터 검증 (2-3) ---
    assert is_hallucination("시청해주셔서 감사합니다", 3.0, 0.5) is True   # 정형구
    assert is_hallucination("네 네 네 네 확인했습니다", 3.0, 0.5) is True  # 토큰 반복 4회
    assert is_hallucination("배송이 어디까지 왔나요", 3.0, 0.01) is True   # 침묵 역설
    assert is_hallucination("배송이 어디까지 왔나요", 3.0, 0.5) is False
    assert is_hallucination("", 3.0, 0.0) is False                       # 빈 텍스트는 아님

    # --- 2.3 발화 비율 + 환각률 검증 (2-3): 1초 음성 + 1초 침묵 x 2 합성 스트림 ---
    _sr = 16000
    _t = np.arange(_sr) / _sr
    _burst = (np.sin(2 * np.pi * 300 * _t) * 0.5).astype(np.float32)   # 1초 톤
    _sil = np.zeros(_sr, dtype=np.float32)                              # 1초 침묵
    _stream = np.concatenate([_burst, _sil, _burst, _sil])              # 50% 발화
    assert abs(speech_ratio(_stream, _sr) - 0.5) < 0.05
    # 침묵 트랙: 텍스트 있는 1건은 환각, 빈 1건은 정상 → 환각률 50%.
    assert abs(hallucination_rate(["텍스트", ""], [0.0, 0.0]) - 0.5) < 1e-9

    # --- 2.2 스트리밍 지연 분해 검증 (2-R) ---
    _events = [{"event": "partial", "t_ms": 1200},
               {"event": "partial", "t_ms": 1800},
               {"event": "final", "t_ms": 2600}]
    _m = streaming_metrics(_events, speech_ms=1600, wall_s=3.0)
    assert _m["first_partial_ms"] == 1200       # 첫 partial 시각.
    assert _m["final_latency_ms"] == 1000       # final(2600) - 발화 종료(1600).
    assert abs(_m["rtf"] - 3.0 / 1.6) < 1e-9
    assert eou_wait_ms(3, 320.0) == 960.0       # 프로덕션 표준 3x320ms.
    assert budget_verdict(480.0) == "pass" and budget_verdict(520.0) == "exceed"

    # --- 2.2 / 2.4 속도·배포 헬퍼 검증 ---
    assert rtf(0.5, 2.0) == 0.25
    assert batch_throughput(10, 2.0) == 5.0
    assert quantization_gap(0.05, 0.08)["absolute"] == 0.03

    # --- 노이즈 합성 교차 검증 (2-W): 실측 SNR ≈ 목표 SNR ---
    _sig = (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, _sr))).astype(np.float32)
    assert abs(measure_snr(_sig, add_noise(_sig, 5.0)) - 5.0) < 0.3

    # --- jiwer 설치 시 CER 교차 검증 (2-1, 선택 사항) ---
    try:
        import jiwer
        assert abs(cer("배송 조회", "배숭 조회")
                   - jiwer.cer(normalize_ko("배송 조회"), normalize_ko("배숭 조회"))) < 1e-9
        print("jiwer 교차 검증 ✅")
    except ImportError:
        print("jiwer 미설치 — 교차 검증 건너뜀")

    print("asr_metrics.py 자체 검증 통과 ✅")
