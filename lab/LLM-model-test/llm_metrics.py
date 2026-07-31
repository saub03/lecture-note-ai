# -*- coding: utf-8 -*-
"""LLM 모델 평가 지표 모듈 (lab/LLM-model-test).

이 파일은 `test-way.md`의 2장(핵심 평가 지표)의 기준선·공식을 그대로
구현한 평가용 모듈입니다.

테스트 목적은 크게 네 가지로 나뉩니다 (test-way.md 2장):

- 2.1 지연(Latency) : `analyze_stream`(TTFT/TTFA/TTFS), `throughput_per_s`,
                      `latency_budget_verdict` — 음성 에이전트의 체감 반응성
- 2.2 계약 준수(Contract Compliance) : `strip_fences`, `classify_output`,
                      `compliance_metrics`, `guarded_generate`, `Saturated`
                      (상류 포화 → DNP 결장 집계)
- 2.3 품질(Quality) : `cer`, `hangul_ratio`, `char_mix`, `is_code_switch`,
                      `sanity_check`, `length_compliant`
- 2.4 배포(Deployment) : `vram_fp16_gb`, `vram_nf4_gb`, `fits_on_t4`,
                      `license_ok`, `load_ladder`, `measure_vram_mb`

그리고 3.1절의 프롬프트 계약(`estimate_tokens`, `validate_prompt_contract`,
`truncate_history`)과 4.3절 비교 매트릭스의 한 행을 생산하는
`run_benchmark`도 포함합니다.

사용법 예시:
    # 생성 함수(엔진의 generate) 하나만 준비하면 비교 매트릭스 행이 나옵니다.
    result = run_benchmark("llama-3.2-korean", my_generate_fn,
                           eval_set, INTENT_SCHEMA)

의존성: numpy, jsonschema (jsonschema가 없으면 required 키 존재 여부만 검사하는
최소 검증으로 대체됩니다 — 스키마 타입 검사는 약해집니다).

모든 함수는 순수 계산 함수라서 모델·GPU 없이 단독으로 테스트할 수 있습니다.
`python3 llm_metrics.py`로 실행하면 `__main__` 블록의 자체 검증(self-test)이
돌아갑니다.
"""

import json
import math
import re
import time

import numpy as np


# ---------------------------------------------------------------------------
# 공통 상수 (test-way.md에서 정한 임계값/기준선 모음)
# ---------------------------------------------------------------------------

# 지연 예산: 음성 에이전트의 첫 답변(TTFA) 예산.
# test-way.md 2.1 — reasoning 모델은 TTFA가 첫 토큰(TTFT)보다 크게 늦어
# 음성 루프에서 탈락 사유가 된다.
TTFA_BUDGET_MS = 500.0

# 출력 길이 한도(문자). test-way.md 2.3 — TTS 합성 약 15초에 해당.
MAX_RESPONSE_CHARS = 200

# 코드 스위칭/혼입 판정 기준선. test-way.md 2.3 —
# EXAONE 공식 권고 T=0.1 실측에서 한글 비율 >= 0.984, 판정 기준은 0.98.
MIN_HANGUL_RATIO = 0.98

# T4 16GB VRAM 상한 (test-way.md 2.4).
T4_VRAM_GB = 16.0

# 계약 준수 판정 시 "재시도 1회"가 프로덕션 기본값 (test-way.md 3.2).
MAX_CONTRACT_RETRY = 1

# 문장 경계 정규식. test-way.md 3.6 — `.?!` + 줄임표(...) 기준.
# 한국어 종결어미("요", "다")만으로는 분리하지 않고, 문장부호가 있어야
# 확실한 문장 경계로 본다.
SENT_END = re.compile(r"[.!?…]")

# reasoning 모델의 "답변 시작" 마커. TTFA는 `</think>` 이후 첫 토큰까지를 잰다
# test-way.md 2.1 — TTFA = 요청 → 첫 *답변* 토큰.
ANSWER_MARKER = "</think>"

# 토큰 추정 계수. test-way.md 3.1 —
# 한국어 문자당 토큰 근사치(실측으로 보정할 상수), +4는 역할 특수 토큰 몫.
TOKENS_PER_CHAR = 0.85

# 상용 배포 가능 라이선스 화이트리스트 / 블랙리스트 (test-way.md 4.2).
# MIT ✓ / Apache 2.0 ✓ / llama3.2 ✓ — CC-BY-NC ✗ / qwen-research ✗.
LICENSE_OK = {"mit", "apache", "apache-2.0", "apache 2.0", "apache2.0",
              "llama3.2", "llama 3.2", "llama3.1", "llama 3.1", "llama2"}
LICENSE_NOK = {"cc-by-nc", "cc-by-nc-4.0", "cc-by-nc-sa", "cc-by-nc-nd",
               "nc", "non-commercial", "비상업", "qwen-research"}


# ---------------------------------------------------------------------------
# 2.1 지연 지표 (test-way.md 2.1)
# ---------------------------------------------------------------------------

def first_sentence(buf: str, sent_end=None):
    """버퍼에서 첫 완전한 문장을 잘라낸다.

    문장 경계는 `[.!?…]`(문장부호 + 줄임표)로 판정합니다.
    TTS 핸드오프 게이트(TTFS)와 문장 단위 TTS 큐 투입에 사용합니다.

    Args:
        buf: 생성 중인 텍스트 버퍼.
        sent_end: 문장 종료 정규식(기본 SENT_END). 한국어 종결어미("다.",
                  "요.") 등 커스텀 경계로 교체 가능.
    Returns:
        첫 문장(경계 포함). 아직 경계가 없으면 None.
    """
    m = (sent_end or SENT_END).search(buf)
    return buf[:m.end()].strip() if m else None


def analyze_stream(events: list, answer_marker: str = ANSWER_MARKER) -> dict:
    """스트리밍 토큰 도착 이벤트에서 TTFT/TTFA/TTFS/처리량을 동시 계측.

    `TextIteratorStreamer` + 스레드 루프 안에서
    첫 청크(TTFT), 첫 문장(TTFS), `</think>` 발견(TTFA) 시각을
    `time.monotonic()`으로 기록합니다. 그 계측 로직을 "이미 찍힌
    (시각, 청크)" 이벤트 리스트에 대한 순수 함수로 옮긴 것입니다 —
    모델 없이도 오프라인에서 검증할 수 있습니다.

    Args:
        events: (t_s, chunk_text) 튜플 리스트. t_s는 요청 시작을 0으로 한
                벽시계 시간(초). chunk_text가 빈 문자열이면 무시합니다.
        answer_marker: TTFA 판정용 마커. reasoning 모델 기본 `</think>`.
    Returns:
        dict —
            ttft_ms / ttfs_ms / ttfa_ms : 첫 토큰/첫 문장/첫 답변 시각(ms).
            gap_ms                     : TTFS - TTFT (문장 경계 겹침 예산).
            total_ms                   : 마지막 이벤트 시각(ms).
            n_chunks / n_chars         : 청크/문자 수.
            chunks_per_s               : 지속 생성 속도(청크/초).
            이벤트가 없거나 해당 사건이 없으면 해당 항목은 None.
    """
    ttft = ttfs = ttfa = None
    buf = ""
    n_chunks = n_chars = 0
    for t_s, chunk in events:
        if not chunk:                       # 빈 청크는 "대기 시간"일 뿐 — 무시.
            continue
        if ttft is None:                    # 첫 비어있지 않은 토큰 = TTFT.
            ttft = t_s
        n_chunks += 1
        n_chars += len(chunk)
        buf += chunk
        # TTFA: answer_marker가 버퍼에 나타난 시각 (</think> 검출).
        if ttfa is None and answer_marker in buf:
            ttfa = t_s
        # TTFS: 첫 완전한 문장이 완성된 시각 (first_sentence 검출).
        if ttfs is None and first_sentence(buf):
            ttfs = t_s

    total = events[-1][0] if events else 0.0
    # 지속 처리량: TTFT 이후 생성 구간 기준 (chunks_per_s 공식).
    sustained = max(total - ttft, 1e-6) if ttft is not None else None
    chunks_per_s = (n_chunks / sustained) if sustained is not None else None

    def _ms(v):
        return v * 1000.0 if v is not None else None

    return {
        "ttft_ms": _ms(ttft),
        "ttfs_ms": _ms(ttfs),
        "ttfa_ms": _ms(ttfa),
        "gap_ms": _ms(ttfs - ttft) if (ttfs is not None and ttft is not None) else None,
        "total_ms": _ms(total),
        "n_chunks": n_chunks,
        "n_chars": n_chars,
        "chunks_per_s": chunks_per_s,
    }


def throughput_per_s(n: int, elapsed_s: float, ttft_s: float = 0.0) -> float:
    """지속 생성 처리량 = (토큰/청크 수) / 생성 구간(초).

    TTFT(첫 토큰)은 프리픽스 비용이고, 이후가 "지속 생성" 구간입니다.
    핵심: TTFT보다 지속 처리량이 문장 단위 TTS 합성 개시 시각을 좌우
    하므로 스트리밍 UX에서는 이 값이 TTFT보다 중요합니다 (test-way.md 2.1).

    Args:
        n: 생성된 토큰/청크 수.
        elapsed_s: 총 소요 시간(초).
        ttft_s: 첫 토큰까지 걸린 시간(초). 생성 구간을 빼기 위해 사용.
    Returns:
        토큰/청크 per 초. 생성 구간이 0이면 0.0 (0 나누기 방지).
    """
    gen_s = elapsed_s - ttft_s
    if gen_s <= 0:
        return 0.0
    return n / gen_s


def latency_ms(t_start_s: float, t_end_s: float) -> float:
    """구간 소요 시간을 밀리초로 환산 (test-way.md 2.1 지연 측정 공통 헬퍼)."""
    return (t_end_s - t_start_s) * 1000.0


def e2e_latency_ms(t_start_s: float, t_end_s: float) -> float:
    """E2E 지연 — 입력 → 최종 검증 출력까지 (test-way.md 2.1).

    예산은 발화당 ≤ 2.5s (ASR + LLM + 검증 합산). latency_ms와 동일한
    계산이지만 "전체 파이프라인"을 재는 것임을 이름으로 명확히 합니다.
    """
    return (t_end_s - t_start_s) * 1000.0


def latency_budget_verdict(value_ms: float, budget_ms: float = TTFA_BUDGET_MS) -> str:
    """지연 예산 통과 여부 판정.

    음성 에이전트 기준은 TTFA ≤ 500ms (test-way.md 2.1).
    TTFT가 아니라 TTFA로 판정하는 이유: reasoning 모델은 첫 토큰이
    `<think>` 혼잣말일 뿐, 고객은 첫 *답변* 토큰까지 침묵을 듣습니다.

    Args:
        value_ms: 측정된 지연(ms). TTFT/TTFS/TTFA/E2E 중 무엇이든 가능.
        budget_ms: 예산(ms). 기본 500.
    Returns:
        "pass" (예산 내) 또는 "exceed" (초과).
    """
    return "pass" if value_ms <= budget_ms else "exceed"


# ---------------------------------------------------------------------------
# 2.2 계약 준수 지표 (test-way.md 2.2 / 3.2)
# ---------------------------------------------------------------------------
# "프롬프트는 계약이다. 게이트가 보장한다." — 모델 출력은
# jsonschema 게이트를 통과하기 전까지 "응답"이 아니다.

def strip_fences(raw: str) -> str:
    """마크다운 펜스(```json ... ```)를 제거하고 깨끗한 JSON 텍스트 추출.

    로컬 모델은 JSON 모드를 지원하지 않아 종종 코드블록을 감싸서 내놓습니다.
    처음과 끝의 ``` 만 제거합니다.

    Args:
        raw: 원시 출력 문자열 (None이면 빈 문자열로 취급).
    Returns:
        펜스가 제거되고 양끝 공백이 정리된 문자열.
        펜스가 없으면 공백만 정리해 그대로 반환.
    """
    s = (raw or "").strip()
    if s.startswith("```"):
        # ``` 다음 줄부터가 실제 JSON — 첫 줄(```json 같은 언어 태그)을 버린다.
        # 태그 줄에 newline이 없으면(예: ```json {...}```) 그대로 둔다.
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3]               # 끝의 닫는 ``` 제거.
    return s.strip()


def _schema_validate(rec: dict, schema: dict) -> None:
    """jsonschema로 레코드를 검증한다. 실패 시 예외 발생.

    jsonschema가 없으면 "required 키 존재"만 검사하는 최소 폴백입니다.
    타입·enum·additionalProperties 검사는 jsonschema가 있을 때만 동작하므로,
    평가 환경에는 jsonschema 설치를 권장합니다 (test-way.md 6).
    """
    try:
        import jsonschema
    except ImportError:
        # 최소 폴백: required 키 존재 여부만 검사 (타입 검사는 약함).
        req = set(schema.get("required", []))
        missing = req - set((rec or {}).keys())
        if missing:
            raise ValueError(f"필수 키 누락: {sorted(missing)}")
        return
    jsonschema.validate(rec, schema)


def validate_contract(rec: dict, schema: dict) -> bool:
    """레코드가 계약(스키마)을 지키는지 여부 (jsonschema 게이트).

    실패를 "조용히 삼키는" 대신 명시적으로 bool로 반환해 상위 루프가
    재시도/폴백을 결정하게 합니다 (test-way.md 3.2의 "출력 게이트").

    Args:
        rec: 모델 출력을 파싱한 dict.
        schema: jsonschema dict. None이면 "제약 없음" → 항상 통과.
    Returns:
        스키마 준수 여부(bool).
    """
    # schema가 None이면 "제약 없음" — classify_output(schema=None)이
    # "JSON이기만 하면 ok"로 보는 것과 일관된 의미를 가진다.
    if schema is None:
        return True
    try:
        _schema_validate(rec, schema)
        return True
    except Exception:
        return False


def parse_json_output(raw: str):
    """원시 출력을 펜스 제거 후 JSON으로 파싱한다.

    Args:
        raw: 모델 원시 출력 문자열.
    Returns:
        (obj, "ok") — 파싱 성공.
        (None, "no_json") — JSON이 아님.
    """
    s = strip_fences(raw)
    try:
        return json.loads(s), "ok"
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, "no_json"


def classify_output(raw: str, schema: dict = None) -> str:
    """출력 하나를 세 범주로 분류한다 (test-way.md 2.2).

      "ok"               : JSON이며 스키마 게이트 통과.
      "no_json"          : JSON이 아님 (비-JSON율의 분자).
      "schema_violation" : JSON이지만 스키마 위반 (스키마 위반율의 분자).

    schema가 None이면 "JSON이냐"만 판정한다 (펜스 제거 성공률의 기반).

    Args:
        raw: 모델 원시 출력.
        schema: jsonschema dict (선택).
    Returns:
        위 세 범주 중 하나.
    """
    obj, kind = parse_json_output(raw)
    if kind != "ok":
        return "no_json"
    if schema is not None and not validate_contract(obj, schema):
        return "schema_violation"
    return "ok"


def compliance_metrics(records: list, schema: dict = None) -> dict:
    """계약 준수 지표 집계 (test-way.md 2.2).

    세 범주 분류를 적용합니다:
        ok   : 계약 준수 (준수율 분자)
        fail : 경기했으나 계약 위반 (분모에 포함)
        dnp  : 결장 — 상류 포화(429) 등으로 경기 자체가 불성립
               (분모에서 제외. "가용성을 준수율로 오측정"하는 함정 방지)

    준수율 = ok / (ok + fail) — "경기한 것"만 잰다.
    dnp 항목의 재시도는 비용이므로 재시도 합계에는 포함합니다.

    Args:
        records: dict 리스트. 각 항목:
            {"raw": 원시 출력, "retries": int, "latency_s": float,
             "dnp": bool}
        schema: jsonschema dict (선택). None이면 JSON 여부만으로 판정.
    Returns:
        dict — n/played/ok/fail/dnp, compliance_rate, no_json_rate,
        schema_violation_rate, retries_total, retries_avg, latency_avg_s.
        played가 0이면 rate 항목들은 None.
    """
    n = len(records)
    ok = fail = dnp = no_json = schema_viol = 0
    retries_total = 0
    lats = []
    for r in records:
        retries_total += int(r.get("retries", 0) or 0)  # None이면 0으로.
        if r.get("dnp"):                 # 결장 — 준수율 분모에서 제외.
            dnp += 1
            continue
        lat = r.get("latency_s")
        if lat is not None:
            lats.append(lat)
        kind = classify_output(r.get("raw", ""), schema)
        if kind == "ok":
            ok += 1
        elif kind == "no_json":
            no_json += 1
            fail += 1
        else:
            schema_viol += 1
            fail += 1

    played = ok + fail
    # 0 나누기 방지: 경기한 항목이 없으면 비율은 정의 불가(None).
    def _rate(num):
        return num / played if played else None

    return {
        "n": n,
        "played": played,
        "ok": ok,
        "fail": fail,
        "dnp": dnp,
        "compliance_rate": _rate(ok),
        "no_json_rate": _rate(no_json),
        "schema_violation_rate": _rate(schema_viol),
        "no_json": no_json,
        "schema_violation": schema_viol,
        "retries_total": retries_total,
        "retries_avg": retries_total / n if n else 0.0,
        "latency_avg_s": sum(lats) / len(lats) if lats else None,
    }


class Saturated(Exception):
    """상류 포화로 경기 자체가 성립하지 않음 — 결장(DNP) 집계.

    계약 실패와는 다른 범주입니다. 429(레이트리밋) 등 상류 제공사 혼잡으로
    generate_fn이 이 예외를 던지면 run_benchmark는 해당 항목을 DNP로 처리해
    준수율 분모에서 제외합니다. "가용성을 계약 준수율로 오측정"하는 함정을
    막기 위한 분류입니다 (test-way.md 3.10 DNP 추적).

    사용 예:
        def generate(prompt):
            if rate_limited():
                raise Saturated(model_id)
            ...
    """


def guarded_generate(generate_fn, prompt: str, schema: dict,
                     max_retry: int = MAX_CONTRACT_RETRY,
                     verbose: bool = False) -> dict:
    """가드된 생성 루프 — JSON 모드 → 펜스 제거 → 스키마 게이트 → 재시도 (test-way.md 3.2).

    "절대 throw하지 말고 상태 dict를 반환" 원칙(3.2)에 따라 실패도 예외가
    아닌 결과로 돌려줍니다. 재시도 시 이전 실패 원인을 짧게 실어 보냅니다
    ("이름을 대라" — 누락된 키를 명시).

    주의: generate_fn이 던지는 예외(엔진 크래시, Saturated 등)는 잡지 않고
    그대로 전파합니다. Saturated는 run_benchmark가 DNP로 집계하고, 그 외
    예외는 "시끄러운 실패"(loud failure) 원칙에 따라 벤치를 중단시킵니다.

    Args:
        generate_fn: (prompt) -> raw 출력 문자열. 재시도 시 err_note가
                     prompt 뒤에 붙어 호출됩니다.
        prompt: 생성 프롬프트 (시스템 + 스키마 지시 포함).
        schema: jsonschema dict. None이면 "JSON이기만 하면" 통과.
        max_retry: 재시도 횟수(기본 1). 총 시도 = max_retry + 1.
        verbose: True면 재시도 로그 출력.
    Returns:
        dict —
            ok: 성공 여부(bool)
            record: 파싱+검증된 dict (실패 시 None)
            raw: 마지막 원시 출력
            latency_s: 성공 시도(또는 마지막 시도) 소요 시간
            retries: 실제 재시도 횟수
    """
    err_note = ""
    raw = ""
    latency_s = 0.0
    for attempt in range(max_retry + 1):
        t0 = time.perf_counter()
        raw = generate_fn(prompt + err_note)
        latency_s = time.perf_counter() - t0
        rec, kind = parse_json_output(raw)
        # schema가 None이면 "JSON이기만 하면" 통과 (classify_output과 일관).
        if kind == "ok" and (schema is None or validate_contract(rec, schema)):
            return {"ok": True, "record": rec, "raw": raw,
                    "latency_s": latency_s, "retries": attempt}
        if verbose:
            reason = "JSON 아님" if kind == "no_json" else "스키마 위반"
            print(f"    [재시도] attempt={attempt}: {reason}")
        err_note = "\n[이전 응답 오류: %s] 스키마를 정확히 지키세요." % (
            "JSON 아님" if kind == "no_json" else "스키마 위반")
    return {"ok": False, "record": None, "raw": raw,
            "latency_s": latency_s, "retries": max_retry + 1}


# ---------------------------------------------------------------------------
# 2.3 품질 지표 (test-way.md 2.3)
# ---------------------------------------------------------------------------

def normalize_ko(text: str, keep_spaces: bool = False) -> str:
    """한국어 전사/텍스트 정규화 — 평가 전에 같은 형태로 맞춘다.

    ASR CER 산정(2.3)에서 정답과 예측을 동일 형태로 만드는 기준선입니다.
    그렇지 않으면 "카드 결제가 안 돼요."와 "카드결제가안돼요"가 같은 의미인데도
    띄어쓰기·문장부호 차이만으로 오류로 집계됩니다.

    하는 일 (asr_metrics.py와 동일 규약):
    1. 문장부호·특수문자 제거 — `[^\\w가-힣\\s]` 를 제거한다.
       (`\\w`는 영숫자와 밑줄(`_`)을 뜻하므로, 밑줄은 다음 줄에서 별도 제거)
    2. 소문자화 — 영문 대문자/소문자 차이(코드 스위칭 트랙)를 무시.
    3. 띄어쓰기 처리:
       - CER(기본): 공백까지 제거 → 음절(문자) 단위 비교.
       - WER(keep_spaces=True): 공백 보존+압축 → 어절 단위 비교.

    Args:
        text: 정규화할 텍스트.
        keep_spaces: True면 공백 보존(WER용), False면 제거(CER용).
    Returns:
        정규화된 문자열.
    """
    # ① 문장부호/특수문자 제거. `\w` = [a-zA-Z0-9_], `가-힣` = 한글 음절.
    t = re.sub(r"[^\w가-힣\s]", "", text or "")
    # ② `\w`가 밑줄(_)을 포함하므로 명시적으로 제거.
    t = t.replace("_", "")
    # ③ 띄어쓰기 처리: WER은 보존+압축, CER은 완전 제거.
    if keep_spaces:
        t = re.sub(r"\s+", " ", t).strip()
    else:
        t = re.sub(r"\s+", "", t)
    # ④ 소문자화 (영어 코드 스위칭 대비).
    return t.lower()


def levenshtein(a, b) -> int:
    """최소 편집 거리(삽입 + 삭제 + 치환 횟수) — 한 줄 DP 최적화.

    "a를 b로 바꾸는 데 필요한 최소 편집 연산 수"를 구하는 전형적인 동적
    프로그래밍(DP) 문제입니다. CER에서는 문자 시퀀스, WER에서는 어절(단어)
    리스트를 그대로 넣을 수 있어 str/list 모두 동작합니다.
    (asr_metrics.py와 동일 구현)

    DP 재귀식:
        dp[i][j] = min(
            dp[i-1][j] + 1,       # a[i] 삭제
            dp[i][j-1] + 1,       # b[j] 삽입
            dp[i-1][j-1] + cost,  # 같으면 유지(0), 다르면 치환(1)
        )
    i는 a의 길이, j는 b의 길이.

    메모리 최적화: 전체 (n+1) x (m+1) 표를 만들지 않고, 바로 윗행 `prev`
    한 줄만 유지합니다. 어떤 셀도 직전 행/열만 참조하기 때문에 한 줄 DP로
    충분합니다.

    Args:
        a: 시퀀스(문자열 또는 리스트).
        b: 시퀀스(문자열 또는 리스트).
    Returns:
        최소 편집 거리(정수).
    """
    # 편의상 항상 a가 더 길도록 스왑 (행 수를 줄여 반복량 최소화).
    if len(a) < len(b):
        a, b = b, a
    # 0행 초기화: a가 ""일 때 b를 만들려면 len(b)번 삽입이 필요.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):           # i = a의 인덱스(1부터).
        cur = [i]                            # 0열: b가 ""일 때 a[i]개 삭제 필요.
        for j, cb in enumerate(b, 1):        # j = b의 인덱스(1부터).
            # 삭제 / 삽입 / (유지 또는 치환) 중 최솟값 선택.
            # `ca != cb`는 True(1)/False(0)로 정수처럼 계산된다.
            cur.append(min(prev[j] + 1,      # 삭제
                           cur[j - 1] + 1,   # 삽입
                           prev[j - 1] + (ca != cb)))  # 유지(0)/치환(1)
        prev = cur                            # 한 줄 이동.
    return prev[-1]                           # 마지막 셀이 답.


def cer(ref: str, hyp: str) -> float:
    """문자 오류율(CER) — 캐스케이드 중간 전사 품질 검증 (test-way.md 2.3).

    단일 호출(옴니/Gemini)에서는 중간 전사가 없으므로, 전사가 계약 필드로
    반환되는 경우 또는 캐스케이드에서 이 CER로
    "ASR-LLM 통합 품질"을 검증합니다. 목표: 노이즈 트랙 CER <= 0.15.

    정규화 정책: 공백·구두점 제거 후 비교. 정답이 빈 문자열이면
    결과도 비면 0.0, 텍스트가 있으면 1.0 (0 나누기 방지).
    """
    ref_n, hyp_n = normalize_ko(ref), normalize_ko(hyp)
    if not ref_n:
        # 0으로 나누기 방지: 정답이 없으면 정의상 빈 결과만 정답.
        return 0.0 if not hyp_n else 1.0
    return levenshtein(ref_n, hyp_n) / len(ref_n)


def hangul_ratio(text: str) -> float:
    """한글 비율 = 알파벳 문자 중 한글의 비율.

    공백·문장부호·숫자를 제외한 알파벳만 센 뒤, 그중 `가-힣` 비율을 반환.
    코드 스위칭(2.3)과 언어 전환을 수치화하는 기본 지표.

    test-way.md 2.3 — T=0.1 실측에서 한글 비율 >= 0.984.
    판정 기준은 0.98 (MIN_HANGUL_RATIO).

    Args:
        text: 모델 출력 문자열.
    Returns:
        0.0 ~ 1.0. 알파벳이 하나도 없으면 0.0.
    """
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if "가" <= c <= "힣") / len(chars)


def char_mix(text: str) -> dict:
    """문자 혼입 3분류 — 한글 / 한자(CJK) / 라틴.

    한글 비율 검출기를 진화시켜, Qwen 계열이 섞기 쉬운 **한자**를
    별도로 식별합니다. 라틴(영어) 혼입보다 한자 혼입이 잠행적이기 때문에
    알파벳을 3분류로 나눠 셉니다 (test-way.md 2.3의 한자 오염).

    Args:
        text: 모델 출력 문자열.
    Returns:
        dict — hangul/hanja/latin/other_alpha 개수와 hangul_ratio.
    """
    counts = {"hangul": 0, "hanja": 0, "latin": 0, "other_alpha": 0}
    for c in text:
        if not c.isalpha():
            continue
        if "가" <= c <= "힣":
            counts["hangul"] += 1
        elif "\u4e00" <= c <= "\u9fff":        # CJK 통합 한자 구간.
            counts["hanja"] += 1
        elif c.isascii():
            counts["latin"] += 1
        else:
            counts["other_alpha"] += 1
    total = sum(counts.values())
    counts["hangul_ratio"] = counts["hangul"] / total if total else 0.0
    return counts


def has_hanja(text: str) -> bool:
    """한자 오염 여부 — CJK 통합 한자가 하나라도 있으면 True (test-way.md 2.3).

    한자 오염은 "한글이 얼마나 되느냐"와 별개 신호입니다. 예: Qwen3가
    "요금"을 "料金"으로 섞는 경우 한글 비율은 높아도 한자가 존재합니다.
    """
    return char_mix(text)["hanja"] > 0


def is_code_switch(text: str, min_hangul_ratio: float = MIN_HANGUL_RATIO) -> bool:
    """코드 스위칭 여부 — 한글 비율이 기준 미만이면 True.

    한국어 응답에 영어/한자 혼입이 심하면 TTS 품질과 음성 에이전트 응대
    품질이 떨어집니다. EXAONE 권고 실측에서 T=0.1 → 한글 비율 0.984,
    판정 기준은 0.98 (test-way.md 2.3).

    Args:
        text: 모델 출력 문자열.
        min_hangul_ratio: 판정 기준(기본 0.98).
    Returns:
        코드 스위칭 여부. 텍스트에 알파벳이 없으면 False.
    """
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return False
    return hangul_ratio(text) < min_hangul_ratio


def response_length_chars(text: str) -> int:
    """응답 길이(문자 수). test-way.md 2.3 — 계약 한도 대비 측정."""
    return len(text or "")


def length_compliant(text: str, max_chars: int = MAX_RESPONSE_CHARS) -> bool:
    """응답 길이가 계약 한도를 넘지 않는지 (test-way.md 2.3).

    한도 200자 ≈ TTS 합성 약 15초. 길이 초과는 TTS 지연·노트 포맷 깨짐의
    원인이 되므로 별도 지표로 잡습니다.
    """
    return len(text or "") <= max_chars


def sanity_check(text: str):
    """모델 출력 정상성 프로브 — 로드 래더의 심장.

    fp16 오버플로(T4에서 bf16 불가)는 "조용한 쓰레기"로 나타납니다 —
    예외도 없이 깨진 텍스트를 반환합니다. 다음 세 증상을 검사합니다:

      ① 빈 출력
      ② 한글 부재 (한국어로 물었는데 한글이 전혀 없음)
      ③ 반복 붕괴 (토큰 종류가 지나치게 적어 루프 의심)

    Args:
        text: 모델 출력 문자열.
    Returns:
        (ok: bool, reason: str) — ok=False면 reason에 원인.
    """
    if not text or not text.strip():
        return False, "빈 출력"
    if not any("가" <= ch <= "힣" for ch in text):
        return False, "한글 부재 (요청은 한국어였음)"
    toks = text.split()
    # 반복 붕괴: 어절이 8개 이상인데 그중 종류가 max(2, n//6)개 이하이면
    # 모델이 같은 토큰을 루프로 돌고 있는 것.
    if len(toks) >= 8 and len(set(toks)) <= max(2, len(toks) // 6):
        return False, "반복 붕괴 의심"
    return True, "정상"


# ---------------------------------------------------------------------------
# 2.4 배포 지표 (test-way.md 2.4 / 3.3 / 3.4)
# ---------------------------------------------------------------------------

def vram_fp16_gb(params_b: float, margin: float = 1.15) -> float:
    """fp16 로드 시 예상 VRAM(GB) = 파라미터(10억) × 2바이트 × 마진.

    test-way.md 2.4: fp16 = 파라미터 × 2바이트. 활성값·
    버퍼 여유 15%를 더해 `params * 2 * 1.15`로 계산합니다 (4.3 표의
    Llama-3.2-Korean 3B → ~6.4GB가 이 공식).

    Args:
        params_b: 파라미터 수(십억 단위, 예: 3.0).
        margin: 메모리 여유 계수(기본 1.15).
    Returns:
        예상 VRAM(GB).
    """
    return params_b * 2.0 * margin


def vram_nf4_gb(params_b: float) -> float:
    """NF4 4bit 양자화 로드 시 예상 VRAM(GB).

    test-way.md 2.4/3.3: NF4 = 파라미터 × 0.55 × 1.3 마진.
    예: Qwen3-8B → 8.2 × 0.55 × 1.3 ≈ 5.4GB — T4(16GB)에서 유일한 경로.
    """
    return params_b * 0.55 * 1.3


def fits_on_t4(gb: float, total_gb: float = T4_VRAM_GB) -> bool:
    """VRAM 추정치가 T4 16GB에 들어가는지 (test-way.md 4.2 ② VRAM 산수).

    Args:
        gb: 모델+양자화 조합의 예상 VRAM(GB).
        total_gb: 사용 가능 VRAM(GB). 기본 T4 16GB.
    Returns:
        탑재 가능 여부(bool).
    """
    return gb <= total_gb


def license_ok(license_name: str) -> bool:
    """상용 배포 가능 라이선스 여부 (test-way.md 3.3/4.2 ① 라이선스).

    화이트리스트(LICENSE_OK): MIT ✓, Apache 2.0 ✓, llama3.2 ✓
    블랙리스트(LICENSE_NOK): CC-BY-NC ✗, qwen-research ✗

    "가정하지 말고 표를 참조한다" — 모델 카드의 라이선스 문자열을
    소문자로 정규화해 매칭합니다. 미지의 라이선스는 fail-closed로 False
    (허가 없이는 상용 배포 불가가 기본 정책).

    Args:
        license_name: 모델 카드의 라이선스 문자열.
    Returns:
        상용 배포 가능 여부(bool).
    """
    lic = (license_name or "").strip().lower()
    if lic in LICENSE_OK:
        return True
    if lic in LICENSE_NOK:
        return False
    return False   # 미지의 라이선스 — 보수적으로 허가하지 않음.


def model_load_time_s(load_fn) -> float:
    """모델 로드(콜드 스타트) 시간 측정(초) (test-way.md 2.4).

    사용자가 앱을 켰을 때 첫 반응까지 걸리는 시간. Gemma E2B ~10s,
    EXAONE 1.2B ~3s가 기준선.

    Args:
        load_fn: 모델을 로드하는 호출 가능 객체(인자 없음).
    Returns:
        로드에 걸린 시간(초).
    """
    t0 = time.perf_counter()
    load_fn()
    return time.perf_counter() - t0


def measure_vram_mb() -> float:
    """현재 프로세스가 점유한 GPU VRAM(MB) 측정 (test-way.md 2.4).

    T4 Colab의 VRAM 상한이 16GB이므로, 모델+양자화 조합이 예산 안에
    들어오는지 확인할 때 씁니다. torch가 설치돼 있고 CUDA를 사용할 수
    있을 때만 측정하며, 그 외 환경(CPU, torch 미설치)에서는 None을
    반환합니다. 호출 시점의 할당량이므로 모델 로드 직후에 불러야 의미가
    있습니다.

    Returns:
        VRAM 사용량(MB, float). 측정 불가 환경이면 None.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated()) / (1024 ** 2)
    except Exception:
        pass   # torch 미설치/CUDA 불가 등 — None 반환.
    return None


def load_ladder(fp16_loader, nf4_loader, probe_fn,
                verbose: bool = True) -> dict:
    """로드 래더 — fp16 시도 → 정상성 프로브 → 4bit NF4 강등 (test-way.md 3.4).

    핵심: fp16 오버플로는 예외가 아니라 "조용한 쓰레기"로 나타납니다.
    따라서 로드 후 반드시 정상성 프로브(sanity_check)를 통과해야만 확정합니다.

    절차:
        1. fp16 로드 시도 → 프로브 통과 시 확정.
        2. 프로브 실패/로드 예외 → 4bit NF4로 강등 (compute float16 명시).
        3. 모두 실패 → 시끄럽게 raise (조용한 실패 금지 원칙).

    Args:
        fp16_loader: () -> model, fp16 로드 호출 가능 객체.
        nf4_loader: () -> model, 4bit NF4 로드 호출 가능 객체.
        probe_fn: (model) -> (ok: bool, reason: str), 정상성 프로브.
        verbose: True면 단계별 로그 출력.
    Returns:
        dict — mode("fp16"|"nf4"), model, load_time_s, probe_ok, probe_note.
    Raises:
        RuntimeError: 전 단계 실패 시 (원인 확인을 위해).
    """
    for mode, loader in (("fp16", fp16_loader), ("nf4", nf4_loader)):
        if verbose:
            print(f"[사다리] {mode} 로드 시도...")
        t0 = time.perf_counter()
        try:
            model = loader()
            load_time_s = time.perf_counter() - t0
        except Exception as e:
            if verbose:
                print(f"  → {mode} 로드 예외: {type(e).__name__}: {e!r}")
            continue
        ok, note = probe_fn(model)
        if verbose:
            print(f"  로드 {load_time_s:.1f}s / [건전성 프로브] {note}")
        if ok:
            return {"mode": mode, "model": model, "load_time_s": load_time_s,
                    "probe_ok": True, "probe_note": note}
        if verbose:
            print(f"  → {mode} 프로브 실패({note}), 다음 단계로 강등")
    raise RuntimeError("로드 사다리 전 단계 실패 — VRAM/게이트/버전 로그 확인")


# ---------------------------------------------------------------------------
# 3.1 프롬프트 계약 (test-way.md 3.1)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str, tokens_per_char: float = TOKENS_PER_CHAR) -> int:
    """한국어 텍스트 토큰 수 근사 추정.

    토크나이저마다 다른 값을 근사하는 헬퍼입니다 — 상수도 실측으로 보정
    합니다 (`tokens_per_char = 0.85`). +4는 역할 특수 토큰 몫.

    Args:
        text: 텍스트.
        tokens_per_char: 문자당 토큰 계수(기본 0.85).
    Returns:
        추정 토큰 수(정수).
    """
    return math.ceil(len(str(text)) * tokens_per_char) + 4


def estimate_tokens_messages(messages: list) -> int:
    """메시지 목록의 총 추정 토큰 수."""
    return sum(estimate_tokens(m["content"]) for m in messages)


def validate_prompt_contract(messages: list, budget: int = None) -> bool:
    """프롬프트 계약 검증 — 위반 시 즉시 AssertionError (loud failure).

    역할 구조(test-way.md 3.1):
        - roles: system/user/assistant만 허용.
        - system은 맨 앞에 정확히 1개 (중복 금지).
        - 마지막은 반드시 user (다음 생성 대상).
        - role/content 필수, content 비어 있으면 안 됨.
        - budget이 주어지면 추정 토큰 <= 예산 (초과 시 실패).

    Args:
        messages: {"role": str, "content": str} 딕셔너리 리스트.
        budget: 토큰 예산(선택). 주어지면 초과 시 실패.
    Returns:
        True (통과). 실패 시 AssertionError를 던진다.
    """
    assert isinstance(messages, list) and messages, "[계약 위반] 빈 메시지 목록"
    for i, m in enumerate(messages):
        # dict가 아닌 항목(문자열 등)도 시끄럽게 잡는다 — set() 비교가
        # TypeError를 내기 전에 isinstance로 명시적 실패.
        assert isinstance(m, dict) and set(m) >= {"role", "content"}, \
            f"[계약 위반] {i}번째: role/content 누락"
        assert m["role"] in {"system", "user", "assistant"}, \
            f"[계약 위반] 미지의 role '{m['role']}'"
        assert str(m["content"]).strip(), f"[계약 위반] {i}번째 content 비어 있음"
    # system은 맨 앞에 정확히 1개 — 규칙 주입 지점이자 권한 경계 (중복은 인젝션 위험).
    assert messages[0]["role"] == "system", "[계약 위반] system이 맨 앞이 아님"
    assert sum(m["role"] == "system" for m in messages) == 1, "[계약 위반] system 중복"
    # 마지막은 반드시 user — 모델이 이어 쓸 다음 생성 대상이어야 함.
    assert messages[-1]["role"] == "user", "[계약 위반] 마지막이 user가 아님"
    if budget is not None:
        n = estimate_tokens_messages(messages)
        assert n <= budget, f"[계약 위반] 토큰 추정 {n} > 예산 {budget}"
    return True


def truncate_history(messages: list, budget: int):
    """전략적 대화 이력 절단 — system과 최신 user는 불가침.

    컨텍스트 창은 유한하고 대화 이력은 무한히 자랍니다. 무엇을 버릴지가
    설계입니다. naive 절단(앞/뒤에서 자르기)은 정작 답해야 할 최신 발화나
    system 규칙을 죽이므로, 가운데 이력을 **오래된 것부터** 제거합니다.

    Args:
        messages: 메시지 목록 (system으로 시작, user로 끝남 가정).
        budget: 토큰 예산.
    Returns:
        (kept, dropped) — 유지된 메시지 리스트와 제거된 턴 수.
        system과 마지막 user는 항상 유지됩니다.
    """
    assert messages[0]["role"] == "system" and messages[-1]["role"] == "user"
    head, tail = [messages[0]], [messages[-1]]
    middle = list(messages[1:-1])
    dropped = 0
    while middle and estimate_tokens_messages(head + middle + tail) > budget:
        middle.pop(0)                       # 가장 오래된 턴부터.
        dropped += 1
    kept = head + middle + tail
    assert estimate_tokens_messages(kept) <= budget or not middle, "예산 초과 잔존"
    return kept, dropped


# ---------------------------------------------------------------------------
# 벤치마크 하네스 — 비교 매트릭스(4.3)의 한 행을 만드는 심판 함수
# ---------------------------------------------------------------------------

def run_benchmark(engine_name: str, generate_fn, eval_set: list,
                  schema: dict = None, max_retry: int = MAX_CONTRACT_RETRY,
                  conditions: list = None, verbose: bool = True) -> dict:
    """평가 세트 전체를 돌려 계약 준수 + 지연 + 품질을 집계.

    결과는 test-way.md 4.3 비교 매트릭스의 한 행 형태입니다:
    engine / 준수율 / 재시도 / 평균 지연(ms) / 한글 비율 / 조건별 준수율.

    Args:
        engine_name: 엔진 이름(표 출력용).
        generate_fn: (prompt) -> raw 출력 문자열 함수 (guarded_generate 사용).
                     상류 포화(429 등)가 감지되면 Saturated를 던질 수 있는데,
                     이 경우 해당 항목은 DNP로 집계되어 준수율 분모에서 제외
                     됩니다.
        eval_set: 항목 리스트. 각 항목은
                  (utterance_id, prompt_text, condition) 튜플.
                  condition 예: "clean" / "noisy" / "code-switched" /
                  "injection" / "streaming" / "reasoning".
        schema: jsonschema dict. None이면 JSON 여부만으로 준수 판정.
        max_retry: guarded_generate의 재시도 횟수(기본 1).
        conditions: 집계할 조건 리스트. None이면 eval_set에서 자동 추출.
        verbose: True면 콘솔에 요약 출력.
    Returns:
        dict — engine, n, 준수/지연/품질 요약, 조건별 준수율(compliance_<cond>).
    """
    records = []
    for uid, prompt, cond in eval_set:
        try:
            out = guarded_generate(generate_fn, prompt, schema,
                                   max_retry=max_retry, verbose=verbose)
            records.append({
                "uid": uid,
                "condition": cond,
                "raw": out["raw"],
                "retries": out["retries"],
                "latency_s": out["latency_s"],
                "dnp": False,
                "record": out["record"],
            })
        except Saturated:
            # 상류 포화 — 결장으로 집계 (재시도/지연 없음, 준수율 분모 제외).
            records.append({"uid": uid, "condition": cond, "raw": "",
                            "retries": 0, "latency_s": None,
                            "dnp": True, "record": None})

    # 엣지 케이스: 항목이 하나도 없으면 평균이 NaN이 되므로 명시적으로 거부.
    if not records:
        raise ValueError("eval_set이 비어 있습니다 — 최소 1개 항목이 필요합니다.")

    m = compliance_metrics(records, schema)
    conditions = conditions or sorted({r["condition"] for r in records})

    result = {"engine": engine_name, "n": len(records)}
    result.update(m)

    # 조건별 준수율 (트랙별 계약 성과). 결장(DNP)은 분모에서 제외해
    # 전체 준수율(compliance_rate)과 같은 규약을 유지한다.
    for cond in conditions:
        sub = [r for r in records
               if r["condition"] == cond and not r.get("dnp")]
        sub_ok = sum(1 for r in sub
                     if classify_output(r["raw"], schema) == "ok")
        result[f"compliance_{cond}"] = sub_ok / len(sub) if sub else None

    # 평균 한글 비율: 성공 레코드의 reply(또는 원시 출력) 기준.
    # 결장(DNP) 항목은 출력이 없으므로 품질 집계에서 제외한다.
    ratios = []
    for r in records:
        if r.get("dnp"):
            continue
        text = ""
        if r["record"] and isinstance(r["record"], dict):
            text = r["record"].get("reply") or r["record"].get("summary") or ""
        text = text or r["raw"]
        ratios.append(hangul_ratio(text))
    result["hangul_ratio_avg"] = float(np.mean(ratios)) if ratios else None

    result["latency_avg_ms"] = (
        m["latency_avg_s"] * 1000.0 if m["latency_avg_s"] is not None else None)

    if verbose:
        cr = result["compliance_rate"]
        header = (f"[{engine_name}] n={result['n']} 준수율 {cr:.0%}"
                  if cr is not None
                  else f"[{engine_name}] n={result['n']} 준수율 판정불가")
        print(header)
        for cond in conditions:
            c = result[f"compliance_{cond}"]
            print(f"    {cond:<14} 준수율 " +
                  (f"{c:.0%}" if c is not None else "판정불가"))
        lat = result["latency_avg_ms"]
        hr = result["hangul_ratio_avg"]
        print(f"    재시도 {result['retries_total']} · 결장 {result['dnp']} · "
              f"평균 지연 {f'{lat:.0f}' if lat is not None else '-'}ms · "
              f"한글 비율 {f'{hr:.3f}' if hr is not None else '-'}")
    return result


# ---------------------------------------------------------------------------
# 자체 검증(self-test) — 모델·GPU 없이 오프라인으로 실행되는 테스트
# ---------------------------------------------------------------------------
# 아래 assert 블록은 각 지표의 동작을 오프라인에서 검증하는 자체 테스트입니다.
# `python3 llm_metrics.py`로 실행하면 아래 모든 검증이 통과해야 합니다.

if __name__ == "__main__":
    # --- 2.1 문장 경계 + 스트리밍 계측 검증 ---
    assert first_sentence("네. 확인합니다.") == "네."
    assert first_sentence("아직 안 끝났") is None

    # 토큰 도착 이벤트로 TTFT/TTFA/TTFS 동시 계측.
    # 시나리오: 0.2s에 첫 토큰, 0.5s에 </think>, 0.8s에 첫 문장("드리겠습니다.") 완성.
    _ev = [(0.2, "안녕"), (0.4, "하세요"),
           (0.5, "</think>"), (0.6, "결제"),
           (0.7, "내역을 확인해"), (0.8, "드리겠습니다.")]
    _a = analyze_stream(_ev)
    assert abs(_a["ttft_ms"] - 200.0) < 1e-9        # 첫 토큰.
    assert abs(_a["ttfa_ms"] - 500.0) < 1e-9        # </think> 이후.
    assert abs(_a["ttfs_ms"] - 800.0) < 1e-9        # 첫 문장 완성.
    assert abs(_a["gap_ms"] - 600.0) < 1e-9         # TTFS(800) - TTFT(200).
    assert _a["n_chunks"] == 6
    # 빈 이벤트 → 모든 시각 None.
    _empty = analyze_stream([])
    assert _empty["ttft_ms"] is None and _empty["ttfs_ms"] is None
    assert _empty["chunks_per_s"] is None

    # --- 2.1 처리량 / 지연 판정 ---
    assert abs(throughput_per_s(120, 2.0, ttft_s=0.2) - 120 / 1.8) < 1e-9
    assert throughput_per_s(10, 0.1, ttft_s=0.2) == 0.0    # 생성 구간 0 → 0.
    assert latency_budget_verdict(480.0) == "pass"
    assert latency_budget_verdict(520.0) == "exceed"

    # --- 2.2 펜스 제거 + JSON 파싱 ---
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('{"a": 1}') == '{"a": 1}'          # 펜스 없음.
    assert strip_fences('```json\n{"a": 1}') == '{"a": 1}'  # 닫는 펜스 누락.
    assert strip_fences(None) == ""                          # None 방어.
    obj, kind = parse_json_output('{"intent": "billing"}')
    assert kind == "ok" and obj["intent"] == "billing"
    obj, kind = parse_json_output("그냥 대화입니다")
    assert kind == "no_json"

    # --- 2.2 스키마 게이트 (INTENT_SCHEMA) ---
    INTENT_SCHEMA = {
        "type": "object",
        "properties": {
            "intent": {"type": "string",
                       "enum": ["billing", "tech_support", "loss_suspend",
                                "plan_change", "payment_change"]},
            "slots": {"type": "object"},
            "reply": {"type": "string", "minLength": 1},
            "handoff_to_human": {"type": "boolean"},
        },
        "required": ["intent", "slots", "reply", "handoff_to_human"],
        "additionalProperties": False,
    }
    assert validate_contract(
        {"intent": "billing", "slots": {}, "reply": "확인해드리겠습니다.",
         "handoff_to_human": False}, INTENT_SCHEMA)
    assert not validate_contract({"intent": "biling"}, INTENT_SCHEMA)  # enum 위반.
    assert validate_contract({"anything": 1}, None)       # schema None = 제약 없음.
    assert classify_output(
        '```json\n{"intent": "billing", "slots": {}, "reply": "네.", '
        '"handoff_to_human": false}\n```', INTENT_SCHEMA) == "ok"
    assert classify_output("그냥 대화입니다", INTENT_SCHEMA) == "no_json"
    assert classify_output(
        '{"intent": "biling", "slots": {}, "reply": "네.", '
        '"handoff_to_human": false}', INTENT_SCHEMA) == "schema_violation"

    # --- 2.2 준수율 집계 ---
    # 4개 레코드: ok 2 / no_json 1 / dnp 1 → played 3, 준수율 2/3.
    _recs = [
        {"raw": '{"intent": "billing", "slots": {}, "reply": "네.", '
                '"handoff_to_human": false}', "retries": 0, "latency_s": 0.5},
        {"raw": '{"intent": "tech_support", "slots": {}, "reply": "확인 중입니다.", '
                '"handoff_to_human": true}', "retries": 1, "latency_s": 1.2},
        {"raw": "장애 때문에 못 하겠습니다.", "retries": 2, "latency_s": 2.0},
        {"raw": "", "retries": 0, "latency_s": 0.1, "dnp": True},   # 결장.
    ]
    _cm = compliance_metrics(_recs, INTENT_SCHEMA)
    assert _cm["ok"] == 2 and _cm["fail"] == 1 and _cm["dnp"] == 1
    assert abs(_cm["compliance_rate"] - 2 / 3) < 1e-9   # 결장은 분모 제외.
    assert _cm["no_json"] == 1 and _cm["no_json_rate"] == 1 / 3
    assert _cm["retries_total"] == 3                     # dnp 재시도도 비용.
    assert abs(_cm["latency_avg_s"] - (0.5 + 1.2 + 2.0) / 3) < 1e-9
    # retries가 None이어도 크래시하지 않는다 (0으로 취급).
    _cm2 = compliance_metrics([{"raw": "x", "retries": None, "latency_s": 0.1}],
                              None)
    assert _cm2["retries_total"] == 0

    # --- 2.2 가드된 생성 루프 (3.2): 실패 → 재시도 → 성공 ---
    _calls = {"n": 0}

    def _gen(prompt):
        _calls["n"] += 1
        if _calls["n"] == 1:
            return "장애 때문에 못 하겠습니다."                     # 1차: 비-JSON.
        return '{"intent": "billing", "slots": {}, "reply": "네.", "handoff_to_human": false}'

    _g = guarded_generate(_gen, "프롬프트", INTENT_SCHEMA, max_retry=1)
    assert _g["ok"] and _g["retries"] == 1 and _g["record"]["intent"] == "billing"
    # 재시도 소진 → 실패 상태 dict (throw하지 않는다).
    _g2 = guarded_generate(lambda p: "계속 실패", "p", INTENT_SCHEMA, max_retry=1)
    assert not _g2["ok"] and _g2["retries"] == 2

    # --- 2.3 한글 비율 / 문자 혼입 ---
    assert hangul_ratio("안녕하세요") == 1.0
    assert hangul_ratio("hello") == 0.0
    assert abs(hangul_ratio("안녕 hello 세요") - 4 / 9) < 1e-9
    m = char_mix("요금은 料金입니다 fee")          # 한글6/한자2/라틴3.
    assert m["hanja"] == 2 and m["latin"] == 3 and m["hangul"] == 6
    assert char_mix("안녕하세요")["hangul_ratio"] == 1.0
    assert has_hanja("요금 料金 확인") and not has_hanja("요금 확인")
    assert is_code_switch("hello world")          # 알파벳 있는 라틴 혼입.
    assert not is_code_switch("안녕하세요")
    assert not is_code_switch("")                 # 알파벳 없음 → False.

    # --- 2.3 CER (ASR-LLM 통합 품질, test-way.md 2.3) ---
    assert cer("배송 조회 부탁해요", "배송 조회 부탁해요") == 0.0
    assert cer("배송 조회", "배숭 조회") == 0.25
    assert cer("", "x") == 1.0 and cer("", "") == 0.0

    # --- 2.3 응답 길이 / 정상성 프로브 ---
    assert response_length_chars("안녕하세요.") == 6
    assert length_compliant("안녕하세요") and not length_compliant("가" * 201)
    assert sanity_check("대한민국의 수도는 서울입니다.")[0] is True
    assert sanity_check("")[0] is False                     # 빈 출력.
    assert sanity_check("Good morning!")[0] is False        # 한글 부재.
    assert sanity_check("네 네 네 네 네 네 네 네")[0] is False  # 반복 붕괴.

    # --- 2.4 VRAM 산수 ---
    assert abs(vram_fp16_gb(3.0) - 3.0 * 2 * 1.15) < 1e-9
    assert abs(vram_nf4_gb(8.2) - 8.2 * 0.55 * 1.3) < 1e-9
    assert not fits_on_t4(vram_fp16_gb(8.2))                # 18.9GB > 16GB.
    assert fits_on_t4(vram_nf4_gb(8.2))                     # ~5.9GB <= 16GB.

    # --- 2.4 라이선스 게이트 (test-way.md 4.2) ---
    assert license_ok("MIT")
    assert license_ok("Apache-2.0")
    assert license_ok("llama3.2")
    assert not license_ok("CC-BY-NC")
    assert not license_ok("qwen-research")
    assert not license_ok("미지의-라이선스")                 # fail-closed.

    # --- 2.4 로드 래더 (3.4): fp16 프로브 실패 → NF4 강등 ---
    def _lp16():
        return "fp16-model"        # fp16 로드는 성공하지만
    def _lnf4():
        return "nf4-model"
    def _probe(model):             # 프로브는 모델 정체로 판정.
        if model == "nf4-model":
            return True, "정상"
        return False, "한글 부재 (fp16 붕괴 의심)"     # fp16 프로브 실패.
    _ll = load_ladder(_lp16, _lnf4, _probe, verbose=False)
    assert _ll["mode"] == "nf4" and _ll["model"] == "nf4-model"
    # 둘 다 실패 → RuntimeError (시끄러운 실패).
    _dead = lambda: (_ for _ in ()).throw(RuntimeError("로드 실패"))
    try:
        load_ladder(_dead, _dead, _probe, verbose=False)
        raise SystemExit("FAIL: 래더 전 단계 실패가 예외가 아님")
    except RuntimeError:
        pass

    # --- 3.1 프롬프트 계약 ---
    _ok_msgs = [{"role": "system", "content": "당신은 콜센터 상담원입니다."},
                {"role": "user", "content": "환불하고 싶어요."}]
    assert validate_prompt_contract(_ok_msgs) is True
    for _bad in (_ok_msgs + [_ok_msgs[0]],               # system 중복.
                 [{"role": "uesr", "content": "환불"}],  # role 오타.
                 [_ok_msgs[0], _ok_msgs[1],
                  {"role": "assistant", "content": "네."}],  # user로 안 끝남.
                 ["문자열-메시지"]):                        # dict가 아님.
        try:
            validate_prompt_contract(_bad)
            raise SystemExit("FAIL: 위반이 통과됨")
        except AssertionError:
            pass
    # 전략적 절단: system과 최신 user 생존, 중간 이력 제거.
    _hist = [{"role": "system", "content": "존댓말만 사용합니다. " * 3}]
    for i in range(12):
        _hist += [{"role": "user", "content": f"{i}번째 문의입니다. " * 6},
                  {"role": "assistant", "content": f"{i}번째 답변입니다. " * 6}]
    _hist += [{"role": "user", "content": "그래서 환불은 언제 되나요?"}]
    _kept, _dropped = truncate_history(_hist, 300)
    assert _kept[0]["role"] == "system"
    assert _kept[-1]["content"].endswith("언제 되나요?")
    assert _dropped > 0

    # --- 벤치마크 하네스 (4.3) ---
    _eval = [("utt_001", "청구 내역 확인", "clean"),
             ("utt_002", "인터넷 끊김 예약", "noisy"),
             ("utt_003", "폰 분실 정지", "clean")]
    _ok_json = ('{"intent": "billing", "slots": {}, "reply": "확인해드리겠습니다.", '
                '"handoff_to_human": false}')
    _res = run_benchmark("fake", lambda p: _ok_json, _eval,
                         INTENT_SCHEMA, verbose=False)
    assert _res["compliance_rate"] == 1.0
    assert _res["compliance_clean"] == 1.0 and _res["compliance_noisy"] == 1.0
    assert _res["retries_total"] == 0
    try:
        run_benchmark("fake", lambda p: _ok_json, [], INTENT_SCHEMA,
                      verbose=False)
        raise SystemExit("FAIL: 빈 eval_set 허용")
    except ValueError:
        pass

    # 상류 포화(Saturated) → DNP 결장: 준수율 분모에서 제외, 별도 집계.
    def _sat(prompt):
        raise Saturated("upstream-saturated")
    _res2 = run_benchmark("saturated", _sat,
                          [("u1", "p1", "clean"), ("u2", "p2", "clean")],
                          INTENT_SCHEMA, verbose=False)
    assert _res2["dnp"] == 2 and _res2["played"] == 0
    assert _res2["compliance_rate"] is None               # 판정불가(None).
    assert _res2["compliance_clean"] is None              # 조건별도 DNP 분모 제외.
    assert _res2["hangul_ratio_avg"] is None              # 결장은 품질 제외.

    print("llm_metrics.py 자체 검증 통과 ✅")
