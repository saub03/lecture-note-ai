# ASR 모델 테스트 계획

> ASR 모델 평가 계획 (2026-07-31)

---

## 1. 개요: ASR 파이프라인 아키텍처

모든 평가 실험이 하나의 아키텍처 패턴으로 수렴합니다:

```
Audio → [전처리: 16kHz 모노, log-mel 스펙트로그램]
     → [VAD: 발화 구간 분할]
     → [ASR 엔진: 어댑터로 래핑]
     → [신뢰도 게이트 + 재시도]
     → [환각(Hallucination) 필터]
     → [후처리: 사전(lexicon), ITN]
     → [데이터 계약 어댑터 → dict]
```

### 평가된 엔진

| 엔진 | 아키텍처 | 한국어 지원 | 신뢰도 지표 | 지연 특성 |
|---|---|---|---|---|
| faster-whisper (large-v3-turbo) | Encoder-Decoder (AR) | 예 | `avg_logprob` | 빠름 (turbo: 디코더 4층) |
| ghost613/ft-whisper-turbo-korean (한국어 FT) | Encoder-Decoder (AR) | 예 (파인튜닝) | `avg_logprob` | turbo와 동일 |
| Qwen3-ASR-0.6B | LLM 기반 (AR) | 예 (52개 언어) | `text != ""` 프록시 | 발화당 수백 ms |
| Cohere Transcribe | Conformer+AED (NAR) | 예 | 없음 (VAD 게이트 필요) | 빠름, 35초 자동 청크 |
| Meta Omnilingual (CTC) | Encoder+CTC (NAR) | 예 (1,672개 언어) | 없음 | 단일 추론 최고 속도 |
| Meta Omnilingual (LLM) | Encoder+LLaMA (AR) | 예 | 없음 | 느림 (자기회귀 패널티) |
| SenseVoice-Small | CTC (NAR) | 예 | `<\|nospeech\|>` 태그 | 오디오 10초당 ~70ms |
| GPT-Live (분석 전용) | Full-duplex 블랙박스 | 예 | N/A | 프레임 단위 40ms |

---

## 2. ASR 모델 평가 핵심 지표

### 2.1 정확도 지표

#### CER (Character Error Rate, 문자 오류율)
- **공식**: `(대체 + 삭제 + 삽입) / N`
- **한국어에서 WER 대신 CER을 쓰는 이유**: 한국어는 띄어쓰기가 일관되지 않아 음절 단위 CER이 더 안정적
- **구현**: `jiwer` 또는 직접 구현한 numpy 기반 편집거리(Levenshtein)
- **평가 세트**: 클린 오디오 + SNR 5dB(노이즈) — 둘 다 필요
- **정규화 정책**: CER 계산 전 공백/구두점 제거; ITN 규약을 정답(GT)과 일치

#### DER (Diarization Error Rate, 화자 분리 오류율)
- **공식**: `(누락 + 오탐 + 혼동) / 전체 참조 발화`
- **하위 구성 요소 → 파이프라인 단계 매핑**:
  - 누락(Miss) → VAD가 너무 보수적
  - 오탐(False Alarm) → VAD가 너무 민감
  - 혼동(Confusion) → 임베딩/클러스터링 실패
- **활용처**: 다중 화자 강의 시나리오 (교수 + 학생 질문)
- **모델**: ECAPA-TDNN 임베딩 + 코사인 거리 응집 클러스터링

### 2.2 속도 지표

| 지표 | 정의 | 예산 |
|---|---|---|
| **지연(Latency, ms)** | 발화 종료 → 최종 전사 | ≤ 500ms (1.5s 왕복 중 ASR 몫) |
| **RTF (Real-Time Factor)** | 처리 시간 / 오디오 길이 | < 1.0 (스트리밍 가능 여부) |
| **first_partial_ms** | 스트림 시작 → 첫 부분 가설 | 체감 반응성 |
| **EoU 대기(ms)** | 발화 경계 판단용 침묵 임계값 | 300ms(공격적) ~ 1200ms(안전) |

### 2.3 견고성 지표

| 지표 | 신호 | 임계값 |
|---|---|---|
| **avg_logprob** | Whisper 고유 신뢰도 | ≥ -1.0 → `confidence_ok = True` |
| **no_speech_prob** | Whisper 침묵 감지 | 높음 → 환각 위험 |
| **발화 비율(Speech ratio)** | RMS 임계값 초과 프레임 비율 | 거의 0 + 텍스트 존재 → 환각 |
| **토큰 반복** | 동일 토큰 ≥ 4회 | 환각 루프 |

### 2.4 배포 지표

| 지표 | 중요 이유 |
|---|---|
| **VRAM (GB)** | T4 Colab = 16GB 상한 |
| **모델 로드 시간 (초)** | 콜드 스타트 UX |
| **배치 처리량 (발화/초)** | 야간 배치 QA vs 실시간 단건 처리 |
| **양자화 허용 오차** | int8 float16 vs float16 CER 차이 |

---

## 3. 성능 개선 처리 방법

### 3.1 전처리

| 방법 | 상세 | 효과 |
|---|---|---|
| **리샘플 → 16kHz 모노** | `librosa.resample` 또는 `soxr` | 모든 ASR의 공통 입력 요건 |
| **Log-mel 스펙트로그램** | 80채널, 25ms 윈도우, 10ms 홉 (n_fft=400, hop=160) | 모든 엔진의 공유 입력 |
| **홉 크기 트레이드오프** | hop=160→320 시 프레임·ASR 부하 절반 | 시간 해상도 vs 속도 |
| **노이즈 주입** | SNR 5dB 스트레스 테스트 | 견고성 상한 측정 |

### 3.2 VAD (Voice Activity Detection, 음성 구간 감지)

| 방법 | 장점 | 단점 |
|---|---|---|
| **에너지 기반 RMS** | 의존성 없음, 빠름 | 노이즈에 취약 |
| **Silero VAD v5+** | 신경망, pip 설치, 오프라인 | 약간 무거움 |
| **SenseVoice `<\|nospeech\|>` 태그** | 내장, 추가 패스 불필요 | 엔진 한정 |
| **silence_chunks=3 × 320ms = 960ms EoU** | 프로덕션 표준 | 고정 지연 하한 |

**환각 방어**: 침묵 → VAD 게이트가 **ASR 이전에** 차단해야 합니다. 모든 엔진이 침묵에서 환각을 생성합니다(Cohere, Whisper, Qwen3 모두 확인됨). 어떤 신뢰도 지표도 이를 확실히 잡아내지 못합니다.

### 3.3 신뢰도 & 재시도 전략 (2-Tier)

```
1. 그리디 디코딩으로 전사 (빠름)
2. 신뢰도 < 임계값이면:
   → beam=5로 재시도 (정확, 최대 1회 재시도)
3. 여전히 낮은 신뢰도면:
   → status="low_confidence" 반환 (절대 예외 던지지 않음)
```

| 엔진 | 신뢰도 임계값 |
|---|---|
| Whisper | `avg_logprob ≥ -1.0` |
| Qwen3-ASR | `text != ""` (프록시) |
| Cohere / Omnilingual | VAD 전용 게이트 (신뢰도 신호 없음) |

### 3.4 환각 탐지 (3가지 신호)

1. **침묵 역설**: 발화 비율이 거의 0인데 텍스트가 반환됨 → 필터
2. **정형구 패턴**: 유튜브식 문구("구독과 좋아요"), 필러 반복
3. **토큰 반복**: 동일 토큰 ≥ 4회 연속 → 필터

### 3.5 디코딩 전략

| 전략 | 속도 | 정확도 | 사용 시점 |
|---|---|---|---|
| **Greedy** | 가장 빠름 | 기준 | 실시간 기본값 |
| **Beam=5** | ~3-5× 느림 | 더 좋음 | 낮은 신뢰도 재시도 |
| **LocalAgreement-2** | 스트리밍 오버헤드 | 텍스트 깜빡임 방지 | 스트리밍 전용 |
| **initial_prompt** | 오버헤드 없음 | 도메인 주입 | 슬라이드 컨텍스트로 시드 |

### 3.6 양자화 (VRAM ≤ 6GB 경로)

| 계산 유형 | VRAM | 속도 | CER 영향 |
|---|---|---|---|
| float16 | ~3.2GB (turbo) | 기준 | — |
| int8_float16 | ~1.6GB (turbo) | 1.1-1.3× 빠름 | 미미 (측정 필수, 가정 금지) |
| int8 | ~1.2GB | 가장 빠름 | 소폭 |

### 3.7 한국어 특화 후처리

| 방법 | 예시 | 사용 시점 |
|---|---|---|
| **도메인 사전** | `{"시퀀스 다이어그램" → "시퀀스 다이어그램"}` (결정적 치환) | 반복 오류 |
| **ITN (Inverse Text Normalization)** | "삼만 원" → "30,000원" | 숫자 정규화 |
| **공백/구두점 제거** | CER 평가 전용 | 지표 계산 전 |
| **코드 스위칭 정책** | "payment" vs "페이먼트" → 일관된 규칙 | 한국어/영어 혼용 |

### 3.8 스트리밍 아키텍처

```
오디오 청크 (320ms) → VAD → find_endpoint()
    → 성장하는 버퍼 재전사 → LocalAgreement-2 커밋
    → STREAM_EVENT_KEYS ⊃ CONTRACT_KEYS
```

**핵심 불변식**:
- `is_final=True` 이벤트는 배치 계약을 충족해야 함 (하위 호환)
- `is_final=False`(부분) 이벤트는 스트림 전용 필드만 전달
- 커밋된 텍스트는 절대 뒤집히지 않음 — LocalAgreement-2 보장

### 3.9 파인튜닝 경로

**필요한 경우**: 실제 전화 회선 오디오, 도메인 특화 용어, 억양 변형

| 접근 | 비용 | 효과 |
|---|---|---|
| 커뮤니티 한국어 FT 체크포인트 | 무료 (다운로드) | 클린 TTS: 미미; 실제 노이즈: 유의미 |
| 양자화 FT 배포 | int8 추론 | VRAM 트레이드오프 |
| `initial_prompt` 주입 | 무비용 | 파인튜닝 없는 도메인 컨텍스트 |

### 3.10 어댑터 패턴 (범용 계약)

```python
CONTRACT_KEYS = {"utt_id", "engine", "text", "language",
                  "confidence_ok", "avg_logprob", "latency_ms"}

# 선택적 확장 필드 (다운스트림 파괴 없음):
# emotion, events, speaker_id
```

모든 엔진은 고유 출력 → 계약 dict로 매핑하는 단일 어댑터 함수를 갖습니다. 다운스트림 코드(로깅, 라우팅, LLM, LiveKit)는 절대 변경되지 않습니다.

---

## 4. lecture-note-ai 평가 프로토콜

### 4.1 테스트 트랙

| 트랙 | 오디오 유형 | 지표 | 목적 |
|---|---|---|---|
| **클린** | 스튜디오 품질 한국어 강의 | CER | 기본 정확도 |
| **노이즈 (SNR 5dB)** | 강의 + 주변 소음 | CER | 견고성 |
| **코드 스위칭** | 한국어 + 영어 기술 용어 | CER + 수동 검수 | 도메인 현실성 |
| **침묵** | 순수 침묵 (0dBFS) | 환각률 | VAD 게이트 검증 |
| **스트리밍** | 시뮬레이션 실시간 청크 | final_latency_ms, RTF | 실시간 가능성 |

### 4.2 모델 선택 프레임워크

세 가지 필터:

1. **언어 지원**: 한국어 필수 → 영어 전용 엔진 제외 (Canary-Qwen, Parakeet, Voxtral)
2. **생태계 성숙도**: pip 설치 가능, T4 호환, 활발한 커뮤니티
3. **지연 예산**: T4에서 발화당 ≤500ms

### 4.3 권장 비교 매트릭스

| 엔진 | CER (클린) | CER (SNR 5dB) | 지연 (ms) | VRAM (GB) | 신뢰도 | 스트리밍 |
|---|---|---|---|---|---|---|
| faster-whisper large-v3-turbo | ? | ? | ? | ~3.2 | avg_logprob | LocalAgreement-2 |
| ghost613/ft-whisper-turbo-korean | ? | ? | ? | ~3.2 | avg_logprob | LocalAgreement-2 |
| Qwen3-ASR-0.6B | ? | ? | ? | ~1.2 | text != "" | Re-transcribe |
| SenseVoice-Small | ? | ? | ? | ~1 | nospeech 태그 | N/A (CTC) |
| Cohere Transcribe | ? | ? | ? | ~4 | VAD 전용 게이트 | 자동 청크 |
| Meta Omnilingual CTC | ? | ? | ? | ? | VAD 전용 게이트 | 배치 |

> **위 평가 프로토콜로 실제 측정값을 채워 `?`를 대체하세요.**

---

## 5. 실행 가능한 제안 (우선순위 순)

### P0: 기본 faster-whisper + VAD 파이프라인
- `src/audio/asr_engine.py`에 faster-whisper large-v3-turbo 구현
- `src/audio/recorder.py`에 Silero VAD + EoU 감지 구현
- 5개 테스트 트랙(4.1) 실행, 비교 매트릭스(4.3)에 결과 기록
- **근거**: Whisper는 업계 표준; 생태계 성숙; `avg_logprob`가 유일한 고유 신뢰도 신호

### P1: 환각 방어
- 3신호 필터(침묵 역설·정형구·토큰 반복) 구현
- 침묵 트랙으로 테스트 → 오탐 텍스트 허용치 0
- LLM 주입 전에 통합 (환각 텍스트는 다운스트림 요약을 오염시킴)

### P2: 2-Tier 신뢰도 전략
- 그리디 기본값, 낮은 avg_logprob 시 beam=5 재시도
- 발화당 최대 1회 재시도 → 지연 상한 확보
- LLM 우아한 복구를 위해 status dict 반환 (절대 예외 던지지 않음)

### P3: 한국어 후처리
- 강의 특화 용어용 도메인 사전
- 숫자 정규화용 ITN
- 코드 스위칭 정책 (영어 용어 → 한글로? 또는 유지?)

### P4: 대체 엔진 평가
- Qwen3-ASR-0.6B 실행: 작은 VRAM, 내장 LID, 코드 스위칭 강점
- SenseVoice-Small 실행: CTC 속도 + 감정/이벤트 태그 (보너스 기능)
- 어댑터 패턴으로 교체 → 파이프라인 코드 변경 0

### P5: 한국어 파인튜닝 Whisper
- 실제 강의 오디오로 ghost613/ft-whisper-turbo-korean 테스트
- 클린 + 노이즈 트랙에서 기본 turbo와 CER 비교
- 유의미한 개선(>5% CER 감소) 시 기본값으로 채택

### P6: 스트리밍 통합
- 실시간 부분 결과용 LocalAgreement-2 구현
- 계약을 STREAM_EVENT_KEYS ⊃ CONTRACT_KEYS로 확장
- `src/pipeline/stream_processor.py`에 필수 (5-6주차)

### P7: 화자 분리 (선택)
- 다중 화자 강의용 ECAPA-TDNN 임베딩 + 클러스터링
- 화자 속성 전사용 ASR 통합
- 강의 시나리오에 Q&A나 패널 토론이 있을 때만 평가

### P8: Full-Duplex 아이디어 (GPT-Live에서 차용)
- 프레임 단위 백채널(backchannel) 인정
- LLM 대기 중 프리앰블 필러 (5주차 구현 목표)
- 위임: 빠른 ASR → 심층 LLM + 우아한 복구

---

## 6. 즉시 수행할 다음 단계

1. **의존성 설치**: `faster-whisper`, `silero-vad`, `librosa`, `jiwer`
2. **테스트 오디오 녹음**: 한국어 강의 발화 5-10개 (클린 + 노이즈 + 코드 스위칭 + 침묵)
3. **베이스라인 실행**: 모든 5개 트랙에 faster-whisper large-v3-turbo
4. **매트릭스 작성**: CER, 지연, VRAM, 환각률 측정
5. **승자 선정**: T4 + 500ms + 한국어 제약을 충족하는 엔진 선택
6. **프로덕션 모듈 구현**: 승자 엔진 + VAD + 신뢰도 + 후처리를 갖춘 `src/audio/asr_engine.py`
