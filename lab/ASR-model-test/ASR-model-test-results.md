# ASR 모델 테스트 결과 — Apple Silicon (M4 Pro, 48GB)

> 실행: `lab/ASR-model-test/asr_test.ipynb` · 평가 프로토콜: `test-way.md` · 측정일: 2026-07-31
> 하드웨어: Mac mini M4 Pro (arm64, 48GB 통합 메모리) · Python 3.14.6 (conda base)

---

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| macOS / 칩 | macOS arm64, M4 Pro 48GB |
| ASR 프레임워크 | `mlx-whisper` 0.4.3 (MLX/Metal) · `openai-whisper` 20250625 (torch MPS) · `funasr` 1.4.0 (SenseVoice, CPU) |
| 오디오 처리 | librosa 0.11 · soundfile · gTTS (TTS 합성) |
| 지표 모듈 | `asr_metrics.py` (CER/WER·RTF·신뢰도·환각·노이즈·벤치마크 하네스) |
| 어댑터 모듈 | `asr_adapters.py` (어댑터 패턴 + 팩토리) |

엔진 가용성: `mlx-whisper ✅ openai-whisper ✅ sensevoice ✅ qwen3-asr ❌`
(`qwen3-asr`은 `transformers==4.57.6` 고정 핀이 `funasr`의 `transformers 5.14.1`과 충돌해 이번 사이클에서 제외. 어댑터는 등록·가드되어 있으며 별도 환경에서 실행 가능.)

### 재현 방법
```bash
streamlit run app.py        # 관련 없음 (본 실험은 노트북 단독 실행)
python3 -m jupyter nbconvert --to notebook --execute asr_test.ipynb
# 데이터셋 v1/v2는 노트북의 DATASET_ID 한 줄로 전환 (gTTS 재합성)
# 생성 오디오: data/temp_audio/asr_test/{v1,v2}/ (gitignore 대상)
```

---

## 2. 평가 세트 (test-way.md §4.1 테스트 트랙)

gTTS(한국어)로 합성한 9개 발화 × 2개 데이터셋 + 변형. 트랙별:

| 트랙 | 구성 | 항목 수 | 지표 |
|---|---|---|---|
| clean | 한국어 강의 문장 (gTTS) | 5 | CER |
| code_switched | 한국어 + 영어 기술 용어 | 4 | CER (+수동 검수) |
| noisy | clean + SNR 5dB 백색소음 (`add_noise`) | 5 | CER |
| silence | 순수 침묵(0dBFS) 2 + 미세소음(≈-60dBFS) 2 | 4 | 환각률 |
| streaming | clean 1건을 1s 청크로 성장 버퍼 재전사 | — | first_partial/final_latency/RTF |

- v1: DB 트랜잭션/CNN/정렬 등 컴퓨터공학 강의 문장
- v2: NLP 임베딩/그래프 순회/운영체제/블록체인/병렬컴퓨팅 강의 문장 (신규 데이터셋, 재현성 검증용)
- 코드 스위칭 발화는 gTTS가 영어 단어를 **한국식 발음**으로 읽으므로, 정답(영문 철자)과의 CER이 구조적으로 높게 잡힙니다(§5.2에서 해석).

---

## 3. 종합 비교 매트릭스 (test-way.md §4.3 채움)

### 사이클 1 (dataset v1) — 18건

| 엔진 | CER clean | CER code_sw | CER noisy | CER silence | 지연(ms) | RTF |
|---|---|---|---|---|---|---|
| **mlx-whisper** large-v3-turbo | **0.7%** | 55.3% | **3.5%** | 100% | **439** | **0.08** |
| openai-whisper turbo (MPS) | 0.7% | 55.3% | 3.5% | 100% | 677 | 0.12 |
| sensevoice-small (CPU) | 8.5% | 64.3% | 12.8% | 100% | 541 | 0.10 |

### 사이클 2 (dataset v2) — 18건 (신규 데이터셋)

| 엔진 | CER clean | CER code_sw | CER noisy | CER silence | 지연(ms) | RTF |
|---|---|---|---|---|---|---|
| **mlx-whisper** large-v3-turbo | **1.5%** | 67.6% | **9.9%** | 100% | **437** | **0.08** |
| openai-whisper turbo (MPS) | 1.5% | 67.6% | 9.9% | 100% | 610 | 0.11 |
| sensevoice-small (CPU) | 5.7% | 70.5% | 16.1% | 100% | 584 | 0.11 |

### 두 사이클 평균

| 엔진 | clean | code_sw | noisy | 지연(ms) | RTF | 신뢰도 신호 |
|---|---|---|---|---|---|---|
| **mlx-whisper large-v3-turbo** | **1.1%** | **61.5%** | **6.7%** | **438** | **0.08** | avg_logprob ✅ |
| openai-whisper turbo (MPS) | 1.1% | 61.5% | 6.7% | 643 | 0.12 | avg_logprob ✅ |
| sensevoice-small (CPU) | 7.1% | 67.4% | 14.5% | 563 | 0.10 | `<\|nospeech\|>` 태그 |

> **채택**: 실시간(≤500ms 예산, test-way.md §2.2) + 한국어 + 신뢰도 신호 기준으로
> **mlx-whisper large-v3-turbo**가 우선 엔진입니다. openai-whisper와 동일 가중치라
> CER이 동일하지만 MLX(Metal)가 ~1.5× 빠르고, `avg_logprob` 신뢰도 신호를 제공합니다.

---

## 4. 심층 실험 결과

### 4.1 침묵 트랙 → 환각 방어 (P1, test-way.md §3.2/§3.4)

**모든 엔진이 침묵에서 100% 환각을 생성**했습니다. (test-way.md의 예측과 일치)

| 엔진 | 환각률 | 침묵 입력에 대한 환각 텍스트 |
|---|---|---|
| mlx-whisper | 100% | "감사합니다." ×4 |
| openai-whisper | 100% | "다음 영상에서 만나요." / "감사합니다." ×4 |
| sensevoice | 100% | "그." ×4 |

→ 신뢰도 지표(`avg_logprob`)로는 이 환각을 못 잡습니다. **VAD 게이트가 ASR보다
먼저 침묵을 차단해야** 합니다 (에너지 RMS `speech_ratio` 또는 Silero VAD).
`asr_metrics.is_hallucination`의 3신호 필터(침묵 역설·정형구·토큰 반복)가 텍스트
레벨 2차 방어로 동작합니다.

### 4.2 2-Tier 신뢰도 전략 (P2, test-way.md §3.3)

mlx-whisper greedy → `avg_logprob < -1.0`이면 beam=5 재시도(최대 1회).

| 사이클 | 최저 신뢰도 noisy 발화 | avg_logprob | greedy conf | greedy CER → beam CER |
|---|---|---|---|---|
| v1 | utt_003_noisy5 | -0.233 | True | 6.7% → 6.7% |
| v2 | utt_102_noisy5 | -0.277 | True | 20.0% → 20.0% |

→ 테스트 발화는 모두 `avg_logprob ≥ -1.0`이라 재시도가 트리거되지 않았고
(정상 동작), greedy가 beam과 동일 결과. 예산(≤500ms)을 지키면서
**신뢰도 게이트가 "재시도 불필요"로 판정**하는 흐름을 검증했습니다.

### 4.3 양자화 허용 오차 (test-way.md §3.6, "측정하라, 가정하지 마라")

mlx-whisper large-v3-turbo **fp16 vs q4** CER 차이:

| 사이클 | clean 평균 Δ | noisy 평균 Δ | 종합 평균 Δ |
|---|---|---|---|
| v1 | 0.0%p | -0.02%p | -0.02%p |
| v2 | 0.0%p | +1.14%p (항목별 최대 +7.4%p) | +1.14%p |

→ 클린 음성에서는 **q4가 fp16과 동일한 정확도** (모델 크기 1.6GB→0.8GB, 절반).
단, **노이즈 구간에서는 q4가 최대 +7.4%p까지 악화**될 수 있어,
음질이 나쁜 환경에서는 fp16을 유지하고 노이즈 구간의 실측이 필요합니다.

### 4.4 initial_prompt 도메인 주입 (P5 대안, test-way.md §3.5/§3.9)

mlx-whisper turbo에 `initial_prompt="API, response, timeout, tokenizer, embedding,
gradient, memory, database"` 주입 전후 (code_switched 트랙):

| 항목 | v1 base → prompt | 항목 | v2 base → prompt |
|---|---|---|---|
| utt_cs_001 | 28.2% → **10.3%** | utt_cs_101 | 48.6% → **16.2%** |
| utt_cs_002 | 51.4% → **27.0%** | utt_cs_102 | 60.5% → 60.5% |
| utt_cs_003 | 65.0% → **42.5%** | utt_cs_103 | 88.0% → 88.0% |
| utt_cs_004 | 76.8% → **0.0%** | utt_cs_104 | 73.3% → **2.2%** |

→ 도메인 용어가 실제 발화와 일치할 때 **큰 폭의 CER 개선**(최대 76.8%→0.0%).
한국어 파인튜닝 없이 무비용으로 도메인 주입이 가능함을 확인했습니다.
(용어가 전혀 들리지 않는 발화는 변화 없음.)

### 4.5 스트리밍 시뮬레이션 (P6, test-way.md §3.8)

1s 청크로 성장 버퍼를 재전사하는 방식. mlx-whisper turbo:

| 사이클 | 오디오 | first_partial_ms | final_latency_ms | RTF(재전사) |
|---|---|---|---|---|
| v1 | utt_001 (6.2s) | 1000 | 0 | 0.50 |
| v2 | utt_101 (5.8s) | 1000 | 0 | 0.46 |

→ 부분 결과가 1s 단위로 갱신되고, EoU 판정(침묵 2청크) 시 final 확정
(지연 0ms). 전체 재전사 방식은 벽시계 기준 실시간 처리(음성 시간의 ~50% 시간)로
동작. 프로덕션에서는 LocalAgreement-2 커밋으로 텍스트 깜빡임을 방지할 수 있습니다.

---

## 5. 해석 & 제한사항

### 5.1 엔진별 특성

| 엔진 | 강점 | 약점 |
|---|---|---|
| **mlx-whisper turbo** | 최고 정확도, 신뢰도 신호, 빠름(MLX) | 1.6GB(fp16)/0.8GB(q4) 모델 |
| openai-whisper (MPS) | 동일 정확도, 익숙한 API | ~1.5× 느림, ffmpeg 비의존 처리 필요 |
| sensevoice-small | 로드·추론 빠름(CPU RTF 0.1), `<\|nospeech\|>` 태그 | CER 5~8× 높음, 띄어쓰기 불안정, 신뢰도 신호 부재 |

### 5.2 코드 스위칭 CER이 높은 이유 (중요)

gTTS가 영어 단어를 **한국식 발음**으로 읽는 반면 정답은 영문 철자("API",
"tokenizer")라서, CER이 **구조적으로 과대 계상**됩니다. 즉 code_switched CER은
"모델 성능"보다 "TTS 발음 vs 정답 표기 불일치"가 지배적입니다. 실제 강의에서
교수님이 영어를 영어 발음으로 말한다면 CER은 크게 낮아질 것입니다. → 트랙
해석 시 clean/noisy를 주 신뢰 지표로, code_switched는 상대 비교용으로 사용.

### 5.3 제한사항 & 후속 작업

- **테스트 음성**: gTTS 합성음(스튜디오급 클린). 실제 강의·잡음·화자 변이는 미반영.
- **qwen3-asr**: transformers 버전 충돌로 제외. 별도 venv에서 `pip install qwen-asr`
  후 어댑터(`qwen3-asr`)로 바로 평가 가능.
- **DER(화자 분리)**: 다중 화자 트랙은 아직 미실행 (test-way.md §2.1 DER, P7).
- **Streaming 프로덕션화**: LocalAgreement-2 + partial→final 커밋 규칙 구현 필요(P6).
- **한국어 FT**: ghost613 모델의 MLX 변환(≈1.6GB)은 추후 선택 사항 — 지금은
  `initial_prompt`로 대체 검증 완료(§4.4).

---

## 6. 결론 (test-way.md §5 우선순위 대응)

1. **P0 (기본 파이프라인)**: Apple Silicon 기준은 `mlx-whisper large-v3-turbo`.
   `src/audio/asr_engine.py` 구현 시 `asr_adapters.MLXWhisperAdapter` 재사용 권장.
2. **P1 (환각 방어)**: 모든 엔진이 침묵에서 100% 환각 → VAD 게이트 필수 확인.
3. **P2 (2-Tier)**: 신뢰도 게이트 로직 검증 완료, 평균 지연 438ms로 예산 내.
4. **P3 (한국어 후처리)**: `initial_prompt` 도메인 주입으로 CER 대폭 개선 확인.
5. **P4 (대체 엔진)**: SenseVoice 등록·측정 완료(CTC 속도). qwen3-asr은 별도 환경에서 가능.
6. **P6 (스트리밍)**: 성장 버퍼 재전사 RTF 0.5, 1s partial 확인.
