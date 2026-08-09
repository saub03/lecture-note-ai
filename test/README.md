# test/ — 테스트 실행 방법

`src/audio/` ASR 엔진(`BaseASR` 어댑터 · `recorder`) 검증용 테스트.

## 테스트 방법 요약

| # | 방법 | 명령 | 필요 조건 | 목적 |
|---|---|---|---|---|
| 1 | Headless 단위 테스트 | `python -m pytest test/` | 없음 (모델·마이크 불필요) | 로직·데이터 계약 검증 (CI용) |
| 2 | 내 목소리 ASR (대화형) | `python test/audio/run_live.py` | 마이크 | 한 번 녹음 → 전사 결과 확인 |
| 3 | Live 테스트 (pytest) | `python -m pytest test/ -m live` | 실물 모델 (+ 마이크·샘플 wav) | 실물 모델 회귀 검증 |
| 4 | 실시간 스트리밍 ASR | `python test/audio/run_stream.py` | 마이크 + 실물 모델 | 말하며 실시간 전사 + 청크 실험 |

## 사전 준비

```bash
pip install -r requirements.txt    # pytest, faster-whisper, mlx-whisper, sounddevice, soundfile
```

## 1) Headless 단위 테스트 (모델·마이크 불필요)

```bash
python -m pytest test/
```

- `test/audio/test_asr_engine.py` — 정규화·데이터 계약·신뢰도·재시도·환각 필터·VAD·Mock 파이프라인
- `pytest.ini`의 `addopts = -m "not live"` 덕분에 `live` 테스트는 자동 제외됩니다.

## 2) 내 목소리로 ASR 테스트 (대화형)

```bash
python test/audio/run_live.py                    # mlx-whisper (기본, Apple Silicon)
ASR_PROVIDER=faster-whisper python test/audio/run_live.py   # CPU int8 교차 검증
```

- Enter → 녹음 시작, Enter → 녹음 정지 → 전사 결과·`confidence_ok`·`avg_logprob`·지연(ms)·RTF 출력
- `initial_prompt`로 샘플 강의 요약(`summarized_markdown` 대체)을 전달합니다.

## 3) Live 테스트 (실물 모델·마이크)

```bash
python -m pytest test/ -m live
```

- **파일 기반**: `test/audio/samples/*.wav` 전사 (wav를 폴더에 넣어 주세요)
- **마이크 기반**: `Recorder`(sounddevice)로 3초 녹음 후 전사
- 첫 실행은 모델 다운로드 포함 (기본 `large-v3-turbo` int8 ≈ 1.5GB)

## 4) 실시간 스트리밍 ASR (슬라이딩 윈도우)

```bash
python test/audio/run_stream.py                        # 60s · window 8 · step 2 (기본)
python test/audio/run_stream.py --window 4 --step 2    # 짧은 청크 — 문장 경계 절단 관찰
python test/audio/run_stream.py --window 15 --step 3 --duration 90   # 긴 청크 — 지연 증가
```

- 말하는 동안 `--step`(초)마다 마지막 `--window`(초)를 재전사 → 새로 확인된 텍스트만 터미널에 출력
- `[⟲ 수정]`: 이전에 출력한 구간이 재인식으로 달라진 경우
- 종료 시 **전체 버퍼를 일괄 전사**해 정확한 최종 전문을 출력 + 통계 (decode ms, RTF 평균/최대, 재인식 횟수)
- 스트리밍 pass는 best-effort — 문장 중간에서 window가 잘리면 누락/재인식이 생길 수 있고, 이는 청크 크기 실험으로 확인하는 부분입니다.
- 청크 실험: 같은 문장으로 `--window`를 2/4/8/15로 바꿔가며 비교. 짧으면 경계 절단으로 인식이 부정확해지고, 길면 pass당 decode 지연이 커집니다.

| `--window` | 예상 트레이드오프 |
|---|---|
| 2s | 경계 절단·오인식 최대 (문장 중간이 잘림) |
| 8s (기본) | 전사 정확도·지연 균형 — MLX RTF ≈ 0.3~0.6 |
| 15s+ | 정확하지만 pass당 수 초 지연 |

`--step ≤ --window` 여야 합니다 (기본 step 2s).

## 엔진·모델 변경

`.env` 또는 환경변수로 선택합니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `ASR_PROVIDER` | `mlx-whisper` | `mlx-whisper` (메인) / `faster-whisper` (교차 검증) |
| `ASR_MODEL` | (엔진별 기본값) | faster: `large-v3-turbo` · mlx: `mlx-community/whisper-large-v3-turbo` (HF 리포명) |
| `ASR_COMPUTE_TYPE` | `int8` | faster-whisper 양자화 |
| `ASR_DEVICE` | `cpu` | faster-whisper 장치 |
| `ASR_LANGUAGE` | `ko` | 전사 언어 |

`ASR_MODEL`을 비워두면 엔진별 기본 모델이 자동 선택됩니다. 주의: **mlx-whisper는 HF 리포명을 요구** — `ASR_MODEL=large-v3-turbo`처럼 faster-whisper용 모델명을 주면 401/RepositoryNotFound 오류가 납니다.

```bash
ASR_PROVIDER=mlx-whisper python test/audio/run_live.py   # mlx 기본 모델 자동 선택
```

## 비고

- `test/audio/samples/*.wav`는 `.gitignore` 처리된 에픽처 데이터입니다.
- 계약 dict 형식: `{engine, text, language, confidence_ok, avg_logprob, latency_ms}` — 노트북 01-ASR-review `CONTRACT_KEYS` 기준.
