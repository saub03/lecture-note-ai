# ⚙️ my-lab 실물 실행 가이드 (Apple Silicon)

my-lab 노트는 기본적으로 **mock·계약·계측**만으로 돌아갑니다.
여기서 설명하는 준비를 마치면 각 노트 끝의 **[REAL] 셀**에서 **실물 모델**이 실제로 동작합니다.

---

## 1. 준비 (1회)

```bash
cd my-lab
bash setup_apple_silicon.sh        # ffmpeg 확인 + pip tier 전체 설치 + 가용성 리포트
```

설치되는 것 (tier별, 대략):

| tier | 패키지 | 용도 |
|---|---|---|
| core | numpy · jsonschema · pytest · matplotlib · scipy · soundfile | 모든 노트의 mock 셀 |
| ml | torch | 05 어텐션 스크래치 (MPS) |
| asr | mlx-whisper · faster-whisper | 01 전사 |
| llm | mlx-lm | 04 로컬 LLM |
| tts | sherpa-onnx (**Supertonic ko = 한국어 우선**) · melotts* | 02 합성 |
| server | fastapi · uvicorn · websockets · httpx · aiortc · nest-asyncio | 07 서버 |
| cloning | elevenlabs · resemblyzer* | 03 클로닝 |
| api | openai | API 경로 |

> \* `melotts`·`resemblyzer` 는 `fugashi`/`webrtcvad` 의존성 때문에 **Python 3.12/3.13 전용**입니다.
> Python 3.14에선 설치가 실패해도 노트가 자동으로 스킵하고 **Supertonic(ko)** / ElevenLabs 로 대체됩니다.
> MeloTTS를 꼭 쓰려면 Python 3.12/3.13 가상환경을 만들어 `pip install melotts` 하세요.

> **Python 버전**: 3.12~3.14 모두 지원 (cp314 arm64 휠 확인됨). 노트 metadata 는 3.13 기준.

## 2. 모델 다운로드 (~5~7GB, 최초 1회)

`[REAL]` 셀에서 **최초 실행 시 자동** 다운로드됩니다.

| 모델 | 크기 | 노트 |
|---|---|---|
| `mlx-community/whisper-large-v3-turbo` | ~1.6GB | 01 ASR |
| faster-whisper `large-v3-turbo`(int8) | ~3.2GB | 01 ASR 교차 |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | ~2.6GB | 04 LLM · 08 |
| sherpa-onnx Supertonic(ko) · Kokoro(en) · MeloTTS-KR | ~0.3GB | 02 TTS · 08 |

모델명 교체: `aicc_env.py` 의 `MODEL_CFG` 또는 환경변수
(`AICC_ASR_MLX`, `AICC_LLM_MLX`, ...) 로 변경.

> ❗ 빠른 테스트: `AICC_ASR_MLX=mlx-community/whisper-small` 같은 소형 모델로 바꾸면
> 다운로드·추론이 훨씬 가벼워집니다.

## 3. API 키 (.env)

핵심 3종만 필요할 때 등록:

```bash
cd my-lab
cp .env.example .env     # 없으면
nano .env
# OPENAI_API_KEY=sk-...
# OPENROUTER_API_KEY=...
# ELEVENLABS_API_KEY=...
```

키가 있는 항목만 해당 `[REAL]` 셀이 실행되고, 없으면 안내 후 스킵됩니다.
`.env` 는 `.gitignore` 에 포함되어 있어 커밋되지 않습니다.

| 키 | 용도 |
|---|---|
| `OPENAI_API_KEY` | 04/06 클라우드 LLM · function-calling |
| `OPENROUTER_API_KEY` | 04 리그전 (다중 모델) |
| `ELEVENLABS_API_KEY` | 03 감정 클로닝 (유료) |

## 4. 실행

각 노트를 Jupyter Lab/VS Code 로 열어 **위에서 아래로** 실행하면 됩니다.
`[REAL]` 셀만 유일하게 실물 모델·서버를 띄우므로 시간이 걸립니다.

| 노트 | [REAL] 셀 |
|---|---|
| 01-ASR | 2.12 mlx-whisper · faster-whisper 전사 + CER/지연 |
| 02-TTS | 2.7 MeloTTS · sherpa-onnx 합성 → 16k 계약 검증 |
| 03-Cloning | 2.7 ElevenLabs 클로닝(키) · resemblyzer 임베딩 |
| 04-LLM-1 | 2.7 mlx-lm(Qwen3-4B) INTENT_SCHEMA 실측 · API 리그 |
| 05-LLM-2 | MPS 가용성 + 미니 LM 실행 |
| 06-LLM-3 | OpenAI function-calling (키) |
| 07-통합 | uvicorn 서버 + REST/WS TTFA + WebRTC 루프백 |
| 08-설계 | 실물 라운드트립 → 4대 게이트 실측 |

## 5. 트러블슈팅

- **`import` 실패** → `aicc_env.engine_report()` 로 어떤 tier 가 비었는지 확인 → `bash setup_apple_silicon.sh <tier>`
- **모델 다운로드 실패** → 네트워크 확인, `~/.cache/huggingface` 캐시 삭제 후 재시도
- **WebRTC/서버** → 07 노트 3-2 가이드대로 `pip install aiortc nest-asyncio` 확인
- **한국어 TTS 품질** → MeloTTS(KR) 가 기본. sherpa-onnx Supertonic(ko) 로 교체 가능
