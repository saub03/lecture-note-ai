# 🧪 my-lab — 음성 에이전트 과정 복습·재사용 노트

> **생성형 AI 기반 음성 에이전트 개발 과정**의 실습 노트북(`ASR/`·`TTS/`·`Cloning/`·`LLM/`·`Modules/`)을
> 챕터별로 하나씩 통합한 **복습·재사용용 노트 8종**입니다.
>
> 🍎 각 노트 끝의 **`[REAL]` 셀**은 **Apple Silicon(M1~M4)에서 실물 모델을 실제로 실행**합니다
> (mlx-whisper · faster-whisper · mlx-lm Qwen3-4B · sherpa-onnx Supertonic · 서버/WebRTC · ElevenLabs).
> 준비: `bash setup_apple_silicon.sh` + `.env` 키 등록 → [`README_실행.md`](./README_실행.md)

## 이 폴더의 학습 순서 🗺️

파일 이름의 숫자가 **읽어야 할 순서**입니다. 음성 에이전트의 파이프라인 구성 요소대로
**귀(ASR) → 입(TTS) → 목소리(Cloning) → 뇌(LLM) → 조립(통합) → 설계(프로젝트)** 흐름을 따릅니다.

| 순서 | 파일 | 챕터 | 파이프라인 역할 | 한 줄 요약 |
|---|---|---|---|---|
| **01** | `01-ASR-review.ipynb` | 2장 (2주차) | 🎧 듣는 귀 | 발화를 텍스트로. 계약·신뢰도·VAD·CER |
| **02** | `02-TTS-review.ipynb` | 3장 (3주차) | 🗣️ 말하는 입 | 텍스트를 음성으로. 정규화·G2P·계약·p95 |
| **03** | `03-Cloning-review.ipynb` | 4장 (4주차) | 👤 목소리 복제 | 동의 계약·SECS+CER 이중 저울·화자 임베딩 |
| **04** | `04-LLM-1-thinking-review.ipynb` | 5장 — LLM 1권 | 🧠 사고부 | LLM 호출·JSON 계약·재시도·리그전 |
| **05** | `05-LLM-2-structure-review.ipynb` | 5장 — LLM 2권 | 🧠 모델 내부 | 어텐션 밑바닥·옴니·파인튜닝 |
| **06** | `06-LLM-3-tools-review.ipynb` | 5장 — LLM 3권 | 🧠 도구 손 | MCP·함수 호출·async·decorator |
| **07** | `07-system-integration.ipynb` | 6장 | 🔗 전체 조립 | 계약·오케스트레이터·서버(REST/WS/WebRTC)·pytest |
| **08** | `08-project-design.ipynb` | 7장 | 🎯 프로젝트 설계 | 4대 게이트·계약서(Charter)·설계 파생 |

> 💡 **빠른 길**: 처음이면 **01 → 02 → 04 → 07** 네 개만 순서대로 읽어도 전체 그림이 잡힙니다.
> (03 클로닝·05 구조·06 도구·08 설계는 선택 심화)

## 모든 노트북의 공통 구조 📐

8종 모두 같은 뼈대를 공유합니다. 구조를 알면 어느 노트든 바로 익숙해집니다.

| 장 | 내용 | 초보자용 장치 |
|---|---|---|
| **0** | 노트북 로드맵 + **용어 사전(0-A)** + 함수 지도(0-B) | 처음 보는 용어는 여기서 확인 |
| **1** | 실험에 필요한 **선행 지식** (개념의 *이유*) | 각 소주제에 원본 노트북 번호 표기 |
| **2** | **함수/클래스** 정의·주석 — ✅ 실행 코드 + 📄 요약 + 🍎 [REAL] 실물 실행 | 코드 셀마다 `assert` 자가 점검 + 초보자용 데모 |
| **3** | **실험 진행 방법** + macOS(Apple Silicon) 실행 가이드 | 노트북별 가이드·판단 기준 모음 |
| **4** | **효율적 설계를 위한 아키텍처** | 설계 원칙 요약 카드 |

## 사전 준비 🛠️

- **기본 실행 환경**: mock·계약·계측 셀은 **GPU·모델·API 키·네트워크 없이** 동작합니다 (macOS·Colab 공통).
  실물 모델은 시그니처+가이드(📄)로 요약되어 있습니다.
- **실물 실행 (Apple Silicon, 선택)**: `[REAL]` 셀에서 실제 모델이 돕니다.
  ```bash
  bash setup_apple_silicon.sh     # ffmpeg + tier별 pip 설치 (mlx-whisper, mlx-lm, sherpa-onnx, fastapi…)
  cp .env.example .env            # OPENAI/OPENROUTER/ELEVENLABS 키가 있다면 등록
  ```
  상세: [`README_실행.md`](./README_실행.md)
- **노트북 열기**: Jupyter Lab 또는 VS Code에서 열고 **위에서 아래로 순서대로 실행**하세요 (`Shift+Enter`).
- **기본 필요 패키지**: mock 셀은 아래 패키지에만 의존합니다.
  ```
  pip install numpy jsonschema pytest
  ```
  (모델 실측·서버 기동·pytest 스위트가 필요한 세션은 각 노트 3장 가이드 참조)

## 노트북 간 관계도 🔗

```
01 ASR ──┐                     (듣기 결과 = 텍스트 계약)
02 TTS ──┼──→ 07 시스템 통합 ──→ 08 프로젝트 설계
03 Cloning ┘      ↑                │
04~06 LLM(3권) ───┘                └─ 04 LLM 계약·TTS/Cloning 계약 재사용
```

- **계약(contract)이 모든 것을 잇습니다** — ASR·TTS·LLM 각각의 표준 레코드 형식이 과정 전체의 근간이며,
  07에서 어댑터로 조립되고 08에서 게이트로 검증됩니다.
- 08(프로젝트 설계)은 04 LLM 1권의 `INTENT_SCHEMA`·02 TTS 계약·03 Cloning 계약을 그대로 재사용합니다.
