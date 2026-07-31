lecture-note-ai/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .env.example
├── 📄 .gitignore
│
├── 🚀 app.py                   # Streamlit 엔트리포인트 (메인 실행 파일)
│
├── ⚙️ config/                  # 환경 변수 및 모델 경로 설정
│   ├── __init__.py
│   └── settings.py
│
├── 🧠 src/                     # 핵심 로직 모듈 (Core Logic)
│   ├── __init__.py
│   │
│   ├── 🎙️ audio/               # 오디오 입력 및 로컬 ASR 엔진
│   │   ├── __init__.py
│   │   ├── recorder.py         # 마이크 스트리밍 / 오디오 버퍼 관리
│   │   └── asr_engine.py       # 로컬 ASR (Faster-Whisper / Whisper.cpp)
│   │
│   ├── 📄 document/            # 강의 교안(PDF/PPT) 전처리
│   │   ├── __init__.py
│   │   ├── parser.py           # 문서 텍스트 추출
│   │   └── summarizer.py       # 교안 사전 요약 및 Context 생성
│   │
│   ├── 🤖 llm/                 # LLM 파이프라인 (API / Local 투트랙)
│   │   ├── __init__.py
│   │   ├── base.py             # LLM 공통 인터페이스 (추상 클래스)
│   │   ├── api_client.py       # API LLM (OpenAI, Gemini 등)
│   │   ├── local_client.py     # Local LLM (Ollama, vLLM 등)
│   │   └── factory.py          # 설정에 따른 LLM 동적 생성
│   │
│   └── ⚡ pipeline/           # 실시간 통합 파이프라인
│       ├── __init__.py
│       └── stream_processor.py # ASR 텍스트 + 교안 Context + LLM 연동
│
├── 🎨 ui/                      # UI 모듈화 (Streamlit 화면 구성)
│   ├── __init__.py
│   ├── components.py           # 녹음 버튼, 파일 업로더, 노트 출력창 등
│   └── styles.css              # 커스텀 웹뷰 스타일
│
└── 💾 data/                    # 임시 파일 저장소 (.gitignore 처리)
    ├── uploads/                # 업로드된 교안 파일
    └── temp_audio/             # 실시간 오디오 버퍼 파일