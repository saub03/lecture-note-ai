#!/usr/bin/env bash
# ── my-lab Apple Silicon 실행 환경 설치 ───────────────────────────
# 1) ffmpeg 확인 (mlx-whisper·sherpa-onnx 오디오 디코딩)
# 2) tier별 pip install (현재 Python 환경 — 3.12~3.14 권장)
# 3) aicc_env 가용성 리포트 + 테스트 오디오 생성
#
# 사용법:
#   bash setup_apple_silicon.sh          # 현재 환경에 전부 설치
#   bash setup_apple_silicon.sh core     # tier 선택 설치
set -euo pipefail
cd "$(dirname "$0")"

TIER="${1:-all}"

echo "── [1/4] ffmpeg 확인 ──────────────────────────────"
if command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"
else
    echo "ffmpeg 없음 → brew install ffmpeg 필요 (mlx-whisper/sherpa-onnx 오디오 디코딩)"
    if command -v brew >/dev/null 2>&1; then
        echo "  brew 가 있으므로 설치를 시도합니다..."
        brew install ffmpeg
    else
        echo "  ⚠️ brew 가 없습니다. https://brew.sh 설치 후 재실행하세요."
        exit 1
    fi
fi

echo "── [2/4] pip 패키지 설치 ───────────────────────────"
python3 - <<'PY'
import sys
print(f"Python {sys.version.split()[0]} ({sys.platform}, {sys.version_info})")
PY

install_tier() {
    local label="$1"; shift
    echo ""
    echo ">> [tier: $label] pip install $*"
    python3 -m pip install --upgrade "$@"
}

case "$TIER" in
  core)    install_tier core numpy jsonschema pytest matplotlib scipy soundfile ;;
  ml)      install_tier ml torch ;;
  asr)     install_tier asr mlx-whisper faster-whisper ;;
  llm)     install_tier llm mlx-lm ;;
  tts)     install_tier tts sherpa-onnx
           # MeloTTS는 fugashi 의존성 때문에 3.12/3.13에서만 — 실패해도 진행
           python3 -m pip install melotts || echo "  (melotts 생략 — Python 3.14에선 비지원, Supertonic이 한국어 대체)"
           ;;
  server)  install_tier server fastapi uvicorn websockets httpx aiortc nest-asyncio ;;
  cloning) install_tier cloning elevenlabs
           python3 -m pip install resemblyzer || echo "  (resemblyzer 생략 — Python 3.14에선 비지원)"
           ;;
  api)     install_tier api openai ;;
  all)
    install_tier core numpy jsonschema pytest matplotlib scipy soundfile
    install_tier ml torch
    install_tier asr mlx-whisper faster-whisper
    install_tier llm mlx-lm
    install_tier tts sherpa-onnx
    python3 -m pip install melotts || echo "  (melotts 생략 — 3.14 비지원)"
    install_tier server fastapi uvicorn websockets httpx aiortc nest-asyncio
    install_tier cloning elevenlabs
    python3 -m pip install resemblyzer || echo "  (resemblyzer 생략 — 3.14 비지원)"
    install_tier api openai
    ;;
  *) echo "모르는 tier: $TIER"; exit 1 ;;
esac

echo ""
echo "── [3/4] 가용성 확인 ──────────────────────────────"
python3 - <<'PY'
import aicc_env as ae
print(ae.engine_report())
PY

echo ""
echo "── [4/4] 테스트 오디오 생성 ───────────────────────"
python3 - <<'PY'
import aicc_env as ae
p = ae.ensure_test_audio()
print("테스트 오디오:", p, f"({p.stat().st_size:,} B)" if p.exists() else "생성 실패")
PY

echo ""
echo "완료 ✅  → my-lab/README_실행.md 에서 실행 가이드를 확인하세요."
echo "  (모델은 노트의 [REAL] 셀에서 최초 실행 시 자동 다운로드 — 약 5~7GB)"
