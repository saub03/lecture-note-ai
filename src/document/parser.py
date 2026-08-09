"""Extract text from lecture slides (PDF/PPT)."""
import json
import requests
from pathlib import Path
from docling.document_converter import DocumentConverter

class LectureNoteToMarkdown:
    def __init__(self, ollama_model="qwen2.5:7b"):
        print("🚀 [1/2] Docling 로드중")
        # Docling 인스턴스를 메모리에 1회만 생성
        self.converter = DocumentConverter()
        
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = ollama_model
        print(f"🚀 [2/2] Ollama 로컬 LLM ({ollama_model}) 연동 완료.")

        # LaTeX 수식 깨짐 방지 및 교안 정리 전용 프롬프트
        self.system_prompt = "전달받은 교안 마크다운 문서를 읽고 오탈자 및 문장 구조를 교정하고, 수식 및 표기법을 일관되게 정리하여 최종적으로 깔끔한 마크다운 문서로 재작성하라. Latex 수식은 그대로 유지하고, 수식 내 변수명도 변경하지 마라. 표기법은 일관되게 통일하라."

    def process_folder(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 처리할 파일 목록 수집
        valid_extensions = {".pdf", ".pptx", ".docx", ".xlsx"}
        files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions] # 파일 확장자 추출해 모두 소문자로 변환 후 비교

        if not files:
            print("⚠️ 처리할 문서 파일이 없습니다.")
            return

        print(f"\n📦 총 {len(files)}개 문서 처리를 시작합니다.\n")

        for idx, file_path in enumerate(files, 1):
            print(f"[{idx}/{len(files)}] 파싱 중: {file_path.name}")
            
            # 1. Docling으로 마크다운 변환 (메모리 재로드 없음)
            result = self.converter.convert(str(file_path))
            raw_markdown = result.document.export_to_markdown()

            # 2. Ollama에 전달하여 재정리
            print(f"  └─ Ollama ({self.ollama_model}) 정제 작업 실행...")
            refined_markdown = self._call_ollama(raw_markdown)

            # 3. 결과 파일 저장
            output_file = output_path / f"{file_path.stem}_cleaned.md"
            output_file.write_text(refined_markdown, encoding="utf-8")
            print(f"  └─ ✅ 저장 완료: {output_file.name}\n")

    def _call_ollama(self, raw_text: str) -> str:
        prompt = f"{self.system_prompt}\n\n[입력 교안 텍스트]:\n{raw_text}"

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2  # 수식 및 서식 변형 최소화를 위해 낮게 설정
            },
            "keep_alive": -1  # LLM을 RAM/VRAM에서 내리지 않고 상주시킴
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=300)
            response.raise_for_status()
            markdown_response = response.json().get("response", "")
            self.summarized_markdown = self._summarize_markdown(markdown_response)  # 요약 프롬프트 호출
            return markdown_response
        except Exception as e:
            print(f"❌ Ollama 호출 실패: {e}")
            self.summarized_markdown = self._summarize_markdown(raw_text)  # 요약 프롬프트 호출
            return raw_text  # 실패 시 원본 마크다운 반환
        
    def unload_model(self):
        """모든 작업이 끝난 후 Ollama 모델을 RAM에서 즉시 해제"""
        print("🧹 Ollama 모델을 RAM에서 해제합니다...")
        requests.post(
            self.ollama_url,
            json={
                "model": self.ollama_model,
                "keep_alive": 0  # 0으로 세팅하면 즉시 RAM에서 내려감
            }
        )

    def _summarize_markdown(self, markdown_text: str) -> str:
        """ASR 모델에게 줄 요약 프롬프트 생성"""
        prompt = f"입력 마크다운 텍스트를 한 문장으로 요약해서 ASR모델에 전달할 프롬프트를 한국어로 작성해주세요\n\n[입력 마크다운 텍스트]:\n{markdown_text}"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            },
            "keep_alive": -1
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=300)
            response.raise_for_status()
            summarized = response.json().get("response", "")
            self.summarized_markdown = summarized
            return summarized
        except Exception as e:
            print(f"❌ Ollama 요약 호출 실패: {e}")
            self.summarized_markdown = None  # 실패 시 원본 마크다운 반환
            return None

if __name__ == "__main__":
    # 실행부
    pipeline = LectureNoteToMarkdown(ollama_model="qwen2.5:7b")

    try:
        # 입력 폴더와 출력 폴더 경로 지정
        pipeline.process_folder(
            input_dir="lab/parser-test",
            output_dir="lab/parser-test"
        )
        print(pipeline.summarized_markdown)  # 요약된 마크다운 출력
    finally:
        # 모든 작업이 끝난 후 Ollama 모델 해제
        pipeline.unload_model()