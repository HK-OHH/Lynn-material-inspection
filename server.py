"""
스마트 자재 검수 V4 — 온라인 배포용 파싱 서버
환경변수로 API 키 관리 (Railway 배포 기준)
"""
from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))

from models.document import PipelineMeta
from pipeline.step1_extract import Step1Extract
from pipeline.step2_layout import Step2Layout
from pipeline.step3_structure import Step3Structure

OUTPUT_ROOT = Path(__file__).parent / "output"


@dataclass
class EnvConfig:
    """환경변수에서 OCR 설정을 읽는 설정 객체"""
    ocr_engine: str
    clova_api_url: str
    clova_secret_key: str


def _load_env_config() -> EnvConfig:
    engine = os.getenv("OCR_ENGINE", "clova")
    url = os.getenv("CLOVA_OCR_URL", "")
    secret = os.getenv("CLOVA_OCR_SECRET", "")

    if engine == "clova" and (not url or not secret):
        print("[경고] CLOVA_OCR_URL 또는 CLOVA_OCR_SECRET 환경변수가 없습니다. EasyOCR로 폴백합니다.")
        engine = "easyocr"

    return EnvConfig(ocr_engine=engine, clova_api_url=url, clova_secret_key=secret)


app = FastAPI(title="스마트 자재 검수 파싱 서버 V4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_config = _load_env_config()
print(f"OCR 엔진: {_config.ocr_engine}")

_step2 = Step2Layout()
_step3 = Step3Structure()


@app.get("/")
def health():
    return {"status": "ok", "service": "스마트 자재 검수 파싱 서버 V4", "ocr": _config.ocr_engine}


@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    run_id = f"web_{uuid.uuid4().hex[:8]}"
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = out_dir / file.filename
    content = await file.read()
    tmp_path.write_bytes(content)

    try:
        meta = PipelineMeta(run_id=run_id, input_file=str(tmp_path), ocr_engine=_config.ocr_engine)
        step1 = Step1Extract(gpu=False, user_config=_config)
        raw_blocks, _, _ = step1.run(str(tmp_path), out_dir, meta)
        _step2.run(raw_blocks, out_dir)
        structured = _step3.run(raw_blocks, out_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파싱 오류: {e}")

    if isinstance(structured, list):
        structured = structured[0] if structured else {}

    return structured


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8888))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
