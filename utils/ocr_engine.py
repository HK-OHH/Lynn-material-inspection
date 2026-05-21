from __future__ import annotations
import uuid
import json
import time
import requests
import numpy as np
from typing import Protocol
from PIL import Image


class OCRResult:
    def __init__(self, bbox: list[float], text: str, confidence: float):
        self.bbox = bbox
        self.text = text
        self.confidence = confidence


class OCRBackend(Protocol):
    def run(self, image: Image.Image) -> list[OCRResult]: ...


class EasyOCRBackend:
    """EasyOCR 기반 한글+영문 OCR."""

    LANGUAGES = ["ko", "en"]

    def __init__(self, gpu: bool = False):
        import easyocr
        self._reader = easyocr.Reader(self.LANGUAGES, gpu=gpu)

    def run(self, image: Image.Image) -> list[OCRResult]:
        img_array = np.array(image)
        raw_results = self._reader.readtext(img_array, detail=1)
        results = []
        for bbox_points, text, conf in raw_results:
            x_coords = [p[0] for p in bbox_points]
            y_coords = [p[1] for p in bbox_points]
            bbox = [float(min(x_coords)), float(min(y_coords)), float(max(x_coords)), float(max(y_coords))]
            results.append(OCRResult(bbox=bbox, text=text.strip(), confidence=float(conf)))
        return results


class ClovaOCRBackend:
    """Naver Clova OCR API 백엔드."""

    def __init__(self, api_url: str, secret_key: str):
        if not api_url or not secret_key:
            raise ValueError("Clova OCR API URL과 Secret Key가 필요합니다.")
        self._api_url = api_url
        self._secret_key = secret_key

    def run(self, image: Image.Image) -> list[OCRResult]:
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        payload = json.dumps({
            "images": [{"format": "png", "name": "page"}],
            "requestId": str(uuid.uuid4()),
            "version": "V2",
            "timestamp": int(time.time() * 1000),
        })
        headers = {"X-OCR-SECRET": self._secret_key}
        files = [("file", ("page.png", buf, "image/png"))]

        response = requests.post(
            self._api_url,
            headers=headers,
            data={"message": payload},
            files=files,
            timeout=30,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def _parse_response(self, data: dict) -> list[OCRResult]:
        results = []
        for image_result in data.get("images", []):
            for field in image_result.get("fields", []):
                vertices = field.get("boundingPoly", {}).get("vertices", [])
                if not vertices:
                    continue
                xs = [v.get("x", 0) for v in vertices]
                ys = [v.get("y", 0) for v in vertices]
                bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                text = field.get("inferText", "").strip()
                conf = float(field.get("inferConfidence", 1.0))
                if text:
                    results.append(OCRResult(bbox=bbox, text=text, confidence=conf))
        return results


class GoogleVisionBackend:
    """Google Cloud Vision API 백엔드."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Google Vision API Key가 필요합니다.")
        self._api_key = api_key

    def run(self, image: Image.Image) -> list[OCRResult]:
        import io, base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "requests": [{
                "image": {"content": b64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["ko", "en"]},
            }]
        }
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self._api_key}"
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return self._parse_response(response.json())

    def _parse_response(self, data: dict) -> list[OCRResult]:
        results = []
        for resp in data.get("responses", []):
            for page in resp.get("fullTextAnnotation", {}).get("pages", []):
                for block in page.get("blocks", []):
                    for para in block.get("paragraphs", []):
                        words = []
                        vertices = para.get("boundingBox", {}).get("vertices", [])
                        for word in para.get("words", []):
                            word_text = "".join(
                                s.get("text", "") for sym in word.get("symbols", [])
                                for s in [sym]
                            )
                            words.append(word_text)
                        text = " ".join(words).strip()
                        if not text or not vertices:
                            continue
                        xs = [v.get("x", 0) for v in vertices]
                        ys = [v.get("y", 0) for v in vertices]
                        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                        results.append(OCRResult(bbox=bbox, text=text, confidence=1.0))
        return results


class OCREngine:
    """OCR 엔진 래퍼. UserConfig를 받아 사용자별 백엔드를 선택한다."""

    LOW_CONFIDENCE_THRESHOLD = 0.7
    PAGE_FAIL_THRESHOLD = 0.6

    def __init__(self, backend: OCRBackend | None = None, gpu: bool = False):
        self._backend: OCRBackend = backend or EasyOCRBackend(gpu=gpu)

    @classmethod
    def from_user_config(cls, user_config, gpu: bool = False) -> OCREngine:
        """UserConfig로부터 적절한 백엔드를 선택해 OCREngine을 생성한다."""
        engine = user_config.ocr_engine

        if engine == "clova":
            backend = ClovaOCRBackend(
                api_url=user_config.clova_api_url,
                secret_key=user_config.clova_secret_key,
            )
        elif engine == "google":
            backend = GoogleVisionBackend(api_key=user_config.google_api_key)
        else:
            backend = EasyOCRBackend(gpu=gpu)

        return cls(backend=backend)

    def run_page(self, image: Image.Image) -> tuple[list[OCRResult], bool]:
        results = self._backend.run(image)

        if not results:
            return [], True

        success_count = sum(1 for r in results if r.confidence >= self.LOW_CONFIDENCE_THRESHOLD)
        success_rate = success_count / len(results)
        needs_retry = success_rate < self.PAGE_FAIL_THRESHOLD
        return results, needs_retry
