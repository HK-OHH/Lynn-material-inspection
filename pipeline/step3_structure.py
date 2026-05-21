from __future__ import annotations
import json
import os
from pathlib import Path

import anthropic

from models.block import RawBlock

EMPTY_SCHEMA: dict = {
    "document_type": "",
    "transaction_date": "",
    "document_number": "",
    "manufacturer": {
        "name": "",
        "brand": ""
    },
    "supplier": {
        "name": "",
        "registration_number": "",
        "address": "",
        "phone": ""
    },
    "buyer": {
        "name": "",
        "registration_number": "",
        "address": "",
        "phone": ""
    },
    "delivery_site": "",
    "items": [],
    "subtotal": None,
    "tax": None,
    "total_amount": None,
    "total_qty": None,
    "receiver_name": "",
    "notes": "",
}

_SYSTEM_PROMPT = """당신은 한국어 거래 문서(거래명세표, 납품서, 송장, 거래명세서 등) OCR 텍스트에서 구조화된 정보를 추출하는 전문가입니다.

규칙:
- 값이 없는 문자열 필드는 "", 숫자 필드는 null로 반환하세요.
- 제조사(manufacturer)는 문서의 '공급자' 란에 기재된 상호입니다.
- 납품회사(supplier)는 문서의 '공급받는자' 란에 기재된 상호입니다.
  - 공급받는자 정보가 없거나 공급자 정보만 있을 경우, supplier는 manufacturer와 동일한 값을 사용하세요.
  - 납품처(현장명·현장주소)는 납품회사가 아닙니다. delivery_site 필드에 별도 기재하세요.
- 품목의 품명과 규격을 최대한 정확하게 분리하세요.
- 숫자 필드(qty, unit_price, amount 등)는 콤마 제거 후 순수 숫자로 반환하세요.
- JSON만 반환, 마크다운 코드블록(```) 사용 금지.

출력 스키마:
{
  "document_type": "거래명세표|납품서|송장|거래명세서|세금계산서|견적서|영수증|기타",
  "transaction_date": "YYYY-MM-DD",
  "document_number": "전표번호 또는 출고번호",
  "manufacturer": {
    "name": "제조사 회사명 (제품을 만든 곳)",
    "brand": "제품 브랜드명 (예: SH, PPI, KCC 등)"
  },
  "supplier": {
    "name": "납품회사 상호명",
    "registration_number": "납품회사 사업자등록번호",
    "address": "납품회사 주소",
    "phone": "납품회사 전화번호"
  },
  "buyer": {
    "name": "공급받는자 상호명",
    "registration_number": "구매자 사업자등록번호",
    "address": "구매자 주소",
    "phone": "구매자 전화번호"
  },
  "delivery_site": "납품 현장명 또는 현장 주소",
  "items": [
    {
      "name": "품명",
      "spec": "규격 (크기·재질·등급 등)",
      "unit": "단위",
      "qty": 수량숫자,
      "unit_price": 단가숫자또는null,
      "amount": 금액숫자또는null,
      "notes": "품목별 비고"
    }
  ],
  "subtotal": 공급가액합계또는null,
  "tax": 세액또는null,
  "total_amount": 합계금액또는null,
  "total_qty": 총수량합계또는null,
  "receiver_name": "인수자 성명",
  "notes": "전체 비고"
}"""


class Step3Structure:
    """
    Step 3: Claude API를 이용한 구조화 필드 추출.
    출력: step3_structured.json
    """

    MODEL = "claude-haiku-4-5"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or self._load_key_file()
        self._client = anthropic.Anthropic(api_key=key)

    def _load_key_file(self) -> str | None:
        # 온라인 배포: ANTHROPIC_API_KEY 환경변수를 anthropic 라이브러리가 자동 인식
        return None

    def run(self, raw_blocks: list[RawBlock], output_dir: Path) -> dict:
        ocr_text = self._join_text(raw_blocks)
        structured = self._extract(ocr_text)
        self._save(structured, output_dir)
        return structured

    def _join_text(self, blocks: list[RawBlock]) -> str:
        lines = []
        current_page = None
        for b in sorted(blocks, key=lambda x: (x.page, x.bbox[1], x.bbox[0])):
            if b.page != current_page:
                if current_page is not None:
                    lines.append(f"--- 페이지 {b.page} ---")
                current_page = b.page
            if b.text.strip():
                lines.append(b.text.strip())
        return "\n".join(lines)

    def _extract(self, ocr_text: str) -> dict:
        try:
            response = self._client.messages.create(
                model=self.MODEL,
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": f"다음 OCR 텍스트에서 정보를 추출하세요:\n\n{ocr_text}",
                    }
                ],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return dict(EMPTY_SCHEMA)
        except Exception as e:
            import sys
            print(f"[Step3] Claude API 오류: {e}", file=sys.stderr)
            return dict(EMPTY_SCHEMA)

    def _save(self, structured: dict, output_dir: Path) -> None:
        path = output_dir / "step3_structured.json"
        path.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")
