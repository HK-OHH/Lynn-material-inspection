from __future__ import annotations
import json
from pathlib import Path

from models.block import RawBlock
from models.document import PageImage, PipelineMeta
from utils.pdf_reader import PDFReader
from utils.preprocessor import Preprocessor
from utils.ocr_engine import OCREngine
from utils.table_detector import TableDetector


class Step1Extract:
    """
    Step 1: OCR + 이미지 + 메타데이터 추출.
    출력: step1_raw_ocr.json
    """

    LOW_CONFIDENCE = 0.7

    def __init__(
        self,
        ocr_engine: OCREngine | None = None,
        preprocessor: Preprocessor | None = None,
        table_detector: TableDetector | None = None,
        gpu: bool = False,
        user_config=None,
    ):
        if ocr_engine:
            self._ocr = ocr_engine
        elif user_config:
            self._ocr = OCREngine.from_user_config(user_config, gpu=gpu)
        else:
            self._ocr = OCREngine(gpu=gpu)
        self._preprocessor = preprocessor or Preprocessor()
        self._table_detector = table_detector or TableDetector()

    def run(
        self,
        pdf_path: str,
        output_dir: Path,
        pipeline_meta: PipelineMeta,
    ) -> tuple[list[RawBlock], list[PageImage], dict]:
        """
        PDF 전체를 처리하고 (raw_blocks, page_images, metadata)를 반환한다.
        중간 결과를 output_dir/step1_raw_ocr.json에 저장한다.
        """
        pdf_path = str(pdf_path)
        raw_blocks: list[RawBlock] = []
        page_images: list[PageImage] = []
        block_counter = 0

        with PDFReader(pdf_path) as reader:
            metadata = reader.get_metadata()

            for page_num in range(reader.page_count):
                is_low_res = reader.is_low_resolution(page_num)
                page_img = reader.get_page_image(page_num)

                processed_img = self._preprocessor.process(page_img, is_low_res=is_low_res)
                ocr_results, needs_retry = self._ocr.run_page(processed_img)

                if needs_retry:
                    pipeline_meta.errors.append(
                        f"page {page_num + 1}: OCR 성공률 낮음, 전처리 재시도"
                    )
                    processed_img = self._preprocessor.process(page_img, is_low_res=True)
                    ocr_results, _ = self._ocr.run_page(processed_img)

                # 표 감지: 표 영역의 bbox 수집
                table_bboxes = self._get_table_bboxes(pdf_path, page_num)

                for result in ocr_results:
                    block_counter += 1
                    block_id = f"B-{block_counter:04d}"
                    flags = []

                    if result.confidence < self.LOW_CONFIDENCE:
                        flags.append("low_confidence")

                    if self._is_in_table(result.bbox, table_bboxes):
                        flags.append("in_table")

                    block = RawBlock(
                        block_id=block_id,
                        page=page_num + 1,  # 1-indexed
                        bbox=result.bbox,
                        text=result.text,
                        confidence=result.confidence,
                        block_type="unknown",
                        flags=flags,
                    )
                    raw_blocks.append(block)

                # 이미지 영역 기록 (Vision 분석은 2단계로 유예)
                page_images.extend(
                    self._extract_image_refs(pdf_path, page_num)
                )

        pipeline_meta.completed_steps.append(1)
        self._save(raw_blocks, page_images, metadata, output_dir)
        return raw_blocks, page_images, metadata

    # ------------------------------------------------------------------

    def _get_table_bboxes(self, pdf_path: str, page_num: int) -> list[list[float]]:
        tables = self._table_detector.extract_tables(pdf_path, page_num)
        return [t.bbox for t in tables]

    def _is_in_table(self, bbox: list[float], table_bboxes: list[list[float]]) -> bool:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        for tb in table_bboxes:
            if tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]:
                return True
        return False

    def _extract_image_refs(self, pdf_path: str, page_num: int) -> list[PageImage]:
        """이미지 bbox만 기록한다. 분류는 2단계에서 수행."""
        import fitz
        images = []
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        for idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            if rects:
                r = rects[0]
                images.append(PageImage(
                    page=page_num + 1,
                    bbox=[r.x0, r.y0, r.x1, r.y1],
                    image_index=idx,
                ))
        doc.close()
        return images

    def _save(
        self,
        blocks: list[RawBlock],
        images: list[PageImage],
        metadata: dict,
        output_dir: Path,
    ) -> None:
        data = {
            "metadata": metadata,
            "blocks": [b.to_dict() for b in blocks],
            "images": [img.to_dict() for img in images],
        }
        path = output_dir / "step1_raw_ocr.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
