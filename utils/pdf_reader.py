from __future__ import annotations
from pathlib import Path
import fitz  # pymupdf


class PDFReader:
    """PDF 파일을 열고 페이지별 이미지와 메타데이터를 추출한다."""

    DEFAULT_DPI = 300
    LOW_DPI_THRESHOLD = 150

    def __init__(self, pdf_path: str | Path, dpi: int = DEFAULT_DPI):
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi
        self._doc: fitz.Document | None = None

    def open(self) -> None:
        self._doc = fitz.open(str(self.pdf_path))

    def close(self) -> None:
        if self._doc:
            self._doc.close()
            self._doc = None

    def __enter__(self) -> PDFReader:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def page_count(self) -> int:
        self._require_open()
        return len(self._doc)

    def get_metadata(self) -> dict:
        """PDF 메타데이터를 반환한다."""
        self._require_open()
        meta = self._doc.metadata
        return {
            "author": meta.get("author", ""),
            "producer": meta.get("producer", ""),
            "creation_date": meta.get("creationDate", ""),
            "modification_date": meta.get("modDate", ""),
            "page_count": self.page_count,
            "title": meta.get("title", ""),
        }

    def get_page_image(self, page_number: int):
        """지정 페이지를 PIL Image로 반환한다. page_number는 0-indexed."""
        from PIL import Image
        import io

        self._require_open()
        page = self._doc[page_number]
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_bytes = pix.tobytes("png")
        return Image.open(io.BytesIO(img_bytes))

    def iter_pages(self):
        """(page_number, PIL Image) 순서로 모든 페이지를 순회한다."""
        for i in range(self.page_count):
            yield i, self.get_page_image(i)

    def estimate_dpi(self, page_number: int) -> float:
        """페이지 내 이미지 해상도를 추정해 실효 DPI를 반환한다."""
        self._require_open()
        page = self._doc[page_number]
        image_list = page.get_images(full=True)
        if not image_list:
            return self.dpi

        widths = []
        for img_info in image_list:
            xref = img_info[0]
            base_image = self._doc.extract_image(xref)
            widths.append(base_image["width"])

        avg_width = sum(widths) / len(widths)
        page_width_pt = page.rect.width
        return (avg_width / page_width_pt) * 72

    def is_low_resolution(self, page_number: int) -> bool:
        return self.estimate_dpi(page_number) < self.LOW_DPI_THRESHOLD

    def _require_open(self) -> None:
        if self._doc is None:
            raise RuntimeError("PDF가 열려있지 않습니다. open() 또는 with 구문을 사용하세요.")
