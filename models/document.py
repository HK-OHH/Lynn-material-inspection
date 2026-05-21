from __future__ import annotations
from dataclasses import dataclass, field
from .block import RawBlock, LayoutBlock
from .chunk import Chunk
from .graph import MindMap


@dataclass
class PageImage:
    """페이지별 이미지 메타데이터."""
    page: int
    bbox: list[float]
    image_index: int           # 페이지 내 이미지 순번
    description: str | None = None   # 향후 Vision 분석 결과

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "bbox": self.bbox,
            "image_index": self.image_index,
            "description": self.description,
        }


@dataclass
class PipelineMeta:
    """파이프라인 실행 메타데이터."""
    run_id: str
    input_file: str
    completed_steps: list[int] = field(default_factory=list)
    ocr_engine: str = "easyocr"
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "input_file": self.input_file,
            "completed_steps": self.completed_steps,
            "ocr_engine": self.ocr_engine,
            "errors": self.errors,
        }


@dataclass
class Document:
    """최종 출력 문서."""
    metadata: dict
    raw_blocks: list[RawBlock] = field(default_factory=list)
    layout_blocks: list[LayoutBlock] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    mindmap: MindMap = field(default_factory=MindMap)
    images: list[PageImage] = field(default_factory=list)
    correction_rules: list[dict] = field(default_factory=list)
    pipeline_meta: PipelineMeta | None = None

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "chunks": [c.to_dict() for c in self.chunks],
            "mindmap": self.mindmap.to_dict() if self.mindmap else {},
            "images": [img.to_dict() for img in self.images],
            "correction_rules": self.correction_rules,
            "pipeline_meta": self.pipeline_meta.to_dict() if self.pipeline_meta else None,
        }
