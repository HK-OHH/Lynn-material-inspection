from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


BlockType = Literal[
    "unknown", "heading", "paragraph", "list_item", "table", "caption", "footnote", "image"
]


@dataclass
class RawBlock:
    """Step 1 출력: OCR 결과 원본 블록."""
    block_id: str
    page: int
    bbox: list[float]          # [x0, y0, x1, y1] 픽셀 좌표
    text: str
    confidence: float
    block_type: BlockType = "unknown"
    flags: list[str] = field(default_factory=list)  # "low_confidence", "page_boundary"

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "page": self.page,
            "bbox": self.bbox,
            "text": self.text,
            "confidence": self.confidence,
            "block_type": self.block_type,
            "flags": self.flags,
        }

    @staticmethod
    def from_dict(d: dict) -> RawBlock:
        return RawBlock(
            block_id=d["block_id"],
            page=d["page"],
            bbox=d["bbox"],
            text=d["text"],
            confidence=d["confidence"],
            block_type=d.get("block_type", "unknown"),
            flags=d.get("flags", []),
        )


@dataclass
class LayoutBlock:
    """Step 2 출력: 레이아웃 분류가 완료된 블록."""
    block_id: str
    page: int
    block_type: BlockType
    level: int                 # heading 레벨 (1~4), 나머지는 0
    text: str
    bbox: list[float]
    confidence: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "page": self.page,
            "block_type": self.block_type,
            "level": self.level,
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "flags": self.flags,
        }

    @staticmethod
    def from_dict(d: dict) -> LayoutBlock:
        return LayoutBlock(
            block_id=d["block_id"],
            page=d["page"],
            block_type=d["block_type"],
            level=d.get("level", 0),
            text=d["text"],
            bbox=d["bbox"],
            confidence=d["confidence"],
            flags=d.get("flags", []),
        )
