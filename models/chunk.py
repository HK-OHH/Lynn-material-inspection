from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


ChunkType = Literal["semantic", "table", "list", "image_ref"]


@dataclass
class Chunk:
    """Step 3 출력: 의미 단위 청크."""
    chunk_id: str
    pages: list[int]
    chunk_type: ChunkType
    parent_heading: str | None     # 상위 heading의 block_id
    text: str
    char_count: int
    source_blocks: list[str]       # 원본 block_id 목록
    flags: list[str] = field(default_factory=list)

    @staticmethod
    def build(
        chunk_id: str,
        pages: list[int],
        chunk_type: ChunkType,
        parent_heading: str | None,
        text: str,
        source_blocks: list[str],
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            pages=pages,
            chunk_type=chunk_type,
            parent_heading=parent_heading,
            text=text,
            char_count=len(text),
            source_blocks=source_blocks,
        )

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "pages": self.pages,
            "chunk_type": self.chunk_type,
            "parent_heading": self.parent_heading,
            "text": self.text,
            "char_count": self.char_count,
            "source_blocks": self.source_blocks,
            "flags": self.flags,
        }

    @staticmethod
    def from_dict(d: dict) -> Chunk:
        return Chunk(
            chunk_id=d["chunk_id"],
            pages=d["pages"],
            chunk_type=d["chunk_type"],
            parent_heading=d.get("parent_heading"),
            text=d["text"],
            char_count=d["char_count"],
            source_blocks=d["source_blocks"],
            flags=d.get("flags", []),
        )
