from __future__ import annotations
import json
from pathlib import Path

from models.block import LayoutBlock
from models.chunk import Chunk, ChunkType


MIN_CHARS = 30
MAX_CHARS = 600


class Step3Chunk:
    """
    Step 3: Semantic Chunk 생성.
    heading 계층 기준으로 블록을 묶고, 크기 기준으로 분할·병합한다.
    출력: step3_chunks.json
    """

    def run(
        self,
        layout_blocks: list[LayoutBlock],
        correction_rules: list[dict],
        output_dir: Path,
    ) -> list[Chunk]:
        chunks = self._build_chunks(layout_blocks)
        chunks = self._enforce_size_limits(chunks)
        chunks = self._apply_corrections(chunks, correction_rules)
        self._save(chunks, output_dir)
        return chunks

    # ------------------------------------------------------------------

    def _build_chunks(self, blocks: list[LayoutBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_counter = 0
        current_heading_id: str | None = None
        buffer: list[LayoutBlock] = []

        def flush_buffer():
            nonlocal chunk_counter
            if not buffer:
                return
            text = " ".join(b.text for b in buffer)
            pages = sorted({b.page for b in buffer})
            chunk_counter += 1
            chunk_type: ChunkType = (
                "list" if all(b.block_type == "list_item" for b in buffer)
                else "semantic"
            )
            chunks.append(Chunk.build(
                chunk_id=f"C-{chunk_counter:04d}",
                pages=pages,
                chunk_type=chunk_type,
                parent_heading=current_heading_id,
                text=text,
                source_blocks=[b.block_id for b in buffer],
            ))
            buffer.clear()

        for block in blocks:
            if block.block_type == "heading":
                flush_buffer()
                chunk_counter += 1
                chunks.append(Chunk.build(
                    chunk_id=f"C-{chunk_counter:04d}",
                    pages=[block.page],
                    chunk_type="semantic",
                    parent_heading=None,
                    text=block.text,
                    source_blocks=[block.block_id],
                ))
                current_heading_id = f"C-{chunk_counter:04d}"

            elif block.block_type == "table":
                flush_buffer()
                chunk_counter += 1
                chunks.append(Chunk.build(
                    chunk_id=f"C-{chunk_counter:04d}",
                    pages=[block.page],
                    chunk_type="table",
                    parent_heading=current_heading_id,
                    text=block.text,
                    source_blocks=[block.block_id],
                ))

            else:
                # paragraph / list_item → 버퍼에 누적
                if buffer and block.block_type != buffer[-1].block_type:
                    flush_buffer()
                buffer.append(block)

        flush_buffer()
        return chunks

    def _enforce_size_limits(self, chunks: list[Chunk]) -> list[Chunk]:
        """최소 미달 → 병합, 최대 초과 → 분할."""
        result = self._merge_small(chunks)
        result = self._split_large(result)
        return result

    def _merge_small(self, chunks: list[Chunk]) -> list[Chunk]:
        result: list[Chunk] = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if (
                chunk.char_count < MIN_CHARS
                and chunk.chunk_type == "semantic"
                and i + 1 < len(chunks)
                and chunks[i + 1].chunk_type == "semantic"
                and chunks[i + 1].parent_heading == chunk.parent_heading
            ):
                next_chunk = chunks[i + 1]
                merged_text = chunk.text + " " + next_chunk.text
                merged = Chunk.build(
                    chunk_id=chunk.chunk_id,
                    pages=sorted(set(chunk.pages + next_chunk.pages)),
                    chunk_type="semantic",
                    parent_heading=chunk.parent_heading,
                    text=merged_text,
                    source_blocks=chunk.source_blocks + next_chunk.source_blocks,
                )
                result.append(merged)
                i += 2
            else:
                result.append(chunk)
                i += 1
        return result

    def _split_large(self, chunks: list[Chunk]) -> list[Chunk]:
        result: list[Chunk] = []
        split_counter = 0
        for chunk in chunks:
            if chunk.char_count <= MAX_CHARS or chunk.chunk_type != "semantic":
                result.append(chunk)
                continue
            # 마침표 기준 분할
            sentences = self._split_sentences(chunk.text)
            buffer = ""
            for sentence in sentences:
                if len(buffer) + len(sentence) > MAX_CHARS and buffer:
                    split_counter += 1
                    result.append(Chunk.build(
                        chunk_id=f"{chunk.chunk_id}-S{split_counter}",
                        pages=chunk.pages,
                        chunk_type="semantic",
                        parent_heading=chunk.parent_heading,
                        text=buffer.strip(),
                        source_blocks=chunk.source_blocks,
                    ))
                    buffer = sentence
                else:
                    buffer += sentence
            if buffer.strip():
                split_counter += 1
                result.append(Chunk.build(
                    chunk_id=f"{chunk.chunk_id}-S{split_counter}",
                    pages=chunk.pages,
                    chunk_type="semantic",
                    parent_heading=chunk.parent_heading,
                    text=buffer.strip(),
                    source_blocks=chunk.source_blocks,
                ))
        return result

    def _split_sentences(self, text: str) -> list[str]:
        """마침표·느낌표·물음표 기준으로 분할한다."""
        import re
        parts = re.split(r"(?<=[.!?。])\s+", text)
        return [p + " " for p in parts if p.strip()]

    def _apply_corrections(
        self, chunks: list[Chunk], rules: list[dict]
    ) -> list[Chunk]:
        """사용자 보정 규칙을 순서대로 적용한다."""
        chunk_map = {c.chunk_id: c for c in chunks}

        for rule in rules:
            rule_type = rule.get("rule")

            if rule_type == "merge_chunk":
                targets = rule.get("target", [])
                if len(targets) >= 2 and all(t in chunk_map for t in targets):
                    base = chunk_map[targets[0]]
                    merged_text = " ".join(chunk_map[t].text for t in targets)
                    merged = Chunk.build(
                        chunk_id=base.chunk_id,
                        pages=sorted(set(p for t in targets for p in chunk_map[t].pages)),
                        chunk_type=base.chunk_type,
                        parent_heading=base.parent_heading,
                        text=merged_text,
                        source_blocks=[b for t in targets for b in chunk_map[t].source_blocks],
                    )
                    chunk_map[base.chunk_id] = merged
                    for t in targets[1:]:
                        chunk_map.pop(t, None)

            elif rule_type == "heading_fix":
                for chunk in chunk_map.values():
                    if chunk.text == rule.get("text"):
                        chunk.flags.append(f"heading_level_{rule.get('level', 1)}")

        # 원래 순서 유지
        return [chunk_map[c.chunk_id] for c in chunks if c.chunk_id in chunk_map]

    def _save(self, chunks: list[Chunk], output_dir: Path) -> None:
        data = {"chunks": [c.to_dict() for c in chunks]}
        path = output_dir / "step3_chunks.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
