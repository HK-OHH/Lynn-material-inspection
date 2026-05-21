from __future__ import annotations
import json
import re
from pathlib import Path

from models.block import RawBlock, LayoutBlock, BlockType


# 제목 패턴: "1.", "제1조", "1.1", "가." 등
HEADING_NUMBER_PATTERN = re.compile(
    r"^(\d+[\.\)]|제\d+조|[가-힣][\.\)]|\d+\.\d+)"
)
# 목록 항목 패턴
LIST_ITEM_PATTERN = re.compile(r"^[•·\-\*○◦▪▫]\s|^\d+\)\s|^[가나다라마바사]\.\s")

# heading에서 제외할 패턴 (이것들은 paragraph로 강제)
PARAGRAPH_FORCE_PATTERNS = [
    re.compile(r"\d{2,4}-\d{3,4}-\d{4}"),          # 전화번호
    re.compile(r"\d{3}-\d{2}-\d{5}"),               # 사업자등록번호
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+"),  # 이메일
    re.compile(r"^(www\.|http|ftp)", re.I),          # URL
    re.compile(r"\.(com|kr|net|org|co\.kr)$", re.I), # 도메인
    re.compile(r"^[\d\s\+\-\=\.\,\:\;\(\)\/\\]+$"), # 숫자·기호만
    re.compile(r"^.{1,2}$"),                         # 1~2글자
]


class Step2Layout:
    """
    Step 2: 레이아웃 재구성.
    위치·패턴 기반 규칙으로 block_type과 level을 결정한다.
    Clova OCR처럼 단어 단위로 반환하는 엔진을 위해 y좌표 기반 행 그룹핑을 수행한다.
    출력: step2_layout.json
    """

    HEADING_HEIGHT_RATIO = 1.4
    LEFT_MARGIN_RATIO = 0.15
    LINE_MERGE_Y_TOLERANCE = 0.5   # 블록 평균 높이의 이 배수 이내면 같은 행으로 판단
    LINE_MERGE_X_GAP_MAX = 1.2     # 블록 평균 너비의 이 배수 이상 x 간격이면 분리

    def run(
        self,
        raw_blocks: list[RawBlock],
        output_dir: Path,
        page_width: float = 2480.0,
    ) -> list[LayoutBlock]:
        if not raw_blocks:
            return []

        avg_height = self._avg_block_height(raw_blocks)

        # 1) y좌표 기반으로 같은 행 블록을 하나로 병합
        merged_blocks = self._merge_same_line(raw_blocks, avg_height)

        # 2) 각 블록을 heading/paragraph/table 등으로 분류
        layout_blocks = [
            self._classify(block, avg_height, page_width)
            for block in merged_blocks
        ]

        # 3) 페이지 경계 병합
        layout_blocks = self._merge_page_boundary(layout_blocks)
        self._save(layout_blocks, output_dir)
        return layout_blocks

    # ------------------------------------------------------------------
    # y좌표 기반 행 그룹핑
    # ------------------------------------------------------------------

    def _merge_same_line(
        self, blocks: list[RawBlock], avg_height: float
    ) -> list[RawBlock]:
        """
        같은 페이지에서 y 중심점이 가깝고 x 간격이 좁은 블록들을 하나의 행으로 병합한다.
        컬럼이 다른 블록(x 간격이 넓음)은 분리 유지한다.
        """
        if not blocks:
            return blocks

        y_tolerance = avg_height * self.LINE_MERGE_Y_TOLERANCE
        avg_width = self._avg_block_width(blocks)
        x_gap_max = avg_width * self.LINE_MERGE_X_GAP_MAX

        pages: dict[int, list[RawBlock]] = {}
        for b in blocks:
            pages.setdefault(b.page, []).append(b)

        result: list[RawBlock] = []
        for page_num in sorted(pages):
            page_blocks = sorted(pages[page_num], key=lambda b: (b.bbox[1], b.bbox[0]))
            lines = self._group_into_lines(page_blocks, y_tolerance)
            for line in lines:
                # 같은 행 내에서도 x 간격이 큰 구간은 다시 분리
                segments = self._split_by_x_gap(line, x_gap_max)
                for seg in segments:
                    if len(seg) == 1:
                        result.append(seg[0])
                    else:
                        result.append(self._merge_line_blocks(seg))

        return result

    def _group_into_lines(
        self, blocks: list[RawBlock], y_tolerance: float
    ) -> list[list[RawBlock]]:
        """y 중심점 기준으로 블록을 행 그룹으로 분류한다."""
        lines: list[list[RawBlock]] = []
        for block in blocks:
            y_center = (block.bbox[1] + block.bbox[3]) / 2
            placed = False
            for line in lines:
                line_y = sum((b.bbox[1] + b.bbox[3]) / 2 for b in line) / len(line)
                if abs(y_center - line_y) <= y_tolerance:
                    line.append(block)
                    placed = True
                    break
            if not placed:
                lines.append([block])
        for line in lines:
            line.sort(key=lambda b: b.bbox[0])
        return lines

    def _split_by_x_gap(
        self, blocks: list[RawBlock], x_gap_max: float
    ) -> list[list[RawBlock]]:
        """x 간격이 x_gap_max를 초과하는 지점에서 행을 분리한다."""
        if len(blocks) <= 1:
            return [blocks]
        segments: list[list[RawBlock]] = [[blocks[0]]]
        for i in range(1, len(blocks)):
            gap = blocks[i].bbox[0] - blocks[i - 1].bbox[2]
            if gap > x_gap_max:
                segments.append([blocks[i]])
            else:
                segments[-1].append(blocks[i])
        return segments

    def _merge_line_blocks(self, blocks: list[RawBlock]) -> RawBlock:
        """같은 행의 블록들을 공백으로 연결해 하나의 블록으로 만든다."""
        text = " ".join(b.text for b in blocks if b.text.strip())
        x0 = min(b.bbox[0] for b in blocks)
        y0 = min(b.bbox[1] for b in blocks)
        x1 = max(b.bbox[2] for b in blocks)
        y1 = max(b.bbox[3] for b in blocks)
        conf = min(b.confidence for b in blocks)
        flags = list({f for b in blocks for f in b.flags})
        return RawBlock(
            block_id=blocks[0].block_id,
            page=blocks[0].page,
            bbox=[x0, y0, x1, y1],
            text=text,
            confidence=conf,
            block_type="unknown",
            flags=flags,
        )

    # ------------------------------------------------------------------
    # 분류
    # ------------------------------------------------------------------

    def _classify(
        self, block: RawBlock, avg_height: float, page_width: float
    ) -> LayoutBlock:
        text = block.text.strip()
        height = block.bbox[3] - block.bbox[1]
        x0 = block.bbox[0]

        block_type: BlockType = "paragraph"
        level = 0

        if "in_table" in block.flags:
            block_type = "table"

        elif LIST_ITEM_PATTERN.match(text):
            block_type = "list_item"

        elif not self._is_forced_paragraph(text):
            # 높이 또는 좌측 정렬 기반 heading 후보
            if (
                height > avg_height * self.HEADING_HEIGHT_RATIO
                or x0 < page_width * self.LEFT_MARGIN_RATIO
            ) and len(text) < 80:
                block_type = "heading"
                level = self._infer_heading_level(text, height, avg_height)

            elif HEADING_NUMBER_PATTERN.match(text) and len(text) < 80:
                block_type = "heading"
                level = self._infer_heading_level(text, height, avg_height)

            elif text.endswith(":") and len(text) < 60:
                block_type = "heading"
                level = 3

        return LayoutBlock(
            block_id=block.block_id,
            page=block.page,
            block_type=block_type,
            level=level,
            text=text,
            bbox=block.bbox,
            confidence=block.confidence,
            flags=block.flags,
        )

    def _is_forced_paragraph(self, text: str) -> bool:
        """전화번호·이메일·URL 등 heading이 되면 안 되는 패턴을 판별한다."""
        for pattern in PARAGRAPH_FORCE_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _infer_heading_level(self, text: str, height: float, avg_height: float) -> int:
        ratio = height / avg_height if avg_height > 0 else 1.0
        if ratio >= 2.0:
            return 1
        if ratio >= 1.6:
            return 2
        match = re.match(r"^(\d+\.)+", text)
        if match:
            depth = match.group().count(".")
            return min(depth, 4)
        return 3

    # ------------------------------------------------------------------
    # 페이지 경계 병합
    # ------------------------------------------------------------------

    def _merge_page_boundary(self, blocks: list[LayoutBlock]) -> list[LayoutBlock]:
        if len(blocks) < 2:
            return blocks

        result = []
        i = 0
        while i < len(blocks):
            current = blocks[i]
            if (
                i + 1 < len(blocks)
                and current.block_type == "paragraph"
                and blocks[i + 1].page == current.page + 1
                and not current.text.endswith((".", "。", "!", "?", ":", ";"))
            ):
                next_block = blocks[i + 1]
                if next_block.block_type == "paragraph":
                    merged = LayoutBlock(
                        block_id=current.block_id,
                        page=current.page,
                        block_type="paragraph",
                        level=0,
                        text=current.text + " " + next_block.text,
                        bbox=current.bbox,
                        confidence=min(current.confidence, next_block.confidence),
                        flags=list(set(current.flags + next_block.flags + ["page_boundary_merged"])),
                    )
                    result.append(merged)
                    i += 2
                    continue
            result.append(current)
            i += 1
        return result

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------

    def _avg_block_height(self, blocks: list[RawBlock]) -> float:
        heights = [b.bbox[3] - b.bbox[1] for b in blocks if b.bbox[3] > b.bbox[1]]
        return sum(heights) / len(heights) if heights else 20.0

    def _avg_block_width(self, blocks: list[RawBlock]) -> float:
        widths = [b.bbox[2] - b.bbox[0] for b in blocks if b.bbox[2] > b.bbox[0]]
        return sum(widths) / len(widths) if widths else 50.0

    def _save(self, blocks: list[LayoutBlock], output_dir: Path) -> None:
        data = {"layout_blocks": [b.to_dict() for b in blocks]}
        path = output_dir / "step2_layout.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
