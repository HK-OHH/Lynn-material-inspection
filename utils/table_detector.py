from __future__ import annotations
import pdfplumber
from PIL import Image


class TableCell:
    def __init__(self, row: int, col: int, text: str):
        self.row = row
        self.col = col
        self.text = text


class DetectedTable:
    def __init__(self, page: int, bbox: list[float], cells: list[list[str]]):
        self.page = page
        self.bbox = bbox
        self.cells = cells   # cells[row][col] = text

    def to_text(self) -> str:
        """표를 탭 구분 텍스트로 직렬화한다."""
        return "\n".join("\t".join(row) for row in self.cells)

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "bbox": self.bbox,
            "cells": self.cells,
            "text": self.to_text(),
        }


class TableDetector:
    """pdfplumber 기반 표 감지 및 구조화."""

    def extract_tables(self, pdf_path: str, page_number: int) -> list[DetectedTable]:
        """
        지정 페이지에서 표를 감지하고 구조화된 데이터를 반환한다.
        page_number는 0-indexed.
        """
        tables = []
        with pdfplumber.open(pdf_path) as pdf:
            if page_number >= len(pdf.pages):
                return tables

            page = pdf.pages[page_number]
            raw_tables = page.extract_tables()
            table_bboxes = self._get_table_bboxes(page)

            for i, raw_table in enumerate(raw_tables):
                cleaned = self._clean_cells(raw_table)
                bbox = table_bboxes[i] if i < len(table_bboxes) else [0, 0, 0, 0]
                tables.append(DetectedTable(
                    page=page_number,
                    bbox=bbox,
                    cells=cleaned,
                ))

        return tables

    def _get_table_bboxes(self, page) -> list[list[float]]:
        """각 표의 bounding box를 추출한다."""
        bboxes = []
        for table in page.find_tables():
            bbox = table.bbox
            bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
        return bboxes

    def _clean_cells(self, raw_table: list[list]) -> list[list[str]]:
        """None 셀을 빈 문자열로 변환하고 줄바꿈을 정리한다."""
        result = []
        for row in raw_table:
            cleaned_row = []
            for cell in row:
                text = (cell or "").replace("\n", " ").strip()
                cleaned_row.append(text)
            result.append(cleaned_row)
        return result
