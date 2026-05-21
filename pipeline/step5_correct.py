from __future__ import annotations
import json
from pathlib import Path
from typing import Literal

from models.chunk import Chunk


RuleType = Literal["merge_chunk", "split_chunk", "heading_fix", "retype_chunk"]

RULES_PATH = Path(__file__).parent.parent / "corrections" / "rules.json"


class Step5Correct:
    """
    사용자 보정 규칙 관리.
    규칙 추가·저장은 이 클래스에서, 적용은 Step3Chunk에서 수행한다.
    """

    def __init__(self, rules_path: Path = RULES_PATH):
        self._path = rules_path
        self._rules: list[dict] = self._load()

    @property
    def rules(self) -> list[dict]:
        return list(self._rules)

    def add_merge(self, chunk_ids: list[str]) -> dict:
        """두 개 이상의 chunk를 하나로 병합하는 규칙을 추가한다."""
        rule = {"rule": "merge_chunk", "target": chunk_ids}
        return self._add(rule)

    def add_heading_fix(self, text: str, level: int) -> dict:
        """특정 텍스트 블록의 heading 레벨을 수정하는 규칙을 추가한다."""
        rule = {"rule": "heading_fix", "text": text, "level": level}
        return self._add(rule)

    def add_retype(self, chunk_id: str, new_type: str) -> dict:
        """특정 chunk의 타입을 변경하는 규칙을 추가한다."""
        rule = {"rule": "retype_chunk", "chunk_id": chunk_id, "new_type": new_type}
        return self._add(rule)

    def remove(self, index: int) -> None:
        """인덱스로 규칙을 삭제한다."""
        if 0 <= index < len(self._rules):
            self._rules.pop(index)
            self._save()

    def clear(self) -> None:
        self._rules.clear()
        self._save()

    def list_rules(self) -> None:
        if not self._rules:
            print("저장된 보정 규칙이 없습니다.")
            return
        for i, rule in enumerate(self._rules):
            print(f"[{i}] {json.dumps(rule, ensure_ascii=False)}")

    # ------------------------------------------------------------------

    def _add(self, rule: dict) -> dict:
        self._rules.append(rule)
        self._save()
        return rule

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        data = json.loads(self._path.read_text(encoding="utf-8"))
        return data.get("rules", [])

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"rules": self._rules}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
