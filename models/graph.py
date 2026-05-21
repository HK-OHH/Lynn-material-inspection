from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


RelationType = Literal["parent-child", "sequence"]


@dataclass
class Node:
    """Mind Map 노드."""
    node_id: str
    label: str
    level: int                     # heading 레벨 (1~4), 루트는 0
    chunk_refs: list[str]          # 연결된 chunk_id 목록
    page: int

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "level": self.level,
            "chunk_refs": self.chunk_refs,
            "page": self.page,
        }

    @staticmethod
    def from_dict(d: dict) -> Node:
        return Node(
            node_id=d["node_id"],
            label=d["label"],
            level=d["level"],
            chunk_refs=d["chunk_refs"],
            page=d["page"],
        )


@dataclass
class Edge:
    """Mind Map 엣지."""
    from_node: str
    to_node: str
    relation: RelationType

    def to_dict(self) -> dict:
        return {
            "from": self.from_node,
            "to": self.to_node,
            "relation": self.relation,
        }

    @staticmethod
    def from_dict(d: dict) -> Edge:
        return Edge(
            from_node=d["from"],
            to_node=d["to"],
            relation=d["relation"],
        )


@dataclass
class MindMap:
    """Step 4 출력: 전체 문서 그래프."""
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @staticmethod
    def from_dict(d: dict) -> MindMap:
        return MindMap(
            nodes=[Node.from_dict(n) for n in d.get("nodes", [])],
            edges=[Edge.from_dict(e) for e in d.get("edges", [])],
        )
