from __future__ import annotations
import json
from pathlib import Path

import networkx as nx

from models.chunk import Chunk
from models.graph import MindMap, Node, Edge


class Step4Graph:
    """
    Step 4: Mind Map / Graph 생성.
    Chunk 계층 관계를 노드-엣지로 변환한다.
    출력: step4_graph.json
    """

    def run(self, chunks: list[Chunk], output_dir: Path) -> MindMap:
        nodes, edges = self._build_nodes_and_edges(chunks)
        mindmap = MindMap(nodes=nodes, edges=edges)
        self._validate(mindmap)
        self._save(mindmap, output_dir)
        return mindmap

    # ------------------------------------------------------------------

    def _build_nodes_and_edges(
        self, chunks: list[Chunk]
    ) -> tuple[list[Node], list[Edge]]:
        nodes: list[Node] = []
        edges: list[Edge] = []
        node_counter = 0

        # heading chunk → 노드, 나머지 → 부모 노드의 chunk_refs에 추가
        heading_nodes: dict[str, str] = {}  # chunk_id → node_id

        prev_node_by_level: dict[int, str] = {}  # level → node_id

        for chunk in chunks:
            is_heading = (
                chunk.chunk_type == "semantic"
                and chunk.parent_heading is None
                and len(chunk.text) < 100
            )

            if is_heading:
                node_counter += 1
                node_id = f"N-{node_counter:04d}"
                level = self._infer_level_from_chunk(chunk)
                node = Node(
                    node_id=node_id,
                    label=chunk.text,
                    level=level,
                    chunk_refs=[chunk.chunk_id],
                    page=chunk.pages[0],
                )
                nodes.append(node)
                heading_nodes[chunk.chunk_id] = node_id

                # parent-child 엣지
                parent_level = level - 1
                while parent_level > 0:
                    if parent_level in prev_node_by_level:
                        edges.append(Edge(
                            from_node=prev_node_by_level[parent_level],
                            to_node=node_id,
                            relation="parent-child",
                        ))
                        break
                    parent_level -= 1

                # sequence 엣지 (같은 레벨 이전 노드와 연결)
                if level in prev_node_by_level:
                    edges.append(Edge(
                        from_node=prev_node_by_level[level],
                        to_node=node_id,
                        relation="sequence",
                    ))

                prev_node_by_level[level] = node_id
                # 하위 레벨 노드 초기화 (새 섹션 시작)
                for l in list(prev_node_by_level.keys()):
                    if l > level:
                        del prev_node_by_level[l]

            else:
                # 비heading chunk → 부모 노드의 chunk_refs에 추가
                parent_node_id = heading_nodes.get(chunk.parent_heading or "")
                if parent_node_id:
                    for node in nodes:
                        if node.node_id == parent_node_id:
                            node.chunk_refs.append(chunk.chunk_id)
                            break

        return nodes, edges

    def _infer_level_from_chunk(self, chunk: Chunk) -> int:
        """chunk flags에 heading_level이 있으면 사용, 없으면 1로 기본."""
        for flag in chunk.flags:
            if flag.startswith("heading_level_"):
                try:
                    return int(flag.split("_")[-1])
                except ValueError:
                    pass
        return 1

    def _validate(self, mindmap: MindMap) -> None:
        """networkx로 그래프 일관성을 검증한다."""
        G = nx.DiGraph()
        for node in mindmap.nodes:
            G.add_node(node.node_id)
        for edge in mindmap.edges:
            G.add_edge(edge.from_node, edge.to_node, relation=edge.relation)

        if not nx.is_directed_acyclic_graph(G):
            # 순환이 있으면 해당 엣지를 제거
            cycles = list(nx.simple_cycles(G))
            cycle_edges = {(c[i], c[i + 1]) for c in cycles for i in range(len(c) - 1)}
            mindmap.edges = [
                e for e in mindmap.edges
                if (e.from_node, e.to_node) not in cycle_edges
            ]

    def _save(self, mindmap: MindMap, output_dir: Path) -> None:
        path = output_dir / "step4_graph.json"
        path.write_text(
            json.dumps(mindmap.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
