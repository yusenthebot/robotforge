"""The behavior tree a script is running, as the viewer's tree panel shows it (see Sim.show_flow, robotforge.flow).

The board only holds what the script last said: the nodes (once), one status letter per node (. not run, r running,
s succeeded, f failed) and short notes. The viewer is sent the whole thing each time it changes: a tree is a few dozen
nodes.
"""

from __future__ import annotations

import threading
from typing import Any

MAX_NODES = 200
MAX_TEXT = 100  # characters of a node's name or note
STATUS_LETTERS = ".rsf"


def _clean_nodes(nodes: list[Any]) -> list[dict[str, Any]]:
    if len(nodes) > MAX_NODES:
        raise ValueError(f"a tree shown in the viewer has at most {MAX_NODES} nodes, this one {len(nodes)}")
    out = []
    for i, node in enumerate(nodes):
        if not isinstance(node, dict) or node.get("id") != i or not isinstance(node.get("parent"), int):
            raise ValueError(f'node {i} needs {{"id": {i}, "parent": <an id or -1>, "kind", "name"}}')
        if node["parent"] >= i or node["parent"] < -1:
            raise ValueError(f"node {i}: a node's parent comes before it in the list (or is -1 for the root)")
        kind, label = str(node.get("kind", "node"))[:20], str(node.get("name", ""))[:MAX_TEXT]
        out.append({"id": i, "parent": node["parent"], "kind": kind, "name": label})
    return out


class FlowBoard:
    """One tree at a time. `rev` goes up with every change, so a viewer sends only when it has moved."""

    def __init__(self) -> None:
        self.rev = 0
        self.name = ""
        self.nodes: list[dict[str, Any]] = []
        self.status = ""
        self.notes: dict[str, str] = {}
        self._lock = threading.Lock()

    def update(self, name: str, nodes: list[Any] | None, status: str, notes: dict[str, Any] | None) -> None:
        """Take what a script says about its tree. nodes=None keeps the nodes shown; [] takes the tree away."""
        with self._lock:
            new_nodes = self.nodes if nodes is None else _clean_nodes(nodes)
            if nodes is None and not self.nodes:
                raise ValueError("show_flow needs the nodes the first time (Tree.run sends them)")
            if len(status) != len(new_nodes) or any(ch not in STATUS_LETTERS for ch in status):
                raise ValueError(f"status is one of {STATUS_LETTERS!r} for each of the {len(new_nodes)} nodes")
            self.name = str(name)[:MAX_TEXT]
            self.nodes = new_nodes
            self.status = status
            self.notes = {str(k): str(v)[:MAX_TEXT] for k, v in (notes or {}).items() if str(k).isdigit()}
            self.rev += 1

    def clear(self) -> None:
        with self._lock:
            if self.nodes:
                self.name, self.nodes, self.status, self.notes = "", [], "", {}
                self.rev += 1

    def message(self) -> dict[str, Any]:
        """What the viewer is sent: {"type": "flow", "name", "nodes", "status", "notes"}; no nodes means no panel."""
        with self._lock:
            return {
                "type": "flow",
                "name": self.name,
                "nodes": self.nodes,
                "status": self.status,
                "notes": self.notes,
            }
