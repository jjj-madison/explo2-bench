"""Exact task-DAG classes, ignoring names and preserving prerequisite choices.

Weisfeiler--Lehman colors only narrow the candidates for comparison. Every
class match is established by an exact, role-preserving directed isomorphism;
the colors are not canonical graph fingerprints.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib

from .._core.types import TaskGraph


@dataclass(frozen=True)
class _EncodedDAG:
    roles: tuple[str, ...]
    colors: tuple[bytes, ...]
    incoming: tuple[frozenset[int], ...]
    outgoing: tuple[frozenset[int], ...]

    @property
    def bucket_key(self) -> tuple:
        return tuple(sorted(Counter(zip(self.roles, self.colors)).items()))


def _encode(graph: TaskGraph) -> _EncodedDAG:
    node_ids = sorted(graph.nodes)
    indices = {node_id: index for index, node_id in enumerate(node_ids)}
    roles = ["goal" if node_id == graph.goal_id else "task" for node_id in node_ids]
    incoming: list[set[int]] = [set() for _ in node_ids]
    outgoing: list[set[int]] = [set() for _ in node_ids]

    # An option is an AND of parents; multiple options entering a task are OR.
    # Keeping option nodes prevents flattening distinct prerequisite formulas.
    for node_id in node_ids:
        child = indices[node_id]
        for parents in graph.nodes[node_id].prerequisites:
            option = len(roles)
            roles.append("option")
            incoming.append({indices[parent] for parent in parents})
            outgoing.append({child})
            incoming[child].add(option)
            for parent in incoming[option]:
                outgoing[parent].add(option)

    colors = [hashlib.sha256(role.encode()).digest() for role in roles]
    for _ in range(6):
        refined = []
        for color, parents, children in zip(colors, incoming, outgoing):
            digest = hashlib.sha256(color)
            for neighbors in (parents, children):
                digest.update(len(neighbors).to_bytes(8, "big"))
                for neighbor_color in sorted(colors[index] for index in neighbors):
                    digest.update(neighbor_color)
            refined.append(digest.digest())
        colors = refined

    return _EncodedDAG(
        tuple(roles), tuple(colors),
        tuple(frozenset(nodes) for nodes in incoming),
        tuple(frozenset(nodes) for nodes in outgoing),
    )


def _isomorphic(left: _EncodedDAG, right: _EncodedDAG) -> bool:
    if len(left.roles) != len(right.roles):
        return False

    def labels(graph: _EncodedDAG) -> list[tuple]:
        return [
            (role, color, len(parents), len(children))
            for role, color, parents, children in zip(
                graph.roles, graph.colors, graph.incoming, graph.outgoing
            )
        ]

    left_labels, right_labels = labels(left), labels(right)
    if Counter(left_labels) != Counter(right_labels):
        return False
    right_classes: dict[tuple, list[int]] = defaultdict(list)
    for index, label in enumerate(right_labels):
        right_classes[label].append(index)
    candidates = [right_classes[label] for label in left_labels]
    mapping: dict[int, int] = {}
    used: set[int] = set()

    def visit() -> bool:
        if len(mapping) == len(left.roles):
            return True

        best = -1
        options: list[int] | None = None
        for source in range(len(left.roles)):
            if source in mapping:
                continue
            valid = []
            for target in candidates[source]:
                if target in used:
                    continue
                if any(
                    ((old_source in left.incoming[source]) != (old_target in right.incoming[target]))
                    or ((old_source in left.outgoing[source]) != (old_target in right.outgoing[target]))
                    for old_source, old_target in mapping.items()
                ):
                    continue
                valid.append(target)
            if not valid:
                return False
            if options is None or len(valid) < len(options):
                best, options = source, valid
            if len(options) == 1:
                break

        assert options is not None
        for target in options:
            mapping[best] = target
            used.add(target)
            if visit():
                return True
            used.remove(target)
            del mapping[best]
        return False

    return visit()


class DAGRegistry:
    """Assign first-seen, zero-based class IDs using exact DAG isomorphism.

    Task names, IDs, depths, and spatial locations do not define a class.
    The distinguished goal and every AND/OR prerequisite group do. IDs are
    local to this registry and depend on insertion order, not graph hashes.
    """

    def __init__(self) -> None:
        self._representatives: list[_EncodedDAG] = []
        self._buckets: dict[tuple, list[int]] = defaultdict(list)
        self._labeled_cache: dict[tuple, int] = {}

    def __len__(self) -> int:
        return len(self._representatives)

    def add(self, graph: TaskGraph) -> int:
        signature = (
            graph.goal_id,
            tuple(
                (node_id, tuple(sorted(tuple(sorted(option)) for option in node.prerequisites)))
                for node_id, node in sorted(graph.nodes.items())
            ),
        )
        if signature in self._labeled_cache:
            return self._labeled_cache[signature]

        encoded = _encode(graph)
        bucket = self._buckets[encoded.bucket_key]
        for class_id in bucket:
            if _isomorphic(encoded, self._representatives[class_id]):
                self._labeled_cache[signature] = class_id
                return class_id

        class_id = len(self._representatives)
        self._representatives.append(encoded)
        bucket.append(class_id)
        self._labeled_cache[signature] = class_id
        return class_id


__all__ = ["DAGRegistry"]
