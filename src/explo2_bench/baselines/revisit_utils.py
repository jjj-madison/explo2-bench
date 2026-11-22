from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from .._core.types import Coord, Observation


@dataclass
class RevisitMemory:
    known_neighbors: dict[Coord, set[Coord]] = field(default_factory=dict)
    visited: set[Coord] = field(default_factory=set)
    visit_count: dict[Coord, int] = field(default_factory=lambda: defaultdict(int))
    discovered_at: dict[str, Coord] = field(default_factory=dict)
    prerequisites: dict[str, list[set[str]]] = field(default_factory=dict)
    ancestors: dict[str, set[str]] = field(default_factory=dict)

    def reset(self) -> None:
        self.known_neighbors.clear()
        self.visited.clear()
        self.visit_count.clear()
        self.discovered_at.clear()
        self.prerequisites.clear()
        self.ancestors.clear()

    def update(self, observation: Observation) -> None:
        position = observation.position
        neighbors = set(observation.neighbors)

        self.visited.add(position)
        self.visit_count[position] += 1
        self.known_neighbors.setdefault(position, set()).update(neighbors)
        for neighbor in neighbors:
            self.known_neighbors.setdefault(neighbor, set()).add(position)

        for node in observation.newly_discovered_visible:
            self.discovered_at.setdefault(node, position)

        for node, prerequisite_options in observation.newly_discovered_prerequisites_visible.items():
            parsed = [set(prerequisite_option) for prerequisite_option in prerequisite_options]
            self.prerequisites[node] = parsed if parsed else [set()]

        for node, parents in observation.newly_discovered_ancestors_visible.items():
            self.ancestors.setdefault(node, set()).update(parents)

    def pending_nodes(self, observation: Observation) -> list[str]:
        activated = set(observation.activated_visible)
        return sorted(node for node in observation.discovered_visible if node not in activated)

    def activatable_pending_nodes(self, observation: Observation) -> list[str]:
        activated = set(observation.activated_visible)
        out: list[str] = []
        for node in self.pending_nodes(observation):
            prerequisite_options = self.prerequisites.get(node, [set()])
            if any(prerequisite_option.issubset(activated) for prerequisite_option in prerequisite_options):
                out.append(node)
        return out

    def goal_distance(self, node: str, goal_visible: str) -> int:
        if node == goal_visible:
            return 0
        queue: deque[tuple[str, int]] = deque([(node, 0)])
        visited = {node}
        while queue:
            current, dist = queue.popleft()
            for anc in sorted(self.ancestors.get(current, set())):
                if anc in visited:
                    continue
                if anc == goal_visible:
                    return dist + 1
                visited.add(anc)
                queue.append((anc, dist + 1))
        return 10**6


def _first_step(parents: dict[Coord, Coord | None], start: Coord, target: Coord) -> Coord | None:
    if target == start:
        return None
    node = target
    while parents[node] is not None and parents[node] != start:
        node = parents[node]  # type: ignore[assignment]
    if parents[node] is None:
        return node if node != start else None
    return node


def nearest_target_step(
    known_neighbors: dict[Coord, set[Coord]],
    start: Coord,
    targets: set[Coord],
) -> Coord | None:
    if not targets:
        return None
    if start in targets:
        return None

    queue: deque[Coord] = deque([start])
    parents: dict[Coord, Coord | None] = {start: None}
    while queue:
        current = queue.popleft()
        for nxt in sorted(known_neighbors.get(current, set())):
            if nxt in parents:
                continue
            parents[nxt] = current
            if nxt in targets:
                return _first_step(parents, start, nxt)
            queue.append(nxt)
    return None


def nearest_frontier_step(
    known_neighbors: dict[Coord, set[Coord]],
    visited: set[Coord],
    start: Coord,
) -> Coord | None:
    queue: deque[Coord] = deque([start])
    parents: dict[Coord, Coord | None] = {start: None}

    while queue:
        current = queue.popleft()
        unseen_neighbors = sorted(n for n in known_neighbors.get(current, set()) if n not in visited)
        if unseen_neighbors:
            if current == start:
                return unseen_neighbors[0]
            return _first_step(parents, start, current)

        for nxt in sorted(known_neighbors.get(current, set())):
            if nxt in parents:
                continue
            parents[nxt] = current
            queue.append(nxt)
    return None
