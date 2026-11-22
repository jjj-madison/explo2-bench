"""Routing over topology learned through ordinary observations."""

from collections import deque

from .._core.types import Coord, Observation
from .revisit_utils import RevisitMemory, nearest_frontier_step


def known_routes(memory: RevisitMemory, start: Coord) -> dict[Coord, tuple[int, Coord | None]]:
    """Return shortest known distance and first step, with stable path ties."""
    routes: dict[Coord, tuple[int, Coord | None]] = {start: (0, None)}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        distance, first_step = routes[current]
        for neighbor in sorted(memory.known_neighbors.get(current, set())):
            if neighbor not in routes:
                routes[neighbor] = (distance + 1, neighbor if current == start else first_step)
                queue.append(neighbor)
    return routes


def exploration_step(memory: RevisitMemory, observation: Observation) -> Coord:
    """Visit local frontiers, then distant frontiers, then the least visited neighbor."""
    unvisited = sorted(neighbor for neighbor in observation.neighbors if neighbor not in memory.visited)
    if unvisited:
        return unvisited[0]
    step = nearest_frontier_step(memory.known_neighbors, memory.visited, observation.position)
    if step is not None:
        return step
    if observation.neighbors:
        return min(
            observation.neighbors,
            key=lambda neighbor: (memory.visit_count.get(neighbor, 0), neighbor[1], neighbor[0]),
        )
    raise RuntimeError(f"No valid MOVE action available at position {observation.position}.")
