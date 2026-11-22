import random

from .._core.types import Action, Observation
from .revisit_utils import RevisitMemory, nearest_frontier_step
from .helpers import known_routes


class RandomRevisitAgent:
    """Choose a random ready target and follow it until reached or ineligible."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self.memory = RevisitMemory()
        self._target_node: str | None = None

    def reset(self) -> None:
        self._rng.seed(self._seed)
        self.memory.reset()
        self._target_node = None

    def act(self, observation: Observation) -> Action:
        self.memory.update(observation)
        routes = known_routes(self.memory, observation.position)
        eligible = {
            node: routes[self.memory.discovered_at[node]][1]
            for node in self.memory.activatable_pending_nodes(observation)
            if node in self.memory.discovered_at
            and self.memory.discovered_at[node] in routes
            and self.memory.discovered_at[node] != observation.position
        }
        if self._target_node not in eligible:
            self._target_node = self._rng.choice(sorted(eligible)) if eligible else None
        if self._target_node is not None:
            step = eligible[self._target_node]
            assert step is not None
            return Action.move(step)

        unvisited = sorted(neighbor for neighbor in observation.neighbors if neighbor not in self.memory.visited)
        if unvisited:
            return Action.move(self._rng.choice(unvisited))
        step = nearest_frontier_step(self.memory.known_neighbors, self.memory.visited, observation.position)
        if step is not None:
            return Action.move(step)
        if observation.neighbors:
            return Action.move(self._rng.choice(sorted(observation.neighbors)))
        raise RuntimeError(f"No valid MOVE action available at position {observation.position}.")
