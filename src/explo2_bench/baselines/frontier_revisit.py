from .._core.types import Action, Observation
from .revisit_utils import RevisitMemory, nearest_target_step
from .helpers import exploration_step


class FrontierRevisitAgent:
    """Explore frontiers and greedily revisit the nearest discovered ready node."""

    def __init__(self) -> None:
        self.memory = RevisitMemory()

    def reset(self) -> None:
        self.memory.reset()

    def act(self, observation: Observation) -> Action:
        self.memory.update(observation)
        targets = {
            self.memory.discovered_at[node]
            for node in self.memory.activatable_pending_nodes(observation)
            if node in self.memory.discovered_at
            and self.memory.discovered_at[node] != observation.position
        }
        step = nearest_target_step(self.memory.known_neighbors, observation.position, targets)
        return Action.move(step if step is not None else exploration_step(self.memory, observation))
