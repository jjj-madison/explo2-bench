from .._core.types import Action, Observation
from .revisit_utils import RevisitMemory
from .helpers import exploration_step, known_routes


class GoalRevisitAgent:
    """Prioritize ready nodes by observed goal distance, then known travel distance."""

    def __init__(self) -> None:
        self.memory = RevisitMemory()

    def reset(self) -> None:
        self.memory.reset()

    def act(self, observation: Observation) -> Action:
        self.memory.update(observation)
        routes = known_routes(self.memory, observation.position)
        candidates = []
        for node in self.memory.activatable_pending_nodes(observation):
            target = self.memory.discovered_at.get(node)
            if target not in routes:
                continue
            distance, step = routes[target]
            if step is not None:
                candidates.append((self.memory.goal_distance(node, observation.goal_visible), distance, node, step))
        if candidates:
            return Action.move(min(candidates)[3])
        return Action.move(exploration_step(self.memory, observation))
