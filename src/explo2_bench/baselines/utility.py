"""Two deterministic policies using only accumulated public observations.

Dependency greedy ranks ready tasks by immediate unlock count, then distance.
Balanced utility ranks tasks by (1 + unlocks)/(1 + distance), and frontiers by
(1 + unknown incident cells)/(1 + distance). Both coefficients are fixed at 1;
neither has been fitted to benchmark maps. Goal readiness overrides all scores.
Targets remain committed until reached/ineligible, except for a ready goal.
"""
from fractions import Fraction

from .._core.types import Action, Coord, Observation
from .helpers import exploration_step, known_routes
from .revisit_utils import RevisitMemory


def immediate_unlock_count(memory: RevisitMemory, observation: Observation, node: str) -> int:
    """Count observed, currently blocked tasks made ready by this activation.

An AND group must be completely satisfied; one satisfied OR group suffices.
Already ready/activated tasks and tasks with unknown prerequisite metadata do
not count. Each task counts at most once regardless of its number of options.
"""
    active = set(observation.activated_visible)
    after = active | {node}
    return sum(
        task != node
        and not any(option <= active for option in memory.prerequisites.get(task, []))
        and any(option <= after for option in memory.prerequisites.get(task, []))
        for task in observation.discovered_visible if task not in active
    )


def _adjacent(position: Coord) -> set[Coord]:
    x, y = position
    return {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}


class DependencyGreedyAgent:
    balanced = False

    def __init__(self) -> None:
        self.memory = RevisitMemory()
        self.reset()

    def reset(self) -> None:
        self.memory.reset()
        self._target: tuple[str, str | Coord] | None = None
        self.diagnostics = {"target_selections": 0, "committed_actions": 0}

    def act(self, observation: Observation) -> Action:
        self.memory.update(observation)
        routes = known_routes(self.memory, observation.position)
        active = set(observation.activated_visible)
        ready = {
            node: self.memory.discovered_at[node]
            for node in observation.discovered_visible if node not in active
            and node in self.memory.discovered_at
            and self.memory.discovered_at[node] in routes
            and self.memory.discovered_at[node] != observation.position
            and any(option <= active for option in self.memory.prerequisites.get(node, []))
        }
        frontier = {
            cell for cell in routes if cell not in self.memory.visited
        }
        eligible: dict[tuple[str, str | Coord], Coord] = {
            ("task", node): cell for node, cell in ready.items()
        }
        # The greedy policy explores only when no ready task is known. The
        # balanced policy compares both target families under one fixed score.
        if self.balanced or not ready:
            eligible.update({("frontier", cell): cell for cell in frontier})

        goal = ("task", observation.goal_visible)
        if goal in eligible:
            chosen = goal
        elif self._target in eligible:
            self.diagnostics["committed_actions"] += 1
            chosen = self._target
        elif eligible:
            known_open = set(self.memory.known_neighbors)
            known_blocked = set().union(*(
                _adjacent(cell) - self.memory.known_neighbors[cell]
                for cell in self.memory.visited
            ))

            def key(target: tuple[str, str | Coord]) -> tuple:
                cell = eligible[target]
                distance = routes[cell][0]
                is_task = target[0] == "task"
                if is_task:
                    unlocks = immediate_unlock_count(self.memory, observation, str(target[1]))
                    score = Fraction(1 + unlocks, 1 + distance) if self.balanced else unlocks
                else:
                    unknown = len(_adjacent(cell) - known_open - known_blocked)
                    score = Fraction(1 + unknown, 1 + distance) if self.balanced else 0
                # Stable ties: task before frontier, distance, task name, x/y.
                return (-score, not is_task, distance, str(target[1]) if is_task else "", cell)

            chosen = min(eligible, key=key)
        else:
            self._target = None
            return Action.move(exploration_step(self.memory, observation))

        if chosen != self._target:
            self.diagnostics["target_selections"] += 1
        self._target = chosen
        step = routes[eligible[chosen]][1]
        assert step is not None
        return Action.move(step)


class BalancedUtilityAgent(DependencyGreedyAgent):
    balanced = True
