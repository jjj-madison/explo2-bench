from __future__ import annotations

from collections import deque

from .._core.types import Action, Coord, Observation
from .revisit_utils import RevisitMemory, nearest_target_step
from .planning import _activate_local_pending_nodes
from .helpers import exploration_step


class OnlinePlannerAgent:
    """Plan shortest known goal routes, with bounded search and frontier fallback.

    The search uses only accumulated observations. ``max_search_states`` caps all
    generated states, including queued states, so both search work and its
    frontier memory are bounded. Cached plans retain expected activations;
    ordinary movement and correctly predicted activations do not cause replans.
    """

    def __init__(self, max_search_states: int = 10_000) -> None:
        if isinstance(max_search_states, bool) or not isinstance(max_search_states, int) or max_search_states < 1:
            raise ValueError("max_search_states must be a positive integer")
        self.max_search_states = max_search_states
        self.memory = RevisitMemory()
        self.reset()

    def reset(self) -> None:
        self.memory.reset()
        self._plan: deque[tuple[Coord, frozenset[str]]] = deque()
        self._plan_knowledge: tuple | None = None
        self._expected_position: Coord | None = None
        self._expected_activated: frozenset[str] | None = None
        self._failed_search_key: tuple | None = None
        self.diagnostics = {
            "plan_searches": 0,
            "search_expansions": 0,
            "search_states_generated": 0,
            "peak_search_states": 0,
            "search_limit_hits": 0,
            "cached_actions": 0,
            "fallback_actions": 0,
        }

    def _knowledge_key(self, observation: Observation) -> tuple:
        # Visit counts, position, and time do not change the planning model.
        return (
            observation.goal_visible,
            tuple(sorted(observation.discovered_visible)),
            tuple((cell, tuple(sorted(neighbors))) for cell, neighbors in sorted(self.memory.known_neighbors.items())),
            tuple(sorted(self.memory.discovered_at.items())),
            tuple(
                (node, tuple(sorted(tuple(sorted(option)) for option in options)))
                for node, options in sorted(self.memory.prerequisites.items())
            ),
        )

    def _search_plan(self, observation: Observation) -> deque[tuple[Coord, frozenset[str]]] | None:
        self.diagnostics["plan_searches"] += 1
        pending = tuple(
            node for node in self.memory.pending_nodes(observation)
            if node in self.memory.discovered_at
        )
        if observation.goal_visible not in pending:
            return None
        base_activated = set(observation.activated_visible)
        # Missing prerequisite metadata is unknown, never an unconditional root.
        prerequisites = {node: self.memory.prerequisites.get(node, []) for node in pending}
        possible = set(base_activated)
        while True:
            added = {
                node for node in pending if node not in possible
                and any(option.issubset(possible) for option in prerequisites[node])
            }
            if not added:
                break
            possible.update(added)
        if observation.goal_visible not in possible:
            return None

        pending_index = {node: index for index, node in enumerate(pending)}
        goal_mask = 1 << pending_index[observation.goal_visible]
        local_nodes: dict[Coord, list[str]] = {}
        for node in pending:
            local_nodes.setdefault(self.memory.discovered_at[node], []).append(node)
        cell_to_nodes = {cell: tuple(nodes) for cell, nodes in local_nodes.items()}
        start = (observation.position, 0)
        queue = deque([start])
        parents: dict[tuple[Coord, int], tuple[Coord, int] | None] = {start: None}
        self.diagnostics["search_states_generated"] += 1
        self.diagnostics["peak_search_states"] = max(self.diagnostics["peak_search_states"], 1)
        while queue:
            state = queue.popleft()
            self.diagnostics["search_expansions"] += 1
            position, mask = state
            for neighbor in sorted(self.memory.known_neighbors.get(position, set())):
                next_mask = _activate_local_pending_nodes(
                    neighbor, mask, base_activated=base_activated,
                    pending_nodes=pending, pending_index=pending_index,
                    cell_to_nodes=cell_to_nodes, prerequisites=prerequisites,
                )
                next_state = (neighbor, next_mask)
                if next_state in parents:
                    continue
                if len(parents) >= self.max_search_states:
                    self.diagnostics["search_limit_hits"] += 1
                    return None
                parents[next_state] = state
                self.diagnostics["search_states_generated"] += 1
                self.diagnostics["peak_search_states"] = max(
                    self.diagnostics["peak_search_states"], len(parents)
                )
                if next_mask & goal_mask:
                    path: deque[tuple[Coord, frozenset[str]]] = deque()
                    cursor = next_state
                    while cursor != start:
                        active = base_activated | {
                            node for index, node in enumerate(pending) if cursor[1] & (1 << index)
                        }
                        path.appendleft((cursor[0], frozenset(active)))
                        parent = parents[cursor]
                        assert parent is not None
                        cursor = parent
                    return path
                queue.append(next_state)
        return None

    def _take_planned_action(self) -> Action:
        position, active = self._plan.popleft()
        self._expected_position = position
        self._expected_activated = active
        return Action.move(position)

    def act(self, observation: Observation) -> Action:
        self.memory.update(observation)
        return self._act_from_memory(observation)

    def _act_from_memory(self, observation: Observation) -> Action:
        knowledge = self._knowledge_key(observation)
        active = frozenset(observation.activated_visible)
        if self._plan:
            if (
                knowledge == self._plan_knowledge
                and observation.position == self._expected_position
                and active == self._expected_activated
                and self._plan[0][0] in observation.neighbors
            ):
                self.diagnostics["cached_actions"] += 1
                return self._take_planned_action()
            self._plan.clear()

        search_key = (knowledge, active)
        if (
            observation.goal_visible in observation.discovered_visible
            and observation.goal_visible not in active
            and search_key != self._failed_search_key
        ):
            plan = self._search_plan(observation)
            if plan:
                self._plan = plan
                self._plan_knowledge = knowledge
                self._failed_search_key = None
                return self._take_planned_action()
            # Moving within the same known component is not new evidence. After
            # a failed or capped search, let fallback progress until it learns
            # topology/tasks or changes the activation state.
            self._failed_search_key = search_key

        self.diagnostics["fallback_actions"] += 1
        ready_targets = {
            self.memory.discovered_at[node]
            for node in self.memory.pending_nodes(observation)
            if node in self.memory.discovered_at
            and self.memory.discovered_at[node] != observation.position
            and any(option.issubset(active) for option in self.memory.prerequisites.get(node, []))
        }
        step = nearest_target_step(self.memory.known_neighbors, observation.position, ready_targets)
        return Action.move(step if step is not None else exploration_step(self.memory, observation))
