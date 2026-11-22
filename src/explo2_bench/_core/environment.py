from __future__ import annotations

from dataclasses import dataclass

from ..baselines.base import Agent, AgentEpisodeFailure
from .types import Action, Coord, EpisodeInstance, Observation, StepRecord, Trajectory


@dataclass
class EnvState:
    t: int
    position: Coord
    discovered: set[str]
    activated: set[str]
    done: bool
    terminated_reason: str


class SymbolicEnvironment:
    def __init__(self, instance: EpisodeInstance, *, exploit_only: bool = False) -> None:
        self.instance = instance
        self.exploit_only = exploit_only
        self.indicator_to_visible = instance.indicator_to_visible()
        self.visible_to_indicator = instance.visible_to_indicator()
        self.goal_id = instance.task_graph.goal_id
        self.history: list[StepRecord] = []
        self.direct_ancestors: dict[str, set[str]] = {
            node_id: set() for node_id in self.instance.task_graph.nodes
        }
        for node_id, node in self.instance.task_graph.nodes.items():
            for prerequisite_option in node.prerequisites:
                for dep in prerequisite_option:
                    if dep in self.direct_ancestors:
                        self.direct_ancestors[dep].add(node_id)
        self.state = EnvState(
            t=0,
            position=instance.grid.start,
            discovered=set(),
            activated=set(),
            done=False,
            terminated_reason="",
        )

    def _state_positions_visible(self) -> dict[str, Coord]:
        positions: dict[str, Coord] = {}
        for coord, node_ids in self.instance.discovery_map.items():
            for node_id in node_ids:
                positions[self.indicator_to_visible[node_id]] = coord
        expected = set(self.indicator_to_visible.values())
        missing = expected - set(positions)
        if missing:
            raise RuntimeError(
                f"Exploit-only visibility requires locations for every symbolic state; missing={sorted(missing)}"
            )
        return dict(sorted(positions.items()))

    def _append_history(
        self,
        action_text: str,
        valid: bool,
        newly_discovered: set[str],
        newly_activated: set[str],
        note: str,
    ) -> None:
        self.history.append(
            StepRecord(
                t=self.state.t,
                position=self.state.position,
                action=action_text,
                valid=valid,
                newly_discovered=set(newly_discovered),
                newly_activated=set(newly_activated),
                discovered=set(self.state.discovered),
                activated=set(self.state.activated),
                note=note,
            )
        )

    @staticmethod
    def _action_text(action: Action) -> str:
        if action.kind == "move":
            return f"MOVE {action.value}"
        return action.kind.upper()

    def _reveal_at_current_position(self) -> set[str]:
        revealed = self.instance.discovery_map.get(self.state.position, set())
        return {state for state in revealed if state not in self.state.discovered}

    def _activate_at_position(self, position: Coord) -> set[str]:
        """Activate only node(s) revealed at the current position."""
        local_nodes = set(self.instance.discovery_map.get(position, set()))
        newly_activated: set[str] = set()
        while True:
            candidates: set[str] = set()
            for node_id in local_nodes:
                if node_id in self.state.activated or node_id not in self.state.discovered:
                    continue
                node = self.instance.task_graph.nodes[node_id]
                if not node.prerequisites:
                    candidates.add(node_id)
                    continue
                for prerequisite_option in node.prerequisites:
                    if prerequisite_option.issubset(self.state.activated):
                        candidates.add(node_id)
                        break
            if not candidates:
                break
            self.state.activated |= candidates
            newly_activated |= candidates
        return newly_activated

    def _discovery_payload(
        self,
        newly_discovered: set[str],
    ) -> tuple[dict[str, list[list[str]]], dict[str, list[str]]]:
        prereq_visible: dict[str, list[list[str]]] = {}
        for node_id in sorted(newly_discovered):
            visible_name = self.indicator_to_visible[node_id]
            node = self.instance.task_graph.nodes[node_id]
            prereq_visible[visible_name] = [
                [self.indicator_to_visible[dep] for dep in sorted(prerequisite_option)]
                for prerequisite_option in sorted(node.prerequisites, key=lambda s: (len(s), sorted(s)))
            ]
        return prereq_visible, self._ancestor_payload(newly_discovered)

    def _ancestor_payload(self, node_ids: set[str]) -> dict[str, list[str]]:
        ancestor_visible: dict[str, list[str]] = {}
        for node_id in sorted(node_ids):
            visible_name = self.indicator_to_visible[node_id]
            ancestor_visible[visible_name] = [
                self.indicator_to_visible[ancestor]
                for ancestor in sorted(self.direct_ancestors.get(node_id, set()))
            ]
        return ancestor_visible

    def _make_observation(
        self,
        newly_discovered: set[str],
        newly_activated: set[str],
    ) -> Observation:
        prereq_visible, ancestor_visible = self._discovery_payload(newly_discovered)
        activated_ancestor_visible = self._ancestor_payload(newly_activated)
        all_nodes = set(self.instance.task_graph.nodes) if self.exploit_only else set()
        all_prerequisites, all_ancestors = self._discovery_payload(all_nodes)
        return Observation(
            t=self.state.t,
            position=self.state.position,
            neighbors=sorted(self.instance.grid.neighbors(self.state.position)),
            newly_discovered_visible=sorted(
                self.indicator_to_visible[s] for s in newly_discovered
            ),
            newly_activated_visible=sorted(self.indicator_to_visible[s] for s in newly_activated),
            discovered_visible=sorted(self.indicator_to_visible[s] for s in self.state.discovered),
            activated_visible=sorted(self.indicator_to_visible[s] for s in self.state.activated),
            goal_visible=self.indicator_to_visible[self.goal_id],
            newly_discovered_prerequisites_visible=prereq_visible,
            newly_discovered_ancestors_visible=ancestor_visible,
            newly_activated_ancestors_visible=activated_ancestor_visible,
            exploit_only=self.exploit_only,
            grid_width=self.instance.grid.width if self.exploit_only else None,
            grid_height=self.instance.grid.height if self.exploit_only else None,
            open_cells=sorted(self.instance.grid.open_cells()) if self.exploit_only else [],
            state_positions_visible=(
                self._state_positions_visible() if self.exploit_only else {}
            ),
            all_prerequisites_visible=all_prerequisites,
            all_ancestors_visible=all_ancestors,
        )

    def reset(self) -> Observation:
        self.history.clear()
        self.state = EnvState(
            t=0,
            position=self.instance.grid.start,
            discovered=set(),
            activated=set(),
            done=False,
            terminated_reason="",
        )

        initial_discovered = (
            set(self.instance.task_graph.nodes)
            if self.exploit_only
            else self._reveal_at_current_position()
        )
        self.state.discovered |= initial_discovered
        initial_activated = self._activate_at_position(self.state.position)

        self._append_history(
            action_text="START",
            valid=True,
            newly_discovered=initial_discovered,
            newly_activated=initial_activated,
            note="initial state",
        )

        if self.goal_id in self.state.activated:
            self.state.done = True
            self.state.terminated_reason = "goal_activated_at_start"

        return self._make_observation(initial_discovered, initial_activated)

    def step(self, action: Action) -> Observation:
        if self.state.done:
            raise RuntimeError("Episode has already terminated; call reset() to start over.")

        self.state.t += 1
        valid = False
        newly_discovered: set[str] = set()
        newly_activated: set[str] = set()
        note = ""

        if action.kind == "move":
            try:
                x_str, y_str = action.value.split(",", maxsplit=1)
                target = (int(x_str), int(y_str))
            except Exception:
                target = None

            if target is not None and target in self.instance.grid.neighbors(self.state.position):
                self.state.position = target
                valid = True
                revealed = self._reveal_at_current_position()
                if revealed:
                    self.state.discovered |= revealed
                    newly_discovered = revealed
                newly_activated = self._activate_at_position(self.state.position)
            else:
                note = "invalid move"

        else:
            note = "unknown action"

        if self.goal_id in self.state.activated:
            self.state.done = True
            self.state.terminated_reason = "goal_activated"
        elif self.state.t >= self.instance.hidden_budget:
            self.state.done = True
            self.state.terminated_reason = "budget_exhausted"

        self._append_history(
            action_text=self._action_text(action),
            valid=valid,
            newly_discovered=newly_discovered,
            newly_activated=newly_activated,
            note=note,
        )

        return self._make_observation(newly_discovered, newly_activated)

    def run(self, agent: Agent) -> Trajectory:
        if hasattr(agent, "reset"):
            agent.reset()
        observation = self.reset()
        while not self.state.done:
            try:
                action = agent.act(observation)
            except AgentEpisodeFailure as exc:
                self.state.t += 1
                self.state.done = True
                self.state.terminated_reason = exc.reason
                self._append_history(
                    action_text="AGENT_FAILURE",
                    valid=False,
                    newly_discovered=set(),
                    newly_activated=set(),
                    note=f"{exc.reason}: {exc.detail}",
                )
                observation = self._make_observation(set(), set())
                break
            observation = self.step(action)

        success = self.goal_id in self.state.activated
        on_episode_end = getattr(agent, "on_episode_end", None)
        if callable(on_episode_end):
            on_episode_end(
                observation,
                success=success,
                terminated_reason=self.state.terminated_reason,
            )

        return Trajectory(
            records=list(self.history),
            success=success,
            terminated_reason=self.state.terminated_reason,
            total_steps=self.state.t,
        )
