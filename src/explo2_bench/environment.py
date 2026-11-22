"""Provider-independent agent interface with an explicit observation boundary.

Only observations returned here should be given to an evaluated policy. Python
object privacy is not a sandbox: agents with filesystem access can still read
public benchmark artifacts. Full maps and release metadata remain evaluator data.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import math
from typing import Any

from ._core.environment import SymbolicEnvironment
from ._core.types import Action, Observation
from .baselines.base import AgentEpisodeFailure

PROTOCOL_VERSION = "partial-v1"
_DIRECTIONS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
ACTION_SCHEMA = {
    "type": "object",
    "properties": {"move": {"type": "string", "enum": list(_DIRECTIONS)}},
    "required": ["move"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = (
    "Navigate a partially observed grid to activate the named goal. Every turn "
    "choose one legal cardinal move. Coordinates use up=(x,y+1), down=(x,y-1), "
    "left=(x-1,y), right=(x+1,y). Visiting a cell reveals its tasks. A task "
    "activates automatically at its cell when every prerequisite in at least "
    "one listed alternative is active. An empty prerequisite group is ready. "
    "Previously discovered tasks may need a revisit after their prerequisites "
    "activate. Success requires activating the goal. The action budget is hidden; "
    "invalid actions consume a turn. Return exactly one JSON object, for example "
    '{"move":"up"}, with no additional keys or text.'
)


def safe_observation(observation: Observation) -> dict[str, Any]:
    """Build a fresh allowlisted view; never serialize the episode or __dict__."""
    x, y = observation.position
    neighbors = set(observation.neighbors)
    # JSON roundtrip gives the policy independent nested lists and dictionaries.
    return json.loads(json.dumps({
        "t": observation.t,
        "position": [x, y],
        "legal_moves": [direction for direction, (dx, dy) in _DIRECTIONS.items()
                        if (x + dx, y + dy) in neighbors],
        "goal": observation.goal_visible,
        "newly_discovered": observation.newly_discovered_visible,
        "newly_activated": observation.newly_activated_visible,
        "discovered": observation.discovered_visible,
        "activated": observation.activated_visible,
        "newly_discovered_prerequisites": observation.newly_discovered_prerequisites_visible,
        "newly_discovered_successors": observation.newly_discovered_ancestors_visible,
        "newly_activated_successors": observation.newly_activated_ancestors_visible,
    }))


def observation_to_text(observation: Observation | Mapping[str, Any]) -> str:
    """Render public observations; discard extra mapping fields as a precaution."""
    if isinstance(observation, Observation):
        payload = safe_observation(observation)
    else:
        keys = (
            "t", "position", "legal_moves", "goal", "newly_discovered", "newly_activated",
            "discovered", "activated", "newly_discovered_prerequisites",
            "newly_discovered_successors", "newly_activated_successors",
        )
        payload = {key: observation[key] for key in keys if key in observation}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def action_from_response(response: Any, position: tuple[int, int]) -> Action:
    """Strict JSON/mapping parser; malformed output becomes one invalid action."""
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return Action("invalid", "")
    if not isinstance(response, Mapping) or set(response) != {"move"}:
        return Action("invalid", "")
    direction = response["move"]
    if not isinstance(direction, str) or direction not in _DIRECTIONS:
        return Action("invalid", "")
    dx, dy = _DIRECTIONS[direction]
    return Action.move((position[0] + dx, position[1] + dy))


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, Any]
    done: bool
    success: bool
    info: dict[str, Any]


class BenchmarkEnv:
    def __init__(self, instance: Any) -> None:
        self._env = SymbolicEnvironment(instance)

    @property
    def done(self) -> bool:
        return self._env.state.done

    @property
    def success(self) -> bool:
        return self._env.goal_id in self._env.state.activated

    def reset(self) -> dict[str, Any]:
        return safe_observation(self._env.reset())

    def step(self, action: Any) -> StepResult:
        parsed = action_from_response(action, self._env.state.position)
        observation = self._env.step(parsed)
        return StepResult(
            observation=safe_observation(observation), done=self.done, success=self.success,
            info={"valid": self._env.history[-1].valid,
                  "terminated_reason": self._env.state.terminated_reason},
        )


def make_env(record: Mapping[str, Any], *, action_budget: int | None = None,
             alpha: float | None = None) -> BenchmarkEnv:
    """Load a release record and privately set its action budget.

    The default preserves the native artifact budget. ``alpha`` instead uses
    floor(alpha * open_cells); an explicit integer budget is exact.
    The supplied record is not mutated. Reset preserves this runtime override.
    """
    from .data import episode_from_record

    if action_budget is not None and alpha is not None:
        raise ValueError("Pass action_budget or alpha, not both")
    instance = episode_from_record(record)
    if alpha is not None:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        action_budget = math.floor(alpha * len(instance.grid.open_cells()))
    if action_budget is not None:
        if isinstance(action_budget, bool) or not isinstance(action_budget, int) or action_budget < 1:
            raise ValueError("action_budget must be a positive integer")
        instance.hidden_budget = action_budget
    if instance.hidden_budget < 1:
        raise ValueError("The episode action budget must be positive")
    return BenchmarkEnv(instance)


class LMCallableAgent:
    """Adapt a provider-independent callback(public_dict) -> action JSON.

    No history truncation, provider retries, or fallback moves are implicit.
    The callback owns its history. If it has reset(), that is called per episode.
    Callback exceptions terminate the episode as callback_error, separately from
    malformed output, which consumes a normal invalid action.
    """
    def __init__(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        self.callback = callback
        self.diagnostics: dict[str, int] = {}

    def reset(self) -> None:
        self.diagnostics = {"callback_calls": 0, "malformed_actions": 0}
        reset = getattr(self.callback, "reset", None)
        if callable(reset):
            reset()

    def act(self, observation: Observation) -> Action:
        self.diagnostics["callback_calls"] = self.diagnostics.get("callback_calls", 0) + 1
        try:
            response = self.callback(safe_observation(observation))
        except Exception as error:
            raise AgentEpisodeFailure("callback_error", type(error).__name__) from error
        action = action_from_response(response, observation.position)
        if action.kind == "invalid":
            self.diagnostics["malformed_actions"] = self.diagnostics.get("malformed_actions", 0) + 1
        return action


__all__ = ["ACTION_SCHEMA", "SYSTEM_PROMPT", "PROTOCOL_VERSION", "BenchmarkEnv", "StepResult",
           "make_env", "safe_observation", "observation_to_text", "action_from_response", "LMCallableAgent"]
