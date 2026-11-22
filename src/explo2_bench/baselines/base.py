from __future__ import annotations

from typing import Protocol

from .._core.types import Action, Observation


class AgentEpisodeFailure(RuntimeError):
    """Request a scored episode failure without crashing the surrounding suite."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


class Agent(Protocol):
    def act(self, observation: Observation) -> Action:
        ...

    def reset(self) -> None:
        ...
