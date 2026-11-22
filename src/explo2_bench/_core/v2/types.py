from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..types import ConstructiveKnobs, Coord, GridSpec, TaskGraph, TaskNode

CONFIGURATION_VERSION = "v2"


class ExplorationLevel(str, Enum):
    HIGHER = "higher"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    LOWER = "lower"
    CUSTOM = "custom"


class TaskDAGDifficulty(str, Enum):
    HARDER = "harder"
    HARD = "hard"
    MEDIUM = "medium"
    EASY = "easy"
    EASIER = "easier"
    CUSTOM = "custom"


_EPISODE_KEYS = {
    "configuration_version",
    "episode_id",
    "exploration_level",
    "task_dag_difficulty",
    "seed",
    "knobs",
    "grid",
    "task_graph",
    "discovery_map",
    "required_nodes",
    "hidden_budget",
    "metadata",
}


@dataclass
class EpisodeInstance:
    episode_id: str
    exploration_level: ExplorationLevel
    task_dag_difficulty: TaskDAGDifficulty
    seed: int
    knobs: ConstructiveKnobs
    grid: GridSpec
    task_graph: TaskGraph
    discovery_map: dict[Coord, set[str]]
    required_nodes: set[str]
    hidden_budget: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def indicator_to_visible(self) -> dict[str, str]:
        return {
            node.indicator_id: node.visible_name
            for node in self.task_graph.nodes.values()
        }

    def visible_to_indicator(self) -> dict[str, str]:
        return {
            visible: indicator
            for indicator, visible in self.indicator_to_visible().items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_version": CONFIGURATION_VERSION,
            "episode_id": self.episode_id,
            "exploration_level": self.exploration_level.value,
            "task_dag_difficulty": self.task_dag_difficulty.value,
            "seed": self.seed,
            "knobs": self.knobs.to_dict(),
            "grid": self.grid.to_dict(),
            "task_graph": self.task_graph.to_dict(),
            "discovery_map": {
                f"{x},{y}": sorted(states)
                for (x, y), states in sorted(self.discovery_map.items())
            },
            "required_nodes": sorted(self.required_nodes),
            "hidden_budget": self.hidden_budget,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EpisodeInstance":
        if not isinstance(data, dict):
            raise TypeError(f"Episode data must be a dict, got {type(data).__name__}")

        missing = _EPISODE_KEYS - data.keys()
        unexpected = data.keys() - _EPISODE_KEYS
        if missing:
            raise ValueError(f"Missing v2 episode fields: {sorted(missing)}")
        if unexpected:
            raise ValueError(f"Unexpected v2 episode fields: {sorted(unexpected)}")
        if data["configuration_version"] != CONFIGURATION_VERSION:
            raise ValueError(
                "Expected configuration_version "
                f"{CONFIGURATION_VERSION!r}, got {data['configuration_version']!r}"
            )

        discovery_map_raw = data["discovery_map"]
        if not isinstance(discovery_map_raw, dict):
            raise TypeError(
                "discovery_map must be a dict[str, list[str]], got "
                f"{type(discovery_map_raw).__name__}"
            )
        discovery_map: dict[Coord, set[str]] = {}
        for coord_str, states in discovery_map_raw.items():
            if not isinstance(coord_str, str):
                raise TypeError("discovery_map coordinate keys must be strings")
            parts = coord_str.split(",")
            if len(parts) != 2:
                raise ValueError(f"Invalid discovery_map coordinate: {coord_str!r}")
            discovery_map[(int(parts[0]), int(parts[1]))] = set(states)

        metadata = data["metadata"]
        if not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(metadata).__name__}")

        return EpisodeInstance(
            episode_id=str(data["episode_id"]),
            exploration_level=ExplorationLevel(data["exploration_level"]),
            task_dag_difficulty=TaskDAGDifficulty(data["task_dag_difficulty"]),
            seed=int(data["seed"]),
            knobs=ConstructiveKnobs.from_dict(data["knobs"]),
            grid=GridSpec.from_dict(data["grid"]),
            task_graph=TaskGraph.from_dict(data["task_graph"]),
            discovery_map=discovery_map,
            required_nodes=set(data["required_nodes"]),
            hidden_budget=int(data["hidden_budget"]),
            metadata=dict(metadata),
        )


__all__ = [
    "CONFIGURATION_VERSION",
    "ConstructiveKnobs",
    "Coord",
    "EpisodeInstance",
    "ExplorationLevel",
    "GridSpec",
    "TaskDAGDifficulty",
    "TaskGraph",
    "TaskNode",
]
