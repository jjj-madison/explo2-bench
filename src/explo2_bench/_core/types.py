from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

Coord = tuple[int, int]


class ExplorationLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CUSTOM = "custom"

    @classmethod
    def parse(cls, raw: str) -> "ExplorationLevel":
        return cls(raw.strip().lower())


class TaskDAGDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    CUSTOM = "custom"


@dataclass(frozen=True)
class TaskNode:
    indicator_id: str
    visible_name: str
    prerequisites: list[set[str]]
    depth: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "visible_name": self.visible_name,
            "prerequisites": [sorted(prerequisite_option) for prerequisite_option in self.prerequisites],
            "depth": self.depth,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TaskNode":
        return TaskNode(
            indicator_id=data["indicator_id"],
            visible_name=data["visible_name"],
            prerequisites=[set(prerequisite_option) for prerequisite_option in data.get("prerequisites", [])],
            depth=int(data["depth"]),
        )


@dataclass
class TaskGraph:
    nodes: dict[str, TaskNode]
    goal_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "nodes": {k: v.to_dict() for k, v in sorted(self.nodes.items())},
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TaskGraph":
        return TaskGraph(
            nodes={k: TaskNode.from_dict(v) for k, v in data["nodes"].items()},
            goal_id=data["goal_id"],
        )


@dataclass
class GridSpec:
    width: int
    height: int
    blocked: set[Coord]
    start: Coord

    def in_bounds(self, coord: Coord) -> bool:
        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height

    def is_open(self, coord: Coord) -> bool:
        return self.in_bounds(coord) and coord not in self.blocked

    def neighbors(self, coord: Coord) -> list[Coord]:
        x, y = coord
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in candidates if self.is_open(c)]

    def open_cells(self) -> list[Coord]:
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.blocked
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "blocked": [list(c) for c in sorted(self.blocked)],
            "start": list(self.start),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GridSpec":
        return GridSpec(
            width=int(data["width"]),
            height=int(data["height"]),
            blocked={tuple(c) for c in data.get("blocked", [])},
            start=tuple(data["start"]),
        )


@dataclass(frozen=True)
class ConstructiveKnobs:
    alpha_budget: float
    num_nodes: int
    dag_density_over_grid: float
    grid_aspect_ratio: float
    max_nodes_per_layer: int
    path_width_probs: dict[int, float]
    option_count_probs: dict[int, float]
    dependency_count_probs: dict[int, float]
    goal_dependency_count_probs: dict[int, float]
    parent_depth_bias: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_budget": self.alpha_budget,
            "num_nodes": self.num_nodes,
            "dag_density_over_grid": self.dag_density_over_grid,
            "grid_aspect_ratio": self.grid_aspect_ratio,
            "max_nodes_per_layer": self.max_nodes_per_layer,
            "path_width_probs": dict(sorted(self.path_width_probs.items())),
            "option_count_probs": dict(sorted(self.option_count_probs.items())),
            "dependency_count_probs": dict(sorted(self.dependency_count_probs.items())),
            "goal_dependency_count_probs": dict(sorted(self.goal_dependency_count_probs.items())),
            "parent_depth_bias": self.parent_depth_bias,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ConstructiveKnobs":
        allowed = {f.name for f in fields(ConstructiveKnobs)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        for field_name in (
            "path_width_probs",
            "option_count_probs",
            "dependency_count_probs",
            "goal_dependency_count_probs",
        ):
            if field_name in filtered:
                filtered[field_name] = ConstructiveKnobs._parse_prob_map(
                    filtered[field_name],
                    field_name,
                )
        return ConstructiveKnobs(**filtered)

    @staticmethod
    def _parse_prob_map(raw: Any, field_name: str) -> dict[int, float]:
        if not isinstance(raw, dict):
            raise TypeError(f"{field_name} must be a dict[int, float], got {type(raw).__name__}")
        parsed: dict[int, float] = {}
        for key, value in raw.items():
            parsed[int(key)] = float(value)
        return parsed

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
        return {v: k for k, v in self.indicator_to_visible().items()}

    def to_dict(self) -> dict[str, Any]:
        return {
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
        knobs = ConstructiveKnobs.from_dict(data["knobs"])
        discovery_map: dict[Coord, set[str]] = {}
        for coord_str, states in data["discovery_map"].items():
            x_str, y_str = coord_str.split(",", maxsplit=1)
            discovery_map[(int(x_str), int(y_str))] = set(states)
        return EpisodeInstance(
            episode_id=data["episode_id"],
            exploration_level=ExplorationLevel.parse(data["exploration_level"]),
            task_dag_difficulty=TaskDAGDifficulty(data["task_dag_difficulty"]),
            seed=int(data["seed"]),
            knobs=knobs,
            grid=GridSpec.from_dict(data["grid"]),
            task_graph=TaskGraph.from_dict(data["task_graph"]),
            discovery_map=discovery_map,
            required_nodes=set(data["required_nodes"]),
            hidden_budget=int(data["hidden_budget"]),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class Action:
    kind: str
    value: str

    @staticmethod
    def move(target: Coord) -> "Action":
        return Action(kind="move", value=f"{target[0]},{target[1]}")


@dataclass
class Observation:
    t: int
    position: Coord
    neighbors: list[Coord]
    newly_discovered_visible: list[str]
    newly_activated_visible: list[str]
    discovered_visible: list[str]
    activated_visible: list[str]
    goal_visible: str
    newly_discovered_prerequisites_visible: dict[str, list[list[str]]] = field(default_factory=dict)
    newly_discovered_ancestors_visible: dict[str, list[str]] = field(default_factory=dict)
    newly_activated_ancestors_visible: dict[str, list[str]] = field(default_factory=dict)
    exploit_only: bool = False
    grid_width: int | None = None
    grid_height: int | None = None
    open_cells: list[Coord] = field(default_factory=list)
    state_positions_visible: dict[str, Coord] = field(default_factory=dict)
    all_prerequisites_visible: dict[str, list[list[str]]] = field(default_factory=dict)
    all_ancestors_visible: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class StepRecord:
    t: int
    position: Coord
    action: str
    valid: bool
    newly_discovered: set[str]
    newly_activated: set[str]
    discovered: set[str]
    activated: set[str]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "position": list(self.position),
            "action": self.action,
            "valid": self.valid,
            "newly_discovered": sorted(self.newly_discovered),
            "newly_activated": sorted(self.newly_activated),
            "discovered": sorted(self.discovered),
            "activated": sorted(self.activated),
            "note": self.note,
        }


@dataclass
class Trajectory:
    records: list[StepRecord]
    success: bool
    terminated_reason: str
    total_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "terminated_reason": self.terminated_reason,
            "total_steps": self.total_steps,
            "records": [record.to_dict() for record in self.records],
        }
