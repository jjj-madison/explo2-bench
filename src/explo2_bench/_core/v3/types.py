"""V3 generator controls and episode serialization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum
import json
import math
from typing import Any

from ..types import Coord, GridSpec, TaskGraph, TaskNode

CONFIGURATION_VERSION = "v3"


class ExplorationLevel(str, Enum):
    X_LOWEST = "x-lowest"
    LOWEST = "lowest"
    LOWER = "lower"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HIGHER = "higher"
    HIGHEST = "highest"
    X_HIGHEST = "x-highest"
    CUSTOM = "custom"


class TaskDAGDifficulty(str, Enum):
    X_EASIEST = "x-easiest"
    EASIEST = "easiest"
    EASIER = "easier"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    HARDER = "harder"
    HARDEST = "hardest"
    X_HARDEST = "x-hardest"
    CUSTOM = "custom"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _keys(data: Any, expected: set[str], name: str, *, required: bool = True) -> None:
    if not isinstance(data, dict):
        raise TypeError(f"{name} must be a dictionary")
    unexpected = set(data) - expected
    missing = expected - set(data) if required else set()
    if unexpected or missing:
        raise ValueError(f"Invalid {name} fields: missing={sorted(missing)}, unexpected={sorted(unexpected)}")


@dataclass(frozen=True)
class V3Knobs:
    alpha_budget: float = 2.0
    dag_density_over_grid: float = 0.16
    path_width_probs: tuple[tuple[int, float], ...] = ((1, 0.35), (2, 0.5), (3, 0.15))
    anchor_spacing: int = 2
    random_wall_fraction: float = 0.01
    min_map_side: int = 5
    max_layout_attempts: int = 128
    # Templates: non-goal widths are seed-permuted; the final goal stays alone.
    layer_profiles: tuple[tuple[int, ...], ...] = ((1, 1, 1), (2, 1))
    task_coverage: float = 1.0
    extra_dependency_probability: float = 0.25
    alternative_probability: float = 0.15
    max_extra_parents: int = 2
    min_open_cells_per_node: float = 2.0

    def __post_init__(self) -> None:
        if _number(self.alpha_budget, "alpha_budget") != 2.0:
            raise ValueError("v3 fixes alpha_budget at 2.0")
        density = _number(self.dag_density_over_grid, "dag_density_over_grid")
        if not 0 < density <= 0.5:
            raise ValueError("dag_density_over_grid must be in (0, 0.5]")
        _integer(self.anchor_spacing, "anchor_spacing", 1)
        _integer(self.min_map_side, "min_map_side", 3)
        _integer(self.max_layout_attempts, "max_layout_attempts", 1)
        if not 0 <= _number(self.random_wall_fraction, "random_wall_fraction") < 1:
            raise ValueError("random_wall_fraction must be in [0, 1)")
        raw_probabilities = self.path_width_probs
        if isinstance(raw_probabilities, dict):
            entries = tuple(raw_probabilities.items())
        elif isinstance(raw_probabilities, tuple):
            entries = raw_probabilities
        else:
            raise ValueError("path_width_probs must be a dictionary or tuple of width/probability pairs")
        probabilities = {}
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("Each path_width_probs entry must contain a width and probability")
            width, probability = entry
            if isinstance(width, str):
                try:
                    parsed_width = int(width)
                except ValueError as error:
                    raise ValueError("Path widths must be positive integers") from error
                if str(parsed_width) != width:
                    raise ValueError("Path width string keys must be canonical integers")
                width = parsed_width
            _integer(width, "path width", 1)
            probability = _number(probability, "path width probability")
            if not 0 <= probability <= 1 or width in probabilities:
                raise ValueError("Path widths must be distinct with probabilities in [0, 1]")
            probabilities[width] = probability
        if not probabilities or not math.isclose(math.fsum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("path_width_probs probabilities must sum to 1")
        object.__setattr__(self, "path_width_probs", tuple(sorted(probabilities.items())))
        for name in ("layer_profiles",):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values or any(not isinstance(row, tuple) or not row for row in values):
                raise ValueError(f"{name} must be a nonempty tuple of nonempty tuples")
            for row in values:
                for value in row:
                    _integer(value, name, 1)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct choices")
        if any(len(profile) < 2 or profile[-1] != 1 for profile in self.layer_profiles):
            raise ValueError("Each layer profile needs at least two layers and a single final goal")
        if len({sum(profile) for profile in self.layer_profiles}) != 1:
            raise ValueError("All layer profiles must have the same node count")
        coverage = _number(self.task_coverage, "task_coverage")
        if not 0 < coverage <= 1:
            raise ValueError("task_coverage must be in (0, 1]")
        for name in ("extra_dependency_probability", "alternative_probability"):
            if not 0 <= _number(getattr(self, name), name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        _integer(self.max_extra_parents, "max_extra_parents")
        if _number(self.min_open_cells_per_node, "min_open_cells_per_node") <= 1:
            raise ValueError("min_open_cells_per_node must be greater than 1")

    @property
    def num_nodes(self) -> int:
        return sum(self.layer_profiles[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: [list(row) for row in getattr(self, item.name)]
            if item.name == "layer_profiles"
            else {str(width): probability for width, probability in self.path_width_probs}
            if item.name == "path_width_probs"
            else getattr(self, item.name)
            for item in fields(self)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "V3Knobs":
        _keys(data, {item.name for item in fields(cls)}, "v3 knobs", required=False)
        values = dict(data)
        for name in ("layer_profiles",):
            if name in values:
                raw = values[name]
                if not isinstance(raw, (list, tuple)) or any(not isinstance(row, (list, tuple)) for row in raw):
                    raise ValueError(f"{name} must be a sequence of sequences")
                values[name] = tuple(tuple(row) for row in raw)
        return cls(**values)


@dataclass
class EpisodeInstanceV3:
    episode_id: str
    exploration_level: ExplorationLevel
    task_dag_difficulty: TaskDAGDifficulty
    seed: int
    knobs: V3Knobs
    grid: GridSpec
    task_graph: TaskGraph
    discovery_map: dict[Coord, set[str]]
    required_nodes: set[str]
    hidden_budget: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a nonempty string")
        if not isinstance(self.exploration_level, ExplorationLevel) or not isinstance(self.task_dag_difficulty, TaskDAGDifficulty):
            raise TypeError("v3 episodes require v3 exploration and task-graph enums")
        if not isinstance(self.knobs, V3Knobs):
            raise TypeError("v3 episodes require V3Knobs")
        _integer(self.seed, "seed")
        _integer(self.hidden_budget, "hidden_budget", 1)
        if not isinstance(self.grid, GridSpec) or not isinstance(self.task_graph, TaskGraph):
            raise TypeError("v3 episodes require GridSpec and TaskGraph")
        _integer(self.grid.width, "grid.width", 1)
        _integer(self.grid.height, "grid.height", 1)
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        json.dumps(self.metadata, allow_nan=False)

    def indicator_to_visible(self) -> dict[str, str]:
        return {node.indicator_id: node.visible_name for node in self.task_graph.nodes.values()}

    def visible_to_indicator(self) -> dict[str, str]:
        return {visible: indicator for indicator, visible in self.indicator_to_visible().items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_version": CONFIGURATION_VERSION,
            "episode_id": self.episode_id,
            "exploration_level": self.exploration_level.value,
            "task_dag_difficulty": self.task_dag_difficulty.value,
            "seed": self.seed, "knobs": self.knobs.to_dict(),
            "grid": self.grid.to_dict(), "task_graph": self.task_graph.to_dict(),
            "discovery_map": {f"{x},{y}": sorted(nodes) for (x, y), nodes in sorted(self.discovery_map.items())},
            "required_nodes": sorted(self.required_nodes), "hidden_budget": self.hidden_budget,
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeInstanceV3":
        expected = {item.name for item in fields(cls)} | {"configuration_version"}
        _keys(data, expected, "v3 episode")
        if data["configuration_version"] != CONFIGURATION_VERSION:
            raise ValueError("Expected configuration_version='v3'")
        json.dumps(data, allow_nan=False)

        def coordinate(raw: Any) -> Coord:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError("Coordinates must contain two integers")
            return _integer(raw[0], "coordinate"), _integer(raw[1], "coordinate")

        def names(raw: Any) -> set[str]:
            if not isinstance(raw, list) or any(not isinstance(name, str) or not name for name in raw) or len(set(raw)) != len(raw):
                raise ValueError("Node lists must contain distinct nonempty strings")
            return set(raw)

        grid = data["grid"]
        _keys(grid, {"width", "height", "blocked", "start"}, "grid")
        _integer(grid["width"], "grid.width", 1)
        _integer(grid["height"], "grid.height", 1)
        if not isinstance(grid["blocked"], list):
            raise ValueError("grid.blocked must be a coordinate list")
        blocked = [coordinate(cell) for cell in grid["blocked"]]
        if len(set(blocked)) != len(blocked):
            raise ValueError("grid.blocked must contain distinct coordinates")
        parsed_grid = GridSpec(grid["width"], grid["height"], set(blocked), coordinate(grid["start"]))
        if not parsed_grid.is_open(parsed_grid.start) or any(not parsed_grid.in_bounds(cell) for cell in blocked):
            raise ValueError("Invalid grid start or blocked cell")

        graph = data["task_graph"]
        _keys(graph, {"nodes", "goal_id"}, "task_graph")
        if not isinstance(graph["nodes"], dict) or not graph["nodes"]:
            raise ValueError("task_graph.nodes must be a nonempty dictionary")
        nodes = {}
        for node_id, node in graph["nodes"].items():
            _keys(node, {"indicator_id", "visible_name", "prerequisites", "depth"}, "task node")
            if not isinstance(node_id, str) or not node_id or node["indicator_id"] != node_id:
                raise ValueError("Task node key must match its nonempty indicator_id")
            if not isinstance(node["visible_name"], str) or not node["visible_name"]:
                raise ValueError("Task visible_name must be a nonempty string")
            if not isinstance(node["prerequisites"], list) or not node["prerequisites"]:
                raise ValueError("Task prerequisites must contain at least one option")
            options = [names(option) for option in node["prerequisites"]]
            if len({frozenset(option) for option in options}) != len(options):
                raise ValueError("Task prerequisite options must be distinct")
            nodes[node_id] = TaskNode(node_id, node["visible_name"], options, _integer(node["depth"], "depth"))
        if graph["goal_id"] not in nodes or len({node.visible_name for node in nodes.values()}) != len(nodes):
            raise ValueError("Invalid goal or duplicate visible names")
        for node in nodes.values():
            for option in node.prerequisites:
                if any(parent not in nodes or nodes[parent].depth >= node.depth for parent in option):
                    raise ValueError("Prerequisites must refer to shallower task nodes")
        if not isinstance(data["discovery_map"], dict):
            raise ValueError("discovery_map must be a dictionary")
        discovery = {}
        placed = set()
        for key, value in data["discovery_map"].items():
            if not isinstance(key, str) or len(key.split(",")) != 2:
                raise ValueError("Invalid discovery_map coordinate")
            try:
                cell = coordinate([int(part) for part in key.split(",")])
            except ValueError as error:
                raise ValueError("Invalid discovery_map coordinate") from error
            if key != f"{cell[0]},{cell[1]}" or not parsed_grid.is_open(cell) or cell == parsed_grid.start:
                raise ValueError("Task coordinates must be canonical, open, and distinct from start")
            at_cell = names(value)
            if len(at_cell) != 1 or not at_cell.issubset(nodes) or placed.intersection(at_cell):
                raise ValueError("Each task needs exactly one distinct map position")
            discovery[cell] = at_cell
            placed.update(at_cell)
        required = names(data["required_nodes"])
        knobs = V3Knobs.from_dict(data["knobs"])
        if placed != set(nodes) or required != set(nodes) or len(nodes) != knobs.num_nodes:
            raise ValueError("v3 node coverage, required nodes, and configured count must agree")
        layers = {
            depth: {node_id for node_id, node in nodes.items() if node.depth == depth}
            for depth in range(max(node.depth for node in nodes.values()) + 1)
        }
        profile = tuple(len(layer) for layer in layers.values())
        matches_template = any(
            len(profile) == len(template) and sorted(profile[:-1]) == sorted(template[:-1])
            for template in knobs.layer_profiles
        )
        if not matches_template or layers[len(profile) - 1] != {graph["goal_id"]}:
            raise ValueError("Task graph layers must permute a configured template with the goal alone last")
        for node in nodes.values():
            if node.depth == 0:
                if node.prerequisites != [set()]:
                    raise ValueError("Primitive tasks must have one empty prerequisite option")
            elif not set.intersection(*node.prerequisites).intersection(layers[node.depth - 1]):
                raise ValueError("Every nonprimitive needs a mandatory immediate-layer parent")
        for depth in range(len(profile) - 1):
            covered = set().union(*(set.intersection(*nodes[child].prerequisites) for child in layers[depth + 1]))
            if not layers[depth].issubset(covered):
                raise ValueError("Every task must belong to the mandatory goal backbone")
        if nodes[graph["goal_id"]].prerequisites != [layers[len(profile) - 2]]:
            raise ValueError("Goal must require the full preceding layer")
        if len(parsed_grid.open_cells()) / knobs.num_nodes < knobs.min_open_cells_per_node:
            raise ValueError("Map has fewer open cells per node than configured")
        if _integer(data["hidden_budget"], "hidden_budget", 1) != 2 * len(parsed_grid.open_cells()):
            raise ValueError("v3 hidden_budget must equal 2 times open cells")
        return cls(data["episode_id"], ExplorationLevel(data["exploration_level"]), TaskDAGDifficulty(data["task_dag_difficulty"]),
                   data["seed"], knobs, parsed_grid, TaskGraph(nodes, graph["goal_id"]), discovery, required,
                   data["hidden_budget"], deepcopy(data["metadata"]))


__all__ = ["CONFIGURATION_VERSION", "V3Knobs", "EpisodeInstanceV3", "ExplorationLevel", "TaskDAGDifficulty", "Coord", "GridSpec", "TaskGraph", "TaskNode"]
