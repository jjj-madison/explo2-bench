"""Bundled v3 presets and configuration loading."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
from typing import Any

from .types import ExplorationLevel, TaskDAGDifficulty, V3Knobs


GEOMETRY_REVISION = "compact-obstacles-1"
_LAYOUT_FIELDS = ("min_map_side", "max_layout_attempts", "min_open_cells_per_node")
_MAP_FIELDS = (*_LAYOUT_FIELDS, "dag_density_over_grid", "path_width_probs", "anchor_spacing", "random_wall_fraction")
_GRAPH_FIELDS = ("layer_profiles", "extra_dependency_probability", "alternative_probability", "max_extra_parents")


def knobs_for(settings: dict[str, Any], exploration: str, difficulty: str) -> V3Knobs:
    """Resolve a complete parameter set without changing its budget."""
    return V3Knobs.from_dict({
        **settings["defaults"],
        **settings["task_dag_difficulties"][difficulty],
        **settings["exploration_levels"][exploration],
    })


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Read bundled presets, or a complete user-supplied settings file."""
    content = files(__package__).joinpath("settings.json").read_text(encoding="utf-8") if path is None else Path(path).read_text(encoding="utf-8")
    settings = json.loads(content)
    if settings.get("configuration_version") != "v3":
        raise ValueError("Active settings must have configuration_version='v3'")
    if settings.get("geometry_revision") != GEOMETRY_REVISION:
        raise ValueError(f"Active settings require geometry_revision={GEOMETRY_REVISION!r}")
    expected_exploration = [level.value for level in ExplorationLevel if level.value != "custom"]
    expected_dags = [level.value for level in TaskDAGDifficulty if level.value != "custom"]
    if set(settings.get("exploration_levels", {})) != set(expected_exploration):
        raise ValueError("Settings require all nine exploration levels")
    if set(settings.get("task_dag_difficulties", {})) != set(expected_dags):
        raise ValueError("Settings require all nine DAG levels")
    if settings.get("preset_count") != 81:
        raise ValueError("Settings require exactly 81 presets")
    if settings.get("axis_order") != {"exploration_levels": expected_exploration, "task_dag_difficulties": expected_dags}:
        raise ValueError("Declared axis order must match the ascending profile order")
    # JSON object order is not semantic; apply the declared profile order after
    # parsing.
    for axis, order in (("exploration_levels", expected_exploration), ("task_dag_difficulties", expected_dags)):
        settings[axis] = {name: settings[axis][name] for name in order}
    resolved = {}
    layout_parameters = None
    map_parameters = {}
    graph_parameters = {}
    coverages = {}
    width_distributions = {}
    densities = {}
    wall_fractions = {}
    node_counts = {}
    for exploration in expected_exploration:
        for difficulty in expected_dags:
            knobs = knobs_for(settings, exploration, difficulty)
            if knobs.alpha_budget != 2.0:
                raise ValueError("V3 calibration fixes alpha_budget=2")
            signature = tuple(getattr(knobs, field) for field in _MAP_FIELDS)
            if exploration in map_parameters and signature != map_parameters[exploration]:
                raise ValueError("DAG profiles must share spatial controls; their node counts determine map sizes")
            map_parameters[exploration] = signature
            layout = tuple(getattr(knobs, field) for field in _LAYOUT_FIELDS)
            if layout_parameters is not None and layout != layout_parameters:
                raise ValueError("All presets must share canvas and feasibility limits")
            layout_parameters = layout
            width_distributions[exploration] = dict(knobs.path_width_probs)
            densities[exploration] = knobs.dag_density_over_grid
            wall_fractions[exploration] = knobs.random_wall_fraction
            graph_signature = tuple(getattr(knobs, field) for field in _GRAPH_FIELDS)
            if difficulty in graph_parameters and graph_parameters[difficulty] != graph_signature:
                raise ValueError("Exploration profiles must not change task graph parameters")
            graph_parameters[difficulty] = graph_signature
            if exploration in coverages and coverages[exploration] != knobs.task_coverage:
                raise ValueError("DAG profiles must not change exploration coverage")
            coverages[exploration] = knobs.task_coverage
            node_counts[difficulty] = knobs.num_nodes
            resolved[exploration, difficulty] = knobs.to_dict()
    if set(coverages.values()) != {1.0}:
        raise ValueError("Active profiles fix task coverage at 1.0")
    for name, values in (("node count", list(node_counts.values())), ("random wall fraction", list(wall_fractions.values()))):
        if not all(a < b for a, b in zip(values, values[1:])):
            raise ValueError(f"Profile {name} must increase strictly along its axis")
    if not all(a > b for a, b in zip(densities.values(), list(densities.values())[1:])):
        raise ValueError("Canvas density must decrease strictly along exploration")
    distributions = list(width_distributions.values())
    support = sorted(set().union(*(set(probabilities) for probabilities in distributions)))
    for easier, harder in zip(distributions, distributions[1:]):
        differences = [sum(easier.get(w, 0) - harder.get(w, 0) for w in support if w <= width)
                       for width in support]
        if any(value < -1e-9 for value in differences) or not any(value > 1e-9 for value in differences):
            raise ValueError("Corridor width distributions must shift probability toward wider paths")
    presets = settings.get("presets", [])
    entries = {(row["exploration_level"], row["task_dag_difficulty"]): row for row in presets}
    if len(presets) != 81 or set(entries) != set(resolved):
        raise ValueError("Presets must cover each of the 81 combinations exactly once")
    for key, row in entries.items():
        if V3Knobs.from_dict(row["knobs"]).to_dict() != resolved[key]:
            raise ValueError(f"Expanded preset does not match its profiles: {key}")
    return settings
