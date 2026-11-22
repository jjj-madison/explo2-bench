"""Construct compact maps with sampled corridors, obstacles, and feasible tasks."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import random
from typing import Any

from ..generator import _inflate_point, _sample_random_coords, _sample_random_line
from .task_graph import construct_task_graph
from ..types import Coord, GridSpec
from .types import EpisodeInstanceV3, ExplorationLevel, TaskDAGDifficulty, V3Knobs
from .config import GEOMETRY_REVISION, knobs_for, load_settings


def _add_random_walls(open_cells: set[Coord], start: Coord, count: int, seed: str) -> tuple[set[Coord], list[Coord]]:
    """Remove exactly count cells while preserving the start and connectivity."""
    remaining = set(open_cells)
    candidates = sorted(remaining - {start})
    random.Random(seed).shuffle(candidates)
    removed = []
    while len(removed) < count:
        progress = False
        for candidate in candidates:
            if candidate not in remaining:
                continue
            remaining.remove(candidate)
            reached = {start}
            pending = [start]
            while pending:
                x, y = pending.pop()
                for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if neighbor in remaining and neighbor not in reached:
                        reached.add(neighbor)
                        pending.append(neighbor)
            if len(reached) == len(remaining):
                removed.append(candidate)
                progress = True
                if len(removed) == count:
                    break
            else:
                remaining.add(candidate)
        # A rejected articulation cell can become removable after a later
        # deletion, so reconsider candidates until the exact target is reached.
        if not progress:
            raise RuntimeError("Cannot add requested walls while preserving connectivity")
    return remaining, removed


def build_layout(knobs: V3Knobs, seed: int) -> tuple[GridSpec, dict[str, Any]]:
    """Sample v2-style paths within the reviewed canvas, then add random walls.

    Canvas size is fixed by node count and density. Rejected layouts are
    resampled within that canvas; they never trigger automatic enlargement.
    Widths have a separate seeded stream, shared across layout attempts.
    """
    count = knobs.num_nodes
    side = max(knobs.min_map_side, math.ceil(math.sqrt(count / knobs.dag_density_over_grid)))
    minimum_open = math.ceil(knobs.min_open_cells_per_node * count)
    if side * side - math.floor(knobs.random_wall_fraction * side * side) < minimum_open:
        raise ValueError("Fixed canvas cannot meet the minimum open-cell requirement after obstacles")
    if math.ceil(side / knobs.anchor_spacing) ** 2 < count + 1:
        raise ValueError("Spatial anchors do not fit the fixed canvas at this spacing")
    probabilities = [(width, probability) for width, probability in knobs.path_width_probs if probability > 0]
    if max(width for width, _ in probabilities) > side:
        raise ValueError("Corridor widths must fit the fixed canvas")
    total_probability = sum(probability for _, probability in probabilities)
    width_rng = random.Random(f"symbolic-v3-corridor-widths:{seed}")
    widths = []
    for _ in range(count + 1):
        threshold = width_rng.random() * total_probability
        cumulative = 0.0
        for width, probability in probabilities:
            cumulative += probability
            if threshold < cumulative:
                widths.append(width)
                break
        else:
            widths.append(probabilities[-1][0])

    for attempt in range(knobs.max_layout_attempts):
        rng = random.Random(f"symbolic-v3-compact-paths:{seed}:{attempt}")
        anchors = _sample_random_coords(width=side, height=side, margin=0, count=count + 1,
                                       spacing=knobs.anchor_spacing, rng=rng)
        if anchors is None:
            continue
        cycle_order = list(range(1, count + 1))
        rng.shuffle(cycle_order)
        edges = list(zip(cycle_order, cycle_order[1:] + cycle_order[:1]))
        edges.append((0, cycle_order[0]))
        open_cells: set[Coord] = set()
        corridors = []
        previous_line = None
        for index, ((a, b), width) in enumerate(zip(edges, widths)):
            line = _sample_random_line(start=anchors[a], end=anchors[b], width=side, height=side,
                                      margin=0, rng=rng, previous_line=previous_line if index < count else None)
            for point in line:
                open_cells.update((x, y) for x, y in _inflate_point(point, width)
                                  if 0 <= x < side and 0 <= y < side)
            corridors.append({"anchor_ids": [a, b], "width": width,
                              "centerline": [list(point) for point in line]})
            previous_line = line
        initial_open_count = len(open_cells)
        requested_walls = math.floor(knobs.random_wall_fraction * initial_open_count)
        if initial_open_count - requested_walls < minimum_open:
            continue
        start = anchors[0]
        open_cells, walls = _add_random_walls(open_cells, start, requested_walls,
                                            f"symbolic-v3-random-walls:{seed}:{attempt}")
        blocked = {(x, y) for y in range(side) for x in range(side)} - open_cells
        scaffold = {"canvas": [side, side], "anchors": anchors, "connections": edges,
                    "centerlines": [corridor["centerline"] for corridor in corridors]}
        metadata = {
            "geometry_revision": GEOMETRY_REVISION,
            "backbone_style": "shared_shortest_cycle_walk",
            "anchor_count": count, "spatial_anchors": [list(point) for point in anchors],
            "canvas_side": side, "dag_density_over_grid": knobs.dag_density_over_grid,
            "anchor_spacing": knobs.anchor_spacing, "layout_attempts": attempt + 1,
            "pre_obstacle_open_cells": initial_open_count,
            "random_wall_fraction_requested": knobs.random_wall_fraction,
            "random_wall_count_requested": requested_walls, "random_wall_count": len(walls),
            "random_wall_fraction": len(walls) / initial_open_count,
            "random_wall_cells": [list(cell) for cell in sorted(walls)],
            "corridors": corridors, "open_fraction": len(open_cells) / (side * side),
            "corridor_count": len(corridors),
            "corridor_width_probs": {str(width): probability for width, probability in knobs.path_width_probs},
            # Keep frozen metadata identical across Python's float sum implementations.
            "corridor_width_expected": math.fsum(width * probability for width, probability in probabilities) / total_probability,
            "corridor_widths": widths, "corridor_width_mean": sum(widths) / len(widths),
            "corridor_width_min": min(widths), "corridor_width_max": max(widths),
            "scaffold_fingerprint": hashlib.sha256(json.dumps(scaffold, sort_keys=True).encode()).hexdigest(),
        }
        return GridSpec(side, side, blocked, start), metadata
    raise ValueError(f"No feasible layout in {knobs.max_layout_attempts} attempts on the fixed {side}x{side} canvas")


def _discovery_walk(grid: GridSpec, seed: int) -> tuple[list[Coord], list[Coord], dict[Coord, int]]:
    """Randomized DFS provides a constructive walk shorter than 2 * open cells."""
    rng = random.Random(f"symbolic-v3-witness:{seed}")
    walk = [grid.start]
    order = [grid.start]
    first_step = {grid.start: 0}

    def shuffled_neighbors(coord: Coord) -> list[Coord]:
        neighbors = grid.neighbors(coord)
        rng.shuffle(neighbors)
        return neighbors

    stack = [(grid.start, iter(shuffled_neighbors(grid.start)))]
    while stack:
        current, iterator = stack[-1]
        target = next(iterator, None)
        if target is None:
            stack.pop()
            if stack:
                walk.append(stack[-1][0])
        elif target not in first_step:
            walk.append(target)
            first_step[target] = len(walk) - 1
            order.append(target)
            stack.append((target, iter(shuffled_neighbors(target))))
    if len(order) != len(grid.open_cells()):
        raise RuntimeError("Constructed grid is disconnected")
    return walk, order, first_step


def verify_witness(instance: EpisodeInstanceV3) -> int:
    """Check movement, local AND/OR activation, and the saved budget certificate."""
    raw = instance.metadata.get("feasibility_witness", [])
    walk = [tuple(coord) for coord in raw]
    if not walk or walk[0] != instance.grid.start:
        raise ValueError("Feasibility witness must begin at the start")
    if len(walk) - 1 > instance.hidden_budget:
        raise ValueError("Feasibility witness exceeds the action budget")
    activated = set()
    for index, position in enumerate(walk):
        if not instance.grid.is_open(position) or (index and position not in instance.grid.neighbors(walk[index - 1])):
            raise ValueError("Feasibility witness contains an invalid movement")
        local = instance.discovery_map.get(position, set())
        while True:
            ready = {node for node in local - activated
                     if not instance.task_graph.nodes[node].prerequisites
                     or any(parents <= activated for parents in instance.task_graph.nodes[node].prerequisites)}
            if not ready:
                break
            activated.update(ready)
    if instance.task_graph.goal_id not in activated:
        raise ValueError("Feasibility witness does not activate the goal")
    return len(walk) - 1


def generate_episode(
    task_dag_difficulty: str | TaskDAGDifficulty,
    exploration_level: str | ExplorationLevel,
    seed: int,
    knobs: V3Knobs | None = None,
) -> EpisodeInstanceV3:
    difficulty = TaskDAGDifficulty(task_dag_difficulty)
    exploration = ExplorationLevel(exploration_level)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if knobs is None:
        knobs = knobs_for(load_settings(), exploration.value, difficulty.value)
    graph = construct_task_graph(knobs, seed)
    grid, geometry = build_layout(knobs, seed)
    open_count = len(grid.open_cells())
    if open_count / len(graph.nodes) < knobs.min_open_cells_per_node:
        raise ValueError("Realized open cells per task node are below the configured minimum")
    walk, visit_order, first_step = _discovery_walk(grid, seed)
    count = len(graph.nodes)
    available = len(visit_order) - 1
    if count > available:
        raise ValueError("Every task requires a distinct open cell beyond the start")
    prefix_count = max(count, math.ceil(knobs.task_coverage * available))
    rng = random.Random(f"symbolic-v3-placement:{seed}")
    ordered_nodes = []
    depths = sorted({node.depth for node in graph.nodes.values()})
    for depth in depths:
        layer = sorted(node_id for node_id, node in graph.nodes.items() if node.depth == depth)
        rng.shuffle(layer)
        ordered_nodes.extend(layer)
    anchors = []
    # One task per stratum avoids accidental clustering while preserving the
    # activation order. The goal fixes the realized discovery prefix exactly.
    for index in range(count):
        lower = index * prefix_count // count
        upper = (index + 1) * prefix_count // count
        rank = prefix_count - 1 if index == count - 1 else rng.randrange(lower, upper)
        anchors.append(visit_order[rank + 1])
    discovery_map = {coord: {node} for coord, node in zip(anchors, ordered_nodes)}
    goal_coord = next(coord for coord, nodes in discovery_map.items() if graph.goal_id in nodes)
    witness = walk[:first_step[goal_coord] + 1]
    budget = math.floor(knobs.alpha_budget * open_count)
    widths = Counter(node.depth for node in graph.nodes.values())
    metadata = {
        "configuration_version": "v3", "generator_version": 3,
        **geometry,
        "exploration_level": exploration.value, "task_dag_difficulty": difficulty.value,
        "open_cells": open_count, "open_cells_per_node": open_count / count,
        "min_open_cells_per_node": knobs.min_open_cells_per_node,
        "task_coverage_requested": knobs.task_coverage,
        "task_coverage": prefix_count / available,
        "layer_widths": [widths[depth] for depth in depths],
        "actual_dag_depth": max(depths), "max_layer_width": max(widths.values()),
        "minimum_goal_activations": count,
        "witness_steps": len(witness) - 1,
        "witness_budget_fraction": (len(witness) - 1) / budget if budget else None,
        "feasibility_witness": [list(coord) for coord in witness],
        "spatial_fingerprint": hashlib.sha256(json.dumps(grid.to_dict(), sort_keys=True).encode()).hexdigest(),
    }
    instance = EpisodeInstanceV3(
        episode_id=f"v3-exploration-{exploration.value}-dag-{difficulty.value}-s{seed}",
        exploration_level=exploration, task_dag_difficulty=difficulty, seed=seed,
        knobs=knobs, grid=grid, task_graph=graph, discovery_map=discovery_map,
        required_nodes=set(graph.nodes), hidden_budget=budget, metadata=metadata,
    )
    verify_witness(instance)
    return instance
