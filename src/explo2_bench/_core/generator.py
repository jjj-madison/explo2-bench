from __future__ import annotations

import math
import random
from collections import deque
from typing import Iterable

from .types import (
    ConstructiveKnobs,
    Coord,
    EpisodeInstance,
    ExplorationLevel,
    GridSpec,
    TaskDAGDifficulty,
    TaskGraph,
    TaskNode,
)

_VISIBLE_TOKEN_LENGTH = 4


def _is_grid_connected(width: int, height: int, blocked: set[Coord], start: Coord) -> bool:
    if start in blocked:
        return False
    open_count = width * height - len(blocked)
    if open_count <= 0:
        return False

    queue: deque[Coord] = deque([start])
    visited: set[Coord] = {start}
    while queue:
        x, y = queue.popleft()
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            nx, ny = nxt
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if nxt in blocked or nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)

    return len(visited) == open_count


def _generate_visible_names(ids: Iterable[str], length: int, rng: random.Random) -> dict[str, str]:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for indicator_id in sorted(ids):
        while True:
            token = "".join(rng.choice(alphabet) for _ in range(length))
            if token in used:
                continue
            used.add(token)
            mapping[indicator_id] = token
            break
    return mapping


def _normalize_prob_map(probs: dict[int, float], *, name: str) -> list[tuple[int, float]]:
    if not probs:
        raise ValueError(f"{name} must be non-empty.")

    normalized: list[tuple[int, float]] = []
    total = 0.0
    for raw_value, raw_weight in sorted(probs.items()):
        value = int(raw_value)
        weight = float(raw_weight)
        if value < 1:
            raise ValueError(f"{name} keys must be >= 1, got {value}.")
        if weight < 0:
            raise ValueError(f"{name} weights must be >= 0, got {weight}.")
        if weight == 0:
            continue
        normalized.append((value, weight))
        total += weight

    if total <= 0:
        raise ValueError(f"{name} must contain at least one positive weight.")

    return [(value, weight / total) for value, weight in normalized]


def _sample_from_prob_map(
    rng: random.Random,
    probs: list[tuple[int, float]],
) -> int:
    target = rng.random()
    cumulative = 0.0
    for value, weight in probs:
        cumulative += weight
        if target <= cumulative:
            return value
    return probs[-1][0]


def _sample_parent_set(
    *,
    rng: random.Random,
    candidates: list[tuple[str, int]],
    count: int,
    child_depth: int,
    parent_depth_bias: float,
) -> set[str]:
    if parent_depth_bias < 0:
        raise ValueError(f"parent_depth_bias must be >= 0, got {parent_depth_bias}")
    if count >= len(candidates):
        return {node_id for node_id, _depth in candidates}

    selected: set[str] = set()
    remaining = list(candidates)
    while len(selected) < count and remaining:
        weights = [
            pow(2.718281828459045, -parent_depth_bias * ((child_depth - 1) - candidate_depth))
            for _node_id, candidate_depth in remaining
        ]
        total_weight = sum(weights)
        if total_weight <= 0:
            choice_idx = rng.randrange(len(remaining))
        else:
            threshold = rng.random() * total_weight
            choice_idx = len(remaining) - 1
            cumulative = 0.0
            for idx, weight in enumerate(weights):
                cumulative += weight
                if threshold <= cumulative:
                    choice_idx = idx
                    break
        node_id, _depth = remaining.pop(choice_idx)
        selected.add(node_id)
    return selected


def _goal_ancestors(nodes: dict[str, TaskNode], goal_id: str) -> set[str]:
    ancestors: set[str] = {goal_id}
    stack = [goal_id]
    while stack:
        node_id = stack.pop()
        node = nodes[node_id]
        for prerequisite_option in node.prerequisites:
            for dep in prerequisite_option:
                if dep not in nodes or dep in ancestors:
                    continue
                ancestors.add(dep)
                stack.append(dep)
    return ancestors


def _dedupe_sorted_prerequisite_options(prerequisite_options: list[set[str]]) -> list[set[str]]:
    unique: set[frozenset[str]] = set()
    deduped: list[set[str]] = []
    for prerequisite_option in prerequisite_options:
        if not prerequisite_option:
            continue
        frozen = frozenset(prerequisite_option)
        if frozen in unique:
            continue
        unique.add(frozen)
        deduped.append(set(prerequisite_option))
    deduped.sort(key=lambda r: (len(r), sorted(r)))
    return deduped


def _min_goal_depth(num_nodes: int, max_nodes_per_layer: int) -> int:
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    if max_nodes_per_layer < 1:
        raise ValueError(f"max_nodes_per_layer must be >= 1, got {max_nodes_per_layer}")

    non_goal_nodes = max(0, num_nodes - 1)
    if non_goal_nodes == 0:
        return 0
    if non_goal_nodes < max_nodes_per_layer:
        raise ValueError(
            "num_nodes must allow a primitive layer of exactly max_nodes_per_layer nodes; "
            f"got num_nodes={num_nodes}, max_nodes_per_layer={max_nodes_per_layer}"
        )

    remaining_after_primitives = non_goal_nodes - max_nodes_per_layer
    if remaining_after_primitives == 0:
        return 1
    return 1 + ((remaining_after_primitives + max_nodes_per_layer - 1) // max_nodes_per_layer)


def _sample_goal_depth(num_nodes: int, max_nodes_per_layer: int, rng: random.Random) -> int:
    min_depth = _min_goal_depth(num_nodes, max_nodes_per_layer)
    non_goal_nodes = max(0, num_nodes - 1)
    if non_goal_nodes == 0:
        max_depth = 0
    elif non_goal_nodes < max_nodes_per_layer:
        raise RuntimeError(
            "No feasible goal depth for "
            f"num_nodes={num_nodes}, max_nodes_per_layer={max_nodes_per_layer}; "
            "the primitive layer alone would exceed the available non-goal nodes."
        )
    else:
        max_depth = 1 + (non_goal_nodes - max_nodes_per_layer)
    if min_depth > max_depth:
        raise RuntimeError(
            f"No feasible goal depth for num_nodes={num_nodes}, "
            f"max_nodes_per_layer={max_nodes_per_layer}"
        )
    candidate_depths = list(range(min_depth, max_depth + 1))
    weights = [
        1.0 / ((depth - min_depth + 1) ** 2)
        for depth in candidate_depths
    ]
    return rng.choices(candidate_depths, weights=weights, k=1)[0]


def _allocate_layer_profile(
    *,
    num_nodes: int,
    goal_depth: int,
    max_nodes_per_layer: int,
    rng: random.Random,
) -> list[int]:
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    if goal_depth < 0:
        raise ValueError(f"goal_depth must be >= 0, got {goal_depth}")
    if max_nodes_per_layer < 1:
        raise ValueError(f"max_nodes_per_layer must be >= 1, got {max_nodes_per_layer}")

    if num_nodes == 1:
        if goal_depth != 0:
            raise RuntimeError(f"num_nodes=1 requires goal_depth=0, got {goal_depth}")
        return [1]

    non_goal_nodes = num_nodes - 1
    if goal_depth < 1:
        raise RuntimeError(f"num_nodes={num_nodes} requires goal_depth >= 1, got {goal_depth}")
    if non_goal_nodes < max_nodes_per_layer:
        raise RuntimeError(
            "num_nodes must allocate exactly max_nodes_per_layer primitive nodes; "
            f"got num_nodes={num_nodes}, max_nodes_per_layer={max_nodes_per_layer}"
        )

    primitive_count = max_nodes_per_layer
    remaining_non_goal_nodes = non_goal_nodes - primitive_count
    additional_depths = goal_depth - 1

    if remaining_non_goal_nodes < additional_depths:
        raise RuntimeError(
            f"goal_depth={goal_depth} is too large for num_nodes={num_nodes}; "
            "need at least one non-primitive node per deeper depth."
        )
    if remaining_non_goal_nodes > additional_depths * max_nodes_per_layer:
        raise RuntimeError(
            f"num_nodes={num_nodes} exceeds capacity for goal_depth={goal_depth} "
            f"and max_nodes_per_layer={max_nodes_per_layer}."
        )

    counts = [primitive_count] + ([1] * additional_depths)
    remaining = remaining_non_goal_nodes - additional_depths
    while remaining > 0:
        eligible = [
            idx
            for idx in range(1, goal_depth)
            if counts[idx] < max_nodes_per_layer and counts[idx - 1] > counts[idx]
        ]
        if not eligible:
            raise RuntimeError("Could not allocate a non-increasing layer profile.")
        weights = [1.0 / (idx + 1) for idx in eligible]
        chosen_idx = rng.choices(eligible, weights=weights, k=1)[0]
        counts[chosen_idx] += 1
        remaining -= 1

    return counts + [1]


def _axis_offsets(width: int) -> list[int]:
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    left = (width - 1) // 2
    right = width // 2
    return list(range(-left, right + 1))


def _inflate_point(coord: Coord, width: int) -> set[Coord]:
    x, y = coord
    offsets = _axis_offsets(width)
    return {(x + dx, y + dy) for dx in offsets for dy in offsets}


def _inflate_segment(start: Coord, end: Coord, width: int) -> set[Coord]:
    if start == end:
        return _inflate_point(start, width)

    x0, y0 = start
    x1, y1 = end
    cells: set[Coord] = set()
    if x0 == x1:
        offsets = _axis_offsets(width)
        y_lo, y_hi = sorted((y0, y1))
        for y in range(y_lo, y_hi + 1):
            for dx in offsets:
                cells.add((x0 + dx, y))
        return cells
    if y0 == y1:
        offsets = _axis_offsets(width)
        x_lo, x_hi = sorted((x0, x1))
        for x in range(x_lo, x_hi + 1):
            for dy in offsets:
                cells.add((x, y0 + dy))
        return cells
    raise ValueError(f"Segments must be axis-aligned, got {start} -> {end}")


def construct_task_graph(knobs: ConstructiveKnobs, seed: int) -> TaskGraph:
    rng = random.Random(seed + 11)
    option_count_probs = _normalize_prob_map(knobs.option_count_probs, name="option_count_probs")
    dependency_count_probs = _normalize_prob_map(
        knobs.dependency_count_probs,
        name="dependency_count_probs",
    )
    goal_dependency_count_probs = _normalize_prob_map(
        knobs.goal_dependency_count_probs,
        name="goal_dependency_count_probs",
    )

    goal_depth = _sample_goal_depth(knobs.num_nodes, knobs.max_nodes_per_layer, rng)
    layer_counts = _allocate_layer_profile(
        num_nodes=knobs.num_nodes,
        goal_depth=goal_depth,
        max_nodes_per_layer=knobs.max_nodes_per_layer,
        rng=rng,
    )

    primitive_count = layer_counts[0] if goal_depth > 0 else 0
    intermediate_count = max(0, knobs.num_nodes - primitive_count - 1)

    primitive_ids = [f"R_{i:02d}" for i in range(1, primitive_count + 1)]
    intermediate_ids = [f"U_{i:02d}" for i in range(1, intermediate_count + 1)]
    goal_id = "G_00"

    layers: dict[int, list[str]] = {}
    cursor = 0
    if goal_depth > 0:
        layers[0] = primitive_ids
        for depth in range(1, goal_depth):
            count = layer_counts[depth]
            layers[depth] = intermediate_ids[cursor : cursor + count]
            cursor += count
    layers[goal_depth] = [goal_id]

    all_ids = primitive_ids + intermediate_ids + [goal_id]
    visible = _generate_visible_names(all_ids, _VISIBLE_TOKEN_LENGTH, rng)

    nodes: dict[str, TaskNode] = {}
    for primitive_id in primitive_ids:
        nodes[primitive_id] = TaskNode(
            indicator_id=primitive_id,
            visible_name=visible[primitive_id],
            prerequisites=[set()],
            depth=0,
        )

    for depth in range(1, goal_depth):
        for node_id in layers.get(depth, []):
            parent_candidates = sorted(
                (
                    candidate_id,
                    candidate_depth,
                )
                for candidate_depth in range(0, depth)
                for candidate_id in layers.get(candidate_depth, [])
            )
            if not parent_candidates:
                raise RuntimeError(f"No parent candidates available for node {node_id} at depth {depth}.")

            prerequisite_option_count = _sample_from_prob_map(rng, option_count_probs)
            prerequisite_options: list[set[str]] = []
            attempts = max(8, prerequisite_option_count * 6)
            for _ in range(attempts):
                target_dependency_count = _sample_from_prob_map(rng, dependency_count_probs)
                dependency_count = max(1, min(target_dependency_count, len(parent_candidates)))
                parents = _sample_parent_set(
                    rng=rng,
                    candidates=parent_candidates,
                    count=dependency_count,
                    child_depth=depth,
                    parent_depth_bias=knobs.parent_depth_bias,
                )
                if not parents:
                    continue
                if parents not in prerequisite_options:
                    prerequisite_options.append(parents)
                if len(prerequisite_options) >= prerequisite_option_count:
                    break
            if not prerequisite_options:
                fallback_parent = _sample_parent_set(
                    rng=rng,
                    candidates=parent_candidates,
                    count=1,
                    child_depth=depth,
                    parent_depth_bias=knobs.parent_depth_bias,
                )
                prerequisite_options = [fallback_parent]

            nodes[node_id] = TaskNode(
                indicator_id=node_id,
                visible_name=visible[node_id],
                prerequisites=_dedupe_sorted_prerequisite_options(prerequisite_options),
                depth=depth,
            )

    if goal_depth == 0:
        nodes[goal_id] = TaskNode(
            indicator_id=goal_id,
            visible_name=visible[goal_id],
            prerequisites=[set()],
            depth=0,
        )
        return TaskGraph(nodes=nodes, goal_id=goal_id)

    deepest_depth = max(node.depth for node in nodes.values())
    deepest_layer = sorted(
        node_id
        for node_id, node in nodes.items()
        if node.depth == deepest_depth
    )
    sampled_goal_dependency_count = _sample_from_prob_map(rng, goal_dependency_count_probs)
    primary_goal_arity = max(1, min(sampled_goal_dependency_count, len(deepest_layer)))
    ranked_goal_parents = sorted(
        deepest_layer,
        key=lambda node_id: (-len(_goal_ancestors(nodes, node_id)), node_id),
    )
    goal_prerequisite_option = set(ranked_goal_parents[:primary_goal_arity])

    provisional_nodes = dict(nodes)
    provisional_nodes[goal_id] = TaskNode(
        indicator_id=goal_id,
        visible_name=visible[goal_id],
        prerequisites=[set(goal_prerequisite_option)],
        depth=goal_depth,
    )

    disconnected = sorted(set(provisional_nodes.keys()) - _goal_ancestors(provisional_nodes, goal_id))
    if disconnected:
        goal_prerequisite_option.update(disconnected)

    nodes[goal_id] = TaskNode(
        indicator_id=goal_id,
        visible_name=visible[goal_id],
        prerequisites=[goal_prerequisite_option],
        depth=goal_depth,
    )

    remaining_disconnected = sorted(set(nodes.keys()) - _goal_ancestors(nodes, goal_id))
    if remaining_disconnected:
        raise RuntimeError(
            f"Task graph contains nodes disconnected from goal: {remaining_disconnected}"
        )

    return TaskGraph(nodes=nodes, goal_id=goal_id)


def _append_path_cells(cells: list[Coord], path: list[Coord]) -> None:
    if not path:
        return
    if cells and path[0] == cells[-1]:
        cells.extend(path[1:])
    else:
        cells.extend(path)


def _manhattan_distance(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _staircase_path(start: Coord, end: Coord, rng: random.Random) -> list[Coord]:
    x, y = start
    x1, y1 = end
    cells = [(x, y)]
    last_axis: str | None = None
    run_budget = rng.randint(1, 3)

    while (x, y) != (x1, y1):
        dx = x1 - x
        dy = y1 - y
        choices: list[tuple[str, tuple[int, int], float]] = []
        if dx != 0:
            choices.append(("x", (1 if dx > 0 else -1, 0), abs(dx)))
        if dy != 0:
            choices.append(("y", (0, 1 if dy > 0 else -1), abs(dy)))

        if len(choices) == 1:
            axis, step, _weight = choices[0]
        else:
            if last_axis is not None and run_budget > 0:
                matching = [choice for choice in choices if choice[0] == last_axis]
                if matching and rng.random() < 0.65:
                    axis, step, _weight = matching[0]
                else:
                    total = sum(weight for _axis, _step, weight in choices)
                    threshold = rng.random() * total
                    cumulative = 0.0
                    axis, step, _weight = choices[-1]
                    for candidate_axis, candidate_step, weight in choices:
                        cumulative += weight
                        if threshold <= cumulative:
                            axis, step, _weight = candidate_axis, candidate_step, weight
                            break
                    if axis != last_axis:
                        run_budget = rng.randint(1, 3)
            else:
                total = sum(weight for _axis, _step, weight in choices)
                threshold = rng.random() * total
                cumulative = 0.0
                axis, step, _weight = choices[-1]
                for candidate_axis, candidate_step, weight in choices:
                    cumulative += weight
                    if threshold <= cumulative:
                        axis, step, _weight = candidate_axis, candidate_step, weight
                        break
                if axis != last_axis:
                    run_budget = rng.randint(1, 3)

        x += step[0]
        y += step[1]
        cells.append((x, y))
        if axis == last_axis:
            run_budget -= 1
        else:
            last_axis = axis
            run_budget -= 1

    return cells


def _sample_random_coords(
    *,
    width: int,
    height: int,
    margin: int,
    count: int,
    spacing: int,
    rng: random.Random,
    attempts: int = 32,
) -> list[Coord] | None:
    if count < 1:
        return []
    if spacing < 1:
        raise ValueError(f"spacing must be >= 1, got {spacing}")
    if width <= (2 * margin) or height <= (2 * margin):
        return None

    candidates = [
        (x, y)
        for y in range(margin, height - margin)
        for x in range(margin, width - margin)
    ]
    if len(candidates) < count:
        return None

    if spacing == 1:
        return rng.sample(candidates, k=count)

    for _ in range(attempts):
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        selected: list[Coord] = []
        for coord in shuffled:
            if all(
                max(abs(coord[0] - other[0]), abs(coord[1] - other[1])) >= spacing
                for other in selected
            ):
                selected.append(coord)
                if len(selected) >= count:
                    return selected
    return None


def _cycle_pairs(items: list[str]) -> list[tuple[str, str]]:
    if len(items) <= 1:
        return []
    return [
        (items[idx], items[(idx + 1) % len(items)])
        for idx in range(len(items))
    ]


def _sample_random_line(
    *,
    start: Coord,
    end: Coord,
    width: int,
    height: int,
    margin: int,
    rng: random.Random,
    previous_line: list[Coord] | None = None,
) -> list[Coord]:
    del width, height, margin
    if start == end:
        return [start]

    if not previous_line or previous_line[-1] != start:
        return _staircase_path(start, end, rng)

    shortest_length = _manhattan_distance(start, end)
    reversed_previous = list(reversed(previous_line))
    valid_turn_indices = [
        idx
        for idx, coord in enumerate(reversed_previous)
        if idx + _manhattan_distance(coord, end) == shortest_length
    ]
    if not valid_turn_indices:
        return _staircase_path(start, end, rng)

    max_shared_steps = max(valid_turn_indices)
    best_turn_indices = [idx for idx in valid_turn_indices if idx == max_shared_steps]
    turn_idx = rng.choice(best_turn_indices)
    shared_prefix = reversed_previous[: turn_idx + 1]

    cells = list(shared_prefix)
    _append_path_cells(cells, _staircase_path(shared_prefix[-1], end, rng))
    return cells


def _clip_cells_to_grid(cells: set[Coord], width: int, height: int) -> set[Coord]:
    return {
        (x, y)
        for x, y in cells
        if 0 <= x < width and 0 <= y < height
    }


def _target_dimensions(
    target_area: int,
    aspect_ratio: float,
) -> tuple[int, int]:
    if aspect_ratio <= 0:
        raise ValueError(f"grid_aspect_ratio must be > 0, got {aspect_ratio}")

    target_width = max(1, int(math.ceil(math.sqrt(target_area * aspect_ratio))))
    target_height = max(1, int(math.ceil(target_area / target_width)))

    alt_height = max(1, int(math.ceil(math.sqrt(target_area / aspect_ratio))))
    alt_width = max(1, int(math.ceil(target_area / alt_height)))

    candidates = (
        (target_width, target_height),
        (alt_width, alt_height),
    )
    return min(
        candidates,
        key=lambda dims: (
            abs((dims[0] / dims[1]) - aspect_ratio),
            dims[0] * dims[1],
            dims[0],
            dims[1],
        ),
    )


def _build_spatial_layout(
    *,
    task_graph: TaskGraph,
    knobs: ConstructiveKnobs,
    seed: int,
    anchor_spacing: int | None = None,
) -> tuple[GridSpec, dict[Coord, set[str]], dict[str, object]]:
    # V3-rl can calibrate compact layouts; existing callers retain their exact
    # spacing rule and random-number sequence when the override is omitted.
    if anchor_spacing is not None and (
        isinstance(anchor_spacing, bool)
        or not isinstance(anchor_spacing, int)
        or anchor_spacing < 1
    ):
        raise ValueError("anchor_spacing must be a positive integer or None")
    rng = random.Random(seed + 29)
    width_probs = _normalize_prob_map(knobs.path_width_probs, name="path_width_probs")
    max_path_width = max(width for width, _ in width_probs)
    if knobs.dag_density_over_grid <= 0:
        raise ValueError(
            "dag_density_over_grid must be > 0, "
            f"got {knobs.dag_density_over_grid}"
        )
    target_area = max(knobs.num_nodes, int((knobs.num_nodes / knobs.dag_density_over_grid) + 0.999999))
    margin = max(1, (max_path_width + 1) // 2)
    coord_spacing = max(2, max_path_width) if anchor_spacing is None else anchor_spacing
    width, height = _target_dimensions(target_area, knobs.grid_aspect_ratio)
    width = max(width, (2 * margin) + 1)
    height = max(height, (2 * margin) + 1)

    required_coord_count = len(task_graph.nodes) + 1
    sampled_coords = _sample_random_coords(
        width=width,
        height=height,
        margin=0,
        count=required_coord_count,
        spacing=coord_spacing,
        rng=rng,
    )
    while sampled_coords is None:
        if (width / height) <= knobs.grid_aspect_ratio:
            width += 1
        else:
            height += 1
        sampled_coords = _sample_random_coords(
            width=width,
            height=height,
            margin=0,
            count=required_coord_count,
            spacing=coord_spacing,
            rng=rng,
        )

    start = sampled_coords[0]
    node_anchor_coords = sampled_coords[1:]

    node_ids = sorted(task_graph.nodes)
    node_positions = {
        node_id: coord
        for node_id, coord in zip(node_ids, node_anchor_coords)
    }

    cycle_order = list(node_ids)
    rng.shuffle(cycle_order)
    cycle_pairs = _cycle_pairs(cycle_order)

    open_cells: set[Coord] = set()
    cell_widths: dict[Coord, int] = {coord: 1 for coord in [start, *node_positions.values()]}

    cycle_traversal: list[Coord] = []
    previous_line: list[Coord] | None = None
    for node_a, node_b in cycle_pairs:
        line_cells = _sample_random_line(
            start=node_positions[node_a],
            end=node_positions[node_b],
            width=width,
            height=height,
            margin=margin,
            rng=rng,
            previous_line=previous_line,
        )
        _append_path_cells(cycle_traversal, line_cells)
        width_value = _sample_from_prob_map(rng, width_probs)
        for seg_start, seg_end in zip(line_cells, line_cells[1:]):
            open_cells |= _clip_cells_to_grid(_inflate_segment(seg_start, seg_end, width_value), width, height)
        for coord in line_cells:
            cell_widths[coord] = max(cell_widths.get(coord, 1), width_value)
        previous_line = line_cells

    if cycle_order:
        start_target = node_positions[cycle_order[0]]
    else:
        start_target = start
    start_line = _sample_random_line(
        start=start,
        end=start_target,
        width=width,
        height=height,
        margin=margin,
        rng=rng,
    )
    start_width = _sample_from_prob_map(rng, width_probs)
    for seg_start, seg_end in zip(start_line, start_line[1:]):
        open_cells |= _clip_cells_to_grid(_inflate_segment(seg_start, seg_end, start_width), width, height)
    for coord in start_line:
        cell_widths[coord] = max(cell_widths.get(coord, 1), start_width)

    if not cycle_pairs and not start_line:
        open_cells.add(start)

    for coord, width_value in cell_widths.items():
        open_cells |= _clip_cells_to_grid(_inflate_point(coord, width_value), width, height)

    goal_depth = max(node.depth for node in task_graph.nodes.values())

    blocked = {
        (x, y)
        for y in range(height)
        for x in range(width)
        if (x, y) not in open_cells
    }
    if not _is_grid_connected(width, height, blocked, start):
        raise RuntimeError("Generated grid is not connected.")

    discovery_map = {
        coord: {node_id}
        for node_id, coord in sorted(node_positions.items())
    }
    metadata = {
        "backbone_style": "shared_shortest_cycle_walk",
        "spatial_cycle_order": list(cycle_order),
        "task_depth": goal_depth,
        "primitive_count": sum(1 for node in task_graph.nodes.values() if node.depth == 0),
        "grid_width": width,
        "grid_height": height,
        "target_grid_aspect_ratio": knobs.grid_aspect_ratio,
        "actual_grid_aspect_ratio": width / height,
        "target_area": target_area,
        "actual_open_cells": len(open_cells),
    }
    grid = GridSpec(
        width=width,
        height=height,
        blocked=blocked,
        start=start,
    )
    return grid, discovery_map, metadata


def _validate_single_cell_per_node(
    discovery_map: dict[Coord, set[str]],
    required_nodes: set[str],
) -> None:
    node_to_cells: dict[str, set[Coord]] = {node_id: set() for node_id in required_nodes}
    for cell, states in discovery_map.items():
        for node_id in states:
            if node_id in node_to_cells:
                node_to_cells[node_id].add(cell)

    missing = sorted(node_id for node_id, cells in node_to_cells.items() if len(cells) == 0)
    duplicated = sorted(node_id for node_id, cells in node_to_cells.items() if len(cells) > 1)
    if missing:
        raise RuntimeError(f"Required nodes were not placed on grid: {missing}")
    if duplicated:
        raise RuntimeError(f"Required nodes were placed on multiple cells: {duplicated}")


def _generate_constructive_episode(
    task_dag_difficulty: TaskDAGDifficulty,
    exploration_level: ExplorationLevel,
    seed: int,
    knobs: ConstructiveKnobs,
) -> EpisodeInstance:

    task_graph = construct_task_graph(knobs=knobs, seed=seed)
    required_nodes = set(task_graph.nodes.keys())
    grid, discovery_map, layout_metadata = _build_spatial_layout(
        task_graph=task_graph,
        knobs=knobs,
        seed=seed,
    )
    _validate_single_cell_per_node(discovery_map=discovery_map, required_nodes=required_nodes)

    hidden_budget = int(layout_metadata["actual_open_cells"] * 3)
    episode_id = (
        f"exploration_{exploration_level.value}-"
        f"task_dag_{task_dag_difficulty.value}-s{seed:04d}"
    )
    metadata = {
        "backbone_style": layout_metadata["backbone_style"],
        "spatial_cycle_order": layout_metadata["spatial_cycle_order"],
        "task_depth": layout_metadata["task_depth"],
        "primitive_count": layout_metadata["primitive_count"],
        "exploration_level": exploration_level.value,
        "task_dag_difficulty": task_dag_difficulty.value,
        "grid_width": grid.width,
        "grid_height": grid.height,
        "target_grid_aspect_ratio": layout_metadata["target_grid_aspect_ratio"],
        "actual_grid_aspect_ratio": layout_metadata["actual_grid_aspect_ratio"],
        "target_area": layout_metadata["target_area"],
        "actual_open_cells": layout_metadata["actual_open_cells"],
    }

    return EpisodeInstance(
        episode_id=episode_id,
        exploration_level=exploration_level,
        task_dag_difficulty=task_dag_difficulty,
        seed=seed,
        knobs=knobs,
        grid=grid,
        task_graph=task_graph,
        discovery_map=discovery_map,
        required_nodes=required_nodes,
        hidden_budget=hidden_budget,
        metadata=metadata,
    )

