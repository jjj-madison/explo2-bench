"""Layered task DAGs with a mandatory backbone and optional scheduling choices."""

from __future__ import annotations

import random

from ..types import TaskGraph, TaskNode
from .types import V3Knobs


def _visible_names(node_ids: list[str], rng: random.Random) -> dict[str, str]:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    used = set()
    result = {}
    for node_id in node_ids:
        while True:
            name = "".join(rng.choice(alphabet) for _ in range(4))
            if name not in used:
                used.add(name)
                result[node_id] = name
                break
    return result


def _required_ancestors(node_ids: set[str], nodes: dict[str, TaskNode]) -> set[str]:
    """Known mandatory ancestors, used to avoid redundant added dependencies."""
    required = set(node_ids)
    frontier = list(node_ids)
    while frontier:
        node = nodes[frontier.pop()]
        for parent in set.intersection(*node.prerequisites):
            if parent not in required:
                required.add(parent)
                frontier.append(parent)
    return required


def construct_task_graph(knobs: V3Knobs, seed: int) -> TaskGraph:
    """Permute one seed-selected layer template independently of spatial controls.

    Non-goal layer widths are shuffled using a separate seeded stream. This
    preserves the selected template's node count, depth, and maximum width
    while allowing the graph to widen and narrow before its singleton goal.

    Every node in a layer is a mandatory parent of some node in the next layer.
    Every nonprimitive has an immediate previous-layer mandatory parent, and
    the goal requires the whole final non-goal layer. Consequently every task
    must activate before success, and declared depths are actual graph depths.
    Extra dependencies and OR alternatives affect permissible activation orders
    without changing that requirement.
    """
    if not isinstance(knobs, V3Knobs):
        raise TypeError("construct_task_graph requires V3Knobs")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(seed + 11)
    template = rng.choice(knobs.layer_profiles)
    non_goal_widths = list(template[:-1])
    random.Random(f"symbolic-v3-layer-widths:{seed}").shuffle(non_goal_widths)
    profile = (*non_goal_widths, 1)
    layers = [
        ["G_00"] if depth == len(profile) - 1 else [f"T_{depth:02d}_{index:02d}" for index in range(width)]
        for depth, width in enumerate(profile)
    ]
    visible = _visible_names([node for layer in layers for node in layer], random.Random(seed + 17))
    nodes = {
        node_id: TaskNode(node_id, visible[node_id], [set()], 0)
        for node_id in layers[0]
    }
    for depth, children in enumerate(layers[1:], 1):
        previous = list(layers[depth - 1])
        if depth == len(layers) - 1:
            nodes["G_00"] = TaskNode("G_00", visible["G_00"], [set(previous)], depth)
            continue
        rng.shuffle(previous)
        child_order = list(children)
        rng.shuffle(child_order)
        mandatory = {node: set() for node in children}
        for index, parent in enumerate(previous):
            mandatory[child_order[index % len(children)]].add(parent)
        for child in children:
            if not mandatory[child]:
                mandatory[child].add(rng.choice(previous))
            base = mandatory[child]
            already_required = _required_ancestors(base, nodes)
            candidates = sorted(node_id for node_id, node in nodes.items() if node.depth < depth and node_id not in already_required)
            extras = set()
            if candidates and knobs.max_extra_parents and rng.random() < knobs.extra_dependency_probability:
                count = rng.randint(1, min(knobs.max_extra_parents, len(candidates)))
                extras = set(rng.sample(candidates, count))
            options = [base | extras]
            alternatives = sorted(set(candidates) - extras)
            if extras and alternatives and rng.random() < knobs.alternative_probability:
                alternative = set(extras)
                alternative.remove(rng.choice(sorted(alternative)))
                alternative.add(rng.choice(alternatives))
                options.append(base | alternative)
            options.sort(key=lambda option: tuple(sorted(option)))
            nodes[child] = TaskNode(child, visible[child], options, depth)
    return TaskGraph(nodes, "G_00")


__all__ = ["construct_task_graph"]
