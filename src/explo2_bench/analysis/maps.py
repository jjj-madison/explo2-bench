"""Structural measurements of complete maps, independent of action budgets."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Iterable

from ..data import episode_from_record, load_benchmark
from .diversity import DAGRegistry

ANALYSIS_VERSION = 1


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fingerprints(episode: Any) -> dict[str, str]:
    """Hash distinct equivalence definitions without budget or seed metadata.

    Coordinates use a fixed orientation: translations/rotations are not merged.
    Exact gameplay retains visible names and task IDs, so graph classes below
    are the separate name-independent comparison.
    """
    walls = {"width": episode.grid.width, "height": episode.grid.height,
             "blocked": sorted(episode.grid.blocked)}
    occupied = sorted(cell for cell, nodes in episode.discovery_map.items() if nodes)
    graph = episode.task_graph
    gameplay = {
        "walls": walls, "start": episode.grid.start,
        "goal_id": graph.goal_id,
        "required_nodes": sorted(episode.required_nodes),
        "nodes": {node_id: {
            "visible_name": node.visible_name,
            "prerequisites": sorted(sorted(option) for option in node.prerequisites),
        } for node_id, node in graph.nodes.items()},
        "discovery_map": {f"{x},{y}": sorted(nodes)
                          for (x, y), nodes in episode.discovery_map.items() if nodes},
    }
    return {"gameplay_sha256": _hash(gameplay), "wall_layout_sha256": _hash(walls),
            "task_coordinates_sha256": _hash(occupied)}


def _distances(grid: Any, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    distance = {start: 0}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in grid.neighbors(cell):
            if neighbor not in distance:
                distance[neighbor] = distance[cell] + 1
                queue.append(neighbor)
    return distance


def _distribution(prefix: str, values: Iterable[int | float]) -> dict[str, Any]:
    values = list(values)
    return {f"{prefix}_count": len(values),
            f"{prefix}_min": min(values) if values else None,
            f"{prefix}_mean": fmean(values) if values else None,
            f"{prefix}_median": median(values) if values else None,
            f"{prefix}_max": max(values) if values else None}


def analyze_map(record: dict[str, Any], *, registry: DAGRegistry | None = None) -> dict[str, Any]:
    """Measure one fixed record; missing distances are null, never zero-filled."""
    episode = episode_from_record(record)
    grid, graph = episode.grid, episode.task_graph
    cells = set(grid.open_cells())
    if not cells or grid.start not in cells:
        raise ValueError("Map needs an open starting cell")
    degrees = {cell: len(grid.neighbors(cell)) for cell in cells}
    start_distances = _distances(grid, grid.start)
    unseen = set(cells)
    components = 0
    while unseen:
        components += 1
        unseen.difference_update(_distances(grid, min(unseen)))
    edges = sum(degrees.values()) // 2

    parents = {node_id: set().union(*node.prerequisites)
               for node_id, node in graph.nodes.items()}
    if any(parent not in graph.nodes for value in parents.values() for parent in value):
        raise ValueError("Task prerequisites reference absent nodes")
    depths: dict[str, int] = {}
    pending = set(graph.nodes)
    while pending:
        ready = sorted(node_id for node_id in pending if parents[node_id] <= depths.keys())
        if not ready:
            raise ValueError("Task dependencies contain a directed cycle")
        for node_id in ready:
            depths[node_id] = max((depths[parent] + 1 for parent in parents[node_id]), default=0)
        pending.difference_update(ready)
    layer_counts = Counter(depths.values())
    options = [option for node in graph.nodes.values() for option in node.prerequisites]

    locations: dict[str, tuple[int, int]] = {}
    for cell, nodes in episode.discovery_map.items():
        if cell not in cells:
            raise ValueError("Task location is not an open cell")
        for node_id in nodes:
            if node_id in locations:
                raise ValueError("A task has multiple spatial locations")
            locations[node_id] = cell
    if locations.keys() != graph.nodes.keys():
        raise ValueError("Every task must have exactly one location")
    task_ids = sorted(graph.nodes)
    task_distances: list[int] = []
    unreachable_pairs = 0
    cache: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    for i, node_id in enumerate(task_ids):
        cell = locations[node_id]
        if cell not in cache:
            cache[cell] = _distances(grid, cell)
        distances = cache[cell]
        for other_id in task_ids[i + 1:]:
            value = distances.get(locations[other_id])
            if value is None:
                unreachable_pairs += 1
            else:
                task_distances.append(value)
    generation = record["generation"]
    row = {
        "map_id": record["map_id"], "version": record["version"],
        "exploration_profile": generation["exploration_profile"],
        "dag_profile": generation["dag_profile"], "generation_seed": generation["seed"],
        "split": record.get("split") or "", "episode_sha256": record["episode_sha256"],
        "width": grid.width, "height": grid.height, "total_cells": grid.width * grid.height,
        "open_cells": len(cells), "blocked_cells": grid.width * grid.height - len(cells),
        "open_fraction": len(cells) / (grid.width * grid.height),
        "connected_components": components, "reachable_cells": len(start_distances),
        "grid_edges": edges, "isolated_cells": sum(degree == 0 for degree in degrees.values()),
        "dead_ends": sum(degree == 1 for degree in degrees.values()),
        "junctions": sum(degree >= 3 for degree in degrees.values()),
        "cycle_rank": edges - len(cells) + components,
        "task_nodes": len(graph.nodes), "non_goal_tasks": len(graph.nodes) - 1,
        "primitive_tasks": sum(not value for value in parents.values()),
        "initially_ready_tasks": sum(not node.prerequisites or any(not option for option in node.prerequisites)
                                     for node in graph.nodes.values()),
        "task_sites": len(set(locations.values())), "task_density": len(graph.nodes) / len(cells),
        "dependency_edges": sum(map(len, parents.values())),
        "prerequisite_groups": len(options), "empty_groups": sum(not option for option in options),
        "and_groups": sum(len(option) >= 2 for option in options),
        "or_tasks": sum(len(node.prerequisites) >= 2 for node in graph.nodes.values()),
        "group_fanin_mean": fmean(map(len, options)) if options else 0.0,
        "group_fanin_max": max(map(len, options), default=0),
        "node_fanin_mean": fmean(map(len, parents.values())),
        "node_fanin_max": max(map(len, parents.values()), default=0),
        "dag_depth": max(depths.values()), "goal_depth": depths[graph.goal_id],
        "dag_layers": max(depths.values()) + 1, "max_layer_width": max(layer_counts.values()),
        "layer_widths": [layer_counts[depth] for depth in range(max(depths.values()) + 1)],
        "start_goal_distance": start_distances.get(locations[graph.goal_id]),
        "unreachable_tasks": sum(cell not in start_distances for cell in locations.values()),
        "unreachable_task_pairs": unreachable_pairs,
        **_distribution("start_task_distance", (start_distances[cell] for cell in locations.values()
                                               if cell in start_distances)),
        **_distribution("task_pair_distance", task_distances),
        **fingerprints(episode),
    }
    if registry is not None:
        row["dag_class"] = registry.add(graph)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty analysis table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def summarize_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = {"generation_seed", "dag_class"}
    metrics = [key for key, value in rows[0].items()
               if isinstance(value, (int, float)) and key not in excluded]
    output = []
    for (exploration, dag), members in sorted(groups_from_rows(rows).items()):
        row = {"version": members[0]["version"], "exploration_profile": exploration,
               "dag_profile": dag, "maps": len(members)}
        for metric in metrics:
            values = [member[metric] for member in members if member[metric] is not None]
            row.update({f"{metric}_{statistic}": value for statistic, value in {
                "n": len(values), "mean": fmean(values) if values else None,
                "std": pstdev(values) if values else None, "min": min(values) if values else None,
                "median": median(values) if values else None, "max": max(values) if values else None,
            }.items()})
        output.append(row)
    return output


def groups_from_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["exploration_profile"], row["dag_profile"])].append(row)
    return groups


def _diversity_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"maps": len(rows)}
    for name, field in (("gameplay", "gameplay_sha256"), ("wall_layouts", "wall_layout_sha256"),
                        ("task_coordinate_sets", "task_coordinates_sha256"), ("dag_classes", "dag_class")):
        classes: dict[Any, list[str]] = defaultdict(list)
        for row in rows:
            classes[row[field]].append(row["map_id"])
        report[name] = {"distinct": len(classes), "duplicate_excess": len(rows) - len(classes),
                        "largest_class": max(map(len, classes.values()), default=0)}
        if name == "gameplay":
            report[name]["duplicate_groups"] = [ids for ids in classes.values() if len(ids) > 1]
    return report


def split_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count structural reuse across assigned splits without equating it with leakage."""
    groups = {split: [row for row in rows if row.get("split") == split]
              for split in ("train", "validation", "test")}
    report: dict[str, Any] = {"split_sizes": {split: len(members) for split, members in groups.items()},
                              "unassigned_maps": sum(not row.get("split") for row in rows), "pairs": {}}
    for left, right in combinations(groups, 2):
        if not groups[left] or not groups[right]:
            continue
        overlaps = {}
        for name, field in (("gameplay", "gameplay_sha256"), ("wall_layouts", "wall_layout_sha256"),
                            ("task_coordinate_sets", "task_coordinates_sha256"), ("dag_classes", "dag_class")):
            first, second = Counter(row[field] for row in groups[left]), Counter(row[field] for row in groups[right])
            shared = first.keys() & second.keys()
            overlaps[name] = {"shared_classes": len(shared),
                              "left_maps": sum(first[key] for key in shared),
                              "right_maps": sum(second[key] for key in shared),
                              "cross_split_map_pairs": sum(first[key] * second[key] for key in shared)}
        report["pairs"][f"{left}/{right}"] = overlaps
    return report


def split_report_lines(report: dict[str, Any]) -> list[str]:
    if not report["pairs"]:
        return []
    lines = ["## Overlap between dataset splits", "",
             "The table compares the assigned training, validation, and test subsets. "
             "Unassigned maps are excluded. Each entry counts structural classes shared by two splits; "
             "see the [split definitions](../../../docs/benchmark.md#split-memberships) for membership counts.", "",
             "| Split pair | Shared gameplay identities | Shared wall layouts | Shared task-coordinate sets | Shared DAG classes |",
             "|---|---:|---:|---:|---:|"]
    for pair, overlaps in report["pairs"].items():
        counts = [overlaps[key]["shared_classes"] for key in
                  ("gameplay", "wall_layouts", "task_coordinate_sets", "dag_classes")]
        lines.append(f"| {pair} | {' | '.join(map(str, counts))} |")
    lines += ["", "Distinct map IDs can share dependency structures. Wall layouts, task-coordinate sets, "
              "and task graphs describe separate components of a map. Their overlap measures structural reuse; "
              "it does not establish whether a model's training data contained these maps.", ""]
    return lines


def analyze_benchmark(version: str, *, root: str | Path | None = None,
                      output_root: str | Path = "analysis/maps") -> dict[str, Any]:
    """Write complete structural tables, exact diversity classes, and provenance."""
    registry = DAGRegistry()
    rows = [analyze_map(record, registry=registry) for record in load_benchmark(version, root=root)]
    if not rows:
        raise ValueError(f"No {version} maps found")
    output = Path(output_root) / version
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_map.csv", rows)
    write_csv(output / "by_profile.csv", summarize_profiles(rows))
    diversity = {"version": version, "complete": True,
                 "dag_equivalence": "exact directed isomorphism preserving goal and AND/OR groups",
                 "global": _diversity_counts(rows),
                 "split_overlap": split_overlap(rows),
                 "by_profile": {f"{exploration}/{dag}": _diversity_counts(members)
                                for (exploration, dag), members in sorted(groups_from_rows(rows).items())}}
    (output / "diversity.json").write_text(json.dumps(diversity, indent=2) + "\n")
    files = ["per_map.csv", "by_profile.csv", "diversity.json"]
    manifest = {"analysis_version": ANALYSIS_VERSION, "version": version, "map_count": len(rows),
                "budget_independent": True, "full_inventory": True,
                "input_inventory_sha256": _hash([(row["map_id"], row["episode_sha256"]) for row in rows]),
                "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_map_report(version, rows, diversity, output)
    return {"version": version, "maps": len(rows), "dag_classes": len(registry), "output": str(output)}


def write_map_report(version: str, rows: list[dict[str, Any]], diversity: dict[str, Any],
                     output: str | Path) -> None:
    """Describe existing structural measurements without recomputing the analysis."""
    benchmark_name = f"(EXPLO)²-Bench {version}"
    lines = [f"# {benchmark_name} map analysis", "",
             "Agents explore a partially observable 2D grid to discover task nodes and their dependencies, "
             "then use the knowledge acquired through past encounters to complete the goal. "
             "The maps vary both the space to explore and the structure of the task dependencies.", "",
             f"This report describes all **{len(rows):,} fixed maps** in {benchmark_name}. "
             "Structural measurements use the complete maps and task DAGs and are independent of action "
             "budgets and agent behavior. See the [map and task-DAG examples](examples/README.md) for "
             "complete views of individual environments.", "",
             "## Environment structure", "",
             "Open cells are traversable grid cells. Task nodes include the goal, and their prerequisites "
             "form a directed acyclic graph (DAG). Exploration-demand and task-DAG difficulty profiles "
             "specify the generator settings used to construct each map.", "",
             "| Structural measurement | Minimum | Mean | Maximum |",
             "|---|---:|---:|---:|"]
    for metric, label in (
        ("open_cells", "Open cells (count)"),
        ("task_nodes", "Task nodes, including goal (count)"),
        ("dependency_edges", "Task-dependency edges (count)"),
        ("dag_depth", "Longest dependency path (edges)"),
        ("max_layer_width", "Largest dependency layer (nodes)"),
        ("dead_ends", "Grid dead ends (count)"),
        ("junctions", "Grid junctions (count)"),
        ("cycle_rank", "Independent grid cycles (count)"),
        ("start_goal_distance", "Start-to-goal distance (moves)"),
        ("task_pair_distance_mean", "Mean task-pair distance (moves)"),
    ):
        values = [float(row[metric]) for row in rows if row[metric] not in (None, "")]
        lines.append(f"| {label} | {min(values):.2f} | {fmean(values):.2f} | {max(values):.2f} |")
    counts = diversity["global"]
    lines.extend(["", "## Structural diversity", "",
                  f"There are **{counts['gameplay']['distinct']:,} distinct gameplay records**, "
                  f"**{counts['wall_layouts']['distinct']:,} wall layouts**, and "
                  f"**{counts['dag_classes']['distinct']:,} task-DAG classes** under the documented equivalence definitions.",
                  "", "Gameplay identity compares the grid, starting cell, task names and placements, and "
                  "dependencies, while excluding generation metadata and action budgets. Wall-layout identity "
                  "compares grid dimensions and blocked cells. Task-coordinate sets compare occupied task "
                  "cells, ignoring walls and task identities. Task-DAG classes group equivalent goal and "
                  "AND/OR dependency structures, ignoring task names and spatial positions.", "",
                  "These counts describe map and task structure. Profile names identify generator settings "
                  "and do not establish increasing difficulty for an agent. Dependency depth includes all "
                  "prerequisite alternatives, so it does not give the minimum number of tasks needed to "
                  "activate the goal. Spatial distances do not account for activation order.", "",
                  "See [metric definitions](../../../docs/map-analysis.md), [per-map data](per_map.csv), "
                  "[setting summaries](by_profile.csv), [diversity results](diversity.json), "
                  "and [provenance checksums](manifest.json).", ""])
    lines.extend(split_report_lines(diversity["split_overlap"]))
    (Path(output) / "README.md").write_text("\n".join(lines).rstrip() + "\n")
