"""Render full maps and AND/OR task graphs as independent publication panels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from .data import episode_from_record, load_benchmark


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    return plt


def _save(fig, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=180)


def render_map(record: dict[str, Any], output: str | Path) -> None:
    """Render the full privileged map, with native cell coordinates and labels."""
    plt = _plotting()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import numpy as np

    episode = episode_from_record(record)
    grid, graph = episode.grid, episode.task_graph
    tiles = np.zeros((grid.height, grid.width))
    for x, y in grid.blocked:
        tiles[y, x] = 1
    for (x, y), nodes in episode.discovery_map.items():
        if nodes:
            tiles[y, x] = 4 if graph.goal_id in nodes else 3
    tiles[grid.start[1], grid.start[0]] = 2
    fig, ax = plt.subplots(figsize=(7.5, 7.2), constrained_layout=True)
    colors = ["#ffffff", "#353941", "#8dcca1", "#a9ccea", "#f2c65e"]
    ax.imshow(tiles, cmap=ListedColormap(colors), vmin=0, vmax=4,
              interpolation="nearest", origin="lower")
    label_size = min(8, 105 / max(grid.width, grid.height))
    for (x, y), nodes in episode.discovery_map.items():
        label = ",".join(graph.nodes[node_id].visible_name for node_id in sorted(nodes))
        ax.text(x, y, label, ha="center", va="center", fontsize=label_size, fontweight="bold")
    ax.text(*grid.start, "S", ha="center", va="center", fontsize=9, fontweight="bold")
    spacing = max(1, max(grid.width, grid.height) // 12)
    ax.set_xticks(range(0, grid.width, spacing))
    ax.set_yticks(range(0, grid.height, spacing))
    ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    generation = record["generation"]
    ax.set(xlabel="x (cell coordinate)", ylabel="y (cell coordinate)",
           title=f"(EXPLO)²-Bench {record['version']} full map\n"
                 f"{generation['exploration_profile']} exploration / {generation['dag_profile']} task DAG\n"
                 f"seed {generation['seed']}; {len(grid.open_cells())} open cells; {len(graph.nodes)} task nodes")
    ax.legend(handles=[Patch(facecolor=colors[index], label=label) for index, label in
                       [(1, "Wall"), (2, "Start (S)"), (3, "Task"), (4, "Goal")]],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    _save(fig, output)
    plt.close(fig)


def render_task_graph(record: dict[str, Any], output: str | Path) -> None:
    """Keep prerequisite options as explicit junctions; never flatten OR choices."""
    plt = _plotting()
    from matplotlib.patches import Circle

    graph = episode_from_record(record).task_graph
    parents = {node_id: set().union(*node.prerequisites) for node_id, node in graph.nodes.items()}
    depths: dict[str, int] = {}
    while len(depths) < len(graph.nodes):
        ready = sorted(node_id for node_id in graph.nodes if node_id not in depths
                       and parents[node_id] <= depths.keys())
        if not ready:
            raise ValueError("Task graph is cyclic")
        for node_id in ready:
            depths[node_id] = max((depths[parent] + 1 for parent in parents[node_id]), default=0)
    layers: dict[int, list[str]] = defaultdict(list)
    for node_id in sorted(graph.nodes):
        layers[depths[node_id]].append(node_id)
    positions = {node_id: (2.0 * depth, 1.8 * (index - (len(nodes) - 1) / 2))
                 for depth, nodes in layers.items() for index, node_id in enumerate(nodes)}
    max_depth, max_width = max(layers), max(map(len, layers.values()))
    fig, ax = plt.subplots(figsize=(max(7.5, 1.8 * (max_depth + 1)), max(4.5, 1.5 * max_width)),
                            constrained_layout=True)
    for node_id, node in graph.nodes.items():
        x, y = positions[node_id]
        options = sorted((sorted(option) for option in node.prerequisites if option))
        for i, option in enumerate(options):
            junction = (x - 0.62, y + 0.40 * (i - (len(options) - 1) / 2))
            for parent in option:
                px, py = positions[parent]
                ax.annotate("", xy=junction, xytext=(px + 0.24, py), zorder=1,
                            arrowprops={"arrowstyle": "->", "color": "#606874", "linewidth": 0.8,
                                        "connectionstyle": "arc3,rad=0.04"})
            ax.annotate("", xy=(x - 0.24, y), xytext=junction, zorder=1,
                        arrowprops={"arrowstyle": "->", "color": "#606874", "linewidth": 0.8})
            ax.plot(*junction, marker="D", markersize=11, color="white", markeredgecolor="#606874", zorder=2)
            ax.text(*junction, "∧" if len(option) > 1 else "·", fontsize=8,
                    ha="center", va="center", zorder=3)
        color = "#f2c65e" if node_id == graph.goal_id else "#a9ccea"
        ax.add_patch(Circle((x, y), 0.24, facecolor=color, edgecolor="#285f9b", linewidth=1.2, zorder=3))
        ax.text(x, y, node.visible_name, ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
        if len(options) > 1:
            ax.text(x, y - 0.43, "OR", ha="center", va="center", color="#606874", fontsize=8)
    generation = record["generation"]
    ax.set(xlim=(-0.75, 2 * max_depth + 0.75),
           ylim=(-0.9 * max_width - 0.4, 0.9 * max_width + 0.4),
           title=f"(EXPLO)²-Bench {record['version']} task graph\n"
                 f"{generation['exploration_profile']} exploration / {generation['dag_profile']} task DAG\n"
                 f"seed {generation['seed']}; {len(graph.nodes)} tasks; gold node is the goal",
           xlabel="Dependency depth (longest path in edges)\n\n"
                  "Arrows: prerequisite → task. Diamonds collect one option (∧: all parents).\n"
                  "Multiple options are alternatives (OR); vertical placement is for readability.")
    ax.set_xticks([2 * value for value in range(max_depth + 1)], list(range(max_depth + 1)))
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, output)
    plt.close(fig)


def render_examples(output_root: str | Path = "analysis/maps", root: str | Path | None = None) -> list[dict]:
    """Render the first recorded seed at three named profiles per version."""
    selected = []
    for version in ("v2", "v3"):
        targets = {("lower", "easier"), ("medium", "medium"), ("higher", "harder")}
        for record in load_benchmark(version, root=root):
            generation = record["generation"]
            pair = (generation["exploration_profile"], generation["dag_profile"])
            if pair not in targets:
                continue
            targets.remove(pair)
            directory = Path(output_root) / version / "examples"
            basename = f"{pair[0]}_{pair[1]}"
            render_map(record, directory / f"{basename}_map")
            render_task_graph(record, directory / f"{basename}_dag")
            selected.append({"version": version, "map_id": record["map_id"],
                             "episode_sha256": record["episode_sha256"],
                             "selection": "first inventory record at named profile", "basename": basename})
            if not targets:
                break
        if targets:
            raise ValueError(f"Missing example settings: {sorted(targets)}")
        directory = Path(output_root) / version / "examples"
        entries = [item for item in selected if item["version"] == version]
        (directory / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
        write_examples_report(version, entries, directory)
    return selected


def write_examples_report(version: str, entries: list[dict], output: str | Path) -> None:
    """Describe the recorded example selection and embed its existing panels."""
    benchmark_name = f"(EXPLO)²-Bench {version}"
    lines = [f"# {benchmark_name} map and task-DAG examples", "",
             "An agent discovers task nodes as it moves through a partially observable 2D grid. "
             "It uses the task dependencies learned through these encounters to satisfy prerequisites "
             "and activate the goal. Each example below shows a complete map and its matching task DAG, "
             "with the same visible task names in both panels.", "",
             "These complete views are for human inspection. Evaluated agents receive partial "
             "observations, not the full map or task graph. Examples use the first seed listed in the "
             "benchmark's seed table at each of three named settings, without selecting for appearance or success. "
             "The profile names identify generator settings rather than measured agent difficulty.", "",
             "In the maps, dark cells are walls, green marks the start, blue marks task locations, "
             "and gold marks the goal. In the DAGs, arrows run from prerequisite to dependent task. "
             "A diamond groups prerequisites that must all be satisfied (AND); separate groups "
             "provide alternatives (OR).", ""]
    for item in entries:
        name = item["basename"]
        exploration, dag = name.split("_", 1)
        seed = int(item["map_id"].rsplit("-s", 1)[1])
        lines += [f"## {exploration.capitalize()} exploration demand · {dag.capitalize()} task DAG", "",
                  f"Map `{item['map_id']}` · generation seed `{seed}`.", "",
                  f"![Complete {benchmark_name} grid map with {exploration} exploration demand and task-DAG profile {dag}, "
                  f"seed {seed}.]({name}_map.png)", "",
                  "The complete grid shows the traversable space, task locations, start, and goal.", "",
                  f"![Task dependencies matching {item['map_id']}, with prerequisite groups and the goal "
                  f"shown explicitly.]({name}_dag.png)", "",
                  "The matching DAG shows which task prerequisites must be satisfied before the goal "
                  "can activate.", ""]
    lines += ["See the [map analysis](../README.md) for the full benchmark's structural measurements "
              "and the [example manifest](manifest.json) for map identities and checksums.", ""]
    (Path(output) / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("analysis/maps"))
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    print(json.dumps(render_examples(args.output, args.root), indent=2))
