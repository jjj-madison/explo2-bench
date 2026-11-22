"""Single-panel figures with shared scales across benchmark versions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

EXPLORATION_ORDER = ["x-lowest", "lowest", "lower", "low", "medium", "high", "higher", "highest", "x-highest"]
DAG_ORDER = ["x-easiest", "easiest", "easier", "easy", "medium", "hard", "harder", "hardest", "x-hardest"]
HEATMAP_METRICS = {
    "open_cells": "Open cells (count)", "task_nodes": "Task nodes, including goal (count)",
    "cycle_rank": "Independent grid cycles (count)", "dag_depth": "Longest dependency path (edges)",
    "and_groups": "Prerequisite groups with ≥2 parents (count)",
    "task_pair_distance_mean": "Mean task-pair distance (moves)",
}


def read_map_table(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in HEATMAP_METRICS:
            row[key] = float(row[key]) if row[key] else None
    return rows


def render_map_figures(rows_by_version: dict[str, list[dict[str, Any]]],
                       output_root: str | Path = "analysis/maps") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    all_rows = [row for rows in rows_by_version.values() for row in rows]
    scale_max = {metric: max(float(row[metric]) for row in all_rows if row[metric] is not None)
                 for metric in HEATMAP_METRICS}
    for version, rows in rows_by_version.items():
        benchmark_name = f"(EXPLO)²-Bench {version}"
        output = Path(output_root) / version / "figures"
        output.mkdir(parents=True, exist_ok=True)
        explorations = [value for value in EXPLORATION_ORDER if any(row["exploration_profile"] == value for row in rows)]
        dags = [value for value in DAG_ORDER if any(row["dag_profile"] == value for row in rows)]
        for metric, label in HEATMAP_METRICS.items():
            values = np.full((len(dags), len(explorations)), np.nan)
            for i, dag in enumerate(dags):
                for j, exploration in enumerate(explorations):
                    members = [float(row[metric]) for row in rows if row["dag_profile"] == dag
                               and row["exploration_profile"] == exploration and row[metric] is not None]
                    if members:
                        values[i, j] = fmean(members)
            fig, ax = plt.subplots(figsize=(8.8, 6.5), constrained_layout=True)
            heat = ax.imshow(values, origin="lower", cmap="viridis", vmin=0,
                             vmax=max(scale_max[metric], 1), aspect="auto")
            for i in range(len(dags)):
                for j in range(len(explorations)):
                    if np.isfinite(values[i, j]):
                        ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=7,
                                color="white" if values[i, j] / max(scale_max[metric], 1) < 0.5 else "black")
            ax.set_xticks(range(len(explorations)), explorations, rotation=35, ha="right")
            ax.set_yticks(range(len(dags)), dags)
            ax.set(xlabel="Exploration profile (generator label)", ylabel="Task DAG profile (generator label)",
                   title=f"{benchmark_name}\n{label}\n"
                         f"{len(rows):,} maps; means by setting; color scale shared across benchmarks")
            fig.colorbar(heat, ax=ax, label=label)
            for extension in ("png", "pdf"):
                fig.savefig(output / f"{metric}_heatmap.{extension}", dpi=180)
            plt.close(fig)
        for metric in ("open_cells", "task_pair_distance_mean"):
            values = sorted(float(row[metric]) for row in rows if row[metric] is not None)
            fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
            if values:
                ax.step([0, *values], [0, *[(i + 1) / len(values) for i in range(len(values))]],
                        where="post", color="#285f9b", linewidth=2, label=f"{benchmark_name}: {len(values):,} maps")
            ax.set(xlim=(0, max(scale_max[metric], 1) * 1.025), ylim=(0, 1.02),
                   xlabel=HEATMAP_METRICS[metric], ylabel="Fraction of maps at or below value",
                   title=f"{benchmark_name}\nFixed-map distribution")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
            ax.grid(alpha=0.2)
            ax.legend(loc="lower right", frameon=False)
            for extension in ("png", "pdf"):
                fig.savefig(output / f"{metric}_distribution.{extension}", dpi=180)
            plt.close(fig)
        (output / "scales.json").write_text(json.dumps({"versions_compared": list(rows_by_version),
            "minimum": 0, "maximum_by_metric": scale_max,
            "profile_order_is_not_a_difficulty_claim": True}, indent=2) + "\n")
        report = Path(output_root) / version / "README.md"
        if report.exists():
            write_figure_section(report, version)


def write_figure_section(report: str | Path, version: str) -> None:
    """Link existing figures without rerendering their data or image assets."""
    report = Path(report)
    benchmark_name = f"(EXPLO)²-Bench {version}"
    text = report.read_text().split("\n## Figures\n")[0].rstrip()
    text += "\n\n## Figures\n\n"
    text += ("Each heatmap shows the mean measurement across maps at a pair of exploration-demand "
             "and task-DAG difficulty profiles. The axes name generator settings; the colors "
             "describe measured structure. Each panel has separate PNG and PDF exports, "
             "with common scales across (EXPLO)²-Bench v2 and (EXPLO)²-Bench v3.\n\n")
    text += "### Space to explore\n\n"
    text += (f"![Mean open-cell count across {benchmark_name} exploration-demand and task-DAG difficulty "
             "profiles.](figures/open_cells_heatmap.png)\n\n")
    text += ("Open cells measure the traversable space in each map. Comparing profile pairs "
             "shows how the amount of space available for exploration changes across the benchmark.\n\n")
    text += "### Task dependencies\n\n"
    text += (f"![Mean task-node count, including the goal, across {benchmark_name} exploration-demand "
             "and task-DAG difficulty profiles.](figures/task_nodes_heatmap.png)\n\n")
    text += ("Task-node counts describe the size of the dependency graph. The prerequisite "
             "structure also matters: node count alone does not determine the work required "
             "to activate the goal.\n\n")
    text += "### Additional structural measurements\n\n"
    for metric, label in HEATMAP_METRICS.items():
        if metric not in ("open_cells", "task_nodes"):
            text += f"- [{label} by profile](figures/{metric}_heatmap.png).\n"
    for metric in ("open_cells", "task_pair_distance_mean"):
        text += f"- [{HEATMAP_METRICS[metric]} distribution](figures/{metric}_distribution.png).\n"
    report.write_text(text)


__all__ = ["read_map_table", "render_map_figures", "write_figure_section"]
