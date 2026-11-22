# (EXPLO)²-Bench v3 map analysis

Agents explore a partially observable 2D grid to discover task nodes and their dependencies, then use the knowledge acquired through past encounters to complete the goal. The maps vary both the space to explore and the structure of the task dependencies.

This report describes all **8,100 fixed maps** in (EXPLO)²-Bench v3. Structural measurements use the complete maps and task DAGs and are independent of action budgets and agent behavior. See the [map and task-DAG examples](examples/README.md) for complete views of individual environments.

## Environment structure

Open cells are traversable grid cells. Task nodes include the goal, and their prerequisites form a directed acyclic graph (DAG). Exploration-demand and task-DAG difficulty profiles specify the generator settings used to construct each map.

| Structural measurement | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| Open cells (count) | 6.00 | 147.65 | 976.00 |
| Task nodes, including goal (count) | 3.00 | 20.33 | 47.00 |
| Task-dependency edges (count) | 2.00 | 34.34 | 124.00 |
| Longest dependency path (edges) | 1.00 | 5.33 | 10.00 |
| Largest dependency layer (nodes) | 1.00 | 4.84 | 10.00 |
| Grid dead ends (count) | 0.00 | 1.14 | 9.00 |
| Grid junctions (count) | 0.00 | 133.02 | 936.00 |
| Independent grid cycles (count) | 0.00 | 112.74 | 826.00 |
| Start-to-goal distance (moves) | 1.00 | 5.75 | 36.00 |
| Mean task-pair distance (moves) | 1.33 | 7.81 | 22.28 |

## Structural diversity

There are **8,100 distinct gameplay records**, **8,032 wall layouts**, and **5,381 task-DAG classes** under the documented equivalence definitions.

Gameplay identity compares the grid, starting cell, task names and placements, and dependencies, while excluding generation metadata and action budgets. Wall-layout identity compares grid dimensions and blocked cells. Task-coordinate sets compare occupied task cells, ignoring walls and task identities. Task-DAG classes group equivalent goal and AND/OR dependency structures, ignoring task names and spatial positions.

These counts describe map and task structure. Profile names identify generator settings and do not establish increasing difficulty for an agent. Dependency depth includes all prerequisite alternatives, so it does not give the minimum number of tasks needed to activate the goal. Spatial distances do not account for activation order.

See [metric definitions](../../../docs/map-analysis.md), [per-map data](per_map.csv), [setting summaries](by_profile.csv), [diversity results](diversity.json), and [provenance checksums](manifest.json).

## Overlap between dataset splits

The table compares the assigned training, validation, and test subsets. Unassigned maps are excluded. Each entry counts structural classes shared by two splits; see the [split definitions](../../../docs/benchmark.md#split-memberships) for membership counts.

| Split pair | Shared gameplay identities | Shared wall layouts | Shared task-coordinate sets | Shared DAG classes |
|---|---:|---:|---:|---:|
| train/validation | 0 | 0 | 0 | 16 |
| train/test | 0 | 0 | 0 | 21 |
| validation/test | 0 | 0 | 0 | 8 |

Distinct map IDs can share dependency structures. Wall layouts, task-coordinate sets, and task graphs describe separate components of a map. Their overlap measures structural reuse; it does not establish whether a model's training data contained these maps.

## Figures

Each heatmap shows the mean measurement across maps at a pair of exploration-demand and task-DAG difficulty profiles. The axes name generator settings; the colors describe measured structure. Each panel has separate PNG and PDF exports, with common scales across (EXPLO)²-Bench v2 and (EXPLO)²-Bench v3.

### Space to explore

![Mean open-cell count across (EXPLO)²-Bench v3 exploration-demand and task-DAG difficulty profiles.](figures/open_cells_heatmap.png)

Open cells measure the traversable space in each map. Comparing profile pairs shows how the amount of space available for exploration changes across the benchmark.

### Task dependencies

![Mean task-node count, including the goal, across (EXPLO)²-Bench v3 exploration-demand and task-DAG difficulty profiles.](figures/task_nodes_heatmap.png)

Task-node counts describe the size of the dependency graph. The prerequisite structure also matters: node count alone does not determine the work required to activate the goal.

### Additional structural measurements

- [Independent grid cycles (count) by profile](figures/cycle_rank_heatmap.png).
- [Longest dependency path (edges) by profile](figures/dag_depth_heatmap.png).
- [Prerequisite groups with ≥2 parents (count) by profile](figures/and_groups_heatmap.png).
- [Mean task-pair distance (moves) by profile](figures/task_pair_distance_mean_heatmap.png).
- [Open cells (count) distribution](figures/open_cells_distribution.png).
- [Mean task-pair distance (moves) distribution](figures/task_pair_distance_mean_distribution.png).
