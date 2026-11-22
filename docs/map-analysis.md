# Structural analysis of the fixed maps

Map analysis describes each complete grid and task graph, independently of an agent's action budget or computation limit. Its units and definitions are shared by (EXPLO)²-Bench v2 and (EXPLO)²-Bench v3. The two inventories have different numbers of seeds per setting, so comparisons should use setting summaries or normalized distributions rather than raw sample counts.

Run the complete analysis from the repository root after installing the package with its analysis dependencies:

```bash
python scripts/analyze_maps.py
python -m explo2_bench.rendering
```

This writes results to `analysis/maps/v2/` and `analysis/maps/v3/`. `--no-figures` produces tables without importing plotting libraries. `--figures-only` redraws existing tables; run both versions together to retain common figure scales.

## Grid and placement measurements

| Column | Definition |
|---|---|
| `width`, `height`, `total_cells` | Grid dimensions and their product. |
| `open_cells`, `blocked_cells`, `open_fraction` | Traversable and blocked cell counts; open cells divided by total cells. |
| `connected_components`, `reachable_cells` | Number of connected components in the full grid and number of cells reachable from the start. |
| `grid_edges` | Undirected, orthogonal adjacency edges between open cells. |
| `isolated_cells`, `dead_ends`, `junctions` | Open cells of degree zero, one, and at least three, respectively. |
| `cycle_rank` | Independent cycles: edges minus open cells plus connected components. |
| `task_sites`, `task_density` | Number of occupied task coordinates; task nodes including the goal divided by open cells. |
| `start_goal_distance` | Shortest path from start to goal location, in moves. |
| `start_task_distance_*` | Count, minimum, mean, median, and maximum shortest-path distance from start to each task, including the goal. |
| `task_pair_distance_*` | The same statistics over unordered pairs of distinct task nodes, including the goal. |
| `unreachable_tasks`, `unreachable_task_pairs` | Missing distances caused by disconnected cells. These distances are excluded from summaries rather than assigned zero. |

Breadth-first search computes distances on the complete undirected grid. Distances describe geometry; they do not describe an agent's observations, a route that satisfies task dependencies, or minimum goal-completion actions. Pair distances weight task pairs equally, including co-located tasks if a custom map permits them.

## Task graph measurements

Each prerequisite group is a conjunction (AND) of parents. Multiple groups are alternatives (OR). An empty group allows activation without parents. Measurements retain these groups to represent the logical structure and also use a union graph with an edge for each distinct parent–child dependency.

| Column | Definition |
|---|---|
| `task_nodes`, `non_goal_tasks` | Number of task nodes including and excluding the goal. |
| `primitive_tasks` | Nodes with no parents in the union graph. |
| `initially_ready_tasks` | Nodes with an empty prerequisite group or no prerequisite groups. Counts logical readiness before spatial discovery. |
| `dependency_edges` | Distinct parent–child pairs across all groups; a repeated parent in alternative groups contributes one edge. |
| `prerequisite_groups`, `empty_groups` | Total number of prerequisite groups, including empty groups, and number of empty groups. |
| `and_groups` | Groups with at least two parents. Single-parent groups are not counted as AND junctions. |
| `or_tasks` | Tasks with at least two alternative groups. |
| `group_fanin_mean`, `group_fanin_max` | Parents per group, including empty groups. |
| `node_fanin_mean`, `node_fanin_max` | Distinct parents per task in the union graph. |
| `dag_depth`, `goal_depth` | Longest directed path in edges, over the whole union graph or ending at the goal. Source nodes have depth zero. |
| `dag_layers`, `max_layer_width`, `layer_widths` | Number of depth levels, largest level, and counts at every level starting with depth zero. |

Depth is recomputed from edges, rather than copied from generator metadata. The union graph includes all alternatives; its depth and ancestor counts do not establish the minimum work needed for goal activation. Nominal exploration/DAG profiles identify generation settings and do not establish a monotonic ordering of empirical difficulty.

## Diversity definitions

`per_map.csv` contains three SHA-256 fingerprints and a task graph class ID:

- `gameplay_sha256` retains grid dimensions and walls, start, task IDs and visible names, task placements, prerequisite groups, goal, and required nodes. It excludes seeds, budgets, derived depth annotations, generation knobs, saved successful routes, and other provenance metadata. `episode_sha256` separately identifies the serialized environment specification, including its saved default action budget.
- `wall_layout_sha256` retains dimensions and blocked coordinates, excluding start and tasks.
- `task_coordinates_sha256` retains only the sorted set of occupied task coordinates, ignoring walls, dimensions, start, task identity, and multiplicity at a coordinate. This matches the upstream spatial-diversity convention; it is not a complete layout identity.
- `dag_class` is an exact directed graph isomorphism class preserving the distinguished goal and every AND/OR group. Auxiliary group nodes preserve alternatives. Task names, IDs, spatial locations, and depth annotations are ignored.

Weisfeiler–Lehman refinement narrows candidate graph matches; every match is checked by exact isomorphism. A matching refinement hash alone is never counted as equality. Class IDs are local to a version's analysis and depend on insertion order; do not compare numeric IDs between version reports. Coordinate fingerprints do not merge rotations, reflections, or translations.

`diversity.json` reports complete inventory and setting-level counts, duplicate excess (maps minus distinct classes), largest class size, and IDs of exact gameplay duplicate groups. A repeated wall layout or task graph does not imply that the full maps are identical.

The same file audits overlap across the assigned training, validation, and test splits. For each pair of splits and each equivalence definition, it reports shared classes, maps from each split belonging to those classes, and the number of matching cross-split map pairs. Unassigned maps are counted separately; see the [split definitions](benchmark.md#split-memberships). A shared component can support memorization, but component reuse alone does not establish evaluation contamination. Zero exact-gameplay duplicates does not imply that all layouts or dependency structures are unseen.

## Tables, provenance, and figures

`per_map.csv` has one row per map. `by_profile.csv` groups by exploration and DAG profiles and reports valid count, mean, population standard deviation, minimum, median, and maximum for each numeric structural metric. These are descriptive summaries of the fixed inventory, not confidence intervals for an inferred population.

`manifest.json` records analysis version, complete inventory size, input ID/hash digest, and checksums for the machine-readable results. It does not substitute for the benchmark's release manifest.

Figures are separate PNG and PDF panels. Six heatmaps summarize open cells, task nodes, independent grid cycles, dependency depth, AND groups, and mean task-pair distance. Two empirical cumulative distributions summarize open cells and task-pair distances. Compared versions use identical limits, units, and visual encodings; `figures/scales.json` records the limits. Representative full-map and task-graph diagrams show the same task names in separate exports.

Structural reports: [(EXPLO)²-Bench v2](../analysis/maps/v2/README.md) and [(EXPLO)²-Bench v3](../analysis/maps/v3/README.md). Example galleries: [(EXPLO)²-Bench v2](../analysis/maps/v2/examples/README.md) and [(EXPLO)²-Bench v3](../analysis/maps/v3/examples/README.md). Behavioral results belong to the separate baseline analysis, where action budgets and planning cost are explicit.
