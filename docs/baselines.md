# Baselines and action-budget evaluation

Eight reference baselines explore partially observable grid maps, discover task
nodes and dependencies, and use that knowledge to activate the goal. Evaluation
measures task success, exploration and exploitation errors, action use, and
planning cost across different action allowances.

All policies see partial observations and may remember their own observations. They receive no map record, generation seed, full graph, or hidden budget. Behavior does not depend on the evaluation cutoff. The same implementation is used for (EXPLO)²-Bench v2 and (EXPLO)²-Bench v3. Run metadata identifies this set of policies as baseline revision 1.

| Method | Decision rule |
|---|---|
| `random_walk` | Uniformly random legal adjacent cell. |
| `random_revisit` | Choose a random ready task and commit until reached/ineligible; otherwise randomly enter an adjacent unvisited cell or route to the nearest frontier. |
| `frontier_revisit` | Route to the nearest known ready task; explore a frontier when no ready task is reachable. |
| `goal_revisit` | Rank ready tasks by observed dependency-graph distance to the goal, then known travel distance; otherwise explore. |
| `sweep_then_exploit` | Sweep the reachable grid first, then use a bounded task planner. |
| `online_planner` | Plan over known position/activation states when goal closure is known; otherwise use frontier/task heuristics. |
| `dependency_greedy` | Rank ready tasks by immediate unlock count, then shortest known travel distance; explore when no ready task is reachable. |
| `balanced_utility` | Compare task and frontier utility divided by one plus shortest known travel distance, with fixed weights. |

The first six methods adapt policies from the
[measurable-explore-exploit framework](https://github.com/jjj-madison/measurable-explore-exploit).
`dependency_greedy` and `balanced_utility` use fixed weights that were not fitted
on the released test maps. The planners cap each search at 10,000 generated
states, including the root, and cache routes. Failed searches use heuristic
fallbacks. Search-cap hits and generated/expanded states are reported.

The sweep enters adjacent unvisited cells before routing to the nearest frontier; it is not a strict depth-first traversal. Tasks activate automatically during exploration. After covering the reachable grid, it uses the same bounded planner as `online_planner`. The online planner starts a search only after the goal has been discovered and the known prerequisites admit its activation. A completed breadth-first search gives the shortest goal route in the known model; neither policy is a globally optimal policy for partial observations. Both fall back to the nearest ready task, then frontier exploration, after failed or capped searches.

For `dependency_greedy` and `balanced_utility`, an **immediate unlock** is an observed, inactive task that is currently blocked but becomes ready if the candidate activates. The whole AND group must be satisfied, and satisfying one OR alternative is sufficient. Already-ready tasks, unknown prerequisite metadata, and the candidate itself do not count. Each newly ready task counts once. `dependency_greedy` maximizes this count before considering travel distance; it does not divide the count by distance.

`balanced_utility` scores a ready task as `(1 + immediate_unlocks) / (1 + distance)` and a frontier cell as `(1 + unknown_adjacent_coordinates) / (1 + distance)`. An unknown adjacent coordinate has not been established as open or blocked from prior observations. This is an observable uncertainty proxy, not the true number of unseen open cells. The coefficients are fixed at one. Score ties prefer tasks, then shorter distance, task token, and coordinate order. Both heuristics commit to a target until reached or ineligible; a ready goal takes priority over the commitment. All path ties are deterministic.

## Fixed configuration

The multipliers are **0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10**, with `B = floor(α × open_cells)`. The planner cap stays fixed. Random walk and random revisit each use three policy replicates per map; deterministic policies run once. Policy seeds are SHA-256-derived from campaign seed, map ID, method, and replicate, separately from generation seeds.

Each policy rolls out once to the largest allowance, stopping early on success. Lower cutoffs are scored from prefixes. This is valid because policies cannot see the cutoff and their compute caps are fixed. Prefix behavior is tested against independently terminated runs. Do not use this optimization for agents that receive the remaining allowance.

## Measurements

Rows identify map hash, policy, replicate, alpha, allowance, and policy seed.
Performance includes success, executed actions, invalid moves, grid coverage,
task discovery/activation, and exploration/exploitation errors. Per-action
errors are computed by the pinned upstream
[`compute_metrics` implementation](https://github.com/jjj-madison/measurable-explore-exploit/blob/2bb260104d7272d3908f312f3b1f67e394fbaa9b/src/symbolic_environment/metrics.py).
See that implementation for the gain-and-staleness rules and error attribution.
Mixed non-goal actions can incur both error types; `union_errors` counts such an
action once.

Each error rate divides by its eligible action count: the number of actions on
which that error type can be assessed. A zero denominator produces null.
The scorer classifies actions as exploration-only, exploitation-only, or mixed.
Exploration uses `explore_only_steps + mixed_non_goal_steps`.
Exploitation uses `exploit_only_steps + mixed_goal_steps + mixed_non_goal_steps`.
Goal-pending actions (`mixed_goal`) contribute only to exploitation.
Union error rate divides by executed actions. Scoring reconstructs knowledge
before each action from the trajectory;
agents never receive privileged scoring data.

Reported rates average map-level rates according to the protocol below.
Pooling error and eligible-action counts across trajectories gives a different
aggregation and need not produce the same value.

Policy CPU and wall times include reset and action selection, excluding environment stepping and metric computation. State counters cover bounded planning; heuristic BFS is included in timing but not those counters. Shared-host parallel execution affects wall time. `run.json` records worker count, platform, Python, code fingerprint, selection, and configuration. Full-rollout scoring CPU time is a separate diagnostic repeated across cutoff rows; do not sum it over cutoffs.

## Aggregation and uncertainty

Average policy replicates within each map, maps within each preset combination, then weight combinations equally. Per-setting tables remain available. Standard errors use within-setting variation among map means; normal 95% intervals use 1.96 standard errors. They describe variation conditional on this generator and selected settings. The fixed inventory's observed mean is exact. Shared structures can violate independence; (EXPLO)²-Bench v2 has ten maps per combination; all-success/all-failure cells have zero estimated variance. Intervals do not establish generalization to unseen generators or contamination-free evaluation.

Complete rows are in `analysis/baselines/{v2,v3}/results.jsonl.gz`, alongside
`summary.csv`, `by_setting.csv`, and PNG/PDF figures. Each evaluation records its
code, metric dependency, maps, and configuration. Atomic per-map checkpoints
support resumption when those inputs match. Start a separate output directory
when changing them; evaluations of a limited map selection record that limit
explicitly. See the [reproduction commands](reproduce.md#reproduce-baseline-analysis).

## Result provenance and validation

The baseline results in `analysis/baselines/{v2,v3}` use the pinned
[measurable-explore-exploit revision `2bb2601`](https://github.com/jjj-madison/measurable-explore-exploit/commit/2bb260104d7272d3908f312f3b1f67e394fbaa9b)
for scoring. Each `run.json` records the evaluator and baseline revisions, metric
dependency revision and source checksums, evaluation configuration, execution
context, and result checksum. These records identify the implementation and
conditions that produced each report.

The [metric audit](../analysis/baselines/metric-audit.json) checked all
**1,002,000 saved result rows** and **25,920 error-rate summary means**. It also
replayed **106 maps**, covering **1,272 trajectories** and **12,720 budget rows**,
with the pinned upstream scorer. The sampled replays agreed on all checked
non-timing results. This sample contains the first map in each profile pair;
the audit does not replay every saved trajectory. Timing fields are excluded
from the comparison, and the audit records its map selection and source hashes.

Policy CPU times in the reports measure reset and action selection and exclude scoring.
They therefore do not describe the total time required to replay or rescore an
evaluation. Use a separate output directory for new evaluations. Resuming an
evaluation requires matching code, dependency, configuration, and selected maps.
