# Baseline evaluation and analysis

Eight reference baselines navigate the fixed benchmark maps using partial
observations. We measure task success, exploration and exploitation errors,
action use, grid coverage, and planning cost across different action budgets.
The reports show how each method discovers tasks and uses known dependencies
as more actions become available.

The complete configuration covers **8,350 maps** and ten action-budget
multipliers. Each stochastic policy uses three replicates; deterministic
policies run once per map. This gives **100,200 trajectories** and **1,002,000
result rows**, because each trajectory is scored at ten budget cutoffs.

| Report | Maps | Result rows |
|---|---:|---:|
| [(EXPLO)²-Bench v2 baseline analysis](v2/README.md) | 250 | 30,000 |
| [(EXPLO)²-Bench v3 baseline analysis](v3/README.md) | 8,100 | 972,000 |

The allowance is `B = floor(alpha × open_cells)`, with
`alpha = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]`. Policies cannot see the allowance,
so lower cutoffs are evaluated from prefixes of the same trajectory. Their
internal planning limits remain fixed.

Exploration and exploitation error rates divide errors by the actions eligible
for each measurement. Actions taken while the goal is ready to activate contribute
only to exploitation. The [measurement definitions](../../docs/baselines.md#measurements)
explain those categories. Union error counts an action once when either or both
error types occur and divides by
executed actions. Reports average replicates within maps, maps within each
exploration-demand/task-DAG setting, then settings equally. The
[upstream framework](https://github.com/jjj-madison/measurable-explore-exploit)
provides the error definitions and implementation.

Each version's `run.json` records the evaluation configuration, code identity,
metric dependency, execution context, and result checksums. See
[result provenance and validation](../../docs/baselines.md#result-provenance-and-validation)
for the pinned scoring implementation and the checks applied to these
results.

The [metric audit](metric-audit.json) checked all **1,002,000 saved result rows**
and **25,920 error-rate summary means** without finding discrepancies. It also
replayed the first inventory map in each of the **106 exploration-demand/task-DAG
settings**, covering all eight methods, policy replicates, and ten budget
cutoffs: **1,272 trajectories and 12,720 budget rows**. Those sampled replays
matched all compared non-timing fields using the pinned upstream scoring
implementation.
The remaining trajectories have not been replayed; timing fields are excluded
from replay comparisons.

After [installing the analysis dependencies](../../docs/reproduce.md#install),
run an evaluation from the repository root using your chosen Python interpreter:

```bash
python scripts/evaluate_baselines.py --version all --workers 1
```

Results go to `outputs/baselines/{v2,v3}` by default; use `--output-root` to
choose another directory. Atomic checkpoints allow a stopped evaluation to
resume with the same code, dependency, maps, and configuration. See the
[reproduction guide](../../docs/reproduce.md) for setup and parallel execution,
and the [baseline protocol](../../docs/baselines.md) for measurement definitions
and interpretation.
