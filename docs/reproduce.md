# Reproduce the release and analyses

Install Python 3.10+ with pip and Git, and run commands from the repository root.
The commands below use `python` for your chosen interpreter; substitute `python3`
if that is its name on your system. An isolated Python environment is recommended.
Each map archive's `manifest.json` records its generation Python version;
each baseline evaluation records its Python version in `run.json`.

## Install

Install the package with analysis and development dependencies:

```bash
python -m pip install -e '.[analysis,dev]'
python -m pip check
```

All dependencies are declared in [pyproject.toml](../pyproject.toml). The
`analysis` extra includes matplotlib and NumPy; the `dev` extra includes pytest
and build. Editable installation makes source edits available in this
environment. For basic environment and agent use, `python -m pip install .`
installs the package and its runtime dependencies.

The package pins
[measurable-explore-exploit revision `2bb2601`](https://github.com/jjj-madison/measurable-explore-exploit/commit/2bb260104d7272d3908f312f3b1f67e394fbaa9b).
Installation uses Git to obtain this dependency from the public repository.
Runtime metadata records the upstream revision and source hashes.

## Validate and build

Run the tests and validate every map against deterministic regeneration:

```bash
python -m pytest
python scripts/validate_release.py --regenerate all
```

The validator checks the pinned metric dependency, bundled source-code
checksums, every release artifact checksum,
map identity, canary metadata, grid connectivity, task placements, dependencies,
saved default action budgets, saved successful routes in (EXPLO)²-Bench v3,
and split counts.
`--regenerate all` also regenerates every map and compares its episode hash. The
default mode regenerates one map in each of the 106 profile combinations after
validating the full inventory. Use `--output analysis/validation.json` to save
the report.

To inspect or validate data using an installed package:

```bash
explo2-bench info v3
explo2-bench validate v2 --regenerate
explo2-bench play v2 --alpha 2
```

Build a source distribution and wheel under `dist/`:

```bash
python -m build
```

The wheel includes both benchmark collections, their settings, seeds, and
checksums.

## Regenerate the maps

```bash
python scripts/freeze_benchmarks.py
```

The command uses the included generator, seed tables, and settings. It validates
each generated map, retains its split assignment, compares every (EXPLO)²-Bench
v3 episode with the upstream hash recorded in `seeds.csv`, and replaces the
compressed exports only after completing a version. The
[dataset provenance](benchmark.md#dataset-provenance) explains these upstream
reference hashes and the evidence available for v2.

The canonical uncompressed episode hash defines instance identity. Gzip archive
bytes may depend on the compression implementation; their manifest hashes
validate the exact artifact produced in that run. Saved episodes retain default
budget multipliers 3 for (EXPLO)²-Bench v2 and 2 for (EXPLO)²-Bench v3.

## Reproduce structural analysis

```bash
python scripts/analyze_maps.py
python -m explo2_bench.rendering
```

This analyzes all 8,350 maps and writes per-map tables, profile summaries,
diversity audits, and figures under `analysis/maps/{v2,v3}`. It does not run an
agent or choose an action budget. Figures use shared scales where the two
versions are compared; each panel is exported separately as PNG and PDF.
The second command regenerates the representative full-map and task-graph panels.

Use `--no-figures` to produce tables without rendering figures, `--versions v2` for
one version, and `--figures-only` to render existing tables. Metric definitions
and limitations are in [map analysis](map-analysis.md).

## Reproduce baseline analysis

```bash
python scripts/evaluate_baselines.py --version all --workers 1
```

Results are written to `outputs/baselines/{v2,v3}` by default. Use
`--output-root` to choose another directory. The evaluator resumes a run only
when its code, dependency, configuration, and map selection match. Each run
records those inputs and its execution context in `run.json`. See
[result provenance and validation](baselines.md#result-provenance-and-validation)
for the scoring implementation and checks applied to the included results in
`analysis/baselines/{v2,v3}`.

The [configuration](../configs/baselines.json) specifies all eight methods and
`alpha = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]`. Each runtime budget is
`floor(alpha × open_cells)`. The two stochastic policies have three recorded
replicates; deterministic methods have one. Generation seeds and policy seeds
are stored separately. The fixed planning-state limit is recorded in the config.

For a small demonstration, keep its output separate from the complete analysis:

```bash
python scripts/evaluate_baselines.py --version v2 --limit 2 \
    --output-root outputs/baselines-demo --workers 1
```

`--limit` takes an inventory prefix; it is a smoke test, not a representative
sample. With `--version v3`, `--split test` selects the 360-map v3 test subset.
Increase `--workers` only as local resources permit. Wall-clock timing depends
on host contention; planning process CPU time is also reported.

The local Bash launcher
[`scripts/evaluate_baselines.sh`](../scripts/evaluate_baselines.sh) uses the
`python` interpreter on your command path with the `analysis` extra installed.
It defaults to both versions, one worker, and `outputs/baselines`. Set the worker
count explicitly to run in parallel:

```bash
bash scripts/evaluate_baselines.sh --workers 4
```

All arguments are forwarded to the evaluator, including `--version`, `--limit`,
and `--output-root`. Windows users can use the Python commands above.

Evaluation checkpoints are written per map and permit interruption and restart.
Rerun the same command to continue. Configuration, code, and selection changes
require a different output root, so results from different protocols cannot
silently mix. A completed `run.json` records the selected maps, code fingerprint,
configuration, upstream metric revision and source hashes, timing context,
output checksum, and `status="complete"`.
Incomplete runs remain labeled `running`.
Each row's `execution_id` identifies its attempt in the manifest's `executions`
list. This preserves the hardware and worker context when a campaign resumes
on another machine; timing summaries then reflect all contributing attempts.

The budget-blind baselines are rolled out once to the largest budget, then
scored at each action cutoff. The [baseline protocol](baselines.md) explains why
this preserves their lower-budget trajectories, defines error denominators and
uncertainty estimates, and separates policy planning cost from evaluator scoring.

## Audit baseline results

Audit the complete runs produced in `outputs/baselines` above:

```bash
python scripts/audit_baseline_metrics.py --results-root outputs/baselines \
    --output outputs/baselines/metric-audit.json --workers 4
```

To audit the included results under `analysis/baselines`, use the defaults:

```bash
python scripts/audit_baseline_metrics.py --workers 4
```

`--results-root` selects the directory containing the `v2` and `v3` result
archives. `--root` selects the benchmark data used for replay; it defaults to
this repository. Without `--results-root`, archives are read from
`<root>/analysis/baselines`. `--output` selects only the audit report destination,
which defaults to
[`analysis/baselines/metric-audit.json`](../analysis/baselines/metric-audit.json).
The audit leaves the archived results unchanged.

The audit checks checksums, error-rate arithmetic, and independently recomputed
error-rate summary means across the saved result tables. By default, it then
replays the first inventory map in each exploration-demand/task-DAG setting
using the pinned upstream scorer, all configured methods and policy replicates,
and all action-budget cutoffs. `--maps-per-setting N` increases that sample;
`--all-maps` replays the full inventory. Set `--workers` to suit the available
CPU allocation. These commands target complete-inventory runs. Replay maps are
selected from the benchmark inventory, and the audit fails if any selected map
is missing from the result archive. A run restricted by `--limit` or `--split`
may not contain that sample.

The included audit's coverage and findings are described in
[result provenance and validation](baselines.md#result-provenance-and-validation).
Its trajectory replay is a sample, with timing fields excluded from the
comparison. The report records the precise map selection and source hashes.
Policy CPU times in the baseline reports exclude scoring, so they do not
describe the total cost of running this audit.

## Connect an LM agent

```bash
python examples/custom_agent.py --version v2
python examples/lm_agent.py --version v3
```

The second example makes no model call until its dummy callback is replaced.
See the [LM-agent protocol](lm-agents.md) for the observation/action contract,
prompt, history policy, and failure accounting. Preserve model/provider versions,
sampling parameters, tokens, latency, and agent seeds with reported experiments.
