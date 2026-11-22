"""Map-level aggregation of complete action-budget sweeps."""
from collections import defaultdict
import csv
import gzip
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import statistics

from ..baselines import STOCHASTIC_BASELINES
from ..data import canonical_sha256

FIELDS = ("success", "steps", "successful_actions", "coverage", "activated_fraction", "discovered_fraction",
          "exploration_error_rate", "exploitation_error_rate", "union_error_rate",
          "invalid_actions", "planning_cpu_seconds", "planning_wall_seconds",
          "planning_cpu_seconds_per_action", "search_states_generated", "search_limit_hits")


def summarize_results(results_path, *, figures=True) -> dict:
    """Average stochastic replicates within each map before map aggregation.

    Uncertainty describes across-map variation within preset cells. It does not
    measure leakage, uncertainty in a fixed inventory's exact mean, or inference
    to another map distribution. Normal intervals can be poor in small cells.
    """
    path = Path(results_path)
    manifest = json.loads((path.parent / "run.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Summarize a complete run only")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["results_sha256"]:
        raise ValueError("Result SHA256 does not match run manifest")
    config = manifest["config"]
    expected_keys = {(method, alpha, replicate)
                     for method in config["methods"] for alpha in config["alphas"]
                     for replicate in range(config["stochastic_replicates"] if method in STOCHASTIC_BASELINES else 1)}
    groups = defaultdict(lambda: defaultdict(list))
    contributing_runs = defaultdict(lambda: defaultdict(int))
    maps = set()
    selection = []
    row_count = 0
    with gzip.open(path, "rt") as handle:
        rows = (json.loads(line) for line in handle)
        for map_id, map_rows in groupby(rows, key=lambda row: row["map_id"]):
            if map_id in maps:
                raise ValueError("Input must keep all rows for each map together")
            maps.add(map_id)
            by_method = defaultdict(list)
            seen_keys = set()
            episode_hash = None
            for row in map_rows:
                row_count += 1
                row_key = (row["method"], row["alpha"], row["replicate"])
                if row_key not in expected_keys or row_key in seen_keys:
                    raise ValueError(f"Unexpected or duplicate result key for {map_id}: {row_key}")
                seen_keys.add(row_key)
                if row["version"] != manifest["version"]:
                    raise ValueError(f"Result version mismatch for {map_id}")
                if episode_hash is None:
                    episode_hash = row["episode_sha256"]
                if row["episode_sha256"] != episode_hash:
                    raise ValueError(f"Mixed episode hashes for {map_id}")
                if row["success"] != (row["goal_activation_step"] is not None):
                    raise ValueError(f"Success and goal activation disagree for {map_id}")
                row["successful_actions"] = row["goal_activation_step"] if row["success"] else None
                row["activated_fraction"] = row["activated_count"] / row["task_nodes"]
                row["discovered_fraction"] = row["discovered_count"] / row["task_nodes"]
                for key in ("search_states_generated", "search_limit_hits"):
                    row[key] = row["planner_diagnostics"].get(key, 0)
                by_method[(row["method"], row["alpha"], row["exploration_profile"], row["dag_profile"])].append(row)
            if seen_keys != expected_keys:
                raise ValueError(f"Incomplete result keys for {map_id}")
            selection.append((map_id, episode_hash))
            for key, replicates in by_method.items():
                for field in FIELDS:
                    values = [row[field] for row in replicates if row[field] is not None]
                    if values:
                        groups[key][field].append(statistics.fmean(values))
                        contributing_runs[key][field] += len(values)
    if len(maps) != manifest["map_count"] or row_count != manifest["row_count"]:
        raise ValueError("Result map/row count does not match run manifest")
    if canonical_sha256(selection) != manifest["selection_sha256"]:
        raise ValueError("Result map selection does not match run manifest")
    by_setting = []
    for (method, alpha, exploration, dag), values in sorted(groups.items()):
        for field in FIELDS:
            sample = values[field]
            n = len(sample)
            mean = statistics.fmean(sample) if sample else None
            se = statistics.stdev(sample) / math.sqrt(n) if n > 1 else None
            by_setting.append(dict(method=method, alpha=alpha, exploration_profile=exploration,
                                   dag_profile=dag, metric=field, maps=n,
                                   runs=contributing_runs[(method, alpha, exploration, dag)][field],
                                   mean=mean, standard_error=se,
                                   ci95_low=mean-1.96*se if se is not None else None,
                                   ci95_high=mean+1.96*se if se is not None else None))
    aggregate = defaultdict(list)
    for row in by_setting:
        aggregate[(row["method"], row["alpha"], row["metric"])].append(row)
    overall = []
    for (method, alpha, field), rows in sorted(aggregate.items()):
        nonempty = [row for row in rows if row["mean"] is not None]
        count = len(nonempty)
        mean = statistics.fmean(row["mean"] for row in nonempty) if count else None
        se = (math.sqrt(sum(row["standard_error"]**2 for row in nonempty)) / count
              if count and all(row["standard_error"] is not None for row in nonempty) else None)
        overall.append(dict(method=method, alpha=alpha, metric=field, cells=count,
                            maps=sum(row["maps"] for row in rows), runs=sum(row["runs"] for row in rows),
                            mean=mean, standard_error=se,
                            ci95_low=mean-1.96*se if se is not None else None,
                            ci95_high=mean+1.96*se if se is not None else None))
    for filename, rows in (("by_setting.csv", by_setting), ("summary.csv", overall)):
        with (path.parent / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "run_identity": manifest["identity"], "maps": len(maps), "result_rows": row_count,
        "aggregation_revision": 1,
        "aggregation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_results_sha256": manifest["results_sha256"],
        "table_sha256": {name: hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
                         for name in ("by_setting.csv", "summary.csv")},
        "aggregation": "replicate mean within map, map mean within preset cell, equal-weight cell mean",
        "uncertainty": "1.96 * standard error of cell-stratified map means; normal approximation, not a bootstrap",
        "uncertainty_limits": "Describes within-cell map variability, conditional on the released generator and cells. Shared map structures can violate independence; (EXPLO)²-Bench v2 cells have only 10 maps. Degenerate success cells have zero estimated variance. Fixed-inventory means are exact, not estimates.",
        "missing_rates": "A rate with zero eligible steps is null, excluded from that metric's mean; maps count reports contributing maps.",
        "successful_actions": "Actions to goal activation among successful runs only; average successful replicates within each map, contributing maps within each cell, then contributing cells equally. maps and runs report valid counts. This conditional mean can increase as harder maps start succeeding.",
        "integrity": "Verified result SHA256, map/row counts, map identity selection, and exactly one row per configured method/alpha/replicate per map.",
        "timing": "CPU and wall time include policy reset and act calls; exclude environment/scoring. Shared host and instrumentation affect measurements.",
    }
    (path.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if figures:
        compared = {manifest["version"]: (overall, path.parent)}
        for other in ("v2", "v3"):
            other_dir = path.parent.parent / other
            if other == manifest["version"] or not (other_dir / "summary.csv").exists():
                continue
            other_manifest = json.loads((other_dir / "run.json").read_text())
            if other_manifest["status"] != "complete":
                continue
            with (other_dir / "summary.csv").open(newline="") as handle:
                other_rows = list(csv.DictReader(handle))
            for row in other_rows:
                row["alpha"] = float(row["alpha"])
                row["mean"] = float(row["mean"]) if row["mean"] else None
            compared[other] = (other_rows, other_dir)
        shared_limits = {metric: max((row["mean"] for table, _ in compared.values() for row in table
                                    if row["metric"] == metric and row["mean"] is not None), default=0)
                         for metric in ("planning_cpu_seconds", "search_states_generated")}
        for version, (table, directory) in compared.items():
            plot_results(table, directory, version, shared_limits=shared_limits,
                         compared_versions=sorted(compared))
    write_report(overall, summary, manifest, path.parent,
                 figures=figures or (path.parent / "figures/success.png").exists())
    return summary


def write_report(rows, summary, manifest, output, *, figures=True):
    """Write a compact report tied to the validated result manifest."""
    output = Path(output)
    alphas = sorted(manifest["config"]["alphas"])
    final_alpha = max(alphas)
    lookup = {(row["method"], row["alpha"], row["metric"]): row for row in rows}

    def number(method, metric, digits=3):
        value = lookup[(method, final_alpha, metric)]["mean"]
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [f"# (EXPLO)²-Bench {manifest['version']} baseline analysis", "",
             "Reference baselines explore partially observable grids, discover task nodes and their dependencies, "
             "and use that knowledge to activate the goal. We measure task success, exploration and exploitation "
             "errors, action use, grid coverage, and planning cost across action budgets.", "",
             f"This report covers **{summary['maps']:,} maps** and **{summary['result_rows']:,} budget rows**. "
             "Each row reports one map, policy, policy replicate, and action budget. "
             f"The action-budget multipliers are {', '.join(str(alpha) for alpha in alphas)}, "
             "with `B = floor(alpha × open_cells)`. Each cutoff is scored from the same trajectory for a map, "
             "method, and policy seed; policies cannot see the allowance.", "",
             f"Stochastic methods use {manifest['config']['stochastic_replicates']} policy replicates per map. "
             f"Each planning search is capped at {manifest['config']['max_search_states']:,} generated states. "
             "Policy CPU and wall time include reset and action selection and exclude environment transitions and scoring.", "",
             f"## Results at α = {final_alpha}", "",
             "| Method | Success rate | Actions to success | Successful maps / runs | Coverage | Union error rate | Policy CPU seconds |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in manifest["config"]["methods"]:
        success = lookup[(method, final_alpha, "successful_actions")]
        lines.append(f"| {method.replace('_', ' ')} | {number(method, 'success')} | "
                     f"{number(method, 'successful_actions', 1)} | {success['maps']} / {success['runs']} | "
                     f"{number(method, 'coverage')} | {number(method, 'union_error_rate')} | "
                     f"{number(method, 'planning_cpu_seconds', 4)} |")
    lines += ["", "Rates are fractions between 0 and 1. Replicates are averaged within maps, maps within each "
              "exploration-demand/task-DAG setting, and settings equally. "
              "Actions to success excludes failures and averages only contributing maps/settings; compare it together "
              "with success rate. Its successful-map and run counts are shown explicitly. Error rates with no eligible "
              "actions are missing and excluded from that metric's mean.", "",
              f"## Exploration and exploitation errors at α = {final_alpha}", "",
              "Separate error rates describe exploration and the use of acquired knowledge. The metric classifies "
              "each action according to what the agent has discovered and which tasks are ready. Eligible actions "
              "are those included in a rate's denominator; some actions count toward both rates. Actions taken "
              "while the goal is ready to activate count only toward exploitation. "
              "The upstream [metric implementation](https://github.com/jjj-madison/measurable-explore-exploit) "
              "defines per-action errors; the [baseline protocol](../../../docs/baselines.md#measurements) "
              "specifies the action categories and denominators.", "",
              "| Method | Exploration error rate | Exploitation error rate |",
              "|---|---:|---:|"]
    for method in manifest["config"]["methods"]:
        lines.append(f"| {method.replace('_', ' ')} | {number(method, 'exploration_error_rate', 6)} | "
                     f"{number(method, 'exploitation_error_rate', 6)} |")
    lines += ["", "These rates average map-level rates rather than pooling counts across trajectories. "
              "They are shown to six decimal places; an em dash indicates no contributing actions. "
              "Union error counts an action once when either or both error types occur and divides by executed "
              "actions, so it is not the sum of the two rates above.", "",
              "## Aggregation and result checks", "",
              "[Per-setting results](by_setting.csv) and [overall results](summary.csv) include contributing counts "
              "and normal-approximation 95% intervals based on within-setting map variation. These intervals describe "
              "variation across the fixed maps under an independence assumption; they do not certify generalization. "
              "Small cells and repeated structural components limit that approximation, and all-success/all-failure "
              "cells yield zero estimated variance. The fixed-inventory means themselves are exact.", "",
              "See the [complete compressed result table](results.jsonl.gz), [run configuration and checksums](run.json), "
              "and [aggregation definitions](summary.json). Result hashes, inventory counts, and all expected "
              "method/budget/replicate combinations were checked before aggregation. "
              "See [result provenance and validation](../../../docs/baselines.md#result-provenance-and-validation) "
              "for the pinned scoring implementation and the checks applied to the included results.", ""]
    audit_path = output.parent / "metric-audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        audited = audit.get("versions", {}).get(manifest["version"], {})
        result_hash = manifest.get("results_sha256")
        if audit.get("valid") is True and result_hash and audited.get("result_sha256") == result_hash:
            replay_map_label = "map" if audited["replay_maps"] == 1 else "maps"
            lines += [
                f"The [upstream metric audit](../metric-audit.json) checked all **{audited['rows_checked']:,} "
                f"recorded rows** and **{audited['summary_means_checked']:,} error-rate summary means** for this result file. "
                f"It replayed **{audited['replay_trajectories']:,} trajectories** on **{audited['replay_maps']:,} selected {replay_map_label}**, "
                f"covering **{audited['replay_rows']:,} budget rows**, with agreement in all compared non-timing fields. "
                "Timing fields are excluded from replay comparisons.", "",
            ]
            if audited["replay_maps"] < audited["maps_checked"]:
                lines += ["Trajectory replay covers the sample listed in the audit; the remaining trajectories "
                          "have not been replayed. The arithmetic and summary checks cover the full saved result table.", ""]
    if figures:
        lines += ["## Budget curves", "",
                  "The curves show how measured behavior changes as the action allowance increases. "
                  "Each plot is a separate panel with PNG and PDF exports. Rate plots share the 0–1 scale.", "",
                  f"![Task success rate across action-budget multipliers for the (EXPLO)²-Bench {manifest['version']} reference baselines.](figures/success.png)", "",
                  "Task success means activating the goal within the allowance. Each point averages the fixed "
                  "maps using the protocol above.", "",
                  f"![Union error rate across action-budget multipliers for the (EXPLO)²-Bench {manifest['version']} reference baselines.](figures/union_error_rate.png)", "",
                  "Union error combines exploration and exploitation error indicators per action. Interpret this "
                  "curve alongside task success: a low error rate alone does not establish that the goal was completed.", "",
                  "Additional measurements:", ""]
        for metric, label in (("coverage", "Grid coverage"),
                              ("activated_fraction", "Task activation"),
                              ("planning_cpu_seconds", "Planning CPU cost"),
                              ("search_states_generated", "Generated planning states")):
            lines.append(f"- [{label}](figures/{metric}.png).")
    (output / "README.md").write_text("\n".join(lines) + "\n")


def plot_results(rows, output, version, *, shared_limits=None, compared_versions=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..baselines import BASELINE_NAMES
    output = Path(output) / "figures"
    output.mkdir(exist_ok=True)
    labels = {"success": "Goal activation rate", "coverage": "Visited / open cells",
              "activated_fraction": "Activated / task nodes", "union_error_rate": "Error steps / executed actions",
              "planning_cpu_seconds": "Policy CPU time per episode (seconds)",
              "search_states_generated": "Generated planner states per episode"}
    for metric, label in labels.items():
        fig, ax = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
        for method in BASELINE_NAMES:
            series = sorted((row for row in rows if row["metric"] == metric and row["method"] == method), key=lambda row: row["alpha"])
            if not series:
                continue
            x, y = [row["alpha"] for row in series], [row["mean"] for row in series]
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.4, label=method.replace("_", " "))
        method_count = len({row["method"] for row in rows})
        ax.set(xlabel="Action budget multiplier α (B = floor(α × open cells))", ylabel=label,
               title=f"(EXPLO)²-Bench {version}\nFixed maps, {method_count} observation-only baselines", xlim=(0.4, 10.1))
        if metric in ("success", "coverage", "activated_fraction", "union_error_rate"):
            ax.set_ylim(0, 1.02)
        if metric in ("planning_cpu_seconds", "search_states_generated"):
            ax.set_yscale("symlog", linthresh=0.001 if metric == "planning_cpu_seconds" else 1)
            maximum = (shared_limits or {}).get(metric, max(
                (row["mean"] for row in rows if row["metric"] == metric and row["mean"] is not None), default=0))
            ax.set_ylim(0, max(maximum * 1.05, 0.001 if metric == "planning_cpu_seconds" else 1))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{metric}.{extension}", dpi=170)
        plt.close(fig)
    (output / "scales.json").write_text(json.dumps({"versions_compared": compared_versions or [version],
        "cost_maximum_before_padding": shared_limits, "rates": [0, 1.02],
        "action_budget_multiplier": [0.4, 10.1]}, indent=2) + "\n")
