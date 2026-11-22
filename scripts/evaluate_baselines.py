#!/usr/bin/env python3
"""Evaluate reference baselines and write result tables, reports, and figures.

Rerun the same command to resume an interrupted evaluation. Use a new output
directory when changing the code, configuration, or selected maps.
"""
import argparse
import json
from pathlib import Path
from explo2_bench.evaluation import run_evaluation
from explo2_bench.analysis.baseline_results import summarize_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v2", "v3", "all"], default="all",
                        help="Benchmark collection; default: all")
    parser.add_argument("--root", help="Repository or benchmarks directory; default: automatic data lookup")
    parser.add_argument("--config", default="configs/baselines.json",
                        help="Evaluation configuration JSON; default: configs/baselines.json")
    parser.add_argument("--output-root", default="outputs/baselines",
                        help="Write results under a subdirectory for each version; default: outputs/baselines")
    parser.add_argument("--workers", type=int, default=1, help="Parallel map workers; default: 1")
    parser.add_argument("--limit", type=int,
                        help="Evaluate the first N maps per version after split filtering; default: all selected maps")
    parser.add_argument("--split", choices=["train", "validation", "test", "unassigned"],
                        help="Select maps by split assignment; default: the full inventory")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure rendering")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    versions = ["v2", "v3"] if args.version == "all" else [args.version]
    for version in versions:
        output = run_evaluation(version, root=args.root, config=config, workers=args.workers,
                                limit=args.limit, split=args.split, output_root=args.output_root)
        print(summarize_results(output, figures=not args.no_figures), flush=True)


if __name__ == "__main__":
    main()
