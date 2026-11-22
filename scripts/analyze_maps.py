#!/usr/bin/env python3
"""Analyze the full released inventory without selecting an agent budget."""

import argparse
import json
from pathlib import Path

from explo2_bench.analysis.maps import analyze_benchmark
from explo2_bench.analysis.plots import read_map_table, render_map_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", choices=("v2", "v3"), default=["v2", "v3"])
    parser.add_argument("--root", type=Path, default=None, help="Repository or benchmarks directory")
    parser.add_argument("--output", type=Path, default=Path("analysis/maps"))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--figures-only", action="store_true", help="Read existing per_map.csv tables")
    args = parser.parse_args()
    if not args.figures_only:
        for version in args.versions:
            print(json.dumps(analyze_benchmark(version, root=args.root, output_root=args.output)), flush=True)
    if not args.no_figures:
        rows = {version: read_map_table(args.output / version / "per_map.csv") for version in args.versions}
        render_map_figures(rows, args.output)


if __name__ == "__main__":
    main()
