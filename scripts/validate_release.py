#!/usr/bin/env python3
"""Check bundled maps, source and artifact checksums, and map regeneration.

Validation uses the installed metric dependency and local release files;
it does not download data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from explo2_bench.data import benchmark_root, load_benchmark, validate_benchmark, validate_record
from explo2_bench.upstream import verify_metric_dependency


def validate_release(root: Path, regenerate: str = "representative") -> dict:
    data = benchmark_root(root)
    sources = json.loads((data / "source_files.json").read_text(encoding="utf-8"))
    source_root = data.parent
    dependency = verify_metric_dependency()
    for source, record in sources.items():
        target = source_root / record["extracted_path"]
        if hashlib.sha256(target.read_bytes()).hexdigest() != record["extracted_sha256"]:
            raise ValueError(f"Bundled source checksum differs: {source} -> {target}")
    results = []
    for version in ("v2", "v3"):
        summary = validate_benchmark(version, data, regenerate=regenerate == "all")
        representative_count = 0
        if regenerate == "representative":
            seen = set()
            for record in load_benchmark(version, data):
                profile = record["generation"]
                pair = profile["exploration_profile"], profile["dag_profile"]
                if pair not in seen:
                    validate_record(record, regenerate=True)
                    representative_count += 1
                    seen.add(pair)
        summary["regeneration_mode"] = regenerate
        summary["representative_regenerations"] = representative_count
        results.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return {"valid": True, "extracted_source_files_verified": len(sources),
            "metric_dependency": dependency,
            "source_files_sha256": hashlib.sha256((data / "source_files.json").read_bytes()).hexdigest(),
            "benchmarks": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository or benchmarks directory; default: this script's repository")
    parser.add_argument("--regenerate", choices=("none", "representative", "all"), default="representative",
                        help="Regenerate no maps, one per profile pair (default), or every map")
    parser.add_argument("--output", type=Path, help="Save the complete validation report to this JSON file")
    args = parser.parse_args()
    report = validate_release(args.root, args.regenerate)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": True, "extracted_source_files_verified": report["extracted_source_files_verified"]}))


if __name__ == "__main__":
    main()
