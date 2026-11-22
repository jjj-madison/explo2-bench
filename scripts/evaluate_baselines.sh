#!/usr/bin/env bash
# Run baseline evaluation locally using the active Python environment.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$ROOT_DIR"

# Prevent numeric libraries used by report generation from creating extra
# threads inside each process. Set parallelism with --workers (default: 1).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exec python scripts/evaluate_baselines.py --output-root outputs/baselines "$@"
