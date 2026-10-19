#!/bin/bash
# Generate TVSeries target_perframe files from labels.csv and exported feature .npy files.
#
# Usage:
#   ./kdoad/scripts/generate_target_perframe.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="$ROOT_DIR/datasets/TVSeries"

exec "$ROOT_DIR/.venv/bin/python" \
    -m kdoad.scripts.labels_to_target_perframe_lstr \
    --labels-csv "$DATA_ROOT/labels.csv" \
    --feature-dir "$DATA_ROOT/features/repr_x3d_test" \
    --output-dir "$DATA_ROOT/target_perframe" \
    --testra-data-info "$ROOT_DIR/testra/data/data_info.json" \
    "$@"
