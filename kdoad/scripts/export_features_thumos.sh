#!/bin/bash
# Extract X3D features from THUMOS14 videos via export_features.py.
#
# Model:
#   X3DEmbedding — X3D-XS without classification head, embedding 2048
#
# Environment:
#   CHECKPOINT, FEATURE_FPS, SOURCE_FPS, FRAMES, STRIDE,
#   DEVICE, BATCH_SIZE, OUTPUT_FOLDER, LABELS_CSV,
#   PARALLEL_GPUS / NUM_GPUS, CUDA_VISIBLE_DEVICES
#
# Temporal sampling:
#   feature_fps + source_fps — how often to place a new clip anchor
#   frames + stride          — clip composition for the model
#
# Examples:
#   ./kdoad/scripts/export_features_thumos.sh
#   LABELS_CSV=/path/to/labels.csv ./kdoad/scripts/export_features_thumos.sh
#   CHECKPOINT=checkpoints/thumos_x3d.pt ./kdoad/scripts/export_features_thumos.sh
#   FRAMES=4 STRIDE=12 OUTPUT_FOLDER=features/x3d_xs_f4_s12 \
#       ./kdoad/scripts/export_features_thumos.sh
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 LABELS_CSV=datasets/THUMOS14/labels.csv \
#       ./kdoad/scripts/export_features_thumos.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DS_ROOT="${DS_ROOT:-$ROOT_DIR/datasets/THUMOS14}"
CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/checkpoints/thumos_x3d.pt}"
LABELS_CSV="${LABELS_CSV:-$DS_ROOT/labels.csv}"
export CHECKPOINT DS_ROOT LABELS_CSV

exec "$SCRIPT_DIR/export_features_common.sh" \
    "$DS_ROOT/processed/validation" \
    "$DS_ROOT/processed/test"
