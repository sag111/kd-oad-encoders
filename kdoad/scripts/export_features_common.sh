#!/bin/bash
# Shared feature export logic used by the dataset wrappers.
#
# Required environment:
#   DS_ROOT — dataset root
#
# Optional environment:
#   CHECKPOINT, FEATURE_FPS, SOURCE_FPS, FRAMES, STRIDE, INPUT_CLIP_LENGTH,
#   DEVICE — torch device (cuda or cpu); a GPU index or list sets
#            CUDA_VISIBLE_DEVICES
#   BATCH_SIZE, OUTPUT_FOLDER, LABELS_CSV,
#   PARALLEL_GPUS / NUM_GPUS, CUDA_VISIBLE_DEVICES

set -euo pipefail

[[ -n "${DS_ROOT:-}" ]] || { echo "ERROR: DS_ROOT is not set."; exit 1; }
[[ $# -gt 0 ]] || { echo "ERROR: at least one input folder is required."; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"

input_folders=("$@")
output_folder_relative="${OUTPUT_FOLDER:-features/x3d_xs_embedding_2048}"
if [[ "$output_folder_relative" == /* ]]; then
    output_folder="$output_folder_relative"
else
    output_folder="$DS_ROOT/${output_folder_relative#/}"
fi

checkpoint="${CHECKPOINT:-}"
feature_fps="${FEATURE_FPS:-4}"
source_fps="${SOURCE_FPS:-24}"
frames="${FRAMES:-6}"
stride="${STRIDE:-1}"
input_clip_length="${INPUT_CLIP_LENGTH:-4}"
batch_size="${BATCH_SIZE:-64}"
device="${DEVICE:-cuda}"

# A numeric DEVICE is a GPU selection, not a torch device string.
if [[ "$device" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$device}"
    device="cuda"
fi

if [[ -z "$checkpoint" ]]; then
    echo "ERROR: X3DEmbedding requires checkpoint. Set CHECKPOINT env variable."
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: project virtual environment not found: $PYTHON"
    exit 1
fi

if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: checkpoint not found: $checkpoint"
    exit 1
fi

for input_folder in "${input_folders[@]}"; do
    if [[ ! -d "$input_folder" ]]; then
        echo "ERROR: input folder not found: $input_folder"
        exit 1
    fi
done

echo "Project root: $ROOT_DIR"
echo "Dataset root: $DS_ROOT"
echo "Model: X3DEmbedding"
echo "Input folders: ${input_folders[*]}"
echo "Output folder: $output_folder"
echo "Checkpoint: $checkpoint"
echo "feature_fps: $feature_fps, source_fps: $source_fps"
echo "frames: $frames, stride: $stride, input_clip_length: $input_clip_length"
echo "batch_size: $batch_size, device: $device"

parallel_args=()
if [[ "${PARALLEL_GPUS:-}" == "1" || "${PARALLEL_GPUS:-}" == "true" ]]; then
    parallel_args=(--parallel-gpus)
elif [[ -n "${NUM_GPUS:-}" && "${NUM_GPUS}" =~ ^[0-9]+$ && "${NUM_GPUS}" -gt 1 ]]; then
    parallel_args=(--parallel-gpus)
fi
echo "parallel_gpus: $([[ ${#parallel_args[@]} -gt 0 ]] && echo on || echo off)"

labels_args=()
if [[ -n "${LABELS_CSV:-}" ]]; then
    if [[ ! -f "$LABELS_CSV" ]]; then
        echo "ERROR: labels CSV not found: $LABELS_CSV"
        exit 1
    fi
    labels_args=(--labels-csv "$LABELS_CSV")
fi
echo "Labels filter: ${LABELS_CSV:-off}"

exec "$PYTHON" -m kdoad.scripts.export_features \
    --input-folders "${input_folders[@]}" \
    --output-folder "$output_folder" \
    --checkpoint "$checkpoint" \
    --model X3DEmbedding \
    --feature-fps "$feature_fps" \
    --source-fps "$source_fps" \
    --frames "$frames" \
    --stride "$stride" \
    --input-clip-length "$input_clip_length" \
    --batch-size "$batch_size" \
    --device "$device" \
    "${labels_args[@]}" \
    "${parallel_args[@]}"
