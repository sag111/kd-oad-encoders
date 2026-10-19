# Copyright (c) 2026 Maksim Skorokhodov, National Research Centre "Kurchatov Institute"
# SPDX-License-Identifier: MIT

"""Convert TVSeries labels and feature files to target_perframe for TeSTra.

For each session (for example, ``Breaking_Bad_ep1``), length ``L`` is read
from its feature ``.npy`` file. Intervals from ``labels.csv`` are converted
to frame-level one-hot targets of shape ``(L, num_classes)`` according to
LSTR rules (4 FPS, background = class 0) and saved to
``target_perframe/{session}.npy``.

Example for TVSeries::

    python -m kdoad.scripts.labels_to_target_perframe_lstr \\
        --labels-csv datasets/TVSeries/labels.csv \\
        --feature-dir datasets/TVSeries/features/repr_x3d_test \\
        --output-dir datasets/TVSeries/target_perframe \\
        --testra-data-info testra/data/data_info.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_NAME = "TVSeries"
BACKGROUND_NAME = "Background"
DEFAULT_LABEL_FPS = 4


def load_class_map(testra_data_info: Path) -> dict[str, int]:
    """Build a mapping from TVSeries class names to TeSTra indices.

    Parameters
    ----------
    testra_data_info:
        Path to ``data/data_info.json`` from the TeSTra repository.

    Returns
    -------
    dict[str, int]
        Class name to index mapping (Background = 0 for TVSeries).
    """
    data_info = json.loads(testra_data_info.read_text(encoding="utf-8"))
    try:
        class_names = data_info[DATASET_NAME]["class_names"]
    except KeyError as exc:
        msg = f"{testra_data_info} does not contain {DATASET_NAME} class names"
        raise KeyError(msg) from exc
    return {name: idx for idx, name in enumerate(class_names)}


def intervals_to_onehot(
    segments: pd.DataFrame,
    length: int,
    class_map: dict[str, int],
    *,
    label_fps: int = DEFAULT_LABEL_FPS,
) -> np.ndarray:
    """Build an LSTR-like one-hot target from labels.csv intervals.

    Parameters
    ----------
    segments:
        ``labels.csv`` rows for one session.
    length:
        Number of temporal steps (usually ``feature.shape[0]``).
    class_map:
        LSTR class name to index mapping.
    label_fps:
        Label frequency (4 FPS for TVSeries/TeSTra).

    Returns
    -------
    np.ndarray
        Array of shape ``(length, num_classes)``, dtype float32, values 0/1.
    """
    num_classes = len(class_map)
    if BACKGROUND_NAME not in class_map:
        msg = f"class_map does not contain class '{BACKGROUND_NAME}'."
        raise KeyError(msg)

    target = np.zeros((length, num_classes), dtype=np.float32)
    background_idx = class_map[BACKGROUND_NAME]

    for _, row in segments.iterrows():
        raw_tag = str(row["tag"]).strip()
        tag = BACKGROUND_NAME if raw_tag.lower() == "background" else raw_tag
        if tag == BACKGROUND_NAME:
            continue

        if tag not in class_map:
            msg = (
                f"Unknown tag '{row['tag']}' (normalized as '{tag}'). "
                "Check labels.csv against the TeSTra class order."
            )
            raise ValueError(msg)

        cls_idx = class_map[tag]
        start_frame = math.floor(float(row["start"]) * label_fps)
        stop_frame = math.ceil(float(row["stop"]) * label_fps) - 1

        if stop_frame < 0 or start_frame >= length:
            continue

        start_frame = max(0, start_frame)
        stop_frame = min(length - 1, stop_frame)
        target[start_frame : stop_frame + 1, cls_idx] = 1.0

    action_cols = [idx for idx in range(num_classes) if idx != background_idx]
    background_mask = np.all(target[:, action_cols] == 0.0, axis=1)
    target[background_mask, background_idx] = 1.0
    return target


def discover_sessions(feature_dir: Path) -> list[str]:
    """Return session names from ``*.npy`` files in the feature directory."""
    return sorted(path.stem for path in feature_dir.glob("*.npy"))


def convert_dataset(
    *,
    labels_csv: Path,
    feature_dir: Path,
    output_dir: Path,
    class_map: dict[str, int],
    sessions: list[str] | None = None,
    label_fps: int = DEFAULT_LABEL_FPS,
) -> int:
    """Convert all sessions and save target_perframe files.

    Returns
    -------
    int
        Number of processed sessions.
    """
    labels = pd.read_csv(labels_csv)
    required_columns = {"video_name", "start", "stop", "tag"}
    missing = required_columns - set(labels.columns)
    if missing:
        msg = f"{labels_csv} is missing columns: {sorted(missing)}"
        raise ValueError(msg)

    session_list = sessions or discover_sessions(feature_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for session in session_list:
        feature_path = feature_dir / f"{session}.npy"
        if not feature_path.is_file():
            print(f"SKIP: feature file not found: {feature_path}")
            continue

        feature = np.load(feature_path, mmap_mode="r")
        if feature.ndim != 2:
            msg = (
                f"Expected a 2D feature array in {feature_path}, "
                f"got shape {feature.shape}"
            )
            raise ValueError(msg)

        length = int(feature.shape[0])
        segments = labels[labels["video_name"] == session]
        target = intervals_to_onehot(
            segments,
            length,
            class_map,
            label_fps=label_fps,
        )

        out_path = output_dir / f"{session}.npy"
        np.save(out_path, target)

        processed += 1
        active_classes = int((target.sum(axis=0) > 0).sum())
        print(
            f"OK {session}: L={length}, active_classes={active_classes}, "
            f"saved {out_path}",
        )

    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert TVSeries labels.csv and feature .npy files to "
            "target_perframe for LSTR/TeSTra."
        ),
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        required=True,
        help="Path to labels.csv with video_name, start, stop, and tag columns.",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Directory with feature .npy files (L is read from shape[0]).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for target_perframe/{session}.npy.",
    )
    parser.add_argument(
        "--testra-data-info",
        type=Path,
        required=True,
        help="Path to TeSTra data/data_info.json.",
    )
    parser.add_argument(
        "--label-fps",
        type=int,
        default=DEFAULT_LABEL_FPS,
        help="Frame-level label frequency (default: 4).",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Session list; defaults to every *.npy file in feature-dir.",
    )
    return parser


def main() -> None:
    """Run the command-line converter."""
    args = build_arg_parser().parse_args()

    class_map = load_class_map(args.testra_data_info)
    processed = convert_dataset(
        labels_csv=args.labels_csv,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        class_map=class_map,
        sessions=args.sessions,
        label_fps=args.label_fps,
    )

    if not processed:
        print("No sessions were processed.")
        return

    print(f"Done: {processed} sessions written to {args.output_dir}")


if __name__ == "__main__":
    main()
