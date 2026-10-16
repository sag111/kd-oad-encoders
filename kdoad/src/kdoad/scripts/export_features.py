# Copyright (c) 2026 Maksim Skorokhodov, National Research Centre "Kurchatov Institute"
# SPDX-License-Identifier: MIT

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import torch
from tqdm import tqdm

from kdoad.data.utils import _import_torchcodec
from kdoad.models.x3d import X3DEmbedding

logger = logging.getLogger(__name__)

_DEFAULT_FRAMES: Final[int] = 6
_DEFAULT_STRIDE: Final[int] = 1
_DEFAULT_INPUT_CLIP_LENGTH: Final[int] = 4


FeatureMetadata = dict[str, object]


class WorkerPayload(TypedDict):
    """Serializable arguments passed to a parallel export worker."""

    video_files: list[str]
    output_folder: str
    checkpoint: str
    feature_fps: float
    source_fps: float | None
    frames: int
    stride: int
    batch_size: int
    input_clip_length: int


def _resolve_checkpoint_path(checkpoint_path: str | None) -> str | None:
    """Normalize a checkpoint string to a path or ``None``."""
    if checkpoint_path is None:
        return None
    normalized = checkpoint_path.strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    return normalized


@dataclass(frozen=True)
class ClipSamplingConfig:
    """Temporal frame sampling parameters for one clip."""

    frames: int
    stride: int
    anchor_stride: int

    @property
    def context_offsets(self) -> tuple[int, ...]:
        """Return frame offsets relative to an anchor frame."""
        return tuple(
            -(self.frames - 1 - index) * self.stride
            for index in range(self.frames)
        )

    @property
    def temporal_span(self) -> int:
        """Return the number of source frames covered by one clip."""
        return abs(min(self.context_offsets))


def resolve_anchor_stride(source_fps: float, feature_fps: float) -> int:
    """Calculate the anchor-frame stride from source and feature FPS."""
    if source_fps <= 0 or feature_fps <= 0:
        msg = f"FPS values must be positive, got {source_fps=} {feature_fps=}"
        raise ValueError(msg)
    return max(int(round(source_fps / feature_fps)), 1)


def find_videos(input_folders: list[str]) -> list[Path]:
    """Find all video files in the specified folders."""
    video_extensions = {".mp4", ".avi", ".MP4", ".AVI"}
    video_files: list[Path] = []

    for folder in input_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning("Folder does not exist: %s", folder)
            continue

        for ext in video_extensions:
            video_files.extend(folder_path.glob(f"*{ext}"))

    return sorted(video_files)


def load_labeled_video_names(labels_csv_path: str) -> set[str]:
    """Load unique video names from a CSV containing ``video_name``."""
    labels_path = Path(labels_csv_path)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv_path}")

    video_names: set[str] = set()
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "video_name" not in reader.fieldnames:
            msg = f"Column 'video_name' not found in {labels_csv_path}"
            raise ValueError(msg)

        for row in reader:
            video_name = row.get("video_name", "").strip()
            if video_name:
                video_names.add(video_name)

    return video_names


def filter_videos_by_labels(
    video_files: list[Path],
    labels_csv_path: str,
) -> list[Path]:
    """Keep videos whose stems are present in the labels CSV."""
    labeled_names = load_labeled_video_names(labels_csv_path)
    return [
        video_file
        for video_file in video_files
        if video_file.stem in labeled_names
    ]


def resolve_torch_device(device: str) -> str:
    """Resolve a torch device, falling back to CPU when CUDA is unavailable."""
    if device == "cpu":
        return "cpu"
    if device == "cuda" or device.startswith("cuda:"):
        if not torch.cuda.is_available():
            logger.warning("CUDA is unavailable; falling back to CPU")
            return "cpu"
        return device
    return device


def build_model(
    checkpoint_path: str | None,
    device: str,
    *,
    input_clip_length: int = _DEFAULT_INPUT_CLIP_LENGTH,
) -> X3DEmbedding:
    """Create an X3D embedding model and prepare it for inference."""
    resolved_device = resolve_torch_device(device)
    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
    model = X3DEmbedding(
        pretrained=False,
        checkpoint_path=resolved_checkpoint,
        input_clip_length=input_clip_length,
    )
    model = model.to(resolved_device)
    model.eval()
    return model


def _close_decoder(decoder: object) -> None:
    """Close a VideoDecoder when it exposes a ``close`` method."""
    close = getattr(decoder, "close", None)
    if callable(close):
        close()


def _flat_frame_indices(
    batch_anchors: list[int],
    sampling: ClipSamplingConfig,
    total_frames: int,
) -> list[int]:
    """Build a flat frame-index list for a batch of anchor frames."""
    flat_indices: list[int] = []
    for anchor in batch_anchors:
        for offset in sampling.context_offsets:
            flat_indices.append(min(max(anchor + offset, 0), total_frames - 1))
    return flat_indices


def _decode_clip_batch(
    decoder: object,
    batch_anchors: list[int],
    sampling: ClipSamplingConfig,
    total_frames: int,
    *,
    pin_memory: bool,
) -> tuple[torch.Tensor, int]:
    """Decode a clip batch into a ``(B, C, T, H, W)`` CPU tensor.

    Args:
        decoder: torchcodec VideoDecoder.
        batch_anchors: Anchor-frame indices.
        sampling: Temporal sampling parameters.
        total_frames: Total number of frames in the video.
        pin_memory: Pin memory for non-blocking host-to-device transfers.

    Returns:
        The clip tensor and batch size.
    """
    flat_indices = _flat_frame_indices(batch_anchors, sampling, total_frames)
    frames = decoder.get_frames_at(indices=flat_indices).data
    batch_len = len(batch_anchors)
    # (B * T, C, H, W) -> (B, C, T, H, W)
    frames = frames.reshape(
        batch_len,
        sampling.frames,
        frames.shape[1],
        frames.shape[2],
        frames.shape[3],
    )
    batch = frames.permute(0, 2, 1, 3, 4).contiguous()
    if pin_memory and batch.device.type == "cpu":
        batch = batch.pin_memory()
    return batch, batch_len


class VideoFeatureExtractor:
    """Extract video features with the torchcodec VideoDecoder."""

    def __init__(
        self,
        model: X3DEmbedding,
        feature_fps: float,
        source_fps: float | None,
        frames: int,
        stride: int,
        batch_size: int = 32,
    ) -> None:
        """Initialize a video feature extractor.

        Args:
            model: X3D embedding model used for inference.
            feature_fps: Frequency of output feature vectors.
            source_fps: Source frame rate used to calculate anchor spacing.
                Video metadata supplies the value when this is ``None``.
            frames: Number of sampled frames in each clip.
            stride: Distance between sampled frames within a clip.
            batch_size: Number of clips processed per model call.
        """
        self.model = model
        self.feature_fps = feature_fps
        self.source_fps = source_fps
        self.frames = frames
        self.stride = stride
        self.device = next(model.parameters()).device
        self.batch_size = batch_size
        self._video_decoder_cls = _import_torchcodec()
        self._copy_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

    def _transfer_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Transfer a batch to the model device asynchronously on CUDA.

        The copy uses a separate stream and can overlap with the previous
        batch's forward pass. ``synchronize`` keeps the pinned CPU buffer
        alive until the host-to-device transfer is complete.
        """
        if self.device.type != "cuda" or self._copy_stream is None:
            return batch.to(self.device)

        with torch.cuda.stream(self._copy_stream):
            gpu_batch = batch.to(self.device, non_blocking=True)
        self._copy_stream.synchronize()
        return gpu_batch

    def _infer_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Run a ``(B, C, T, H, W)`` clip batch through the model.

        Returns:
            Float32 features on the model device with shape
            ``(B, feature_dim)``. The CPU transfer is deferred until the
            entire video is processed to avoid per-batch GPU synchronization.
        """
        batch = self._transfer_batch(batch)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            processed_batch = self.model.preprocess_clip(batch)
            features = self.model(processed_batch)
        return features.float()

    def extract_features_from_video(
        self,
        video_path: str,
    ) -> tuple[np.ndarray, FeatureMetadata]:
        """Extract features from one video.

        Frames are read in clip batches with torchcodec ``get_frames_at``.
        The next batch is decoded in the background while the GPU processes
        the current batch
        (decode[N+1] || H2D[N] + forward[N]).

        Args:
            video_path: Path to the video file.

        Returns:
            A ``(num_clips, feature_dim)`` feature array and metadata.
        """
        decoder = self._video_decoder_cls(video_path, device="cpu")
        try:
            meta = decoder.metadata
            if meta.num_frames is None or int(meta.num_frames) <= 0:
                raise ValueError(f"Unable to read frames from: {video_path}")
            if meta.average_fps is None or float(meta.average_fps) <= 0:
                raise ValueError(f"Invalid FPS in video metadata: {video_path}")

            original_fps = float(meta.average_fps)
            total_frames = int(meta.num_frames)
            original_width = int(meta.width) if meta.width is not None else 0
            original_height = int(meta.height) if meta.height is not None else 0

            effective_source_fps = (
                self.source_fps if self.source_fps is not None else original_fps
            )
            sampling = ClipSamplingConfig(
                frames=self.frames,
                stride=self.stride,
                anchor_stride=resolve_anchor_stride(
                    effective_source_fps,
                    self.feature_fps,
                ),
            )

            anchors = list(range(0, total_frames, sampling.anchor_stride))
            video_name = Path(video_path).name
            feature_chunks: list[torch.Tensor] = []
            anchor_count = 0
            pin_memory = self.device.type == "cuda"
            batch_ranges = [
                anchors[start : start + self.batch_size]
                for start in range(0, len(anchors), self.batch_size)
            ]

            def decode_job(batch_anchors: list[int]) -> tuple[torch.Tensor, int]:
                """Decode one batch in the decoder's sole worker thread."""
                return _decode_clip_batch(
                    decoder,
                    batch_anchors,
                    sampling,
                    total_frames,
                    pin_memory=pin_memory,
                )

            with tqdm(
                total=len(anchors),
                desc=f"Extracting [{video_name}]",
                leave=False,
            ) as progress:
                if batch_ranges:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        pending: Future[tuple[torch.Tensor, int]] = pool.submit(
                            decode_job,
                            batch_ranges[0],
                        )
                        for batch_idx, _batch_anchors in enumerate(batch_ranges):
                            batch, batch_len = pending.result()
                            next_idx = batch_idx + 1
                            # Start decode[N+1] before forward[N].
                            if next_idx < len(batch_ranges):
                                pending = pool.submit(
                                    decode_job,
                                    batch_ranges[next_idx],
                                )
                            feature_chunks.append(self._infer_batch(batch))
                            anchor_count += batch_len
                            progress.update(batch_len)

            if not feature_chunks:
                features = np.array([])
            else:
                features = torch.cat(feature_chunks, dim=0).cpu().numpy()

            metadata: FeatureMetadata = {
                "model": type(self.model).__name__,
                "frames": sampling.frames,
                "stride": sampling.stride,
                "original_fps": original_fps,
                "effective_source_fps": effective_source_fps,
                "feature_fps": self.feature_fps,
                "anchor_stride": sampling.anchor_stride,
                "context_offsets": list(sampling.context_offsets),
                "original_resolution": [original_width, original_height],
                "total_frames_in_video": total_frames,
                "num_clips": anchor_count,
                "feature_shape": list(features.shape),
                "source_video": video_name,
                "decoder": "torchcodec",
            }
            if hasattr(self.model, "embedding_dim"):
                metadata["embedding_dim"] = self.model.embedding_dim
            return features, metadata
        finally:
            _close_decoder(decoder)

    def process_video(self, video_path: str, output_path: str) -> bool:
        """Process one video and save its features."""
        output_file = Path(output_path)
        if output_file.is_file():
            logger.info("SKIP (already exists): %s", output_path)
            return True

        logger.info("Processing: %s", video_path)
        features, metadata = self.extract_features_from_video(video_path)

        if features.size == 0:
            logger.error("No features extracted from: %s", video_path)
            return False

        np.save(output_path, features)

        metadata_path = output_path.replace(".npy", "_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        logger.info("Saved: %s (%s)", output_path, features.shape)
        return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract X3D embedding features from videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-folders",
        nargs="+",
        required=True,
        help="Folders containing input videos",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Output folder for .npy files",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--model",
        choices=["X3DEmbedding"],
        default="X3DEmbedding",
        help="Model class used for inference",
    )
    parser.add_argument(
        "--feature-fps",
        type=float,
        required=True,
        help=(
            "Anchor clip frequency: anchor_stride = source_fps / feature_fps "
            "(for example, 24/4=6 starts a clip every 6 source frames)"
        ),
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Source video FPS; read from metadata when omitted",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=_DEFAULT_FRAMES,
        help="Number of frames in each model clip",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=_DEFAULT_STRIDE,
        help="Frame stride within a clip (1 means consecutive frames)",
    )
    parser.add_argument(
        "--input-clip-length",
        type=int,
        default=_DEFAULT_INPUT_CLIP_LENGTH,
        help=(
            "Architectural X3D clip length used by temporal head pooling; "
            "independent of --frames"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Clip batch size",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help=(
            "Inference device; ignored with --parallel-gpus, which uses "
            "cuda:0 through cuda:N-1"
        ),
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help=(
            "Filter videos by video_name from a CSV; process every input "
            "video when omitted"
        ),
    )
    parser.add_argument(
        "--parallel-gpus",
        action="store_true",
        help=(
            "Shard videos across all visible CUDA devices, respecting "
            "CUDA_VISIBLE_DEVICES; use one process by default"
        ),
    )
    return parser.parse_args()


def _log_sampling_config(
    source_fps: float | None,
    feature_fps: float,
    frames: int,
    stride: int,
    input_clip_length: int,
) -> None:
    """Log temporal sampling parameters."""
    if source_fps is not None:
        anchor_stride = resolve_anchor_stride(source_fps, feature_fps)
        logger.info(
            "source_fps=%.3f feature_fps=%.3f anchor_stride=%d "
            "frames=%d stride=%d input_clip_length=%d",
            source_fps,
            feature_fps,
            anchor_stride,
            frames,
            stride,
            input_clip_length,
        )
    else:
        logger.info(
            "source_fps from video metadata, feature_fps=%.3f "
            "frames=%d stride=%d input_clip_length=%d",
            feature_fps,
            frames,
            stride,
            input_clip_length,
        )


def export_video_shard(
    video_files: list[Path],
    output_path: Path,
    *,
    checkpoint: str,
    device: str,
    feature_fps: float,
    source_fps: float | None,
    frames: int,
    stride: int,
    batch_size: int,
    input_clip_length: int = _DEFAULT_INPUT_CLIP_LENGTH,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[int, int]:
    """Process a video-list shard on one device.

    Returns:
        A ``(successful, failed)`` tuple.
    """
    shard = video_files[rank::world_size]
    logger.info(
        "rank=%d/%d device=%s: %d videos",
        rank,
        world_size,
        device,
        len(shard),
    )
    if not shard:
        return 0, 0

    model = build_model(
        checkpoint,
        device,
        input_clip_length=input_clip_length,
    )
    extractor = VideoFeatureExtractor(
        model=model,
        feature_fps=feature_fps,
        source_fps=source_fps,
        frames=frames,
        stride=stride,
        batch_size=batch_size,
    )

    successful = 0
    failed = 0
    for video_path in shard:
        output_file = output_path / f"{video_path.stem}.npy"
        if extractor.process_video(str(video_path), str(output_file)):
            successful += 1
        else:
            failed += 1
    return successful, failed


def _parallel_worker(
    rank: int,
    world_size: int,
    payload: WorkerPayload,
) -> None:
    """Process one video shard in a worker on ``cuda:{rank}``."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [gpu{rank}] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    video_files = [Path(path) for path in payload["video_files"]]
    output_path = Path(payload["output_folder"])
    started_at = time.perf_counter()
    try:
        successful, failed = export_video_shard(
            video_files,
            output_path,
            checkpoint=payload["checkpoint"],
            device=f"cuda:{rank}",
            feature_fps=payload["feature_fps"],
            source_fps=payload["source_fps"],
            frames=payload["frames"],
            stride=payload["stride"],
            batch_size=payload["batch_size"],
            input_clip_length=payload["input_clip_length"],
            rank=rank,
            world_size=world_size,
        )
    except KeyboardInterrupt:
        elapsed_s = time.perf_counter() - started_at
        logger.warning(
            "Interrupted by user after %.2f seconds",
            elapsed_s,
        )
        raise

    elapsed_s = time.perf_counter() - started_at
    logger.info(
        "Done: success=%d failed=%d elapsed time %.2f s",
        successful,
        failed,
        elapsed_s,
    )


def _collect_video_files(args: argparse.Namespace) -> list[Path]:
    """Find and filter videos according to CLI arguments."""
    video_files = find_videos(args.input_folders)
    if not video_files:
        logger.error("No videos found in the input folders")
        return []

    total_before_filter = len(video_files)
    if args.labels_csv is not None:
        video_files = filter_videos_by_labels(video_files, args.labels_csv)
        logger.info(
            "Labels filter: %s -> %d of %d videos",
            args.labels_csv,
            len(video_files),
            total_before_filter,
        )

    if not video_files:
        logger.error("No videos remain after labels filtering")
        return []

    logger.info("Videos to process: %d", len(video_files))
    return video_files


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = _collect_video_files(args)
    if not video_files:
        return

    _log_sampling_config(
        args.source_fps,
        args.feature_fps,
        args.frames,
        args.stride,
        args.input_clip_length,
    )

    use_parallel = (
        args.parallel_gpus
        and args.device == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    )

    if args.parallel_gpus and not use_parallel:
        logger.warning(
            "--parallel-gpus requested, but fewer than two GPUs are visible "
            "(device=%s, cuda_available=%s, count=%d); using one process",
            args.device,
            torch.cuda.is_available(),
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )

    started_at = time.perf_counter()

    if use_parallel:
        world_size = torch.cuda.device_count()
        logger.info(
            "Parallel export on %d GPUs (respecting CUDA_VISIBLE_DEVICES)",
            world_size,
        )
        payload: WorkerPayload = {
            "video_files": [str(path) for path in video_files],
            "output_folder": str(output_path),
            "checkpoint": args.checkpoint,
            "feature_fps": args.feature_fps,
            "source_fps": args.source_fps,
            "frames": args.frames,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "input_clip_length": args.input_clip_length,
        }
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
            torch.multiprocessing.spawn(
                _parallel_worker,
                args=(world_size, payload),
                nprocs=world_size,
                join=True,
            )
        except KeyboardInterrupt:
            elapsed_s = time.perf_counter() - started_at
            logger.warning(
                "Interrupted by user after %.2f seconds",
                elapsed_s,
            )
            sys.exit(130)

        elapsed_s = time.perf_counter() - started_at
        logger.info(
            "All workers finished: total=%d elapsed time %.2f s",
            len(video_files),
            elapsed_s,
        )
        return

    try:
        successful, failed = export_video_shard(
            video_files,
            output_path,
            checkpoint=args.checkpoint,
            device=args.device,
            feature_fps=args.feature_fps,
            source_fps=args.source_fps,
            frames=args.frames,
            stride=args.stride,
            batch_size=args.batch_size,
            input_clip_length=args.input_clip_length,
        )
    except KeyboardInterrupt:
        elapsed_s = time.perf_counter() - started_at
        logger.warning(
            "Interrupted by user after %.2f seconds",
            elapsed_s,
        )
        sys.exit(130)

    elapsed_s = time.perf_counter() - started_at
    logger.info(
        "Done: total=%d success=%d failed=%d elapsed time %.2f s",
        len(video_files),
        successful,
        failed,
        elapsed_s,
    )


if __name__ == "__main__":
    main()
