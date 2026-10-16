# Copyright (c) 2026 Maksim Skorokhodov, National Research Centre "Kurchatov Institute"
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path
from typing import Final

import torch
import torch.nn
import torch.nn.functional as F
from pytorchvideo.models.hub.x3d import _x3d, x3d_xs

logger = logging.getLogger(__name__)

_STATE_DICT_PREFIXES: Final[tuple[str, ...]] = (
    "model.",
    "module.",
    "backbone.",
    "_orig_mod.",
)


class X3D(torch.nn.Module):
    """Base X3D-XS model with RGB clip preprocessing."""

    _MEAN: Final[tuple[float, float, float]] = (0.45, 0.45, 0.45)
    _STD: Final[tuple[float, float, float]] = (0.225, 0.225, 0.225)
    _X3D_XS_EMBEDDING_DIM: Final[int] = 2048
    _X3D_XS_DEFAULT_INPUT_CLIP_LENGTH: Final[int] = 4

    def __init__(
        self,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        side_size: int = 182,
        crop_size: int = 182,
        assume_uint8_scale: bool = False,
        input_clip_length: int = _X3D_XS_DEFAULT_INPUT_CLIP_LENGTH,
    ) -> None:
        """Initialize the base X3D-XS model.

        Args:
            pretrained: Load pretrained PyTorchVideo weights.
            checkpoint_path: Deprecated checkpoint path accepted for
                compatibility. The base model ignores this value.
            side_size: Size of the shorter side after spatial resizing.
            crop_size: Height and width of the center crop.
            assume_uint8_scale: Treat floating-point inputs as values in
                ``[0, 255]`` instead of ``[0, 1]``.
            input_clip_length: Architectural temporal input length.
        """
        super().__init__()

        self.input_clip_length = input_clip_length
        if input_clip_length == self._X3D_XS_DEFAULT_INPUT_CLIP_LENGTH:
            self.model = x3d_xs(pretrained=pretrained)
        else:
            # x3d_xs() already passes input_clip_length=4, so call the same
            # hub builder directly for other values.
            self.model = _x3d(
                pretrained=pretrained,
                input_clip_length=input_clip_length,
            )

        self.side_size = side_size
        self.crop_size = crop_size
        self._assume_uint8_scale = assume_uint8_scale

        mean = torch.tensor(self._MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor(self._STD, dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

        if checkpoint_path is not None:
            logger.warning(
                "checkpoint_path is ignored by the base X3D model; "
                "use X3DEmbedding instead.",
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the X3D model on a preprocessed video batch."""
        return self.model(x)

    def preprocess_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """Convert clip to ``[B, C, T, H, W]`` and apply video transforms.

        Args:
            clip: Video tensor with shape ``[T, C, H, W]``, ``[B, C, T, H, W]``,
                or ``[B, T, C, H, W]`` (DataLoader collate).

        Returns:
            Preprocessed tensor with shape ``[B, C, T, H', W']``.
        """
        clip = self._coerce_input_shape(clip)
        return self._preprocess_batch(clip)

    @staticmethod
    def _coerce_input_shape(clip: torch.Tensor) -> torch.Tensor:
        """Convert a clip to ``[B, C, T, H, W]`` format."""
        if clip.ndim == 4:
            return clip.permute(1, 0, 2, 3).unsqueeze(0)
        if clip.ndim == 5:
            if clip.shape[1] == 3:
                return clip
            if clip.shape[2] == 3:
                return clip.permute(0, 2, 1, 3, 4)
            raise ValueError(
                "Expected clip shape [B, C, T, H, W] or [B, T, C, H, W], "
                f"got {tuple(clip.shape)}."
            )
        raise ValueError(
            "Expected clip shape [T, C, H, W], [B, C, T, H, W] or "
            f"[B, T, C, H, W], got {tuple(clip.shape)}."
        )

    def _preprocess_batch(self, clip: torch.Tensor) -> torch.Tensor:
        """Apply X3D normalization, short-side scaling, and center cropping.

        Args:
            clip: A ``(B, 3, T, H, W)`` uint8 tensor in ``[0, 255]`` or
                floating-point tensor in ``[0, 1]``. When
                ``assume_uint8_scale`` is enabled, floating-point values are
                expected in ``[0, 255]``.

        Returns:
            A normalized and center-cropped ``(B, 3, T, H', W')`` tensor.

        Raises:
            ValueError: If the input does not have shape
                ``(B, 3, T, H, W)``.
        """
        if clip.ndim != 5:
            raise ValueError(
                "Expected clip shape (B, 3, T, H, W), "
                f"got {tuple(clip.shape)}"
            )
        if clip.shape[1] != 3:
            raise ValueError(
                "Expected clip shape (B, 3, T, H, W), "
                f"got {tuple(clip.shape)}"
            )

        if clip.dtype == torch.uint8:
            x = clip.float() / 255.0
        else:
            x = clip.float()
            if self._assume_uint8_scale:
                x = x / 255.0

        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std

        b, c, t, h, w = x.shape
        if h == self.crop_size and w == self.crop_size:
            return x

        return self._short_side_scale_and_center_crop(
            x,
            side_size=self.side_size,
            crop_size=self.crop_size,
        )

    @staticmethod
    def _short_side_scale_and_center_crop(
        x: torch.Tensor,
        side_size: int,
        crop_size: int,
    ) -> torch.Tensor:
        """Scale and center-crop a ``[B, C, T, H, W]`` batch."""
        b, c, t, h, w = x.shape

        if h >= w:
            new_w = side_size
            new_h = max(int(round(h * side_size / w)), 1)
        else:
            new_h = side_size
            new_w = max(int(round(w * side_size / h)), 1)

        flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        flat = F.interpolate(
            flat,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

        top = max((new_h - crop_size) // 2, 0)
        left = max((new_w - crop_size) // 2, 0)
        flat = flat[:, :, top:top + crop_size, left:left + crop_size]

        return flat.reshape(b, t, c, crop_size, crop_size).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _load_raw_checkpoint(checkpoint_path: str) -> object:
        """Load a checkpoint with a fallback for mixed checkpoint content."""
        resolved_path = Path(checkpoint_path).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")

        try:
            return torch.load(
                str(resolved_path),
                map_location="cpu",
                weights_only=True,
            )
        except Exception:
            return torch.load(
                str(resolved_path),
                map_location="cpu",
                weights_only=False,
            )

    @staticmethod
    def _extract_state_dict(raw_checkpoint: object) -> dict[str, torch.Tensor]:
        """Extract a state dict from supported checkpoint formats."""
        if not isinstance(raw_checkpoint, dict):
            msg = f"Unsupported checkpoint format: {type(raw_checkpoint)}"
            raise TypeError(msg)

        for key in ("model_state", "state_dict", "model", "module"):
            value = raw_checkpoint.get(key)
            if isinstance(value, dict):
                return value

        return raw_checkpoint

    @staticmethod
    def _clean_state_dict_keys(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Normalize state-dict keys."""
        cleaned_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            while True:
                stripped = False
                for prefix in _STATE_DICT_PREFIXES:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        stripped = True
                        break
                if not stripped:
                    break
            cleaned_state_dict[new_key] = value
        return cleaned_state_dict

    @staticmethod
    def _extract_backbone_state(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Extract X3D backbone weights and discard projection heads."""
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith("blocks.")
        }

    def _load_backbone_state(
        self,
        backbone_state: dict[str, torch.Tensor],
        *,
        label: str,
    ) -> None:
        """Load backbone weights into ``self.model``."""
        missing, unexpected = self.model.load_state_dict(
            backbone_state,
            strict=False,
        )
        if unexpected:
            logger.warning(
                "[%s] Unexpected backbone keys: %d",
                label,
                len(unexpected),
            )
        if missing:
            logger.warning(
                "[%s] Missing backbone keys: %d",
                label,
                len(missing),
            )


class X3DEmbedding(X3D):
    """X3D without its classification head, returning 2048-D embeddings."""

    def __init__(
        self,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        side_size: int = 182,
        crop_size: int = 182,
        assume_uint8_scale: bool = False,
        input_clip_length: int = X3D._X3D_XS_DEFAULT_INPUT_CLIP_LENGTH,
    ) -> None:
        """Initialize an X3D embedding model.

        Args:
            pretrained: Load pretrained PyTorchVideo weights before applying
                an optional checkpoint.
            checkpoint_path: Optional X3D or distillation checkpoint.
            side_size: Size of the shorter side after spatial resizing.
            crop_size: Height and width of the center crop.
            assume_uint8_scale: Treat floating-point inputs as values in
                ``[0, 255]`` instead of ``[0, 1]``.
            input_clip_length: Architectural temporal input length.
        """
        super().__init__(
            pretrained=pretrained,
            side_size=side_size,
            crop_size=crop_size,
            assume_uint8_scale=assume_uint8_scale,
            input_clip_length=input_clip_length,
        )
        backbone_dim = self.model.blocks[-1].proj.in_features
        if backbone_dim != self._X3D_XS_EMBEDDING_DIM:
            logger.warning(
                "Expected embedding_dim=%d, got %d",
                self._X3D_XS_EMBEDDING_DIM,
                backbone_dim,
            )
        self.embedding_dim = self._X3D_XS_EMBEDDING_DIM
        self._remove_classification_head()
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return 2048-dimensional embeddings for a preprocessed batch."""
        embeddings = self.model(x)
        if embeddings.ndim > 2:
            embeddings = embeddings.flatten(start_dim=1)
        return embeddings

    def load_checkpoint(
        self,
        checkpoint_path: str,
    ) -> None:
        """Load backbone weights from an X3D or distillation checkpoint.

        Args:
            checkpoint_path: Path to a ``.pyth`` or ``.pt`` checkpoint.
        """
        raw_checkpoint = self._load_raw_checkpoint(checkpoint_path)
        state_dict = self._clean_state_dict_keys(
            self._extract_state_dict(raw_checkpoint),
        )
        backbone_state = self._extract_backbone_state(state_dict)
        if not backbone_state:
            raise ValueError(
                f"No X3D backbone weights found in checkpoint: {checkpoint_path}"
            )
        self._load_backbone_state(backbone_state, label="X3DEmbedding")

    def _remove_classification_head(self) -> None:
        """Replace the classification projection and activation with identities."""
        head = self.model.blocks[-1]
        if hasattr(head, "proj"):
            head.proj = torch.nn.Identity()
        if hasattr(head, "activation"):
            head.activation = torch.nn.Identity()
