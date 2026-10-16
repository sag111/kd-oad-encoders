# Copyright (c) 2026 Maksim Skorokhodov, National Research Centre "Kurchatov Institute"
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torchcodec.decoders import VideoDecoder


def _import_torchcodec() -> type[VideoDecoder]:
    """Lazily import VideoDecoder from torchcodec."""
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as exc:
        raise ImportError("torchcodec is not installed") from exc
    return VideoDecoder
