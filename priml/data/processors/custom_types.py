"""Type definitions for media processing.

This module contains type definitions used across media processing components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.data.custom_types import FilterableSample


if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from torch import Tensor

    from priml.data.sources.tarhandle import TarFileProtocol


__all__ = [
    "Sample",
]


class Sample(FilterableSample, total=False):
    """Sample dictionary schema.

    All fields are optional to support progressive enrichment through pipeline.
    Samples start with metadata, get filtered, then processed into tensors.
    """

    # Common metadata.
    key: str
    shard_index: int
    file_name: str
    file_path: str
    """On-disk path; set for extracted datasets, absent for tar shards."""

    format: str
    """Extension without the dot, such as ``jpg``, ``webp`` or ``png``."""

    # Text-to-image / diffusion fields.
    url: str
    caption: str

    # Image/video metadata.
    image: PILImage
    """Decoded image before tensorization; ``media_tensor`` supersedes it."""

    frames: int
    width: int
    height: int
    fps: float

    # Resize dimensions.
    target_frames: int
    target_width: int
    target_height: int

    # Processed tensors.
    media_tensor: Tensor

    # Classification fields.
    label: int | str
    """Class index when int, synset or class name when str."""

    # Embedding processors (from core.data.processors.embedding)
    clip_embeddings: dict[int, list[float]]
    siglip1_embeddings: dict[int, list[float]]
    siglip2_embeddings: dict[int, list[float]]
    dino2_embeddings: dict[int, list[float]]
    dino3_embeddings: dict[int, list[float]]

    # OpticalFlowCV2.
    optical_flow_mean: float
    optical_flow_median: float
    optical_flow_raw: dict[int, float]
    camera_motion: dict[int, list[float]]

    # MotionEstimator.
    mean_fg_flow: float
    mean_bg_flow: float
    median_fg_flow: float
    median_bg_flow: float
    max_fg_flow: float
    max_bg_flow: float
    foreground_percent: float

    # Quality and content scores: Tensor during the GPU pipeline, dict after
    # unbatch, list after a parquet round trip.
    aesthetic_score: float
    aesthetic_scores: Tensor | dict[int, float] | list[float]
    qalign_score: float
    qalign_scores: Tensor | dict[int, float] | list[float]
    nsfw_scores: Tensor | dict[int, float] | list[float]
    nsfw_scores_v2: Tensor | dict[int, float] | list[float]
    watermark_scores: Tensor | dict[int, float] | list[float]

    # FirstFrameLastFrameSimilarityScore / KenBurnsScore.
    first_frame_last_frame_similarity: float
    is_ken_burns: bool
    ken_burns_reprojection_error: float
    ken_burns_optical_flow: float

    # EasyOCRBoundingBoxProcessor.
    text_bbox_coverage_ratios: dict[int, float]
    text_bbox_detections: dict[int, list[list[float]]]
    text_bbox_best_crops: dict[int, tuple[tuple[int, int], tuple[int, int]]]

    _filter_counts: dict[str, int]
    _tar_handle: TarFileProtocol
