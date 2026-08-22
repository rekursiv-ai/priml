from typing import Any

from torch import Tensor, nn

import torch

from .anchor_utils import AnchorGenerator
from .image_list import ImageList

class RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, conv_depth=...) -> None: ...
    def forward(self, x: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[Tensor], list[Tensor]]: ...

def permute_and_flatten(
    layer: Tensor, N: int, A: int, C: int, H: int, W: int
) -> Tensor: ...
def concat_box_prediction_layers(
    box_cls: list[Tensor], box_regression: list[Tensor]
) -> tuple[Tensor, Tensor]: ...

class RegionProposalNetwork(torch.nn.Module):
    __annotations__ = ...
    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_top_n: dict[str, int],
        post_nms_top_n: dict[str, int],
        nms_thresh: float,
        score_thresh: float = ...,
    ) -> None: ...
    def pre_nms_top_n(self) -> int: ...
    def post_nms_top_n(self) -> int: ...
    def assign_targets_to_anchors(
        self, anchors: list[Tensor], targets: list[dict[str, Tensor]]
    ) -> tuple[list[Tensor], list[Tensor]]: ...
    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: list[tuple[int, int]],
        num_anchors_per_level: list[int],
    ) -> tuple[list[Tensor], list[Tensor]]: ...
    def compute_loss(
        self,
        objectness: Tensor,
        pred_bbox_deltas: Tensor,
        labels: list[Tensor],
        regression_targets: list[Tensor],
    ) -> tuple[Tensor, Tensor]: ...
    def forward(
        self,
        images: ImageList,
        features: dict[str, Tensor],
        targets: list[dict[str, Tensor]] | None = ...,
    ) -> tuple[list[Tensor], dict[str, Tensor]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[Tensor], dict[str, Tensor]]: ...
