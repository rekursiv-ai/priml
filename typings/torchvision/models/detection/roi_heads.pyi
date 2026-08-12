from typing import Any
from torch import Tensor
from torch import nn

import torch

def fastrcnn_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: list[torch.Tensor],
    regression_targets: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]: ...
def maskrcnn_inference(
    x: torch.Tensor, labels: list[torch.Tensor]
) -> list[torch.Tensor]: ...
def project_masks_on_boxes(
    gt_masks: Tensor, boxes: Tensor, matched_idxs: Tensor, M: int
) -> Tensor: ...
def maskrcnn_loss(
    mask_logits: Tensor,
    proposals: list[Tensor],
    gt_masks: list[Tensor],
    gt_labels: list[Tensor],
    mask_matched_idxs: list[Tensor],
) -> Tensor: ...
def keypoints_to_heatmap(
    keypoints: Tensor, rois: Tensor, heatmap_size: int
) -> tuple[Tensor, Tensor]: ...
def heatmaps_to_keypoints(maps, rois):  # -> tuple[Tensor, Tensor]:
    ...
def keypointrcnn_loss(
    keypoint_logits: Tensor,
    proposals: list[Tensor],
    gt_keypoints: list[Tensor],
    keypoint_matched_idxs: list[Tensor],
) -> Tensor: ...
def keypointrcnn_inference(
    x: Tensor, boxes: list[Tensor]
) -> tuple[list[Tensor], list[Tensor]]: ...
def expand_boxes(boxes: Tensor, scale: float) -> Tensor: ...
@torch.jit.unused
def expand_masks_tracing_scale(M: int, padding: int) -> float: ...
def expand_masks(mask: Tensor, padding: int) -> tuple[Tensor, float]: ...
def paste_mask_in_image(mask: Tensor, box: Tensor, im_h: int, im_w: int) -> Tensor: ...
def paste_masks_in_image(
    masks: Tensor, boxes: Tensor, img_shape: tuple[int, int], padding: int = ...
) -> Tensor: ...

class RoIHeads(nn.Module):
    __annotations__ = ...
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        score_thresh,
        nms_thresh,
        detections_per_img,
        mask_roi_pool=...,
        mask_head=...,
        mask_predictor=...,
        keypoint_roi_pool=...,
        keypoint_head=...,
        keypoint_predictor=...,
    ) -> None: ...
    def has_mask(self):  # -> bool:
        ...
    def has_keypoint(self):  # -> bool:
        ...
    def assign_targets_to_proposals(
        self, proposals: list[Tensor], gt_boxes: list[Tensor], gt_labels: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]: ...
    def subsample(self, labels: list[Tensor]) -> list[Tensor]: ...
    def add_gt_proposals(
        self, proposals: list[Tensor], gt_boxes: list[Tensor]
    ) -> list[Tensor]: ...
    def check_targets(self, targets: list[dict[str, Tensor]] | None) -> None: ...
    def select_training_samples(
        self, proposals: list[Tensor], targets: list[dict[str, Tensor]] | None
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]: ...
    def postprocess_detections(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]: ...
    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]] | None = ...,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]: ...
