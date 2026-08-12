from typing import Any
from collections.abc import Callable

from torch import Tensor, nn
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock

from .. import mobilenet, resnet
from .._api import WeightsEnum, _get_enum_from_fn
from .._utils import handle_legacy_interface

class BackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        return_layers: dict[str, str],
        in_channels_list: list[int],
        out_channels: int,
        extra_blocks: ExtraFPNBlock | None = ...,
        norm_layer: Callable[..., nn.Module] | None = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> dict[str, Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]: ...

@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: _get_enum_from_fn(resnet.__dict__[kwargs["backbone_name"]])[
            "IMAGENET1K_V1"
        ],
    )
)
def resnet_fpn_backbone(
    *,
    backbone_name: str,
    weights: WeightsEnum | None,
    norm_layer: Callable[..., nn.Module] = ...,
    trainable_layers: int = ...,
    returned_layers: list[int] | None = ...,
    extra_blocks: ExtraFPNBlock | None = ...,
) -> BackboneWithFPN: ...
@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: _get_enum_from_fn(mobilenet.__dict__[kwargs["backbone_name"]])[
            "IMAGENET1K_V1"
        ],
    )
)
def mobilenet_backbone(
    *,
    backbone_name: str,
    weights: WeightsEnum | None,
    fpn: bool,
    norm_layer: Callable[..., nn.Module] = ...,
    trainable_layers: int = ...,
    returned_layers: list[int] | None = ...,
    extra_blocks: ExtraFPNBlock | None = ...,
) -> nn.Module: ...
