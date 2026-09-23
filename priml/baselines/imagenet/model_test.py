"""Tests for the ffcv-imagenet ResNet."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Final

from configgle import PartialConfig
from torch import nn
from torchvision.models import resnet18, resnet50

import pytest
import torch

from priml.baselines.imagenet.model import (
    BlurPoolConv2d,
    TorchvisionResNet,
    apply_blurpool,
)
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from collections.abc import Callable


_CWD: Final = Path(__file__).resolve().parent


def strided_net(num_classes: int) -> nn.Module:
    """Return a conv-BN net whose second conv qualifies for BlurPool.

    torchvision fixes ResNet stage widths at 64-512, so its smallest ResNet
    is ~3M parameters -- a 12 MB golden. This pins the wrapper and the blur
    rewrite; torchvision's own ResNet is checked by the ffcv parity script.
    Named ``bn`` like torchvision's norms, so the recipe's optimizer split
    has a BatchNorm group to route.
    """
    return nn.Sequential(
        OrderedDict(
            conv1=nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            bn1=nn.BatchNorm2d(16),
            relu=nn.ReLU(),
            conv2=nn.Conv2d(16, 8, 3, stride=2, padding=1, bias=False),
            pool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(),
            fc=nn.Linear(8, num_classes),
        ),
    )


def tiny_resnet() -> TorchvisionResNet.Config:
    """Return the wrapper around ``strided_net`` over 10 classes."""
    config = TorchvisionResNet.Config()
    config.arch = PartialConfig(strided_net)
    config.channels_out = 10
    return config


def test_blurpool_wraps_only_strided_convolutions_with_16_or_more_inputs() -> None:
    root = nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2),  # 3 inputs: the stem stays sharp.
        nn.Sequential(nn.Conv2d(16, 16, 3, stride=2)),  # Nested, strided: wrapped.
        nn.Conv2d(16, 16, 3, stride=1),  # Unstrided: left alone.
    )
    apply_blurpool(root)
    nested = root[1]
    assert isinstance(nested, nn.Sequential)
    assert isinstance(root[0], nn.Conv2d)
    assert isinstance(nested[0], BlurPoolConv2d)
    assert isinstance(root[2], nn.Conv2d)


def test_blurpool_filter_is_the_normalized_binomial_kernel() -> None:
    blur = BlurPoolConv2d(nn.Conv2d(4, 4, 3, stride=2))
    assert blur.blur_filter.shape == (4, 1, 3, 3)
    assert torch.allclose(blur.blur_filter.sum(dim=(-2, -1)), torch.ones(4, 1))
    x = torch.ones(1, 4, 6, 6)
    # A constant image passes the blur unchanged away from the zero-padded border.
    assert torch.equal(
        nn.functional.conv2d(x, blur.blur_filter, padding=1, groups=4)[..., 1:-1, 1:-1],
        x[..., 1:-1, 1:-1],
    )


def test_resnet50_blurpools_the_same_convolutions_ffcv_imagenet_does() -> None:
    with torch.device("meta"):
        model = TorchvisionResNet.Config().make()
    blurred = [n for n, m in model.named_modules() if isinstance(m, BlurPoolConv2d)]
    # Three stride-2 3x3 convs and three stride-2 downsample 1x1s.
    assert len(blurred) == 6


def test_forward_maps_images_to_logits() -> None:
    model = tiny_resnet().make()
    assert model.forward(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert isinstance(model.net.get_submodule("conv2"), BlurPoolConv2d)
    assert isinstance(model.net.get_submodule("conv1"), nn.Conv2d)


@pytest.mark.parametrize(
    "arch",
    [strided_net, resnet18, resnet50],
    ids=["strided_net", "resnet18", "resnet50"],
)
def test_cost_matches_what_torch_dispatches(arch: Callable[..., nn.Module]) -> None:
    """Every conv, blur, and head matmul is costed, and every parameter owned."""
    config = TorchvisionResNet.Config()
    config.channels_out = 10
    config.image_size = (32, 32)
    config.arch = PartialConfig(arch)
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 32, 32, requires_grad=True),
        batch_size=2,
        dtype=None,
    )
    assert analytical["flops", "primal", "elementwise"].sum() > 0


def test_cost_refuses_a_layer_it_cannot_price() -> None:
    config = tiny_resnet()
    config.arch = PartialConfig(_with_gelu)
    with pytest.raises(TypeError, match="GELU"):
        _ = config.cost(batch_size=1, dtype=None)


def _with_gelu(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3),
        nn.GELU(),
        nn.Flatten(),
        nn.LazyLinear(num_classes),
    )


def test_resnet_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="blurpool_net",
        build_module=lambda: tiny_resnet().make().eval(),
        build_input=lambda: torch.randn(1, 3, 16, 16),
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
