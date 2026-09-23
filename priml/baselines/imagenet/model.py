"""ResNet-50 as ffcv-imagenet trains it: torchvision's module, BlurPool'd.

The reference builds ``torchvision.models.resnet50`` and wraps every strided
convolution with at least 16 input channels in a fixed 3x3 binomial blur.
``exp000`` reproduces that module bit-for-bit, so the model is torchvision's
rather than a priml rewrite; a fork that wants priml's ``ResNet`` supplies it
in the same slot.

References:
  https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py
  https://arxiv.org/abs/1904.11486
    Zhang 2019. Making convolutional networks shift-invariant again.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, cast, override

import dataclasses

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, elementwise_cost, matmul_cost
from priml.model.conv import conv_cost
from priml.model.norm import BatchNorm2d
from priml.model.pool import avg_pool_cost, max_pool_cost
from priml.model.swiglu import relu


if TYPE_CHECKING:
    import torchvision.models
    import torchvision.models.resnet
else:
    from wrapt import lazy_import

    torchvision = lazy_import("torchvision")  # ~400 ms; only the builder needs it.


class BlurPoolConv2d(nn.Module):
    """A convolution preceded by a fixed 3x3 binomial blur, per input channel."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        kernel = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        self.conv = conv
        self.blur_filter: Tensor
        self.register_buffer("blur_filter", kernel.repeat(conv.in_channels, 1, 1, 1))

    @override
    def forward(self, x: Tensor) -> Tensor:
        blurred = functional.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


def apply_blurpool(module: nn.Module) -> None:
    """Wrap every strided convolution with >= 16 input channels in a blur.

    ffcv-imagenet's rule, verbatim; a fork changing it changes the model.

    Args:
      module: Module rewritten in place.

    """
    for name, child in module.named_children():
        if (
            isinstance(child, nn.Conv2d)
            and max(child.stride) > 1
            and child.in_channels >= 16
        ):
            setattr(module, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


class TorchvisionResNet(nn.Module):
    """A torchvision ResNet, optionally BlurPool'd, called with ``media``."""

    class Config(Fig["TorchvisionResNet"]):
        arch: Makeable[Callable[..., nn.Module]] = field(
            default_factory=lambda: PartialConfig(torchvision.models.resnet50),
        )
        """torchvision builder, called with ``num_classes``."""

        channels_out: int = 1_000
        """Classes; torchvision's ``num_classes``."""

        use_blurpool: bool = True
        """Blur before every strided convolution, as in ffcv-imagenet."""

        image_size: tuple[int, int] = (160, 160)
        """``(height, width)`` the cost is taken at; ffcv's first-epoch side."""

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost every layer the forward runs, at the shape it runs at.

            torchvision owns the architecture, so the layers are read off a
            meta-device build by one hooked forward rather than restated. A
            layer this cannot price is a ``TypeError``, never zero.

            Args:
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; nothing here reads it.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            del kwargs
            with torch.device("meta"):
                net = TorchvisionResNet(self)
            costs: list[Cost] = []

            def record(
                module: nn.Module,
                args: tuple[object, ...],
                out: object,
            ) -> None:
                (x,) = args
                assert isinstance(x, Tensor)
                assert isinstance(out, Tensor)
                costs.append(_layer_cost(module, x, out, dtype=dtype))

            handles = [
                module.register_forward_hook(record)
                for module in net.modules()
                if _is_costed_layer(module)
            ]
            try:
                with torch.device("meta"):
                    _ = net(torch.empty(batch_size, 3, *self.image_size))
            finally:
                for handle in handles:
                    handle.remove()
            return sum(costs, Cost())

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.net = config.arch.make()(num_classes=config.channels_out)
        if config.use_blurpool:
            apply_blurpool(self.net)

    @override
    def forward(self, media: Tensor) -> Tensor:
        return cast(Tensor, self.net(media))


def _is_costed_layer(module: nn.Module) -> bool:
    """Leaves, plus the containers that do work of their own (blur, residual add)."""
    residual = (
        torchvision.models.resnet.BasicBlock,
        torchvision.models.resnet.Bottleneck,
    )
    return isinstance(module, (BlurPoolConv2d, *residual)) or not any(
        True for _ in module.children()
    )


def _layer_cost(
    module: nn.Module,
    x: Tensor,
    out: Tensor,
    *,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost one hooked layer at the shapes its forward saw."""
    batch_size, channels = x.shape[0], x.shape[1]
    grid = (x.shape[2], x.shape[3]) if x.ndim == 4 else (1, 1)
    if isinstance(module, BlurPoolConv2d):
        # ``forward`` calls ``conv.forward`` directly, so the inner conv fires
        # no hook and is costed here. The blur is a buffer: it owns no
        # parameters and takes no weight gradient, but passes an input one.
        blur = conv_cost(
            channels_in=channels,
            channels_out=channels,
            kernel_size=3,
            ndim=2,
            groups=channels,
            bias=False,
            input_grid=grid,
            batch_size=batch_size,
            padding=1,
            dtype=dtype,
            weight_grad=False,
        )
        return dataclasses.replace(blur, params=0, params_active=0) + _layer_cost(
            module.conv,
            x,
            out,
            dtype=dtype,
        )
    if isinstance(module, nn.Conv2d):
        return conv_cost(
            channels_in=module.in_channels,
            channels_out=module.out_channels,
            kernel_size=module.kernel_size,
            ndim=2,
            groups=module.groups,
            bias=module.bias is not None,
            input_grid=grid,
            batch_size=batch_size,
            stride=module.stride,
            padding=cast(tuple[int, int], module.padding),
            dilation=module.dilation,
            dtype=dtype,
        )
    if isinstance(module, nn.BatchNorm2d):
        return cost(
            BatchNorm2d.Config(channels, elementwise_affine=module.affine),
            seq_len=grid[0] * grid[1],
            batch_size=batch_size,
            dtype=dtype,
        )
    if isinstance(module, nn.ReLU):
        return cost(relu, channels=x.numel(), dtype=dtype)
    if isinstance(module, nn.MaxPool2d):
        return max_pool_cost(
            channels=channels,
            kernel_size=cast(int, module.kernel_size),
            rows=out.numel() // channels,
            dtype=dtype,
        )
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return avg_pool_cost(
            channels=channels,
            positions=grid[0] * grid[1],
            batch_size=batch_size,
            dtype=dtype,
        )
    if isinstance(module, nn.Linear):
        return matmul_cost(
            channels_in=module.in_features,
            channels_out=module.out_features,
            bias=module.bias is not None,
            rows=x.numel() // module.in_features,
            dtype=dtype,
        )
    if isinstance(module, nn.Flatten):
        return Cost()
    if isinstance(
        module,
        (torchvision.models.resnet.BasicBlock, torchvision.models.resnet.Bottleneck),
    ):
        # The block's own work is the residual add; its layers hook themselves.
        return elementwise_cost(
            primal=out.numel(),
            adjoint=0,
            channels=out.shape[1],
            inputs=2,
            adjoint_inputs=0,
            adjoint_outputs=0,
            rows=out.numel() // out.shape[1],
            dtype=dtype,
        )
    raise TypeError(f"{type(module).__qualname__} has no cost.")
