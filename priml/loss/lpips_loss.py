"""LPIPS perceptual loss for videos."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import functools
import math
import warnings

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    elementwise_cost,
    traffic,
)
from priml.model.conv import conv_cost
from priml.model.pool import max_pool_cost


if TYPE_CHECKING:
    import lpips

    from priml.loss.custom_types import LossOutput
else:
    from wrapt import lazy_import

    lpips = lazy_import("lpips")


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss for videos."""

    class Config(Fig["LPIPSLoss"]):
        max_num_random_frames: int = 2
        """Maximum number of random frames to subsample per video."""

        net: str = "vgg"
        """Backbone network for LPIPS ("vgg", "alex", "squeeze")."""

        image_size: tuple[int, int] = (-1, -1)
        """``(height, width)`` of each frame. The data fixes it, so the
        experiment sets it; ``cost`` refuses the sentinel."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost both branches through the frozen trunk and the linear head.

            A complete invocation scores ``batch_size * seq_len`` frames, each
            with ``height * width`` positions. ``seq_len`` is the number of
            sampled frames per video, bounded at runtime by
            ``min(max_num_random_frames, T)``; ``batch_size`` is the number of
            videos. The returned ledger contains the complete FLOP and logical
            byte totals for both branches, the trainable heads, and frame
            selection. Shared trunk parameters are counted once even though
            the frozen trunk executes in both branches.

            The trunk is built with random weights on the meta device, so
            costing downloads nothing and allocates nothing. Its layers are
            traced through one forward at ``image_size`` and each is costed at
            its own output grid: a convolution as the matmul over its receptive
            field, a max pool as ``k * k - 1`` compares per pooled element with
            one gradient routed back to the argmax, and a ReLU as one operation
            per element each way. Every trunk parameter is frozen but owned:
            it is in ``params``, and its adjoint is the input gradient alone --
            one convolution of the primal's size, not the two a trainable layer
            pays. The trunk, the ``ScalingLayer`` (shift and scale per input
            element), and the per-stage channel normalization run once per
            branch, ``x`` and ``xhat``. Per stage, the squared feature
            difference, trainable 1x1 ``NetLinLayer`` convolution, and spatial
            mean run once; stage and frame reductions use their complete
            concrete tensors. ``NetLinLayer``'s dropout is uncosted:
            ``lpips.LPIPS`` puts itself in eval mode at construction, where it
            is the identity.

            Each branch gathers a selected RGB frame: read/write its pixels
            and read the shared frame index once. Both input adjoints scatter
            to those selected pixels, reading the gradient and destination
            before writing it. Unique frame indices need no collision adds.
            Permutation generation and unsampled gradient initialization are
            excluded: their original frame count is not on this scored-frame
            bus. No sort is assumed for ``randperm`` and no layout-copy is
            guessed.

            Args:
              seq_len: Sampled frames per video.
              batch_size: Videos per invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Complete invocation cost; ``bytes_state`` is zero.

            Raises:
              ValueError: ``image_size`` was never set.

            """
            if min(self.image_size) < 1:
                raise ValueError(
                    "LPIPSLoss.cost needs image_size; the data fixes it, so set it "
                    f"on the config. Got {self.image_size}.",
                )
            del kwargs
            image_size = self.image_size
            dt = dtype
            positions = math.prod(image_size)
            frames = batch_size * seq_len
            with torch.device("meta"):
                model = _lpips(self.net, pretrained=False)
            traced: list[tuple[nn.Module, tuple[int, ...], tuple[int, ...]]] = []
            for module in model.net.modules():
                if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.ReLU)):
                    module.register_forward_hook(
                        functools.partial(_record_output_shape, traced),
                    )
            # The trunk returns a per-net namedtuple of stage outputs; the stub
            # types ``net`` as a bare module, so its result is narrowed here.
            stages = cast(
                tuple[object, ...],
                model.net(torch.empty(1, 3, *image_size, device="meta")),
            )

            branch = (
                traffic(
                    "primal",
                    "elementwise",
                    elements=2 * 3 * 3,
                    flops=2 * 3,
                    dtype=dt,
                )
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=3 * 3,
                    flops=3,
                    dtype=dt,
                )
            ).tile(frames * positions)
            for module, input_shape, shape in traced:
                channels, grid = shape[1], math.prod(shape[2:])
                output_rows = frames * grid
                if isinstance(module, nn.Conv2d):
                    costed = _conv2d_cost(
                        module,
                        input_grid=input_shape[2:],
                        batch_size=frames,
                        dtype=dt,
                    )
                elif isinstance(module, nn.MaxPool2d):
                    costed = max_pool_cost(
                        channels=channels,
                        kernel_size=module.kernel_size,
                        rows=output_rows,
                        dtype=dt,
                    )
                else:
                    costed = elementwise_cost(
                        primal=channels,
                        adjoint=channels,
                        channels=channels,
                        dtype=dt,
                    ).tile(output_rows)
                branch += costed

            head = (
                traffic(
                    "primal",
                    "elementwise",
                    elements=3 * model.L,
                    flops=model.L + 1,
                    dtype=dt,
                )
                + traffic("primal", "reduction", elements=2, dtype=dt)
                + traffic("adjoint", "elementwise", elements=2, flops=1, dtype=dt)
                + traffic("adjoint", "reduction", elements=2, dtype=dt)
            ).tile(frames)
            heads = model.lins  # codespell:ignore lins
            for lin, out in zip(heads, stages, strict=True):
                assert isinstance(out, Tensor)
                channels, grid = out.shape[1], math.prod(out.shape[2:])
                branch += _normalize_cost(channels, dtype=dt).tile(frames * grid)
                conv = next(m for m in lin.modules() if isinstance(m, nn.Conv2d))
                head += (
                    traffic(
                        "primal",
                        "elementwise",
                        elements=5 * channels,
                        flops=2 * channels,
                        dtype=dt,
                    )
                    + traffic(
                        "adjoint",
                        "elementwise",
                        elements=7 * channels,
                        flops=3 * channels,
                        dtype=dt,
                    )
                ).tile(frames)
                head += _conv2d_cost(
                    conv,
                    input_grid=out.shape[2:],
                    batch_size=frames,
                    dtype=dt,
                )
                head += _spatial_average_cost(grid, dtype=dt).tile(frames)
            # Both branches gather a frame's pixels and read the shared frame
            # index, one ``int64`` each; the adjoints scatter to those pixels.
            selected = (
                traffic(
                    "primal",
                    "selection",
                    elements=2 * 3 * 2 * frames * positions,
                    dtype=dt,
                )
                + traffic("primal", "selection", elements=2, dtype=torch.int64)
                + traffic(
                    "adjoint",
                    "selection",
                    elements=2 * 3 * 3 * frames * positions,
                    dtype=dt,
                )
                + traffic("adjoint", "selection", elements=2, dtype=torch.int64)
            )
            return branch.tile(2, copies=1) + head + selected

    def __init__(self, config: Config):
        super().__init__()
        self.max_num_random_frames = config.max_num_random_frames
        self.lpips_criterion = _lpips(config.net, pretrained=True)

    @override
    def forward(
        self,
        model_output: Tensor,
        *,
        x: Tensor,
        xhat: Tensor,
        **batch: object,
    ) -> LossOutput:
        """Compute LPIPS on subsampled frames (pointwise).

        Args:
          model_output: Model output (ignored, use xhat instead).
          x: [B, C, T, H, W] video tensor.
          xhat: [B, C, T, H, W] reconstructed video tensor.
          **batch: Additional batch keys (ignored).

        Returns:
          loss: Dict with pointwise loss [B], one entry per input sample.

        """
        del model_output, batch
        if x.ndim != 5:
            raise ValueError("Expected 5D tensor [B, C, T, H, W]")

        # Subsample unique random frames (at most max_num_random_frames or total frames)
        num_frames = x.shape[-3]
        num_sample_frames = min(self.max_num_random_frames, num_frames)
        rand_indices = torch.randperm(num_frames, device=x.device)[:num_sample_frames]

        x_sub = x[:, :, rand_indices, :, :]
        xhat_sub = xhat[:, :, rand_indices, :, :]

        # Reshape to [B*T, C, H, W] for LPIPS.
        b, c, t, h, w = x_sub.shape
        x_sub = x_sub.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        xhat_sub = xhat_sub.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        # LPIPS returns [B*T, 1, 1, 1].
        loss = self.lpips_criterion(x_sub, xhat_sub)

        # Reshape back to [B, T] and mean over frames → [B].
        loss = loss.reshape(b, t).mean(dim=1)

        return {"loss": loss}


def _lpips(net: str, *, pretrained: bool) -> lpips.LPIPS:
    """Build ``lpips.LPIPS``; ``pretrained=False`` gives a random trunk and head."""
    with warnings.catch_warnings():
        # `lpips` reaches torchvision through its legacy ``pretrained=`` argument,
        # which torchvision deprecation-warns on every construction. The call is
        # lpips's, not ours, so the filter sits at the one site that makes it.
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"torchvision\.models\._utils",
        )
        return lpips.LPIPS(
            net=net,
            pretrained=pretrained,
            pnet_rand=not pretrained,
            verbose=False,
        )


def _record_output_shape(
    traced: list[tuple[nn.Module, tuple[int, ...], tuple[int, ...]]],
    module: nn.Module,
    args: tuple[object, ...],
    output: object,
) -> None:
    """Record both operand grids; strided convolutions change their sizes."""
    input_tensor = args[0]
    assert isinstance(input_tensor, Tensor)
    assert isinstance(output, Tensor)
    traced.append((module, tuple(input_tensor.shape), tuple(output.shape)))


def _conv2d_cost(
    module: nn.Conv2d,
    *,
    input_grid: tuple[int, ...],
    batch_size: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost a convolution over its measured input grid and module geometry."""
    return conv_cost(
        channels_in=module.in_channels,
        channels_out=module.out_channels,
        kernel_size=module.kernel_size,
        ndim=2,
        groups=module.groups,
        bias=module.bias is not None,
        input_grid=input_grid,
        batch_size=batch_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        dtype=dtype,
        weight_grad=module.weight.requires_grad,
        bias_grad=module.bias is not None and module.bias.requires_grad,
    )


# ``y = x / (||x|| + eps)``: square, sum, sqrt, add, divide forward. Back,
# ``g / d - x (g . x) / (d^2 ||x||)``: one dot product, three scalar ops, then
# a divide, a multiply, and a subtract per channel.
def _normalize_cost(channels: int, *, dtype: torch.dtype | None) -> Cost:
    """Cost ``lpips.normalize_tensor`` at one position over ``channels``."""
    c = channels
    return (
        traffic(
            "primal",
            "elementwise",
            elements=4 * c + 5,
            flops=2 * c + 2,
            dtype=dtype,
        )
        + traffic("primal", "reduction", elements=c + 1, flops=c - 1, dtype=dtype)
        + traffic(
            "adjoint",
            "elementwise",
            elements=10 * c + 9,
            flops=4 * c + 3,
            dtype=dtype,
        )
        + traffic("adjoint", "reduction", elements=c + 1, flops=c - 1, dtype=dtype)
    )


def _spatial_average_cost(positions: int, *, dtype: torch.dtype | None) -> Cost:
    """Cost the mean of one channel over ``positions``: a sum, a scale, a broadcast back."""
    n = positions
    return (
        traffic("primal", "reduction", elements=n + 1, flops=n - 1, dtype=dtype)
        + traffic("primal", "elementwise", elements=2, flops=1, dtype=dtype)
        + traffic("adjoint", "elementwise", elements=n + 1, flops=n, dtype=dtype)
    )
