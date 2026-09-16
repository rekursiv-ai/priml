"""LPIPS perceptual loss for videos."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast, override

import functools
import math
import warnings

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.loss.custom_types import LossOutput
from priml.model.conv import conv_cost
from priml.model.cost import Bytes, Compute, Cost, Flops, elementwise_cost


if TYPE_CHECKING:
    import lpips
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

        def cost(
            self,
            *,
            image_size: tuple[int, int],
            num_tokens: int = 1,
            **kwargs: object,
        ) -> Cost:
            """Price both branches through the frozen trunk and the linear head.

            A token is one ``(h, w)`` position of one SCORED frame of one
            video, so ``num_tokens`` is ``batch * frames_scored * height *
            width``. ``forward`` scores ``min(max_num_random_frames, T)`` of
            each video's ``T`` frames, so a caller holding ``T`` frames of
            which ``k`` are scored spreads the cost over ``k / T`` of its
            frame positions; the unscored frames cost nothing.

            The trunk is built with random weights on the meta device, so
            pricing downloads nothing and allocates nothing. Its layers are
            traced through one forward at ``image_size`` and each is priced at
            its own output grid: a convolution as the matmul over its receptive
            field, a max pool as ``k * k - 1`` compares per pooled element with
            one gradient routed back to the argmax, a ReLU as one operation per
            element each way. Every trunk parameter is frozen but owned: it is
            in ``params``, and its adjoint is the input gradient alone -- one
            convolution of the primal's size, not the two a trainable layer
            pays. The trunk, the ``ScalingLayer`` (shift and scale per input
            element), and the per-stage channel normalization run once per
            branch, ``x`` and ``xhat``, so their work is doubled and their
            parameters counted once. Per stage, the squared feature difference,
            the trainable 1x1 ``NetLinLayer`` convolution, and the spatial mean
            run once; the stage sum and the mean over frames are one add each
            per frame. Work at every grid is spread over the image's positions in
            one division. ``NetLinLayer``'s dropout is unpriced: ``lpips.LPIPS``
            puts itself in eval mode at construction, where it is the identity.

            Args:
              image_size: ``(height, width)`` of one frame; the token grid.
              num_tokens: Frame positions sharing each parameter, batch included.
              **kwargs: The rest of the bus, unread.

            Returns:
              cost: Per-frame-position cost of this loss; ``bytes_state`` is zero.

            """
            del kwargs
            positions = math.prod(image_size)
            with torch.device("meta"):
                model = _lpips(self.net, pretrained=False)
            traced: list[tuple[nn.Module, tuple[int, ...]]] = []
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

            branch = _repeat(elementwise_cost(primal=2 * 3, adjoint=3), positions)
            for module, shape in traced:
                channels, grid = shape[1], math.prod(shape[2:])
                if isinstance(module, nn.Conv2d):
                    priced = _conv2d_cost(
                        module,
                        num_tokens=_rows(num_tokens, positions=positions, grid=grid),
                    )
                elif isinstance(module, nn.MaxPool2d):
                    priced = _max_pool_cost(channels, kernel_size=module.kernel_size)
                else:
                    priced = elementwise_cost(primal=channels, adjoint=channels)
                branch += _repeat(priced, grid)

            head = elementwise_cost(primal=model.L + 1, adjoint=1)
            heads = model.lins  # codespell:ignore lins
            for lin, out in zip(heads, stages, strict=True):
                assert isinstance(out, Tensor)
                channels, grid = out.shape[1], math.prod(out.shape[2:])
                branch += _repeat(_normalize_cost(channels), grid)
                conv = next(m for m in lin.modules() if isinstance(m, nn.Conv2d))
                head += _repeat(
                    elementwise_cost(
                        primal=2 * channels,
                        adjoint=3 * channels,
                        channels=channels,
                    )
                    + _conv2d_cost(
                        conv,
                        num_tokens=_rows(num_tokens, positions=positions, grid=grid),
                    ),
                    grid,
                )
                head += _spatial_average_cost(grid)
            return _per_position(_repeat(branch, 2) + head, positions)

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
        assert x.ndim == 5, "Expected 5D tensor [B, C, T, H, W]"

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
    traced: list[tuple[nn.Module, tuple[int, ...]]],
    module: nn.Module,
    args: tuple[object, ...],
    output: object,
) -> None:
    """Forward hook: append the module and its output shape in execution order."""
    del args
    assert isinstance(output, Tensor)
    traced.append((module, tuple(output.shape)))


def _conv2d_cost(module: nn.Conv2d, *, num_tokens: int) -> Cost:
    """Price one output position of ``module``; a frozen weight pays no weight gradient."""
    priced = conv_cost(
        channels_in=module.in_channels,
        channels_out=module.out_channels,
        kernel_size=module.kernel_size,
        ndim=2,
        groups=module.groups,
        bias=module.bias is not None,
        num_tokens=num_tokens,
    )
    if module.weight.requires_grad:
        return priced
    return replace(
        priced,
        adjoint=replace(
            priced.adjoint,
            flops=Flops(matmul=priced.primal.flops.matmul),
        ),
    )


def _max_pool_cost(channels: int, *, kernel_size: int | tuple[int, ...]) -> Cost:
    """Price one pooled position: compares forward, one gradient routed to the argmax."""
    taps = math.prod(
        kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2,
    )
    return Cost(
        primal=Compute(
            flops=Flops(reduction=channels * (taps - 1)),
            bytes=Bytes(reduction=channels),
        ),
        adjoint=Compute(
            flops=Flops(selection=channels),
            bytes=Bytes(selection=channels),
        ),
    )


# ``y = x / (||x|| + eps)``: square, sum, sqrt, add, divide forward. Back,
# ``g / d - x (g . x) / (d^2 ||x||)``: one dot product, three scalar ops, then
# a divide, a multiply, and a subtract per channel.
def _normalize_cost(channels: int) -> Cost:
    """Price ``lpips.normalize_tensor`` at one position over ``channels``."""
    return elementwise_cost(
        primal=2 * channels + 2,
        adjoint=4 * channels + 3,
        channels=channels,
    ) + Cost(
        primal=Compute(flops=Flops(reduction=channels - 1)),
        adjoint=Compute(flops=Flops(reduction=channels - 1)),
    )


def _spatial_average_cost(positions: int) -> Cost:
    """Price the mean of one channel over ``positions``: a sum, a scale, a broadcast back."""
    return Cost(
        primal=Compute(
            flops=Flops(reduction=positions - 1, elementwise=1),
            bytes=Bytes(reduction=1),
        ),
        adjoint=Compute(
            flops=Flops(elementwise=positions),
            bytes=Bytes(elementwise=positions),
        ),
    )


# Rounded up, so the one-row shape estimate stays one row through a downsampling
# layer instead of dividing by zero.
def _rows(num_tokens: int, *, positions: int, grid: int) -> int:
    """Return the rows a layer on ``grid`` sees when a frame holds ``num_tokens``."""
    return (num_tokens * grid + positions - 1) // positions


def _repeat(priced: Cost, rows: int) -> Cost:
    """Run ``rows`` times; parameters stay owned once."""
    return replace(priced, primal=priced.primal * rows, adjoint=priced.adjoint * rows)


# One division by the frame's positions, never a chain of per-stage ratios: a
# ``49 / 961`` followed by ``961 / 1024`` lands an ulp off ``49 / 1024``, and the
# torch comparison is exact.
def _per_position(priced: Cost, positions: int) -> Cost:
    """Spread work done once per frame over the frame's positions."""
    return replace(
        priced,
        primal=Compute(
            flops=priced.primal.flops / positions,
            bytes=priced.primal.bytes / positions,
        ),
        adjoint=Compute(
            flops=priced.adjoint.flops / positions,
            bytes=priced.adjoint.bytes / positions,
        ),
    )
