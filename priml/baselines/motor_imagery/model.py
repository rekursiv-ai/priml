"""EEG classifiers composed from Priml convolution, normalization, and linear layers."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, Final, Self, cast, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.baselines.motor_imagery.custom_types import (
    ChannelsHidden,
    Groups,
    KernelSize,
    SpatialConfig,
)
from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    traffic,
)
from priml.math.custom_types import TensorFn
from priml.model.conv import Conv2d
from priml.model.custom_types import (
    ActivationFn,
    ChannelsIn,
    ChannelsOut,
    TensorModule,
    propagate_attr,
)
from priml.model.legacy_keys import absorb_legacy_keys
from priml.model.linear import Linear
from priml.model.norm import BatchNorm2d
from priml.model.pool import AvgPool2d
from priml.model.special import Dropout
from priml.model.squeeze_excitation import SqueezeExcitation
from priml.model.swiglu import elu


if TYPE_CHECKING:
    from collections.abc import Sequence


SOURCE_NAMES: Final = {
    "bn_t": "norm_t",
    "bn_s": "norm_s",
    "bn_sep": "norm_sep",
    "bn": "norm",
    "classifier": "proj_out",
}
"""Source attribute names and the blessed nouns that replaced them."""


class MultiScaleEEGNet(nn.Module):
    """Concatenate three temporal scales before depthwise spatial filtering."""

    class Config(Fig["MultiScaleEEGNet"], kw_only=False):
        channels_in: int = -1
        """Recording electrodes; upstream n_channels."""

        channels_out: int = -1
        """Class logits; upstream n_classes."""

        _: KW_ONLY

        num_samples: int = -1
        """Temporal input width; upstream n_samples."""

        channels_hidden: int = 8
        """Filters per temporal scale; upstream F1."""

        multiplier: int = 2
        """Depthwise spatial multiplier; upstream D."""

        channels_projection: int = 16
        """Half the separable projection width; upstream F2."""

        activation: ActivationFn = elu
        """Nonlinearity after the spatial and separable normalizations."""

        conv_t1: SpatialConfig = field(
            default_factory=lambda: Conv2d.Config(kernel_size=(1, 63), padding=(0, 31)),
        )
        """Longest temporal filter."""

        conv_t2: SpatialConfig = field(
            default_factory=lambda: Conv2d.Config(kernel_size=(1, 31), padding=(0, 15)),
        )
        """Middle temporal filter."""

        conv_t3: SpatialConfig = field(
            default_factory=lambda: Conv2d.Config(kernel_size=(1, 15), padding=(0, 7)),
        )
        """Shortest temporal filter."""

        norm_t: Makeable[TensorModule] = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Normalization of concatenated temporal channels."""

        conv_s: SpatialConfig = field(default_factory=lambda: Conv2d.Config(padding=0))
        """Grouped electrode projection."""

        norm_s: Makeable[TensorModule] = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Spatial feature normalization."""

        conv_sep_depth: SpatialConfig = field(
            default_factory=lambda: Conv2d.Config(kernel_size=(1, 15), padding=(0, 7)),
        )
        """Depthwise temporal refinement."""

        conv_sep_point: SpatialConfig = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0),
        )
        """Pointwise feature mixing."""

        norm_sep: Makeable[TensorModule] = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Mixed feature normalization."""

        pool1: SpatialConfig = field(
            default_factory=lambda: AvgPool2d.Config(kernel_size=(1, 4)),
        )
        """First temporal downsampling."""

        pool2: SpatialConfig = field(
            default_factory=lambda: AvgPool2d.Config(kernel_size=(1, 8)),
        )
        """Second temporal downsampling."""

        drop1: Makeable[TensorFn] = field(
            default_factory=lambda: Dropout.Config(p=0.25),
        )
        """First feature dropout."""

        drop2: Makeable[TensorFn] = field(
            default_factory=lambda: Dropout.Config(p=0.25),
        )
        """Second feature dropout."""

        proj_out: Makeable[TensorModule] = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Flattened feature projection to class logits."""

        @override
        def finalize(self) -> Self:
            temporal = 3 * self.channels_hidden
            spatial = temporal * self.multiplier
            projected = 2 * self.channels_projection
            for conv in (self.conv_t1, self.conv_t2, self.conv_t3):
                _size(conv, channels_in=1, channels_out=self.channels_hidden)
            _size(self.norm_t, channels_in=temporal)
            _size(self.conv_s, channels_in=temporal, channels_out=spatial)
            propagate_attr(self.conv_s, "groups", temporal, protocol=Groups)
            propagate_attr(
                self.conv_s,
                "kernel_size",
                (self.channels_in, 1),
                protocol=KernelSize,
            )
            _size(self.norm_s, channels_in=spatial)
            _size(self.conv_sep_depth, channels_in=spatial)
            propagate_attr(self.conv_sep_depth, "groups", spatial, protocol=Groups)
            _size(self.conv_sep_point, channels_in=spatial, channels_out=projected)
            _size(self.norm_sep, channels_in=projected)
            _size(
                self.proj_out,
                channels_in=projected
                * math.prod(
                    _feature_grid(
                        (self.channels_in, self.num_samples),
                        operations=[
                            self.conv_t1,
                            self.conv_s,
                            self.pool1,
                            self.conv_sep_depth,
                            self.conv_sep_point,
                            self.pool2,
                        ],
                    ),
                ),
                channels_out=self.channels_out,
            )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost all three branches, spatial filters, pools, and classifier.

            Args:
              seq_len: Temporal width, matching the configured classifier geometry.
              batch_size: Windows in the complete invocation.
              dtype: Activation and parameter dtype.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Forward and adjoint work, traffic, and owned parameters.

            """
            if seq_len != self.num_samples:
                raise ValueError("Cost sequence length must match num_samples.")
            temporal = 3 * self.channels_hidden
            spatial = temporal * self.multiplier
            projected = 2 * self.channels_projection
            grid = (self.channels_in, self.num_samples)
            result = Cost()
            for conv in (self.conv_t1, self.conv_t2, self.conv_t3):
                result += cost(
                    conv,
                    input_grid=grid,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            grid = self.conv_t1.output_grid(grid)
            result += traffic(
                "primal",
                "elementwise",
                elements=2 * batch_size * temporal * math.prod(grid),
                dtype=dtype,
            )
            result += cost(
                self.norm_t,
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            part, grid = _conv_cost(
                self.conv_s,
                grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            result += part + _norm_activation_cost(
                self.norm_s,
                self.activation,
                channels=spatial,
                positions=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
            )
            part, grid = _pool_dropout_cost(
                self.pool1,
                self.drop1,
                channels=spatial,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            result += part
            for conv in (self.conv_sep_depth, self.conv_sep_point):
                part, grid = _conv_cost(conv, grid, batch_size=batch_size, dtype=dtype)
                result += part
            result += _norm_activation_cost(
                self.norm_sep,
                self.activation,
                channels=projected,
                positions=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
            )
            part, grid = _pool_dropout_cost(
                self.pool2,
                self.drop2,
                channels=projected,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            return (
                result
                + part
                + cost(
                    self.proj_out,
                    seq_len=1,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        absorb_legacy_keys(self, SOURCE_NAMES)
        self.activation = _activation(config.activation)
        self.conv_t1 = config.conv_t1.make()
        self.conv_t2 = config.conv_t2.make()
        self.conv_t3 = config.conv_t3.make()
        self.norm_t = config.norm_t.make()
        self.conv_s = config.conv_s.make()
        self.norm_s = config.norm_s.make()
        self.pool1 = config.pool1.make()
        self.drop1 = config.drop1.make()
        self.conv_sep_depth = config.conv_sep_depth.make()
        self.conv_sep_point = config.conv_sep_point.make()
        self.norm_sep = config.norm_sep.make()
        self.pool2 = config.pool2.make()
        self.drop2 = config.drop2.make()
        # Warm running statistics and advance dropout before the classifier draw.
        with torch.no_grad():
            self._features(torch.zeros(1, 1, config.channels_in, config.num_samples))
        self.proj_out = config.proj_out.make()

    @override
    def forward(self, media: Tensor) -> Tensor:
        """Return the source's two class logits."""
        features = self._features(media)
        return self.proj_out(features.view(features.size(0), -1))

    def _features(self, media: Tensor) -> Tensor:
        x = torch.cat(
            [self.conv_t1(media), self.conv_t2(media), self.conv_t3(media)],
            dim=1,
        )
        x = self.norm_t(x)
        x = self.drop1(self.pool1(self.activation(self.norm_s(self.conv_s(x)))))
        x = self.conv_sep_point(self.conv_sep_depth(x))
        return self.drop2(self.pool2(self.activation(self.norm_sep(x))))


class ShallowConvNet(nn.Module):
    """Model spatially filtered temporal energy using square, pool, and log."""

    class Config(Fig["ShallowConvNet"], kw_only=False):
        channels_in: int = -1
        """Recording electrodes; upstream n_channels."""

        channels_out: int = -1
        """Class logits; upstream n_classes."""

        _: KW_ONLY

        num_samples: int = -1
        """Temporal input width; upstream n_samples."""

        channels_hidden: int = 40
        """Temporal and spatial filter count; upstream F."""

        eps: float = 1e-6
        """Floor on the pooled energy before its logarithm."""

        conv_t: SpatialConfig = field(
            default_factory=lambda: Conv2d.Config(kernel_size=(1, 25), padding=0),
        )
        """Valid temporal filter."""

        conv_s: SpatialConfig = field(default_factory=lambda: Conv2d.Config(padding=0))
        """Full electrode projection."""

        norm: Makeable[TensorModule] = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Spatial energy normalization."""

        pool: SpatialConfig = field(
            default_factory=lambda: AvgPool2d.Config(
                kernel_size=(1, 75),
                stride=(1, 15),
            ),
        )
        """Overlapping temporal energy average."""

        drop: Makeable[TensorFn] = field(
            default_factory=lambda: Dropout.Config(p=0.5),
        )
        """Log-energy dropout."""

        proj_out: Makeable[TensorModule] = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Log-energy class projection."""

        @override
        def finalize(self) -> Self:
            _size(self.conv_t, channels_in=1, channels_out=self.channels_hidden)
            _size(self.conv_s, channels_in=self.channels_hidden)
            propagate_attr(
                self.conv_s,
                "kernel_size",
                (self.channels_in, 1),
                protocol=KernelSize,
            )
            _size(self.norm, channels_in=self.channels_hidden)
            _size(
                self.proj_out,
                channels_in=self.channels_hidden
                * math.prod(
                    _feature_grid(
                        (self.channels_in, self.num_samples),
                        operations=[self.conv_t, self.conv_s, self.pool],
                    ),
                ),
                channels_out=self.channels_out,
            )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost valid convolutions, squared energy, overlapping pool, and log.

            Args:
              seq_len: Temporal width, matching the configured classifier geometry.
              batch_size: Windows in the complete invocation.
              dtype: Activation and parameter dtype.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Forward and adjoint work, traffic, and owned parameters.

            """
            if seq_len != self.num_samples:
                raise ValueError("Cost sequence length must match num_samples.")
            grid = (self.channels_in, self.num_samples)
            result = Cost()
            for conv in (self.conv_t, self.conv_s):
                part, grid = _conv_cost(conv, grid, batch_size=batch_size, dtype=dtype)
                result += part
            elements = batch_size * self.channels_hidden * math.prod(grid)
            result += cost(
                self.norm,
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            result += elementwise_cost(
                primal=elements,
                adjoint=2 * elements,
                channels=elements,
                dtype=dtype,
            )
            part, grid = _pool_dropout_cost(
                self.pool,
                self.drop,
                channels=self.channels_hidden,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            outputs = batch_size * self.channels_hidden * math.prod(grid)
            return (
                result
                + part
                + elementwise_cost(
                    primal=2 * outputs,
                    adjoint=2 * outputs,
                    channels=outputs,
                    inputs=2,
                    outputs=2,
                    adjoint_inputs=4,
                    adjoint_outputs=2,
                    dtype=dtype,
                )
                + cost(
                    self.proj_out,
                    seq_len=1,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        absorb_legacy_keys(self, SOURCE_NAMES)
        self.eps = config.eps
        self.conv_t = config.conv_t.make()
        self.conv_s = config.conv_s.make()
        self.norm = config.norm.make()
        self.pool = config.pool.make()
        self.drop = config.drop.make()
        with torch.no_grad():
            self._features(torch.zeros(1, 1, config.channels_in, config.num_samples))
        self.proj_out = config.proj_out.make()

    @override
    def forward(self, media: Tensor) -> Tensor:
        """Return the source's two class logits."""
        features = self._features(media)
        return self.proj_out(features.view(features.size(0), -1))

    def _features(self, media: Tensor) -> Tensor:
        x = self.norm(self.conv_s(self.conv_t(media)))
        x = torch.log(torch.clamp(self.pool(x**2), min=self.eps))
        return self.drop(x)


def temporal_squeeze(media: Tensor) -> Tensor:
    """Mean time while retaining the source's collapsed-height vector layout.

    Args:
      media: Activations shaped ``[batch, channels, 1, time]``.

    Returns:
      squeezed: Time means shaped ``[batch, channels]``.

    """
    return media.mean(dim=-1).view(media.shape[0], media.shape[1])


def temporal_expand(gates: Tensor) -> Tensor:
    """Broadcast a channel vector over the EEG spatial and temporal axes.

    Args:
      gates: Channel gates shaped ``[batch, channels]``.

    Returns:
      expanded: The gates shaped ``[batch, channels, 1, 1]``.

    """
    return gates.view(gates.shape[0], gates.shape[1], 1, 1)


class AttentionEEGNet(nn.Module):
    """Apply squeeze-excitation before both temporal pooling operations."""

    class Config(Fig["AttentionEEGNet"], kw_only=False):
        channels_in: int = -1
        """Recording electrodes; upstream n_channels."""

        channels_out: int = -1
        """Class logits; upstream n_classes."""

        _: KW_ONLY

        num_samples: int = -1
        """Temporal input width; upstream n_samples."""

        channels_hidden: int = 8
        """Temporal filters; upstream F1."""

        multiplier: int = 2
        """Depthwise spatial multiplier; upstream D."""

        channels_projection: int = 16
        """Separable projection width; upstream F2."""

        reduction: int = 4
        """Integer bottleneck reduction of each gate; the public implementation's."""

        activation: ActivationFn = elu
        """Nonlinearity after the spatial and separable normalizations."""

        conv_t: SpatialConfig = field(
            default_factory=lambda: Conv2d.Config(kernel_size=(1, 63), padding=(0, 31)),
        )
        """Temporal filter bank."""

        norm_t: Makeable[TensorModule] = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Temporal feature normalization."""

        conv_s: SpatialConfig = field(default_factory=lambda: Conv2d.Config(padding=0))
        """Depthwise electrode projection."""

        norm_s: Makeable[TensorModule] = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Spatial feature normalization."""

        conv_sep_depth: SpatialConfig = field(
            default_factory=lambda: Conv2d.Config(kernel_size=(1, 15), padding=(0, 7)),
        )
        """Depthwise temporal refinement."""

        conv_sep_point: SpatialConfig = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0),
        )
        """Pointwise feature mixing."""

        norm_sep: Makeable[TensorModule] = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Mixed feature normalization."""

        pool1: SpatialConfig = field(
            default_factory=lambda: AvgPool2d.Config(kernel_size=(1, 4)),
        )
        """First temporal average."""

        pool2: SpatialConfig = field(
            default_factory=lambda: AvgPool2d.Config(kernel_size=(1, 8)),
        )
        """Second temporal average."""

        drop1: Makeable[TensorFn] = field(
            default_factory=lambda: Dropout.Config(p=0.25),
        )
        """First gated-feature dropout."""

        drop2: Makeable[TensorFn] = field(
            default_factory=lambda: Dropout.Config(p=0.25),
        )
        """Second gated-feature dropout."""

        proj_out: Makeable[TensorModule] = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Gated feature class projection."""

        gate: Makeable[TensorFn] = field(
            default_factory=lambda: SqueezeExcitation.Config(
                fc1=Linear.Config(),
                fc2=Linear.Config(),
                squeeze=temporal_squeeze,
                expand=temporal_expand,
            ),
        )
        """Channel-gating template; copied and sized for each position."""

        gates: list[Makeable[TensorFn]] = field(
            default_factory=list[Makeable[TensorFn]],
            init=False,
        )
        """Independently sized gate instances, expanded before child finalization."""

        @override
        def finalize(self) -> Self:
            spatial = self.channels_hidden * self.multiplier
            _size(self.conv_t, channels_in=1, channels_out=self.channels_hidden)
            _size(self.norm_t, channels_in=self.channels_hidden)
            _size(self.conv_s, channels_in=self.channels_hidden, channels_out=spatial)
            propagate_attr(self.conv_s, "groups", self.channels_hidden, protocol=Groups)
            propagate_attr(
                self.conv_s,
                "kernel_size",
                (self.channels_in, 1),
                protocol=KernelSize,
            )
            _size(self.norm_s, channels_in=spatial)
            _size(self.conv_sep_depth, channels_in=spatial)
            propagate_attr(self.conv_sep_depth, "groups", spatial, protocol=Groups)
            _size(
                self.conv_sep_point,
                channels_in=spatial,
                channels_out=self.channels_projection,
            )
            _size(self.norm_sep, channels_in=self.channels_projection)
            _size(
                self.proj_out,
                channels_in=self.channels_projection
                * math.prod(
                    _feature_grid(
                        (self.channels_in, self.num_samples),
                        operations=[
                            self.conv_t,
                            self.conv_s,
                            self.pool1,
                            self.conv_sep_depth,
                            self.conv_sep_point,
                            self.pool2,
                        ],
                    ),
                ),
                channels_out=self.channels_out,
            )
            self.gates = [self.gate.copy_tree(), self.gate.copy_tree()]
            for gate, width in zip(
                self.gates,
                (spatial, self.channels_projection),
                strict=True,
            ):
                propagate_attr(
                    gate,
                    name="channels_in",
                    value=width,
                    protocol=ChannelsIn,
                )
                propagate_attr(gate, name="channels_out", value=width)
                propagate_attr(
                    gate,
                    name="channels_hidden",
                    value=max(1, width // self.reduction),
                    protocol=ChannelsHidden,
                )
                gate.finalize()
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost both injected gates and the complete convolutional classifier.

            Args:
              seq_len: Temporal width, matching the configured classifier geometry.
              batch_size: Windows in the complete invocation.
              dtype: Activation and parameter dtype.
              **kwargs: Forwarded to both injected gate costs.

            Returns:
              cost: Forward and adjoint work, traffic, and owned parameters.

            """
            if seq_len != self.num_samples:
                raise ValueError("Cost sequence length must match num_samples.")
            spatial = self.channels_hidden * self.multiplier
            grid = (self.channels_in, self.num_samples)
            result, grid = _conv_cost(
                self.conv_t,
                grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            result += cost(
                self.norm_t,
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            part, grid = _conv_cost(
                self.conv_s,
                grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            result += part + _norm_activation_cost(
                self.norm_s,
                self.activation,
                channels=spatial,
                positions=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
            )
            result += cost(
                self.gates[0],
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            part, grid = _pool_dropout_cost(
                self.pool1,
                self.drop1,
                channels=spatial,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            result += part
            for conv in (self.conv_sep_depth, self.conv_sep_point):
                part, grid = _conv_cost(conv, grid, batch_size=batch_size, dtype=dtype)
                result += part
            result += _norm_activation_cost(
                self.norm_sep,
                self.activation,
                channels=self.channels_projection,
                positions=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
            )
            result += cost(
                self.gates[1],
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            part, grid = _pool_dropout_cost(
                self.pool2,
                self.drop2,
                channels=self.channels_projection,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            return (
                result
                + part
                + cost(
                    self.proj_out,
                    seq_len=1,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        absorb_legacy_keys(self, SOURCE_NAMES)
        self.activation = _activation(config.activation)
        self.conv_t = config.conv_t.make()
        self.norm_t = config.norm_t.make()
        self.conv_s = config.conv_s.make()
        self.norm_s = config.norm_s.make()
        self.se1 = config.gates[0].make()
        self.pool1 = config.pool1.make()
        self.drop1 = config.drop1.make()
        self.conv_sep_depth = config.conv_sep_depth.make()
        self.conv_sep_point = config.conv_sep_point.make()
        self.norm_sep = config.norm_sep.make()
        self.se2 = config.gates[1].make()
        self.pool2 = config.pool2.make()
        self.drop2 = config.drop2.make()
        with torch.no_grad():
            self._features(torch.zeros(1, 1, config.channels_in, config.num_samples))
        self.proj_out = config.proj_out.make()

    @override
    def forward(self, media: Tensor) -> Tensor:
        """Return the source's two class logits."""
        features = self._features(media)
        return self.proj_out(features.view(features.size(0), -1))

    def _features(self, media: Tensor) -> Tensor:
        x = self.norm_t(self.conv_t(media))
        x = self.activation(self.norm_s(self.conv_s(x)))
        x = self.se1(x)
        x = self.drop1(self.pool1(x))
        x = self.conv_sep_point(self.conv_sep_depth(x))
        x = self.activation(self.norm_sep(x))
        x = self.se2(x)
        return self.drop2(self.pool2(x))


def _activation(activation: ActivationFn) -> TensorFn:
    """Build an activation from a config, or pass a callable through."""
    if isinstance(activation, Makeable):
        return cast(TensorFn, activation.make())
    return activation


def _size(
    config: object,
    *,
    channels_in: int,
    channels_out: int | None = None,
) -> None:
    """Push a child's widths; one preserving its width takes ``channels_in``."""
    propagate_attr(config, "channels_in", channels_in, protocol=ChannelsIn)
    propagate_attr(
        config,
        "channels_out",
        channels_in if channels_out is None else channels_out,
        protocol=ChannelsOut,
    )


def _feature_grid(
    input_grid: tuple[int, ...],
    *,
    operations: Sequence[SpatialConfig],
) -> tuple[int, ...]:
    for operation in operations:
        operation.finalize()
        input_grid = operation.output_grid(input_grid)
    return input_grid


def _conv_cost(
    config: SpatialConfig,
    input_grid: tuple[int, ...],
    *,
    batch_size: int,
    dtype: torch.dtype | None,
) -> tuple[Cost, tuple[int, ...]]:
    return (
        cost(config, input_grid=input_grid, batch_size=batch_size, dtype=dtype),
        config.output_grid(input_grid),
    )


def _norm_activation_cost(
    norm: Makeable[TensorModule],
    activation: ActivationFn,
    *,
    channels: int,
    positions: int,
    batch_size: int,
    dtype: torch.dtype | None,
) -> Cost:
    return cost(
        norm,
        seq_len=positions,
        batch_size=batch_size,
        dtype=dtype,
    ) + cost(activation, channels=batch_size * channels * positions, dtype=dtype)


def _pool_dropout_cost(
    pool: SpatialConfig,
    dropout: Makeable[TensorFn],
    *,
    channels: int,
    input_grid: tuple[int, ...],
    batch_size: int,
    dtype: torch.dtype | None,
) -> tuple[Cost, tuple[int, ...]]:
    grid = pool.output_grid(input_grid)
    return (
        cost(
            pool,
            input_grid=input_grid,
            channels=channels,
            batch_size=batch_size,
            dtype=dtype,
        )
        + cost(
            dropout,
            seq_len=math.prod(grid),
            channels=channels,
            batch_size=batch_size,
            dtype=dtype,
        ),
        grid,
    )
