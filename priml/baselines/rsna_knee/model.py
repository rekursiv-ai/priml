"""Every neural network the RSNA knee recipe owns beyond the shared priml layers."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast, override, runtime_checkable

import math
import re

from configgle import Fig, Makeable, Makes, traverse
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.rsna_knee.data import FINDINGS
from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    map_cost,
    matmul_cost,
    nbytes,
    reduction_cost,
    set_cost,
    traffic,
)
from priml.math.custom_types import TensorFn
from priml.model.ae.checkpoint import LocalFile
from priml.model.coatnet import CoAtNet
from priml.model.conv import Conv2d
from priml.model.convnext import (
    ConvNeXt,
    ConvNeXtBlock,
    ConvNeXtHead,
    ConvNeXtStage,
    ConvNeXtStem,
    DropPathSchedule,
)
from priml.model.custom_types import (
    ChannelsIn,
    HasResetParameters,
    TensorModule,
    propagate_attr,
)
from priml.model.embedding import Embedding
from priml.model.init import InitFn, call_init, normal
from priml.model.linear import Linear
from priml.model.norm import LayerNorm, LayerNorm2d
from priml.model.sequential import Sequential
from priml.model.special import Activation, Dropout, Identity
from priml.model.state_dict import remap_state_dict, validate_state_dict
from priml.model.swiglu import SwiGLU, gelu
from priml.model.vnet import VNet
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Mapping


class R7(nn.Module):
    """Encode windows and combine spatial, state, extent, and synovitis evidence."""

    class Config(Fig["R7"]):
        channels_hidden: int = 384
        """Width of each class-specific window and study token."""

        encoder: ConvNeXt.Config = field(
            default_factory=lambda: ConvNeXt.Config(
                blocks_per_stage=(3, 3, 27, 3),
                drop_path=0.1,
                head=Identity.Config(),
            ),
        )
        """Raw spatial encoder; the pooled head normalization lives separately."""

        proj_in: Linear.Config = field(default_factory=lambda: Linear.Config(bias=True))
        """Projection from pooled encoder features to window tokens."""

        norm_in: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Normalization immediately after the window projection."""

        slot: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(10, init_weight=nn.init.zeros_),
        )
        """Series-slot embeddings."""

        view: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(6, init_weight=nn.init.zeros_),
        )
        """Rendered-view embeddings."""

        geom: Linear.Config = field(
            default_factory=lambda: Linear.Config(4, init_weight=nn.init.zeros_),
        )
        """Projection of relative position, spacing, extent, and fluid metadata."""

        dropout: Dropout.Config = field(default_factory=lambda: Dropout.Config(p=0.1))
        """Dropout of the window tokens."""

        win_proj: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True),
        )
        """Projection before finding-specific window-attention queries."""

        norm_out: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Normalization of each pooled class token."""

        norm_spatial: Makeable[TensorModule] = field(default_factory=LayerNorm.Config)
        """Nonaffine normalization of spatial keys before their projection."""

        norm_sp_head: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(eps=1e-6, elementwise_affine=True),
        )
        """Retained encoder head normalization, applied after spatial pooling."""

        mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
        """Per-channel center subtracted from pixels scaled to [0, 1]."""

        std: tuple[float, float, float] = (0.229, 0.224, 0.225)
        """Per-channel divisor of the centered pixels."""

        pixel_scale: float = 255.0
        """Divisor taking raw uint8 pixels to [0, 1]."""

        pad_multiple: int = 32
        """Real windows are padded to a multiple of this before encoding."""

        init_weight: InitFn = partial(normal, std=0.02)
        """Initializer of the window query, class, state and binary readout weights."""

        init_bias_extent: InitFn = partial(nn.init.constant_, val=-4.0)
        """Initializer of the extent logit offsets."""

        init_prior_state: InitFn = partial(nn.init.constant_, val=0.5)
        """Initializer of the state priors, replaced by the checkpoint's."""

        sp_key: Linear.Config = field(
            default_factory=lambda: Linear.Config(channels_out=128, bias=True),
        )
        """Projection from normalized spatial features to attention keys."""

        sp_pos: Linear.Config = field(
            default_factory=lambda: Linear.Config(init_weight=nn.init.zeros_),
        )
        """Projection of two-dimensional Fourier position features."""

        frequencies: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
        """Position frequencies, multiplied by pi/2."""

        state_counts: tuple[int, ...] = (6, 6, 5, 5, 6, 6, 6, 6, 5, 6, 6, 8)
        """Valid state slots in canonical finding order; priors load from checkpoint."""

        binary_parents: tuple[int, ...] = (4, 5, 6, 4, 5, 6, 4, 5, 6, 1, 11, 10, 0)
        """Parent-finding indices in the immutable state-spec binary order."""

        extent_classes: tuple[int, int, int] = (4, 5, 6)
        """Medial, lateral, and patellofemoral OA indices, respectively."""

        @override
        def finalize(self) -> Self:
            width = self.encoder.channels_hidden[-1]
            self.proj_in.channels_in = width
            self.proj_in.channels_out = self.channels_hidden
            self.slot.channels_out = self.view.channels_out = self.channels_hidden
            self.geom.channels_out = self.channels_hidden
            self.win_proj.channels_in = self.win_proj.channels_out = (
                self.channels_hidden
            )
            self.dropout.channels_in = self.channels_hidden
            for norm in (self.norm_in, self.norm_out):
                propagate_attr(
                    norm,
                    name="channels_in",
                    value=self.channels_hidden,
                    protocol=ChannelsIn,
                )
            for norm in (self.norm_spatial, self.norm_sp_head):
                propagate_attr(
                    norm,
                    name="channels_in",
                    value=width,
                    protocol=ChannelsIn,
                )
            self.sp_key.channels_in = width
            self.sp_pos.channels_in = 4 * len(self.frequencies)
            self.sp_pos.channels_out = self.sp_key.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            real_windows: int = 0,
            encoded_windows: int = 0,
            num_series: int = 16,
            chunk: int = 0,
            **kwargs: object,
        ) -> Cost:
            """Count all encoded windows, spatial pooling and complete clinical readouts.

            Args:
                seq_len: Window slots per study, including masked slots.
                input_grid: Height and width of each three-slice window.
                batch_size: Studies in this invocation.
                dtype: Training operand dtype; None uses Torch's default.
                real_windows: Total real windows; zero uses all supplied slots.
                encoded_windows: Actual encoder rows; zero uses the released
                    multiple-of-32 padding policy. Supply unpadded rows when
                    injecting an encoder with batch normalization in training.
                num_series: Actual extent membership width, at least sixteen.
                chunk: Encoder rows per deployed chunk; zero uses one chunk.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Whole training work and once-owned model state.

            """
            del kwargs
            real = real_windows or seq_len * batch_size
            encoded = encoded_windows or real + (-real) % self.pad_multiple
            step = chunk or encoded
            positions = math.prod(self.encoder.output_grid(input_grid))
            total = Cost()
            shared = Cost()
            for offset in range(0, encoded, step):
                count = min(step, encoded - offset)
                shared = cost(
                    self.encoder,
                    input_grid=input_grid,
                    batch_size=count,
                    dtype=dtype,
                    input_grad=False,
                )
                shared += self._spatial_cost(
                    positions=positions,
                    batch_size=count,
                    dtype=dtype,
                )
                total += shared
            total = Cost(
                cells=total.cells,
                params=shared.params,
                params_active=shared.params_active,
                bytes_state=shared.bytes_state,
            )
            elements = encoded * math.prod(input_grid) * self.encoder.channels_in
            total += traffic(
                "primal",
                "elementwise",
                elements=elements,
                dtype=torch.uint8,
            )
            total += elementwise_cost(
                primal=3 * elements,
                adjoint=0,
                channels=elements,
                dtype=dtype,
            )
            return total + self._head_cost(
                seq_len=seq_len,
                real_windows=real,
                batch_size=batch_size,
                num_series=num_series,
                dtype=dtype,
            )

        def _spatial_cost(
            self,
            *,
            positions: int,
            batch_size: int,
            dtype: torch.dtype | None,
        ) -> Cost:
            classes, width, key = (
                len(self.state_counts),
                self.encoder.channels_hidden[-1],
                self.sp_key.channels_out,
            )
            total = cost(
                self.norm_spatial,
                seq_len=positions,
                batch_size=batch_size,
                dtype=dtype,
            )
            total += cost(
                self.sp_key,
                seq_len=positions,
                batch_size=batch_size,
                dtype=dtype,
            )
            total += cost(
                self.sp_pos,
                seq_len=positions,
                batch_size=1,
                dtype=dtype,
                input_grad=False,
            )
            total += matmul_cost(
                channels_in=key,
                channels_out=classes,
                rows=batch_size * positions,
                dtype=dtype,
            )
            total += matmul_cost(
                channels_in=positions,
                channels_out=width,
                rows=classes,
                weight=False,
                dtype=dtype,
            ).tile(batch_size)
            total += cost(
                self.norm_sp_head,
                seq_len=classes,
                batch_size=batch_size,
                dtype=dtype,
            )
            elements = batch_size * positions * key
            total += elementwise_cost(
                primal=2 * elements,
                adjoint=4 * elements,
                channels=elements,
                inputs=2,
                dtype=dtype,
            )
            total += _softmax_cost(
                elements=batch_size * positions * classes,
                groups=batch_size * classes,
                dtype=dtype,
            )
            coordinate_elements = positions * len(self.frequencies)
            return total + traffic(
                "primal",
                "elementwise",
                elements=12 * coordinate_elements,
                flops=6 * coordinate_elements,
                dtype=dtype,
            )

        def _head_cost(
            self,
            *,
            seq_len: int,
            real_windows: int,
            batch_size: int,
            num_series: int,
            dtype: torch.dtype | None,
        ) -> Cost:
            classes, hidden = len(self.state_counts), self.channels_hidden
            rows = batch_size * seq_len
            total = Cost()
            for child in (self.proj_in, self.norm_in):
                total += cost(
                    child,
                    seq_len=classes,
                    batch_size=real_windows,
                    dtype=dtype,
                )
            for child in (self.slot, self.view):
                total += cost(
                    child,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            total += cost(
                self.geom,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                input_grad=False,
            )
            total += cost(
                self.win_proj,
                seq_len=seq_len * classes,
                batch_size=batch_size,
                dtype=dtype,
            )
            total += matmul_cost(
                channels_in=hidden,
                channels_out=1,
                rows=rows,
                dtype=dtype,
            ).tile(classes, copies=classes)
            total += matmul_cost(
                channels_in=seq_len,
                channels_out=hidden,
                rows=1,
                weight=False,
                dtype=dtype,
            ).tile(batch_size * classes)
            total += cost(
                self.norm_out,
                seq_len=classes,
                batch_size=batch_size,
                dtype=dtype,
            )
            total += matmul_cost(
                channels_in=hidden,
                channels_out=max(self.state_counts),
                rows=batch_size,
                bias=True,
                dtype=dtype,
            ).tile(classes, copies=classes)
            total += matmul_cost(
                channels_in=hidden,
                channels_out=1,
                rows=batch_size,
                bias=True,
                dtype=dtype,
            ).tile(len(self.binary_parents), copies=len(self.binary_parents))
            elements = rows * classes * hidden
            total += elementwise_cost(
                primal=3 * elements + 2 * rows * hidden,
                adjoint=5 * elements + 2 * rows * hidden,
                channels=elements,
                dtype=dtype,
            )
            total += cost(
                self.dropout,
                seq_len=seq_len * classes,
                batch_size=batch_size,
                dtype=dtype,
            )
            total += _softmax_cost(
                elements=rows * classes,
                groups=batch_size * classes,
                dtype=dtype,
            )
            total += _readout_cost(
                classes=classes,
                hidden=hidden,
                rows=batch_size,
                dtype=dtype,
            )
            total += _readout_cost(
                classes=classes,
                hidden=hidden,
                rows=rows,
                dtype=dtype,
            )
            total += self._extent_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                num_series=num_series,
                dtype=dtype,
            )
            states = batch_size * classes * max(self.state_counts)
            total += _softmax_cost(
                elements=states,
                groups=batch_size * classes,
                dtype=dtype,
            )
            total += elementwise_cost(
                primal=states + 7 * batch_size * classes,
                adjoint=2 * states + 7 * batch_size * classes,
                channels=classes,
                params=classes,
                rows=batch_size,
                dtype=dtype,
            )
            total += reduction_cost(
                input_elements=states,
                output_groups=batch_size * classes,
                dtype=dtype,
            )
            total += elementwise_cost(
                primal=4 * batch_size * classes,
                adjoint=5 * batch_size * classes,
                channels=classes,
                params=classes,
                rows=batch_size,
                dtype=dtype,
            )
            total += elementwise_cost(
                primal=batch_size * (1 + 2 * classes),
                adjoint=batch_size * classes,
                channels=classes,
                params=1,
                rows=batch_size,
                dtype=dtype,
            )
            return total + Cost(
                bytes_state=nbytes(
                    6 + classes * max(self.state_counts) + 2 * classes,
                    torch.float32,
                )
                + nbytes(classes * max(self.state_counts), torch.bool),
            )

        def _extent_cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            num_series: int,
            dtype: torch.dtype | None,
        ) -> Cost:
            classes, views = len(self.state_counts), self.view.channels_in
            total = matmul_cost(
                channels_in=seq_len,
                channels_out=classes,
                rows=num_series * views,
                weight=False,
                dtype=dtype,
                input_grad=False,
            ).tile(batch_size)
            total += (
                matmul_cost(
                    channels_in=seq_len,
                    channels_out=1,
                    rows=num_series,
                    weight=False,
                    dtype=dtype,
                )
                .only("primal")
                .tile(2 * batch_size)
            )
            total += _softmax_cost(
                elements=batch_size * num_series * views * classes,
                groups=batch_size * num_series * classes,
                dtype=dtype,
            )
            total += elementwise_cost(
                primal=2 * batch_size * num_series * views * classes,
                adjoint=3 * batch_size * num_series * views * classes,
                channels=views * classes,
                params=views * classes,
                rows=batch_size * num_series,
                dtype=dtype,
            )
            total += reduction_cost(
                input_elements=batch_size * num_series * views * classes,
                output_groups=batch_size * num_series * classes,
                dtype=dtype,
            )
            total += reduction_cost(
                input_elements=batch_size * seq_len * num_series,
                output_groups=batch_size * num_series,
                dtype=dtype,
            )
            elements = batch_size * seq_len * classes
            total += elementwise_cost(
                primal=5 * elements,
                adjoint=4 * elements,
                channels=elements,
                dtype=dtype,
            )
            total += traffic(
                "primal",
                "selection",
                elements=batch_size * seq_len * num_series * (views + 1),
                dtype=dtype,
            )
            total += traffic(
                "primal",
                "reduction",
                elements=4 * batch_size * num_series * classes,
                flops=3 * batch_size * classes * (num_series - 1),
                dtype=dtype,
            )
            return total + traffic(
                "adjoint",
                "selection",
                elements=4 * batch_size * num_series * classes,
                flops=3 * batch_size * num_series * classes,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        if len(config.state_counts) != len(FINDINGS):
            raise ValueError("State counts must match the canonical finding order.")
        if config.encoder.channels_in != 3:
            raise ValueError("The released R7 model consumes three adjacent slices.")
        super().__init__()
        classes = len(config.state_counts)
        hidden = config.channels_hidden
        self.state_counts = config.state_counts
        self.bin_parent = config.binary_parents
        self.extent_classes = config.extent_classes
        self.frequencies = config.frequencies
        self.num_slots = config.slot.channels_in
        self.num_views = config.view.channels_in
        self.area: Tensor | None = None
        self.syn_specific: Tensor | None = None
        self.syn_beta = nn.Parameter(torch.zeros(()))
        self.norm_sp_head = config.norm_sp_head.make()
        encoder = config.encoder.make()
        self.enc: TensorModule = encoder
        self.has_bn = any(
            isinstance(module, nn.BatchNorm2d) for module in encoder.modules()
        )
        self.proj = Sequential.Config(elements=[config.proj_in, config.norm_in]).make()
        self.slot = config.slot.make()
        self.view = config.view.make()
        self.geom = config.geom.make()
        self.drop = config.dropout.make()
        self.win_proj = config.win_proj.make()
        self.q_win = nn.Parameter(torch.empty(classes, hidden))
        call_init(config.init_weight, self.q_win)
        self.norm_out = config.norm_out.make()
        self.cls_w = nn.Parameter(torch.empty(classes, hidden))
        call_init(config.init_weight, self.cls_w)
        self.bias = nn.Parameter(torch.zeros(classes))
        self.mean = nn.Buffer(torch.tensor(config.mean).view(1, 3, 1, 1))
        self.std = nn.Buffer(torch.tensor(config.std).view(1, 3, 1, 1))
        self.pixel_scale = config.pixel_scale
        self.pad_multiple = config.pad_multiple
        kmax = max(config.state_counts)
        self.state_w = nn.Parameter(torch.empty(classes, kmax, hidden))
        call_init(config.init_weight, self.state_w)
        self.state_b = nn.Parameter(torch.zeros(classes, kmax))
        self.state_prior = nn.Buffer(torch.empty(classes, kmax))
        call_init(config.init_prior_state, self.state_prior)
        self.state_valid = nn.Buffer(
            torch.arange(kmax)[None] < torch.tensor(config.state_counts)[:, None],
        )
        self.state_base_logit = nn.Buffer(torch.zeros(classes))
        self.state_gate = nn.Parameter(torch.zeros(classes))
        self.bin_w = nn.Parameter(torch.empty(len(config.binary_parents), hidden))
        call_init(config.init_weight, self.bin_w)
        self.bin_b = nn.Parameter(torch.zeros(len(config.binary_parents)))
        self.norm_spatial = config.norm_spatial.make()
        self.sp_key = config.sp_key.make()
        self.sp_pos = config.sp_pos.make()
        self.sp_q = nn.Parameter(torch.zeros(classes, config.sp_key.channels_out))
        self.ext_u = nn.Parameter(torch.zeros(classes, hidden))
        self.ext_b = nn.Parameter(torch.empty(classes))
        call_init(config.init_bias_extent, self.ext_b)
        self.ext_gamma = nn.Parameter(torch.zeros(classes))
        self.ext_view = nn.Parameter(torch.zeros(classes, self.num_views))
        extent_mask = torch.zeros(classes)
        extent_mask[list(config.extent_classes)] = 1.0
        self.ext_mask = nn.Buffer(extent_mask)

    def spatial_pool(self, x: Tensor) -> Tensor:
        """Pool the raw map separately per finding, then apply encoder head norm.

        Args:
            x: Raw NCHW encoder feature maps.

        Returns:
            pooled: Normalized feature vectors per window and finding.

        """
        height, width = x.shape[2:]
        features = x.flatten(2).transpose(1, 2)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device),
            torch.linspace(-1, 1, width, device=x.device),
            indexing="ij",
        )
        frequencies = torch.tensor(self.frequencies, device=x.device) * (math.pi / 2)
        yy, xx = yy.reshape(-1, 1) * frequencies, xx.reshape(-1, 1) * frequencies
        positions = torch.cat([yy.sin(), yy.cos(), xx.sin(), xx.cos()], 1)
        keys_in = self.norm_spatial(features.float()).to(features.dtype)
        keys = torch.tanh(
            self.sp_key(keys_in) + self.sp_pos(positions.to(features.dtype))[None],
        )
        weights = (
            torch.einsum(keys, [0, 1, 2], self.sp_q.to(keys.dtype), [3, 2])
            .float()
            .softmax(1)
            .to(features.dtype)
        )
        pooled = torch.einsum(weights, [0, 1, 2], features, [0, 1, 3], [0, 2, 3])
        return self.norm_sp_head(pooled.float()).to(features.dtype)

    def extent_area(self, logits: Tensor, meta: Tensor, mask: Tensor) -> Tensor:
        """Combine view means within series and orthogonal-plane defect extents.

        Args:
            logits: Per-window finding extent logits.
            meta: Window geometry, series, and rendered-view metadata.
            mask: Real-window membership in each study.

        Returns:
            area: Orthogonal-plane defect areas in canonical finding order.

        """
        series = meta[..., 6].long()
        count_series = max(16, int(series.max()) + 1)
        views = meta[..., 1].long().clamp(0, self.num_views - 1)
        probability = torch.sigmoid(logits.float()) * mask[..., None]
        view_membership = (
            functional.one_hot(
                series * self.num_views + views,
                count_series * self.num_views,
            ).float()
            * mask[..., None]
        )
        view_counts = view_membership.sum(1)
        view_means = (
            torch.einsum(view_membership, [0, 1, 2], probability, [0, 1, 3], [0, 2, 3])
            / view_counts.clamp_min(1)[..., None]
        ).view(len(logits), count_series, self.num_views, -1)
        present = (view_counts > 0).view(len(logits), count_series, self.num_views, 1)
        weights = (
            self.ext_view.float()
            .t()[None, None]
            .masked_fill(~present, torch.finfo(torch.float32).min)
            .softmax(2)
            * present
        )
        series_means = (weights * view_means).sum(2)
        membership = functional.one_hot(series, count_series).float() * mask[..., None]
        counts = membership.sum(1)
        span = torch.expm1(meta[..., 3].float()).clamp_min(0) * meta[..., 8].float()
        span_series = torch.einsum(
            membership,
            [0, 1, 2],
            span,
            [0, 1],
            [0, 2],
        ) / counts.clamp_min(
            1,
        )
        plane_series = (
            torch.einsum(membership, [0, 1, 2], meta[..., 7].float(), [0, 1], [0, 2])
            / counts.clamp_min(1)
        ).round()
        extents = series_means * span_series[..., None]
        planes = [
            extents.masked_fill(
                ~(plane_series == plane)[..., None],
                0.0,
            ).amax(1)
            for plane in (0, 1, 2)
        ]
        sagittal, coronal, axial = planes
        area = torch.zeros_like(sagittal)
        medial, lateral, patellofemoral = self.extent_classes
        for finding in (medial, lateral):
            area[:, finding] = sagittal[:, finding] * coronal[:, finding]
        area[:, patellofemoral] = axial[:, patellofemoral] * sagittal[:, patellofemoral]
        return area

    def head(
        self,
        tokens: Tensor,
        meta: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return direct logits plus extent/state corrections and all state readouts.

        Args:
            tokens: Encoded real-window features, grouped by finding.
            meta: Window geometry and acquisition metadata.
            mask: Real-window membership in each study.

        Returns:
            outputs: Finding, padded state, and binary-state logits.

        """
        batch, windows = mask.shape
        features = self.proj(tokens.float())
        embeddings = (
            self.slot(meta[..., 0].long().clamp(0, self.num_slots - 1))
            + self.view(meta[..., 1].long().clamp(0, self.num_views - 1))
            + self.geom(meta[..., 2:6])
        )
        token = features.new_zeros((batch, windows, *features.shape[1:]))
        token[mask] = features
        token = self.drop(token + embeddings[:, :, None, :])
        scores = torch.einsum(
            torch.tanh(self.win_proj(token)),
            [0, 1, 2, 3],
            self.q_win,
            [2, 3],
            [0, 1, 2],
        ).float()
        scores = scores.masked_fill(~mask[:, :, None], -math.inf)
        attention = scores.softmax(1)
        study = self.norm_out(
            torch.einsum(attention, [0, 1, 2], token, [0, 1, 2, 3], [0, 2, 3]),
        )
        extent_logits = (token.float() * self.ext_u[None, None]).sum(-1) + self.ext_b
        logits = (study * self.cls_w).sum(-1).float() + self.bias
        self.area = self.extent_area(extent_logits, meta=meta, mask=mask)
        logits = logits + self.ext_gamma * self.ext_mask * torch.log1p(
            self.area / 100.0,
        )
        states = (
            torch.einsum(study, [0, 1, 2], self.state_w, [1, 3, 2], [0, 1, 3]).float()
            + self.state_b
        )
        states = states.masked_fill(~self.state_valid[None], -math.inf)
        binaries = (
            torch.einsum(
                study[:, self.bin_parent],
                [0, 1, 2],
                self.bin_w,
                [1, 2],
                [0, 1],
            ).float()
            + self.bin_b
        )
        implied = (
            (states.softmax(-1) * self.state_prior[None]).sum(-1).clamp(1e-3, 1 - 1e-3)
        )
        logits = logits + self.state_gate[None] * (
            torch.logit(implied) - self.state_base_logit[None]
        )
        return self.decompose_syn(logits), states, binaries

    def decompose_syn(self, logits: Tensor) -> Tensor:
        """Add detached effusion evidence only to the synovitis output.

        Args:
            logits: Finding logits before effusion adjustment.

        Returns:
            adjusted: Finding logits with the synovitis adjustment.

        """
        synovitis, effusion = FINDINGS.index("Synovitis"), FINDINGS.index("Effusion")
        self.syn_specific = logits[:, synovitis].detach().float()
        covariate = logits[:, effusion].detach().clone()
        select = torch.zeros(logits.shape[1], device=logits.device, dtype=logits.dtype)
        select[synovitis] = 1
        return logits + (self.syn_beta * covariate)[:, None] * select[None, :]

    def state_readout(self, states: Tensor, binaries: Tensor) -> Tensor:
        """Return valid state probabilities followed by binary probabilities.

        Args:
            states: Padded categorical-state logits in finding order.
            binaries: Binary-state logits in state-spec order.

        Returns:
            probabilities: All 84 exported state probabilities.

        """
        probabilities = states.softmax(-1)
        return torch.cat(
            [
                probabilities[:, finding, :count]
                for finding, count in enumerate(self.state_counts)
            ]
            + [torch.sigmoid(binaries)],
            1,
        ).float()

    @override
    def forward(
        self,
        image: Tensor,
        meta: Tensor,
        mask: Tensor,
        *,
        chunk: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Normalize and encode real windows, padding and chunking as in deployment."""
        selected = image[mask]
        count = len(selected)
        if count % self.pad_multiple and (not self.training or not self.has_bn):
            selected = torch.cat(
                [
                    selected,
                    selected.new_zeros(
                        (
                            self.pad_multiple - count % self.pad_multiple,
                            *selected.shape[1:],
                        ),
                    ),
                ],
            )
        features: list[Tensor] = []
        step = chunk or len(selected)
        for start in range(0, len(selected), step):
            normalized = selected[start : start + step].float().div_(self.pixel_scale)
            normalized = ((normalized - self.mean) / self.std).contiguous(
                memory_format=torch.channels_last,
            )
            features.append(self.spatial_pool(self.enc(normalized)))
        return self.head(torch.cat(features)[:count], meta=meta, mask=mask)

    def load_checkpoint(self, state: Mapping[str, Tensor]) -> None:
        """Validate and load all deployed model tensors through a name bijection.

        Args:
            state: Complete checkpoint mapping with original or native names.

        """
        encoder = {
            name.removeprefix("enc."): value
            for name, value in state.items()
            if name.startswith("enc.")
        }
        rest = remap_state_dict(
            {
                name: value
                for name, value in state.items()
                if not name.startswith("enc.")
            },
            lambda name: name.replace("out_norm.", "norm_out.").replace(
                "sp_head_norm.",
                "norm_sp_head.",
            ),
        )
        validate_state_dict(
            rest,
            expected={
                name: value
                for name, value in self.state_dict().items()
                if not name.startswith("enc.")
            },
        )
        assert isinstance(self.enc, ConvNeXt)
        self.enc.load_timm_state_dict(encoder)
        self.load_state_dict({**self.enc.state_dict(prefix="enc."), **rest})


@set_cost(map_cost(primal=1, adjoint=3))
def tanh(x: Tensor) -> Tensor:
    """Apply the hyperbolic tangent with its measured elementwise cost."""
    return torch.tanh(x)


class FindingAttention(nn.Module):
    """Pool image features separately for each finding, following Raptor's head."""

    class Config(Fig["FindingAttention"]):
        channels_in: int = -1
        """Image encoder feature width."""

        channels_out: int = -1
        """Number of findings."""

        _: KW_ONLY

        channels_hidden: int = 256
        """Attention projection width."""

        norm: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Normalization of every window before attention and pooling."""

        proj_in: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Window projection before the activation and dropout."""

        activation: Makeable[nn.Module] = field(
            default_factory=lambda: Activation.Config(act=tanh),
        )
        """Nonlinearity of the attention hidden layer."""

        dropout: Dropout.Config = field(default_factory=lambda: Dropout.Config(p=0.2))
        """Dropout of the attention hidden activation."""

        proj_out: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Projection to the finding-specific attention scores."""

        init_weight: InitFn = partial(nn.init.trunc_normal_, std=0.02)
        """Initializer of the per-finding classifier weights."""

        init_bias: InitFn = nn.init.zeros_
        """Initializer of the per-finding classifier biases."""

        @override
        def finalize(self) -> Self:
            propagate_attr(
                self.norm,
                name="channels_in",
                value=self.channels_in,
                protocol=ChannelsIn,
            )
            propagate_attr(
                self.activation,
                name="channels_in",
                value=self.channels_hidden,
                protocol=ChannelsIn,
            )
            self.proj_in.channels_in = self.channels_in
            self.proj_in.channels_out = self.proj_out.channels_in = self.channels_hidden
            self.dropout.channels_in = self.channels_hidden
            self.proj_out.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count normalization, attention, finding pooling and classification.

            Args:
              seq_len: Images in each study's bag.
              batch_size: Studies in this invocation.
              dtype: Operand dtype; None uses torch's default.
              **kwargs: Unread cost bus.

            Returns:
              execution: Complete primal/adjoint cost and parameter ownership.

            """
            del kwargs
            config = self.copy_tree().finalize()
            width, findings = self.channels_in, self.channels_out
            rows = batch_size * seq_len
            attention = rows * findings
            pooled = batch_size * findings * width
            return (
                cost(
                    config.norm,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    config.proj_in,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    config.activation,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    config.dropout,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    config.proj_out,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    kernel="elementwise",
                    elements=6 * attention + 2 * batch_size * findings,
                    flops=3 * attention,
                    dtype=dtype,
                )
                + reduction_cost(
                    input_elements=attention,
                    output_groups=batch_size * findings,
                    dtype=dtype,
                ).tile(2)
                + traffic(
                    "adjoint",
                    kernel="elementwise",
                    elements=7 * attention + batch_size * findings,
                    flops=3 * attention,
                    dtype=dtype,
                )
                + reduction_cost(
                    input_elements=attention,
                    output_groups=batch_size * findings,
                    dtype=dtype,
                    phase="adjoint",
                )
                + matmul_cost(
                    channels_in=seq_len,
                    channels_out=width,
                    rows=findings,
                    weight=False,
                    dtype=dtype,
                ).tile(batch_size)
                + elementwise_cost(
                    primal=pooled + batch_size * findings,
                    adjoint=2 * pooled,
                    channels=findings * width,
                    params=findings * (width + 1),
                    rows=batch_size,
                    dtype=dtype,
                )
                + reduction_cost(
                    input_elements=pooled,
                    output_groups=batch_size * findings,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=0,
                    adjoint=rows * width,
                    channels=width,
                    rows=rows,
                    inputs=0,
                    outputs=0,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm = config.norm.make()
        self.att = nn.Sequential(
            config.proj_in.make(),
            config.activation.make(),
            config.dropout.make(),
            config.proj_out.make(),
        )
        self.init_weight = config.init_weight
        self.init_bias = config.init_bias
        self.clsW = nn.Parameter(torch.empty(config.channels_out, config.channels_in))
        self.clsb = nn.Parameter(torch.empty(config.channels_out))
        call_init(self.init_weight, self.clsW)
        call_init(self.init_bias, self.clsb)

    def reset_parameters(self) -> None:
        """Reset the finding classifier and all injected pooling projections."""
        self.norm.reset_parameters()
        for child in self.att:
            if isinstance(child, HasResetParameters):
                child.reset_parameters()
        call_init(self.init_weight, self.clsW)
        call_init(self.init_bias, self.clsb)

    @override
    def forward(self, features: Tensor) -> Tensor:
        """Return study-by-finding logits from study-by-window-by-feature inputs."""
        normalized = self.norm(features)
        attention = torch.softmax(self.att(normalized), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", attention, normalized)
        return (pooled * self.clsW).sum(-1) + self.clsb


@runtime_checkable
class FeatureWidth(Protocol):
    """An encoder config that knows the width of the features it produces."""

    def feature_width(self) -> int:
        """Return the width of each encoded feature vector."""
        ...


class Raptor(nn.Module):
    """Encode every window of a study and pool all windows per finding."""

    class Config(Fig["Raptor"]):
        backbone: Makeable[nn.Module] = field(default_factory=CoAtNet.Config)
        """Encoder mapping each window to one feature vector."""

        head: FindingAttention.Config = field(
            default_factory=lambda: FindingAttention.Config(channels_out=len(FINDINGS)),
        )
        """Finding-specific attention and classification head."""

        chunk_size: float = math.inf
        """Encoder windows per pass; infinity encodes all windows together."""

        @override
        def finalize(self) -> Self:
            if isinstance(self.backbone, FeatureWidth):
                self.head.channels_in = self.backbone.feature_width()
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count every encoder chunk and the once-owned whole-study MIL head.

            Args:
                seq_len: Windows per study.
                input_grid: Spatial extents of each window.
                batch_size: Studies in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Outer cost arguments forwarded to both components.

            Returns:
                execution: Whole training work with shared encoder ownership.

            """
            config = self.copy_tree().finalize()
            count = seq_len * batch_size
            step = int(min(config.chunk_size, count))
            chunks = [
                cost(
                    config.backbone,
                    input_grid=input_grid,
                    batch_size=min(step, count - offset),
                    dtype=dtype,
                    **kwargs,
                )
                for offset in range(0, count, step)
            ]
            total = sum(chunks, start=Cost())
            encoder = Cost(
                cells=total.cells,
                params=chunks[0].params,
                params_active=chunks[0].params_active,
                bytes_state=chunks[0].bytes_state,
            )
            return encoder + cost(
                config.head,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.chunk_size < 1:
            raise ValueError("Encoder chunk size must be positive.")
        self.chunk_size = config.chunk_size
        self.backbone = config.backbone.make()
        self.head = config.head.make()

    def reset_parameters(self) -> None:
        """Reset the injected encoder and every finding-head parameter in place."""
        if isinstance(self.backbone, HasResetParameters):
            self.backbone.reset_parameters()
        self.head.reset_parameters()

    def encode(self, media: Tensor) -> Tensor:
        """Return complete per-window features with the deployed encoder batching.

        Args:
            media: Study-by-window-by-channel-by-height-by-width inputs.

        Returns:
            features: Study-by-window-by-feature encoder outputs.

        """
        batch, count = media.shape[:2]
        flat = media.flatten(0, 1)
        encoder = cast(TensorFn, self.backbone)
        encoded = torch.cat(
            [
                encoder(chunk)
                for chunk in flat.split(int(min(self.chunk_size, len(flat))))
            ],
        )
        return encoded.view(batch, count, -1)

    @override
    def forward(self, media: Tensor, **batch: object) -> Tensor:
        """Return finding logits; ignore labels forwarded by the generic train step.

        Args:
            media: Study-by-window-by-channel-by-height-by-width image tensor.
            **batch: Labels and other fields consumed by the training loss.

        Returns:
            logits: One score per study and finding.

        """
        del batch
        return self.head(self.encode(media))

    def load_checkpoint(self, state: Mapping[str, Tensor]) -> None:
        """Validate every source or native tensor before mutating any model state.

        Args:
            state: Complete CoAtNet plus finding-head checkpoint.

        """
        mapped = remap_state_dict(
            state,
            lambda name: (
                "head." + name
                if name in {"clsW", "clsb"} or name.startswith(("norm.", "att."))
                else name
            ),
        )
        validate_state_dict(mapped, expected=self.state_dict())
        self.load_state_dict(mapped)


def tiny_drop_path(maximum: float, count: int) -> tuple[float, ...]:
    """Retain the source binary64 linear stochastic-depth arithmetic."""
    return tuple(maximum * index / max(count - 1.0, 1.0) for index in range(count))


class TinyConvNeXt(ConvNeXt):
    """Compose the cold recipe's Tiny encoder from existing native ConvNeXt blocks."""

    class Config(Makes["TinyConvNeXt"], ConvNeXt.Config):
        init_weight: InitFn = partial(nn.init.trunc_normal_, std=0.02)
        """Weight initializer of every convolution and linear layer.

        Applied after construction, so each layer's own constructor draws keep
        the source's random stream before this pass overwrites them.
        """

        init_bias: InitFn = nn.init.zeros_
        """Bias initializer of every convolution and linear layer, applied likewise."""

        drop_path: float = 0.1
        """Original linear row-stochastic-depth schedule."""

        stem: ConvNeXtStem.Config = field(
            default_factory=lambda: ConvNeXtStem.Config(
                conv=Conv2d.Config(
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    bias=True,
                    init_bias=None,
                ),
            ),
        )
        """Patch stem including original constructor bias draws."""

        stage: ConvNeXtStage.Config | list[ConvNeXtStage.Config] = field(
            default_factory=lambda: ConvNeXtStage.Config(
                downsample=Sequential.Config(
                    elements=[
                        LayerNorm2d.Config(),
                        Conv2d.Config(
                            kernel_size=2,
                            stride=2,
                            padding=0,
                            bias=True,
                            init_bias=None,
                        ),
                    ],
                ),
                block=ConvNeXtBlock.Config(
                    conv_dw=Conv2d.Config(
                        kernel_size=7,
                        padding=3,
                        bias=True,
                        init_bias=None,
                    ),
                    mlp=SwiGLU.Config(
                        gate=False,
                        round_to=1,
                        act=gelu,
                        up_proj=Linear.Config(bias=True, init_bias=None),
                        down_proj=Linear.Config(bias=True, init_bias=None),
                    ),
                ),
            ),
        )
        """Existing native spatial stages with source-compatible initialization."""

        head: Makeable[TensorModule] = field(
            default_factory=lambda: ConvNeXtHead.Config(
                proj_out=Linear.Config(channels_out=1_000, bias=True, init_bias=None),
            ),
        )
        """Adaptive pooled normalization and all1000 initializer classes."""

        drop_path_schedule: DropPathSchedule = tiny_drop_path
        """Source Python arithmetic, independent of the shared float32 schedule."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.init_weight = config.init_weight
        self.init_bias = config.init_bias
        self._initialize()

    @override
    def reset_parameters(self) -> None:
        """Repeat constructor draws and the complete global initializer pass."""
        super().reset_parameters()
        self._initialize()

    def native_state(self, state: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Return a complete ImageNet initializer under this model's names."""
        return checkpoint_state(state)

    def remap_checkpoint(self, state: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Validate and remap every complete ImageNet initializer tensor.

        Args:
            state: Every original or native parameter and buffer before head removal.

        Returns:
            mapped: A strict name/shape/dtype bijection containing every tensor.

        """
        mapped = self.native_state(state)
        validate_state_dict(mapped, expected=self.state_dict())
        return mapped

    def load_checkpoint(self, state: Mapping[str, Tensor]) -> None:
        """Load the complete acquired ImageNet initializer before removing its head."""
        self.load_state_dict(self.remap_checkpoint(state))

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                call_init(self.init_weight, module.weight)
                if module.bias is not None:
                    call_init(self.init_bias, module.bias)


def checkpoint_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Normalize complete torchvision or native checkpoint names and layer scales.

    Args:
        state: Complete acquired initializer, before classifier removal.

    Returns:
        mapped: Native names and vector layer scales, without filtering tensors.

    """
    mapped = remap_state_dict(state, _checkpoint_name)
    for name in state:
        if name.endswith(".layer_scale"):
            target = _checkpoint_name(name)
            value = mapped[target]
            if value.ndim != 3 or value.shape[1:] != (1, 1):
                raise ValueError(f"Checkpoint layer-scale shape differs: {name}.")
            mapped[target] = value.reshape(-1)
    return mapped


def _checkpoint_name(name: str) -> str:
    parts = name.split(".")
    if parts[0] not in ("classifier", "features"):
        return name
    match parts:
        case ["classifier", component, parameter] if component in (
            "0",
            "2",
        ) and parameter in ("weight", "bias"):
            return "head." + {"0": "norm", "2": "proj_out"}[component] + "." + parameter
        case ["features", feature, component, parameter] if (
            _checkpoint_index(feature)
            and int(feature) % 2 == 0
            and component in ("0", "1")
            and parameter in ("weight", "bias")
        ):
            if feature == "0":
                return "stem." + {"0": "conv", "1": "norm"}[component] + "." + parameter
            return f"stages.{int(feature) // 2}.downsample.{component}.{parameter}"
        case ["features", feature, block, "layer_scale"] if (
            _checkpoint_index(feature)
            and int(feature) % 2 == 1
            and _checkpoint_index(block)
        ):
            return f"stages.{(int(feature) - 1) // 2}.blocks.{block}.gamma"
        case ["features", feature, block, "block", component, parameter] if (
            _checkpoint_index(feature)
            and int(feature) % 2 == 1
            and _checkpoint_index(block)
            and component in ("0", "2", "3", "5")
            and parameter in ("weight", "bias")
        ):
            component = {
                "0": "conv_dw",
                "2": "norm",
                "3": "mlp.up_proj",
                "5": "mlp.down_proj",
            }[component]
            return f"stages.{(int(feature) - 1) // 2}.blocks.{block}.{component}.{parameter}"
        case _:
            raise ValueError(f"Checkpoint source name is invalid: {name}.")


def _checkpoint_index(value: str) -> bool:
    return value.isascii() and value.isdecimal() and str(int(value)) == value


@runtime_checkable
class NativeCheckpointNames(Protocol):
    """A backbone that maps a foreign checkpoint's names onto its own."""

    def native_state(self, state: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Return the checkpoint under this backbone's parameter names."""
        ...


class ImageNetEncoder(nn.Module):
    """Load an acquired ImageNet checkpoint, removing only its final classifier."""

    class Config(Fig["ImageNetEncoder"]):
        backbone: Makeable[nn.Module] = field(
            default_factory=TinyConvNeXt.Config,
        )
        """ConvNeXt architecture before its ImageNet classifier is removed."""

        base_dir: Path | str | None = None
        """Resource root beneath which `working_dir` resolves; the owner supplies it."""

        working_dir: Path | str = "/datasets/rsna-knee-imagenet"
        """Logical directory holding the acquired initializer."""

        checkpoint: LocalFile.Config = field(
            default_factory=lambda: LocalFile.Config(
                path="convnext_tiny-983f1562.pth",
                sha256="983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
            ),
        )
        """Acquired torchvision ImageNet-1K initializer, named within `working_dir`.

        An absolute path overrides `working_dir`. Never a knee-selected checkpoint.
        """

        classifier: str = "head.proj_out"
        """Final classification module replaced by an identity after strict loading."""

        @override
        def finalize(self) -> Self:
            self.checkpoint.path = (
                resolve_working_dir(self.base_dir, self.working_dir)
                / self.checkpoint.path
            )
            return super().finalize()

        def feature_width(self) -> int:
            """Return the width of each feature once the classifier is removed."""
            if not isinstance(self.backbone, ConvNeXt.Config):
                raise TypeError("Feature width requires a ConvNeXt architecture.")
            return self.backbone.channels_hidden[-1]

        def cost(
            self,
            *,
            input_grid: tuple[int, int],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count the declared feature architecture without loading weights.

            Args:
              input_grid: Input image height and width.
              batch_size: Images in this invocation.
              dtype: Operand dtype; None uses torch's default.
              **kwargs: Unread cost bus.

            Returns:
              execution: Complete feature encoder cost.

            Raises:
              ValueError: The classifier path does not name a declared config leaf.
              TypeError: The injected feature architecture has no declared cost.

            """
            del kwargs
            backbone = self.backbone.copy_tree()
            for match in traverse(backbone, Makeable, recurse=True):
                if match.fqn == self.classifier:
                    match.replace(Identity.Config())
                    return cost(
                        backbone.finalize(),
                        input_grid=input_grid,
                        batch_size=batch_size,
                        dtype=dtype,
                    )
            raise ValueError("No cost model for the configured classifier path.")

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        state = self._read_checkpoint()
        self.backbone = config.backbone.make()
        if isinstance(self.backbone, NativeCheckpointNames):
            state = self.backbone.native_state(state)
        expected = self.backbone.state_dict()
        validate_state_dict(state, expected=expected)
        if not any(value.is_meta for value in expected.values()):
            self.backbone.load_state_dict(state)
        self.backbone.set_submodule(
            config.classifier,
            module=Identity.Config().make(),
            strict=True,
        )

    def reset_parameters(self) -> None:
        """Restore pretrained features in place after allocation or an explicit reset."""
        state = self._read_checkpoint()
        if isinstance(self.backbone, NativeCheckpointNames):
            state = self.backbone.native_state(state)
        classifier = self.config.classifier + "."
        self.backbone.load_state_dict(
            {
                name: value
                for name, value in state.items()
                if not name.startswith(classifier)
            },
        )

    @override
    def forward(self, media: Tensor) -> Tensor:
        """Return a feature vector for each image."""
        return cast(TensorFn, self.backbone)(media)

    def _read_checkpoint(self) -> dict[str, Tensor]:
        path = self.config.checkpoint.make().path()
        return cast(
            dict[str, Tensor],
            torch.load(path, map_location="cpu", weights_only=True),
        )


class ReleasedVNet(VNet):
    """The priml VNet loading the released MONAI-0.6 anatomical checkpoint."""

    class Config(Makes["ReleasedVNet"], VNet.Config):
        """The priml VNet configuration, producing the checkpoint-aware model."""

    def load_checkpoint(self, state: Mapping[str, Tensor]) -> None:
        """Load a complete released model state through a strict name bijection.

        Args:
          state: Every VNet parameter and buffer, with an optional model.net prefix.

        Raises:
          ValueError: Names collide, or names, shapes, or dtypes do not match.

        """
        mapped = remap_state_dict(state, _vnet_name)
        validate_state_dict(mapped, expected=self.state_dict())
        self.load_state_dict(mapped)


def _vnet_name(name: str) -> str:
    target = name.removeprefix("model.net.")
    for index, width in enumerate((32, 64, 128, 256)):
        target = target.replace(f"down_tr{width}.", f"down.{index}.")
    for index, width in enumerate((256, 128, 64, 32)):
        target = target.replace(f"up_tr{width}.", f"up.{index}.")
    target = target.replace("in_tr.conv_block.", "input.layer.")
    target = target.replace("out_tr.conv_block.", "output.conv.")
    target = target.replace("out_tr.conv2.", "output.proj.")
    target = target.replace(".down_conv.", ".proj.conv.")
    target = target.replace(".up_conv.", ".proj.")
    target = target.replace(
        ".bn1.",
        ".proj.norm." if target.startswith("down.") else ".norm.",
    )
    target = target.replace(".ops.", ".blocks.").replace(".conv_block.", ".")
    # The upstream container abbreviates activation, dropout, and normalization.
    return re.sub(r"\.ad[n]\.N\.", ".norm.", target)


def _softmax_cost(*, elements: int, groups: int, dtype: torch.dtype | None) -> Cost:
    return (
        elementwise_cost(
            primal=3 * elements,
            adjoint=3 * elements,
            channels=elements,
            dtype=dtype,
        )
        + reduction_cost(
            input_elements=elements,
            output_groups=groups,
            dtype=dtype,
        ).tile(2)
        + reduction_cost(
            input_elements=elements,
            output_groups=groups,
            phase="adjoint",
            dtype=dtype,
        )
    )


def _readout_cost(
    *,
    classes: int,
    hidden: int,
    rows: int,
    dtype: torch.dtype | None,
) -> Cost:
    elements = rows * classes * hidden
    return elementwise_cost(
        primal=elements + rows * classes,
        adjoint=2 * elements,
        channels=classes * hidden,
        params=classes * (hidden + 1),
        rows=rows,
        dtype=dtype,
    ) + reduction_cost(
        input_elements=elements,
        output_groups=rows * classes,
        dtype=dtype,
    )
