"""Every CASMI network that is not a promoted architecture, in one module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Self, TypedDict, override

import copy
import math

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

import numpy as np
import torch

from priml.baselines.casmi2026.isolated import SIMULATOR_HYDROGEN_MASS
from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    nbytes,
    reduction_cost,
    resolve_dtype,
    traffic,
)
from priml.model.attention.attention import Attention, AttentionProjections
from priml.model.attention.kernel import SdpaFused, attention_kernel_cost
from priml.model.custom_types import (
    AttentionKernel,
    ChannelsInOutConfig,
    ChannelsOut,
    TensorModule,
    infer_same_width,
)
from priml.model.embedding import Embedding
from priml.model.ggnn import GGNN, GraphBatch
from priml.model.graphormer import Graphormer
from priml.model.init import InitFn, call_init
from priml.model.linear import Linear
from priml.model.norm import LayerNorm
from priml.model.sequential import Sequential
from priml.model.special import Activation, Dropout
from priml.model.state_dict import validate_state_dict
from priml.model.swiglu import SwiGLU, gelu, relu
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.model.custom_types import LayerCache


class ReleasedModel(nn.Module, ABC):
    """Load a complete released schema through an architecture-owned key map."""

    @abstractmethod
    def checkpoint_names(self) -> dict[str, str]:
        """Return every released-to-native key in released registration order."""

    def released_state(self) -> dict[str, Tensor]:
        """Return every native tensor under its released checkpoint name."""
        state = self.state_dict()
        return {
            released: state[native]
            for released, native in self.checkpoint_names().items()
        }

    def load_released_state(self, state: Mapping[str, Tensor]) -> None:
        """Validate the complete released schema before changing any weight.

        Args:
          state: All released tensors, including persistent numerical buffers.

        Raises:
          ValueError: A key or shape differs from the declared architecture.

        """
        names = self.checkpoint_names()
        if len(set(names.values())) != len(names):
            raise ValueError("Checkpoint mapping has duplicate native tensor names.")
        if set(state) != set(names):
            raise ValueError(
                f"Checkpoint keys differ: missing={sorted(set(names) - set(state))}, "
                f"unexpected={sorted(set(state) - set(names))}.",
            )
        current = self.state_dict()
        mapped = {native: state[name] for name, native in names.items()}
        validate_state_dict(mapped, expected=current)
        _ = self.load_state_dict(mapped)


def feed_forward() -> Sequential.Config:
    """Return the released affine/GELU/affine feed-forward and its residual dropout.

    Returns:
      config: SwiGLU without a gate, then one dropout on the residual update.

    """
    swiglu = SwiGLU.Config(gate=False, act=gelu, round_to=1)
    swiglu.up_proj = Linear.Config(bias=True, init_bias=None)
    swiglu.middle = Dropout.Config(p=0.1)
    swiglu.down_proj = Linear.Config(bias=True, init_bias=None)
    ffn = Sequential.Config()
    ffn.elements = [swiglu, Dropout.Config(p=0.1)]
    return ffn


class PeakAttention(nn.Module):
    """Use shared attention projections/kernel with the released residual dropout."""

    class Config(Fig["PeakAttention"]):
        channels_in: int = -1
        """Residual-stream width entering attention; -1 reads it from the kernel."""

        channels_out: int = -1
        """Residual-stream width leaving attention; -1 reads it from the kernel."""

        _: KW_ONLY

        attention: Attention.Config = field(
            default_factory=lambda: Attention.Config(
                proj_qkv=Linear.Config(bias=True, init_bias=None),
                proj_out=Linear.Config(bias=True, init_bias=None),
            ),
        )
        """Shared geometry, projections and independently injectable kernel."""

        drop: Dropout.Config = field(default_factory=lambda: Dropout.Config(p=0.1))
        """Drop the projected residual update."""

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            if self.attention.channels_in == -1:
                self.attention.channels_in = self.channels_in
            if self.attention.channels_out == -1:
                self.attention.channels_out = self.channels_out
            self.drop.channels_in = (
                self.attention.channels_in
                if self.attention.channels_in != -1
                else self.attention.channels_out
            )
            self = super().finalize()
            self.channels_in = self.channels_out = self.attention.channels_in
            return self

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count shared attention and residual dropout without a decoding cache.

            Args:
              seq_len: Peak tokens per spectrum.
              batch_size: Spectra per invocation.
              dtype: Activation dtype; None uses the torch default.
              **kwargs: Shape and policy messages forwarded to both children.

            Returns:
              execution: Complete attention and residual-dropout cost.

            """
            return replace(
                cost(
                    self.attention,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                ),
                bytes_state=0,
            ) + cost(
                self.drop,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention = config.attention.make()
        self.drop = config.drop.make()

    def reset_parameters(self) -> None:
        """Reset the actual injected projections in their registration order."""
        self.attention.reset_parameters()
        self.drop.reset_parameters()

    @override
    def forward(
        self,
        input: Tensor,
        *,
        memory: Tensor | None = None,
        cache: LayerCache | None = None,
        positions: Tensor | None = None,
        cos_sin: tuple[Tensor, Tensor] | None = None,
        dropout_p: float | None = None,
        is_causal: bool | None = None,
        attn_mask: Tensor | None = None,
        window: int = -1,
        **kwargs: object,
    ) -> Tensor:
        return self.drop(
            self.attention.forward(
                input,
                memory=memory,
                cache=cache,
                positions=positions,
                cos_sin=cos_sin,
                dropout_p=dropout_p,
                is_causal=is_causal,
                attn_mask=attn_mask,
                window=window,
                **kwargs,
            ),
        )


class SinEmb(nn.Module):
    """Embed scalar masses in the reference's logarithmic wavelength basis."""

    class Config(Fig["SinEmb"]):
        channels_out: int = -1
        """Twice the number of sinusoidal wavelengths; -1 inherits the owning width."""

        lo: float = -2.0
        """Base-ten logarithm of the smallest wavelength."""

        hi: float = 3.3
        """Base-ten logarithm of the largest wavelength."""

        power: float = 1.0
        """Exponent on the unit-interval wavelength coordinates."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count angle multiplication and both sinusoidal branches and their adjoints.

            Args:
              seq_len: Scalar masses per spectrum.
              batch_size: Spectra in the invocation.
              dtype: Operand dtype; None uses torch's default.
              **kwargs: Unread cost bus.

            Returns:
              execution: Complete primal/adjoint cost and persistent wavelength bytes.

            """
            del kwargs
            half = self.channels_out // 2
            rows = seq_len * batch_size
            return (
                elementwise_cost(
                    primal=3 * rows * half,
                    adjoint=7 * rows * half,
                    channels=half,
                    rows=rows,
                    inputs=3,
                    outputs=4,
                    adjoint_inputs=8,
                    adjoint_outputs=4,
                    dtype=dtype,
                )
                + reduction_cost(
                    input_elements=rows * half,
                    output_groups=rows,
                    dtype=dtype,
                    phase="adjoint",
                )
                + Cost(bytes_state=nbytes(half, None))
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        wav = torch.pow(
            10.0,
            (config.hi - config.lo)
            * torch.pow(torch.linspace(0, 1, config.channels_out // 2), config.power)
            + config.lo,
        )
        self.register_buffer("inv", (2 * math.pi) / wav)
        self.inv: Tensor

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Append sinusoidal wavelength features to each scalar input."""
        angle = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(angle), torch.cos(angle)], -1)


class Block(TransformerBlock):
    """Configure a Priml pre-norm transformer with both residual dropout sites."""

    class Config(Makes["Block"], TransformerBlock.Config, kw_only=False):
        attn: Makeable[TensorModule] = field(default_factory=PeakAttention.Config)
        """Packed affine projections and the fused SDPA kernel."""

        ffn: Makeable[TensorModule] = field(default_factory=feed_forward)
        """Affine, GELU, hidden dropout, affine, then the residual dropout."""

        norm1: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Normalization preceding attention."""

        norm2: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Normalization preceding the feed-forward branch."""


def fingerprint_head() -> SwiGLU.Config:
    """Return the released fingerprint head: affine, GELU, dropout, affine.

    Returns:
      config: Ungated SwiGLU with the dropout between its two affine maps.

    """
    head = SwiGLU.Config(gate=False, act=gelu, channels_hidden=2_048)
    head.up_proj = Linear.Config(bias=True, init_bias=None)
    head.middle = Dropout.Config(p=0.1)
    head.down_proj = Linear.Config(bias=True, init_bias=None)
    return head


def element_head() -> SwiGLU.Config:
    """Return the released element-count head: affine, GELU, affine.

    Returns:
      config: Ungated SwiGLU regressing C, H, N, O, P, S, F, Cl, Br and I.

    """
    head = SwiGLU.Config(channels_out=10, gate=False, act=gelu, channels_hidden=256)
    head.up_proj = Linear.Config(bias=True, init_bias=None)
    head.down_proj = Linear.Config(bias=True, init_bias=None)
    return head


class FPNet(ReleasedModel):
    """Predict molecular fingerprint logits and log element counts from spectra."""

    class Config(Fig["FPNet"]):
        channels_out: int = 10_226
        """Selected fingerprint bits; verified against the checkpoint on loading."""

        channels_hidden: int = 512
        """Shared mass-embedding and residual-stream width."""

        num_layers: int = 8
        """Copies of the transformer block."""

        block: Block.Config = field(default_factory=Block.Config)
        """Attention and feed-forward recipe, including heads and dropout."""

        mz_emb: SinEmb.Config = field(default_factory=SinEmb.Config)
        """Peak mass-to-charge sinusoidal embedding."""

        nl_emb: SinEmb.Config = field(default_factory=SinEmb.Config)
        """Neutral-loss sinusoidal embedding."""

        prec_emb: SinEmb.Config = field(default_factory=SinEmb.Config)
        """Precursor mass-to-charge sinusoidal embedding."""

        ce_emb: SinEmb.Config = field(
            default_factory=lambda: SinEmb.Config(lo=0.0, hi=2.2),
        )
        """Collision-energy embedding, a quarter of the residual width."""

        int_emb: Linear.Config = field(
            default_factory=lambda: Linear.Config(
                channels_in=2,
                bias=True,
                init_bias=None,
            ),
        )
        """Embed each peak's square-root intensity and its square."""

        pk: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Fuse the mass, neutral-loss and intensity embeddings of each peak."""

        adduct: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(
                60,
                init_weight=nn.init.normal_,
            ),
        )
        """Adduct vocabulary from the pinned reference's ADDUCT_LIST."""

        gl: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Project precursor and collision-energy features to the global token."""

        norm: LayerNorm.Config = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Final token normalization."""

        head: SwiGLU.Config = field(default_factory=fingerprint_head)
        """Fingerprint logits from the global token and the mean peak token."""

        head_elements: SwiGLU.Config = field(default_factory=element_head)
        """Element counts from the global token and the mean peak token."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count every embedding, global token, transformer and both prediction heads.

            Args:
              seq_len: Peak slots before prepending the global token.
              batch_size: Spectra in the invocation.
              dtype: Operand dtype; None uses torch's default.
              **kwargs: Cost bus forwarded to the model's children.

            Returns:
              execution: Full fingerprint and element-head primal/adjoint cost.

            """
            width = self.channels_hidden
            context = self.gl.channels_in
            rows = seq_len * batch_size
            total = cost(
                self.block,
                seq_len=seq_len + 1,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ).tile(self.num_layers, copies=self.num_layers) + cost(
                self.norm,
                seq_len=seq_len + 1,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            for child, positions in (
                (self.mz_emb, seq_len),
                (self.nl_emb, seq_len),
                (self.prec_emb, 1),
                (self.ce_emb, 1),
                (self.adduct, 1),
                (self.int_emb, seq_len),
                (self.pk, seq_len),
                (self.gl, 1),
                (self.head, 1),
                (self.head_elements, 1),
            ):
                total += cost(
                    child,
                    seq_len=positions,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            selected = 2 * (
                3 * rows * width
                + batch_size * context
                + batch_size * (seq_len + 1) * width
            )
            return (
                total
                + elementwise_cost(
                    primal=3 * rows + 4 * batch_size + batch_size * width,
                    adjoint=4 * rows + 4 * batch_size + batch_size * width,
                    channels=1,
                    rows=rows,
                    inputs=5,
                    outputs=4,
                    dtype=dtype,
                )
                + traffic("primal", kernel="selection", elements=selected, dtype=dtype)
                + traffic("adjoint", kernel="selection", elements=selected, dtype=dtype)
                + reduction_cost(
                    input_elements=rows * (width + 1),
                    output_groups=batch_size * (width + 1),
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=rows * width + batch_size * (width + 1),
                    adjoint=3 * rows * width,
                    channels=width,
                    rows=rows,
                    inputs=3,
                    outputs=2,
                    dtype=dtype,
                )
            )

        @override
        def finalize(self) -> Self:
            width = self.channels_hidden
            if self.block.channels_in == -1:
                self.block.channels_in = width
            for embedding in (self.mz_emb, self.nl_emb, self.prec_emb):
                if embedding.channels_out == -1:
                    embedding.channels_out = width
            if self.ce_emb.channels_out == -1:
                self.ce_emb.channels_out = width // 4
            if self.int_emb.channels_out == -1:
                self.int_emb.channels_out = width
            if self.pk.channels_in == -1:
                self.pk.channels_in = (
                    self.mz_emb.channels_out
                    + self.nl_emb.channels_out
                    + self.int_emb.channels_out
                )
            if self.pk.channels_out == -1:
                self.pk.channels_out = width
            if self.adduct.channels_out == -1:
                self.adduct.channels_out = width
            if self.gl.channels_in == -1:
                # Precursor and collision-energy embeddings, then four scalar features.
                self.gl.channels_in = (
                    self.prec_emb.channels_out + self.ce_emb.channels_out + 4
                )
            if self.gl.channels_out == -1:
                self.gl.channels_out = width
            if self.norm.channels_in == -1:
                self.norm.channels_in = width
            for head in (self.head, self.head_elements):
                if head.channels_in == -1:
                    head.channels_in = 2 * width
            if self.head.channels_out == -1:
                self.head.channels_out = self.channels_out
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_hidden < 8 or config.channels_hidden % 8:
            raise ValueError("channels_hidden must be a positive multiple of eight.")
        self.mz_emb = config.mz_emb.make()
        self.nl_emb = config.nl_emb.make()
        self.int_emb = config.int_emb.make()
        self.pk = config.pk.make()
        self.prec_emb = config.prec_emb.make()
        self.adduct = config.adduct.make()
        self.ce_emb = config.ce_emb.make()
        self.gl = config.gl.make()
        self.blocks = nn.ModuleList(
            [config.block.make() for _ in range(config.num_layers)],
        )
        self.norm = config.norm.make()
        self.head = config.head.make()
        self.head_elements = config.head_elements.make()

    @override
    def checkpoint_names(self) -> dict[str, str]:
        """Map every released tensor name to one native tensor, without reshaping.

        Returns:
          names: Complete released-to-native key map in released registration order.

        """
        names: dict[str, str] = {"mz_emb.inv": "mz_emb.inv", "nl_emb.inv": "nl_emb.inv"}
        for prefix in ("int_emb", "pk"):
            for suffix in ("weight", "bias"):
                names[f"{prefix}.{suffix}"] = f"{prefix}.{suffix}"
        names["prec_emb.inv"] = "prec_emb.inv"
        names["ad.weight"] = "adduct.weight"
        names["ce_emb.inv"] = "ce_emb.inv"
        for suffix in ("weight", "bias"):
            names[f"gl.{suffix}"] = f"gl.{suffix}"
        for index in range(len(self.blocks)):
            for released, native in (
                ("n1", "norm1"),
                ("qkv", "attn.attention.proj_qkv"),
                ("o", "attn.attention.proj_out"),
                ("n2", "norm2"),
                ("ff.0", "ffn.0.up_proj"),
                ("ff.3", "ffn.0.down_proj"),
            ):
                for suffix in ("weight", "bias"):
                    names[f"blocks.{index}.{released}.{suffix}"] = (
                        f"blocks.{index}.{native}.{suffix}"
                    )
        for released, native in (
            ("norm", "norm"),
            ("head.0", "head.up_proj"),
            ("head.3", "head.down_proj"),
            ("elem_head.0", "head_elements.up_proj"),
            ("elem_head.2", "head_elements.down_proj"),
        ):
            for suffix in ("weight", "bias"):
                names[f"{released}.{suffix}"] = f"{native}.{suffix}"
        if set(names.values()) != set(self.state_dict()):
            raise ValueError("Native FPNet state does not match its checkpoint schema.")
        return names

    def encode(
        self,
        mz: Tensor,
        *,
        it: Tensor,
        pad: Tensor,
        prec: Tensor,
        ad: Tensor,
        ce: Tensor,
        ce_known: Tensor,
        merged: Tensor,
        mode: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return normalized tokens and their mask, including the global token.

        Args:
          mz: Peak masses, with batch and peak axes.
          it: Square-root peak intensities.
          pad: True at padded peak positions.
          prec: Precursor mass-to-charge per spectrum.
          ad: Reference adduct-vocabulary indices.
          ce: Collision energies in electron volts.
          ce_known: Indicators for measured collision energies.
          merged: Number of merged spectra divided by four.
          mode: Positive one or negative one for ionization polarity.

        Returns:
          tokens: Normalized global and peak states.
          padding: Mask including the unmasked global token.

        """
        neutral_loss = (prec[:, None] - mz).clamp(min=0)
        peaks = self.pk(
            torch.cat(
                [
                    self.mz_emb(mz),
                    self.nl_emb(neutral_loss),
                    self.int_emb(torch.stack([it, it * it], -1)),
                ],
                -1,
            ),
        )
        global_token = self.gl(
            torch.cat(
                [
                    self.prec_emb(prec),
                    self.ce_emb(ce / 10.0 + 1.0),
                    ce_known.unsqueeze(-1),
                    merged.unsqueeze(-1),
                    mode.unsqueeze(-1),
                    torch.log1p(prec).unsqueeze(-1) / 10.0,
                ],
                -1,
            ),
        ) + self.adduct(ad)
        x = torch.cat([global_token.unsqueeze(1), peaks], 1)
        pad = torch.cat(
            [
                torch.zeros(mz.shape[0], 1, dtype=torch.bool, device=pad.device),
                pad,
            ],
            1,
        )
        for block in self.blocks:
            x = block(x, attn_mask=(~pad)[:, None, None, :])
        return self.norm(x), pad

    @override
    def forward(
        self,
        mz: Tensor,
        it: Tensor,
        pad: Tensor,
        prec: Tensor,
        ad: Tensor,
        ce: Tensor,
        ce_known: Tensor,
        merged: Tensor,
        mode: Tensor,
        **batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict fingerprint logits and log element counts for each spectrum.

        Args:
          mz: Batched peak masses.
          it: Square-root relative peak intensities.
          pad: True at padded peak positions.
          prec: Precursor mass-to-charge per spectrum.
          ad: Reference adduct-vocabulary indices.
          ce: Collision energies in electron volts.
          ce_known: Indicators for measured collision energies.
          merged: Number of merged spectra divided by four.
          mode: Positive one or negative one for ionization polarity.
          **batch: Unread targets and identities that a training batch carries.

        Returns:
          fingerprints: Unnormalized scores for the selected fingerprint bits.
          elements: Predicted log element counts.

        """
        del batch
        x, pad = self.encode(
            mz,
            it=it,
            pad=pad,
            prec=prec,
            ad=ad,
            ce=ce,
            ce_known=ce_known,
            merged=merged,
            mode=mode,
        )
        mask = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * mask).sum(1) / mask.sum(1).clamp(min=1)
        embedding = torch.cat([x[:, 0], mean], -1)
        return self.head(embedding), self.head_elements(embedding)


@dataclass(slots=True, kw_only=True)
class WidthNormal:
    """Draw from a normal of variance 2 / (5 * width), the released projection scale."""

    width: int = -1
    """Residual width; -1 inherits it from the owning module."""

    def __call__(self, tensor: Tensor) -> Tensor:
        """Overwrite ``tensor`` with the released draw.

        Args:
          tensor: Parameter initialized in place.

        Returns:
          tensor: The same, initialized parameter.

        """
        return nn.init.normal_(tensor, std=(2 / (5 * self.width)) ** 0.5)


def rectified_stack(
    num_layers: int,
    channels_hidden: int,
    dropout: float,
) -> Sequential.Config:
    """Return affine layers with hidden dropout and a rectifier after each one.

    Args:
      num_layers: Number of affine projections.
      channels_hidden: Width between successive projections.
      dropout: Probability between a hidden projection and its rectifier.

    Returns:
      stack: Elements whose first input and last output widths the owner pushes.

    """
    elements: list[Makeable[TensorModule]] = []
    for index in range(num_layers):
        last = index == num_layers - 1
        elements.append(
            Linear.Config(
                channels_out=-1 if last else channels_hidden,
                bias=True,
                init_bias=None,
            ),
        )
        if not last:
            elements.append(Dropout.Config(p=dropout))
        elements.append(Activation.Config(act=relu))
    return Sequential.Config(elements=elements)


def size_stack(
    stack: Sequential.Config,
    *,
    channels_in: int,
    channels_out: int,
) -> None:
    """Size the first and last projections of a rectified stack where unset.

    The stack ends in a rectifier, so its owner cannot push the output width
    through the sequence; the final projection receives it directly.

    Args:
      stack: Stack made by ``rectified_stack`` or an edit of it.
      channels_in: Width entering the first projection.
      channels_out: Width leaving the final projection.

    """
    assert isinstance(stack.elements, list)
    projections = [
        element for element in stack.elements if isinstance(element, Linear.Config)
    ]
    if projections[0].channels_in == -1:
        projections[0].channels_in = channels_in
    if projections[-1].channels_out == -1:
        projections[-1].channels_out = channels_out


class FourierFeatures(nn.Module):
    """Build the fractional and integer reciprocal-mass Fourier basis."""

    class Config(Fig["FourierFeatures"]):
        num_fractional: int = 4_999
        """Even reciprocal fractional-mass frequencies."""

        num_integer: int = 998
        """Reciprocal integer-mass frequencies."""

        resolution: float = 1e-4
        """Mass resolution setting the fractional wavelengths."""

        channels_out: int = -1
        """Derived concatenated cosine and sine width."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price the fixed basis with an input-only matrix adjoint.

            Args:
              seq_len: Scalar masses per example.
              batch_size: Examples in the invocation.
              dtype: Input dtype before the explicit precision promotion.
              **kwargs: Unused surrounding cost arguments.

            Returns:
              execution: Scaling, frozen matrix expansion, trig features and buffer.

            """
            del kwargs
            rows = seq_len * batch_size
            width = self.num_fractional + self.num_integer
            precise = (
                torch.float64
                if resolve_dtype(dtype) == torch.float64
                else torch.float32
            )
            product = matmul_cost(
                channels_in=1,
                channels_out=width,
                rows=rows,
                weight=False,
                dtype=precise,
            ).only("primal")
            return (
                product
                + product.relabel("adjoint")
                + Cost(bytes_state=nbytes(width, torch.get_default_dtype()))
                + elementwise_cost(
                    primal=rows,
                    adjoint=rows,
                    channels=1,
                    rows=rows,
                    dtype=precise,
                )
                + elementwise_cost(
                    primal=2 * rows * width,
                    adjoint=6 * rows * width,
                    channels=width,
                    rows=rows,
                    outputs=2,
                    adjoint_inputs=3,
                    dtype=precise,
                )
                + traffic(
                    "primal",
                    kernel="selection",
                    elements=4 * rows * width,
                    dtype=precise,
                )
                + traffic(
                    "adjoint",
                    kernel="selection",
                    elements=4 * rows * width,
                    dtype=precise,
                )
            )

        @override
        def finalize(self) -> Self:
            self.channels_out = 2 * (self.num_fractional + self.num_integer)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        frequency = [
            1 / (config.resolution * index)
            for index in range(2, 2 * (config.num_fractional + 1), 2)
        ] + [1 / index for index in range(2, config.num_integer + 2)]
        self.register_buffer("b", torch.tensor(frequency).unsqueeze(0))
        self.b: Tensor

    @override
    def forward(self, input: Tensor) -> Tensor:
        with torch.autocast(device_type=input.device.type, enabled=False):
            phase = (2 * torch.pi * precise_float(input)) @ self.b
            return torch.cat([phase.cos(), phase.sin()], dim=-1)


class DreamKernel(nn.Module):
    """Apply padded-query zeroing and source precision around shared attention."""

    class Config(Fig["DreamKernel"]):
        dtype: torch.dtype | None = torch.float32
        """Arithmetic dtype the operands are promoted to, apart from projection
        autocast; None keeps them in the projection dtype."""

        kernel: Makeable[AttentionKernel] = field(default_factory=SdpaFused.Config)
        """Shared sequence/head kernel, including its independent implementation."""

        def cost(self, **kwargs: object) -> Cost:
            """Delegate the projected head geometry to the configured kernel."""
            return cost(self.kernel, **kwargs)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dtype = config.dtype
        self.kernel = config.kernel.make()

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        query_mask: Tensor,
        attn_mask: Tensor,
        dropout_p: float = 0.0,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        q = q.masked_fill(query_mask.movedim(-3, -2), 0.0)
        dtype = self.dtype
        if dtype is not None:
            with torch.autocast(device_type=q.device.type, enabled=False):
                return self.kernel(
                    _promote(q, dtype),
                    _promote(k, dtype),
                    _promote(v, dtype),
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                )
        return self.kernel(
            q,
            k,
            v,
            attn_mask=attn_mask.to(q.dtype),
            dropout_p=dropout_p,
        )


def _promote(input: Tensor, dtype: torch.dtype) -> Tensor:
    return input.to(torch.promote_types(input.dtype, dtype))


class DreamAttention(nn.Module):
    """Use shared projections with one packed source initialization draw."""

    class Config(Fig["DreamAttention"]):
        channels_in: int = -1
        """Residual-stream width entering attention; -1 reads it from the kernel."""

        channels_out: int = -1
        """Residual-stream width leaving attention; -1 reads it from the kernel."""

        _: KW_ONLY

        attention: Attention.Config = field(
            default_factory=lambda: Attention.Config(
                dropout=0.1,
                proj_qkv=Linear.Config(init_weight=nn.init.zeros_),
                proj_out=Linear.Config(init_weight=nn.init.zeros_),
                attn_kernel=DreamKernel.Config(),
            ),
        )
        """Shared projections and explicit query-mask/precision kernel."""

        dropout: Dropout.Config = field(default_factory=lambda: Dropout.Config(p=0.1))
        """Drop the projected residual update."""

        @property
        def precision(self) -> DreamKernel.Config:
            """Expose the released precision policy for checkpoint interpretation."""
            kernel = self.attention.attn_kernel
            if not isinstance(kernel, DreamKernel.Config):
                raise TypeError("Released Dream precision requires DreamKernel.Config.")
            return kernel

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            if self.attention.channels_in == -1:
                self.attention.channels_in = self.channels_in
            if self.attention.channels_out == -1:
                self.attention.channels_out = self.channels_out
            self.dropout.channels_in = (
                self.attention.channels_in
                if self.attention.channels_in != -1
                else self.attention.channels_out
            )
            self = super().finalize()
            self.channels_in = self.channels_out = self.attention.channels_in
            return self

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Compose projection precision and the explicitly promoted masked kernel.

            Args:
              seq_len: Spectral tokens per example.
              batch_size: Spectra in the invocation.
              dtype: Projection activation dtype.
              **kwargs: Unused surrounding cost arguments.

            Returns:
              execution: Complete attention without an unused decoding cache.

            """
            del kwargs
            attention = self.attention
            kernel_dtype = (
                dtype
                if self.precision.dtype is None
                else torch.promote_types(resolve_dtype(dtype), self.precision.dtype)
            )
            elements = (
                batch_size * seq_len * attention.num_heads * attention.channels_head
            )
            total = (
                AttentionProjections.Config.cost(
                    attention,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.precision,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_heads=attention.num_heads,
                    channels_head=attention.channels_head,
                    channels_v_head=attention.channels_head,
                    dropout_p=attention.dropout,
                    dtype=kernel_dtype,
                )
                + elementwise_cost(
                    primal=0,
                    adjoint=0,
                    channels=attention.num_heads * attention.channels_head,
                    rows=batch_size * seq_len,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    kernel="elementwise",
                    elements=batch_size * seq_len,
                    dtype=torch.bool,
                )
                + cost(
                    self.dropout,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            )
            if resolve_dtype(kernel_dtype) != resolve_dtype(dtype):
                for phase in ("primal", "adjoint"):
                    total += traffic(
                        phase,
                        kernel="elementwise",
                        elements=4 * elements,
                        dtype=dtype,
                    )
                    total += traffic(
                        phase,
                        kernel="elementwise",
                        elements=4 * elements,
                        dtype=kernel_dtype,
                    )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention = config.attention.make()
        self.dropout = config.dropout.make()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Draw every projection value in one source-ordered packed matrix."""
        width = self.attention.proj_qkv.weight.shape[1]
        packed = torch.empty(4 * width, width)
        _ = WidthNormal(width=width)(packed)
        with torch.no_grad():
            _ = self.attention.proj_qkv.weight.copy_(packed[: 3 * width])
            _ = self.attention.proj_out.weight.copy_(packed[3 * width :])

    @override
    def forward(
        self,
        input: Tensor,
        *,
        bias: Tensor,
        query_mask: Tensor,
    ) -> Tensor:
        attention: TensorModule = self.attention
        return self.dropout(attention(input, attn_mask=bias, query_mask=query_mask))


def released_swiglu() -> SwiGLU.Config:
    """Return the released bias-free expansion, rectifier, dropout and contraction.

    Returns:
      config: Ungated SwiGLU whose projections keep Priml's default initialization.

    """
    ffn = SwiGLU.Config(gate=False, act=relu, round_to=1)
    ffn.up_proj = Linear.Config()
    ffn.middle = Dropout.Config(p=0.1)
    ffn.down_proj = Linear.Config()
    return ffn


class DreamFFN(nn.Module):
    """Redraw both feed-forward projections in the released second pass."""

    class Config(Fig["DreamFFN"]):
        channels_in: int = -1
        """Residual width."""

        ffn: SwiGLU.Config = field(default_factory=released_swiglu)
        """Expansion, rectifier, hidden dropout and contraction."""

        init_final: InitFn = field(default_factory=WidthNormal)
        """Second initialization applied to both projection weights."""

        def cost(self, **kwargs: object) -> Cost:
            """Delegate the complete invocation to the feed-forward.

            Args:
              **kwargs: Whole sequence geometry and activation dtype.

            Returns:
              execution: Both projections and the expanded hidden operations.

            """
            return cost(self.ffn, **kwargs)

        @override
        def finalize(self) -> Self:
            if self.ffn.channels_in == -1:
                self.ffn.channels_in = self.channels_in
            if isinstance(self.init_final, WidthNormal) and self.init_final.width == -1:
                self.init_final.width = self.channels_in
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ffn = config.ffn.make()
        self.init_final = config.init_final
        self._redraw()

    def reset_parameters(self) -> None:
        """Repeat the constructor's default draws and then the released second pass."""
        self.ffn.reset_parameters()
        self._redraw()

    def _redraw(self) -> None:
        call_init(self.init_final, self.ffn.up_proj.weight)
        call_init(self.init_final, self.ffn.down_proj.weight)

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return self.ffn(input)


def precise_float(input: Tensor) -> Tensor:
    """Use fp32 arithmetic while preserving an existing exact-math fp64 input."""
    return input if input.dtype == torch.float64 else input.float()


def released_ffn() -> Sequential.Config:
    """Return the released feed-forward followed by its residual-update dropout.

    Returns:
      config: The second-pass feed-forward, then one dropout on the update.

    """
    elements: list[Makeable[TensorModule]] = [
        DreamFFN.Config(),
        Dropout.Config(p=0.1),
    ]
    return Sequential.Config(elements=elements)


def released_block() -> TransformerBlock.Config:
    """Return the released pre-norm attention and feed-forward residual pair.

    Returns:
      config: Mass-biased attention and the released feed-forward, each normalized
        beforehand and followed by its own residual-update dropout.

    """
    block = TransformerBlock.Config(prenorm=True)
    block.attn = DreamAttention.Config()
    block.ffn = released_ffn()
    block.norm1 = LayerNorm.Config(elementwise_affine=True)
    block.norm2 = LayerNorm.Config(elementwise_affine=True)
    return block


class DreamEncoder(Transformer):
    """Drop the input, apply the residual pairs and normalize the final tokens."""

    class Config(Makes["DreamEncoder"], Transformer.Config, kw_only=False):
        num_layers: int = 7
        """Attention/feed-forward pairs."""

        proj_in: Makeable[TensorModule] | None = field(
            default_factory=lambda: Dropout.Config(p=0.1),
        )
        """Input dropout."""

        block: ChannelsInOutConfig | list[ChannelsInOutConfig] = field(
            default_factory=released_block,
        )
        """Residual pair template broadcast over the layers."""

        proj_out: Makeable[TensorModule] | None = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Affine final normalization."""


class DreamBackbone(nn.Module):
    """Embed raw mass/intensity pairs and use Fourier sums as attention key bias."""

    class Config(Fig["DreamBackbone"]):
        channels_fourier: int = 980
        """Projected Fourier feature width."""

        channels_peak: int = 44
        """Projected normalized mass/intensity feature width."""

        max_mass: float = 1_000.0
        """Mass normalization denominator for peak features."""

        fourier: Sequential.Config = field(
            default_factory=lambda: rectified_stack(5, 512, 0.1),
        )
        """Fourier-feature rectified affine stack."""

        peak: Sequential.Config = field(
            default_factory=lambda: rectified_stack(1, 44, 0.1),
        )
        """Normalized mass/intensity rectified affine stack."""

        basis: FourierFeatures.Config = field(default_factory=FourierFeatures.Config)
        """Fixed fractional and integer reciprocal-mass Fourier basis."""

        encoder: DreamEncoder.Config = field(default_factory=DreamEncoder.Config)
        """Pre-norm attention and feed-forward residual stack."""

        chunk_size: int | None = 4_096
        """Flattened tokens per Fourier projection chunk; None projects them all at once."""

        checkpoint_fourier: bool = True
        """Recompute each Fourier chunk during gradient-enabled training."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price every Fourier chunk with one shared parameter and buffer owner.

            Args:
              seq_len: Peak and precursor tokens per spectrum.
              batch_size: Spectra in the invocation.
              dtype: Peak-projection activation dtype.
              **kwargs: Additional encoder cost arguments.

            Returns:
              execution: Complete backbone; checkpoint recomputation is excluded
                under Priml's ordinary MFU cost convention.

            """
            rows = batch_size * seq_len
            precise = (
                torch.float64
                if resolve_dtype(dtype) == torch.float64
                else torch.float32
            )
            chunk_size = rows if self.chunk_size is None else self.chunk_size
            total = Cost()
            for start in range(0, rows, chunk_size):
                chunk = min(chunk_size, rows - start)
                piece = cost(
                    self.basis,
                    seq_len=chunk,
                    batch_size=1,
                    dtype=precise,
                ) + cost(self.fourier, seq_len=chunk, batch_size=1, dtype=precise)
                if start:
                    total += replace(piece, params=0, params_active=0, bytes_state=0)
                    total += elementwise_cost(
                        primal=0,
                        adjoint=piece.params,
                        channels=piece.params,
                        rows=1,
                        dtype=precise,
                    )
                else:
                    total += piece
            width = self.channels_peak + self.channels_fourier
            bus = {
                **kwargs,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "dtype": dtype,
            }
            return (
                total
                + cost(self.peak, **bus)
                + cost(self.encoder, **bus)
                + elementwise_cost(
                    primal=2 * rows + batch_size * width + rows,
                    adjoint=2 * rows + batch_size * width + rows,
                    channels=2,
                    rows=rows,
                    dtype=precise,
                )
                + reduction_cost(
                    input_elements=rows * self.channels_fourier,
                    output_groups=rows,
                    dtype=precise,
                )
                + traffic(
                    "adjoint",
                    kernel="reduction",
                    elements=rows * (self.channels_fourier + 1),
                    dtype=precise,
                )
                + traffic(
                    "primal",
                    kernel="selection",
                    elements=2
                    * (rows + rows * self.channels_fourier + 2 * rows * width),
                    dtype=precise,
                )
                + traffic(
                    "adjoint",
                    kernel="selection",
                    elements=2
                    * (rows + rows * self.channels_fourier + 2 * rows * width),
                    dtype=precise,
                )
                + elementwise_cost(
                    primal=0,
                    adjoint=0,
                    channels=seq_len,
                    rows=rows,
                    dtype=precise,
                )
                + traffic(
                    "primal",
                    kernel="elementwise",
                    elements=rows * (seq_len + 1),
                    dtype=torch.bool,
                )
            )

        @override
        def finalize(self) -> Self:
            size_stack(
                self.fourier,
                channels_in=self.basis.copy_tree().finalize().channels_out,
                channels_out=self.channels_fourier,
            )
            size_stack(self.peak, channels_in=2, channels_out=self.channels_peak)
            if self.encoder.channels_in == -1:
                self.encoder.channels_in = self.channels_fourier + self.channels_peak
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.basis = config.basis.make()
        self.fourier: Sequential = config.fourier.make()
        self.peak: Sequential = config.peak.make()
        self.encoder: DreamEncoder = config.encoder.make()

    @override
    def forward(self, input: Tensor, *, condition: Tensor) -> Tensor:
        """Condition the precursor token and retain padded keys in mass-biased attention.

        Args:
          input: Mass/intensity pairs including the precursor token.
          condition: Additive precursor context from adduct, polarity and CE.

        Returns:
          tokens: Normalized peak and precursor representations.

        """
        spectrum = precise_float(input)
        mass = spectrum[..., [0]]
        flat = mass.reshape(-1, 1)
        chunk_size = (
            len(flat) if self.config.chunk_size is None else self.config.chunk_size
        )
        chunks: list[Tensor] = []
        for start in range(0, len(flat), chunk_size):
            chunk = flat[start : start + chunk_size]
            chunks.append(
                checkpoint(self._fourier_chunk, chunk, use_reentrant=False)
                if self.config.checkpoint_fourier
                and self.training
                and torch.is_grad_enabled()
                else self._fourier_chunk(chunk),
            )
        frequencies = (chunks[0] if len(chunks) == 1 else torch.cat(chunks)).view(
            *mass.shape[:-1],
            -1,
        )
        peak = self.peak(spectrum / spectrum.new_tensor([self.config.max_mass, 1.0]))
        tokens = torch.cat([peak.to(frequencies.dtype), frequencies], dim=-1)
        tokens = torch.cat(
            [tokens[:, :1] + condition.to(tokens.dtype)[:, None], tokens[:, 1:]],
            dim=1,
        )
        pad = spectrum[..., 0] == 0
        scores = precise_float(frequencies).sum(-1)
        length = pad.shape[1]
        query_mask = pad[:, None, :, None]
        bias = (-scores)[:, None, None, :].expand(-1, 1, length, length)
        bias = bias.masked_fill(query_mask, 0.0).contiguous()
        return self.encoder.forward(tokens, bias=bias, query_mask=query_mask)

    def _fourier_chunk(self, mass: Tensor) -> Tensor:
        with torch.autocast(device_type=mass.device.type, enabled=False):
            return self.fourier.forward(self.basis(mass))


def condition_mlp() -> Sequential.Config:
    """Return the released collision-energy context: affine, GELU, affine.

    Returns:
      config: Elements sized by the owner's context and residual widths.

    """
    return Sequential.Config(
        elements=[
            Linear.Config(channels_out=256, bias=True, init_bias=None),
            Activation.Config(act=gelu),
            Linear.Config(bias=True, init_bias=None),
        ],
    )


def fingerprint_mlp() -> Sequential.Config:
    """Return the released fingerprint head: affine, GELU, dropout, affine.

    Returns:
      config: Elements sized by the owner's pooled width and vocabulary.

    """
    return Sequential.Config(
        elements=[
            Linear.Config(channels_out=2_048, bias=True, init_bias=None),
            Activation.Config(act=gelu),
            Dropout.Config(p=0.1),
            Linear.Config(bias=True, init_bias=None),
        ],
    )


def element_mlp() -> Sequential.Config:
    """Return the released element head: affine, GELU, affine.

    Returns:
      config: Elements sized by the owner's pooled width and element count.

    """
    return Sequential.Config(
        elements=[
            Linear.Config(channels_out=256, bias=True, init_bias=None),
            Activation.Config(act=gelu),
            Linear.Config(bias=True, init_bias=None),
        ],
    )


class DreamsFP(ReleasedModel):
    """Predict fingerprints and elements from a Fourier spectral backbone."""

    class Config(Fig["DreamsFP"]):
        channels_out: int = 10_226
        """Selected molecular fingerprint vocabulary."""

        num_elements: int = 10
        """Element regression outputs."""

        max_peaks: int = 60
        """Intensity-selected peak slots after the precursor token."""

        negative_mass_shift: float = 2 * 1.007276467
        """Convert negative spectra to the positive precursor mass scale."""

        precursor_intensity: float = 1.1
        """Artificial intensity assigned to the precursor token."""

        backbone: DreamBackbone.Config = field(default_factory=DreamBackbone.Config)
        """Fourier and peak embeddings followed by mass-biased attention."""

        adduct: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(60, init_weight=nn.init.normal_),
        )
        """Adduct context, zeroed after its initialization draw."""

        polarity: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(2, init_weight=nn.init.normal_),
        )
        """Positive/negative context, zeroed after its initialization draw."""

        ce: SinEmb.Config = field(
            default_factory=lambda: SinEmb.Config(channels_out=32, lo=0.0, hi=2.2),
        )
        """Collision-energy sinusoidal features."""

        condition: Sequential.Config = field(default_factory=condition_mlp)
        """Conditioning MLP with a zeroed final projection."""

        head: Sequential.Config = field(default_factory=fingerprint_mlp)
        """Fingerprint head with GELU and dropout."""

        elements: Sequential.Config = field(default_factory=element_mlp)
        """Element head with GELU."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price raw peak preparation, conditioning, backbone and both heads.

            Args:
              seq_len: Raw peak slots before intensity selection.
              batch_size: Spectra in the invocation.
              dtype: Model activation dtype.
              **kwargs: Additional backbone cost arguments.

            Returns:
              execution: Complete fingerprint and element prediction invocation.

            """
            raw = batch_size * seq_len
            kept = batch_size * min(seq_len, self.max_peaks)
            tokens = self.max_peaks + 1
            rows = batch_size * tokens
            width = self.backbone.channels_fourier + self.backbone.channels_peak
            scalar = {"seq_len": 1, "batch_size": batch_size, "dtype": dtype}
            total = (
                cost(self.adduct, **scalar)
                + cost(self.polarity, **scalar)
                + cost(self.ce, **scalar)
                + cost(self.condition, **scalar)
                + cost(
                    self.backbone,
                    seq_len=tokens,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + cost(self.head, **scalar)
                + cost(self.elements, **scalar)
            )
            return (
                total
                + elementwise_cost(
                    primal=2 * batch_size * width
                    + batch_size * (self.ce.channels_out + 2),
                    adjoint=2 * batch_size * width
                    + batch_size * (self.ce.channels_out + 2),
                    channels=width,
                    rows=batch_size,
                    inputs=3,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    kernel="sort",
                    elements=raw + kept,
                    dtype=torch.float32,
                )
                + traffic("primal", kernel="sort", elements=kept, dtype=torch.int64)
                + traffic(
                    "adjoint",
                    kernel="selection",
                    elements=raw + kept,
                    dtype=torch.float32,
                )
                + traffic(
                    "primal",
                    kernel="selection",
                    elements=2 * raw + 4 * kept + 4 * rows,
                    dtype=torch.float32,
                )
                + traffic(
                    "adjoint",
                    kernel="selection",
                    elements=2 * raw + 4 * kept + 4 * rows,
                    dtype=torch.float32,
                )
                + elementwise_cost(
                    primal=raw + 2 * kept,
                    adjoint=2 * raw + 3 * kept,
                    channels=1,
                    rows=max(1, raw),
                    dtype=torch.float32,
                )
                + elementwise_cost(
                    primal=batch_size + kept,
                    adjoint=0,
                    channels=1,
                    rows=batch_size + kept,
                    dtype=torch.float64,
                )
                + traffic(
                    "primal",
                    kernel="elementwise",
                    elements=3 * raw + kept + rows,
                    dtype=torch.bool,
                )
                + reduction_cost(
                    input_elements=kept,
                    output_groups=batch_size,
                    dtype=torch.float32,
                )
                + reduction_cost(
                    input_elements=rows * (width + 1),
                    output_groups=batch_size * (width + 1),
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=rows * width + batch_size * width,
                    adjoint=3 * rows * width + 2 * batch_size * width,
                    channels=width,
                    rows=rows,
                    inputs=3,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    kernel="selection",
                    elements=4 * batch_size * width,
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    kernel="selection",
                    elements=4 * batch_size * width,
                    dtype=dtype,
                )
            )

        @override
        def finalize(self) -> Self:
            width = self.backbone.channels_fourier + self.backbone.channels_peak
            if self.adduct.channels_out == -1:
                self.adduct.channels_out = width
            if self.polarity.channels_out == -1:
                self.polarity.channels_out = width
            for head, channels_in, channels_out in (
                (self.condition, self.ce.channels_out + 2, width),
                (self.head, 2 * width, self.channels_out),
                (self.elements, 2 * width, self.num_elements),
            ):
                if head.channels_in == -1:
                    head.channels_in = channels_in
                if head.channels_out == -1:
                    head.channels_out = channels_out
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.backbone = config.backbone.make()
        self.adduct = config.adduct.make()
        self.polarity = config.polarity.make()
        self.ce = config.ce.make()
        self.condition = config.condition.make()
        # Source draw order: each context is initialized, then zeroed.
        _ = nn.init.zeros_(self.adduct.weight)
        _ = nn.init.zeros_(self.polarity.weight)
        output = self.condition[-1]
        assert isinstance(output, Linear)
        _ = nn.init.zeros_(output.weight)
        if output.bias is None:
            raise ValueError("Expected output.bias is not None.")
        _ = nn.init.zeros_(output.bias)
        self.head = config.head.make()
        self.elements = config.elements.make()

    def checkpoint_parts(self) -> dict[str, tuple[str, ...]]:
        """Map every released tensor to its native tensors, concatenated in order.

        Returns:
          parts: Released schema; only the packed attention matrices hold two parts.

        """
        parts: dict[str, tuple[str, ...]] = {
            "backbone.fourier_enc.b": ("backbone.basis.b",),
        }
        for released, native, mlp in (
            ("backbone.ff_fourier.ff", "backbone.fourier", self.backbone.fourier),
            ("backbone.ff_peak.ff", "backbone.peak", self.backbone.peak),
        ):
            for index, layer in enumerate(mlp):
                if isinstance(layer, Linear):
                    for suffix in ("weight", "bias"):
                        parts[f"{released}.{index}.{suffix}"] = (
                            f"{native}.{index}.{suffix}",
                        )
        source = "backbone.transformer_encoder"
        for index in range(len(self.backbone.encoder.blocks)):
            native = f"backbone.encoder.blocks.{index}"
            parts[f"{source}.atts.{index}.weights"] = (
                f"{native}.attn.attention.proj_qkv.weight",
                f"{native}.attn.attention.proj_out.weight",
            )
            for released, projection in (
                ("in_proj", "up_proj"),
                ("out_proj", "down_proj"),
            ):
                parts[f"{source}.ffs.{index}.{released}.weight"] = (
                    f"{native}.ffn.0.ffn.{projection}.weight",
                )
            for offset, norm in enumerate((f"{native}.norm1", f"{native}.norm2")):
                for suffix in ("weight", "bias"):
                    parts[f"{source}.scales.{2 * index + offset}.{suffix}"] = (
                        f"{norm}.{suffix}",
                    )
        for suffix in ("weight", "bias"):
            parts[
                f"{source}.scales.{2 * len(self.backbone.encoder.blocks)}.{suffix}"
            ] = (f"backbone.encoder.proj_out.{suffix}",)
        parts["ad_emb.weight"] = ("adduct.weight",)
        parts["pol_emb.weight"] = ("polarity.weight",)
        parts["ce_emb.inv"] = ("ce.inv",)
        for released, native in (
            ("cond_mlp.0", "condition.0"),
            ("cond_mlp.2", "condition.2"),
            ("head.0", "head.0"),
            ("head.3", "head.3"),
            ("elem_head.0", "elements.0"),
            ("elem_head.2", "elements.2"),
        ):
            for suffix in ("weight", "bias"):
                parts[f"{released}.{suffix}"] = (f"{native}.{suffix}",)
        return parts

    @override
    def checkpoint_names(self) -> dict[str, str]:
        """Name each native tensor, numbering the parts of a packed released one."""
        names: dict[str, str] = {}
        for released, parts in self.checkpoint_parts().items():
            if len(parts) == 1:
                names[released] = parts[0]
            else:
                names.update(
                    {f"{released}#{index}": name for index, name in enumerate(parts)},
                )
        return names

    @override
    def released_state(self) -> dict[str, Tensor]:
        """Join the packed parts and expose the complete released state.

        Returns:
          state: Complete released tensor names and values.

        """
        state = super().released_state()
        return {
            released: state[released]
            if len(parts) == 1
            else torch.cat(
                [state[f"{released}#{index}"] for index in range(len(parts))],
            )
            for released, parts in self.checkpoint_parts().items()
        }

    @override
    def load_released_state(self, state: Mapping[str, Tensor]) -> None:
        """Split the packed matrices, then validate and load like any released model.

        Args:
          state: Complete DreamsFP released checkpoint tensors.

        Raises:
          ValueError: Any key or shape differs from the declared architecture.

        """
        native = self.state_dict()
        split: dict[str, Tensor] = dict(state)
        for released, parts in self.checkpoint_parts().items():
            if len(parts) == 1 or released not in state:
                continue
            sizes = [native[name].shape[0] for name in parts]
            if state[released].shape[0] != sum(sizes):
                raise ValueError(f"DreamsFP checkpoint shapes differ: [{released!r}].")
            del split[released]
            split.update(
                {
                    f"{released}#{index}": value
                    for index, value in enumerate(state[released].split(sizes))
                },
            )
        super().load_released_state(split)

    @override
    def forward(
        self,
        mz: Tensor,
        it: Tensor,
        pad: Tensor,
        prec: Tensor,
        ad: Tensor,
        ce: Tensor,
        ce_known: Tensor,
        merged: Tensor,
        mode: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict both heads with precursor-only conditioning and peak-only pooling.

        Args:
          mz: Batched peak masses.
          it: Square-root relative peak intensities.
          pad: True at padded peak slots.
          prec: Precursor masses.
          ad: Adduct vocabulary indices.
          ce: Collision energies in electron volts.
          ce_known: Indicators for measured collision energy.
          merged: Merged spectrum count divided by four.
          mode: Ionization polarity, positive or negative one.

        Returns:
          logits: Selected fingerprint scores.
          elements: Element regression predictions.

        """
        spectrum = self.prepare_spectrum(
            mz,
            intensity=it,
            pad=pad,
            precursor=prec,
            mode=mode,
        )
        features = torch.cat(
            [
                self.ce(ce / 10 + 1) * ce_known[:, None],
                ce_known[:, None],
                merged[:, None],
            ],
            dim=-1,
        )
        condition = (
            self.adduct(ad)
            + self.polarity((mode > 0).long())
            + self.condition(features)
        )
        tokens = self.backbone(spectrum, condition=condition)
        real = spectrum[..., 0] != 0
        real = torch.cat([torch.zeros_like(real[:, :1]), real[:, 1:]], dim=1)
        weights = real[..., None].to(tokens.dtype)
        pooled = (tokens * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        representation = torch.cat([tokens[:, 0], pooled], dim=-1)
        return self.head(representation), self.elements(representation)

    def prepare_spectrum(
        self,
        mass: Tensor,
        *,
        intensity: Tensor,
        pad: Tensor,
        precursor: Tensor,
        mode: Tensor,
    ) -> Tensor:
        """Select intensity-ranked peaks and prepend the shifted precursor token.

        Args:
          mass: Batched raw peak masses.
          intensity: Square-root relative peak intensities.
          pad: True at padded raw peak slots.
          precursor: Precursor masses.
          mode: Ionization polarity controlling the mass shift.

        Returns:
          spectrum: Fixed-width mass/intensity pairs including the precursor.

        """
        valid = (~pad) & (mass > 0)
        values = torch.where(
            valid,
            intensity.float() ** 2,
            torch.full_like(intensity, -1.0, dtype=torch.float32),
        )
        count = min(self.config.max_peaks, mass.shape[1])
        top = values.topk(count, dim=1)
        selected = mass.float().gather(1, top.indices)
        keep = top.values > 0
        maximum = (
            torch.where(keep, top.values, torch.zeros_like(top.values))
            .amax(1, keepdim=True)
            .clamp(min=1e-12)
        )
        values = torch.where(keep, top.values / maximum, torch.zeros_like(top.values))
        shift = (mode < 0).double() * self.config.negative_mass_shift
        selected = torch.where(
            keep,
            (selected.double() + shift[:, None]).float(),
            torch.zeros_like(selected),
        )
        if count < self.config.max_peaks:
            selected = functional.pad(selected, (0, self.config.max_peaks - count))
            values = functional.pad(values, (0, self.config.max_peaks - count))
        mass = torch.cat(
            [(precursor.float().double() + shift).float()[:, None], selected],
            dim=1,
        )
        values = torch.cat(
            [
                torch.full(
                    (len(mass), 1),
                    self.config.precursor_intensity,
                    device=mass.device,
                ),
                values,
            ],
            dim=1,
        )
        return torch.stack([mass, values], dim=-1)


class SecondFPNet(ReleasedModel):
    """Predict the second engine's fingerprint vocabulary with instrument context."""

    class Config(Fig["SecondFPNet"]):
        channels_hidden: int = 512
        """Token width shared by all embeddings and transformer blocks."""

        channels_out: int = 6_930
        """Selected molecular fingerprint vocabulary."""

        num_layers: int = 6
        """Repeated pre-norm transformer blocks."""

        mz_emb: SinEmb.Config = field(default_factory=lambda: SinEmb.Config(hi=3.2))
        """Mass-to-charge wavelength features."""

        nl_emb: SinEmb.Config = field(default_factory=lambda: SinEmb.Config(hi=3.2))
        """Neutral-loss wavelength features."""

        prec_emb: SinEmb.Config = field(default_factory=lambda: SinEmb.Config(hi=3.2))
        """Precursor wavelength features."""

        peak_projection: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Project mass, neutral loss and one intensity channel."""

        adduct: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(26, init_weight=nn.init.normal_),
        )
        """Adduct lookup, including the unknown token."""

        instrument: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(5, init_weight=nn.init.normal_),
        )
        """timsTOF, Orbitrap, QTOF, ion trap and other instrument lookups."""

        context_projection: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Project precursor features, scaled CE, polarity and log precursor mass."""

        block: Block.Config = field(default_factory=Block.Config)
        """Priml attention, affine norms, GELU feed-forward and residual dropout."""

        norm: LayerNorm.Config = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Final affine token normalization."""

        head: SwiGLU.Config = field(default_factory=fingerprint_head)
        """Linear, GELU, dropout and output projection of the pooled tokens."""

        @override
        def finalize(self) -> Self:
            width = self.channels_hidden
            for embedding in (self.mz_emb, self.nl_emb, self.prec_emb):
                embedding.channels_out = width
            self.peak_projection.channels_in = 2 * width + 1
            self.peak_projection.channels_out = width
            self.context_projection.channels_in = width + 3
            self.context_projection.channels_out = width
            self.adduct.channels_out = self.instrument.channels_out = width
            self.block.channels_in = self.norm.channels_in = width
            if self.head.channels_in == -1:
                self.head.channels_in = 2 * width
            if self.head.channels_out == -1:
                self.head.channels_out = self.channels_out
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.mz_emb = config.mz_emb.make()
        self.nl_emb = config.nl_emb.make()
        self.peak_projection = config.peak_projection.make()
        self.prec_emb = config.prec_emb.make()
        self.adduct = config.adduct.make()
        self.instrument = config.instrument.make()
        self.context_projection = config.context_projection.make()
        self.blocks = nn.ModuleList(
            [config.block.make() for _ in range(config.num_layers)],
        )
        self.norm = config.norm.make()
        self.head = config.head.make()

    @override
    def checkpoint_names(self) -> dict[str, str]:
        """Map the complete second-engine schema without filtering or shape inference."""
        names: dict[str, str] = {"mz_emb.inv": "mz_emb.inv", "nl_emb.inv": "nl_emb.inv"}
        for suffix in ("weight", "bias"):
            names[f"pk.{suffix}"] = f"peak_projection.{suffix}"
        names["prec_emb.inv"] = "prec_emb.inv"
        names["ad.weight"] = "adduct.weight"
        names["ins.weight"] = "instrument.weight"
        for suffix in ("weight", "bias"):
            names[f"gl.{suffix}"] = f"context_projection.{suffix}"
        for index in range(len(self.blocks)):
            for released, native in (
                ("n1", "norm1"),
                ("qkv", "attn.attention.proj_qkv"),
                ("o", "attn.attention.proj_out"),
                ("n2", "norm2"),
                ("ff.0", "ffn.0.up_proj"),
                ("ff.3", "ffn.0.down_proj"),
            ):
                for suffix in ("weight", "bias"):
                    names[f"blocks.{index}.{released}.{suffix}"] = (
                        f"blocks.{index}.{native}.{suffix}"
                    )
        for released, native in (
            ("norm", "norm"),
            ("head.0", "head.up_proj"),
            ("head.3", "head.down_proj"),
        ):
            for suffix in ("weight", "bias"):
                names[f"{released}.{suffix}"] = f"{native}.{suffix}"
        if set(names.values()) != set(self.state_dict()):
            raise ValueError(
                "Native SecondFPNet state differs from its checkpoint schema.",
            )
        return names

    @override
    def forward(
        self,
        mz: Tensor,
        it: Tensor,
        pad: Tensor,
        prec: Tensor,
        ad: Tensor,
        ins: Tensor,
        ce: Tensor,
        mode: Tensor,
    ) -> Tensor:
        """Encode instrument context and pool the global and nonpadding peak states.

        Args:
          mz: Batched peak masses.
          it: Square-root relative peak intensities.
          pad: True at padded peak positions.
          prec: Precursor masses.
          ad: Adduct indices.
          ins: Instrument family indices.
          ce: Collision energy in electron volts.
          mode: Ionization polarity, positive or negative one.

        Returns:
          logits: Unnormalized scores for the second engine's selected bits.

        """
        mass_features = torch.cat(
            [
                self.mz_emb(mz),
                self.nl_emb((prec[:, None] - mz).clamp(min=0)),
                it[..., None],
            ],
            dim=-1,
        )
        context_features = torch.cat(
            [
                self.prec_emb(prec),
                (ce / 100)[..., None],
                mode[..., None],
                torch.log1p(prec)[..., None] / 10,
            ],
            dim=-1,
        )
        context = (
            self.context_projection(context_features)
            + self.adduct(ad)
            + self.instrument(ins)
        )
        tokens = torch.cat(
            [context[:, None], self.peak_projection(mass_features)],
            dim=1,
        )
        valid = torch.cat([torch.ones_like(pad[:, :1]), ~pad], dim=1)
        for block in self.blocks:
            tokens = block(tokens, attn_mask=valid[:, None, None])
        tokens = self.norm(tokens)
        weight = valid[:, 1:].float()[..., None]
        pooled = (tokens[:, 1:] * weight).sum(1) / weight.sum(1).clamp(min=1)
        return self.head(torch.cat([tokens[:, 0], pooled], dim=-1))


class FormulaEncoding(nn.Module):
    """Absolute-sine integer-count features with an oversized-count token."""

    class Config(Fig["FormulaEncoding"]):
        max_count: int = 255
        """First count represented by the learned oversized-count token."""

        channels_out: int = -1
        """Derived number of powers-of-two count features."""

        extra: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(init_weight=nn.init.normal_),
        )
        """One normally initialized token with the absolute-sine table's width."""

        @override
        def finalize(self) -> Self:
            self.channels_out = (self.max_count - 1).bit_length()
            self.extra.channels_in = 1
            self.extra.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            num_elements: int,
            oversized: int,
            normal_rows: int,
            counts_dtype: torch.dtype = torch.int64,
            **kwargs: object,
        ) -> Cost:
            """Count masked formula gathers, frozen storage and the learned-token VJP.

            Args:
              seq_len: Formula rows per example.
              batch_size: Examples represented by these formula rows.
              dtype: Outer activation dtype; table/output dtypes are explicit here.
              num_elements: Element-count entries in each formula.
              oversized: Actual entries at or above max_count in this invocation.
              normal_rows: Actual distinct ordinary table rows selected by counts.
              counts_dtype: Storage dtype of the mutable input counts.
              **kwargs: Cost metadata forwarded to the learned token.

            Returns:
              execution: Frozen table ownership, selection traffic and token adjoint.

            Raises:
              ValueError: Oversized entries exceed the supplied input geometry.
                Normal rows cannot fit the ordinary entries and fixed table.

            """
            del dtype
            entries = seq_len * batch_size * num_elements
            if oversized < 0 or oversized > entries:
                raise ValueError(
                    "oversized must be between zero and the count entries.",
                )
            width = self.channels_out
            ordinary = entries - oversized
            if normal_rows < (1 if ordinary else 0) or normal_rows > min(
                ordinary,
                self.max_count,
            ):
                raise ValueError("Normal rows must fit ordinary counts and the table.")
            output_dtype = resolve_dtype(None)
            token_dtype = resolve_dtype(self.extra.dtype)
            token = cost(
                self.extra,
                seq_len=oversized,
                batch_size=1,
                dtype=token_dtype,
                **kwargs,
            )
            token = replace(token, params_active=width if oversized else 0)
            total = (
                token
                # The absolute-sine table is a frozen Parameter, not a buffer;
                # constructor sin/abs work is not repeated by a lookup.
                + Cost(
                    params=self.max_count * width,
                    params_active=normal_rows * width,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=3 * ordinary * width,
                    dtype=torch.float32,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=oversized * width,
                    dtype=token_dtype,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=entries * width,
                    dtype=output_dtype,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=entries + 6 * oversized,
                    dtype=counts_dtype,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=3 * ordinary + 4 * oversized,
                    dtype=torch.int64,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=11 * entries,
                    dtype=torch.bool,
                )
                + traffic(
                    "adjoint",
                    "selection",
                    elements=2 * oversized * width,
                    dtype=output_dtype,
                )
                + traffic(
                    "adjoint",
                    "selection",
                    elements=entries,
                    dtype=torch.bool,
                )
            )
            if counts_dtype != torch.int64:
                total += traffic(
                    "primal",
                    "selection",
                    elements=entries,
                    dtype=counts_dtype,
                ) + traffic(
                    "primal",
                    "selection",
                    elements=entries,
                    dtype=torch.int64,
                )
            if token_dtype != output_dtype:
                total += traffic(
                    "adjoint",
                    "selection",
                    elements=oversized * width,
                    dtype=output_dtype,
                ) + traffic(
                    "adjoint",
                    "selection",
                    elements=oversized * width,
                    dtype=token_dtype,
                )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.extra = config.extra.make()
        frequency = (0.5 ** torch.arange(config.channels_out + 2).float())[2:]
        phase = torch.arange(config.max_count).float()[:, None] * (
            2 * math.pi * frequency[None, :]
        )
        self.table = nn.Parameter(phase.sin().abs().float(), requires_grad=False)

    @override
    def forward(self, counts: Tensor) -> Tensor:
        """Flatten elemental features, clamping oversized counts in the input.

        Args:
          counts: Integer formulas, mutated at entries outside the fixed table.

        Returns:
          encoded: Formula axes with their element and frequency axes flattened.

        """
        oversized = counts >= self.config.max_count
        counts[oversized] = counts[oversized].clamp(
            min=self.config.max_count,
            max=self.config.max_count,
        )
        indices = counts.long()
        output = torch.empty(
            (*counts.shape, self.config.channels_out),
            device=counts.device,
        )
        output[~oversized] = self.table[indices[~oversized]]
        output[oversized] = self.extra.weight[
            indices[oversized] - self.config.max_count
        ].to(output.dtype)
        return output.reshape(*counts.shape[:-1], -1)


class GraphConditions(nn.Module):
    """Frozen adduct, collision-energy and instrument graph context."""

    class Config(Fig["GraphConditions"]):
        adduct_groups: list[list[int]] = field(
            default_factory=lambda: [
                [0, 3],
                [0, 2],
                [0, 2],
                [0, 3, 6],
                [0, 3],
                [0],
                [0, 3, 6],
                [0, 3],
                [0, 3],
                [1],
                [1, 4],
                [1, 5],
                [1, 4, 6],
                [1, 4, 7],
            ],
        )
        """Mode, proton, salt and neutral-loss indicators in physical adduct order."""

        collision_channels: int = 64
        """Concatenated sine and cosine collision-energy dimensions."""

        collision_scalar: float = 10_000
        """Geometric denominator base for the collision-energy sinusoids."""

        num_instruments: int = 4
        """Instrument identities in released checkpoint order."""

        channels_out: int = -1
        """Derived complete adduct, collision-energy and instrument width."""

        @override
        def finalize(self) -> Self:
            self.channels_out = (
                len(self.adduct_groups)
                + len({mode for groups in self.adduct_groups for mode in groups})
                + self.collision_channels
                + self.num_instruments
            )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            adduct_rows: int,
            instrument_rows: int,
            missing_collision: bool,
            input_requires_grad: bool = True,
            index_dtype: torch.dtype = torch.int64,
            **kwargs: object,
        ) -> Cost:
            """Count frozen physical tables and missing-aware collision sinusoids.

            Args:
              seq_len: Graph-condition rows per example.
              batch_size: Examples represented by the flattened condition rows.
              dtype: Collision-energy input dtype; None uses Torch's default.
              adduct_rows: Actual distinct selected adduct rows in this invocation.
              instrument_rows: Actual distinct selected instrument rows.
              missing_collision: Whether any collision feature selects missing context.
              input_requires_grad: Differentiate collision energy, not frozen tables.
              index_dtype: Input storage dtype of adduct and instrument identities.
              **kwargs: Unread outer cost metadata.

            Returns:
              execution: Frozen parameter ownership and complete condition arithmetic.

            Raises:
              ValueError: Distinct rows cannot fit the batch or declared tables.

            """
            del kwargs
            rows = seq_len * batch_size
            adducts = len(self.adduct_groups)
            modes = len({mode for groups in self.adduct_groups for mode in groups})
            minimum = 1 if rows else 0
            if (
                adduct_rows < minimum
                or adduct_rows > min(rows, adducts)
                or instrument_rows < minimum
                or instrument_rows > min(rows, self.num_instruments)
            ):
                raise ValueError(
                    "Distinct condition rows must fit the batch and tables.",
                )
            identity_width = adducts + modes + self.num_instruments
            half = self.collision_channels // 2
            collision = 2 * rows * half
            default_dtype = resolve_dtype(None)
            input_dtype = resolve_dtype(dtype)
            phase_dtype = torch.promote_types(input_dtype, default_dtype)
            output_dtype = torch.promote_types(phase_dtype, torch.float32)
            total = (
                Cost(
                    params=adducts * (adducts + modes)
                    + half
                    + self.collision_channels
                    + self.num_instruments**2,
                    params_active=adduct_rows * (adducts + modes)
                    + instrument_rows * self.num_instruments
                    + (half if rows else 0)
                    + (self.collision_channels if missing_collision else 0),
                )
                + traffic(
                    "primal",
                    "elementwise",
                    elements=rows,
                    dtype=input_dtype,
                )
                + traffic(
                    "primal",
                    "elementwise",
                    elements=half + self.collision_channels,
                    dtype=default_dtype,
                )
                + traffic(
                    "primal",
                    "elementwise",
                    elements=5 * rows * half + 3 * collision,
                    flops=3 * rows * half + collision,
                    dtype=phase_dtype,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=3 * rows * identity_width,
                    dtype=torch.float32,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=3 * collision,
                    dtype=phase_dtype,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=rows * self.channels_out,
                    dtype=output_dtype,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=2 * rows,
                    dtype=torch.int64,
                )
                + traffic(
                    "primal",
                    "elementwise",
                    elements=2 * collision,
                    dtype=torch.bool,
                )
            )
            if index_dtype != torch.int64:
                total += traffic(
                    "primal",
                    "selection",
                    elements=2 * rows,
                    dtype=index_dtype,
                ) + traffic(
                    "primal",
                    "selection",
                    elements=2 * rows,
                    dtype=torch.int64,
                )
            if input_requires_grad:
                # Only CE is differentiable: categorical indices and every
                # physical table are frozen. Broadcasting CE reduces half waves.
                total += (
                    traffic(
                        "adjoint",
                        "elementwise",
                        elements=17 * rows * half + 2 * collision,
                        flops=7 * rows * half + collision,
                        dtype=phase_dtype,
                    )
                    + traffic(
                        "adjoint",
                        "elementwise",
                        elements=half,
                        dtype=default_dtype,
                    )
                    + traffic(
                        "adjoint",
                        "elementwise",
                        elements=collision,
                        dtype=torch.bool,
                    )
                    + reduction_cost(
                        input_elements=rows * half,
                        output_groups=rows,
                        dtype=phase_dtype,
                        phase="adjoint",
                    )
                )
                if input_dtype != phase_dtype:
                    total += traffic(
                        "adjoint",
                        "elementwise",
                        elements=rows,
                        dtype=phase_dtype,
                    ) + traffic(
                        "adjoint",
                        "elementwise",
                        elements=rows,
                        dtype=input_dtype,
                    )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        modes = len({mode for groups in config.adduct_groups for mode in groups})
        multi = torch.zeros(len(config.adduct_groups), modes)
        for index, groups in enumerate(config.adduct_groups):
            multi[index, groups] = 1
        self.adduct = nn.Parameter(
            torch.cat([torch.eye(len(config.adduct_groups)), multi], dim=-1).float(),
            requires_grad=False,
        )
        power = 2 * torch.arange(config.collision_channels // 2)
        self.collision_denominators = nn.Parameter(
            torch.pow(config.collision_scalar, power / config.collision_channels),
            requires_grad=False,
        )
        self.collision_missing = nn.Parameter(
            torch.zeros(config.collision_channels),
            requires_grad=False,
        )
        self.instrument = nn.Parameter(
            torch.eye(config.num_instruments).float(),
            requires_grad=False,
        )

    @override
    def forward(
        self,
        collision_energy: Tensor,
        adduct: Tensor,
        instrument: Tensor,
    ) -> Tensor:
        """Join frozen physical identities and missing-aware collision features."""
        phase = collision_energy[:, None] / self.collision_denominators[None, :]
        collision = torch.cat([phase.sin(), phase.cos()], dim=1)
        collision = torch.where(
            collision.isnan(),
            self.collision_missing[None, :],
            collision,
        )
        return torch.cat(
            [self.adduct[adduct.long()], collision, self.instrument[instrument.long()]],
            dim=-1,
        )


class PackedAttention(nn.MultiheadAttention):
    """Use Priml-owned projections with the standard packed-attention ABI.

    The standard kernel supplies its training SDPA and inference dispatch. Packed
    parameters consume no draws before the output affine, then receive Xavier
    initialization. No source-specific attention implementation is imported.
    """

    class Config(Fig["PackedAttention"]):
        projections: AttentionProjections.Config = field(
            default_factory=lambda: AttentionProjections.Config(
                channels_in=256,
                dropout=0.1,
                proj_qkv=Linear.Config(bias=True, init_weight=nn.init.zeros_),
                proj_out=Linear.Config(bias=True, init_bias=None),
            ),
        )
        """Shared attention geometry and source-ordered projection initialization."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            memory_len: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Price packed self-attention or Torch's sliced cross projections.

            Args:
              seq_len: Query positions per example.
              batch_size: Examples in the invocation.
              dtype: Projection and attention operand dtype.
              memory_len: Key/value positions; negative denotes self-attention.
              **kwargs: Additional child cost arguments.

            Returns:
              execution: Complete projections and dense attention, without a cache.

            """
            projection = self.projections
            width = projection.channels_in
            memory = seq_len if memory_len < 0 else memory_len
            if memory_len < 0:
                total = cost(
                    projection,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            else:
                assert isinstance(projection.proj_qkv, Linear.Config)
                total = (
                    matmul_cost(
                        channels_in=width,
                        channels_out=width,
                        rows=batch_size * seq_len,
                        bias=projection.proj_qkv.bias,
                        dtype=dtype,
                    )
                    + matmul_cost(
                        channels_in=width,
                        channels_out=2 * width,
                        rows=batch_size * memory,
                        bias=projection.proj_qkv.bias,
                        dtype=dtype,
                    )
                    + cost(
                        projection.proj_out,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                )
            return (
                total
                + attention_kernel_cost(
                    seq_len=memory,
                    rows=seq_len,
                    batch_size=batch_size,
                    num_heads=projection.num_heads,
                    channels_head=projection.channels_head,
                    dropout_p=projection.dropout,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    kernel="selection",
                    elements=2 * batch_size * width * (seq_len + 2 * memory),
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    kernel="selection",
                    elements=2 * batch_size * width * (seq_len + 2 * memory),
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        cfg = config.projections
        self.embed_dim = self.kdim = self.vdim = cfg.channels_in
        self._qkv_same_embed_dim = True
        self.num_heads = cfg.num_heads
        self.dropout = cfg.dropout
        self.batch_first = True
        projections = cfg.make()
        packed, output = projections.proj_qkv, projections.proj_out
        assert isinstance(packed, Linear)
        assert isinstance(output, Linear)
        self.head_dim = cfg.channels_head
        self.in_proj_weight = packed.weight
        self.register_parameter("q_proj_weight", None)
        self.register_parameter("k_proj_weight", None)
        self.register_parameter("v_proj_weight", None)
        self.in_proj_bias = packed.bias
        self.out_proj = output
        self.bias_k = self.bias_v = None
        self.add_zero_attn = False
        _ = nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            _ = nn.init.zeros_(self.in_proj_bias)
        if self.out_proj.bias is not None:
            _ = nn.init.zeros_(self.out_proj.bias)


class LayerSettings:
    """Configure the common Priml leaves of standard encoder and decoder layers."""

    class Config(Fig[nn.Module]):
        channels_in: int = 256
        """Residual width."""

        attention: PackedAttention.Config = field(
            default_factory=PackedAttention.Config,
        )
        """Standard packed self-attention."""

        expand: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Fourfold hidden expansion."""

        contract: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Projection back to residual width."""

        norm: LayerNorm.Config = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True, eps=1e-5),
        )
        """Independent affine normalization at each residual."""

        dropout: Dropout.Config = field(default_factory=lambda: Dropout.Config(p=0.1))
        """Independent dropout modules with the same probability."""

        activation: Activation.Config = field(
            default_factory=lambda: Activation.Config(act=relu),
        )
        """Rectify the expanded hidden activations."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            memory_len: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Compose the encoder's attention, FFN and two residual normalizations.

            Args:
              seq_len: Positions per example.
              batch_size: Examples in the invocation.
              dtype: Floating operand dtype.
              memory_len: Memory geometry reserved for the decoder override.
              **kwargs: Additional child cost arguments.

            Returns:
              execution: Every encoder leaf and residual operation.

            """
            del memory_len
            bus = {
                **kwargs,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "dtype": dtype,
            }
            return (
                cost(self.attention, **bus)
                + cost(self.expand, **bus)
                + cost(self.activation, channels=self.expand.channels_out, **bus)
                + cost(self.dropout, channels=self.expand.channels_out, **bus)
                + cost(self.contract, **bus)
                + cost(self.norm, **bus).tile(2, copies=2)
                + cost(self.dropout, channels=self.channels_in, **bus).tile(2)
                + elementwise_cost(
                    primal=2 * batch_size * seq_len * self.channels_in,
                    adjoint=2 * batch_size * seq_len * self.channels_in,
                    channels=self.channels_in,
                    rows=batch_size * seq_len,
                    inputs=2,
                    dtype=dtype,
                )
            )

        @override
        def finalize(self) -> Self:
            self.attention.projections.channels_in = self.norm.channels_in = (
                self.channels_in
            )
            self.expand.channels_in = self.contract.channels_out = self.channels_in
            self.expand.channels_out = self.contract.channels_in = 4 * self.channels_in
            return super().finalize()


class Encoder(nn.TransformerEncoderLayer):
    """Assemble a post-norm encoder from Priml leaves and stock torch dispatch."""

    class Config(Makes["Encoder"], LayerSettings.Config, kw_only=False):
        """Configure the common encoder leaves."""

    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.self_attn = config.attention.make()
        self.linear1 = config.expand.make()
        self.dropout = config.dropout.make()
        self.linear2 = config.contract.make()
        self.norm_first = False
        self.norm1 = config.norm.make()
        self.norm2 = config.norm.make()
        self.dropout1 = config.dropout.make()
        self.dropout2 = config.dropout.make()
        self.activation_relu_or_gelu = 1
        self.activation = config.activation.make()


class Decoder(nn.TransformerDecoderLayer):
    """Assemble self/cross attention with independent Priml projections and norms."""

    class Config(Makes["Decoder"], LayerSettings.Config, kw_only=False):
        """Use the encoder geometry for two independently initialized attentions."""

        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            memory_len: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Add the decoder's independent cross-attention residual.

            Args:
              seq_len: Target positions per example.
              batch_size: Examples in the invocation.
              dtype: Floating operand dtype.
              memory_len: Memory positions; negative uses the target length.
              **kwargs: Additional child cost arguments.

            Returns:
              execution: Self/cross attention, FFN and three residual norms.

            """
            bus = {
                **kwargs,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "dtype": dtype,
            }
            return (
                super().cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    memory_len=-1,
                    **kwargs,
                )
                + cost(
                    self.attention,
                    memory_len=seq_len if memory_len < 0 else memory_len,
                    **bus,
                )
                + cost(self.norm, **bus)
                + cost(self.dropout, channels=self.channels_in, **bus)
                + elementwise_cost(
                    primal=batch_size * seq_len * self.channels_in,
                    adjoint=batch_size * seq_len * self.channels_in,
                    channels=self.channels_in,
                    rows=batch_size * seq_len,
                    inputs=2,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.self_attn = config.attention.make()
        self.multihead_attn = config.attention.make()
        self.linear1 = config.expand.make()
        self.dropout = config.dropout.make()
        self.linear2 = config.contract.make()
        self.norm_first = False
        self.norm1 = config.norm.make()
        self.norm2 = config.norm.make()
        self.norm3 = config.norm.make()
        self.dropout1 = config.dropout.make()
        self.dropout2 = config.dropout.make()
        self.dropout3 = config.dropout.make()
        self.activation = config.activation.make()


class Slots(nn.Module):
    """Decode every layer of learned orthogonal fragment queries."""

    class Config(Fig["Slots"]):
        count: int = 200
        """Fragment query count."""

        num_layers: int = 4
        """Independent copies of a single initialized decoder prototype."""

        embedding: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(
                init_weight=partial(nn.init.orthogonal_, gain=0.02),
            ),
        )
        """Orthogonal learned query rows."""

        decoder: Decoder.Config = field(default_factory=Decoder.Config)
        """Decoder geometry and Priml leaf configurations."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            root_len: int = 1,
            **kwargs: object,
        ) -> Cost:
            """Price every retained decoder output and the shared learned query table.

            Args:
              seq_len: Padded atom positions, excluding root tokens.
              batch_size: Graphs in the invocation.
              dtype: Floating operand dtype.
              root_len: Root tokens concatenated before atoms.
              **kwargs: Additional decoder cost arguments.

            Returns:
              execution: All decoder layers, query ownership and stack traffic.

            """
            width = self.decoder.channels_in
            memory = seq_len + root_len
            queries = self.count * width
            return (
                cost(
                    self.decoder,
                    seq_len=self.count,
                    memory_len=memory,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                ).tile(self.num_layers, copies=self.num_layers)
                + Cost(params=queries, params_active=queries)
                + reduction_cost(
                    input_elements=batch_size * queries,
                    output_groups=queries,
                    dtype=dtype,
                    phase="adjoint",
                )
                + elementwise_cost(
                    primal=0,
                    adjoint=max(0, self.num_layers - 1) * batch_size * memory * width,
                    channels=memory * width,
                    rows=batch_size,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    kernel="selection",
                    elements=2
                    * batch_size
                    * (memory * width + self.num_layers * queries),
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    kernel="selection",
                    elements=2
                    * batch_size
                    * (memory * width + self.num_layers * queries),
                    dtype=dtype,
                )
            )

        @override
        def finalize(self) -> Self:
            self.embedding.channels_in = self.count
            self.embedding.channels_out = self.decoder.channels_in
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.embedding = config.embedding.make()
        prototype = config.decoder.make()
        self.layers = nn.ModuleList(
            [copy.deepcopy(prototype) for _ in range(config.num_layers)],
        )

    @override
    def forward(self, root: Tensor, atoms: Tensor, padding: Tensor) -> Tensor:
        memory = torch.cat([root, atoms], dim=1)
        hidden = self.embedding.weight.unsqueeze(0).expand(len(root), -1, -1)
        outputs: list[Tensor] = []
        for layer in self.layers:
            hidden = layer(tgt=hidden, memory=memory, memory_key_padding_mask=padding)
            outputs.append(hidden)
        return torch.stack(outputs)


class Breakpoints:
    """Project cardinality constraints and retain the released discrete graph semantics."""

    class Config(Fig["Breakpoints"]):
        tau: float = 15
        """Temperature of the shipped single-iteration equality projection."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def rank(self, logits: Tensor, counts: Tensor) -> Tensor:
        """Return zero/one/two/three-atom equality projections for every slot.

        Args:
          logits: Per-slot atom logits, including padded atoms.
          counts: Real atom counts in each molecule.

        Returns:
          ranked: Cardinality probabilities in the final four-entry axis.

        """
        atoms = logits.shape[-1]
        padding = torch.arange(atoms, device=logits.device)[None, :] >= counts[:, None]
        masked = logits.masked_fill(padding[:, None, :], -1e9).reshape(-1, atoms)
        coefficient = masked.new_ones(atoms).log()
        ranked: list[Tensor] = []
        for cardinality in (1, 2, 3):
            # Separate division nodes preserve the released adjoint accumulation.
            pair = torch.stack(
                [
                    masked / self.config.tau,
                    torch.full_like(masked, 0.0 / self.config.tau),
                ],
                dim=-2,
            )
            target = masked.new_tensor([cardinality, atoms - cardinality]).clamp(min=0)
            pair = pair - torch.logsumexp(pair + coefficient, dim=-1, keepdim=True)
            pair = pair + target.log().view(1, 2, 1)
            pair = pair - torch.logsumexp(pair, dim=-2, keepdim=True)
            ranked.append(pair[:, 0, :].exp().reshape(logits.shape))
        return torch.stack([torch.zeros_like(ranked[0]), *ranked], dim=-1)

    def select(self, logits: Tensor, cards: Tensor, counts: Tensor) -> Tensor:
        """Choose the predicted number of atoms, preserving torch top-k ties.

        Args:
          logits: Per-slot atom logits.
          cards: Four cardinality logits for each slot.
          counts: Real atom counts per molecule.

        Returns:
          removed: Boolean per-slot atom removal patterns.

        """
        ranks = self.rank(logits, counts)
        chosen = cards.argmax(dim=-1)
        projected = ranks.gather(
            -1,
            chosen[:, :, None, None].expand(*logits.shape, 1),
        ).squeeze(-1)
        flat = projected.reshape(-1, logits.shape[-1])
        cardinality = chosen.reshape(-1)
        removed = torch.zeros_like(flat, dtype=torch.bool)
        largest = int(cardinality.max())
        if largest > 0:
            indices = flat.topk(largest, dim=-1).indices
            active = (
                torch.arange(largest, device=flat.device)[None, :]
                < cardinality[:, None]
            )
            rows = torch.arange(len(flat), device=flat.device)[:, None].expand_as(
                indices,
            )
            removed[rows[active], indices[active]] = True
        return removed.view(logits.shape)

    def patterns(
        self,
        removed: Tensor,
        adjacency: Tensor,
        counts: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Extract sorted unique nonempty reachable components within each molecule.

        Args:
          removed: Per-slot atom removal patterns.
          adjacency: Padded molecular bond matrices.
          counts: Real atom counts per molecule.

        Returns:
          patterns: Padded integer component-membership rows.
          sizes: Number of components, with the source's full-root fallback.

        """
        batch, slots, atoms = removed.shape
        graphs = adjacency[:, None].expand(-1, slots, -1, -1).reshape(-1, atoms, atoms)
        reachable = self.closure(
            graphs,
            removed=removed.reshape(-1, atoms),
            counts=counts,
        )
        molecule = torch.arange(batch, device=adjacency.device).repeat_interleave(slots)
        molecule = molecule[:, None].expand(-1, atoms).reshape(-1, 1)
        rows = torch.cat([molecule, reachable.long().reshape(-1, atoms)], dim=-1)
        unique = torch.unique(rows, dim=0)
        unique = unique[unique[:, 1:].sum(dim=-1) != 0]
        sizes = torch.bincount(unique[:, 0], minlength=batch)
        patterns = pad_atoms(unique[:, 1:], sizes)
        full_root = (
            torch.arange(atoms, device=adjacency.device)[None, :] < counts[:, None]
        )
        full_root = full_root & (sizes[:, None] == 0)
        patterns[:, 0, :] = patterns[:, 0, :].bool() | full_root
        return patterns, sizes.clamp(min=1)

    def closure(
        self,
        adjacency: Tensor,
        *,
        removed: Tensor | None = None,
        counts: Tensor | None = None,
    ) -> Tensor:
        """Compute the source's non-reflexive transitive bond closure.

        Args:
          adjacency: Bond matrices, optionally repeated once per removal slot.
          removed: Atoms whose incoming and outgoing bonds are removed.
          counts: Real atom counts before optional slot repetition.

        Returns:
          reachable: Reachability without adding self-loops for isolated atoms.

        """
        reachable = adjacency > 0
        if removed is not None:
            reachable = reachable & ~removed[..., None] & ~removed[:, None, :]
        for atom in range(adjacency.shape[-1]):
            reachable = reachable | (
                reachable[:, :, atom : atom + 1] & reachable[:, atom : atom + 1, :]
            )
        if counts is not None:
            valid = (
                torch.arange(adjacency.shape[-1], device=adjacency.device)[None, :]
                < counts[:, None]
            )
            valid = (valid[:, :, None] & valid[:, None, :]).repeat_interleave(
                len(adjacency) // len(counts),
                0,
            )
            reachable = reachable & valid
        return reachable


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphInput:
    """Carry prepared padded atom, path, degree and attention-bias tensors."""

    features: Tensor
    bias: Tensor
    edges: Tensor
    spatial: Tensor
    degrees: Tensor
    paths: Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class GlacierBatch:
    """Carry complete prepared chemical inputs for joint fragmentation/intensity."""

    graph: GraphInput
    atom_counts: Tensor
    adduct: Tensor
    collision_energy: Tensor
    instrument: Tensor
    root_forms: Tensor
    atom_forms: Tensor
    masses: Tensor
    adduct_shifts: Tensor
    adjacency: Tensor
    atom_hs: Tensor
    total_hs: Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class Fragmentation:
    """Carry the embeddings, slot-layer logits and chosen fragments of one batch."""

    root: Tensor
    atoms: Tensor
    logits: Tensor
    cards: Tensor
    fragments: Tensor
    counts: Tensor

    def fields(self, batch: GlacierBatch) -> dict[str, Tensor]:
        """Return the tensors the intensity decoder reads for this batch.

        Args:
          batch: The prepared batch this fragmentation was selected from.

        Returns:
          fields: Embeddings and chemical inputs, keyed by decoder argument.

        """
        return {
            "root": self.root,
            "atoms": self.atoms,
            "root_forms": batch.root_forms,
            "atom_forms": batch.atom_forms,
            "atom_counts": batch.atom_counts,
            "atom_hs": batch.atom_hs,
            "total_hs": batch.total_hs,
            "adjacency": batch.adjacency,
            "adduct_shifts": batch.adduct_shifts,
            "masses": batch.masses,
        }


class BreakpointOutput(TypedDict):
    """All slot-layer atom and cardinality logits."""

    frag_logits: Tensor
    frag_card_logits: Tensor


class IntensityOutput(TypedDict):
    """All unbinned mass/intensity pairs."""

    output: Tensor


class FragmentOutput(TypedDict):
    """Packed predicted fragments and their molecule-level counts."""

    fragments: Tensor
    fragment_count: Tensor


class GlacierOutput(TypedDict):
    """Complete joint model outputs."""

    breakpoints_pred: BreakpointOutput
    inten_pred_end_to_end: IntensityOutput
    frags_pred: FragmentOutput


def pad_atoms(features: Tensor, counts: Tensor) -> Tensor:
    """Scatter packed atoms to padded graph rows without changing encounter order.

    Args:
      features: Concatenated atom or fragment features.
      counts: Segment lengths in their input encounter order.

    Returns:
      padded: Feature rows with zeros after each segment's final real position.

    """
    width = int(counts.max())
    segment = torch.repeat_interleave(counts)
    first = counts.cumsum(0) - counts
    index = torch.arange(len(features), device=features.device)
    positions = index - first[segment] + width * segment
    padded = features.new_zeros((len(counts) * width, *features.shape[1:]))
    padded[positions] = features
    return padded.view(len(counts), width, *features.shape[1:])


def released_graphormer() -> Graphormer.Config:
    """Return the released GLACIER graph encoder geometry.

    Returns:
      config: Graphormer with the shipped degree, shortest-path and bond-feature
        vocabularies and ten learned multi-hop distance matrices.

    """
    config = Graphormer.Config()
    config.channels_hidden = 256
    config.heads = 8
    config.num_layers = 5
    config.edge_distances = 10
    config.max_hops = 10
    config.degree.channels_in = 8
    config.spatial.channels_in = 1_025
    config.edge.channels_in = 28
    return config


class Glacier(ReleasedModel):
    """Predict graph fragments and spectra with the complete public GLACIER family."""

    class Config(Fig["Glacier"]):
        channels_hidden: int = 256
        """Shared graph, slot and intensity residual width."""

        heads: int = 8
        """Attention heads throughout the model."""

        channels_in: int = 88
        """Prepared atom features, before physical conditions."""

        intensity_decoder_layers: int = 4
        """Fragment-to-atom intensity decoder layers."""

        intensity_encoder_layers: int = 3
        """Fragment intensity encoder layers."""

        max_broken: int = 6
        """Broken-bond embedding and hydrogen-shift radius."""

        num_elements: int = 18
        """Element counts per root or fragment formula."""

        formula: FormulaEncoding.Config = field(default_factory=FormulaEncoding.Config)
        """Absolute-sine formula features and oversized-count token."""

        conditions: GraphConditions.Config = field(
            default_factory=GraphConditions.Config,
        )
        """Physical adduct, collision and instrument features."""

        root: Graphormer.Config = field(default_factory=released_graphormer)
        """Root graph encoder with degree and multi-hop bias."""

        formula_mapper: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Combine graph token and root formula before slot attention."""

        token_mapper: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Combine root token, fragment/difference formulas and broken-bond count."""

        queries: Slots.Config = field(default_factory=Slots.Config)
        """Orthogonal learned removal queries and all slot-layer outputs."""

        cardinality: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Zero/one/two/three removed-atom logits."""

        decoder: Decoder.Config = field(default_factory=Decoder.Config)
        """Fragment-to-atom intensity attention prototype."""

        encoder: Encoder.Config = field(default_factory=Encoder.Config)
        """Fragment intensity self-attention prototype."""

        output: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Two ion groups at every hydrogen shift."""

        breakpoints: Breakpoints.Config = field(default_factory=Breakpoints.Config)
        """The shipped equality projection and discrete component semantics."""

        @override
        def finalize(self) -> Self:
            self.root.channels_in = (
                self.channels_in + self.conditions.copy_tree().finalize().channels_out
            )
            self.root.channels_hidden = self.channels_hidden
            self.root.heads = self.heads
            formula_width = (
                self.num_elements * self.formula.copy_tree().finalize().channels_out
            )
            self.formula_mapper.channels_in = self.channels_hidden + formula_width
            self.token_mapper.channels_in = (
                self.channels_hidden + 2 * formula_width + self.max_broken + 1
            )
            self.formula_mapper.channels_out = self.token_mapper.channels_out = (
                self.channels_hidden
            )
            for decoder in (self.queries.decoder, self.decoder, self.encoder):
                decoder.channels_in = self.channels_hidden
                decoder.attention.projections.num_heads = self.heads
            self.cardinality.channels_in = self.output.channels_in = (
                self.channels_hidden
            )
            self.cardinality.channels_out = 4
            self.output.channels_out = 2 * (2 * self.max_broken + 1)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.conditions = config.conditions.make()
        self.root = config.root.make()
        self.formula = config.formula.make()
        self.formula_mapper = config.formula_mapper.make()
        self.token_mapper = config.token_mapper.make()
        self.queries = config.queries.make()
        self.cardinality = config.cardinality.make()
        self.decoder = nn.TransformerDecoder(
            config.decoder.make(),
            config.intensity_decoder_layers,
        )
        self.encoder = nn.TransformerEncoder(
            config.encoder.make(),
            config.intensity_encoder_layers,
        )
        self.output = config.output.make()
        self.breakpoints = config.breakpoints.make()

    @override
    def checkpoint_names(self) -> dict[str, str]:
        names = {
            "adduct_embedder": "conditions.adduct",
            "collision_embedder_denominators": "conditions.collision_denominators",
            "collision_embed_merged": "conditions.collision_missing",
            "instrument_embedder": "conditions.instrument",
        }
        for old, new in (
            ("graph_node_feature.atom_encoder", "node"),
            ("graph_node_feature.degree_encoder", "degree"),
            ("graph_node_feature.graph_token", "token"),
            ("graph_attn_bias.edge_encoder", "edge"),
            ("graph_attn_bias.edge_dis_encoder", "distance"),
            ("graph_attn_bias.spatial_pos_encoder", "spatial"),
            ("graph_attn_bias.graph_token_virtual_distance", "virtual"),
        ):
            module = self.root.get_submodule(new)
            for name in module.state_dict():
                names[f"root_module.{old}.{name}"] = f"root.{new}.{name}"
        for index in range(self.config.root.num_layers):
            for old, new in (
                ("self_attn.qkv_proj", "attn.attention.proj_qkv"),
                ("self_attn.out_proj", "attn.attention.proj_out"),
                ("self_attn_layer_norm", "norm1"),
                ("mlp.0", "ffn.0.up_proj"),
                ("mlp.3", "ffn.0.down_proj"),
                ("final_layer_norm", "norm2"),
            ):
                for suffix in ("weight", "bias"):
                    names[f"root_module.layers.{index}.{old}.{suffix}"] = (
                        f"root.blocks.{index}.{new}.{suffix}"
                    )
        names["embedder._extra_embeddings"] = "formula.extra.weight"
        names["embedder.int_to_feat_matrix"] = "formula.table"
        for name in ("formula_mapper", "token_mapper"):
            for suffix in ("weight", "bias"):
                names[f"{name}.{suffix}"] = f"{name}.{suffix}"
        names["fragment_decoder.slots"] = "queries.embedding.weight"
        for name in self.queries.layers.state_dict():
            names[f"fragment_decoder.decoder_layers.{name}"] = f"queries.layers.{name}"
        for suffix in ("weight", "bias"):
            names[f"frag_card_mapper.{suffix}"] = f"cardinality.{suffix}"
        for old, new in (
            ("inten_decoder", "decoder"),
            ("inten_encoder", "encoder"),
            ("output_map", "output"),
        ):
            for name in self.get_submodule(new).state_dict():
                names[f"{old}.{name}"] = f"{new}.{name}"
        if set(names.values()) != set(self.state_dict()):
            raise ValueError("Native GLACIER state differs from its checkpoint schema.")
        return names

    def fragment(self, batch: GlacierBatch) -> Fragmentation:
        """Embed the molecule and select the predicted fragments.

        Args:
          batch: Complete prepared chemical inputs.

        Returns:
          fragmentation: Embeddings, all slot-layer logits and the chosen fragments.

        """
        context = self.conditions(
            batch.collision_energy,
            batch.adduct,
            batch.instrument,
        )
        features = batch.graph.features
        # Keep the three concatenation nodes and their released adjoint order.
        adduct_width = self.conditions.adduct.shape[-1]
        collision_width = self.conditions.collision_missing.numel()
        for part in (
            context[:, :adduct_width],
            context[:, adduct_width : adduct_width + collision_width],
            context[:, adduct_width + collision_width :],
        ):
            features = torch.cat(
                [features, part[:, None, :].expand(-1, features.shape[1], -1)],
                dim=-1,
            )
        root, atoms = self.root(
            features,
            bias=batch.graph.bias,
            spatial=batch.graph.spatial,
            degrees=batch.graph.degrees,
            paths=batch.graph.paths,
        )
        root = root.unsqueeze(1)
        padding = (
            torch.arange(atoms.shape[1], device=atoms.device)[None, :]
            >= batch.atom_counts[:, None]
        )
        query_root = self.formula_mapper(
            torch.cat([root, self.formula(batch.root_forms)[:, None, :]], dim=-1),
        )
        slot_states = self.queries(
            query_root,
            atoms,
            torch.nn.functional.pad(padding, (1, 0), value=False),
        )
        cards = self.cardinality(slot_states)
        logits = torch.einsum("nbij,bkj->nbik", slot_states, atoms)
        with torch.no_grad():
            removed = self.breakpoints.select(logits[-1], cards[-1], batch.atom_counts)
            patterns, counts = self.breakpoints.patterns(
                removed,
                batch.adjacency,
                batch.atom_counts,
            )
            valid = (
                torch.arange(patterns.shape[1], device=patterns.device)[None, :]
                < counts[:, None]
            )
            fragments = patterns[valid].bool()
        return Fragmentation(
            root=root,
            atoms=atoms,
            logits=logits,
            cards=cards,
            fragments=fragments,
            counts=counts,
        )

    @override
    def forward(self, batch: GlacierBatch) -> GlacierOutput:
        part = self.fragment(batch)
        output = self.intensities(
            **part.fields(batch),
            masks=part.fragments,
            counts=part.counts,
        )
        return GlacierOutput(
            breakpoints_pred=BreakpointOutput(
                frag_logits=part.logits,
                frag_card_logits=part.cards,
            ),
            inten_pred_end_to_end=IntensityOutput(output=output),
            frags_pred=FragmentOutput(
                fragments=part.fragments,
                fragment_count=part.counts,
            ),
        )

    def intensities(
        self,
        *,
        root: Tensor,
        atoms: Tensor,
        root_forms: Tensor,
        atom_forms: Tensor,
        atom_counts: Tensor,
        atom_hs: Tensor,
        total_hs: Tensor,
        adjacency: Tensor,
        adduct_shifts: Tensor,
        masses: Tensor,
        masks: Tensor,
        counts: Tensor,
    ) -> Tensor:
        """Decode every fragment, ion-group and permitted hydrogen-shift output.

        Args:
          root: Root graph tokens.
          atoms: Padded atom embeddings.
          root_forms: Root element counts.
          atom_forms: Packed atom element counts.
          atom_counts: Real atom counts per molecule.
          atom_hs: Padded atom hydrogen counts.
          total_hs: Total molecular hydrogen counts.
          adjacency: Padded chemical bond matrices.
          adduct_shifts: Two ion-group mass offsets per molecule.
          masses: Padded atom masses, including attached hydrogens.
          masks: Packed fragment membership masks.
          counts: Fragment counts per molecule.

        Returns:
          spectrum: All padded fragment mass/intensity pairs.

        """
        radius = self.config.max_broken
        molecule = torch.arange(len(root), device=root.device).repeat_interleave(counts)
        mask = masks.bool()
        hs = (mask.float() * atom_hs[molecule].float()).sum(1)
        remove = hs.clamp(max=radius)
        add = (total_hs[molecule].float() - hs).clamp(max=radius)
        crossings = mask[:, :, None] & ~mask[:, None, :]
        broken = (adjacency[molecule] * crossings.float()).sum((1, 2))
        formula = (
            pad_atoms(atom_forms, atom_counts).repeat_interleave(counts, 0)
            * masks.unsqueeze(-1).float()
        ).sum(1)
        formula = pad_atoms(formula, counts)
        difference = root_forms[:, None, :] - formula
        encoded = self.formula(formula)
        encoded_difference = self.formula(difference)
        broken = torch.nn.functional.one_hot(
            pad_atoms(broken, counts).clamp(max=radius).long(),
            num_classes=radius + 1,
        ).float()
        width = formula.shape[1]
        tokens = self.token_mapper(
            torch.cat(
                [root.expand(-1, width, -1), encoded, encoded_difference, broken],
                dim=-1,
            ),
        )
        padding = torch.arange(width, device=root.device)[None, :] >= counts[:, None]
        padded_masks = pad_atoms(masks, counts)
        padded_masks = padded_masks | padding[:, :, None]
        hidden = self.decoder(
            tgt=tokens,
            memory=atoms,
            memory_mask=~padded_masks.repeat_interleave(self.config.heads, 0),
            tgt_key_padding_mask=padding,
        )
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        shift = (
            torch.arange(-radius, radius + 1, device=root.device)
            * SIMULATOR_HYDROGEN_MASS
        )
        mass = pad_atoms((masses[molecule] * masks.float()).sum(-1), counts)
        mass = (
            mass[:, :, None, None]
            + shift[None, None, None, :]
            + adduct_shifts[:, None, :, None]
        )
        mass = torch.where(mass > 0, mass, torch.zeros_like(mass))
        positions = torch.arange(2 * radius + 1, device=root.device)[None, None, :]
        valid = (positions <= pad_atoms(add, counts)[:, :, None] + radius) & (
            positions >= -pad_atoms(remove, counts)[:, :, None] + radius
        )
        valid = (
            (valid & ~padding[:, :, None])
            .unsqueeze(-2)
            .expand(-1, -1, 2, -1)
            .reshape(len(root), width, -1)
        )
        values = (
            self.output(hidden).sigmoid().masked_fill(~valid, 0).reshape(len(root), -1)
        )
        return torch.stack([mass.reshape(len(root), -1), values], dim=-1)


class FragmentKernel(nn.Module):
    """Divide queries before BMM and retain same-dtype masked softmax."""

    class Config(Fig["FragmentKernel"]):
        """Use the source query-scaling and padding arithmetic."""

        @classmethod
        def cost(cls, **kwargs: object) -> Cost:
            """Use the shared two-product logical attention accounting.

            Args:
              **kwargs: Complete geometry and dtype from the projection owner.

            Returns:
              execution: Shared attention-kernel cost.

            """
            return cost(SdpaFused.Config(), **kwargs)

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        padding: Tensor,
        dropout_p: float = 0.0,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        length, batch, heads, width = q.shape
        query, key, value = (
            projection.contiguous().view(length, batch * heads, width).transpose(0, 1)
            for projection in (q, k, v)
        )
        scores = torch.bmm(query / width**0.5, key.transpose(-2, -1))
        mask = padding.view(batch, 1, 1, length).expand(-1, heads, -1, -1)
        mask = mask.reshape(batch * heads, 1, length)
        bias = torch.zeros_like(mask, dtype=query.dtype)
        _ = bias.masked_fill_(mask, float("-inf"))
        probabilities = functional.dropout(
            (scores + bias).softmax(dim=-1),
            p=dropout_p,
            training=self.training,
        )
        return (
            torch.bmm(probabilities, value)
            .transpose(0, 1)
            .contiguous()
            .view(length, batch, heads, width)
        )


class FragmentAttention(PeakAttention):
    """Use shared projections with the source's sequence-major affine calls."""

    class Config(Makes["FragmentAttention"], PeakAttention.Config):
        attention: Attention.Config = field(
            default_factory=lambda: Attention.Config(
                dropout=0.1,
                proj_qkv=Linear.Config(bias=True, init_weight=nn.init.zeros_),
                proj_out=Linear.Config(bias=True, init_bias=None),
                attn_kernel=FragmentKernel.Config(),
            ),
        )
        """Shared projections and exact query-scaled eager fragment kernel."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        _ = nn.init.xavier_uniform_(self.attention.proj_qkv.weight)
        if self.attention.proj_out.bias is not None:
            _ = nn.init.zeros_(self.attention.proj_out.bias)

    @override
    def reset_parameters(self) -> None:
        """Retain output draws before packed Xavier initialization."""
        self.attention.proj_out.reset_parameters()
        _ = nn.init.xavier_uniform_(self.attention.proj_qkv.weight)
        if self.attention.proj_qkv.bias is not None:
            _ = nn.init.zeros_(self.attention.proj_qkv.bias)
        if self.attention.proj_out.bias is not None:
            _ = nn.init.zeros_(self.attention.proj_out.bias)

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        padding = kwargs.pop("padding", None)
        if not isinstance(padding, Tensor):
            raise TypeError("Fragment attention requires a padding tensor.")
        attended = self.attention(input.transpose(0, 1), padding=padding)
        return self.drop(attended.transpose(0, 1))


def fragment_feed_forward() -> Sequential.Config:
    """Return the released two affine maps, rectification and both dropout sites.

    Returns:
      config: Ungated SwiGLU with a ReLU, then one dropout on the residual update.

    """
    swiglu = SwiGLU.Config(gate=False, act=relu, round_to=1)
    swiglu.up_proj = Linear.Config(init_bias=None, bias=True)
    swiglu.middle = Dropout.Config(p=0.1)
    swiglu.down_proj = Linear.Config(init_bias=None, bias=True)
    return Sequential.Config(elements=[swiglu, Dropout.Config(p=0.1)])


class FragmentBlock(TransformerBlock):
    """Compose post-attention and post-feed-forward normalization in Priml."""

    class Config(Makes["FragmentBlock"], TransformerBlock.Config, kw_only=False):
        prenorm: bool = False
        """Normalize after each residual addition."""

        attn: Makeable[TensorModule] = field(default_factory=FragmentAttention.Config)
        """Query-scaled eager attention with a packed affine."""

        ffn: Makeable[TensorModule] = field(default_factory=fragment_feed_forward)
        """Two affine projections, rectification and both dropout sites."""

        norm1: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Affine normalization after attention."""

        norm2: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Affine normalization after the feed-forward residual."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MolecularGraph:
    """Carry ordered graph topology and its concatenated node features."""

    topology: GraphBatch
    features: Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationBatch:
    """Carry the full root/fragment neural input, including mutable formula counts."""

    fragments: MolecularGraph
    roots: MolecularGraph
    root_index: Tensor
    broken: Tensor
    collision_energy: Tensor
    precursor: Tensor
    adduct: Tensor
    instrument: Tensor
    root_forms: Tensor
    fragment_forms: Tensor


def hidden_stack() -> Sequential.Config:
    """Return the released input affine, rectification and hidden dropout.

    Returns:
      config: Elements whose widths the owning head pushes.

    """
    return Sequential.Config(
        elements=[
            Linear.Config(init_bias=None, bias=True),
            Activation.Config(act=relu),
            Dropout.Config(p=0.2),
        ],
    )


class SharedRootModel(ReleasedModel, ABC):
    """Reject checkpoints whose two copies of the shared root encoder disagree."""

    @override
    def load_released_state(self, state: Mapping[str, Tensor]) -> None:
        """Reject conflicting shared-root tensors before the strict atomic load."""
        for name in self.checkpoint_names():
            if name.startswith("gnn."):
                alias = "root_module." + name.removeprefix("gnn.")
                if (
                    name in state
                    and alias in state
                    and not torch.equal(state[name], state[alias])
                ):
                    raise ValueError(f"Shared root checkpoint tensors differ: {name}.")
        super().load_released_state(state)


class MoleculeMLP(nn.Module):
    """Compose the released single-hidden-layer head from Priml affine leaves."""

    class Config(Fig["MoleculeMLP"]):
        channels_in: int = -1
        """Conditioned graph and formula feature width."""

        channels_out: int = -1
        """Head output width."""

        _: KW_ONLY

        channels_hidden: int = 512
        """Rectified hidden width."""

        hidden: Sequential.Config = field(default_factory=hidden_stack)
        """Input affine, rectification and dropout composition."""

        discarded: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Unused middle affine whose construction consumes released random draws."""

        output: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Final affine projection."""

        @override
        def finalize(self) -> Self:
            if self.hidden.channels_in == -1:
                self.hidden.channels_in = self.channels_in
            assert isinstance(self.hidden.elements, list)
            projection = self.hidden.elements[0]
            if isinstance(projection, ChannelsOut) and projection.channels_out == -1:
                projection.channels_out = self.channels_hidden
            self.discarded.channels_in = self.channels_hidden
            self.discarded.channels_out = self.channels_hidden
            self.output.channels_in = self.channels_hidden
            self.output.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Compose only the two registered affine paths and hidden operations.

            Args:
              seq_len: Feature rows per example.
              batch_size: Examples in this invocation.
              dtype: Floating operand dtype; None uses Torch's default.
              **kwargs: Cost metadata forwarded to the executed children.

            Returns:
              execution: Hidden affine/ReLU/dropout and output affine cost.

            """
            # The discarded affine consumes initialization draws but is neither
            # registered nor executed, so it owns no invocation parameters.
            return cost(
                self.hidden,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + cost(
                self.output,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.hidden: Sequential = config.hidden.make()
        _ = config.discarded.make()
        self.output: Linear = config.output.make()

    @override
    def forward(self, features: Tensor) -> Tensor:
        return self.output.forward(self.hidden.forward(features))


def released_ggnn() -> GGNN.Config:
    """Return the released ICE message network: four typed bond maps and a GRU.

    Returns:
      config: GGNN whose edge maps start from torch's default bias draw and are
        redrawn with a sqrt(2)-gain Xavier normal after the GRU is redrawn.

    """
    config = GGNN.Config()
    config.channels_hidden = 256
    config.num_layers = 4
    config.num_edge_types = 4
    config.edge.init_bias = None
    config.init_edge = partial(nn.init.xavier_normal_, gain=2**0.5)
    return config


def intensity_ggnn() -> GGNN.Config:
    """Return the released message network with the six steps ICE intensity uses.

    Returns:
      config: The released GGNN with six recurrent steps.

    """
    config = released_ggnn()
    config.num_layers = 6
    return config


def ggnn_checkpoint_names(messages: GGNN) -> dict[str, str]:
    """Declare every edge map and GRU gate tensor in released registration order.

    Args:
      messages: The built message network.

    Returns:
      names: Each released tensor name, which is also its native name.

    """
    names: dict[str, str] = {}
    for index, linear in enumerate(messages.linears):
        names[f"linears.{index}.weight"] = f"linears.{index}.weight"
        if linear.bias is not None:
            names[f"linears.{index}.bias"] = f"linears.{index}.bias"
    for suffix in ("weight_ih", "weight_hh"):
        names[f"gru.{suffix}"] = f"gru.{suffix}"
    if messages.gru.bias:
        for suffix in ("bias_ih", "bias_hh"):
            names[f"gru.{suffix}"] = f"gru.{suffix}"
    return names


class MoleculeEncoder(nn.Module):
    """Project prepared atom features and apply the released shared GGNN."""

    class Config(Fig["MoleculeEncoder"]):
        channels_in: int = -1
        """Prepared atoms plus physical condition features."""

        channels_out: int = -1
        """Projected and recurrent node-state width."""

        _: KW_ONLY

        projection: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Initial prepared-feature affine map."""

        messages: GGNN.Config = field(default_factory=intensity_ggnn)
        """Shared typed-edge maps and recurrent message updates."""

        @override
        def finalize(self) -> Self:
            self.projection.channels_in = self.channels_in
            self.projection.channels_out = self.channels_out
            self.messages.channels_hidden = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            edge_counts: tuple[int, ...],
            **kwargs: object,
        ) -> Cost:
            """Compose prepared-feature projection and topology-sized shared messages.

            Args:
              seq_len: Packed real nodes per example, not padded atom slots.
              batch_size: Examples represented by the packed node rows.
              dtype: Floating operand dtype; None uses Torch's default.
              edge_counts: Actual whole-invocation directed edges per declared type.
              **kwargs: Cost metadata forwarded to the executed children.

            Returns:
              execution: One owned projection and the recurrent message composition.

            """
            return cost(
                self.projection,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + cost(
                self.messages,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                edge_counts=edge_counts,
                channels=self.channels_out,
                **kwargs,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.projection = config.projection.make()
        self.messages = config.messages.make()

    @override
    def forward(self, topology: GraphBatch, features: Tensor) -> Tensor:
        return self.messages(topology, self.projection(features))


class GraphGenerator(SharedRootModel):
    """Predict leaving-atom probabilities using the public ICE generator family."""

    class Config(Fig["GraphGenerator"]):
        channels_in: int = 46
        """Prepared root and fragment atom width, including random-walk features."""

        channels_hidden: int = 512
        """Shared graph-encoder and output-head hidden width."""

        num_elements: int = 18
        """Element counts per molecular formula."""

        max_broken: int = 6
        """Largest broken-bond count, also representing all larger counts."""

        formula: FormulaEncoding.Config = field(default_factory=FormulaEncoding.Config)
        """Absolute-sine elemental-count table and learned oversized-count token."""

        conditions: GraphConditions.Config = field(
            default_factory=GraphConditions.Config,
        )
        """Frozen adduct, collision and instrument context."""

        encoder: MoleculeEncoder.Config = field(default_factory=MoleculeEncoder.Config)
        """One graph encoder shared by root and fragment evaluations."""

        head: MoleculeMLP.Config = field(default_factory=MoleculeMLP.Config)
        """Per-atom leaving-probability affine composition."""

        @override
        def finalize(self) -> Self:
            self.encoder.channels_in = (
                self.channels_in + self.conditions.copy_tree().finalize().channels_out
            )
            self.encoder.channels_out = self.channels_hidden
            self.head.channels_in = (
                3 * self.channels_hidden
                + self.max_broken
                + 1
                + 2
                * self.num_elements
                * self.formula.copy_tree().finalize().channels_out
            )
            self.head.channels_hidden = self.channels_hidden
            self.head.channels_out = 1
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.formula = config.formula.make()
        self.broken = nn.Parameter(
            torch.eye(config.max_broken + 1),
            requires_grad=False,
        )
        self.conditions = config.conditions.make()
        self.encoder = config.encoder.make()
        self.root_encoder = self.encoder
        self.head = config.head.make()

    @override
    def checkpoint_names(self) -> dict[str, str]:
        """Declare the complete released schema, including shared encoder aliases."""
        names = {
            "broken_onehot": "broken",
            "adduct_embedder": "conditions.adduct",
            "collision_embedder_denominators": "conditions.collision_denominators",
            "collision_embed_merged": "conditions.collision_missing",
            "instrument_embedder": "conditions.instrument",
            "embedder._extra_embeddings": "formula.extra.weight",
            "embedder.int_to_feat_matrix": "formula.table",
        }
        for released, native in (("gnn", "encoder"), ("root_module", "root_encoder")):
            for suffix in ("weight", "bias"):
                names[f"{released}.input_project.{suffix}"] = (
                    f"{native}.projection.{suffix}"
                )
            for old, new in ggnn_checkpoint_names(self.encoder.messages).items():
                names[f"{released}.gnn.model.{old}"] = f"{native}.messages.{new}"
        for released, native in (
            ("input_layer", "hidden.0"),
            ("output_layer", "output"),
        ):
            for suffix in ("weight", "bias"):
                names[f"output_map.{released}.{suffix}"] = f"head.{native}.{suffix}"
        if set(names.values()) != set(self.state_dict()):
            raise ValueError(
                "Native generator state differs from its checkpoint schema.",
            )
        return names

    @override
    def forward(self, batch: GenerationBatch) -> Tensor:
        """Return padded leaving-atom probabilities for every prepared fragment."""
        context = self.conditions(
            batch.collision_energy,
            batch.adduct,
            batch.instrument,
        )
        root_context = context.repeat_interleave(batch.roots.topology.node_counts, 0)
        root_states = self.root_encoder(
            batch.roots.topology,
            torch.cat([batch.roots.features, root_context], dim=-1),
        )
        roots = batch.roots.topology.average(root_states)[batch.root_index]
        counts = batch.fragments.topology.node_counts
        root_atoms = roots.repeat_interleave(counts, 0)
        fragment_context = context[batch.root_index].repeat_interleave(counts, 0)
        fragment_states = self.encoder(
            batch.fragments.topology,
            torch.cat([batch.fragments.features, fragment_context], dim=-1).float(),
        )
        fragment_atoms = batch.fragments.topology.average(
            fragment_states,
        ).repeat_interleave(counts, 0)
        broken = batch.broken.clamp(max=self.config.max_broken)
        broken_atoms = self.broken[broken.repeat_interleave(counts, 0).long()]
        difference = batch.root_forms[batch.root_index] - batch.fragment_forms
        formula = self.formula(batch.fragment_forms).repeat_interleave(counts, 0)
        difference = self.formula(difference).repeat_interleave(counts, 0)
        features = torch.cat(
            [
                root_atoms,
                root_atoms - fragment_atoms,
                fragment_states,
                broken_atoms,
                formula,
                difference,
            ],
            dim=1,
        )
        return pad_atoms(self.head(features).sigmoid(), counts).squeeze(-1)


@dataclass(frozen=True, slots=True, kw_only=True)
class IntensityBatch(GenerationBatch):
    """Carry root/fragment graphs, padded formulas and all possible shifted masses."""

    fragment_counts: Tensor
    max_add_h: Tensor
    max_remove_h: Tensor
    masses: Tensor


def fragment_head() -> MoleculeMLP.Config:
    """Return the fragment feature head, dropping hidden units at the lighter rate.

    Returns:
      config: A molecule head whose hidden dropout is 0.1.

    """
    head = MoleculeMLP.Config()
    dropout = (
        head.hidden.elements[2] if isinstance(head.hidden.elements, list) else None
    )
    assert isinstance(dropout, Dropout.Config)
    dropout.p = 0.1
    return head


class GraphIntensity(SharedRootModel):
    """Predict binned spectra and attribute their intensities to graph fragments."""

    class Config(Fig["GraphIntensity"]):
        channels_in: int = 32
        """Prepared root and fragment atom features."""

        channels_hidden: int = 256
        """Shared graph, fragment-MLP and attention residual width."""

        num_elements: int = 18
        """Element counts per padded root/fragment formula."""

        max_broken: int = 6
        """Largest distinct broken-bond count."""

        formula: FormulaEncoding.Config = field(default_factory=FormulaEncoding.Config)
        """Absolute-sine elemental-count features with one oversized-count token."""

        conditions: GraphConditions.Config = field(
            default_factory=GraphConditions.Config,
        )
        """Frozen physical adduct, collision and instrument context."""

        encoder: MoleculeEncoder.Config = field(
            default_factory=lambda: MoleculeEncoder.Config(
                messages=released_ggnn(),
            ),
        )
        """Shared root and fragment encoder, with four recurrent steps."""

        intermediate: MoleculeMLP.Config = field(default_factory=fragment_head)
        """Conditioned fragment features projected to attention residual width."""

        num_layers: int = 3
        """Independent attention blocks copied from one initialized prototype."""

        block: FragmentBlock.Config = field(default_factory=FragmentBlock.Config)
        """Post-norm attention and feed-forward composition."""

        max_hydrogen_shift: int = 6
        """Both directions of mass-shift output, independent of broken-count width."""

        output: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Intensity affine, copied once for isomer attention without random draws."""

        num_buckets: int = 15_000
        """Linearly spaced m/z bin boundaries."""

        max_mass: float = 1_500
        """Final m/z boundary; larger masses share its clipped final bin."""

        @override
        def finalize(self) -> Self:
            width = self.channels_hidden
            self.encoder.channels_in = (
                self.channels_in + self.conditions.copy_tree().finalize().channels_out
            )
            self.encoder.channels_out = width
            self.intermediate.channels_in = (
                3 * width
                + self.max_broken
                + 1
                + 2
                * self.num_elements
                * self.formula.copy_tree().finalize().channels_out
            )
            self.intermediate.channels_hidden = self.intermediate.channels_out = width
            self.block.channels_in = width
            self.output.channels_in = width
            self.output.channels_out = 2 * (2 * self.max_hydrogen_shift + 1)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.formula = config.formula.make()
        self.broken = nn.Parameter(
            torch.eye(config.max_broken + 1),
            requires_grad=False,
        )
        self.conditions = config.conditions.make()
        self.encoder = config.encoder.make()
        self.root_encoder = self.encoder
        self.intermediate = config.intermediate.make()
        prototype = config.block.make()
        self.blocks = nn.ModuleList(
            [copy.deepcopy(prototype) for _ in range(config.num_layers)],
        )
        self.output: Linear = config.output.make()
        self.isomer = copy.deepcopy(self.output)
        self.boundaries = nn.Parameter(
            torch.from_numpy(np.linspace(0, config.max_mass, config.num_buckets)),
            requires_grad=False,
        )

    @override
    def checkpoint_names(self) -> dict[str, str]:
        """Declare every released tensor, including root aliases and cloned blocks."""
        names = {
            "broken_onehot": "broken",
            "adduct_embedder": "conditions.adduct",
            "collision_embedder_denominators": "conditions.collision_denominators",
            "collision_embed_merged": "conditions.collision_missing",
            "instrument_embedder": "conditions.instrument",
            "inten_buckets": "boundaries",
            "embedder._extra_embeddings": "formula.extra.weight",
            "embedder.int_to_feat_matrix": "formula.table",
        }
        for released, native in (("gnn", "encoder"), ("root_module", "root_encoder")):
            for suffix in ("weight", "bias"):
                names[f"{released}.input_project.{suffix}"] = (
                    f"{native}.projection.{suffix}"
                )
            for old, new in ggnn_checkpoint_names(self.encoder.messages).items():
                names[f"{released}.gnn.model.{old}"] = f"{native}.messages.{new}"
        for released, native in (
            ("input_layer", "hidden.0"),
            ("output_layer", "output"),
        ):
            for suffix in ("weight", "bias"):
                names[f"intermediate_out.{released}.{suffix}"] = (
                    f"intermediate.{native}.{suffix}"
                )
        for index in range(len(self.blocks)):
            prefix = f"trans_layers.{index}"
            native = f"blocks.{index}"
            for suffix in ("weight", "bias"):
                names[f"{prefix}.self_attn.in_proj_{suffix}"] = (
                    f"{native}.attn.attention.proj_qkv.{suffix}"
                )
            for released, leaf in (
                ("self_attn.out_proj", "attn.attention.proj_out"),
                ("linear1", "ffn.0.up_proj"),
                ("linear2", "ffn.0.down_proj"),
                ("norm1", "norm1"),
                ("norm2", "norm2"),
            ):
                for suffix in ("weight", "bias"):
                    names[f"{prefix}.{released}.{suffix}"] = f"{native}.{leaf}.{suffix}"
        for released, native in (
            ("output_map", "output"),
            ("isomer_attn_out", "isomer"),
        ):
            for suffix in ("weight", "bias"):
                names[f"{released}.{suffix}"] = f"{native}.{suffix}"
        if set(names.values()) != set(self.state_dict()):
            raise ValueError(
                "Native intensity state differs from its checkpoint schema.",
            )
        return names

    @override
    def forward(self, batch: IntensityBatch) -> dict[str, Tensor]:
        """Evaluate the released binned-intensity and per-fragment attribution path."""
        context = self.conditions(
            batch.collision_energy,
            batch.adduct,
            batch.instrument,
        )
        root_states = self.root_encoder(
            batch.roots.topology,
            torch.cat(
                [
                    batch.roots.features,
                    context.repeat_interleave(batch.roots.topology.node_counts, 0),
                ],
                dim=-1,
            ),
        )
        roots = batch.roots.topology.average(root_states)[batch.root_index]
        counts = batch.fragments.topology.node_counts
        fragment_states = self.encoder(
            batch.fragments.topology,
            torch.cat(
                [
                    batch.fragments.features,
                    context[batch.root_index].repeat_interleave(counts, 0),
                ],
                dim=-1,
            ).float(),
        )
        fragments = batch.fragments.topology.average(fragment_states)
        broken_valid = (
            torch.arange(batch.broken.shape[-1], device=counts.device)[None, :]
            < batch.fragment_counts[:, None]
        )
        broken = batch.broken[broken_valid].clamp(max=self.config.max_broken)
        hidden = pad_atoms(
            torch.cat(
                [roots, roots - fragments, fragments, self.broken[broken.long()]],
                dim=1,
            ),
            batch.fragment_counts,
        )
        valid = (
            torch.arange(hidden.shape[1], device=counts.device)[None, :]
            < batch.fragment_counts[:, None]
        )
        difference = batch.root_forms[:, None, :] - batch.fragment_forms
        hidden = torch.cat(
            [
                hidden,
                self.formula(batch.fragment_forms),
                self.formula(difference),
            ],
            dim=-1,
        )
        hidden = self.intermediate(hidden)
        for block in self.blocks:
            hidden = block(hidden, padding=~valid)
        shifts = torch.arange(
            2 * self.config.max_hydrogen_shift + 1,
            device=counts.device,
        )[None, None, :]
        valid_shifts = (
            (shifts <= (batch.max_add_h + self.config.max_hydrogen_shift)[:, :, None])
            & (
                shifts
                >= (-batch.max_remove_h + self.config.max_hydrogen_shift)[:, :, None]
            )
            & valid[:, :, None]
        )
        valid_shifts = valid_shifts[:, :, None, None, :].expand(-1, -1, 1, 2, -1)
        geometry = valid_shifts.shape
        output = self.output(hidden).reshape(geometry).transpose(1, 2)
        attention = self.isomer(hidden).reshape(geometry)
        _ = attention.masked_fill_(~valid_shifts, -99_999)
        attention = attention.transpose(1, 2)
        inverse = torch.bucketize(batch.masses, self.boundaries, right=False).clamp(
            max=len(self.boundaries) - 1,
        )
        inverse = inverse[:, None, :, :].expand(attention.shape)
        return attribute_bins(
            output,
            attention,
            inverse,
            valid_shifts,
            num_buckets=self.config.num_buckets,
        )


def attribute_bins(
    logits: Tensor,
    attention: Tensor,
    inverse: Tensor,
    valid: Tensor,
    *,
    num_buckets: int,
) -> dict[str, Tensor]:
    """Normalize isomers within m/z bins, activate bins and recover fragment shares.

    Args:
      logits: Raw intensities, batch/output/fragment/mass-group/shift layout.
      attention: Masked isomer logits in the same layout.
      inverse: Clipped bin indices, expanded to the same layout.
      valid: Valid positions, in batch/fragment/output/mass-group/shift layout.
      num_buckets: Complete output spectrum width, including empty bins.

    Returns:
      predictions: Complete binned spectrum and per-fragment intensity attribution.

    """
    batch, outputs, fragments = logits.shape[:3]
    logits = logits.reshape(batch, outputs, -1)
    attention = attention.reshape(batch, outputs, -1)
    inverse = inverse.reshape(batch, outputs, -1)
    valid = valid.reshape(batch, outputs, -1)
    extent = (batch, outputs, int(inverse.max()) + 1)
    maximum = attention.new_full(extent, -float("inf")).scatter_reduce(
        -1,
        inverse,
        attention,
        "amax",
        include_self=True,
    )
    exponent = (attention - maximum.gather(-1, inverse)).exp()
    denominator = attention.new_zeros(extent).scatter_add_(-1, inverse, exponent)
    weights = exponent / denominator.gather(-1, inverse)
    binned = logits.new_zeros((batch, outputs, num_buckets)).scatter_add_(
        -1,
        inverse,
        weights * logits,
    )
    admitted = (
        valid.long()
        .new_zeros((batch, outputs, num_buckets))
        .scatter_reduce(
            -1,
            inverse,
            valid.long(),
            "amax",
            include_self=False,
        )
        .bool()
    )
    binned = binned.sigmoid().masked_fill(~admitted, 0)
    attributed = torch.take_along_dim(binned, inverse, dim=-1)
    attributed = attributed.reshape(batch, outputs, fragments, -1).transpose(1, 2)
    weights = weights.reshape(batch, outputs, fragments, -1).transpose(1, 2)
    return {"output_binned": binned, "output": attributed * weights}
