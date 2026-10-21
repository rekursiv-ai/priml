"""Every CASMI network that is not a promoted architecture, checked together."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Final, cast, override

import itertools
import math
import re
import warnings

from configgle import Fig, InlineConfig, PartialConfig
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel
from treekle import from_plain

import pytest
import torch

from priml.baselines.casmi2026.model import (
    Block,
    BreakpointOutput,
    Breakpoints,
    Decoder,
    DreamAttention,
    DreamBackbone,
    DreamEncoder,
    DreamFFN,
    DreamsFP,
    Encoder,
    FormulaEncoding,
    FourierFeatures,
    FPNet,
    FragmentAttention,
    FragmentBlock,
    FragmentKernel,
    FragmentOutput,
    GenerationBatch,
    Glacier,
    GlacierBatch,
    GlacierOutput,
    GraphConditions,
    GraphGenerator,
    GraphInput,
    GraphIntensity,
    IntensityBatch,
    IntensityOutput,
    MolecularGraph,
    MoleculeEncoder,
    MoleculeMLP,
    PackedAttention,
    PeakAttention,
    SecondFPNet,
    SinEmb,
    Slots,
    feed_forward,
    pad_atoms,
    precise_float,
    rectified_stack,
    released_ggnn,
    size_stack,
)
from priml.baselines.casmi2026.testing import float32_comparands, set_heads
from priml.cost import cost
from priml.loss.simple_loss import SimpleLoss
from priml.model.attention.kernel import SdpaFused, SdpaNaive
from priml.model.embedding import Embedding
from priml.model.ggnn import GGNN, GraphBatch
from priml.model.graphormer import GraphBlock
from priml.model.linear import Linear
from priml.model.norm import LayerNorm
from priml.model.sequential import Sequential
from priml.model.special import Activation, Dropout, Identity
from priml.model.swiglu import SwiGLU, gelu, relu
from priml.model.transformer.block import TransformerBlock
from priml.optimizers.composite import CompositeOptimizer
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import (
    assert_tensor_golden,
    mismatches,
    read_tensors,
    rng_fingerprint,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class FragmentTargets:
    """Carry packed training fragments and molecule-level segment lengths."""

    masks: Tensor
    counts: Tensor


class LabelledOutput(GlacierOutput):
    """Add the labelled-fragment intensity path to the inference outputs."""

    inten_pred_magma: IntensityOutput | None


def labelled_forward(
    model: Glacier,
    batch: GlacierBatch,
    labels: FragmentTargets,
) -> LabelledOutput:
    """Decode the labelled and the predicted fragments together for training.

    Args:
      model: GLACIER whose predicted fragments join the labelled ones.
      batch: Complete prepared chemical inputs.
      labels: Packed labelled fragment masks and per-molecule counts.

    Returns:
      output: The inference outputs plus the labelled-fragment intensities.

    """
    part = model.fragment(batch)
    masks = torch.cat([labels.masks, part.fragments])
    sizes = torch.cat([labels.counts, part.counts])
    fields = {
        name: torch.cat([value, value], dim=0)
        for name, value in part.fields(batch).items()
    }
    output = model.intensities(**fields, masks=masks, counts=sizes)
    size = len(batch.atom_counts)
    return LabelledOutput(
        breakpoints_pred=BreakpointOutput(
            frag_logits=part.logits,
            frag_card_logits=part.cards,
        ),
        inten_pred_end_to_end=IntensityOutput(output=output[size:]),
        inten_pred_magma=IntensityOutput(output=output[:size]),
        frags_pred=FragmentOutput(
            fragments=part.fragments,
            fragment_count=part.counts,
        ),
    )


def optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Tensor]:
    """Key every optimizer state tensor by its native parameter name.

    Args:
      model: Module whose parameters the optimizer updates.
      optimizer: Optimizer after its updates.

    Returns:
      record: ``optimizer/<parameter>/<state>`` float32 tensors.

    """
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        f"optimizer/{names[id(parameter)]}/{key}": value.detach().clone().float()
        for parameter, state in optimizer.state.items()
        for key, value in state.items()
    }


def assignment(cost: list[list[float]]) -> tuple[list[int], list[int]]:
    """Assign every column to a distinct row at the least total cost.

    Args:
      cost: Rows (query slots) by columns (boundaries), with no more columns than rows.

    Returns:
      rows: The row chosen for each column, first in permutation order among ties.
      columns: Every column index in order.

    """
    chosen = min(
        itertools.permutations(range(len(cost)), len(cost[0])),
        key=lambda rows: sum(cost[row][column] for column, row in enumerate(rows)),
    )
    return list(chosen), list(range(len(cost[0])))


def boundaries(adjacency: Tensor, masks: Tensor) -> dict[str, Tensor]:
    """Find distinct training boundaries using the released signed bit packing.

    Args:
      adjacency: Boolean bond matrices per molecule.
      masks: Fragment-membership rows per molecule.

    Returns:
      boundaries: Float membership rows and counts in signed-packed unique order.

    """
    batch, fragments, atoms = masks.shape
    frontier = (masks[..., None] & adjacency[:, None, :, :]).any(dim=2) & ~masks
    chunks = (atoms + 63) // 64
    if chunks * 64 > atoms:
        frontier = functional.pad(frontier, (0, chunks * 64 - atoms))
    powers = 1 << torch.arange(64, device=adjacency.device, dtype=torch.int64)
    packed = (frontier.view(batch, fragments, chunks, 64).long() * powers).sum(
        dim=-1,
    )
    molecule = torch.arange(batch, device=adjacency.device)[:, None, None].expand(
        -1,
        fragments,
        -1,
    )
    unique = torch.unique(
        torch.cat([molecule, packed], dim=-1).reshape(-1, chunks + 1),
        dim=0,
    )
    identifiers, packed = unique[:, 0], unique[:, 1:]
    sizes = torch.bincount(identifiers, minlength=batch)
    # Signed comparison intentionally drops bit63, as in the active source.
    recovered = ((packed[..., None] & powers) > 0).reshape(-1, chunks * 64)[
        :,
        :atoms,
    ]
    output = torch.zeros(
        batch,
        int(sizes.max()),
        atoms,
        dtype=torch.float32,
        device=adjacency.device,
    )
    starts = sizes.cumsum(0) - sizes
    position = torch.arange(len(unique), device=adjacency.device) - starts[identifiers]
    output[identifiers, position] = recovered.float()
    return {"boundary_mask": output, "unique_boundary_patterns": sizes}


class GlacierCriterion:
    """Apply visible noncontrastive spectral and auxiliary fragment training."""

    class Config(Fig["GlacierCriterion"]):
        num_bins: int = 150_000
        """Sparse spectral-loss bin vocabulary."""

        max_mass: float = 1500
        """Final sparse spectral-loss mass boundary."""

        fragment_weight: float = 0.1
        """Assigned-boundary loss coefficient."""

        auxiliary_decay: float = 0.9
        """Geometric weighting of earlier slot-layer losses."""

        magma_warmup: int = 2500
        """Updates using only the labelled-fragment intensity path."""

        magma_decay: float = 0.9
        """Staircase decrease of labelled-fragment weighting after warmup."""

        magma_decay_steps: int = 2500
        """Updates between labelled-fragment weighting decreases."""

        breakpoints: Breakpoints.Config = field(default_factory=Breakpoints.Config)
        """Shipped equality projection and signed boundary extraction."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.step = 0
        self.breakpoints = config.breakpoints.make()

    def coalesce(self, spectrum: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Keep positive in-range intensities and maximum-pool repeated sparse bins.

        Args:
          spectrum: Padded unbinned mass/intensity pairs per molecule.

        Returns:
          sparse: Batch indices, bin indices and maximum intensities.

        """
        bins = (
            torch.floor(
                spectrum[:, :, 0] * ((self.config.num_bins - 1) / self.config.max_mass),
            ).long()
            + 1
        )
        values = spectrum[:, :, 1].float()
        valid = (values > 0) & (bins >= 0) & (bins < self.config.num_bins)
        if not bool(valid.any()):
            indices = torch.empty(0, dtype=torch.long, device=spectrum.device)
            values = torch.empty(0, dtype=torch.float32, device=spectrum.device)
            return indices, indices, values
        batch = torch.arange(len(spectrum), device=spectrum.device)[:, None].expand_as(
            bins,
        )
        keys, inverse = torch.unique(
            batch[valid] * self.config.num_bins + bins[valid],
            sorted=True,
            return_inverse=True,
        )
        pooled = values.new_full((len(keys),), -float("inf")).scatter_reduce(
            0,
            inverse,
            values[valid],
            "amax",
            include_self=True,
        )
        pooled = torch.where(pooled.isfinite(), pooled, torch.zeros_like(pooled))
        return (
            torch.div(keys, self.config.num_bins, rounding_mode="floor"),
            keys % self.config.num_bins,
            pooled,
        )

    def cosine(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute per-molecule sparse cosine distance using sorted-bin joins.

        Args:
          prediction: Predicted unbinned mass/intensity pairs.
          target: Observed unbinned mass/intensity pairs.

        Returns:
          distance: One sparse cosine distance per molecule.

        """
        pb, pi, pv = self.coalesce(prediction)
        tb, ti, tv = self.coalesce(target)
        pn = prediction.new_zeros(len(prediction), dtype=torch.float32)
        tn = torch.zeros_like(pn)
        _ = pn.scatter_add_(0, pb, pv * pv)
        _ = tn.scatter_add_(0, tb, tv * tv)
        dot = torch.zeros_like(pn)
        if len(pv) and len(tv):
            target_keys = tb * self.config.num_bins + ti
            order = torch.argsort(target_keys)
            target_keys = target_keys[order]
            positions = torch.searchsorted(target_keys, pb * self.config.num_bins + pi)
            matched = (positions < len(target_keys)) & (
                target_keys[positions.clamp_max(len(target_keys) - 1)]
                == pb * self.config.num_bins + pi
            )
            if bool(matched.any()):
                _ = dot.scatter_add_(
                    0,
                    pb[matched],
                    pv[matched] * tv[order][positions[matched]],
                )
        pn, tn = pn.sqrt(), tn.sqrt()
        loss = 1 - dot / (pn * tn + 1e-22)
        return torch.where((pn > 0) & (tn > 0), loss, torch.ones_like(loss))

    @classmethod
    def cross_entropy(
        cls,
        probability: Tensor,
        target: Tensor,
        *,
        normalized: bool = True,
    ) -> Tensor:
        """Cross entropy for normalized atom ranks or unnormalized cardinality logits."""
        logarithm = (
            (probability + 1e-9).log() if normalized else probability.log_softmax(-1)
        )
        return -(target * logarithm).sum(-1)

    def fragment_loss(
        self,
        logits: Tensor,
        cards: Tensor,
        batch: GlacierBatch,
        targets: FragmentTargets,
    ) -> Tensor:
        """Assign query slots to unique chemical boundaries with the exact LAP cost.

        Args:
          logits: Per-slot atom-removal logits.
          cards: Per-slot cardinality logits.
          batch: Prepared molecular counts and bond matrices.
          targets: Packed labelled fragment masks and counts.

        Returns:
          loss: Mean assigned atom/cardinality cross entropy.

        """
        boundary = boundaries(
            batch.adjacency > 0,
            pad_atoms(targets.masks, targets.counts),
        )
        truth = boundary["boundary_mask"]
        counts = boundary["unique_boundary_patterns"]
        ranked = self.breakpoints.rank(logits, batch.atom_counts)
        truth_cards = functional.one_hot(truth.sum(-1).long(), num_classes=4)
        paired = (ranked.unsqueeze(2) * truth_cards[:, None, :, None, :]).sum(-1)
        rank_cost = self.cross_entropy(
            functional.normalize(paired, p=1, dim=-1),
            functional.normalize(truth[:, None].expand_as(paired), p=1, dim=-1),
        )
        card_cost = self.cross_entropy(
            cards.unsqueeze(2),
            truth_cards.unsqueeze(1),
            normalized=False,
        )
        cost = rank_cost + card_cost
        chosen = torch.zeros_like(cost)
        # Assignment is discrete; the selected logits retain their gradients.
        for index in range(len(cost)):
            rows, columns = assignment(
                from_plain(
                    cost[index, :, : int(counts[index])].detach().cpu().tolist(),
                    list[list[float]],
                ),
            )
            chosen[index, rows, columns] = 1
        flat = ranked.reshape(len(logits), logits.shape[1], -1)
        selected = (
            (flat.transpose(1, 2) @ chosen)
            .transpose(1, 2)
            .reshape(len(logits), truth.shape[1], logits.shape[-1], 4)
        )
        selected = (selected * truth_cards.unsqueeze(-2)).sum(-1)
        selected_cards = (cards.transpose(1, 2) @ chosen).transpose(1, 2)
        loss = self.cross_entropy(
            functional.normalize(selected, p=1, dim=-1),
            functional.normalize(truth, p=1, dim=-1),
        ) + self.cross_entropy(selected_cards, truth_cards, normalized=False)
        padding = (
            counts[:, None]
            <= torch.arange(truth.shape[1], device=truth.device)[None, :]
        )
        return (loss.masked_fill(padding, 0).sum(-1) / counts).mean()

    def __call__(
        self,
        prediction: LabelledOutput,
        *,
        batch: GlacierBatch,
        labels: FragmentTargets,
        spectrum: Tensor,
    ) -> Tensor:
        """Advance the MAGMA schedule and combine all auxiliary slot-layer losses."""
        self.step += 1
        magma = prediction["inten_pred_magma"]
        if magma is None:
            raise ValueError("Training requires labelled-fragment intensity output.")
        weight = 1.0
        if self.step >= self.config.magma_warmup:
            weight = self.config.magma_decay ** (
                (self.step - self.config.magma_warmup) // self.config.magma_decay_steps
            )
        end_to_end = self.cosine(
            prediction["inten_pred_end_to_end"]["output"],
            spectrum,
        ).mean()
        labelled = self.cosine(magma["output"], spectrum).mean()
        breakpoints = prediction["breakpoints_pred"]
        losses = [
            self.fragment_loss(logits, cards, batch, labels)
            * self.config.auxiliary_decay
            ** (len(breakpoints["frag_logits"]) - index - 1)
            for index, (logits, cards) in enumerate(
                zip(
                    breakpoints["frag_logits"],
                    breakpoints["frag_card_logits"],
                    strict=True,
                ),
            )
        ]
        fragment_loss = sum(losses)
        if weight == 1:
            return labelled + self.config.fragment_weight * fragment_loss
        return (
            weight * labelled
            + (1 - weight) * end_to_end
            + self.config.fragment_weight * weight * fragment_loss
        )


def generation_multiplier(
    step: int,
    *,
    warmup: int = 1_000,
    decay: float = 0.7214,
    decay_steps: int = 5_000,
) -> float:
    """Match the generator's exponential warmup and staircase decay."""
    if step >= warmup:
        return decay ** ((step - warmup) // decay_steps)
    return 1 - math.exp(-step / warmup)


_CWD: Final = Path(__file__).resolve().parent

_BASELINE: Final = _CWD


def tiny_model() -> FPNet.Config:
    """Preserve the reference recipe while reducing every configurable geometry."""
    config = FPNet.Config()
    config.channels_hidden = 8
    config.channels_out = 7
    config.num_layers = 1  # The golden holds one complete block.
    set_block_heads(config.block, 2)
    config.adduct.channels_in = 11
    config.head.channels_hidden = 9
    config.head_elements.channels_hidden = 10
    config.head_elements.channels_out = 6
    return config


def set_block_heads(block: Block.Config, heads: int) -> None:
    """Set the attention head count, which only the kernel config owns."""
    assert isinstance(block.attn, PeakAttention.Config)
    block.attn.attention.num_heads = heads


def inputs() -> tuple[Tensor, ...]:
    """Construct three distinct spectra with five peak slots and both polarities."""
    mz = torch.arange(15, dtype=torch.float32).reshape(3, 5) + 20
    return (
        mz,
        torch.arange(15, dtype=torch.float32).reshape(3, 5) / 15,
        torch.tensor([[False] * 5, [False, False, True, True, True], [False] * 5]),
        torch.tensor([100.0, 200.0, 30.0]),
        torch.tensor([0, 2, 4]),
        torch.tensor([10.0, 0.0, 30.0]),
        torch.tensor([1.0, 0.0, 1.0]),
        torch.tensor([0.25, 0.5, 1.0]),
        torch.tensor([1.0, -1.0, 1.0]),
    )


def test_forward_emits_fingerprints_and_elements() -> None:
    model = tiny_model().make().eval()
    fingerprints, elements = model.forward(*inputs())
    assert fingerprints.shape == (3, 7)
    assert elements.shape == (3, 6)
    mz, it, padding, prec, ad, ce, ce_known, merged, mode = inputs()
    tokens, pad = model.encode(
        mz,
        it=it,
        pad=padding,
        prec=prec,
        ad=ad,
        ce=ce,
        ce_known=ce_known,
        merged=merged,
        mode=mode,
    )
    assert tokens.shape == (3, 6, 8)
    assert torch.equal(pad[:, 0], torch.zeros(3, dtype=torch.bool))


def test_fpnet_composes_priml_primitives() -> None:
    model = tiny_model().make()
    assert isinstance(model.pk, Linear)
    assert isinstance(model.int_emb, Linear)
    assert isinstance(model.adduct, Embedding)
    assert isinstance(model.norm, LayerNorm)
    assert isinstance(model.head, SwiGLU)
    assert isinstance(model.head_elements, SwiGLU)
    assert isinstance(model.blocks[0], TransformerBlock)


@pytest.mark.parametrize(
    "config",
    [Block.Config(), GraphBlock.Config(), FragmentBlock.Config()],
)
def test_transformer_finalization_preserves_an_injected_feedforward(
    config: TransformerBlock.Config,
) -> None:
    config.channels_in = 8
    config.ffn = Identity.Config()
    finalized = config.copy_tree().finalize()
    assert isinstance(finalized.ffn, Identity.Config)
    assert finalized.ffn.channels_in == finalized.ffn.channels_out == 8


def test_padding_values_do_not_change_pooled_predictions() -> None:
    model = tiny_model().make().eval()
    original = inputs()
    expected = model.forward(*original)
    changed = list(original)
    changed[0] = changed[0].masked_fill(original[2], 900)
    changed[1] = changed[1].masked_fill(original[2], 200)
    actual = model.forward(*changed)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))


def test_parameter_registration_order() -> None:
    model = tiny_model().make()
    assert list(model._modules) == [
        "mz_emb",
        "nl_emb",
        "int_emb",
        "pk",
        "prec_emb",
        "adduct",
        "ce_emb",
        "gl",
        "blocks",
        "norm",
        "head",
        "head_elements",
    ]
    source_names = list(model.checkpoint_names())
    assert source_names[11:23] == [
        f"blocks.0.{name}.{suffix}"
        for name in ("n1", "qkv", "o", "n2", "ff.0", "ff.3")
        for suffix in ("weight", "bias")
    ]


@pytest.mark.parametrize("damage", ["missing", "unexpected", "shape", "dtype"])
def test_released_checkpoint_validation_precedes_loading(damage: str) -> None:
    model = tiny_model().make()
    original = {name: tensor.clone() for name, tensor in model.released_state().items()}
    malformed = {name: tensor + 1 for name, tensor in original.items()}
    if damage == "missing":
        del malformed["norm.bias"]
    elif damage == "unexpected":
        malformed["surprise"] = torch.zeros(2, 3)
    elif damage == "shape":
        malformed["norm.bias"] = torch.zeros(2, 3)
    else:
        malformed["norm.bias"] = malformed["norm.bias"].double()
    with pytest.raises(ValueError, match="Checkpoint"):
        model.load_released_state(malformed)
    assert all(
        torch.equal(tensor, original[name])
        for name, tensor in model.released_state().items()
    )


def test_released_checkpoint_remap_loads_every_tensor() -> None:
    model = tiny_model().make()
    expected = {name: tensor + 1 for name, tensor in model.released_state().items()}
    model.load_released_state(expected)
    assert all(
        torch.equal(tensor, expected[name])
        for name, tensor in model.released_state().items()
    )


def test_checkpoint_schema_rejects_an_unmapped_native_tensor() -> None:
    model = tiny_model().make()
    model.register_buffer("unmapped", torch.zeros(2, 3))
    with pytest.raises(ValueError, match="Native FPNet state"):
        _ = model.checkpoint_names()


def test_checkpoint_mapping_rejects_collisions_before_any_weight_changes() -> None:
    model = _CollisionMap(tiny_model().copy_tree().finalize())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    proposed = {name: value + 1 for name, value in model.released_state().items()}
    with pytest.raises(ValueError, match="duplicate native"):
        model.load_released_state(proposed)
    assert not mismatches(before, model.state_dict())


def test_sinusoidal_basis_matches_the_closed_form_expression() -> None:
    config = SinEmb.Config()
    config.channels_out = 6
    config.power = 2.0
    layer = config.make()
    values = torch.tensor([0.0, 11.0])
    phase = values[:, None] * (
        (2 * torch.pi) / torch.pow(10.0, 5.3 * torch.linspace(0, 1, 3).pow(2) - 2)
    )
    assert torch.equal(layer.forward(values), torch.cat([phase.sin(), phase.cos()], -1))


def test_invalid_mass_embedding_geometry_fails_before_training() -> None:
    config = tiny_model()
    config.channels_hidden = 4
    with pytest.raises(ValueError, match="multiple of eight"):
        _ = config.make()


def test_sinusoidal_cost_matches_complete_adjoint() -> None:
    config = SinEmb.Config()
    config.channels_out = 8
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(3, 5, requires_grad=True),
        seq_len=5,
        batch_size=3,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_block_cost_matches_complete_adjoint(dropout: float) -> None:
    config = Block.Config()
    config.channels_in = 8
    set_block_heads(config, 2)
    set_dropout(config, dropout)
    # CPU flash SDPA has no Torch FLOP formula; math exposes the same dense products.
    with sdpa_kernel(SDPBackend.MATH):
        assert_cost_matches_torch(
            config,
            build_input=lambda: torch.randn(3, 5, 8, requires_grad=True),
            seq_len=5,
            batch_size=3,
            dtype=torch.float32,
        )


def test_fpnet_cost_includes_both_heads_and_input_gradients() -> None:
    with sdpa_kernel(SDPBackend.MATH):
        assert_cost_matches_torch(
            tiny_model(),
            build_input=_differentiable_inputs,
            run=_both_heads,
            seq_len=5,
            batch_size=3,
            dtype=torch.float32,
        )


def set_dropout(block: Block.Config, dropout: float) -> None:
    """Set all three drop sites of the block, which the child configs own."""
    assert isinstance(block.attn, PeakAttention.Config)
    assert isinstance(block.ffn, Sequential.Config)
    assert isinstance(block.ffn.elements, list)
    swiglu, residual = block.ffn.elements
    assert isinstance(swiglu, SwiGLU.Config)
    assert isinstance(swiglu.middle, Dropout.Config)
    assert isinstance(residual, Dropout.Config)
    block.attn.drop.p = swiglu.middle.p = residual.p = dropout


def _differentiable_inputs() -> tuple[Tensor, ...]:
    return tuple(
        value.requires_grad_() if value.is_floating_point() else value
        for value in inputs()
    )


def _both_heads(model: torch.nn.Module, values: tuple[Tensor, ...]) -> Tensor:
    assert isinstance(model, FPNet)
    return torch.cat(model(*values), dim=-1)


class _CollisionMap(FPNet):
    @override
    def checkpoint_names(self) -> dict[str, str]:
        names = super().checkpoint_names()
        names["duplicate"] = names["norm.weight"]
        return names


def test_fourier_basis_and_zero_mass_features() -> None:
    config = FourierFeatures.Config()
    config.num_fractional = 2
    config.num_integer = 3
    basis = config.make()
    assert config.channels_out == -1
    assert config.finalized().channels_out == 10
    assert torch.equal(basis.b, torch.tensor([[5_000.0, 2_500.0, 0.5, 1 / 3, 0.25]]))
    # FourierFeatures.forward consumes a scalar mass on its final axis.
    output = basis.forward(torch.zeros(2, 3, 1))
    assert torch.equal(output[..., :5], torch.ones(2, 3, 5))
    assert torch.equal(output[..., 5:], torch.zeros(2, 3, 5))
    exact = torch.zeros(2, 3, dtype=torch.float64)
    assert precise_float(exact) is exact
    reduced = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    promoted = precise_float(reduced)
    assert promoted.dtype == torch.float32
    assert torch.equal(promoted, reduced.float())


def test_rectified_stack_preserves_hidden_dropout_before_activation() -> None:
    config = rectified_stack(2, 4, 0.1)
    size_stack(config, channels_in=3, channels_out=5)
    stack = config.make()
    assert [type(layer) for layer in stack] == [
        Linear,
        Dropout,
        Activation,
        Linear,
        Activation,
    ]
    assert stack(torch.randn(2, 7, 3)).shape == (2, 7, 5)
    assert isinstance(config.elements, list)
    config.elements = config.elements[:-1]
    assert [type(layer) for layer in config.make()] == [
        Linear,
        Dropout,
        Activation,
        Linear,
    ]


def test_mass_attention_and_feedforward_preserve_residual_geometry() -> None:
    config = DreamAttention.Config()
    config.channels_in = 6
    config.attention.num_heads = 2
    attention = config.make().eval()
    input = torch.randn(4, 5, 6)
    # DreamAttention.forward broadcasts query masks over heads and keys.
    query_mask = torch.zeros(4, 1, 5, 1, dtype=torch.bool)
    query_mask[:, :, -2:] = True
    # DreamAttention.forward uses equal self-attention axes and one broadcast head.
    output = attention(input, bias=torch.zeros(4, 1, 5, 5), query_mask=query_mask)
    assert output.shape == input.shape
    assert torch.equal(output[:, -1], output[:, -2])
    ffn = DreamFFN.Config()
    ffn.channels_in = 6
    assert ffn.make()(input).shape == input.shape


def test_released_dream_precision_requires_its_actual_kernel_policy() -> None:
    config = DreamAttention.Config()
    config.attention.attn_kernel = SdpaFused.Config()
    with pytest.raises(TypeError, match=r"requires DreamKernel\.Config"):
        _ = config.precision


def test_feedforward_cost_matches_the_complete_training_invocation() -> None:
    config = DreamFFN.Config()
    config.channels_in = 6
    assert isinstance(config.ffn.middle, Dropout.Config)
    config.ffn.middle.p = 0.0
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(3, 5, 6, requires_grad=True),
        seq_len=5,
        batch_size=3,
        dtype=torch.float32,
    )


def test_feedforward_reset_repeats_the_constructor_draws() -> None:
    config = DreamFFN.Config()
    config.channels_in = 6
    with torch.random.fork_rng(devices=[]):
        _ = torch.manual_seed(7)
        model = config.make()
        initial = {name: value.clone() for name, value in model.state_dict().items()}
        _ = torch.manual_seed(7)
        model.reset_parameters()
        assert not mismatches(initial, model.state_dict())


def test_feedforward_preserves_an_injected_final_initializer() -> None:
    config = DreamFFN.Config()
    config.channels_in = 6
    config.init_final = nn.init.zeros_
    model = config.make()
    assert torch.equal(
        model.ffn.up_proj.weight,
        torch.zeros_like(model.ffn.up_proj.weight),
    )
    assert torch.equal(
        model.ffn.down_proj.weight,
        torch.zeros_like(model.ffn.down_proj.weight),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fourier_cost_counts_only_the_unfrozen_input_adjoint(
    dtype: torch.dtype,
) -> None:
    config = FourierFeatures.Config()
    config.num_fractional = 2
    config.num_integer = 3
    measured = assert_cost_matches_torch(
        config,
        # FourierFeatures.forward requires scalar masses on its final axis.
        build_input=lambda: torch.randn(3, 5, 1, dtype=dtype, requires_grad=True),
        seq_len=5,
        batch_size=3,
        dtype=dtype,
    )
    assert measured.params == 0
    assert (
        measured.bytes_state
        == (config.num_fractional + config.num_integer)
        * torch.tensor(0.0).element_size()
    )


@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_cost_matches_the_complete_masked_training_invocation(
    fp32: bool,
    dtype: torch.dtype,
) -> None:
    config = DreamAttention.Config()
    config.channels_in = 8
    config.attention.num_heads = 2
    config.attention.dropout = 0.0
    config.dropout.p = 0.0
    config.precision.dtype = torch.float32 if fp32 else None
    # CPU math SDPA promotes BF16 internally; the logical byte convention
    # is measured through explicit products at the configured operand dtype.
    config.precision.kernel = SdpaNaive.Config()

    def run(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
        assert isinstance(module, DreamAttention)
        states, bias, mask = inputs
        with torch.autocast(
            "cpu",
            dtype=torch.bfloat16,
            enabled=dtype == torch.bfloat16,
        ):
            return module.forward(states, bias=bias, query_mask=mask)

    with sdpa_kernel(SDPBackend.MATH):
        assert_cost_matches_torch(
            config,
            # DreamAttention.forward broadcasts the bias and mask across heads.
            build_input=lambda: (
                torch.randn(3, 5, 8, requires_grad=True),
                torch.zeros(3, 1, 5, 5, requires_grad=True),
                torch.zeros(3, 1, 5, 1, dtype=torch.bool),
            ),
            run=run,
            seq_len=5,
            batch_size=3,
            dtype=dtype,
        )


def test_complete_both_attention_precision_branches_and_three_updates() -> None:
    golden = _BASELINE / "testdata" / "dream_precision.pt"
    expected = read_tensors(golden)
    actual = {name: expected[name].clone() for name in ("input", "bias", "mask")}
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        for fp32 in (True, False):
            _ = torch.manual_seed(42)
            config = DreamAttention.Config()
            config.channels_in = 8
            config.attention.num_heads = 2
            config.dropout.p = 0.0
            config.precision.dtype = torch.float32 if fp32 else None
            model = config.make()
            label = "fp32" if fp32 else "bf16"
            actual[f"{label}/initial"] = (
                torch.cat(
                    [model.attention.proj_qkv.weight, model.attention.proj_out.weight],
                )
                .detach()
                .clone()
            )
            actual[f"{label}/initial_rng"] = rng_fingerprint()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            losses: list[Tensor] = []
            outputs: list[Tensor] = []
            gradients: list[Tensor] = []
            for _ in range(3):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = model(
                        actual["input"],
                        bias=actual["bias"],
                        query_mask=actual["mask"],
                    )
                    loss = output.float().square().mean()
                outputs.append(output.detach().clone())
                losses.append(loss.detach().clone())
                first, second = (
                    model.attention.proj_qkv.weight,
                    model.attention.proj_out.weight,
                )
                loss.backward()
                assert first.grad is not None
                assert second.grad is not None
                gradients.append(torch.cat([first.grad, second.grad]).detach().clone())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            actual[f"{label}/outputs"] = torch.stack(outputs)
            actual[f"{label}/losses"] = torch.stack(losses)
            actual[f"{label}/gradients"] = torch.stack(gradients)
            actual[f"{label}/final"] = (
                torch.cat(
                    [model.attention.proj_qkv.weight, model.attention.proj_out.weight],
                )
                .detach()
                .clone()
            )
            states = from_plain(
                cast(object, optimizer.state_dict()["state"]),
                dict[int, dict[str, Tensor]],
            )
            assert tuple(states) == (0, 1)
            for name, tensor in states[0].items():
                if name == "step":
                    assert torch.equal(tensor, states[1][name])
                    actual[f"{label}/optimizer/{name}"] = tensor.detach().clone()
                else:
                    actual[f"{label}/optimizer/{name}"] = (
                        torch.cat([tensor, states[1][name]]).detach().clone()
                    )
            model.eval()
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                actual[f"{label}/inference"] = model(
                    actual["input"],
                    bias=actual["bias"],
                    query_mask=actual["mask"],
                ).clone()
            actual[f"{label}/final_rng"] = rng_fingerprint()
    assert_tensor_golden(golden, actual)


DREAMS_GOLDEN: Final = _BASELINE / "testdata" / "dreams.pt"


def tiny_dreams() -> DreamsFP.Config:
    """Keep every selected numerical mechanism while reducing only geometry."""
    config = DreamsFP.Config()
    config.channels_out = 7
    config.num_elements = 6
    config.max_peaks = 4
    for head, hidden in ((config.head, 3), (config.elements, 2), (config.condition, 2)):
        assert isinstance(head.elements, list)
        first = head.elements[0]
        assert isinstance(first, Linear.Config)
        first.channels_out = hidden
    config.adduct.channels_in = 11
    config.ce.channels_out = 6
    config.backbone.channels_fourier = 6
    config.backbone.channels_peak = 2
    config.backbone.fourier = rectified_stack(2, 2, 0.1)
    config.backbone.basis.num_fractional = 2
    config.backbone.basis.num_integer = 3
    config.backbone.encoder.num_layers = (
        1  # The golden holds one complete encoder layer.
    )
    block = config.backbone.encoder.block
    assert isinstance(block, TransformerBlock.Config)
    set_heads(block.attn, 2)
    return config


def test_encoder_cost_counts_each_owned_residual_pair_and_final_norm() -> None:
    config = tiny_dreams().backbone.encoder
    config.channels_in = 8
    config.num_layers = 2

    def run(module: nn.Module, values: tuple[Tensor, ...]) -> Tensor:
        assert isinstance(module, DreamEncoder)
        states, bias, mask = values
        return module.forward(states, bias=bias, query_mask=mask)

    with sdpa_kernel(SDPBackend.MATH):
        assert_cost_matches_torch(
            config,
            # DreamEncoder.forward uses dense self-attention and broadcast masks.
            build_input=lambda: (
                torch.randn(3, 5, 8, requires_grad=True),
                torch.zeros(3, 1, 5, 5, requires_grad=True),
                torch.zeros(3, 1, 5, 1, dtype=torch.bool),
            ),
            run=run,
            seq_len=5,
            batch_size=3,
            dtype=torch.float32,
        )


@pytest.mark.parametrize("chunk_size", [None, 7])
def test_backbone_cost_counts_complete_and_uneven_chunks_with_shared_weights(
    chunk_size: int | None,
) -> None:
    config = tiny_dreams().backbone
    config.chunk_size = chunk_size
    # Priml costs exclude activation-checkpoint recomputation by convention.
    config.checkpoint_fourier = False

    def run(module: nn.Module, values: tuple[Tensor, ...]) -> Tensor:
        assert isinstance(module, DreamBackbone)
        spectrum, condition = values
        return module.forward(spectrum, condition=condition)

    with sdpa_kernel(SDPBackend.MATH):
        measured = assert_cost_matches_torch(
            config,
            build_input=lambda: (
                torch.rand(3, 5, 2, requires_grad=True),
                torch.randn(3, 8, requires_grad=True),
            ),
            run=run,
            seq_len=5,
            batch_size=3,
            dtype=torch.float32,
        )
    assert (
        measured.bytes_state
        == (config.basis.num_fractional + config.basis.num_integer)
        * torch.tensor(0.0).element_size()
    )


def test_dreams_cost_matches_both_heads_and_all_preparation_paths() -> None:
    config = tiny_dreams()
    config.backbone.checkpoint_fourier = False
    config.backbone.chunk_size = 7

    def run(module: nn.Module, values: tuple[Tensor, ...]) -> Tensor:
        assert isinstance(module, DreamsFP)
        logits, elements = module.forward(*values)
        return torch.cat([logits, elements], dim=-1)

    with sdpa_kernel(SDPBackend.MATH):
        assert_cost_matches_torch(
            config,
            build_input=lambda: tuple(
                value.requires_grad_() if value.is_floating_point() else value
                for value in inputs()
            ),
            run=run,
            seq_len=5,
            batch_size=3,
            dtype=torch.float32,
        )


def test_condition_and_prediction_heads_preserve_injected_activations() -> None:
    config = tiny_dreams()
    for head in (config.condition, config.head, config.elements):
        assert isinstance(head.elements, list)
        head.elements[1] = Activation.Config(act=relu)
    model = config.make()
    for head in (model.condition, model.head, model.elements):
        activation = head[1]
        assert isinstance(activation, Activation)
        assert activation.act is relu


def test_spectrum_preparation_keeps_positive_subunit_masses_and_float32_output() -> (
    None
):
    config = tiny_dreams()
    config.max_peaks = 2
    model = config.make()
    mass = torch.tensor(
        [
            [0.0, 0.5, 1.5, 2.5, 3.5],
            [0.0, 0.25, 1.25, 2.25, 3.25],
            [0.0, 0.125, 1.125, 2.125, 3.125],
            [0.0, 0.375, 1.375, 2.375, 3.375],
        ],
        dtype=torch.float64,
    )
    intensity = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.125], dtype=torch.float64).repeat(
        4,
        1,
    )
    precursor = torch.tensor([100.0, 200.0, 300.0, 400.0], dtype=torch.float64)
    mode = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    output = model.prepare_spectrum(
        mass,
        intensity=intensity,
        pad=torch.zeros(4, 5, dtype=torch.bool),
        precursor=precursor,
        mode=mode,
    )
    shift = torch.tensor(
        [0.0, config.negative_mass_shift, 0.0, config.negative_mass_shift],
        dtype=torch.float64,
    )
    expected_mass = torch.cat(
        [(precursor + shift)[:, None].float(), (mass[:, 1:3] + shift[:, None]).float()],
        dim=1,
    )
    expected_intensity = torch.full((4, 3), config.precursor_intensity)
    expected_intensity[:, 1] = 1
    expected_intensity[:, 2] = torch.tensor(0.5).square() / torch.tensor(0.75).square()
    assert output.dtype == torch.float32
    assert torch.equal(output, torch.stack([expected_mass, expected_intensity], dim=-1))


def test_spectrum_preparation_keeps_every_allocation_on_the_input_device() -> None:
    model = tiny_dreams().make()
    output = model.prepare_spectrum(
        torch.ones(3, 7, device="meta"),
        intensity=torch.ones(3, 7, device="meta"),
        pad=torch.zeros(3, 7, dtype=torch.bool, device="meta"),
        precursor=torch.ones(3, device="meta"),
        mode=torch.ones(3, device="meta"),
    )
    assert output.device.type == "meta"
    # DreamsFP.prepare_spectrum prepends one precursor and stacks mass/intensity.
    assert output.shape == (3, model.config.max_peaks + 1, 2)


def test_three_updates_match_every_tensor_and_random_position() -> None:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        _ = torch.manual_seed(42)
        model = tiny_dreams().make()
        actual = {
            f"initial/{name}": value.detach().clone()
            for name, value in model.state_dict().items()
        }
        actual["initial/rng"] = rng_fingerprint()
        config = CompositeOptimizer.Config()
        config.optimizers = [PartialConfig(torch.optim.AdamW, lr=1e-5)]
        optimizer = config.make()(model)
        loss_fn = SimpleLoss.Config().make()
        labels = torch.arange(21).reshape(3, 7).remainder(2).float()
        losses: list[Tensor] = []
        for _ in range(3):
            logits, _ = model.forward(*inputs())
            loss = loss_fn(logits.float(), label=labels)["loss"].mean(-1)
            losses.append(loss.detach().clone())
            loss.mean().backward()
            _ = nn.utils.clip_grad_norm_(model.parameters(), math.inf, foreach=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        actual["loss"] = torch.stack(losses).float()
        for name, value in model.state_dict().items():
            actual[f"final/{name}"] = value.detach().clone()
        actual.update(optimizer_state(model, optimizer.optimizers[0]))
        model.eval()
        logits, elements = model.forward(*inputs())
        actual["logits"] = logits.detach().float()
        actual["elements"] = elements.detach().float()
        with torch.inference_mode():
            logits, elements = model.forward(*inputs())
        actual["inference/logits"] = logits.detach().float()
        actual["inference/elements"] = elements.detach().float()
        actual["final/rng"] = rng_fingerprint()
        model.config.max_peaks = 7
        mz, intensity, pad, precursor, _, _, _, _, mode = inputs()
        actual["padded_spectrum"] = model.prepare_spectrum(
            mz,
            intensity=intensity,
            pad=pad,
            precursor=precursor,
            mode=mode,
        )
    assert_tensor_golden(DREAMS_GOLDEN, actual)


@pytest.mark.parametrize("damage", ["missing", "shape", "none"])
def test_complete_checkpoint_split_and_preload_validation(damage: str) -> None:
    model = tiny_dreams().make()
    original = {name: tensor.clone() for name, tensor in model.released_state().items()}
    expected = {name: value + 1 for name, value in original.items()}
    if damage == "missing":
        del expected["head.0.bias"]
    elif damage == "shape":
        expected["backbone.transformer_encoder.atts.0.weights"] = torch.zeros(2, 3)
    if damage != "none":
        message = (
            "Checkpoint keys differ: missing=['head.0.bias'], unexpected=[]."
            if damage == "missing"
            else "DreamsFP checkpoint shapes differ: ['backbone.transformer_encoder.atts.0.weights']."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model.load_released_state(expected)
        expected = original
    else:
        model.load_released_state(expected)
    assert not mismatches(expected, model.released_state())


def test_the_zeroed_condition_projection_must_have_a_bias() -> None:
    config = tiny_dreams()
    assert isinstance(config.condition.elements, list)
    output = config.condition.elements[-1]
    assert isinstance(output, Linear.Config)
    output.bias = False
    with pytest.raises(ValueError, match="bias"):
        _ = config.make()


def test_full_record_detects_one_packed_projection_ulp() -> None:
    model = tiny_dreams().make()
    expected = {name: tensor.clone() for name, tensor in model.released_state().items()}
    attn = cast(object, model.backbone.encoder.blocks[0].attn)
    assert isinstance(attn, DreamAttention)
    weight = attn.attention.proj_qkv.weight
    with torch.no_grad():
        _ = weight.view(torch.int32).reshape(-1)[0].bitwise_xor_(1)
    assert mismatches(expected, model.released_state()) == [
        "backbone.transformer_encoder.atts.0.weights: 1/256 differ",
    ]
    model.register_buffer("unmapped", torch.zeros(2, 3))
    with pytest.raises(ValueError, match="unmapped"):
        model.load_released_state(model.released_state())


def test_dream_checkpoint_mapping_rejects_collisions_before_loading() -> None:
    model = _DreamCollision(tiny_dreams().finalized())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    proposed = {name: value + 1 for name, value in model.released_state().items()}
    with pytest.raises(
        ValueError,
        match=r"^Checkpoint mapping has duplicate native tensor names\.$",
    ):
        model.load_released_state(proposed)
    assert not mismatches(before, model.state_dict())


class _DreamCollision(DreamsFP):
    @override
    def checkpoint_parts(self) -> dict[str, tuple[str, ...]]:
        parts = super().checkpoint_parts()
        parts["duplicate"] = ("head.0.weight",)
        return parts


SECOND_GOLDEN: Final = _BASELINE / "testdata" / "second_fpnet.pt"


def tiny_second() -> SecondFPNet.Config:
    """Reduce only the geometry of the second engine's released model family."""
    config = SecondFPNet.Config()
    config.channels_hidden = 8
    config.channels_out = 7
    config.num_layers = 1  # The golden holds one complete block.
    config.head.channels_hidden = 9
    config.adduct.channels_in = 11
    assert isinstance(config.block.attn, PeakAttention.Config)
    config.block.attn.attention.num_heads = 2
    return config


def second_inputs() -> tuple[Tensor, ...]:
    """Use the recorded spectra with three different instrument families."""
    mz, intensity, pad, precursor, adduct, ce, _, _, polarity = inputs()
    return mz, intensity, pad, precursor, adduct, torch.tensor([0, 2, 4]), ce, polarity


def test_second_three_updates_match_every_tensor_and_random_position() -> None:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        _ = torch.manual_seed(42)
        model = tiny_second().make()
        actual = {
            f"initial/{name}": value.detach().clone()
            for name, value in model.state_dict().items()
        }
        actual["initial/rng"] = rng_fingerprint()
        config = CompositeOptimizer.Config()
        config.optimizers = [PartialConfig(torch.optim.AdamW, lr=1e-5)]
        optimizer = config.make()(model)
        loss_fn = SimpleLoss.Config().make()
        labels = torch.arange(21).reshape(3, 7).remainder(2).float()
        losses: list[Tensor] = []
        for _ in range(3):
            logits = model.forward(*second_inputs())
            loss = loss_fn(logits.float(), label=labels)["loss"].mean(-1)
            losses.append(loss.detach().clone())
            loss.mean().backward()
            _ = nn.utils.clip_grad_norm_(model.parameters(), math.inf, foreach=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        actual["loss"] = torch.stack(losses).float()
        for name, value in model.state_dict().items():
            actual[f"final/{name}"] = value.detach().clone()
        actual.update(optimizer_state(model, optimizer.optimizers[0]))
        model.eval()
        actual["logits"] = model.forward(*second_inputs()).detach().float()
        with torch.inference_mode():
            actual["inference/logits"] = (
                model.forward(*second_inputs()).detach().float()
            )
        actual["final/rng"] = rng_fingerprint()
    assert_tensor_golden(SECOND_GOLDEN, actual)


def test_checkpoint_remap_is_complete_and_detects_a_single_ulp() -> None:
    expected = {
        name: value + 1 for name, value in tiny_second().make().released_state().items()
    }
    model = tiny_second().make()
    model.load_released_state(expected)
    assert not mismatches(expected, model.released_state())
    with torch.no_grad():
        _ = model.instrument.weight.view(torch.int32).reshape(-1)[0].bitwise_xor_(1)
    assert mismatches(expected, model.released_state()) == ["ins.weight: 1/40 differ"]
    model.register_buffer("unmapped", torch.zeros(2, 3))
    with pytest.raises(ValueError, match="checkpoint schema"):
        _ = model.checkpoint_names()


@pytest.mark.parametrize("oversized", [0, 12, 40])
@pytest.mark.parametrize("counts_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("token_dtype", [None, torch.float16])
def test_formula_cost_includes_frozen_table_and_only_learned_token_adjoint(
    oversized: int,
    counts_dtype: torch.dtype,
    token_dtype: torch.dtype | None,
) -> None:
    config = FormulaEncoding.Config()
    config.max_count = 7
    config.extra.dtype = token_dtype
    counts = torch.arange(40, dtype=counts_dtype).remainder(7)
    counts[:oversized] = 9
    normal_rows = 0 if oversized == 40 else 7
    # FormulaEncoding.finalize forces one oversized-count embedding row.
    execution = assert_cost_matches_torch(
        config,
        build_input=lambda: counts.reshape(2, 5, 4).clone(),
        seq_len=5,
        batch_size=2,
        num_elements=4,
        oversized=oversized,
        normal_rows=normal_rows,
        counts_dtype=counts_dtype,
        dtype=torch.float32,
    )
    assert execution.params == (7 + 1) * 3
    assert execution.params_active == normal_rows * 3 + (3 if oversized else 0)
    assert execution.bytes_state == 0
    learned = token_dtype or torch.float32
    assert execution["flops", "adjoint", "selection", learned] == oversized * 3


@pytest.mark.parametrize(
    ("oversized", "normal_rows", "message"),
    [
        (-1, 7, "oversized"),
        (41, 0, "oversized"),
        (0, 8, "Normal rows"),
        (40, 1, "Normal rows"),
    ],
)
def test_formula_cost_rejects_metadata_that_cannot_describe_its_input(
    oversized: int,
    normal_rows: int,
    message: str,
) -> None:
    config = FormulaEncoding.Config()
    config.max_count = 7
    with pytest.raises(ValueError, match=message):
        _ = config.finalized().cost(
            seq_len=5,
            batch_size=2,
            dtype=torch.float32,
            num_elements=4,
            oversized=oversized,
            normal_rows=normal_rows,
        )


@pytest.mark.parametrize("missing_collision", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_frozen_graph_condition_cost_matches_real_sinusoidal_input_adjoint(
    missing_collision: bool,
    dtype: torch.dtype,
) -> None:
    config = GraphConditions.Config()
    config.adduct_groups = [[0, 1], [2, 3], [1]]
    config.collision_channels = 10
    config.num_instruments = 8
    # GraphConditions.__init__ forces the instrument table to be a square identity.
    execution = assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.tensor(
                [2.0, 3.0, float("nan") if missing_collision else 4.0, 5.0, 7.0, 11.0],
                dtype=dtype,
                requires_grad=True,
            ),
            torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32),
            torch.tensor([0, 2, 4, 0, 2, 4], dtype=torch.int32),
        ),
        seq_len=2,
        batch_size=3,
        dtype=dtype,
        adduct_rows=2,
        instrument_rows=3,
        missing_collision=missing_collision,
        index_dtype=torch.int32,
    )
    assert execution.params == 3 * (3 + 4) + 10 // 2 + 10 + 8 * 8
    assert execution.params_active == (
        2 * (3 + 4) + 3 * 8 + 10 // 2 + (10 if missing_collision else 0)
    )
    assert execution["flops", "adjoint"].sum() > 0
    assert execution.bytes_state == 0


@pytest.mark.parametrize(
    ("adduct_rows", "instrument_rows"),
    [(0, 1), (1, 0), (7, 1), (1, 7)],
)
def test_graph_condition_cost_rejects_impossible_selected_row_counts(
    adduct_rows: int,
    instrument_rows: int,
) -> None:
    with pytest.raises(ValueError, match="Distinct condition rows"):
        _ = (
            GraphConditions.Config()
            .finalized()
            .cost(
                seq_len=2,
                batch_size=3,
                dtype=torch.float32,
                adduct_rows=adduct_rows,
                instrument_rows=instrument_rows,
                missing_collision=False,
            )
        )


def test_attention_rejects_indivisible_head_geometry() -> None:
    config = PackedAttention.Config()
    config.projections.channels_in = 3
    config.projections.num_heads = 2
    with pytest.raises(ValueError, match="channels_in=3 not divisible by num_heads=2"):
        config.make()


@pytest.mark.parametrize("cross", [False, True])
def test_packed_attention_cost_includes_complete_self_or_cross_products(
    cross: bool,
) -> None:
    config = PackedAttention.Config()
    config.projections.channels_in = 8
    config.projections.num_heads = 2
    config.projections.dropout = 0.0

    def run(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
        assert isinstance(module, PackedAttention)
        query, memory = inputs
        # PackedAttention's self-attention dispatch requires identical Q/K/V objects.
        key = memory if cross else query
        output, _ = module.forward(query, key, key, need_weights=False)
        return output

    with sdpa_kernel(SDPBackend.MATH):
        assert_cost_matches_torch(
            config,
            build_input=lambda: (
                torch.randn(3, 5, 8, requires_grad=True),
                torch.randn(3, 7, 8, requires_grad=True),
            ),
            run=run,
            seq_len=5,
            batch_size=3,
            memory_len=7 if cross else -1,
            dtype=torch.float32,
        )


@pytest.mark.parametrize("cross", [False, True])
def test_transformer_layer_cost_counts_independent_attention_and_norms(
    cross: bool,
) -> None:
    config = Decoder.Config() if cross else Encoder.Config()
    config.channels_in = 8
    config.attention.projections.num_heads = 2
    config.attention.projections.dropout = 0.0
    config.dropout.p = 0.0

    def run(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
        query, memory = inputs
        if isinstance(module, Decoder):
            return module.forward(query, memory)
        assert isinstance(module, Encoder)
        return module.forward(query)

    with sdpa_kernel(SDPBackend.MATH):
        assert_cost_matches_torch(
            config,
            build_input=lambda: (
                torch.randn(3, 5, 8, requires_grad=True),
                torch.randn(3, 7, 8, requires_grad=True),
            ),
            run=run,
            seq_len=5,
            batch_size=3,
            dtype=torch.float32,
            **({"memory_len": 7} if cross else {}),
        )


def test_slot_cost_counts_all_independent_decoder_layers_and_query_ownership() -> None:
    config = Slots.Config()
    config.count = 5
    config.num_layers = 2
    config.decoder.channels_in = 8
    config.decoder.attention.projections.num_heads = 2
    config.decoder.attention.projections.dropout = 0.0
    config.decoder.dropout.p = 0.0
    with sdpa_kernel(SDPBackend.MATH):
        assert_cost_matches_torch(
            config,
            build_input=lambda: (
                torch.randn(3, 2, 8, requires_grad=True),
                torch.randn(3, 7, 8, requires_grad=True),
                torch.zeros(3, 9, dtype=torch.bool),
            ),
            seq_len=7,
            root_len=2,
            batch_size=3,
            dtype=torch.float32,
        )


GLACIER_GOLDEN: Final = _BASELINE / "testdata" / "glacier.pt"


def tiny_glacier() -> Glacier.Config:
    """Reduce geometry while retaining every selected neural family."""
    config = Glacier.Config()
    config.channels_hidden = 4
    config.heads = 2
    config.channels_in = 4
    config.root.edge.channels_in = 3
    config.root.num_layers = 1
    config.root.degree.channels_in = 5
    config.root.spatial.channels_in = 7
    config.root.edge_distances = config.root.max_hops = 3
    config.queries.count = 2
    config.queries.num_layers = config.intensity_decoder_layers = (
        config.intensity_encoder_layers
    ) = 1
    config.max_broken = 3
    config.num_elements = 2
    config.formula.max_count = 3
    config.conditions.adduct_groups = [[0, 3], [0, 2], [1, 4]]
    config.conditions.collision_channels = 4
    config.conditions.num_instruments = 2
    return config


def prepared_batch() -> tuple[GlacierBatch, FragmentTargets, Tensor]:
    """Cover multi-hop paths, padded atoms, a full graph and all tiny contexts."""
    counts = torch.tensor([4, 5, 5])
    # Glacier.forward consumes square atom adjacency axes.
    adjacency = torch.zeros(3, 5, 5)
    for graph, count in enumerate([4, 5, 5]):
        for source in range(count):
            for target in range(count):
                if source != target and (graph == 2 or abs(source - target) == 1):
                    adjacency[graph, source, target] = 1
    features = torch.arange(60).reshape(3, 5, 4).float() / 60
    features[0, 4] = 0
    # Glacier.forward adds its graph token to both square attention axes.
    bias = torch.zeros(3, 6, 6)
    bias[0, 5, :] = bias[0, :, 5] = -99_999
    spatial = (
        (torch.arange(5)[:, None] - torch.arange(5)[None, :])
        .abs()
        .expand(3, -1, -1)
        .clone()
    )
    spatial[2] = 1 - torch.eye(5, dtype=torch.long)
    spatial[0, 4, :] = spatial[0, :, 4] = 0
    # Glacier.forward consumes paired source/target atom axes in path features.
    edges = torch.zeros(3, 5, 5, 3, 3)
    direct = torch.zeros(3, 5, 5, 3)
    for graph, count in enumerate([4, 5, 5]):
        for source in range(count):
            for target in range(count):
                if source == target:
                    continue
                path = (
                    [source, target]
                    if graph == 2
                    else list(
                        range(
                            source,
                            target + (1 if target > source else -1),
                            1 if target > source else -1,
                        ),
                    )
                )
                for hop, (start, end) in enumerate(itertools.pairwise(path)):
                    if hop < 3:
                        edges[graph, source, target, hop, (start + end) % 3] = 1
                if adjacency[graph, source, target]:
                    direct[graph, source, target, (source + target) % 3] = 1
    hs = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 0, 1], [1, 0, 1, 2, 0]])
    atom_forms = torch.stack([torch.ones_like(hs), hs], dim=-1)
    masks: list[Tensor] = []
    for count in [4, 5, 5]:
        masks.extend(torch.arange(5) < keep for keep in (count, count - 1))
    batch = GlacierBatch(
        graph=GraphInput(
            features=features,
            bias=bias,
            edges=direct,
            spatial=spatial,
            degrees=adjacency.sum(-1).long(),
            paths=edges,
        ),
        atom_counts=counts,
        adduct=torch.tensor([0.0, 2.0, 1.0]),
        collision_energy=torch.tensor([10.0, float("nan"), 60.0]),
        instrument=torch.tensor([0.0, 1.0, 0.0]),
        root_forms=torch.tensor([[4, 3], [5, 4], [5, 4]]),
        atom_forms=torch.cat(
            [atom_forms[index, :count] for index, count in enumerate([4, 5, 5])],
        ),
        masses=torch.tensor(
            [
                [12.0, 13.007825032, 14.015650064, 12.0, 0.0],
                [12.0, 13.007825032, 14.015650064, 12.0, 13.007825032],
                [13.007825032, 12.0, 13.007825032, 14.015650064, 12.0],
            ],
        ),
        adduct_shifts=torch.tensor(
            [
                [1.007276452, -0.00054858],
                [-1.007276452, 0.00054858],
                [22.98922070, -0.00054858],
            ],
        ),
        adjacency=adjacency,
        atom_hs=hs,
        total_hs=hs.sum(-1),
    )
    targets = FragmentTargets(masks=torch.stack(masks), counts=torch.tensor([2, 2, 2]))
    spectrum = torch.tensor(
        [
            [[10.0, 1.0], [50.0, 0.5], [100.0, 0.25]],
            [[10.0, 0.2], [50.0, 0.7], [100.0, 0.4]],
            [[10.0, 0.3], [50.0, 0.2], [100.0, 0.9]],
        ],
    )
    return batch, targets, spectrum


def capture(record: dict[str, Tensor], prefix: str, output: object) -> None:
    """Flatten every native nested output, retaining explicit absent fields."""
    if isinstance(output, Tensor):
        record[prefix] = output.detach().clone()
    elif output is None:
        record[f"{prefix}/is_none"] = torch.tensor(True)
    else:
        for name, value in from_plain(output, dict[str, object]).items():
            capture(record, f"{prefix}/{name}", value)


def assert_glacier_replay() -> None:
    """Train GLACIER for three steps and compare every tensor with its golden."""
    golden = GLACIER_GOLDEN
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        _ = torch.manual_seed(42)
        model = tiny_glacier().make()
        batch, labels, spectrum = prepared_batch()
        actual = {
            f"initial/{name}": value.detach().clone()
            for name, value in model.state_dict().items()
        }
        actual["initial/rng"] = rng_fingerprint()
        actual["requires_grad"] = torch.tensor(
            [parameter.requires_grad for parameter in model.parameters()],
        )
        optimizer = CompositeOptimizer(
            [
                PartialConfig(torch.optim.AdamW, lr=0.0004, weight_decay=0.0).make()(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                ),
            ],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            partial(generation_multiplier, warmup=2000, decay=0.9),
        )
        criterion_config = GlacierCriterion.Config()
        criterion_config.num_bins = 31
        criterion = criterion_config.make()
        losses: list[Tensor] = []
        for _ in range(3):
            prediction = labelled_forward(model, batch, labels)
            loss = criterion(prediction, batch=batch, labels=labels, spectrum=spectrum)
            losses.append(loss.detach().clone())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        actual["loss"] = torch.stack(losses)
        for name, value in model.state_dict().items():
            actual[f"final/{name}"] = value.detach().clone()
        actual.update(optimizer_state(model, optimizer.optimizers[0]))
        actual["learning_rate"] = torch.tensor(
            [
                from_plain(group["lr"], float)
                for group in from_plain(optimizer.param_groups, list[dict[str, object]])
            ],
        )
        actual["step"] = torch.tensor(criterion.step)
        model.eval()
        capture(actual, "output", labelled_forward(model, batch, labels))
        with warnings.catch_warnings(record=True) as emitted, torch.inference_mode():
            # Pytest raises warnings before record=True can capture them. Admit
            # only this measured standard-torch prototype notice into the log.
            warnings.filterwarnings(
                "always",
                message="The PyTorch API of nested tensors is in prototype stage.*",
                category=UserWarning,
            )
            capture(
                actual,
                "inference",
                {**model.forward(batch), "inten_pred_magma": None},
            )
        assert all(
            "nested tensors is in prototype stage" in str(warning.message)
            for warning in emitted
        )
        actual["post/graph_features"] = batch.graph.features.clone()
        actual["post/root_forms"] = batch.root_forms.clone()
        actual["post/atom_forms"] = batch.atom_forms.clone()
        actual["final/rng"] = rng_fingerprint()
    assert_tensor_golden(golden, actual)


def test_whole_glacier_three_released_training_steps() -> None:
    assert_glacier_replay()


def test_complete_checkpoint_mapping_and_atomic_shape_rejection() -> None:
    with torch.random.fork_rng(devices=[]):
        source = {
            name: value + 1
            for name, value in tiny_glacier().make().released_state().items()
        }
        model = tiny_glacier().make()
    model.load_released_state(source)
    assert not mismatches(source, model.released_state())
    invalid = dict(source)
    invalid["formula_mapper.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape"):
        model.load_released_state(invalid)
    assert not mismatches(source, model.released_state())
    with torch.no_grad():
        parameter = model.formula_mapper.weight
        parameter[0, 0] = parameter[0, 0].nextafter(parameter.new_tensor(float("inf")))
    assert mismatches(source, model.released_state()) == [
        "formula_mapper.weight: 1/32 differ",
    ]
    model.register_parameter("unexpected", nn.Parameter(torch.zeros(2)))
    with pytest.raises(ValueError, match="differs from its checkpoint schema"):
        model.checkpoint_names()


@pytest.mark.parametrize("bias", [False, True])
def test_fragment_attention_reset_repeats_complete_constructor(bias: bool) -> None:
    config = FragmentAttention.Config()
    config.attention.channels_in = 4
    config.attention.num_heads = 2
    assert isinstance(config.attention.proj_qkv, Linear.Config)
    assert isinstance(config.attention.proj_out, Linear.Config)
    config.attention.proj_qkv.bias = config.attention.proj_out.bias = bias
    with torch.random.fork_rng(devices=[]):
        _ = torch.manual_seed(42)
        attention = config.make()
        initial = {
            name: value.clone() for name, value in attention.state_dict().items()
        }
        _ = torch.manual_seed(42)
        attention.reset_parameters()
        assert not mismatches(initial, attention.state_dict())


def test_fragment_kernel_cost_is_the_shared_fused_kernel_cost() -> None:
    geometry = {
        "seq_len": 5,
        "batch_size": 3,
        "num_heads": 2,
        "channels_head": 4,
        "channels_v_head": 4,
        "dropout_p": 0.0,
        "dtype": torch.float32,
    }
    assert FragmentKernel.Config.cost(**geometry) == cost(
        SdpaFused.Config(),
        **geometry,
    )


def test_fragment_attention_rejects_missing_padding_at_its_message_boundary() -> None:
    config = FragmentAttention.Config()
    config.attention.channels_in = 4
    config.attention.num_heads = 2
    with pytest.raises(TypeError, match="requires a padding tensor"):
        config.make()(torch.ones(2, 3, 4))


GEN_GOLDEN: Final = _BASELINE / "testdata" / "gen.pt"


def tiny_generator() -> GraphGenerator.Config:
    """Reduce only graph, vocabulary, formula-count and recurrent geometry."""
    config = GraphGenerator.Config()
    config.channels_in = 8
    config.channels_hidden = 5
    config.num_elements = 2
    config.max_broken = 3
    config.formula.max_count = 7
    config.conditions.adduct_groups = [[0, 3], [0, 2], [1, 4]]
    config.conditions.collision_channels = 4
    config.conditions.num_instruments = 2
    config.encoder.messages.num_layers = 2
    return config


def molecular_graph(counts: list[int], *, offset: int) -> MolecularGraph:
    """Construct deterministic directed typed chains for the source input geometry."""
    source: list[int] = []
    destination: list[int] = []
    start = 0
    for count in counts:
        for index in range(start, start + count - 1):
            source.extend((index, index + 1))
            destination.extend((index + 1, index))
        start += count
    return MolecularGraph(
        topology=GraphBatch(
            source=torch.tensor(source),
            destination=torch.tensor(destination),
            edge_type=torch.arange(len(source)).remainder(4),
            node_counts=torch.tensor(counts),
        ),
        features=(
            torch.arange(start * 8, dtype=torch.float32).reshape(start, 8) + offset
        )
        / (start * 8 + offset),
    )


def generation_batch() -> GenerationBatch:
    """Cover nonfinite CE, positive/negative adducts and oversized formula counts."""
    return GenerationBatch(
        roots=molecular_graph([4, 5], offset=0),
        fragments=molecular_graph([2, 3, 4], offset=1),
        root_index=torch.tensor([0, 1, 1]),
        broken=torch.tensor([0, 1, 5]),
        collision_energy=torch.tensor([10.0, float("nan")]),
        precursor=torch.tensor([100.0, 200.0]),
        adduct=torch.tensor([0, 2]),
        instrument=torch.tensor([0, 1]),
        root_forms=torch.tensor([[9, 2], [6, 8]]),
        fragment_forms=torch.tensor([[7, 0], [3, 1], [0, 8]]),
    )


def test_three_adam_updates_match_all_generator_values() -> None:
    golden = GEN_GOLDEN
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        _ = torch.manual_seed(42)
        model = tiny_generator().make()
        batch = generation_batch()
        actual = {
            f"initial/{name}": value.detach().clone()
            for name, value in model.state_dict().items()
        }
        actual["initial/rng"] = rng_fingerprint()
        optimizer = CompositeOptimizer(
            [
                PartialConfig(torch.optim.Adam, lr=0.000996, weight_decay=0.0).make()(
                    model.parameters(),
                ),
            ],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            partial(generation_multiplier, warmup=1000, decay=0.7214),
        )
        loss_config = SimpleLoss.Config()
        loss_config.loss_fn = functional.binary_cross_entropy
        loss_fn = loss_config.make()
        counts = batch.fragments.topology.node_counts
        labels = torch.arange(12).reshape(3, 4).remainder(2).float()
        losses: list[Tensor] = []
        for _ in range(3):
            probabilities = model.forward(batch)
            per_atom = loss_fn(probabilities, label=labels)["loss"]
            valid = torch.arange(per_atom.shape[1])[None, :] < counts[:, None]
            loss = (per_atom * valid).sum() / counts.sum()
            losses.append(loss.detach().clone())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        actual["loss"] = torch.stack(losses).float()
        for name, value in model.state_dict().items():
            actual[f"final/{name}"] = value.detach().clone()
        actual.update(optimizer_state(model, optimizer.optimizers[0]))
        actual["learning_rate"] = torch.tensor(
            [
                from_plain(group["lr"], float)
                for group in from_plain(optimizer.param_groups, list[dict[str, object]])
            ],
        )
        model.eval()
        actual["output"] = model.forward(batch).detach().float()
        with torch.inference_mode():
            actual["inference"] = model.forward(batch).detach().float()
        actual["post/root_forms"] = batch.root_forms.clone()
        actual["post/fragment_forms"] = batch.fragment_forms.clone()
        actual["post/root_features"] = batch.roots.features.clone()
        actual["post/fragment_features"] = batch.fragments.features.clone()
        actual["final/rng"] = rng_fingerprint()
    assert_tensor_golden(golden, actual)


def test_complete_checkpoint_loading_rejects_alias_conflicts_atomically() -> None:
    state = {
        name: value + 1
        for name, value in tiny_generator().make().released_state().items()
    }
    model = tiny_generator().make()
    model.load_released_state(state)
    assert not mismatches(state, model.released_state())
    assert model.encoder is model.root_encoder
    assert any(not parameter.requires_grad for parameter in model.parameters())
    conflict = {
        **state,
        "root_module.input_project.weight": state["gnn.input_project.weight"] + 1,
    }
    with pytest.raises(ValueError, match="Shared root checkpoint tensors differ"):
        model.load_released_state(conflict)
    with pytest.raises(ValueError, match="Checkpoint keys differ"):
        model.load_released_state(
            {key: value for key, value in state.items() if key != "broken_onehot"},
        )
    with pytest.raises(ValueError, match="Checkpoint tensor shape or dtype differs"):
        model.load_released_state({**state, "broken_onehot": torch.zeros(3)})
    assert not mismatches(state, model.released_state())
    model.register_buffer("undeclared", torch.zeros(1))
    with pytest.raises(ValueError, match="checkpoint schema"):
        _ = model.checkpoint_names()


def test_molecule_mlp_cost_excludes_constructor_only_discarded_affine() -> None:
    config = MoleculeMLP.Config()
    config.channels_in = 5
    config.channels_hidden = 7
    config.channels_out = 11
    execution = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 5, requires_grad=True),
        seq_len=3,
        batch_size=2,
        dtype=torch.float32,
    )
    assert execution.params == (5 * 7 + 7) + (7 * 11 + 11)


@pytest.mark.parametrize("edge_counts", [(2, 3, 0, 0), (0, 0, 0, 0)])
def test_molecule_encoder_cost_composes_projection_and_shared_messages(
    edge_counts: tuple[int, ...],
) -> None:
    config = MoleculeEncoder.Config()
    config.channels_in = 7
    config.channels_out = 5
    config.messages.num_layers = 2
    # MolecularGGNN.finalize forces its message/state widths to channels_out.
    source = torch.arange(sum(edge_counts)).remainder(6)
    graph = GraphBatch(
        source=source,
        destination=(source + 1).remainder(6),
        edge_type=torch.repeat_interleave(torch.arange(4), torch.tensor(edge_counts)),
        node_counts=torch.tensor([2, 4]),
    )
    execution = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(6, 7, requires_grad=True),
        run=lambda module, features: cast(Tensor, module(graph, features)),
        seq_len=2,
        batch_size=3,
        dtype=torch.float32,
        edge_counts=edge_counts,
    )
    assert execution.params == (7 * 5 + 5 + 4 * (5 * 5 + 5) + 3 * 5 * (5 + 5) + 6 * 5)


INTENSITY_GOLDEN: Final = _BASELINE / "testdata" / "intensity.pt"


def tiny_intensity() -> GraphIntensity.Config:
    """Reduce only graph, vocabulary, formula, attention and bin geometry."""
    config = GraphIntensity.Config()
    config.channels_in = 8
    config.channels_hidden = 4
    config.num_elements = 2
    config.max_broken = 2
    config.formula.max_count = 3
    config.conditions.adduct_groups = [[0, 3], [0, 2], [1, 4]]
    config.conditions.collision_channels = 4
    config.conditions.num_instruments = 2
    config.encoder.messages.num_layers = 2
    config.num_layers = 2
    set_heads(config.block.attn, 2)
    config.num_buckets = 23
    return config


def intensity_batch() -> IntensityBatch:
    """Cover shared roots, padded fragments, NaN CE and colliding shifted masses."""
    # GraphIntensity.forward fixes two mass groups before hydrogen-shift bins.
    masses = torch.arange(104).remainder(29).reshape(2, 2, 2, 13).float() * 67
    masses[0, 1] = 0
    return IntensityBatch(
        roots=molecular_graph([4, 5], offset=0),
        fragments=molecular_graph([2, 3, 4], offset=1),
        root_index=torch.tensor([0, 1, 1]),
        fragment_counts=torch.tensor([1, 2]),
        broken=torch.tensor([[0, 9], [1, 5]]),
        collision_energy=torch.tensor([10.0, float("nan")]),
        precursor=torch.tensor([100.0, 200.0]),
        adduct=torch.tensor([0, 2]),
        instrument=torch.tensor([0, 1]),
        max_add_h=torch.tensor([[2, 0], [1, 6]]),
        max_remove_h=torch.tensor([[1, 0], [2, 5]]),
        masses=masses,
        root_forms=torch.tensor([[9, 2], [6, 8]]),
        fragment_forms=torch.tensor([[[3, 0], [0, 0]], [[2, 1], [0, 8]]]),
    )


def test_complete_intensity_three_updates() -> None:
    golden = INTENSITY_GOLDEN
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        _ = torch.manual_seed(42)
        model = tiny_intensity().make()
        batch = intensity_batch()
        actual = {
            f"initial/{name}": value.detach().clone()
            for name, value in model.state_dict().items()
        }
        actual["initial/rng"] = rng_fingerprint()
        actual["requires_grad"] = torch.tensor(
            [parameter.requires_grad for parameter in model.parameters()],
        )
        optimizer = CompositeOptimizer(
            [
                PartialConfig(torch.optim.Adam, lr=0.000736, weight_decay=1e-7).make()(
                    model.parameters(),
                ),
            ],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            partial(generation_multiplier, warmup=1000, decay=0.825),
        )
        cosine = torch.nn.CosineSimilarity()
        labels = torch.arange(46).reshape(2, 23).remainder(7).float()
        losses: list[Tensor] = []
        for _ in range(3):
            prediction = model.forward(batch)["output_binned"][:, 0, :]
            loss = (1 - cosine(prediction, labels)).mean()
            losses.append(loss.detach().clone())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        actual["loss"] = torch.stack(losses).float()
        for name, value in model.state_dict().items():
            actual[f"final/{name}"] = value.detach().clone()
        actual.update(optimizer_state(model, optimizer.optimizers[0]))
        actual["learning_rate"] = torch.tensor(
            [
                from_plain(group["lr"], float)
                for group in from_plain(optimizer.param_groups, list[dict[str, object]])
            ],
        )
        model.eval()
        for name, value in model.forward(batch).items():
            actual[f"output/{name}"] = value.detach().float()
        with torch.inference_mode():
            for name, value in model.forward(batch).items():
                actual[f"inference/{name}"] = value.detach().float()
        actual["post/root_forms"] = batch.root_forms.clone()
        actual["post/fragment_forms"] = batch.fragment_forms.clone()
        actual["post/root_features"] = batch.roots.features.clone()
        actual["post/fragment_features"] = batch.fragments.features.clone()
        actual["final/rng"] = rng_fingerprint()
    assert_tensor_golden(golden, float32_comparands(actual))


def test_checkpoint_schema_rejects_extra_registration() -> None:
    model = tiny_intensity().make()
    model.register_parameter("unexpected", torch.nn.Parameter(torch.zeros(1)))
    with pytest.raises(
        ValueError,
        match=r"^Native intensity state differs from its checkpoint schema\.$",
    ):
        model.checkpoint_names()


def test_intensity_checkpoint_loading_and_clone_identity() -> None:
    state = {
        name: value + 1
        for name, value in tiny_intensity().make().released_state().items()
    }
    model = tiny_intensity().make()
    model.load_released_state(state)
    assert not mismatches(state, model.released_state())
    assert model.encoder is model.root_encoder
    for first, second in zip(
        model.blocks[0].parameters(),
        model.blocks[1].parameters(),
        strict=True,
    ):
        assert first is not second
        assert torch.equal(first, second)
    assert model.output.weight is not model.isomer.weight
    assert torch.equal(model.output.weight, model.isomer.weight)
    conflict = {
        **state,
        "root_module.input_project.weight": state["gnn.input_project.weight"] + 1,
    }
    with pytest.raises(ValueError, match="Shared root checkpoint tensors differ"):
        model.load_released_state(conflict)
    with pytest.raises(ValueError, match="Checkpoint keys differ"):
        model.load_released_state(
            {key: value for key, value in state.items() if key != "inten_buckets"},
        )
    with pytest.raises(ValueError, match="Checkpoint tensor shape or dtype differs"):
        model.load_released_state({**state, "inten_buckets": torch.zeros(3)})
    assert not mismatches(state, model.released_state())
    model.register_buffer("undeclared", torch.zeros(1))
    with pytest.raises(ValueError, match="checkpoint schema"):
        _ = model.checkpoint_names()


CONSTRAINTS_GOLDEN: Final = _BASELINE / "testdata" / "constraints.pt"


def test_all_constraint_gradients_selections_and_graph_patterns() -> None:
    golden = CONSTRAINTS_GOLDEN
    model = Breakpoints.Config().make()
    actual: dict[str, Tensor] = {}
    with host_agnostic_numerics():
        for index, (logits, counts, cards) in enumerate(
            [
                (
                    (torch.arange(60).reshape(3, 4, 5).remainder(13).float() - 6) / 3,
                    torch.tensor([5, 3, 1]),
                    torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 0, 0]]),
                ),
                (
                    # Breakpoints.rank replays two molecules and two tied slots.
                    torch.zeros(2, 2, 3),
                    torch.tensor([3, 2]),
                    # Breakpoints.select matches the same molecule-slot axes.
                    torch.zeros(2, 2, dtype=torch.long),
                ),
                (
                    torch.tensor(
                        [[[1e9, 1e9, 0, -1e9, -1e9], [-3000, -2000, -1000, 0, 1000]]],
                    ),
                    torch.tensor([5]),
                    torch.tensor([[2, 3]]),
                ),
            ],
        ):
            logits.requires_grad_()
            ranked = model.rank(logits, counts)
            actual[f"{index}/rank"] = ranked.detach().clone()
            gradient = torch.autograd.grad(
                (ranked * torch.arange(ranked.numel()).reshape_as(ranked)).sum(),
                logits,
            )[0]
            actual[f"{index}/gradient"] = gradient.detach().clone()
            chosen = model.select(
                logits,
                torch.nn.functional.one_hot(cards, num_classes=4).float(),
                counts,
            )
            actual[f"{index}/selected"] = chosen.clone()
            if index == 0:
                # Breakpoints.patterns requires square atom adjacency axes.
                adjacency = torch.zeros(3, 5, 5)
                adjacency[0, torch.arange(4), torch.arange(1, 5)] = 1
                adjacency[0, torch.arange(1, 5), torch.arange(4)] = 1
                adjacency[1, 0, 1] = adjacency[1, 1, 0] = 1
                patterns, pattern_counts = model.patterns(chosen, adjacency, counts)
                actual["patterns/masks"] = patterns.clone()
                actual["patterns/counts"] = pattern_counts.clone()
                actual["reachability/all"] = model.closure(adjacency)
                actual["reachability/limited"] = model.closure(adjacency, counts=counts)
        for size in (64, 65):
            adjacency = torch.zeros(2, size, size, dtype=torch.bool)
            adjacency[:, 0, 63] = adjacency[:, 63, 0] = True
            adjacency[:, 1, 62] = adjacency[:, 62, 1] = True
            if size == 65:
                adjacency[:, 63, 64] = adjacency[:, 64, 63] = True
            masks = torch.zeros(2, 3, size, dtype=torch.bool)
            masks[:, 0] = True
            masks[:, 1, 0] = True
            masks[:, 2, 1] = True
            if size == 65:
                masks[1, 2, 0] = masks[1, 2, 63] = True
            for name, value in boundaries(adjacency, masks).items():
                actual[f"boundary/{size}/{name}"] = value.clone()
    assert_tensor_golden(golden, actual)
    assert not actual["boundary/64/boundary_mask"][..., 63].any()


def test_typed_unique_return_contracts() -> None:
    values = torch.tensor([3, 1, 3, 1, 2])
    unique = torch.unique(values)
    assert torch.equal(unique, torch.tensor([1, 2, 3]))
    result, inverse = torch.unique(values, return_inverse=True)
    assert torch.equal(result[inverse], values)
    result, counts = torch.unique(values, return_counts=True)
    assert torch.equal(counts, torch.tensor([2, 1, 2]))
    result_both, inverse_both, counts_both = torch.unique(
        values,
        return_inverse=True,
        return_counts=True,
    )
    assert torch.equal(result_both, result)
    assert torch.equal(result_both[inverse_both], values)
    assert torch.equal(counts_both, counts)


def test_dense_matches_complete_torch_initialization_and_random_position() -> None:
    with torch.random.fork_rng(devices=[]):
        _ = torch.manual_seed(42)
        expected = torch.nn.Linear(3, 5)
        expected_next = torch.rand(7)
        _ = torch.manual_seed(42)
        actual = Linear.Config(3, 5, bias=True, init_bias=None).make()
        assert torch.equal(actual.weight, expected.weight)
        assert actual.bias is not None
        assert expected.bias is not None
        assert torch.equal(actual.bias, expected.bias)
        assert torch.equal(torch.rand(7), expected_next)


@pytest.mark.parametrize(
    "config",
    [Dropout.Config(p=0.1), Activation.Config(act=gelu), Activation.Config(act=relu)],
)
def test_parameterless_leaves_accept_bus_and_reset(
    config: Dropout.Config | Activation.Config,
) -> None:
    config.channels_in = 5
    module = config.make()
    module.reset_parameters()
    assert module(torch.zeros(2, 3, 5), unused="bus").shape == (2, 3, 5)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 5, requires_grad=True),
        seq_len=3,
        batch_size=2,
        channels_in=5,
        dtype=torch.float32,
    )


def test_feed_forward_is_the_released_affine_gelu_affine_pair_with_two_dropouts() -> (
    None
):
    config = feed_forward()
    config.channels_in = 6
    model = config.make()
    model.eval()
    elements = config.elements
    assert isinstance(elements, list)
    swiglu, residual = elements
    assert isinstance(swiglu, SwiGLU.Config)
    assert swiglu.act is gelu
    assert not swiglu.gate
    assert isinstance(swiglu.middle, Dropout.Config)
    assert swiglu.middle.p == 0.1
    assert isinstance(residual, Dropout.Config)
    assert residual.p == 0.1
    state = model.state_dict()
    input = torch.randn(2, 3, 6)
    hidden = torch.nn.functional.gelu(
        torch.nn.functional.linear(
            input,
            state["0.up_proj.weight"],
            state["0.up_proj.bias"],
        ),
    )
    expected = torch.nn.functional.linear(
        hidden,
        state["0.down_proj.weight"],
        state["0.down_proj.bias"],
    )
    assert torch.allclose(model(input), expected, atol=1e-6)


def test_residual_width_reaches_the_kernel_and_is_read_back_from_it() -> None:
    pushed = PeakAttention.Config()
    pushed.channels_out = 8
    pushed = pushed.copy_tree().finalize()
    assert pushed.channels_in == pushed.attention.channels_in == pushed.drop.channels_in
    assert pushed.attention.channels_out == 8
    read = PeakAttention.Config()
    read.attention.channels_in = 6
    read = read.copy_tree().finalize()
    assert read.channels_in == read.channels_out == read.drop.channels_in == 6


def test_attention_cost_and_reset_cover_the_complete_branch() -> None:
    config = PeakAttention.Config()
    config.attention.channels_in = 6
    config.attention.num_heads = 2
    attention = config.make()
    expected = attention.attention.proj_qkv.weight.clone()
    attention.reset_parameters()
    assert not torch.equal(attention.attention.proj_qkv.weight, expected)
    with sdpa_kernel(SDPBackend.MATH):
        assert_cost_matches_torch(
            config,
            build_input=lambda: torch.randn(4, 5, 6, requires_grad=True),
            seq_len=5,
            batch_size=4,
            dtype=torch.float32,
        )


def test_attention_uses_the_injected_kernel_before_the_real_output_affine() -> None:
    config = PeakAttention.Config()
    config.attention.channels_in = 8
    config.attention.num_heads = 2
    config.attention.attn_kernel = InlineConfig(_ZeroAttention)
    attention = config.make().eval()
    values = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) / 17
    assert attention.attention.proj_out.bias is not None
    assert torch.equal(
        attention(values),
        attention.attention.proj_out.bias.expand_as(values),
    )


@pytest.mark.parametrize("heads", [0, 5])
def test_invalid_head_geometry_fails_before_projection_allocation(heads: int) -> None:
    config = PeakAttention.Config()
    config.attention.channels_in = 6
    config.attention.num_heads = heads
    with pytest.raises(ValueError, match="num_heads"):
        _ = config.make()


class _ZeroAttention(nn.Module):
    @override
    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs: object) -> Tensor:
        del k, v, kwargs
        return torch.zeros_like(q)


GGNN_GOLDEN: Final = _BASELINE / "testdata" / "ggnn.pt"


def ggnn_config() -> GGNN.Config:
    """Keep all four edge categories and the shared GRU while reducing geometry."""
    config = released_ggnn()
    config.channels_hidden = 5
    config.num_layers = 2
    return config


def ggnn_topology() -> GraphBatch:
    """Represent three disconnected molecules with ordered directed edges."""
    return GraphBatch(
        source=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 5]),
        destination=torch.tensor([1, 0, 3, 4, 2, 6, 7, 8, 5, 7]),
        edge_type=torch.arange(10).remainder(4),
        node_counts=torch.tensor([2, 3, 4]),
    )


def test_ggnn_three_updates_match_every_active_tensor_and_random_position() -> None:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        _ = torch.manual_seed(42)
        model = ggnn_config().make()
        actual = {
            f"initial/{name}": tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        actual["initial/rng"] = rng_fingerprint()
        config = CompositeOptimizer.Config()
        config.optimizers = [PartialConfig(torch.optim.AdamW, lr=1e-5)]
        optimizer = config.make()(model)
        loss_fn = SimpleLoss.Config().make()
        features = torch.arange(45, dtype=torch.float32).reshape(9, 5) / 45
        labels = torch.arange(45).reshape(9, 5).remainder(2).float()
        losses: list[Tensor] = []
        for _ in range(3):
            logits = model.forward(ggnn_topology(), features)
            loss = loss_fn(logits.float(), label=labels)["loss"].mean(-1)
            losses.append(loss.detach().clone())
            loss.mean().backward()
            _ = nn.utils.clip_grad_norm_(model.parameters(), math.inf, foreach=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        actual["loss"] = torch.stack(losses).float()
        for name, tensor in model.state_dict().items():
            actual[f"final/{name}"] = tensor.detach().clone()
        actual.update(optimizer_state(model, optimizer.optimizers[0]))
        model.eval()
        output = model.forward(ggnn_topology(), features)
        actual["output"] = output.detach().float()
        actual["missing_type"] = (
            model.forward(
                replace(
                    ggnn_topology(),
                    edge_type=ggnn_topology().edge_type.remainder(3),
                ),
                features,
            )
            .detach()
            .float()
        )
        actual["zero_pad"] = (
            model.forward(ggnn_topology(), features[:, :3]).detach().float()
        )
        empty = torch.zeros(0, dtype=torch.long)
        actual["empty_edges"] = (
            model.forward(
                replace(
                    ggnn_topology(),
                    source=empty,
                    destination=empty,
                    edge_type=empty,
                ),
                features,
            )
            .detach()
            .float()
        )
        actual["pool"] = ggnn_topology().average(output).detach().float()
        actual["pool/single"] = (
            replace(ggnn_topology(), node_counts=torch.tensor([1, 3, 5]))
            .average(output)
            .detach()
            .float()
        )
        actual["pool/empty"] = (
            replace(ggnn_topology(), node_counts=torch.tensor([0, 3, 6]))
            .average(output)
            .detach()
            .float()
        )
        actual["final/rng"] = rng_fingerprint()
    assert_tensor_golden(GGNN_GOLDEN, actual)


def test_packed_rows_scatter_into_zero_padded_segments_in_encounter_order() -> None:
    features = torch.arange(1, 7, dtype=torch.float32).reshape(3, 2)
    padded = pad_atoms(features, torch.tensor([2, 1]))
    assert padded.shape == (2, 2, 2)
    assert padded.tolist() == [[[1, 2], [3, 4]], [[5, 6], [0, 0]]]


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, 0.0),
        (999, 1 - math.exp(-999 / 1000)),
        (1000, 1.0),
        (5999, 1.0),
        (6000, 0.7214),
        (10_999, 0.7214),
        (11_000, 0.7214**2),
    ],
)
def test_warmup_and_decay_boundaries(step: int, expected: float) -> None:
    assert generation_multiplier(step) == expected


def test_empty_spectrum_and_zero_boundary_have_analytic_scheduled_loss() -> None:
    batch, _, spectrum = prepared_batch()
    labels = FragmentTargets(
        masks=torch.arange(5)[None, :] < batch.atom_counts[:, None],
        counts=torch.ones(3, dtype=torch.long),
    )
    empty = torch.zeros_like(spectrum)
    prediction = LabelledOutput(
        breakpoints_pred=BreakpointOutput(
            # GlacierCriterion consumes a leading decoder-layer axis, here one.
            frag_logits=torch.zeros(1, 3, 2, 5),
            frag_card_logits=torch.zeros(1, 3, 2, 4),
        ),
        inten_pred_end_to_end=IntensityOutput(output=empty),
        inten_pred_magma=IntensityOutput(output=empty),
        frags_pred=FragmentOutput(fragments=labels.masks, fragment_count=labels.counts),
    )
    criterion = GlacierCriterion.Config().make()
    criterion.step = 4999
    loss = criterion(prediction, batch=batch, labels=labels, spectrum=empty)
    expected = (
        0.9 * torch.tensor(1.0)
        + 0.1 * torch.tensor(1.0)
        + 0.1 * 0.9 * torch.log(torch.tensor(4.0))
    )
    assert torch.equal(loss, expected)
    assert criterion.step == 5000
    prediction["inten_pred_magma"] = None
    with pytest.raises(ValueError, match="requires labelled-fragment"):
        criterion(prediction, batch=batch, labels=labels, spectrum=empty)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
