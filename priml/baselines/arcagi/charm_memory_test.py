"""Tests for the CoSE task-prefix module."""

from __future__ import annotations

from pathlib import Path
from typing import Final, override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import pytest
import torch

from priml.cost import Cost
from priml.model.conditioning import (
    CategorySlotFiLM,
    SigmoidResidualGate,
)
from priml.model.grid_embedding import GridEmbedding
from priml.model.grid_solver import GridSolver
from priml.model.prefix import ComposedSparsePrefix
from priml.model.special import Identity
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import read_tensors, rng_fingerprint


_CWD: Final = Path(__file__).resolve().parent

_TESTDATA: Final = _CWD / "testdata"


class InjectedComposer(nn.Module):
    """Independent injectable task composer used by the slot-contract test."""

    class Config(Fig["InjectedComposer"]):
        """Output width for the independent composer fake."""

        channels_out: int = 21
        """Output width produced by the injected composer."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Return a zero fake cost for injection-only coverage."""
            del seq_len, batch_size, dtype, kwargs
            return Cost()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.channels_out = config.channels_out

    @override
    def forward(
        self,
        category_indices: Tensor,
        augmentation_indices: Tensor,
        slot_permutation: Tensor,
    ) -> Tensor:
        """Broadcast puzzle IDs across the configured output width."""
        del augmentation_indices, slot_permutation
        return (
            category_indices.to(torch.float32)
            .unsqueeze(-1)
            .expand(-1, self.channels_out)
        )

    def reset_parameters(self) -> None:
        """Leave the fake unchanged."""


class InjectedGate(nn.Module):
    """Independent injectable residual gate used by the slot-contract test."""

    class Config(Fig["InjectedGate"]):
        """Output width for the independent gate fake."""

        channels_out: int = 21
        """Output width consumed by the injected residual gate."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Return a zero fake cost for injection-only coverage."""
            del seq_len, batch_size, dtype, kwargs
            return Cost()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.channels_out = config.channels_out

    @override
    def forward(self, task: Tensor) -> Tensor:
        """Return a fractional gate for every task value."""
        return torch.full_like(task, 0.5)

    def reset_parameters(self) -> None:
        """Leave the fake unchanged."""


class InjectedProjection(nn.Linear):
    """Independent weighted projection used by the slot-contract test."""

    class Config(Fig["InjectedProjection"]):
        """Projection input/output widths for the independent fake."""

        channels_in: int = 16
        """Residual rank accepted by the injected projection."""

        channels_out: int = 21
        """Task-vector width produced by the injected projection."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Return a zero fake cost for injection-only coverage."""
            del seq_len, batch_size, dtype, kwargs
            return Cost()

    def __init__(self, config: Config) -> None:
        super().__init__(config.channels_in, config.channels_out, bias=False)
        self.channels_in = config.channels_in
        self.channels_out = config.channels_out


@pytest.mark.parametrize("gated", [False, True])
def test_pinned_source_tensor_golden(gated: bool) -> None:
    """Replay source init, RNG, output, dense grads, and sparse local grads."""
    suffix = "gated" if gated else "ungated"
    expected = read_tensors(_TESTDATA / "charm_memory.pt")
    with host_agnostic_numerics():
        torch.manual_seed(0xC0E5)
        module = _source_module(gated=gated)
        local_params = list(module.named_parameters())
        source_names = (
            "composition.slot_base",
            "composition.slot_embeddings",
            "composition.category_embed.weight",
            "composition.augmentation_embed.weight",
            "composition.film_gamma.weight",
            "composition.film_gamma.bias",
            "composition.film_beta.weight",
            "composition.film_beta.bias",
        )
        if gated:
            source_names += ("residual_gate.linear.weight", "residual_gate.linear.bias")
        source_names += ("residual_projection.weight",)
        names = tuple(name for name, _ in local_params)
        assert names == source_names
        shapes = torch.zeros(len(local_params), 4, dtype=torch.int64)
        for index, (_, parameter) in enumerate(local_params):
            shapes[index, 0] = parameter.ndim
            shapes[index, 1 : parameter.ndim + 1] = torch.tensor(parameter.shape)
        assert torch.equal(shapes, expected[f"{suffix}/param_shapes"])
        values = torch.cat(
            [parameter.detach().flatten() for _, parameter in local_params],
        )
        assert torch.equal(values, expected[f"{suffix}/initial_params"])
        assert torch.equal(rng_fingerprint(), expected[f"{suffix}/rng_next8"])
        roles = torch.tensor(
            [1 if parameter.ndim == 2 else 0 for _, parameter in local_params],
            dtype=torch.int64,
        )
        assert torch.equal(roles, expected[f"{suffix}/param_roles"])
        assert torch.equal(module.residual.weights, expected[f"{suffix}/initial_table"])
        _source_set_parameters(module)
        inputs = _source_inputs()
        prefix = module(3, **inputs)
        weights = torch.arange(78, dtype=torch.float32).reshape(3, 2, 13) / 17 + 1
        (prefix.float() * weights).sum().backward()
        assert torch.equal(prefix, expected[f"{suffix}/prefix"])
        assert torch.equal(module.residual.local_ids, expected[f"{suffix}/local_ids"])
        assert module.residual.local_weights.grad is not None
        assert torch.equal(
            module.residual.local_weights.grad,
            expected[f"{suffix}/local_grad"],
        )
        dense_grads = [
            parameter.grad.detach().flatten()
            for _, parameter in local_params
            if parameter.grad is not None
        ]
        assert len(dense_grads) == len(local_params)
        assert torch.equal(
            torch.cat(dense_grads),
            expected[f"{suffix}/dense_grads"],
        )
        if module.residual_gate is not None:
            assert isinstance(module.residual_gate, SigmoidResidualGate)
            gate_grad = module.residual_gate.linear.weight.grad
            assert gate_grad is not None
            assert torch.count_nonzero(gate_grad) > 0


def test_prefix_and_complete_sudoku_costs_include_co_se() -> None:
    """The prefix prices child work and plugs into a complete Sudoku config."""
    prefix = _source_config(gated=True).finalize()
    prefix_cost = prefix.cost(
        seq_len=1,
        batch_size=3,
        dtype=torch.bfloat16,
    )
    prefix_module = prefix.make()
    assert prefix_cost.params == sum(
        parameter.numel() for parameter in prefix_module.parameters()
    )
    assert prefix_cost["flops", "primal"].sum() > 0
    assert prefix_cost["bytes", "primal", "selection"].sum() > 0

    model = GridSolver.Config(channels_in=13, num_layers=1, vocab_size=11)
    model.embedding = GridEmbedding.Config(grid_shape=(2,))
    model.block = Identity.Config()
    model.prefix = prefix
    complete = model.finalize().cost(batch_size=3, dtype=torch.bfloat16)

    plain_model = GridSolver.Config(channels_in=13, num_layers=1, vocab_size=11)
    plain_model.embedding = GridEmbedding.Config(grid_shape=(2,))
    plain_model.block = Identity.Config()
    plain = plain_model.finalize().cost(batch_size=3, dtype=torch.bfloat16)
    assert complete.params - plain.params == prefix_cost.params
    assert complete["flops", "primal"].sum() > plain["flops", "primal"].sum()


def test_categorical_film_caps_active_rows_to_the_tables_that_can_be_read() -> None:
    """Active ownership includes unique task/dihedral rows and all color slots."""
    cost = CategorySlotFiLM.Config(
        num_categories=2,
        num_augmentations=8,
        num_slots=9,
        channels_out=12,
    ).cost(seq_len=1, batch_size=16, dtype=torch.bfloat16)

    assert cost.params == 732
    assert cost.params_active == 732


@pytest.mark.parametrize("gated", [False, True])
def test_prefix_matches_closed_form_and_has_gradients(gated: bool) -> None:
    """Exercise nonzero slot permutation, FiLM, residual, and gate paths."""
    module = _module(gated=gated)
    _set_parameters(module)
    inputs = _inputs()

    actual = module(3, **inputs)
    expected = _closed_form(module, inputs=inputs)

    assert actual.shape == (3, 2, 21)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[:, 0]) > 0
    assert torch.equal(actual[:, 1], torch.zeros(3, 21))
    assert torch.equal(actual[:, 0], expected[:, 0])
    if module.residual_gate is not None:
        assert isinstance(module.residual_gate, SigmoidResidualGate)
        composition = module.composition
        assert isinstance(composition, CategorySlotFiLM)
        logits = module.residual_gate.linear(
            composition(
                category_indices=inputs["category_indices"],
                augmentation_indices=inputs["augmentation_indices"],
                slot_permutation=inputs["slot_permutation"],
            ),
        )
        gates = torch.sigmoid(logits)
        assert torch.all(gates > 0)
        assert torch.all(gates < 1)
    actual.sum().backward()
    composition = module.composition
    assert isinstance(composition, CategorySlotFiLM)
    assert composition.category_embed.weight.grad is not None
    assert composition.augmentation_embed.weight.grad is not None
    assert composition.slot_embeddings.grad is not None
    assert composition.film_gamma.weight.grad is not None
    assert composition.film_beta.weight.grad is not None
    assert module.residual.local_weights.grad is not None
    assert module.residual_projection.weight.grad is not None
    if module.residual_gate is not None:
        assert isinstance(module.residual_gate, SigmoidResidualGate)
        gate_grad = module.residual_gate.linear.weight.grad
        assert gate_grad is not None
        assert torch.count_nonzero(gate_grad) > 0


def test_rank_only_residual_width_updates_the_lookup_and_projection() -> None:
    """Rank alone sets the sparse lookup width and projection boundary."""
    config = ComposedSparsePrefix.Config(channels_out=21)
    config.composition = CategorySlotFiLM.Config(
        num_categories=5,
        num_augmentations=8,
        num_slots=9,
    )
    config.residual.num_puzzles = 7
    config.residual.batch_size = 3
    config.residual.channels_out = 16
    config.residual.dtype = torch.float32
    config.residual_gate = None

    finalized = config.finalize()
    module = finalized.make()

    assert finalized.residual.channels_token == 16
    assert finalized.residual_projection.channels_in == 16
    assert module.residual_projection.weight.shape == (21, 16)
    output = module(
        3,
        category_indices=torch.tensor([1, 2, 3]),
        augmentation_indices=torch.tensor([0, 1, 2]),
        slot_permutation=torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                [8, 7, 6, 5, 4, 3, 2, 1, 0],
                [1, 3, 5, 7, 0, 2, 4, 6, 8],
            ],
        ),
        puzzle_identifiers=torch.tensor([0, 1, 2]),
    )
    assert output.shape == (3, 16, 21)
    assert torch.equal(output[:, 1:], torch.zeros(3, 15, 21))


def test_independent_composition_gate_and_projection_injection() -> None:
    """Typed child slots accept unrelated modules and execute their contracts."""
    config = ComposedSparsePrefix.Config(channels_out=21)
    config.composition = InjectedComposer.Config()
    config.residual_gate = InjectedGate.Config()
    config.residual_projection = InjectedProjection.Config(channels_in=16)
    config.residual.channels_out = 16
    config.residual.num_puzzles = 7
    config.residual.batch_size = 3

    module = config.make()

    assert isinstance(module.composition, InjectedComposer)
    assert isinstance(module.residual_gate, InjectedGate)
    assert isinstance(module.residual_projection, InjectedProjection)
    output = module(
        3,
        category_indices=torch.tensor([1, 2, 3]),
        augmentation_indices=torch.tensor([0, 1, 2]),
        slot_permutation=torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                [8, 7, 6, 5, 4, 3, 2, 1, 0],
                [1, 3, 5, 7, 0, 2, 4, 6, 8],
            ],
        ),
        puzzle_identifiers=torch.tensor([0, 1, 2]),
    )
    assert output.shape == (3, 16, 21)
    assert torch.equal(
        output[:, 0],
        torch.arange(1, 4).float().unsqueeze(-1).expand(-1, 21),
    )
    assert torch.equal(output[:, 1:], torch.zeros(3, 15, 21))


def test_slotperm_retains_source_head_axes_and_optimizer_roles() -> None:
    """Source head=1 is retained; matrix and tensor roles stay distinct."""
    module = _module()
    composition = module.composition
    assert isinstance(composition, CategorySlotFiLM)

    assert composition.slot_base.shape == (1, 3)
    assert composition.slot_embeddings.shape == (1, 9, 2)
    params = dict(module.named_parameters())
    assert params["composition.slot_base"].ndim == 2
    assert params["composition.slot_embeddings"].ndim == 3


def test_training_uses_residual_gradient_buffer_and_eval_uses_table() -> None:
    """The residual lookup remains sparse during training and direct in eval."""
    module = _module(gated=False)
    _set_parameters(module)
    inputs = _inputs()

    module.train()
    train_output = module(3, **inputs)
    train_output.sum().backward()
    assert module.residual.local_weights.grad is not None
    assert module.residual.weights.grad is None
    assert torch.equal(
        module.residual.local_ids,
        inputs["puzzle_identifiers"].int(),
    )

    module.eval()
    eval_output = module(3, **inputs)
    assert torch.equal(eval_output, _closed_form(module, inputs=inputs))
    assert module.residual.local_weights.grad is not None
    assert module.residual.weights.grad is None


def test_prefix_preserves_configured_dtype_after_module_move() -> None:
    """Moving the module keeps the sparse gradient buffer trainable."""
    module = _module(gated=True, dtype=torch.float64).to(dtype=torch.float64)
    _set_parameters(module)
    output = module(3, **_inputs())

    assert output.dtype == torch.float64
    assert module.residual.local_weights.dtype == torch.float64
    assert module.residual.local_weights.requires_grad
    assert module.residual.local_ids.dtype == torch.int32
    output.sum().backward()
    assert module.residual.local_weights.grad is not None


def test_default_gate_starts_with_a_large_open_probability() -> None:
    """The gate's zero weights and bias two match the reference init."""
    gate = SigmoidResidualGate.Config(channels_out=21).make()

    assert gate.channels_out == 21
    assert torch.equal(gate.linear.weight, torch.zeros_like(gate.linear.weight))
    bias = gate.linear.bias
    assert bias is not None
    assert torch.equal(bias, torch.full_like(bias, 2.0))
    expected = torch.sigmoid(torch.tensor(2.0)).expand(2, 21)
    assert torch.equal(gate(torch.zeros(2, 21)), expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_prefix_cast_and_padding_match_closed_form(dtype: torch.dtype) -> None:
    """The output cast precedes zero-padding; model owner scales the sequence."""
    module = _module(gated=True, dtype=dtype)
    _set_parameters(module)

    actual = module(3, **_inputs())

    assert actual.dtype == dtype
    assert torch.equal(actual, _closed_form(module, inputs=_inputs()))


@pytest.mark.parametrize(
    ("channels_out", "num_tokens"),
    [(0, 2), (21, 0)],
)
def test_invalid_prefix_shape_is_rejected(channels_out: int, num_tokens: int) -> None:
    """The prefix requires positive width and at least one token."""
    config = ComposedSparsePrefix.Config(
        channels_out=channels_out,
        num_tokens=num_tokens,
    )
    with pytest.raises(ValueError, match=r"positive|num_tokens"):
        config.make()


def test_config_inherits_child_widths() -> None:
    """The prefix finalizer fills injected child widths from its own width."""
    config = ComposedSparsePrefix.Config(channels_out=21)
    config.residual.channels_in = 2
    config.residual.channels_out = 4

    finalized = config.finalize()

    assert isinstance(finalized.composition, CategorySlotFiLM.Config)
    assert finalized.composition.channels_out == 21
    assert finalized.residual_projection.channels_in == 2
    assert finalized.residual_projection.channels_out == 21
    assert isinstance(finalized.residual_gate, SigmoidResidualGate.Config)
    assert finalized.residual_gate.channels_out == 21


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("category_indices", "wrong"),
        ("augmentation_indices", None),
        ("slot_permutation", 1),
        ("puzzle_identifiers", False),
    ],
)
def test_prefix_rejects_non_tensor_kwargs(name: str, value: object) -> None:
    """Every required tensor is checked before composition or lookup."""
    inputs: dict[str, object] = dict(_inputs())
    inputs[name] = value
    with pytest.raises(TypeError, match=name):
        _module()(3, **inputs)


def test_slot_film_rejects_nonpositive_width() -> None:
    """The composition must have room for a positive task vector."""
    config = ComposedSparsePrefix.Config(channels_out=0)
    config.composition.channels_out = 0

    with pytest.raises(
        ValueError,
        match="CategorySlotFiLM requires positive channels_out",
    ):
        config.composition.make()


def test_gate_none_remains_ungated_after_finalize() -> None:
    """The ungated ARC2 behavior survives config finalization."""
    config = ComposedSparsePrefix.Config(channels_out=21, residual_gate=None)
    config.residual.channels_in = 2
    config.residual.channels_out = 4

    finalized = config.finalize()

    assert finalized.residual_gate is None


def _source_module(*, gated: bool) -> ComposedSparsePrefix:
    """Build the minimum source-parity module geometry."""
    return _source_config(gated=gated).make()


def _source_config(*, gated: bool) -> ComposedSparsePrefix.Config:
    """Build source-parity geometry; production uses hidden=512/prefix=16/rank=32."""
    config = ComposedSparsePrefix.Config(
        channels_out=13,
        num_tokens=2,
        forward_dtype=torch.bfloat16,
        composition=CategorySlotFiLM.Config(
            num_categories=7,
            num_augmentations=8,
            num_slots=9,
        ),
        residual_gate=SigmoidResidualGate.Config() if gated else None,
    )
    config.residual.num_puzzles = 11
    config.residual.channels_out = 4
    config.residual.num_tokens = 1
    config.residual.batch_size = 3
    config.residual.dtype = torch.bfloat16
    config.residual.embed_scale = 1.0
    return config


def _source_set_parameters(module: ComposedSparsePrefix) -> None:
    """Install the source harness's deterministic nonzero parameter values."""
    composition = module.composition
    assert isinstance(composition, CategorySlotFiLM)
    with torch.no_grad():
        composition.category_embed.weight.copy_(
            torch.arange(91, dtype=torch.float32).reshape(7, 13) / 64,
        )
        composition.augmentation_embed.weight.copy_(
            torch.arange(104, dtype=torch.float32).reshape(8, 13) / 128,
        )
        # Immutable CHARM source oracle fixes the slot-base shape at [1, 4].
        composition.slot_base.copy_(torch.arange(4).reshape(1, 4) / 16)
        # Immutable CHARM source oracle keeps one rank-one row per color slot.
        composition.slot_embeddings.copy_(torch.arange(9).reshape(1, 9, 1) / 32)
        composition.film_gamma.weight.copy_(
            torch.arange(338, dtype=torch.float32).reshape(13, 26) / 1024,
        )
        assert composition.film_gamma.bias is not None
        composition.film_gamma.bias.copy_(torch.arange(13) / 32)
        composition.film_beta.weight.copy_(
            torch.arange(338, dtype=torch.float32).flip(0).reshape(13, 26) / 1024,
        )
        assert composition.film_beta.bias is not None
        composition.film_beta.bias.copy_(-torch.arange(13) / 64)
        module.residual.weights.copy_(
            torch.arange(44, dtype=torch.float32).reshape(11, 4) / 32,
        )
        module.residual_projection.weight.copy_(
            torch.arange(52, dtype=torch.float32).reshape(13, 4) / 128,
        )


def _source_inputs() -> dict[str, Tensor]:
    """Return independent semantic and instance IDs with nontrivial slots."""
    return {
        "category_indices": torch.tensor([1, 6, 2]),
        "augmentation_indices": torch.tensor([7, 2, 5]),
        "slot_permutation": torch.tensor(
            [
                [8, 1, 6, 3, 0, 5, 2, 7, 4],
                [2, 5, 8, 1, 4, 7, 0, 3, 6],
                [4, 0, 7, 2, 6, 1, 8, 5, 3],
            ],
        ),
        "puzzle_identifiers": torch.tensor([7, 1, 9]),
    }


def _module(
    *,
    gated: bool = True,
    dtype: torch.dtype = torch.float32,
) -> ComposedSparsePrefix:
    # Small test geometry: production uses hidden=512, prefix=16, rank=32,
    # semantic task count and instance count from separate dataset mappings.
    config = ComposedSparsePrefix.Config()
    config.channels_out = 21
    config.forward_dtype = dtype
    config.num_tokens = 2
    config.composition = CategorySlotFiLM.Config(
        num_categories=5,
        num_augmentations=8,
        num_slots=9,
        channels_out=21,
    )
    config.residual.num_puzzles = 7
    config.residual.channels_in = 2
    config.residual.channels_out = 4
    config.residual.num_tokens = 2
    config.residual.batch_size = 3
    config.residual.embed_scale = 1.0
    config.residual.dtype = dtype
    config.residual_projection.channels_in = 4
    config.residual_projection.channels_out = 21
    config.residual_gate = (
        SigmoidResidualGate.Config(channels_out=21) if gated else None
    )
    return config.make()


def _set_parameters(module: ComposedSparsePrefix) -> None:
    composition = module.composition
    assert isinstance(composition, CategorySlotFiLM)
    with torch.no_grad():
        composition.category_embed.weight.copy_(
            torch.arange(105, dtype=torch.float32).reshape(5, 21) / 16,
        )
        composition.augmentation_embed.weight.copy_(
            torch.arange(168, dtype=torch.float32).reshape(8, 21) / 32,
        )
        # CategorySlotFiLM stores one base row for this three-value fixture.
        composition.slot_base.copy_(
            torch.arange(3, dtype=torch.float32).reshape(1, 3) / 8,
        )
        # CategorySlotFiLM stores one group axis and nine color slots.
        composition.slot_embeddings.copy_(
            torch.arange(18, dtype=torch.float32).reshape(1, 9, 2) / 8,
        )
        composition.film_gamma.weight.copy_(
            torch.arange(882, dtype=torch.float32).reshape(21, 42) / 512,
        )
        gamma_bias = composition.film_gamma.bias
        assert gamma_bias is not None
        gamma_bias.copy_(
            torch.arange(21, dtype=torch.float32) / 16,
        )
        composition.film_beta.weight.copy_(
            torch.arange(882, dtype=torch.float32).flip(0).reshape(21, 42) / 512,
        )
        beta_bias = composition.film_beta.bias
        assert beta_bias is not None
        beta_bias.copy_(
            -torch.arange(21, dtype=torch.float32) / 32,
        )
        module.residual.weights.copy_(
            torch.arange(28, dtype=torch.float32).reshape(7, 4) / 16,
        )
        module.residual_projection.weight.copy_(
            torch.arange(84, dtype=torch.float32).reshape(21, 4) / 32,
        )
        if module.residual_gate is not None:
            assert isinstance(module.residual_gate, SigmoidResidualGate)
            # SigmoidResidualGate maps the task width back to the same width.
            module.residual_gate.linear.weight.copy_(
                torch.arange(441, dtype=torch.float32).reshape(21, 21) / 50_000_000,
            )
            gate_bias = module.residual_gate.linear.bias
            assert gate_bias is not None
            gate_bias.copy_(
                torch.arange(21, dtype=torch.float32) / 1_600_000 - 0.00000625,
            )


def _inputs() -> dict[str, Tensor]:
    return {
        "category_indices": torch.tensor([1, 4, 0]),
        "augmentation_indices": torch.tensor([7, 2, 5]),
        "slot_permutation": torch.tensor(
            [
                [8, 1, 6, 3, 0, 5, 2, 7, 4],
                [2, 5, 8, 1, 4, 7, 0, 3, 6],
                [4, 0, 7, 2, 6, 1, 8, 5, 3],
            ],
        ),
        "puzzle_identifiers": torch.tensor([4, 1, 3]),
    }


def _closed_form(
    module: ComposedSparsePrefix,
    *,
    inputs: dict[str, Tensor],
) -> Tensor:
    composition = module.composition
    assert isinstance(composition, CategorySlotFiLM)
    puzzle = composition.category_embed(inputs["category_indices"])
    dih = composition.augmentation_embed(inputs["augmentation_indices"])
    slots = composition.slot_embeddings.expand(3, -1, -1, -1)
    # CategorySlotFiLM gathers one indexed group over its slot_dim axis.
    slot_indices = (
        inputs["slot_permutation"].unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, 2)
    )
    slots = torch.gather(slots, 2, slot_indices)
    # This fixture's ARC color permutation contains nine slots.
    colorperm = torch.cat(
        [
            composition.slot_base.expand(3, -1),
            slots.flatten(start_dim=2).squeeze(1),
        ],
        dim=-1,
    )
    aug = torch.cat([dih, colorperm], dim=-1)
    gamma = composition.film_gamma(aug)
    beta = composition.film_beta(aug)
    combined = (1 + gamma) * puzzle + beta
    residual_rows = module.residual(
        3,
        puzzle_identifiers=inputs["puzzle_identifiers"],
    ).flatten(start_dim=1)
    residual_rows = residual_rows.to(module.residual_projection.weight.dtype)
    residual = module.residual_projection(residual_rows)
    if module.residual_gate is not None:
        assert isinstance(module.residual_gate, SigmoidResidualGate)
        gate = torch.sigmoid(module.residual_gate.linear(combined))
        residual = residual.to(combined.dtype) * gate
    combined = combined + residual.to(combined.dtype)
    prefix = combined.to(module.forward_dtype).unsqueeze(1)
    return functional.pad(prefix, (0, 0, 0, 1))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
