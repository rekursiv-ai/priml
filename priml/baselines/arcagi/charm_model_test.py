"""Source-minted numerical contracts for the generic ARC grid solver."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import torch

from priml.baselines.arcagi.charm_checkpoint import (
    _target_key,
    remap_charm_checkpoint,
)
from priml.baselines.arcagi.experiments import (
    _SOURCE_DEPTHWISE_KERNEL_2_BIAS_INIT,
)
from priml.cost import cost
from priml.math.linear import cast_parameters_to_input
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.conditioning import CategorySlotFiLM
from priml.model.conv import Conv1d, SequenceConv1d
from priml.model.embedding import Embedding
from priml.model.grid_embedding import GridEmbedding
from priml.model.grid_solver import SingleLatentGridSolver
from priml.model.init import corrected_fan_in_normal
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.prefix import ComposedSparsePrefix
from priml.model.recurrence import SingleLatentRecurrence
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import read_tensors


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    import pytest


def test_source_minted_model_init_forward_and_gradients_are_bit_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay the immutable CPU oracle against generic priml composition."""
    with host_agnostic_numerics():
        source = read_tensors(_CWD / "testdata" / "charm_model.pt")
        checkpoint = read_tensors(_CWD / "testdata" / "charm_checkpoint.pt")
        live = {
            key.removeprefix("live/"): value
            for key, value in checkpoint.items()
            if key.startswith("live/")
        }

        torch.manual_seed(51_405)
        model = _source_parity_config().make()

        expected_state = remap_charm_checkpoint(live, model=model)
        assert model.state_dict().keys() == expected_state.keys()
        for name, value in model.state_dict().items():
            assert torch.equal(value, expected_state[name]), name

        generator = torch.Generator()
        generator.set_state(torch.get_rng_state())
        actual_next8 = torch.randint(2_147_483_647, (8,), generator=generator)
        assert torch.equal(actual_next8, source["rng/init_next8"])

        block = cast(TransformerBlock, model.reasoning[0])
        attention = cast(Attention, block.attn)
        probe = torch.linspace(-1, 1, 2 * 22 * 12, dtype=torch.bfloat16).reshape(
            2,
            22,
            12,
        )
        positions = torch.arange(22)
        assert model.rope is not None
        cos, sin = model.rope(positions)
        assert torch.equal(
            attention(probe, cos_sin=(cos, sin)),
            source["probe/attention"],
        )
        assert torch.equal(block.ffn(probe), source["probe/conv"])

        norm = RMSNorm.Config(channels_in=12, eps=1e-5).make()
        norm_input = torch.linspace(-1, 1, 2 * 22 * 12, dtype=torch.bfloat16).reshape(
            2,
            22,
            12,
        )
        assert torch.equal(norm(norm_input), source["probe/norm"])

        tokens = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 5, 4, 3, 2, 1]])
        category_indices = torch.tensor([0, 2])
        augmentation_indices = torch.tensor([7, 3])
        slot_permutation = torch.tensor(
            [[8, 1, 6, 3, 0, 5, 2, 7, 4], [2, 5, 8, 1, 4, 7, 0, 3, 6]],
        )
        puzzle_identifiers = torch.tensor([1, 4])
        captured: list[torch.Tensor] = []
        original_embed = cast(
            Callable[[torch.Tensor, dict[str, object]], torch.Tensor],
            model._embed,
        )

        def capture_embedding(
            input_tokens: torch.Tensor,
            prefix_kwargs: dict[str, object],
        ) -> torch.Tensor:
            value = original_embed(input_tokens, prefix_kwargs)
            value.retain_grad()
            captured.append(value)
            return value

        monkeypatch.setattr(model, "_embed", capture_embedding)
        output = model(
            tokens,
            category_indices=category_indices,
            augmentation_indices=augmentation_indices,
            slot_permutation=slot_permutation,
            puzzle_identifiers=puzzle_identifiers,
        )
        weights = torch.arange(84, dtype=torch.float32).reshape(2, 6, 7) / 64 + 1
        objective = (output.logits.float() * weights).sum() + output.halt.sum()
        objective.backward()

        assert len(captured) == 1
        embedded = captured[0]
        assert torch.equal(embedded, source["forward/input_embedding"])
        assert embedded.grad is not None
        assert torch.equal(embedded.grad, source["forward/input_embedding_grad"])
        assert torch.equal(output.logits, source["forward/logits"])
        assert torch.equal(output.halt, source["forward/halt"])
        assert torch.equal(output.z_slow, source["forward/hidden"])
        assert torch.equal(objective, source["forward/objective"])

        parameters = dict(model.named_parameters())
        source_gradient_keys = [
            key.removeprefix("gradient/")
            for key in source
            if key.startswith("gradient/")
        ]
        assert [_target_key(key) for key in source_gradient_keys] == list(parameters)
        for source_key in source_gradient_keys:
            parameter = parameters[_target_key(source_key)]
            assert parameter.grad is not None
            assert torch.equal(
                parameter.grad,
                source[f"gradient/{source_key}"],
            ), source_key


def test_single_latent_cost_prices_real_forward_and_gradient_pass_counts() -> None:
    """Cost uses the recurrence's declared executions and counts head ownership once."""
    config = _config().finalize()
    config.recurrence = SingleLatentRecurrence.Config(
        slow_cycles=3,
        fast_cycles=2,
        inner_grad_loops=4,
    )
    total = config.cost(batch_size=2, dtype=torch.bfloat16)
    block = cost(
        cast(TransformerBlock.Config, config.block),
        seq_len=6,
        batch_size=2,
        dtype=torch.bfloat16,
    ).tile(config.num_layers, copies=config.num_layers)
    assert total["flops", "primal"].sum() >= (3 * 4 * block["flops", "primal"].sum())
    module = config.make()
    assert total.params == (
        cast(GridEmbedding.Config, config.embedding)
        .cost(seq_len=6, batch_size=2, dtype=torch.bfloat16)
        .params
        + block.params
        + config.head.cost(seq_len=6, batch_size=2, dtype=torch.bfloat16).params
        + config.halt_head.cost(seq_len=1, batch_size=2, dtype=torch.bfloat16).params
        + config.recurrence.cost(
            seq_len=6,
            batch_size=2,
            dtype=torch.bfloat16,
        ).params
    )
    assert total.params == sum(parameter.numel() for parameter in module.parameters())
    assert total.bytes_state == 2 * config.channels_in * torch.bfloat16.itemsize


def test_single_latent_attention_uses_rank_two_generic_projection_slots() -> None:
    """The native block keeps the source's Muon-compatible rank-two matrices."""
    model = _config().make()
    block = cast(TransformerBlock, model.reasoning[0])
    attention = cast(Attention, block.attn)

    assert attention.proj_qkv.weight.shape == (36, 12)
    assert attention.proj_qkv.weight.ndim == 2
    assert attention.proj_out.weight.shape == (12, 12)
    assert cast(SwiGLU, block.ffn).middle is not None
    assert [name for name, _ in model.named_parameters()][:4] == [
        "embedding.embed_tokens.weight",
        "head.weight",
        "halt_head.weight",
        "halt_head.bias",
    ]


def _config() -> SingleLatentGridSolver.Config:
    config = SingleLatentGridSolver.Config()
    config.channels_in = 12
    config.num_layers = 1
    config.vocab_size = 7
    embedding = cast(GridEmbedding.Config, config.embedding)
    embedding.grid_shape = (2, 3)
    embedding.embed_scale = 1.0
    embedding.token_table = Embedding.Config(
        init_weight=corrected_fan_in_normal,
        lookup_dtype=torch.bfloat16,
    )
    block = TransformerBlock.Config(prenorm=False)
    attention = Attention.Config(
        channels_head=4,
        num_heads=3,
        init_weight=corrected_fan_in_normal,
    )
    config.rope = RoPE.Config(channels_head=4)
    attention.proj_qkv = Linear.Config(
        bias=False,
        init_weight=corrected_fan_in_normal,
        linear_fn=cast_parameters_to_input,
    )
    attention.proj_out = Linear.Config(
        bias=False,
        init_weight=corrected_fan_in_normal,
        linear_fn=cast_parameters_to_input,
    )
    block.attn = attention
    ffn = SwiGLU.Config(
        channels_hidden=9,
        expansion=4.0,
        init_weight=corrected_fan_in_normal,
        init_weight_out=corrected_fan_in_normal,
    )
    ffn.up_proj = Linear.Config(
        bias=False,
        init_weight=corrected_fan_in_normal,
        linear_fn=cast_parameters_to_input,
    )
    ffn.down_proj = Linear.Config(
        bias=False,
        init_weight=corrected_fan_in_normal,
        linear_fn=cast_parameters_to_input,
    )
    ffn.middle = SequenceConv1d.Config(
        conv=Conv1d.Config(
            kernel_size=2,
            padding=1,
            groups=9,
            bias=True,
            dtype=torch.bfloat16,
            init_bias=_SOURCE_DEPTHWISE_KERNEL_2_BIAS_INIT,
        ),
    )
    block.ffn = ffn
    block.norm1 = RMSNorm.Config(eps=1e-5)
    block.norm2 = RMSNorm.Config(eps=1e-5)
    config.block = block
    config.halt_outputs = 1
    config.recurrence = SingleLatentRecurrence.Config(
        slow_cycles=1,
        fast_cycles=2,
        inner_grad_loops=1,
    )
    config.prefix = None
    return config


def _source_parity_config() -> SingleLatentGridSolver.Config:
    config = _config()
    config.num_layers = 1
    config.recurrence = SingleLatentRecurrence.Config(
        slow_cycles=1,
        fast_cycles=4,
        inner_grad_loops=2,
    )
    embedding = cast(GridEmbedding.Config, config.embedding)
    embedding.embed_scale = 1.0
    prefix = ComposedSparsePrefix.Config()
    prefix.channels_out = 12
    prefix.num_tokens = 16
    composition = cast(CategorySlotFiLM.Config, prefix.composition)
    composition.num_categories = 3
    composition.num_augmentations = 8
    composition.num_slots = 9
    prefix.residual.num_puzzles = 5
    prefix.residual.num_tokens = 1
    prefix.residual.channels_in = 4
    prefix.residual.channels_out = 4
    prefix.residual.batch_size = 2
    prefix.residual.dtype = torch.bfloat16
    prefix.residual.dtype_scale = torch.bfloat16
    prefix.residual.embed_scale = 1.0
    prefix.residual_projection.channels_in = 4
    prefix.residual_projection.channels_out = 12
    config.prefix = prefix
    return config


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
