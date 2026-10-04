"""Tests for TRM model behavior and configuration."""

from __future__ import annotations

from typing import cast

from torch import Tensor

import pytest
import torch

from priml.baselines.sudoku import trm
from priml.baselines.sudoku.trm import (
    TRM,
    SparsePuzzleEmbedding,
    _corrected,
    _feedback_table,
    _pos_table,
    recipe_block,
    trm_truncated_normal_corrected,
)
from priml.model.attention.attention import Attention
from priml.model.init import truncated_normal
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU


def _config(**overrides: object) -> TRM.Config:
    config = TRM.Config(
        vocab_size=5,
        puzzle_grid_shape=(2, 3),
        channels_in=12,
        num_layers=1,
        num_heads=3,
        slow_cycles=2,
        fast_cycles=2,
        num_puzzle_identifiers=2,
        puzzle_emb_len=2,
        puzzle_emb_ndim=12,
        puzzle_emb_batch_size=2,
        pos2d_grid_shape=(2, 3),
        pos2d_box_shape=(2, 1),  # Production geometry permits one-cell box width.
        compile=False,
        dtype=None,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_recipe_block_passes_explicit_swiglu_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def capture_config(**kwargs: object) -> dict[str, object]:
        return kwargs

    monkeypatch.setattr(SwiGLU, "Config", capture_config)
    recipe = recipe_block()

    assert isinstance(recipe.ffn, dict)
    assert recipe.ffn == {
        "expansion": 8 / 3,
        "round_to": 256,
        "gate": True,
        "norm": recipe.ffn["norm"],
        "init_weight": trm_truncated_normal_corrected,
        "init_weight_out": trm_truncated_normal_corrected,
    }


def test_config_finalizes_recipe_and_rejects_invalid_geometry() -> None:
    recipe = recipe_block()
    recipe_attn = recipe.attn
    recipe_ffn = recipe.ffn
    assert isinstance(recipe_attn, Attention.Config)
    assert recipe_attn.num_heads == -1
    assert isinstance(recipe_attn.norm_qk, RMSNorm.Config)
    assert recipe_attn.init_weight is trm_truncated_normal_corrected
    assert isinstance(recipe_ffn, SwiGLU.Config)
    assert recipe_ffn.expansion == 8 / 3
    assert recipe_ffn.round_to == 256
    assert recipe_ffn.gate is True
    assert isinstance(recipe_ffn.norm, RMSNorm.Config)
    assert recipe_ffn.init_weight is trm_truncated_normal_corrected
    assert recipe_ffn.init_weight_out is trm_truncated_normal_corrected
    assert recipe.prenorm is False
    assert isinstance(recipe.norm1, RMSNorm.Config)
    assert recipe.norm1.eps == 1e-5
    assert isinstance(recipe.norm2, RMSNorm.Config)
    assert recipe.norm2.eps == 1e-5

    config = _config().finalize()
    assert config.block is not None
    assert config.block.channels_in == 12
    assert isinstance(config.block.attn, Attention.Config)
    assert config.block.attn.num_heads == 3
    assert config.block.attn.channels_head == 4
    assert isinstance(config.block.attn.norm_qk, RMSNorm.Config)
    assert config.block.attn.init_weight is trm_truncated_normal_corrected
    assert isinstance(config.block.ffn, SwiGLU.Config)
    assert config.block.ffn.expansion == 8 / 3
    assert config.block.ffn.round_to == 256
    assert config.block.ffn.gate
    assert isinstance(config.block.norm1, RMSNorm.Config)
    assert config.block.norm1.eps == 1e-5
    assert isinstance(config.block.norm2, RMSNorm.Config)
    assert config.block.norm2.eps == 1e-5
    assert not config.block.prenorm
    assert config.puzzle_emb_ndim == 12
    with pytest.raises(ValueError, match="does not factor"):
        _config(pos2d_grid_shape=(2, 2)).make()
    with pytest.raises(ValueError, match="does not tile"):
        _config(pos2d_box_shape=(2, 2)).make()
    with pytest.raises(
        ValueError,
        match=r"^TRM requires vocabulary and grid shape from the dataset\.$",
    ):
        _config(vocab_size=-1).make()
    with pytest.raises(
        ValueError,
        match=r"^TRM requires pos2d grid and box shapes from the dataset\.$",
    ):
        _config(pos2d_grid_shape=(0, 3)).make()
    with pytest.raises(
        ValueError,
        match=r"^TRM requires pos2d grid and box shapes from the dataset\.$",
    ):
        _config(pos2d_box_shape=(0, 1)).make()


def test_constructor_rejects_unfinalized_block_with_exact_message() -> None:
    config = _config()
    config.block = None
    with pytest.raises(
        ValueError,
        match=r"^finalize\(\) fills the default block$",
    ) as error:
        TRM(config)
    assert str(error.value) == "finalize() fills the default block"


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("embed_pos_row", "Expected self.embed_pos_row is not None."),
        ("embed_pos_col", "Expected self.embed_pos_col is not None."),
        ("embed_pos_box", "Expected self.embed_pos_box is not None."),
    ],
)
def test_pos2d_rejects_each_missing_table_with_exact_message(
    attribute: str,
    message: str,
) -> None:
    model = _config().make()
    setattr(model, attribute, None)
    with pytest.raises(ValueError, match=message) as error:
        model._pos2d()
    assert str(error.value) == message


def test_forward_returns_grid_logits_and_consumes_feedback() -> None:
    model = _config().make()
    inputs = torch.tensor([[1, 2, 3, 1, 4, 2], [3, 1, 2, 4, 1, 3]])
    identifiers = torch.tensor([0, 1])
    z_slow, z_fast = model.init_z(2)
    feedback = torch.tensor([[2, 1, 3, 2, 1, 4], [1, 2, 3, 1, 2, 3]])

    model.set_feedback(feedback)
    output = model(inputs, z_slow, z_fast, identifiers)

    logits = cast(Tensor, output["logits"])
    all_logits = cast(list[Tensor], output["all_logits"])
    q_halt = cast(Tensor, output["q_halt"])
    assert logits.shape == (2, 6, 5)
    assert len(all_logits) == 2
    assert all(item.shape == (2, 6, 5) for item in all_logits)
    assert q_halt.shape == (2,)
    z_slow_output = output["z_slow"]
    assert isinstance(z_slow_output, Tensor)
    assert not z_slow_output.requires_grad
    assert model._feedback_ids is None


def test_forward_requires_puzzle_ids_and_act_step_has_expected_keys() -> None:
    model = _config().make()
    inputs = torch.tensor([[1, 2, 3, 1, 4, 2], [3, 1, 2, 4, 1, 3]])
    z_slow, z_fast = model.init_z(2)

    with pytest.raises(
        ValueError,
        match=r"^puzzle_identifiers is required when num_puzzle_identifiers > 0\.$",
    ):
        model(inputs, z_slow, z_fast)

    result = model.act_step(inputs, z_slow, z_fast, torch.tensor([0, 1]))
    assert set(result) == {"logits", "q_halt", "z_slow", "z_fast"}
    assert result["logits"].shape == (2, 6, 5)


def test_act_step_forwards_feedback_and_returns_only_act_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _config().make()
    inputs = torch.zeros(2, 6, dtype=torch.long)
    z_slow = torch.zeros(2, 8, 12)
    z_fast = torch.ones(2, 8, 12)
    identifiers = torch.tensor([0, 1])
    feedback = torch.full((2, 6), 3)
    logits = torch.ones(2, 6, 5)
    q_halt = torch.ones(2)
    next_slow = torch.full((2, 8, 12), 2.0)
    next_fast = torch.full((2, 8, 12), 3.0)
    calls: list[tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]] = []

    def forward(
        input_ids: Tensor,
        slow: Tensor,
        fast: Tensor,
        puzzle_ids: Tensor | None = None,
        feedback_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        assert puzzle_ids is not None
        assert feedback_ids is not None
        calls.append((input_ids, slow, fast, puzzle_ids, feedback_ids))
        return {
            "logits": logits,
            "q_halt": q_halt,
            "z_slow": next_slow,
            "z_fast": next_fast,
        }

    monkeypatch.setattr(model, "forward", forward)
    result = model.act_step(inputs, z_slow, z_fast, identifiers, feedback)

    assert len(calls) == 1
    assert all(
        actual is expected
        for actual, expected in zip(
            calls[0],
            (inputs, z_slow, z_fast, identifiers, feedback),
            strict=True,
        )
    )
    assert set(result) == {"logits", "q_halt", "z_slow", "z_fast"}
    assert result["logits"] is logits
    assert result["q_halt"] is q_halt
    assert result["z_slow"] is next_slow
    assert result["z_fast"] is next_fast


def test_single_output_q_head_returns_one_halt_logit_per_puzzle() -> None:
    model = _config(q_head_outputs=1, slow_cycles=1, fast_cycles=1).make()
    inputs = torch.tensor([[1, 2, 3, 1, 4, 2], [3, 1, 2, 4, 1, 3]])
    z_slow, z_fast = model.init_z(2)

    output = model(inputs, z_slow, z_fast, torch.tensor([0, 1]))
    q_halt = output["q_halt"]

    assert isinstance(q_halt, Tensor)
    assert q_halt.shape == (2,)


def test_sparse_puzzle_embedding_training_eval_and_device_apply() -> None:
    embedding = SparsePuzzleEmbedding(
        3,
        embedding_dim=4,
        batch_size=2,
        init_std=0.0,
        cast_to=torch.float64,
    )
    ids = torch.tensor([2, 0])
    train_output = embedding(ids)
    assert train_output.dtype == torch.float64
    assert train_output.shape == (2, 4)
    assert embedding.local_ids.tolist() == [2, 0]
    train_output.sum().backward()
    assert embedding.local_weights.grad is not None

    embedding.eval()
    eval_output = embedding(torch.tensor([1]))
    assert torch.equal(eval_output, embedding.weights[[1]].to(torch.float64))
    assert "local_weights" not in embedding.state_dict()
    embedding.to(torch.float64)
    assert embedding.local_weights.requires_grad


def test_init_helpers_use_expected_shapes_and_corrected_init() -> None:
    config = _config().finalize()
    torch.manual_seed(13)
    table = _pos_table(3, config)
    torch.manual_seed(13)
    expected_table = torch.empty(3, 12)
    truncated_normal(
        expected_table,
        std=config.pos2d_init_std / config.channels_in**0.5,
        depth_index=(),
        variance_correction=True,
    )
    assert torch.equal(table, expected_table)

    feedback_config = _config(feedback_init_std=0.5).finalize()
    torch.manual_seed(17)
    feedback = _feedback_table(feedback_config)
    torch.manual_seed(17)
    expected_feedback = torch.zeros(5, 12)
    truncated_normal(
        expected_feedback,
        std=feedback_config.feedback_init_std / feedback_config.channels_in**0.5,
        depth_index=(),
        variance_correction=True,
    )
    assert torch.equal(feedback, expected_feedback)
    assert torch.count_nonzero(_feedback_table(_config(feedback_init_std=0.0))) == 0

    torch.manual_seed(19)
    tensor = torch.empty(3, 4)
    assert _corrected(tensor, std=0.5) is tensor
    torch.manual_seed(19)
    expected = torch.empty(3, 4)
    truncated_normal(expected, std=0.5, depth_index=(), variance_correction=True)
    assert torch.equal(tensor, expected)

    torch.manual_seed(23)
    trm_truncated_normal_corrected(tensor, depth=99)
    torch.manual_seed(23)
    expected = torch.empty(3, 4)
    truncated_normal(expected, std=4**-0.5, depth_index=(), variance_correction=True)
    assert torch.equal(tensor, expected)


def test_init_helpers_pass_exact_truncation_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Tensor, dict[str, object]]] = []

    def capture(tensor: Tensor, **kwargs: object) -> None:
        calls.append((tensor, kwargs))

    monkeypatch.setattr(trm, "truncated_normal", capture)
    config = _config().finalize()
    _pos_table(3, config)
    _feedback_table(_config(feedback_init_std=0.5).finalize())
    zero_feedback = _feedback_table(config)
    _corrected(torch.empty(2, 3), std=0.25)
    weight = torch.empty(2, 3, 5)
    trm_truncated_normal_corrected(weight, depth=27)

    assert torch.count_nonzero(zero_feedback) == 0
    assert [tensor.shape for tensor, _ in calls] == [
        (3, 12),
        (5, 12),
        (2, 3),
        (2, 3, 5),
    ]
    assert [kwargs for _, kwargs in calls] == [
        {
            "std": 1.0 / 12**0.5,
            "depth_index": (),
            "variance_correction": True,
        },
        {
            "std": 0.5 / 12**0.5,
            "depth_index": (),
            "variance_correction": True,
        },
        {"std": 0.25, "depth_index": (), "variance_correction": True},
        {
            "std": 5**-0.5,
            "depth_index": (),
            "variance_correction": True,
        },
    ]


def test_run_h_cycles_collects_each_cycle_and_gradients_only_last() -> None:
    model = _config(slow_cycles=3).make()
    inputs = torch.tensor([[1, 2, 3, 1, 4, 2], [3, 1, 2, 4, 1, 3]])
    embedded, cos_sin = model._embed_and_prepare(inputs, torch.tensor([0, 1]))
    z_slow, z_fast = model.init_z(2)

    result = model.run_h_cycles(
        embedded,
        z_slow,
        z_fast,
        cos_sin,
        collect_intermediates=True,
    )

    assert len(result.all_logits) == 3
    assert len(result.all_z_slow) == 3
    assert not result.all_logits[0].requires_grad
    assert not result.all_z_slow[-1].requires_grad
    assert result.logits.requires_grad


def test_core_updates_fast_state_then_slow_state_with_rope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _config(fast_cycles=2, slow_cycles=1).make().to(torch.float64)
    # One offset per position; the unit axes are expanded to [2, 8, 12] below.
    offsets = torch.arange(8, dtype=torch.float64).reshape(1, 8, 1).expand(2, 8, 12)
    input_emb = offsets + 2
    z_slow = offsets + 3
    z_fast = offsets + 5
    cos_sin = (torch.ones(8, 3), torch.full((8, 3), 2.0))
    eager_inputs: list[Tensor] = []
    compiled_inputs: list[Tensor] = []

    def eager(value: Tensor, **kwargs: object) -> Tensor:
        eager_inputs.append(value)
        assert kwargs.get("cos_sin") is cos_sin
        return value + 1

    def compiled(value: Tensor, **kwargs: object) -> Tensor:
        compiled_inputs.append(value)
        assert kwargs.get("cos_sin") is cos_sin
        return value + 2

    monkeypatch.setattr(model.reasoning, "forward", eager)
    model._reasoning_compiled = compiled
    with torch.no_grad():
        assert model.q_head.bias is not None
        model.q_head.weight.copy_(torch.arange(24).reshape(2, 12))
        model.q_head.bias.copy_(torch.tensor([1.0, -2.0]))

    logits, q_halt, next_slow, next_fast = model.core(
        input_emb,
        z_slow,
        z_fast,
        cos_sin,
    )
    assert [value[0, 0, 0].item() for value in eager_inputs] == [10, 16, 20]
    assert compiled_inputs == []
    expected_offset = offsets * 6
    assert torch.equal(next_fast, torch.full_like(z_fast, 17) + offsets * 5)
    assert torch.equal(next_slow, torch.full_like(z_slow, 21) + expected_offset)
    assert torch.equal(logits, model.head(next_slow))
    expected_halt = model.q_head(next_slow[:, 0]).to(torch.float32)[..., 0]
    assert q_halt.dtype is torch.float32
    assert torch.equal(q_halt, expected_halt)

    model.config.compile = True
    eager_inputs.clear()
    _, compiled_halt, compiled_slow, compiled_fast = model.core(
        input_emb,
        z_slow,
        z_fast,
        cos_sin,
    )
    assert eager_inputs == []
    assert [value[0, 0, 0].item() for value in compiled_inputs] == [10, 17, 22]
    assert torch.equal(compiled_fast, torch.full_like(z_fast, 19) + offsets * 5)
    assert torch.equal(compiled_slow, torch.full_like(z_slow, 24) + offsets * 6)
    assert torch.equal(
        compiled_halt,
        model.q_head(compiled_slow[:, 0]).to(torch.float32)[..., 0],
    )


def test_run_h_cycles_detaches_prior_cycles_and_rope_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _config(slow_cycles=3).make()
    input_emb = torch.ones(2, 8, 12, requires_grad=True)
    z_slow = torch.ones(2, 8, 12, requires_grad=True)
    z_fast = torch.ones(2, 8, 12, requires_grad=True)
    cos_sin = (
        torch.ones(8, 3, requires_grad=True),
        torch.full((8, 3), 2.0, requires_grad=True),
    )
    calls: list[tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor] | None]] = []

    def core(
        embedded: Tensor,
        slow: Tensor,
        fast: Tensor,
        rotary: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        calls.append((embedded, slow, fast, rotary))
        next_slow = slow + embedded
        next_fast = fast + embedded
        return next_slow, next_slow[:, 0, 0], next_slow, next_fast

    monkeypatch.setattr(model, "core", core)
    result = model.run_h_cycles(input_emb, z_slow, z_fast, cos_sin)

    assert len(calls) == 3
    assert all(not value.requires_grad for value in calls[0][:3])
    assert all(not value.requires_grad for value in calls[1][:3])
    assert calls[2][0] is input_emb
    assert calls[2][3] is cos_sin
    for rotary in (calls[0][3], calls[1][3]):
        assert rotary is not None
        assert not rotary[0].requires_grad
        assert not rotary[1].requires_grad
        assert torch.equal(rotary[0], cos_sin[0])
        assert torch.equal(rotary[1], cos_sin[1])
    assert result.logits.requires_grad
    assert result.all_logits == ()
    assert result.all_z_slow == ()

    calls.clear()
    collected = model.run_h_cycles(
        input_emb,
        z_slow,
        z_fast,
        cos_sin,
        collect_intermediates=True,
    )
    assert len(collected.all_logits) == 3
    assert len(collected.all_z_slow) == 3
    assert all(isinstance(value, Tensor) for value in collected.all_z_slow)
    assert all(value.shape == (2, 8, 12) for value in collected.all_z_slow)

    calls.clear()
    model.run_h_cycles(input_emb, z_slow, z_fast, None)
    assert len(calls) == 3
    assert all(rotary is None for *_, rotary in calls)


def test_position_and_feedback_embeddings_are_applied_to_grid_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _config(
        puzzle_emb_ndim=5,
        dtype=torch.float64,
        feedback_init_std=0.5,
    ).make()
    model.eval()
    batch_size = 32
    inputs = torch.arange(batch_size * 6).reshape(batch_size, 6) % 5
    identifiers = torch.arange(batch_size) % 2
    feedback = (inputs + 1) % 5
    with torch.no_grad():
        model.embed_tokens.weight.copy_(torch.arange(60).reshape(5, 12) / 10)
        assert model.puzzle_emb is not None
        model.puzzle_emb.weights.copy_(torch.arange(10).reshape(2, 5) + 1)
        model.embed_pos_row = torch.nn.Parameter(
            torch.arange(24, dtype=torch.float64).reshape(2, 12) / 100,
        )
        model.embed_pos_col = torch.nn.Parameter(
            torch.arange(36, dtype=torch.float64).reshape(3, 12) / 100,
        )
        model.embed_pos_box = torch.nn.Parameter(
            torch.arange(36, dtype=torch.float64).reshape(3, 12) / 100,
        )
        model.embed_feedback = torch.nn.Parameter(
            torch.arange(60, dtype=torch.float64).reshape(5, 12) / 10,
        )

    token_embeddings = model.embed_scale * model.embed_tokens(inputs)
    puzzle_vectors = model.puzzle_emb(identifiers)
    puzzle_prefix = torch.nn.functional.pad(puzzle_vectors, (0, 19)).reshape(
        batch_size,
        2,
        12,
    )
    puzzle_prefix = model.embed_scale * puzzle_prefix.to(token_embeddings.dtype)
    assert model.embed_pos_row is not None
    assert model.embed_pos_col is not None
    assert model.embed_pos_box is not None
    position_table = (
        model.embed_pos_row[model.row_index]
        + model.embed_pos_col[model.col_index]
        + model.embed_pos_box[model.box_index]
    )
    position_embeddings = model.embed_scale * position_table.to(
        token_embeddings.dtype,
    )
    feedback_embeddings = model.embed_scale * model.embed_feedback[feedback].to(
        token_embeddings.dtype,
    )
    expected_grid = token_embeddings + position_embeddings
    expected_grid = expected_grid + feedback_embeddings
    expected = torch.cat([puzzle_prefix, expected_grid], dim=1)
    arange = torch.arange
    arange_calls: list[tuple[tuple[int, ...], dict[str, object]]] = []

    def capture_arange(*args: int, **kwargs: object) -> Tensor:
        arange_calls.append((args, kwargs))
        assert args == (8,)
        assert kwargs == {"device": token_embeddings.device}
        return arange(*args, device=token_embeddings.device)

    monkeypatch.setattr(torch, "arange", capture_arange)
    model.set_feedback(feedback)
    embedded, cos_sin = model._embed_and_prepare(inputs, identifiers)

    assert embedded.dtype is torch.float32
    assert torch.equal(embedded, expected)
    assert embedded.shape == (batch_size, 8, 12)
    assert cos_sin is not None
    assert cos_sin[0].shape[0] == 8
    assert arange_calls == [((8,), {"device": token_embeddings.device})]
    assert model._feedback_ids is None


def test_exact_width_puzzle_embedding_is_not_padded() -> None:
    model = _config(puzzle_emb_ndim=24, pos2d_grid_shape=None).make()
    model.eval()
    assert model.puzzle_emb is not None
    identifiers = torch.tensor([0, 1])
    with torch.no_grad():
        model.puzzle_emb.weights.copy_(torch.arange(48).reshape(2, 24))
    embedded, _ = model._embed_and_prepare(
        torch.zeros(2, 6, dtype=torch.long),
        identifiers,
    )
    # Two puzzles, each 24 = 2 prefix tokens x 12 hidden.
    expected_prefix = model.embed_scale * model.puzzle_emb.weights[identifiers].reshape(
        2,
        2,
        12,
    )
    assert torch.equal(embedded[:, :2], expected_prefix)
    assert embedded.shape == (2, 8, 12)


def test_non_square_grid_preserves_padded_puzzle_prefix() -> None:
    model = _config(puzzle_emb_ndim=5).make()
    model.eval()
    assert model.puzzle_emb is not None
    with torch.no_grad():
        model.puzzle_emb.weights.copy_(torch.arange(10).reshape(2, 5) + 1)
    inputs = torch.zeros(32, 6, dtype=torch.long)
    identifiers = torch.arange(32) % 2

    embedded, _ = model._embed_and_prepare(inputs, identifiers)

    prefix = embedded[:, :2]
    assert prefix.shape == (32, 2, 12)
    assert torch.equal(
        prefix[:, 0, :5],
        model.embed_scale * (torch.arange(10).reshape(2, 5) + 1)[identifiers],
    )
    assert torch.equal(prefix[:, 0, 5:], torch.zeros_like(prefix[:, 0, 5:]))
    assert torch.equal(prefix[:, 1], torch.zeros_like(prefix[:, 1]))
    assert embedded[:, 2:].shape == (32, 6, 12)


def test_model_parameters_and_positional_indices_match_grid_geometry() -> None:
    model = _config().make()

    assert model.embed_tokens.weight.shape == (5, 12)
    assert model.head.weight.shape == (5, 12)
    assert model.q_head.weight.shape == (2, 12)
    assert model.puzzle_emb is not None
    assert model.puzzle_emb.weights.shape == (2, 12)
    assert model.embed_pos_row is not None
    assert model.embed_pos_row.shape == (2, 12)
    assert model.embed_pos_col is not None
    assert model.embed_pos_col.shape == (3, 12)
    assert model.embed_pos_box is not None
    assert model.embed_pos_box.shape == (3, 12)
    assert model.embed_feedback.shape == (5, 12)
    assert model.row_index.tolist() == [0, 0, 0, 1, 1, 1]
    assert model.col_index.tolist() == [0, 1, 2, 0, 1, 2]
    assert model.box_index.tolist() == [0, 1, 2, 0, 1, 2]
    assert model.slow_init.shape == (1, 12)
    assert model.fast_init.shape == (1, 12)
    assert model.rope.channels_head == (4,)
    z_slow, z_fast = model.init_z(2)
    assert z_slow.shape == (2, 8, 12)
    assert z_fast.shape == (2, 8, 12)
    assert torch.equal(z_fast[0], z_fast[1])
    state = model.state_dict()
    assert "row_index" not in state
    assert "col_index" not in state
    assert "box_index" not in state
    assert "slow_init" in state
    assert "fast_init" in state
    assert "_dummy" in state


def test_constructor_pins_head_initialization_and_accepts_one_token_vocab() -> None:
    model = _config(vocab_size=1, num_puzzle_identifiers=0).make()

    assert model.embed_scale == 12**0.5
    assert model.embed_tokens.weight.shape == (1, 12)
    assert model.head.weight.shape == (1, 12)
    assert model.head.bias is None
    assert model.q_head.weight.shape == (2, 12)
    assert model.q_head.bias is not None
    assert torch.equal(model.q_head.weight, torch.zeros_like(model.q_head.weight))
    assert torch.equal(model.q_head.bias, torch.full_like(model.q_head.bias, -5.0))
    assert model.puzzle_emb is None
    assert model.embed_pos_row is not None
    assert model.embed_pos_col is not None
    assert model.embed_pos_box is not None


def test_position_indices_register_as_nonpersistent_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool | None]] = []
    register_buffer = torch.nn.Module.register_buffer

    def capture_register_buffer(
        module: torch.nn.Module,
        name: str,
        tensor: Tensor | None,
        persistent: bool = True,
    ) -> None:
        if name in {"row_index", "col_index", "box_index"}:
            calls.append((name, persistent))
        register_buffer(module, name, tensor, persistent=persistent)

    monkeypatch.setattr(torch.nn.Module, "register_buffer", capture_register_buffer)
    _config().make()
    assert calls == [
        ("row_index", False),
        ("col_index", False),
        ("box_index", False),
    ]


def test_constructor_pins_init_and_device_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_calls: list[tuple[tuple[int, ...], float]] = []

    def capture_init(tensor: Tensor, *, std: float) -> Tensor:
        init_calls.append((tuple(tensor.shape), std))
        return tensor

    monkeypatch.setattr(trm, "_corrected", capture_init)
    model = _config().make()

    assert init_calls == [((2, 12), 0.0), ((1, 12), 1.0), ((1, 12), 1.0)]
    assert model.slow_init.shape == (1, 12)
    assert model.fast_init.shape == (1, 12)
    assert model._dummy.shape == (0,)
    assert model.device == torch.device("cpu")
    assert model._reasoning_compiled is None


def test_constructor_binds_compiled_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def compile_fake(function: object, **kwargs: object) -> object:
        calls.append((function, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", compile_fake)
    model = _config(compile=True).make()

    assert calls == [(model.reasoning.forward, {"fullgraph": True})]
    assert model._reasoning_compiled == model.reasoning.forward


def test_constructor_pins_embedding_init_and_repeats_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Tensor, dict[str, object]]] = []

    def capture(tensor: Tensor, **kwargs: object) -> None:
        calls.append((tensor, kwargs))

    monkeypatch.setattr(trm, "truncated_normal", capture)
    model = _config(
        num_puzzle_identifiers=1,
        puzzle_emb_ndim=6,
        dtype=torch.float64,
        num_layers=2,
    ).make()

    assert tuple(calls[0][0].shape) == (5, 12)
    assert calls[0][1] == {
        "std": 1.0 / 12**0.5,
        "depth_index": (),
        "variance_correction": True,
    }
    assert model.puzzle_emb is not None
    assert model.puzzle_emb.weights.shape == (1, 6)
    assert model.puzzle_emb.cast_to is torch.float64
    assert len(model.reasoning) == 2
    assert model.reasoning[0] is not model.reasoning[1]
    first_params = list(model.reasoning[0].parameters())
    second_params = list(model.reasoning[1].parameters())
    assert len(first_params) == len(second_params)
    assert all(
        left is not right
        for left, right in zip(first_params, second_params, strict=True)
    )


def test_parameter_count_logs_preserve_each_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_puzzles = _config().make()
    without_puzzles = _config(num_puzzle_identifiers=0).make()
    calls: list[tuple[object, tuple[object, ...]]] = []

    def log_info(message: object, *args: object) -> None:
        calls.append((message, args))

    monkeypatch.setattr(trm.logger, "info", log_info)
    with_puzzles._log_parameter_counts()
    without_puzzles._log_parameter_counts()

    body_with_puzzles = sum(param.numel() for param in with_puzzles.parameters())
    assert with_puzzles.puzzle_emb is not None
    puzzle_params = with_puzzles.puzzle_emb.weights.numel()
    body_without_puzzles = sum(param.numel() for param in without_puzzles.parameters())
    assert calls == [
        (
            "model parameters: %.2fM total (%.2fM body + %.2fM puzzle-emb)",
            (
                (body_with_puzzles + puzzle_params) / 1e6,
                body_with_puzzles / 1e6,
                puzzle_params / 1e6,
            ),
        ),
        (
            "model parameters: %.2fM (body)",
            (body_without_puzzles / 1e6,),
        ),
    ]


def test_constructor_uses_corrected_head_initializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, ...]] = []

    def capture(weight: Tensor, **kwargs: object) -> None:
        del kwargs
        calls.append(tuple(weight.shape))

    monkeypatch.setattr(trm, "trm_truncated_normal_corrected", capture)
    _config().make()

    assert calls[0] == (5, 12)


def test_model_without_optional_embeddings_uses_plain_grid_sequence() -> None:
    model = _config(num_puzzle_identifiers=0, pos2d_grid_shape=None).make()
    inputs = torch.tensor([[1, 2, 3, 1, 4, 2], [3, 1, 2, 4, 1, 3]])
    z_slow, z_fast = model.init_z(2)

    output = model(inputs, z_slow, z_fast)

    assert model.puzzle_emb is None
    assert model.embed_pos_row is None
    assert model.embed_pos_col is None
    assert model.embed_pos_box is None
    logits = output["logits"]
    assert isinstance(logits, Tensor)
    assert logits.shape == (2, 6, 5)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
