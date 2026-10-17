"""Tests for native released-checkpoint CHARM evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override
from unittest.mock import Mock

import hashlib

from pyarrow import parquet

import numpy as np
import pyarrow as pa
import pytest
import torch

from priml.baselines.arcagi.metric import ParquetPassK
from priml.baselines.arcagi.parquet_data import (
    ArcBatch,
    ArcParquetData,
    PairSource,
)
from priml.baselines.arcagi.scripts.charm_eval import evaluate
from priml.baselines.arcagi.scripts.charm_eval.evaluate import (
    _halt_column,
    _predict_batch,
    _prediction_tokens,
    evaluate_checkpoints,
    select_checkpoint_steps,
)
from priml.model.attention.attention import Attention
from priml.model.conditioning import CategorySlotFiLM
from priml.model.custom_types import ForwardOutput
from priml.model.grid_embedding import GridEmbedding
from priml.model.grid_solver import SingleLatentGridSolver
from priml.model.prefix import ComposedSparsePrefix
from priml.model.recurrence import SingleLatentRecurrence
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from pathlib import Path


def test_checkpoint_windows_match_the_released_scripts() -> None:
    assert select_checkpoint_steps("arc1", checkpoint_step=None) == list(
        range(1_510_000, 1_590_001, 10_000),
    )
    assert select_checkpoint_steps("arc2", checkpoint_step=None) == list(
        range(2_290_000, 2_380_001, 10_000),
    )
    assert select_checkpoint_steps("arc1", checkpoint_step=1_530_000) == [
        1_530_000,
    ]
    with pytest.raises(ValueError, match="outside the released window"):
        select_checkpoint_steps("arc1", checkpoint_step=1_600_000)


def test_predict_batch_carries_state_and_passes_each_prefix_field() -> None:
    batch = _prediction_batch()
    model = _RecordingModel()

    packed = _predict_batch(
        cast(SingleLatentGridSolver, model),
        batch,
        device=torch.device("cpu"),
        max_steps=2,
    )

    assert len(model.inputs) == 2
    assert all(torch.equal(tokens, batch["media"]) for tokens in model.inputs)
    assert [latent[0, 0, 0].item() for latent in model.slow_inputs] == [0.0, 1.0]
    assert [latent[0, 0, 0].item() for latent in model.fast_inputs] == [0.0, 2.0]
    assert set(model.prefix_values) == {
        "category_indices",
        "augmentation_indices",
        "slot_permutation",
        "puzzle_identifiers",
    }
    expected_prefix = {
        "category_indices": batch["puzzle_idxs"],
        "augmentation_indices": batch["dih_idxs"],
        "slot_permutation": batch["colorperm_slots"],
        "puzzle_identifiers": batch["puzzle_identifiers"],
    }
    for key, expected in expected_prefix.items():
        assert torch.equal(model.prefix_values[key], expected)
    assert torch.equal(packed[:, 0], torch.tensor([0.2, 0.2]))
    assert torch.equal(packed[:, 1:], torch.full((2, 4), 2.0))
    assert packed.dtype is torch.float32


def test_predict_batch_moves_inputs_to_runtime_device() -> None:
    model = _RecordingModel()

    packed = _predict_batch(
        cast(SingleLatentGridSolver, model),
        _prediction_batch(),
        device=torch.device("meta"),
        max_steps=1,
    )

    assert model.inputs[0].device.type == "meta"
    assert all(value.device.type == "meta" for value in model.prefix_values.values())
    assert packed.device.type == "meta"
    assert packed.dtype is torch.float32


def test_predict_batch_uses_the_shared_label_free_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass all source prefix ids through the common fixed-depth rollout."""
    batch = _prediction_batch()
    model = cast(SingleLatentGridSolver, _RecordingModel())
    logits = torch.zeros(2, 4, 12)
    logits[..., 3] = 1.0
    halt = torch.full((2,), -0.25)
    shared_rollout = Mock(return_value=(logits, halt))
    monkeypatch.setattr(evaluate, "rollout", shared_rollout)

    packed = _predict_batch(
        model,
        batch,
        device=torch.device("cpu"),
        max_steps=4,
    )

    shared_rollout.assert_called_once_with(
        model,
        media=batch["media"],
        max_steps=4,
        carry=None,
        prefix_kwargs={
            "category_indices": batch["puzzle_idxs"],
            "augmentation_indices": batch["dih_idxs"],
            "slot_permutation": batch["colorperm_slots"],
            "puzzle_identifiers": batch["puzzle_identifiers"],
        },
    )
    assert torch.equal(packed[:, 0], halt.reshape(-1))
    assert torch.equal(packed[:, 1:], torch.full((2, 4), 3.0))


def test_predict_batch_rejects_zero_act_steps() -> None:
    with pytest.raises(
        ValueError,
        match="CHARM evaluation requires at least one ACT step",
    ) as error:
        _predict_batch(
            cast(SingleLatentGridSolver, _RecordingModel()),
            _prediction_batch(),
            device=torch.device("cpu"),
            max_steps=0,
        )
    assert str(error.value) == "CHARM evaluation requires at least one ACT step."


def test_prediction_fields_are_float32() -> None:
    halt = torch.tensor([0.123456789], dtype=torch.float64)
    logits = torch.tensor([[[0.0, 3.0, 1.0]]], dtype=torch.float64)

    halt_column = _halt_column(halt)
    predictions = _prediction_tokens(logits)

    assert halt_column.dtype is torch.float32
    assert torch.equal(halt_column, torch.tensor([[0.12345679]]))
    assert predictions.dtype is torch.float32
    assert torch.equal(predictions, torch.tensor([[1.0]]))


def test_tiny_checkpoint_replays_act16_and_accumulates_metric_events(
    tmp_path: Path,
) -> None:
    parquet = _write_tiny_source(tmp_path / "tiny.parquet")
    config = _tiny_model_config()
    seed_model = config.make()
    live = _source_state(seed_model, parameters_only=False, halt_bias=-2.0)
    ema_states = [
        _source_state(seed_model, parameters_only=True, halt_bias=bias)
        for bias in (-1.0, 1.0)
    ]
    checkpoint_paths = [
        tmp_path / "step_1510000.pt",
        tmp_path / "step_1520000.pt",
    ]
    for step, path, ema_state in zip(
        (1_510_000, 1_520_000),
        checkpoint_paths,
        ema_states,
        strict=True,
    ):
        torch.save(
            {
                "model_state_dict": live,
                "ema_state": ema_state,
                "step": step,
            },
            path,
        )

    data = ArcParquetData.Config(
        train_sources=[PairSource.Config(path=parquet, pair_selection="train")],
        eval_source=parquet,
        canvas_size=2,
        eval_batch_size=1,
        num_workers=0,
        p_translate=0.0,
    ).make()
    aggregate_config = ParquetPassK.Config(
        eval_source=parquet,
        canvas_size=2,
        pass_ks=(1, 2),
        recent_window_size=10,
    )
    aggregate = aggregate_config.make()
    current_config = aggregate_config.copy_tree()
    current_config.aggregated_voting = False
    current = current_config.make()
    model = config.make()
    counter = _ForwardCounter()
    model.register_forward_hook(counter)
    output_dir = tmp_path / "output"

    results = evaluate_checkpoints(
        model,
        data,
        aggregate,
        current,
        checkpoint_paths,
        device=torch.device("cpu"),
        output_dir=output_dir,
        max_steps=16,
        submission_k=2,
        provenance={"benchmark": "tiny"},
    )

    assert counter.calls == 32
    halt_bias = model.halt_head.bias
    assert halt_bias is not None
    assert torch.equal(halt_bias, torch.tensor([1.0]))
    assert [result["step"] for result in results] == [1_510_000, 1_520_000]
    assert [result["act_steps"] for result in results] == [16, 16]
    assert results[0]["current_scores"] == {"pass@1": 0.0, "pass@2": 0.0}
    assert results[1]["current_scores"] == {"pass@1": 0.0, "pass@2": 0.0}
    assert results[1]["accumulated_scores"] == {
        "pass@1": 0.0,
        "pass@2": 0.0,
    }
    assert set(aggregate.state_dict()["events"]) == {0, 1}
    assert set(current.state_dict()["events"]) == {1}
    assert (output_dir / "submissions" / "step_1510000.json").is_file()
    assert (output_dir / "submissions" / "step_1520000.json").is_file()
    assert (output_dir / "submissions" / "current_step_1510000.json").is_file()
    assert (output_dir / "submissions" / "current_step_1520000.json").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "provenance.json").is_file()
    assert (output_dir / "submission.json").read_text(encoding="utf-8") == (
        '{\n  "tiny": [\n    {\n      "attempt_1": [],\n'
        '      "attempt_2": []\n    }\n  ]\n}\n'
    )
    assert [result["checkpoint_sha256"] for result in results] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoint_paths
    ]


@pytest.mark.parametrize("include_ema", [False, True])
def test_released_evaluation_rejects_missing_or_none_ema(
    tmp_path: Path,
    include_ema: bool,
) -> None:
    model = _tiny_model_config().make()
    live = _source_state(model, parameters_only=False, halt_bias=0.75)
    checkpoint = tmp_path / "step_1510000.pt"
    payload: dict[str, object] = {
        "model_state_dict": live,
        "step": 1_510_000,
    }
    if include_ema:
        payload["ema_state"] = None
    torch.save(payload, checkpoint)

    parquet = _write_tiny_source(tmp_path / "tiny.parquet")
    data = ArcParquetData.Config(
        train_sources=[PairSource.Config(path=parquet, pair_selection="train")],
        eval_source=parquet,
        canvas_size=2,
        eval_batch_size=2,
        num_workers=0,
        p_translate=0.0,
    ).make()
    aggregate_config = ParquetPassK.Config(
        eval_source=parquet,
        canvas_size=2,
        pass_ks=(1,),
    )
    aggregate = aggregate_config.make()
    current_config = aggregate_config.copy_tree()
    current_config.aggregated_voting = False
    current = current_config.make()

    with pytest.raises(TypeError, match="ema_state must be a mapping"):
        evaluate_checkpoints(
            model,
            dataset=data,
            accumulated_metric=aggregate,
            current_metric=current,
            checkpoints=[checkpoint],
            device=torch.device("cpu"),
            output_dir=tmp_path / "output",
            max_steps=1,
            submission_k=1,
            provenance={},
        )


def _prediction_batch() -> ArcBatch:
    return {
        "media": torch.arange(8, dtype=torch.int32).reshape(2, 4),
        "puzzle_identifiers": torch.tensor([11, 12]),
        "puzzle_idxs": torch.tensor([2, 3]),
        "dih_idxs": torch.tensor([4, 5]),
        "colorperm_slots": torch.arange(18).reshape(2, 9),
        "spatial_tags": torch.ones((2, 3), dtype=torch.int64),
        "valid_count": 2,
        "canonical_puzzle_names": ["a", "b"],
        "canonical_inputs": [
            np.array([[1]], dtype=np.uint8),
            np.array([[2]], dtype=np.uint8),
        ],
        "canonical_outputs": [
            np.array([[1]], dtype=np.uint8),
            np.array([[2]], dtype=np.uint8),
        ],
        "input_metadata": [{}, {}],
    }


class _RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []
        self.slow_inputs: list[torch.Tensor] = []
        self.fast_inputs: list[torch.Tensor] = []
        self.prefix_values: dict[str, torch.Tensor] = {}

    def init_latents(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((batch_size, 3, 2)),
            torch.zeros((batch_size, 3, 2)),
        )

    @override
    def forward(
        self,
        tokens: torch.Tensor,
        z_slow: torch.Tensor | None = None,
        z_fast: torch.Tensor | None = None,
        *,
        collect_intermediates: bool = False,
        **prefix_kwargs: object,
    ) -> ForwardOutput:
        del collect_intermediates
        assert z_slow is not None
        assert z_fast is not None
        self.inputs.append(tokens)
        self.slow_inputs.append(z_slow)
        self.fast_inputs.append(z_fast)
        for key, value in prefix_kwargs.items():
            assert isinstance(value, torch.Tensor)
            self.prefix_values[key] = value
        logits = torch.zeros(
            (tokens.shape[0], 4, 12),
            dtype=torch.float64,
            device=tokens.device,
        )
        logits[..., len(self.inputs)] = 1.0
        halt = torch.full(
            (tokens.shape[0],),
            0.1 * len(self.inputs),
            device=tokens.device,
            dtype=torch.float64,
        )
        return ForwardOutput(logits, halt, z_slow + 1, z_fast + 2)


def _write_tiny_source(path: Path) -> Path:
    rows: list[dict[str, object]] = [
        {
            "format_version": 2,
            "puzzle_id": "tiny",
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[1]], "output": [[2]]}],
            "offline_augs": [""],
        },
    ]
    table = pa.Table.from_pylist(rows)
    parquet.write_table(table, path)
    return path


def _tiny_model_config() -> SingleLatentGridSolver.Config:
    config = SingleLatentGridSolver.Config()
    config.channels_in = 8
    config.num_layers = 1
    config.vocab_size = 12
    cast(GridEmbedding.Config, config.embedding).grid_shape = (2, 2)
    block = cast(TransformerBlock.Config, config.block)
    attention = cast(Attention.Config, block.attn)
    ffn = cast(SwiGLU.Config, block.ffn)
    attention.num_heads = 2
    attention.channels_head = 4
    ffn.channels_hidden = 6
    prefix = cast(ComposedSparsePrefix.Config, config.prefix)
    prefix.channels_out = 8
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
    prefix.residual_projection.channels_out = 8
    recurrence = cast(SingleLatentRecurrence.Config, config.recurrence)
    recurrence.slow_cycles = 1
    recurrence.fast_cycles = 1
    recurrence.inner_grad_loops = 1
    config.compile_core = None
    return config


def _source_state(
    model: SingleLatentGridSolver,
    *,
    parameters_only: bool,
    halt_bias: float,
) -> dict[str, torch.Tensor]:
    state = dict(model.named_parameters()) if parameters_only else model.state_dict()
    source: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        source_name = _source_key(name)
        source_value = torch.zeros_like(value)
        if name == "halt_head.bias":
            source_value.fill_(halt_bias)
        source[source_name] = source_value
    return source


def _source_key(native_key: str) -> str:
    if native_key == "embedding.embed_tokens.weight":
        key = "embed_tokens.embedding_weight"
    elif native_key == "head.weight":
        key = "lm_head.weight"
    elif native_key.startswith("halt_head."):
        key = "q_head." + native_key.removeprefix("halt_head.")
    elif native_key == "slow_init":
        key = "init_hidden"
    elif native_key.startswith("prefix.composition."):
        key = "puzzle_emb." + native_key.removeprefix("prefix.composition.")
    elif native_key == "prefix.residual.weights":
        key = "puzzle_emb.instance_residual_embed.weights"
    elif native_key.startswith("prefix.residual_projection."):
        key = "puzzle_emb.instance_residual_up_proj." + native_key.removeprefix(
            "prefix.residual_projection.",
        )
    elif native_key.startswith("prefix.residual_gate.linear."):
        key = "puzzle_emb.interaction_gate." + native_key.removeprefix(
            "prefix.residual_gate.linear.",
        )
    elif native_key.startswith("reasoning."):
        key = "layers." + native_key.removeprefix("reasoning.")
        key = key.replace(".attn.", ".self_attn.")
        key = key.replace(".ffn.", ".mlp.")
        key = key.replace(".self_attn.proj_qkv.", ".self_attn.qkv_proj.")
        key = key.replace(".self_attn.proj_out.", ".self_attn.o_proj.")
        key = key.replace(".mlp.up_proj.", ".mlp.gate_up_proj.")
        key = key.replace(".mlp.middle.conv.", ".mlp.dwconv.")
    else:
        raise AssertionError(native_key)
    return "_orig_mod.model.inner." + key


class _ForwardCounter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        module: torch.nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        del module, inputs, output
        self.calls += 1


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
