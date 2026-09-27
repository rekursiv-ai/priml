"""Focused behavioral checks for HPS atomic feedback state."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, cast

import copy
import traceback

from torch._inductor import config as inductor_config
from torch.nn.functional import binary_cross_entropy_with_logits

import pytest
import torch

from priml.baselines.arcagi1.model import ConvSwiGLU
from priml.baselines.arcagi1.train_step import (
    HPSFeedbackTrainStep,
    _precision_casts,
)
from priml.baselines.sudoku.embedding import GridEmbedding, PredictionFeedback
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.prefix import PrefixStack, SparsePuzzleEmbedding
from priml.model.attention.self_attention import SelfAttention
from priml.model.transformer.block import TransformerBlock
from priml.optimizers.lr import learning_rate
from priml.runtime import MultiProcess
from priml.train.checkpointer import SyncLocalStateDictStorer
from priml.train.parallelism import DataParallel, NoParallel


if TYPE_CHECKING:
    from pathlib import Path

    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter


def _tiny_step(*, data_parallel: bool = False) -> HPSFeedbackTrainStep:
    config = HPSFeedbackTrainStep.Config()
    config.batch_size = 2
    config.max_act_steps = 2
    config.total_train_steps = 3
    config.warmup_steps = 4
    config.use_ema = False
    config.ignore_label_id = -100
    config.parallelism = (
        DataParallel.Config() if data_parallel else NoParallel.Config(device="cpu")
    )
    model = config.model
    model.channels_in = 16
    model.num_layers = 1
    model.vocab_size = 12
    model.embedding = GridEmbedding.Config(
        grid_shape=(4,),
        channels=[PredictionFeedback.Config()],
    )
    model.recurrence = DeepRecurrence.Config(slow_cycles=1, fast_cycles=2)
    model.prefix = PrefixStack.Config(
        parts=[
            SparsePuzzleEmbedding.Config(
                num_puzzles=20,
                num_tokens=2,
                batch_size=2,
            ),
        ],
    )
    model.block = TransformerBlock.Config(
        channels_in=16,
        channels_out=16,
        prenorm=False,
        attn=SelfAttention.Config(
            channels_in=16,
            channels_out=16,
            num_heads=2,
            channels_head=8,
        ),
        ffn=ConvSwiGLU.Config(
            channels_in=16,
            channels_out=16,
            channels_hidden=32,
            kernel_size=2,
        ),
    )
    return config.copy_tree().finalize().make()


def test_act_requires_at_least_two_steps() -> None:
    config = HPSFeedbackTrainStep.Config()
    config.max_act_steps = 1
    with pytest.raises(ValueError, match="max_act_steps"):
        config.finalize()


def test_nan_feedback_corruption_rate_is_rejected() -> None:
    config = HPSFeedbackTrainStep.Config()
    config.feedback_corruption_rate = float("nan")
    with pytest.raises(ValueError, match="feedback_corruption_rate"):
        config.finalize()


def test_atomic_slots_keep_their_ids_and_feedback_until_halted() -> None:
    step = _tiny_step()
    media = torch.tensor([[2, 3, 0, 0], [4, 5, 0, 0]])
    labels = torch.tensor([[2, 3, -100, -100], [4, 5, -100, -100]])
    step._refill(media, labels=labels, ids=torch.tensor([7, 8]), valid_count=2)
    step._pool_halted[0] = False
    step._pool_feedback[0] = torch.tensor([9, 9, 9, 9])

    step._refill(media + 2, labels=labels, ids=torch.tensor([17, 18]), valid_count=2)

    assert step._pool_ids.tolist() == [7, 18]
    assert step._pool_feedback[0].tolist() == [9, 9, 9, 9]
    assert step._pool_feedback[1].tolist() == [6, 7, 2, 2]


def test_partial_batch_ignores_empty_pool_slots_in_halt_loss() -> None:
    step = _tiny_step()
    media = torch.tensor([[2, 3, 0, 0], [4, 5, 0, 0]])
    labels = torch.tensor([[2, 3, -100, -100], [4, 5, -100, -100]])
    step._refill(media, labels=labels, ids=torch.tensor([7, 8]), valid_count=1)
    step._pool_halted[0] = False
    step._refill(media + 1, labels=labels, ids=torch.tensor([9, 10]), valid_count=0)

    logits = torch.zeros(2, 4, 12)
    halt = torch.tensor([0.0, 3.0], requires_grad=True)
    _, _, halt_loss, _ = step._loss(logits, halt=halt, labels=step._pool_labels)
    expected = (
        binary_cross_entropy_with_logits(
            halt[0],
            torch.zeros_like(halt[0]),
        )
        / step.config.batch_size
    )
    torch.testing.assert_close(halt_loss, expected)
    halt_loss.backward()
    assert halt.grad is not None
    assert halt.grad[0] != 0
    assert halt.grad[1] == 0


def test_feedback_rng_is_checkpointed_and_color_outputs_are_not_clamped() -> None:
    step = _tiny_step()
    step.config.feedback_corruption_rate = 0.5
    prediction = torch.tensor([[11, 10, 9, 8]])
    torch.manual_seed(1)
    before = torch.random.get_rng_state()
    checkpoint = step.state_dict()
    expected = step._corrupt(prediction)
    assert torch.equal(torch.random.get_rng_state(), before)
    step._feedback_rng.manual_seed(123)
    step.load_state_dict(checkpoint)
    assert torch.equal(step._corrupt(prediction), expected)


def test_checkpoint_restores_inflight_act_pool() -> None:
    step = _tiny_step()
    media = torch.tensor([[2, 3, 0, 0], [4, 5, 0, 0]])
    labels = torch.tensor([[2, 3, -100, -100], [4, 5, -100, -100]])
    step.train_step(
        media=media,
        label=labels,
        puzzle_identifiers=torch.tensor([7, 8]),
        valid_count=2,
    )
    step._pool_halted.fill_(True)
    step._refill(media, labels=labels, ids=torch.tensor([7, 8]), valid_count=2)
    step._pool_feedback.fill_(9)
    step._pool_z_slow.fill_(1)
    step._pool_z_fast.fill_(2)
    step._pool_halted[0] = False
    step._pool_steps[:] = torch.tensor([1, 2])
    checkpoint = copy.deepcopy(step.state_dict())

    resumed = _tiny_step()
    resumed.load_state_dict(checkpoint)
    for name in (
        "_pool_inputs",
        "_pool_labels",
        "_pool_ids",
        "_pool_feedback",
        "_pool_z_slow",
        "_pool_z_fast",
        "_pool_halted",
        "_pool_steps",
    ):
        torch.testing.assert_close(getattr(resumed, name), getattr(step, name))

    next_batch = {
        "media": media + 1,
        "label": labels,
        "puzzle_identifiers": torch.tensor([9, 10]),
        "valid_count": 0,
    }
    expected = step.train_step(**next_batch)
    actual = resumed.train_step(**next_batch)
    torch.testing.assert_close(actual["loss"], expected["loss"])
    torch.testing.assert_close(actual["model"], expected["model"])
    for name, weight in step.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], weight)


def test_checkpoint_rejects_changed_dp_size() -> None:
    step = _tiny_step()
    checkpoint = dict(step.state_dict())
    checkpoint["hps_dp_size"] = 2
    with pytest.raises(ValueError, match="DP size"):
        step.load_state_dict(checkpoint)


def _distributed_pool_checkpoint_worker(root: Path, mesh: DeviceMesh) -> None:
    rank = mesh.get_rank()
    runtime_config = MultiProcess.Config()
    runtime_config.device = "cpu"
    runtime_config.mesh_topology = {"dp": 2, "pp": 1, "tp": 1}
    runtime = runtime_config.make()
    try:
        runtime.initialize()
        step = _tiny_step(data_parallel=True)
        step._pool_inputs.fill_(rank + 1)
        step._pool_labels.fill_(rank + 2)
        step._pool_ids.fill_(rank + 3)
        step._pool_feedback.fill_(rank + 4)
        step._pool_z_slow.fill_(rank + 5)
        step._pool_z_fast.fill_(rank + 6)
        step._pool_halted.fill_(rank == 0)
        step._pool_steps.fill_(rank + 7)
        step._halt_rng.manual_seed(rank + 101)
        step._feedback_rng.manual_seed(rank + 201)
        path = root / "step.pt"
        storer = SyncLocalStateDictStorer()
        storer.write(path, {"step": step.state_dict()})

        resumed = _tiny_step(data_parallel=True)
        saved = storer.read(path, {"step": resumed.state_dict()})["step"]
        assert isinstance(saved, Mapping)
        resumed.load_state_dict(cast(Mapping[str, object], saved))
        for name in (
            "_pool_inputs",
            "_pool_labels",
            "_pool_ids",
            "_pool_feedback",
            "_pool_z_slow",
            "_pool_z_fast",
            "_pool_halted",
            "_pool_steps",
        ):
            torch.testing.assert_close(getattr(resumed, name), getattr(step, name))
        assert torch.equal(resumed._halt_rng.get_state(), step._halt_rng.get_state())
        assert torch.equal(
            resumed._feedback_rng.get_state(),
            step._feedback_rng.get_state(),
        )
        assert path.is_dir()
        result = "ok"
    except Exception:  # noqa: BLE001 -- The worker must report failures to pytest.
        result = traceback.format_exc()
    finally:
        runtime.destroy()
    (root / f"rank_{rank}").write_text(result)


@pytest.mark.compute_distributed
def test_distributed_checkpoint_preserves_rank_local_pool(
    tmp_path: Path,
    warm_pools: WarmPoolGetter,
) -> None:
    warm_pools({"dp": 2})(partial(_distributed_pool_checkpoint_worker, tmp_path))
    assert {
        path.name: path.read_text()
        for path in tmp_path.iterdir()
        if path.name.startswith("rank_")
    } == {
        "rank_0": "ok",
        "rank_1": "ok",
    }


def test_precision_cast_emulation_is_scoped_to_train_and_eval() -> None:
    with _precision_casts(False):
        step = _tiny_step()
        assert cast(bool, inductor_config.emulate_precision_casts) is False
        forward_values: list[bool] = []
        backward_values: list[bool] = []

        def record_forward(module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
            del module, inputs
            forward_values.append(cast(bool, inductor_config.emulate_precision_casts))

        def record_backward(gradient: torch.Tensor) -> torch.Tensor:
            backward_values.append(cast(bool, inductor_config.emulate_precision_casts))
            return gradient

        step.model.register_forward_pre_hook(record_forward)
        next(step.model.parameters()).register_hook(record_backward)
        media = torch.tensor([[2, 3, 0, 0], [4, 5, 0, 0]])
        labels = torch.tensor([[2, 3, -100, -100], [4, 5, -100, -100]])
        batch = {
            "media": media,
            "label": labels,
            "puzzle_identifiers": torch.tensor([7, 8]),
            "valid_count": 2,
        }
        step.train_step(**batch)
        assert cast(bool, inductor_config.emulate_precision_casts) is False
        step.eval_loss(**batch)
        assert cast(bool, inductor_config.emulate_precision_casts) is False
        assert forward_values == [True] * (1 + step.config.max_act_steps)
        assert backward_values == [True]


def test_tiny_step_runs_train_and_clean_eval_with_padding() -> None:
    step = _tiny_step()
    step.config.feedback_corruption_rate = 0
    torch.manual_seed(42)
    media = torch.tensor([[2, 3, 0, 0], [4, 5, 0, 0]])
    labels = torch.tensor([[2, 3, -100, -100], [4, 5, -100, -100]])
    identifiers = torch.tensor([1, 2])
    train = step.train_step(
        media=media,
        label=labels,
        puzzle_identifiers=identifiers,
        valid_count=1,
    )
    evaluation = step.eval_loss(
        media=media,
        label=labels,
        puzzle_identifiers=identifiers,
        valid_count=1,
    )
    assert torch.isfinite(train["loss"]).all()
    assert not torch.equal(step._pool_feedback[:, :2], media[:, :2])
    assert evaluation["model"].shape == (2, 5)
    assert torch.isfinite(evaluation["loss"]).all()


def test_dense_optimizer_warms_up_but_sparse_table_rate_stays_flat() -> None:
    step = _tiny_step()
    sparse_lr = learning_rate(step._sparse_optimizer)
    group_count = len(step.optimizer.param_groups)
    initial_lrs = [
        learning_rate(step.optimizer, group_index=index) for index in range(group_count)
    ]
    step.timer_step.global_count = 0
    step.apply_learning_rate()
    assert [
        learning_rate(step.optimizer, group_index=index) for index in range(group_count)
    ] == [0.0] * group_count
    assert learning_rate(step._sparse_optimizer) == sparse_lr

    step.timer_step.global_count = step.config.warmup_steps // 2
    step.apply_learning_rate()
    assert [
        learning_rate(step.optimizer, group_index=index) for index in range(group_count)
    ] == [rate / 2 for rate in initial_lrs]
    assert learning_rate(step._sparse_optimizer) == sparse_lr

    step.timer_step.global_count = step.config.warmup_steps
    step.apply_learning_rate()
    assert [
        learning_rate(step.optimizer, group_index=index) for index in range(group_count)
    ] == initial_lrs
    assert learning_rate(step._sparse_optimizer) == sparse_lr


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
