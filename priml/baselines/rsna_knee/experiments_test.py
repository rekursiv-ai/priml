"""Pin the released-inference and cold-training experiment configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import math

from configgle import PartialConfig
from torch import Tensor

import torch

from priml.baselines.rsna_knee.experiments import (
    exp000,
    train_exp000,
    train_exp001,
)
from priml.baselines.rsna_knee.imaging import unchanged
from priml.baselines.rsna_knee.inference import accept_logits
from priml.loss.simple_loss import SimpleLoss, bce_with_logits
from priml.model.coatnet import CoAtNet
from priml.model.linear import Linear
from priml.optimizers.composite import CompositeOptimizer
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
    assert_pprint_golden,
    assert_tensor_golden,
    put_steps,
    rng_fingerprint,
)
from priml.train.parallelism import NoParallel


_CWD: Final = Path(__file__).resolve().parent


def test_released_recipe_pins_every_finalized_field() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_second_raptor_arm_changes_only_its_released_choices() -> None:
    config = exp000()
    arm, parent = config.raptor224, config.raptor384
    assert isinstance(arm.model.backbone, CoAtNet.Config)
    assert isinstance(parent.model.backbone, CoAtNet.Config)
    assert arm.model.backbone.image_size == (224, 224)
    assert arm.model.chunk_size == math.inf
    assert arm.windows.crop_size == 307
    assert arm.views == [unchanged]
    assert arm.score.dtypes_autocast == [torch.float16, None]
    assert arm.score.check_logits is accept_logits
    arm.checkpoint.file.path = parent.checkpoint.file.path
    arm.checkpoint.file.sha256 = parent.checkpoint.file.sha256
    arm.model.backbone.image_size = parent.model.backbone.image_size
    arm.model.chunk_size = parent.model.chunk_size
    arm.windows.crop_size = parent.windows.crop_size
    arm.views = parent.views
    arm.score.dtypes_autocast = parent.score.dtypes_autocast
    arm.score.check_logits = parent.score.check_logits
    assert arm == parent, arm.udiff(parent)


def test_cold_training_tree_finalizes_without_source_data_or_a_gpu() -> None:
    config = train_exp000().copy_tree().finalize()
    assert config.cold_loop.working_dir == Path(
        "/opt/scratch/runs/rsna_knee_training/exp000",
    )
    assert config.cold_loop.seed is None
    assert config.cold_loop.num_steps_eval == -1
    assert config.cold_loop.eval_every_epoch is False
    assert config.cold_loop.checkpointer is None
    assert config.cold_loop.max_steps == 10_000
    assert config.cold_loop.max_time == config.cold_loop.step.train_budget_sec == 180
    assert config.cold_loop.max_eval_time == config.budget.reserve_evaluation_sec == 45
    assert "ColdStudies.Config" in config.pformat(hide_default_values=False)


def test_training_recipe_pins_complete_model_optimizer_and_precision() -> None:
    assert_pprint_golden(test_file=__file__, name="train_exp000", config=train_exp000())


def test_training_recipe_declares_the_original_new_protocol_choices() -> None:
    config = train_exp000().cold_loop.step
    assert config.compile is None
    assert config.dtype_autocast == torch.bfloat16
    assert config.model.head.dropout.p == 0
    assert isinstance(config.loss, SimpleLoss.Config)
    assert config.loss.loss_fn is bce_with_logits
    assert isinstance(config.optimizer, CompositeOptimizer.Config)
    optimizer = config.optimizer.optimizers[0]
    assert isinstance(optimizer, PartialConfig)
    function, args, kwargs = optimizer.deferred_call()
    assert function is torch.optim.AdamW
    assert args == []
    assert kwargs == {"lr": 1e-4, "weight_decay": 1e-2}


def test_batch_size_fork_changes_only_the_named_parent_choice() -> None:
    base, fork = train_exp000(), train_exp001()
    assert base.cold_loop.dataset.batch_size == 2
    assert fork.cold_loop.dataset.batch_size == 4
    assert fork.cold_loop.experiment_name == "exp001"
    fork.cold_loop.dataset.batch_size = base.cold_loop.dataset.batch_size
    fork.cold_loop.experiment_name = base.cold_loop.experiment_name
    assert fork == base, fork.udiff(base)


def test_optimizer_stack_replays_three_updates_bit_for_bit() -> None:
    """Replay train_exp000's loss, optimizer and schedule on a minimum-size model.

    Only size shrinks, and autocast is dropped because CPU half-precision
    kernels are host-specific; the optimizer, loss and learning rate stay.
    """
    config = train_exp000().cold_loop.step
    encoder = Linear.Config()
    encoder.channels_in, encoder.channels_out = 4, 7
    config.model.backbone = encoder
    config.model.head.channels_in, config.model.head.channels_hidden = 7, 6
    config.model.head.channels_out = 5
    config.dtype_autocast = None
    config.parallelism = NoParallel.Config(device="cpu")
    records: list[dict[str, Tensor]] = []
    with host_agnostic_numerics():
        torch.manual_seed(19)
        step = config.make()
        assert isinstance(step.optimizer, CompositeOptimizer)
        initial = {
            f"initial/{key}": value.detach().clone()
            for key, value in step.model.state_dict().items()
        }
        for index in range(3):
            media = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
            output = step.train_step(
                media=media / 17 - 1 + index / 8,
                label=((torch.arange(2 * 5).reshape(2, 5) + index) % 2).float(),
            )
            records.append(
                {
                    "loss": output["loss"].detach().float(),
                    **{
                        f"parameter/{key}": value.detach().clone()
                        for key, value in step.model.state_dict().items()
                    },
                    **{
                        f"optimizer/{member}/{slot}/{name}": value.detach().clone()
                        for member, optimizer in enumerate(step.optimizer.optimizers)
                        for slot, state in cast(
                            "dict[int, dict[str, object]]",
                            optimizer.state_dict()["state"],
                        ).items()
                        for name, value in state.items()
                        if isinstance(value, Tensor)
                    },
                },
            )
    record = {**initial, "rng": rng_fingerprint()}
    put_steps(record, "step", records)
    assert_tensor_golden(_CWD / "testdata" / "train_exp000.pt", record)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
