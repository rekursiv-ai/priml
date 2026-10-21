"""Pin the released-inference and cold-training experiment configurations.

The finalized inference recipe is too large for one golden, so each test pins a
disjoint set of its subtrees; together they cover every field.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from configgle import PartialConfig
from torch import Tensor

import torch

from priml.baselines.rsna_knee.experiments import (
    exp000,
    train_exp000,
    train_exp001,
)
from priml.loss.simple_loss import SimpleLoss, bce_with_logits
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


if TYPE_CHECKING:
    from collections.abc import Callable

    from configgle import Maker

    from priml.baselines.rsna_knee.inference import PublicRecipe


_CWD: Final = Path(__file__).resolve().parent


class _Subtrees:
    """Render named subtrees of one finalized recipe as a single pprint."""

    def __init__(self, **subtrees: object) -> None:
        self.subtrees = subtrees

    def pformat(self, **kwargs: object) -> str:
        """Render each subtree under its name, keeping the golden's own options."""
        parts: list[str] = []
        for name, subtree in self.subtrees.items():
            pformat = cast(
                "Callable[..., str] | None",
                getattr(subtree, "pformat", None),
            )
            parts.append(f"{name}: {pformat(**kwargs) if pformat else repr(subtree)}")
        return "\n".join(parts)


@cache
def _recipe() -> PublicRecipe.Config:
    return exp000().copy_tree().finalize()


def test_recipe_top_level_configuration() -> None:
    config = _recipe()
    assert_pprint_golden(
        test_file=__file__,
        name="exp000",
        config=cast(
            "Maker[object]",
            _Subtrees(
                study_name=config.study_name,
                experiment_name=config.experiment_name,
                base_dir=config.base_dir,
                working_dir=config.working_dir,
                data=config.data,
                blend=config.blend,
                budget=config.budget,
            ),
        ),
    )


def test_raptor384_configuration() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="exp000_raptor384",
        config=cast("Maker[object]", _Subtrees(raptor384=_recipe().raptor384)),
    )


def test_raptor224_configuration() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="exp000_raptor224",
        config=cast("Maker[object]", _Subtrees(raptor224=_recipe().raptor224)),
    )


def test_r7_model_configuration() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="exp000_r7_model",
        config=cast("Maker[object]", _Subtrees(model=_recipe().r7.model)),
    )


def test_r7_auxiliary_model_configuration() -> None:
    r7 = _recipe().r7
    assert_pprint_golden(
        test_file=__file__,
        name="exp000_r7_auxiliary",
        config=cast("Maker[object]", _Subtrees(anatomy=r7.anatomy, side=r7.side)),
    )


def test_r7_remaining_configuration() -> None:
    r7 = _recipe().r7
    assert_pprint_golden(
        test_file=__file__,
        name="exp000_r7_rest",
        config=cast(
            "Maker[object]",
            _Subtrees(
                base_dir=r7.base_dir,
                checkpoints=r7.checkpoints,
                anatomy_checkpoint=r7.anatomy_checkpoint,
                side_checkpoint=r7.side_checkpoint,
                preparation=r7.preparation,
                score=r7.score,
                postprocess=r7.postprocess,
                execution=r7.execution,
                cudnn_benchmark=r7.cudnn_benchmark,
                allow_tf32_cudnn=r7.allow_tf32_cudnn,
                allow_tf32_matmul=r7.allow_tf32_matmul,
            ),
        ),
    )


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
