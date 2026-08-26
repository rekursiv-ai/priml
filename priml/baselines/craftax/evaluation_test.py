"""Shared evaluation contracts for every Craftax trainer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import copy

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.gtrxl_train_step import CraftaxGTrXLTrainStep
from priml.baselines.craftax.pqn_train_step import CraftaxPQNTrainStep
from priml.baselines.craftax.rnn_train_step import CraftaxRNNTrainStep
from priml.baselines.craftax.train_step import CraftaxTrainStep
from priml.train.parallelism import NoParallel


pytestmark = pytest.mark.compute_training

type CraftaxStep = (
    CraftaxTrainStep | CraftaxRNNTrainStep | CraftaxPQNTrainStep | CraftaxGTrXLTrainStep
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Case:
    name: str
    build: Callable[[], CraftaxStep]
    fields: tuple[str, ...]


def _ppo() -> CraftaxStep:
    config = CraftaxTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.env.device = "cpu"
    config.env.num_envs = 2
    config.rollout_steps = 2
    config.num_epochs = 1
    config.num_minibatches = 1
    config.total_train_steps = 10
    config.model.channels_in = 4
    config.model.num_layers = 1
    return config.make()


def _rnn() -> CraftaxStep:
    config = CraftaxRNNTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.env.device = "cpu"
    config.env.num_envs = 2
    config.env.optimistic_reset_ratio = 1
    config.env.view = (3, 3)
    config.rollout_steps = 2
    config.num_epochs = 1
    config.num_minibatches = 1
    config.total_train_steps = 10
    config.model.channels_in = 4
    return config.make()


def _pqn() -> CraftaxStep:
    config = CraftaxPQNTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.env.device = "cpu"
    config.env.num_envs = 2
    config.env.optimistic_reset_ratio = 1
    config.env.view = (3, 3)
    config.rollout_steps = 2
    config.num_epochs = 1
    config.num_minibatches = 1
    config.total_train_steps = 10
    config.model.channels_in = 4
    return config.make()


def _gtrxl() -> CraftaxStep:
    config = CraftaxGTrXLTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.env.device = "cpu"
    config.env.num_envs = 2
    config.rollout_steps = 4
    config.gradient_window = 2
    config.num_epochs = 1
    config.num_minibatches = 1
    config.total_train_steps = 10
    config.model.embed_dim = 4
    config.model.num_heads = 1
    config.model.num_layers = 1
    config.model.qkv_dim = 4
    config.model.channels_in = 4
    config.model.memory_length = 2
    return config.make()


_CASES = (
    _Case(
        name="ppo",
        build=_ppo,
        fields=(
            "_observation",
            "_done",
            "_episode_return",
            "_episode_length",
            "_finished_returns",
            "_finished_lengths",
        ),
    ),
    _Case(
        name="rnn",
        build=_rnn,
        fields=(
            "_observation",
            "_state",
            "_previous_done",
            "_episode_return",
            "_episode_length",
            "_finished_returns",
            "_finished_lengths",
        ),
    ),
    _Case(
        name="pqn",
        build=_pqn,
        fields=(
            "_observation",
            "_state",
            "_previous_action",
            "_previous_done",
            "_episode_return",
            "_episode_length",
            "_finished_returns",
            "_finished_lengths",
        ),
    ),
    _Case(
        name="gtrxl",
        build=_gtrxl,
        fields=(
            "_observation",
            "_memory",
            "_valid_length",
            "_previous_done",
            "_episode_return",
            "_episode_length",
            "_finished_returns",
            "_finished_lengths",
        ),
    ),
)


def _snapshot(step: CraftaxStep, fields: tuple[str, ...]) -> dict[str, object]:
    return {
        "state_dict": copy.deepcopy(step.state_dict()),
        "fields": {name: copy.deepcopy(getattr(step, name)) for name in fields},
        "training": step.model.training,
    }


def _assert_tree_equal(left: object, right: object, path: str = "root") -> None:
    if isinstance(left, Tensor):
        assert isinstance(right, Tensor), f"{path}: {type(right).__name__}"
        assert torch.equal(left, right), f"{path}: tensors differ"
        return
    if isinstance(left, dict):
        assert isinstance(right, dict), f"{path}: {type(right).__name__}"
        left_dict = cast(dict[object, object], left)
        right_dict = cast(dict[object, object], right)
        assert left_dict.keys() == right_dict.keys(), f"{path}: keys differ"
        for key in left_dict:
            _assert_tree_equal(left_dict[key], right_dict[key], f"{path}.{key}")
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        assert isinstance(right, Sequence), f"{path}: {type(right).__name__}"
        assert len(left) == len(right), f"{path}: lengths differ"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_tree_equal(left_item, right_item, f"{path}[{index}]")
        return
    assert left == right, f"{path}: {left!r} != {right!r}"


def _seed_finished_banks(step: CraftaxStep) -> None:
    step._finished_returns.append(123.0)
    step._finished_lengths.append(456)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_eval_loss_preserves_complete_training_lifecycle(case: _Case) -> None:
    step = case.build()
    step.train_step()
    _seed_finished_banks(step)
    step.model.eval()
    before = _snapshot(step, case.fields)

    step.eval_loss()

    _assert_tree_equal(before, _snapshot(step, case.fields))


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_checkpoint_round_trips_complete_training_lifecycle(case: _Case) -> None:
    step = case.build()
    step.train_step()
    _seed_finished_banks(step)
    if isinstance(step, CraftaxTrainStep):
        step._done.fill_(True)
    before = _snapshot(step, case.fields)
    saved = copy.deepcopy(step.state_dict())

    resumed = case.build()
    resumed.load_state_dict(saved)

    _assert_tree_equal(before, _snapshot(resumed, case.fields))
