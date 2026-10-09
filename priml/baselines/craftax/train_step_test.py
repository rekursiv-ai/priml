"""Unit tests for the learner minibatch and the pipeline at tiny sizes on the CPU.

Goldens freeze whole epochs of the tiny pipeline, the same bits on every CPU:
exp000's recipe in its torch forms from portable seed-73 masters (four epochs
that walk its whole cosine, which a resume after epoch 2 and a checkpoint
loaded two epochs later must reproduce), and exp002's from the port's own
init. On the GPU, host-keyed goldens freeze exp000's kernels at the same tiny
size: four epochs, the last three as captured learner graphs (captured again
after a load drops them), and three steps of its fused Muon. The fp32
cosine's rate at every one of exp000's epochs is frozen on each platform's
libm.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import fields, replace
from functools import cache
from typing import TYPE_CHECKING, cast, override

import copy
import io
import math

from configgle import Fig, Makes, PartialConfig
from torch import Tensor, nn

import numpy as np
import pytest
import torch

from priml.baselines.craftax.env import StallCap, WorldPool
from priml.baselines.craftax.experiments import exp000
from priml.baselines.craftax.game.state import ATN_DIM, OBS_SIZE
from priml.baselines.craftax.learners.imitation import BranchImitation
from priml.baselines.craftax.learners.practice import FrontierPractice
from priml.baselines.craftax.learners.update import ShuffledTransitions
from priml.baselines.craftax.model import DenseObservation, MinGRUPolicy
from priml.baselines.craftax.rollout import (
    CAPTURE_LOCK,
    PhiloxSampler,
    Rollout,
)
from priml.baselines.craftax.testing import (
    assert_golden,
    digest,
    forward_parameters,
    fp32,
    gpu_key,
    host_agnostic_pipeline,
    optimizer_state,
    packed_observations,
    portable_checkpoint,
    portable_uniform,
    read_golden,
    require_golden,
    smoke_feature,
    tiny_env,
    tiny_exp000_step,
    tiny_exp002_step,
    tiny_policy,
    tiny_train_step,
)
from priml.baselines.craftax.train_step import (
    AgentWindows,
    CraftaxTrainStep,
    LearnerRollout,
    ProgressSchedule,
    _check_recipe,
    cosine_annealing_fp32,
    learn_joint_minibatch,
    learn_minibatch,
    load_masters,
    minibatch_count,
    minibatch_offsets,
    score_minibatch,
)
from priml.baselines.craftax.world_model.context import (
    ContextReplay,
    JointWorldModel,
)
from priml.baselines.craftax.world_model.feature import (
    FeatureEngine,
    Sliding,
    WorldModelFeature,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.testing import (
    context_reference,
    random_contexts,
    small_schema,
    tiny_model,
)
from priml.lib.codec import from_plain
from priml.loss.policy_gradient import TorchPPO
from priml.loss.policy_gradient_kernel import TritonPPO
from priml.model.linear import Linear
from priml.optimizers.fused_muon import FusedMuon, clip_coefficient


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from priml.baselines.craftax.rollout import RolloutStorage
    from priml.baselines.craftax.world_model.model import WorldModel
    from priml.train.custom_types import TrainStepOutput


def test_exp000_learns_18_minibatches_wrapping_past_16() -> None:
    count = minibatch_count(
        agents=2048,
        horizon=256,
        minibatch_size=32_768,
        replay_ratio=1.17263889,
    )
    assert count == 18
    offsets = minibatch_offsets(agents=2048, rows=128, count=count)
    assert offsets[:3] == [0, 128, 256]
    assert offsets[15] == 1920
    assert offsets[16:] == [0, 128]


def _rollout(
    config: MinGRUPolicy.Config,
    *,
    agents: int,
    horizon: int,
    seed: int = 0,
) -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    observations = packed_observations(config, batch=horizon, time=agents, seed=seed)
    actions = torch.randint(0, 43, (horizon, agents), generator=generator).float()
    logprobs = (-torch.rand(horizon, agents, generator=generator) * 3).bfloat16()
    rewards = (torch.randn(horizon, agents, generator=generator) * 2).bfloat16()
    terminals = (torch.rand(horizon, agents, generator=generator) < 0.1).bfloat16()
    values = torch.randn(horizon, agents, generator=generator).bfloat16()
    action_mask = torch.ones(horizon, agents, 43).bfloat16()
    initial_states = torch.randn(
        config.num_layers,
        agents,
        config.channels_hidden,
        generator=generator,
    ).bfloat16()
    branch_starts = (torch.rand(agents, generator=generator) < 0.5).to(torch.uint8)
    return (
        observations,
        actions,
        logprobs,
        rewards,
        terminals,
        values,
        action_mask,
        initial_states,
        branch_starts,
    )


def test_the_learner_rollout_is_agent_major_with_clamped_rewards() -> None:
    config = tiny_policy()
    buffers = _rollout(config, agents=4, horizon=5)
    features = torch.randn(5, 4, 3, generator=torch.Generator().manual_seed(1))
    rollout = LearnerRollout.from_time_major(
        *buffers,
        reward_scale=1.0,
        reward_clip=1.0,
        features=features.bfloat16(),
    )
    assert rollout.observations.shape == (4, 5, 843)
    assert rollout.features is not None
    assert torch.equal(rollout.features[1, 2], features[2, 1].bfloat16())
    assert rollout.actions.shape == (4, 5)
    assert rollout.action_mask.shape == (4, 5, 43)
    assert rollout.initial_states.shape == (2, 4, 8)
    assert rollout.branch_starts.shape == (4,)
    assert torch.equal(rollout.observations[1, 2], buffers[0][2, 1])
    assert rollout.rewards.abs().max() <= 1
    # The bf16 clamp is PufferLib's fp32 clamp before the store, and a unit
    # scale leaves the bits as they were.
    assert torch.equal(
        rollout.rewards,
        buffers[3].transpose(0, 1).float().clamp(-1, 1).bfloat16(),
    )
    assert not torch.equal(rollout.rewards, buffers[3].transpose(0, 1))
    # Every field is agent-major in memory too, so no kernel copies it again.
    for entry in fields(rollout):
        value = cast("torch.Tensor | None", getattr(rollout, entry.name))
        assert value is None or value.is_contiguous(), entry.name
    minibatch = rollout.minibatch(1, 2)
    assert torch.equal(minibatch.values, rollout.values[1:3])
    assert torch.equal(minibatch.initial_states, rollout.initial_states[:, 1:3])
    assert torch.equal(minibatch.branch_starts, buffers[8][1:3])
    assert minibatch.observations.is_contiguous()
    assert minibatch.features is not None
    assert torch.equal(minibatch.features, rollout.features[1:3])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_reward_scale_of_an_eighth_is_exact_before_the_clamp(
    dtype: torch.dtype,
) -> None:
    """exp103's 0.125 with no clamp is an exact power of two, in either storage.

    With a clamp at 0.25 the scale comes first: 3 becomes 0.375, clamped to
    0.25, where clamping first would give 1/32.
    """
    buffers = list(_rollout(tiny_policy(), agents=4, horizon=5))
    rewards = buffers[3] = (buffers[3].float() * 4).to(dtype)
    scaled = LearnerRollout.from_time_major(
        *buffers,
        reward_scale=0.125,
        reward_clip=math.inf,
    )
    assert torch.equal(scaled.rewards, (rewards.transpose(0, 1).double() / 8).to(dtype))
    assert scaled.rewards.abs().max() > 1
    buffers[3] = torch.full_like(rewards, 3.0)
    clamped = LearnerRollout.from_time_major(
        *buffers,
        reward_scale=0.125,
        reward_clip=0.25,
    )
    assert torch.equal(clamped.rewards, torch.full((4, 5), 0.25, dtype=dtype))


def _take_gradients(policy: nn.Module) -> list[Tensor]:
    """Take every weight's gradient, leaving ``grad`` None for the next backward."""
    gradients: list[Tensor] = []
    for parameter in policy.parameters():
        assert parameter.grad is not None
        gradients.append(parameter.grad)
        parameter.grad = None
    return gradients


def test_learn_minibatch_leaves_a_gradient_on_every_weight() -> None:
    config = tiny_policy()
    policy = config.make()
    rollout = LearnerRollout.from_time_major(
        *_rollout(config, agents=4, horizon=5),
        reward_scale=1.0,
        reward_clip=1.0,
    )
    losses, auxiliary_loss = learn_minibatch(
        policy,
        _objective(),
        rollout.minibatch(2, 2),
    )
    assert losses.shape == (len(TorchPPO.Config.LOSS_NAMES),)
    assert not losses.requires_grad
    assert float(auxiliary_loss) == 0.0
    gradients = _take_gradients(policy)
    for gradient, parameter in zip(gradients, policy.parameters(), strict=True):
        assert gradient.shape == parameter.shape
        assert gradient.dtype == parameter.dtype
        assert not torch.equal(gradient, torch.zeros_like(gradient))
    # The minibatch starts from the rollout's carry for its rows, so a
    # different carry changes the gradients.
    learn_minibatch(policy, _objective(), rollout.minibatch(0, 2))
    assert not torch.equal(_take_gradients(policy)[1], gradients[1])


def _objective() -> TorchPPO:
    """Return exp000's learning rule and coefficients, in torch for the CPU."""
    windows = exp000().step.learner
    assert isinstance(windows, AgentWindows.Config)
    assert isinstance(windows.objective, TorchPPO.Config)
    return TorchPPO(windows.objective)


def test_scoring_is_learning_without_the_backward() -> None:
    config = tiny_policy()
    policy = config.make()
    rollout = LearnerRollout.from_time_major(
        *_rollout(config, agents=4, horizon=5),
        reward_scale=1.0,
        reward_clip=1.0,
    )
    with torch.no_grad():
        total, scored, _ = score_minibatch(
            policy,
            _objective(),
            rollout.minibatch(0, 2),
        )
    assert all(parameter.grad is None for parameter in policy.parameters())
    learned, _ = learn_minibatch(policy, _objective(), rollout.minibatch(0, 2))
    assert torch.equal(scored, learned)
    assert torch.equal(total, scored[TorchPPO.Config.LOSS_NAMES.index("total_loss")])


def test_the_total_adds_the_policys_auxiliary_loss() -> None:
    policy = _LinearPolicy(_LinearPolicy.Config())
    rollout = LearnerRollout.from_time_major(
        *_rollout(tiny_policy(), agents=4, horizon=5),
        reward_scale=1.0,
        reward_clip=1.0,
    )
    total, losses, auxiliary_loss = score_minibatch(
        policy,
        _objective(),
        rollout.minibatch(0, 2),
    )
    rule = losses[TorchPPO.Config.LOSS_NAMES.index("total_loss")]
    assert torch.equal(auxiliary_loss, policy.head.weight.float().mean())
    assert torch.equal(total, rule + auxiliary_loss)


def test_an_extra_loss_joins_the_minibatchs_one_backward() -> None:
    """The extra loss is added to the total before its one backward, not after.

    The extra is built before the forward, as the windows build it: autograd
    adds a weight's gradients in the reverse of the order their nodes were
    built, so the order is part of the bits.
    """
    policy = _LinearPolicy(_LinearPolicy.Config())
    rollout = LearnerRollout.from_time_major(
        *_rollout(tiny_policy(), agents=4, horizon=5),
        reward_scale=1.0,
        reward_clip=1.0,
    )
    minibatch = rollout.minibatch(0, 2)
    extra_loss = policy.head.weight.sum() * 3
    total, losses, _ = score_minibatch(policy, _objective(), minibatch)
    (total + extra_loss).backward()
    (expected,) = _take_gradients(policy)
    joined, auxiliary_loss = learn_minibatch(
        policy,
        _objective(),
        minibatch,
        extra_loss=policy.head.weight.sum() * 3,
    )
    assert torch.equal(joined, losses)
    assert torch.equal(auxiliary_loss, policy.head.weight.float().mean())
    (gradient,) = _take_gradients(policy)
    assert torch.equal(gradient, expected)
    learn_minibatch(policy, _objective(), minibatch)
    assert not torch.equal(_take_gradients(policy)[0], expected)


def _train_step(config: CraftaxTrainStep.Config | None = None) -> CraftaxTrainStep:
    if config is None:
        config = _quick(tiny_train_step())
    torch.manual_seed(0)
    return config.make()


# An epoch of the tiny pipeline at a horizon of 4 in two windows took 25 ms on x86;
# here 15 ms. The windows' tiling is checked by its own tests.
def _quick(config: CraftaxTrainStep.Config) -> CraftaxTrainStep.Config:
    """Cut a tiny step's rollouts to 2 steps, learned as one window of every agent."""
    config.rollout.horizon = 2
    windows = config.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.minibatch_size = 2 * config.env.num_envs
    return config


def test_the_pipeline_boots_then_prefetches_one_rollout_ahead_of_the_learner() -> None:
    step = _train_step()
    try:
        # The training loop drives it through the step protocol alone.
        initial = [p.detach().clone() for p in step.model.parameters()]
        assert isinstance(step.learner, AgentWindows)
        assert step.learner.offsets == [0]
        first = step.train_step()
        # Boot and epoch 0 both collected with the initial weights, which the
        # actor still holds; the learner has moved on, and the slots swapped.
        assert (step.global_step, step.ready, step.write) == (1, 1, 0)
        for actor, weight in zip(step.actor.parameters(), initial, strict=True):
            assert torch.equal(actor, weight)
        assert not all(
            torch.equal(parameter, weight)
            for parameter, weight in zip(step.model.parameters(), initial, strict=True)
        )
        assert first["model"].shape == (len(TorchPPO.Config.LOSS_NAMES),)
        assert "metrics" in first
        assert "learning_rate" in first["metrics"]
        assert first["metrics"]["agent_steps"] == 1 * 4 * 2
        step.train_step()
        # Epoch 1 copied the epoch-0 update into the actor before its rollout.
        assert (step.global_step, step.ready, step.write) == (2, 0, 1)
        last = step.train_step()
        # The last epoch does not prefetch, so the slots stay, and it timed no
        # rollout.
        assert (step.global_step, step.ready, step.write) == (3, 0, 1)
        assert "metrics" in last
        assert last["metrics"]["rollout_seconds"] == 0.0
        # Past the budget an epoch would relearn the last rollout; it is refused.
        with pytest.raises(RuntimeError, match="spent"):
            step.train_step()
        assert float(last["metrics"]["learning_rate"]) < float(
            first["metrics"]["learning_rate"],
        )
        assert step.train_loss()["model"].shape == (len(TorchPPO.Config.LOSS_NAMES),)
        # The evaluation trainer mirrors the training geometry and sampler.
        evaluator = step.make_evaluator()
        assert evaluator.env.num_envs == step.env.num_envs
        evaluator.close()
    finally:
        step.close()


def test_the_envs_are_readied_before_every_rollout_and_report_practice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three epochs run three rollouts: the boot and two prefetches.

    Each is readied with no rollout in flight, and each epoch reports the
    environments' practice metrics and the learner's own.
    """
    step = _train_step()
    try:
        in_flight: list[bool] = []
        monkeypatch.setattr(
            step.env,
            "prepare_rollout",
            lambda: in_flight.append(step._pending is not None),
        )
        monkeypatch.setattr(
            step.env,
            "practice_metrics",
            lambda: {"practice/fraction": 0.25},
        )
        results = [step.train_step() for _ in range(3)]
    finally:
        step.close()
    assert in_flight == [False, False, False]
    transitions = step.env.num_envs * step.config.rollout.horizon
    for result in results:
        metrics = result.get("metrics", {})
        assert metrics["practice/fraction"] == 0.25
        assert float(metrics["auxiliary_loss"]) == 0.0
        assert metrics["transitions_per_second"] == pytest.approx(
            transitions / float(metrics["epoch_seconds"]),
        )


def test_the_episode_metrics_count_only_episodes_finished_since_the_last() -> None:
    """Each report differences the env's running log sums against the last read."""
    step = _train_step()
    try:
        logs = step.env.stats["log"]
        logs["n"][:2] = 2.0
        logs["perf"][:2] = (0.5, 1.5)
        first = step._episode_metrics()
        again = step._episode_metrics()
        logs["n"][0] += 1.0
        logs["perf"][0] += 0.25
        third = step._episode_metrics()
    finally:
        step.close()
    assert (first["env/n"], first["env/perf"]) == (4.0, 0.5)
    assert again == {}
    assert (third["env/n"], third["env/perf"]) == (1.0, 0.25)


def test_a_resumed_step_reports_only_the_episodes_after_the_resume() -> None:
    """The checkpoint carries the logs' running sums; they are the new baseline."""
    source, target = _train_step(), _train_step()
    try:
        source.env.stats["log"]["n"][:2] = 2.0
        source.env.stats["log"]["perf"][:2] = (0.5, 1.5)
        saved = io.BytesIO()
        torch.save(source.state_dict(), saved)
        target.load_state_dict(_read(saved))
        resumed = target._episode_metrics()
        target.env.stats["log"]["n"][0] += 1.0
        target.env.stats["log"]["perf"][0] += 0.25
        after = target._episode_metrics()
    finally:
        source.close()
        target.close()
    assert resumed == {}
    assert (after["env/n"], after["env/perf"]) == (1.0, 0.25)


def test_the_optimizer_holds_the_weights_in_parameters_order() -> None:
    # The global norm reduces the gradients in the optimizer's order.
    step = _train_step()
    try:
        held = [
            id(parameter)
            for group in step.optimizer.param_groups
            for parameter in cast("list[Tensor]", group["params"])
        ]
        assert held == [id(parameter) for parameter in step.model.parameters()]
    finally:
        step.close()


def _masters_and_momentum(step: CraftaxTrainStep) -> list[Tensor]:
    muon = step.optimizer
    assert isinstance(muon, FusedMuon)
    return [value.clone() for value in (*muon.master_weights, *muon.momentum_buffers)]


# Closing waits for the rollout the last epoch prefetched into the ready slot.
def _prefetched(step: CraftaxTrainStep) -> list[Tensor]:
    """Return a closed step's ready slot: the rollout its last epoch prefetched."""
    return _slot(step, step.ready)


# The carries and draw counts are the buffers' running state, not the slot's: they
# move on as the next rollout starts.
def _slot(step: CraftaxTrainStep, slot: int) -> list[Tensor]:
    """Return a copy of one slot's rollout: what it stores, the carries at its start."""
    return [
        value.clone()
        for name, value in step.rollout.state_dict(slot=slot).items()
        if name not in {"carry", "draws"}
    ]


def _read(saved: io.BytesIO) -> dict[str, object]:
    """Read back a step's state as the checkpointer's plain ``torch.save`` wrote it."""
    return from_plain(
        cast("object", torch.load(io.BytesIO(saved.getvalue()), weights_only=True)),
        dict[str, object],
    )


def test_a_resumed_run_equals_an_uninterrupted_one() -> None:
    """Two straight epochs against a checkpoint after the first and one more (CKPT-1).

    The resumed step's epoch learns from the saved slot and prefetches a
    rollout from the saved environments, streams and carries. Each saved part
    is needed: without the slot its losses and Muon's state differ, without
    the rest the rollout it prefetched does. Same-host numerics: the golden
    above holds the bits.
    """
    config = _quick(tiny_train_step())
    config.train_budget_steps = 3
    straight = _train_step(config)
    try:
        straight.train_step()
        # As the checkpointer writes a plain checkpoint, to be read back below.
        saved = io.BytesIO()
        torch.save(straight.state_dict(), saved)
        loss = straight.train_step()["model"]
    finally:
        straight.close()
    expected = [loss, *_masters_and_momentum(straight), *_prefetched(straight)]
    resumed = _train_step(config)
    try:
        resumed.load_state_dict(_read(saved))
        loss = resumed.train_step()["model"]
    finally:
        resumed.close()
    actual = [loss, *_masters_and_momentum(resumed), *_prefetched(resumed)]
    assert len(actual) == len(expected)
    for index, (ours, theirs) in enumerate(zip(actual, expected, strict=True)):
        assert torch.equal(ours, theirs), index


def test_a_loaded_state_restores_every_part_of_the_step() -> None:
    """Each part a checkpoint carries, changed on one step, lands on a fresh one.

    The weights, the optimizer's groups, the learner's archive, the epoch
    count, the worlds' streams, the pipeline's slots and carries, and which
    slot is ready.
    """
    config = tiny_train_step()
    windows = config.learner
    assert isinstance(windows, AgentWindows.Config)
    imitation = windows.auxiliary = BranchImitation.Config()
    imitation.rows = 3
    source, target = _train_step(config), _train_step(config)
    try:
        storage = source.rollout.slots[1]
        storage.branch_starts.fill_(1)
        storage.action_mask.fill_(1)
        storage.observations.add_(1)
        learner = source.learner
        assert isinstance(learner, AgentWindows)
        assert learner.auxiliary is not None
        learner.auxiliary.ingest(source._learner_rollout(storage))
        with torch.no_grad():
            for parameter in source.model.parameters():
                parameter.add_(1)
        source.optimizer.param_groups[0]["lr"] = 0.5
        for _ in range(2):
            with source.timer_step:
                pass
        source.env.rngs[0] += 1
        source.rollout.graphs[0][0].state.add_(1)
        source.ready, source.write, source._booted = 1, 0, True
        saved = io.BytesIO()
        torch.save(source.state_dict(), saved)
        target.load_state_dict(_read(saved))
        for ours, theirs in zip(
            target.model.parameters(),
            source.model.parameters(),
            strict=True,
        ):
            assert torch.equal(ours, theirs)
        assert target.optimizer.param_groups[0]["lr"] == 0.5
        assert target.global_step == 2
        assert target.env.rngs[0] == source.env.rngs[0]
        assert (target.ready, target.write, target._booted) == (1, 0, True)
        assert torch.equal(target.rollout.slots[1].observations, storage.observations)
        assert torch.equal(
            target.rollout.graphs[0][0].state,
            source.rollout.graphs[0][0].state,
        )
        restored = target.learner.state_dict()
        assert int(restored["count"]) == 3
        for name, value in learner.state_dict().items():
            assert torch.equal(restored[name], value), name
        partial_state = {
            key: value for key, value in source.state_dict().items() if key != "learner"
        }
        with pytest.raises(ValueError, match=r"\['learner'\] are missing"):
            target.load_state_dict(partial_state)
    finally:
        source.close()
        target.close()


def test_windows_without_an_auxiliary_refuse_another_recipes_state() -> None:
    windows = AgentWindows(AgentWindows.Config())
    assert windows.state_dict() == {}
    windows.load_state_dict({})
    with pytest.raises(ValueError, match=r"carry no state, not \['count'\]"):
        windows.load_state_dict({"count": torch.tensor(1)})


def test_the_step_starts_from_its_checkpoints_masters(tmp_path: Path) -> None:
    """The parameters take the file's fp32 masters rounded; Muon's take them exactly."""
    config = tiny_train_step()
    generator = torch.Generator().manual_seed(5)
    masters = {
        name: torch.randn(parameter.shape, generator=generator)
        for name, parameter in config.model.make().named_parameters()
    }
    torch.save(masters, tmp_path / "init.pt")
    config.checkpoint = tmp_path / "init.pt"
    step = _train_step(config)
    try:
        for name, parameter in step.model.named_parameters():
            assert torch.equal(parameter, masters[name].bfloat16()), name
        loaded = optimizer_state(step.model, step.optimizer, "master_weight")
        assert loaded.keys() == masters.keys()
        for name, master in loaded.items():
            assert torch.equal(master, masters[name]), name
    finally:
        step.close()


def test_masters_refuse_an_optimizer_parameter_the_policy_does_not_name(
    tmp_path: Path,
) -> None:
    """Only the parameters named beside the policy's keep their own masters."""
    config = tiny_policy()
    model = config.make()
    torch.save(
        {name: p.detach().float() for name, p in model.named_parameters()},
        tmp_path / "masters.pt",
    )
    stray = nn.Parameter(torch.zeros(2, 3))
    optimizer = FusedMuon.Config().make()([*model.parameters(), stray])
    with pytest.raises(ValueError, match="does not name"):
        load_masters(model, optimizer, tmp_path / "masters.pt")
    load_masters(model, optimizer, tmp_path / "masters.pt", others=[stray])


def test_a_failed_rollout_still_releases_every_thread_on_close() -> None:
    step = _train_step()
    failed: Future[RolloutStorage] = Future()
    failed.set_exception(RuntimeError("the rollout failed"))
    step._pending = failed
    with pytest.raises(RuntimeError, match="rollout failed"):
        step.close()
    assert step.rollout.workers._shutdown
    assert step._prefetch._shutdown
    step.close()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("reward_clip", -1.0, "reward_clip"),
        ("reward_scale", 0.0, "reward_scale"),
        ("reward_scale", math.nan, "reward_scale"),
        ("reward_scale", math.inf, "reward_scale"),
        ("train_budget_steps", math.inf, "train_budget_steps"),
    ],
)
def test_the_step_refuses_a_recipe_it_would_run_wrongly(
    field: str,
    value: float,
    match: str,
) -> None:
    config = tiny_train_step()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=match):
        config.make()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("minibatch_size", 24, "tile the 4 environments"),
        ("minibatch_size", 10, "multiple of the horizon"),
        ("replay_ratio", 0.1, "no minibatch"),
        ("replay_ratio", math.nan, "replay_ratio must be positive"),
    ],
)
def test_the_windows_refuse_a_geometry_that_does_not_tile_the_rollout(
    field: str,
    value: float,
    match: str,
) -> None:
    config = tiny_train_step()
    setattr(config.learner, field, value)
    with pytest.raises(ValueError, match=match):
        config.make()


@pytest.mark.parametrize(
    ("num_episodes", "num_envs", "match"),
    [(0, 4, "num_episodes"), (10, 5, "multiple of num_buffers")],
)
def test_the_step_refuses_an_evaluation_it_could_not_play(
    num_episodes: int,
    num_envs: int,
    match: str,
) -> None:
    """Refused at construction, not when the final eval starts after training."""
    config = tiny_train_step()
    config.evaluation.num_episodes = num_episodes
    env = config.evaluation.env = tiny_env()
    env.num_envs = num_envs
    with pytest.raises(ValueError, match=match):
        config.make()


def test_a_world_model_feature_trains_through_the_pipeline_and_its_evaluation() -> None:
    """The learner reads the stored features; the evaluation steps engines of its own."""
    config = _quick(tiny_train_step())
    config.train_budget_steps = 2
    config.feature = _smoke_feature()
    model = config.model
    assert isinstance(model, MinGRUPolicy.Config)
    proj = model.proj_feature = Linear.Config()
    proj.channels_in = 36
    step = _train_step(config.copy_tree().finalize())
    try:
        # The policy refuses a window without its feature, so an epoch that
        # learns is one whose windows read the store.
        result = step.train_step()
        assert "metrics" in result
        metrics = result["metrics"]
        assert step.rollout.slots[step.ready].features is not None
        assert float(metrics["feature/rms"]) > 0
        assert step.train_loss()["model"].isfinite().all()
        evaluator = step.make_evaluator()
        try:
            evaluator.collect()
            engines = evaluator._rollout.engines
            assert engines
            assert all(
                isinstance(engine, FeatureEngine) and engine.keys.numel()
                for engine in engines
            )
        finally:
            evaluator.close()
        assert step.feature is not None
        # The evaluation's engines are its own; training's still hold their cache.
        assert all(
            isinstance(engine, FeatureEngine) and engine.keys.numel()
            for engine in step.rollout.engines
        )
    finally:
        step.close()


def test_a_feature_is_refused_the_symbolic_view() -> None:
    """Refused before anything is built, the world model's weights included."""
    config = tiny_train_step()
    config.feature = _smoke_feature()
    config.env.rules.symbolic_observation = True
    model = config.model = MinGRUPolicy.Config()
    model.embedding = DenseObservation.Config()
    with pytest.raises(ValueError, match="symbolic view is refused"):
        config.make()


def test_the_step_accepts_a_feature_with_practice() -> None:
    """The recipe check passes practice restores beside a feature: the rollout owns their history."""
    config = tiny_train_step()
    config.feature = _smoke_feature()
    config.env.practice = FrontierPractice.Config()
    _check_recipe(config)


def test_the_joint_gradient_is_autograds_through_each_steps_own_context() -> None:
    """Two phases equal one autograd graph through the per-step training forward.

    ``learn_joint_minibatch`` recomputes the features, learns the policy from
    them as a leaf and hands their gradient to the world model's backward,
    measuring them against the stored ones, which it does not read. The
    reference is one graph: each step's context alone through the world
    model's training forward, the policy's loss, one backward. The policy
    reads the replay's values in both, the reference's carrying the training
    forward's graph, so both take the loss's gradient at one point. The loss
    is not smooth in the features: for 5 or 6 of these 24 samples inside the
    value clip, rounding alone picks its branch, and features a rounding apart,
    as the replay's and the training forward's are, moved the policy's
    gradients by up to 16% of their largest on x86 (measured). The
    policy's gradients are then equal, and every world-model gradient agrees
    within 1e-5 of its largest entry. Contexts restart (row 0) and slide (rows
    1 and 2).
    """
    config = tiny_policy(dtype=torch.float32)
    proj = config.proj_feature = Linear.Config()
    proj.channels_in = 36  # The tiny world model's width.
    policy = config.make()
    reference_policy = copy.deepcopy(policy)
    model = tiny_model(small_schema(), global_layers=2)
    reference_model = copy.deepcopy(model).requires_grad_()
    contexts = random_contexts(
        torch.tensor(
            ((4, 5, 6, 4, 5, 6, 1, 2), (4,) * 8, (1, 2, 3, 4, 4, 4, 4, 4)),
        ),
        torch.tensor(
            ((0, 0, 0, 0, 0, 0, 1, 1), (0,) * 8, (1, 1, 1, 1, 0, 0, 0, 0)),
        ).bool(),
        schema=small_schema(),
        slots=6,
        counts=torch.tensor((5, 6, 0)),
        seed=3,
    )
    buffers = list(_rollout(config, agents=3, horizon=8))
    buffers[7] = buffers[7].float()
    traced = context_reference(reference_model, contexts, layers=2)
    # Scaled away from the replay's: the gradient must not read them; the gap must.
    stored = traced.detach() * 1.25
    minibatch = LearnerRollout.from_time_major(
        *buffers,
        reward_scale=1.0,
        reward_clip=1.0,
        features=stored.transpose(0, 1),
        contexts=contexts,
    )
    joint = JointWorldModel(model, layers=2, replay=_small_bins().make())
    _, _, gap = learn_joint_minibatch(policy, _objective(), minibatch, joint)
    # The stored features are scaled, so the gap is 1 - 1 / 1.25 of them.
    assert float(gap) == pytest.approx(0.2, abs=1e-6)
    replayed = joint.forward(contexts).features
    # The replay's values exactly, ``traced - traced`` being 0, and the training
    # forward's gradient.
    features = replayed + (traced - traced.detach())
    learn_minibatch(
        reference_policy,
        _objective(),
        replace(minibatch, features=features),
    )
    for (name, parameter), reference in zip(
        policy.named_parameters(),
        reference_policy.parameters(),
        strict=True,
    ):
        assert parameter.grad is not None, name
        assert reference.grad is not None, name
        assert torch.equal(parameter.grad, reference.grad), name
    pairs = [
        (name, leaf.grad, reference_model.get_parameter(name).grad)
        for name, leaf in zip(joint.weights(), joint.parameters(), strict=True)
    ]
    for name, got, want in pairs:
        assert got is not None, name
        assert want is not None, name
        torch.testing.assert_close(
            got.view(want.shape),
            want,
            rtol=0,
            atol=1e-5 * float(want.abs().max()),
            msg=name,
        )


def test_a_joint_step_optimizes_the_world_model_after_the_policy() -> None:
    """One group: the policy's parameters, then the trained world model's."""
    step = _joint_step()
    try:
        assert step.joint is not None
        held = [
            id(parameter)
            for group in step.optimizer.param_groups
            for parameter in cast("list[Tensor]", group["params"])
        ]
        assert held == [
            id(parameter)
            for parameter in (*step.model.parameters(), *step.joint.parameters())
        ]
        assert isinstance(step.feature, WorldModelFeature)
        assert step.joint.source is step.feature.model
        assert step.joint.guard is CAPTURE_LOCK
        assert not any(p.requires_grad for p in step.feature.model.parameters())
    finally:
        step.close()


def test_a_joint_steps_evaluation_shares_its_weights_and_stores_nothing() -> None:
    """No learner reads an evaluation: no frames in its engines, no inputs in its slot."""
    step = _joint_step()
    try:
        evaluator = step.make_evaluator()
        try:
            rollout = evaluator._rollout
            assert rollout.slots[0].prefix_cells is None
            assert rollout.slots[0].frame_cells is None
            assert rollout.engines
            for engine in rollout.engines:
                assert isinstance(engine, FeatureEngine)
                assert engine.cell_ring is None
                assert engine.ring == 4
            assert isinstance(rollout.feature, WorldModelFeature)
            assert isinstance(step.feature, WorldModelFeature)
            assert rollout.feature.model is step.feature.model
            assert step.feature.joint
        finally:
            evaluator.close()
    finally:
        step.close()


def test_a_joint_epoch_trains_both_and_its_checkpoint_carries_the_world_model() -> None:
    """The windows learn from contexts; reloaded after it moves on, the step holds them.

    The rollout is drawn, its contexts as the store will hold them, rather than
    played: the actor's side is ``rollout_test``'s. The source the actor reads
    keeps its weights until a publication, and a load publishes: after the
    step's trained weights, their publication and Muon's state all move, the
    checkpoint brings back each.
    """
    step = _joint_step()
    try:
        joint = step.joint
        assert joint is not None
        model = step.config.model
        assert isinstance(model, MinGRUPolicy.Config)
        agents, horizon = step.env.num_envs, step.config.rollout.horizon
        features = torch.randn(
            horizon,
            agents,
            36,
            generator=torch.Generator().manual_seed(6),
        )
        rollout = LearnerRollout.from_time_major(
            *_rollout(model, agents=agents, horizon=horizon),
            reward_scale=1.0,
            reward_clip=1.0,
            features=features.bfloat16(),
            contexts=random_contexts(
                torch.tensor(((2, 3, 4, 1), (1, 2, 1, 2))),
                torch.tensor(((0, 0, 0, 1), (1, 1, 1, 1))).bool(),
                schema=craftax_schema(),
                slots=3,
                counts=torch.tensor((1, 0)),
                seed=5,
            ),
        )
        before = {name: weight.clone() for name, weight in joint.weights().items()}
        policy = [parameter.detach().clone() for parameter in step.model.parameters()]
        losses, _ = step.learner(step, rollout)
        assert bool(losses.isfinite().all())
        assert not all(
            torch.equal(weight, before[name])
            for name, weight in joint.weights().items()
        )
        assert not all(
            torch.equal(parameter, old)
            for parameter, old in zip(step.model.parameters(), policy, strict=True)
        )
        for name, weight in before.items():
            assert torch.equal(joint.source.get_parameter(name), weight), name
        total, _ = step.learner.loss(step, rollout)
        assert bool(total.isfinite())
        with pytest.raises(ValueError, match="stored context"):
            step.learner(step, replace(rollout, contexts=None))
        saved = io.BytesIO()
        torch.save(step.state_dict(), saved)
        trained = {name: weight.clone() for name, weight in joint.weights().items()}
        optimizer = _masters_and_momentum(step)
        muon = step.optimizer
        assert isinstance(muon, FusedMuon)
        with torch.no_grad():
            for value in (
                *joint.parameters(),
                *joint.source.parameters(),
                *muon.master_weights,
                *muon.momentum_buffers,
            ):
                value.add_(1)
        step.load_state_dict(_read(saved))
        for name, weight in joint.weights().items():
            assert torch.equal(weight, trained[name]), name
            assert torch.equal(joint.source.get_parameter(name), trained[name]), name
        for ours, theirs in zip(_masters_and_momentum(step), optimizer, strict=True):
            assert torch.equal(ours, theirs)
    finally:
        step.close()


def test_a_joint_and_a_frozen_step_refuse_each_others_checkpoints() -> None:
    """The trained world model is in the one checkpoint and not the other."""
    joint, frozen = _joint_step(), _train_step()
    try:
        with pytest.raises(ValueError, match="only there"):
            frozen.load_state_dict(joint.state_dict())
        with pytest.raises(ValueError, match="only there"):
            joint.load_state_dict(frozen.state_dict())
    finally:
        for step in (joint, frozen):
            step.close()


class _AgreementWindows(AgentWindows):
    """PufferLib's windows, measuring first the replayed features against the stored."""

    class Config(Makes["_AgreementWindows"], AgentWindows.Config):
        """PufferLib's windows' config."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.gaps: list[float] = []

    @override
    def __call__(
        self,
        step: CraftaxTrainStep,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Record the largest gap between the replay and the store, then learn."""
        assert step.joint is not None
        assert rollout.contexts is not None
        assert rollout.features is not None
        replayed = step.joint.forward(rollout.contexts).features
        self.gaps.append(float((replayed - rollout.features).abs().max()))
        return super().__call__(step, rollout)


@pytest.mark.parametrize("history", ["refill", "sliding"])
def test_a_joint_learner_replays_the_features_its_actor_read_at_the_same_weights(
    history: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every epoch's stored contexts replay to the stored features, after a rebuild.

    One slot, so each epoch learns from a rollout of the weights it starts
    with: published into the source, the actor's histories rebuilt under them
    for the second epoch. Within ``feature_test``'s 2e-6 of the engine's
    training forward, in float32 with masked attention, for both histories.
    Two agents, one a buffer, from a pool of one world: each replays the full
    schema's frames.
    """
    config = tiny_train_step()
    config.env.num_envs = 2
    _one_world(config)
    config = _quick(config)
    config.train_budget_steps = 2
    feature = config.feature = _smoke_feature()
    if history == "sliding":
        window = feature.history = Sliding.Config()
        window.decisions = 4
    model = config.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.dtype = model.state_dtype = model.output_dtype = torch.float32
    proj = model.proj_feature = Linear.Config()
    proj.channels_in = 36  # The smoke world model's width.
    config.feature_training = _small_bins()
    config.learner = _AgreementWindows.Config().update(
        config.learner,
        skip_missing=True,
    )
    config.rollout.num_slots = 1
    config.rollout.dtype = torch.float32
    step = _train_step(config)
    try:
        assert step.joint is not None
        rebuild = step.rollout.rebuild_features
        calls: list[int] = []
        monkeypatch.setattr(
            step.rollout,
            "rebuild_features",
            lambda: (calls.append(step.global_step), rebuild())[1],
        )
        before = {name: w.clone() for name, w in step.joint.weights().items()}
        results = [step.train_step() for _ in range(2)]
        learner = step.learner
        assert isinstance(learner, _AgreementWindows)
        assert len(learner.gaps) == 2
        assert max(learner.gaps) <= 2e-6, learner.gaps
        reported = [float(r.get("metrics", {})["joint/feature_gap"]) for r in results]
        assert max(reported) <= 1e-5, reported
        # A rebuild before the rollout after the first epoch's learning.
        assert calls == [1]
        rebuilds = [float(r.get("metrics", {})["rebuild_seconds"]) for r in results]
        assert rebuilds[0] == 0.0
        assert rebuilds[1] > 0
        assert not all(
            torch.equal(weight, before[name])
            for name, weight in step.joint.weights().items()
        )
    finally:
        step.close()


def test_a_feature_is_joint_exactly_when_the_learner_trains_it() -> None:
    """A joint feature no learner trains would store inputs nothing reads."""
    config = tiny_train_step()
    config.feature = _smoke_feature()
    config.feature.joint = True
    frozen = config.copy_tree().finalize().feature
    assert isinstance(frozen, WorldModelFeature.Config)
    assert not frozen.joint
    config.feature_training = ContextReplay.Config()
    final = config.copy_tree().finalize()
    assert isinstance(final.feature, WorldModelFeature.Config)
    assert final.feature.joint


def test_the_evaluation_reads_the_trained_world_model() -> None:
    step = _joint_step()
    try:
        assert step.joint is not None
        with torch.no_grad():
            for leaf in step.joint.parameters():
                leaf.add_(1)
        evaluator = step.make_evaluator()
        try:
            for name, weight in step.joint.weights().items():
                assert torch.equal(step.joint.source.get_parameter(name), weight), name
        finally:
            evaluator.close()
    finally:
        step.close()


@pytest.mark.parametrize("missing", ["feature", "windows"])
def test_feature_training_needs_a_world_model_feature_and_whole_windows(
    missing: str,
) -> None:
    """Refused before anything is built."""
    config = tiny_train_step()
    config.feature_training = ContextReplay.Config()
    if missing == "windows":
        config.feature = _smoke_feature()
        config.learner = ShuffledTransitions.Config()
    match = "learner must be AgentWindows" if missing == "windows" else "reads no such"
    with pytest.raises(ValueError, match=match):
        config.make()


def _joint_step() -> CraftaxTrainStep:
    """Return the tiny step on 2 agents, training the smoke world model's feature with its policy."""
    config = tiny_train_step()
    config.env.num_envs = 2
    _one_world(config)
    config.feature = _smoke_feature()
    model = config.model
    assert isinstance(model, MinGRUPolicy.Config)
    proj = model.proj_feature = Linear.Config()
    proj.channels_in = 36  # The smoke world model's width.
    config.feature_training = _small_bins()
    return _train_step(config)


# A pool's world takes 0.7 ms to generate on x86; these steps' tests read none apart.
def _one_world(config: CraftaxTrainStep.Config) -> None:
    """Shrink the step's world pool to one world."""
    pool = config.env.restart
    assert isinstance(pool, WorldPool.Config)
    pool.num_worlds = 1


def _smoke_feature() -> WorldModelFeature.Config:
    """Return ``smoke_feature`` whose weights are copies of one build of its model."""
    config = smoke_feature()
    config.weights = _SmokeWeights.Config()
    return config


class _SmokeWeights:
    """``smoke_feature``'s world model, built once a process and copied for each source.

    The build finalizes ``exp_smoke``'s whole config and draws its ~300 weights:
    30 ms on x86, a third of a test that makes a source. ``feature_test`` checks
    ``InitialWeights`` itself.
    """

    class Config(Fig["_SmokeWeights"]):
        """Nothing to configure: the weights are ``smoke_feature``'s."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self) -> WorldModel:
        """Return a copy of the model, at the seeded init and in eval mode."""
        return copy.deepcopy(_smoke_model())


@cache
def _smoke_model() -> WorldModel:
    """Build ``smoke_feature``'s world model."""
    return smoke_feature().weights.make()()


# The default micro-batches, 16,384 tokens of passes and 512 frames, are padded full:
# at these sizes nearly all of a replay's work.
def _small_bins() -> ContextReplay.Config:
    """Return the replay in micro-batches as small as tiny windows want."""
    config = ContextReplay.Config()
    config.bin_tokens = 1
    config.pass_tokens = 26
    config.frames_per_batch = 5
    return config


def test_a_checkpoint_needs_an_optimizer_that_keeps_masters(tmp_path: Path) -> None:
    config = tiny_train_step()
    config.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
    config.checkpoint = tmp_path / "init.pt"
    with pytest.raises(TypeError, match="masters"):
        config.make()


def test_the_step_refuses_a_geometry_its_rule_cannot_learn() -> None:
    config = tiny_train_step()
    windows = config.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.objective = TritonPPO.Config()
    assert config.rollout.horizon == 4
    with pytest.raises(ValueError, match="multiple of 8"):
        config.make()
    windows.objective = TorchPPO.Config()
    config.rollout.num_slots = 3
    with pytest.raises(ValueError, match="num_slots"):
        config.make()


def test_the_step_refuses_a_first_stage_that_does_not_read_the_env() -> None:
    """Refused from the configs alone, before the environments are built."""
    config = tiny_train_step()
    assert isinstance(config.model, MinGRUPolicy.Config)
    config.model.embedding = DenseObservation.Config()
    # Buffers for 2**40 environments cannot be allocated: building the env
    # first would fail on that, not on the width.
    config.env.num_envs = 2**40
    with pytest.raises(ValueError, match="843-float"):
        config.make()


def test_an_explicit_evaluation_env_sampler_and_horizon_survive_finalize() -> None:
    """Finalize fills only what the evaluation leaves unset (CFG-6).

    Measured on the base: exp000's step finalized an evaluation of 64
    environments into 2,048, its sampler seed 5 into 73 and its horizon 64
    into 256.
    """
    config = exp000().step
    env = config.evaluation.env = tiny_env()
    env.num_envs = 64
    sampler = config.evaluation.sampler = PhiloxSampler.Config()
    sampler.seed = 5
    rollout = config.evaluation.rollout = Rollout.Config()
    rollout.num_slots = 1
    rollout.horizon = 64
    final = config.copy_tree().finalize().evaluation
    assert final.env is not None
    assert final.env.num_envs == 64
    assert isinstance(final.sampler, PhiloxSampler.Config)
    assert final.sampler.seed == 5
    assert final.rollout is not None
    assert final.rollout.horizon == 64


def test_the_evaluation_copies_training_env_less_its_training_only_options() -> None:
    config = tiny_train_step()
    config.env.stall_cap = StallCap.Config()
    config.env.practice = FrontierPractice.Config()
    config.env.rules.end_on_boss_defeat = True
    final = config.copy_tree().finalize()
    assert final.evaluation.env is not None
    assert final.evaluation.env.stall_cap is None
    assert final.evaluation.env.practice is None
    assert final.evaluation.env.rules.end_on_boss_defeat
    assert final.env.stall_cap is not None
    assert final.env.practice is not None


@pytest.mark.parametrize(
    "name",
    [
        "loss",
        "lr_schedule",
        "compile",
        "ema",
        "gradient_clip_norm",
        "accumulate_grad_batches",
        "dtype_autocast",
        "skip_step_on_nonfinite_grad",
        "objective",
        "minibatch_size",
        "replay_ratio",
    ],
)
def test_the_step_has_no_field_it_would_ignore(name: str) -> None:
    """A knob the step itself never reads is not in its config (CFG-7, CFG-8).

    On the base a custom ``loss`` built and was silently ignored, and the
    printed exp000 named a constant schedule and a BCE loss it never ran. The
    learning rule and the minibatches are the learner's own, so exp003's
    learner no longer carries exp000's unread PPO coefficients.
    """
    config = tiny_train_step()
    with pytest.raises(AttributeError):
        setattr(config, name, None)


def test_the_step_anneals_by_its_schedule_slot() -> None:
    """The rate each epoch comes from ``schedule``, called with the configured rate."""
    config = _quick(tiny_train_step())
    config.schedule = PartialConfig(_half_rate)
    step = _train_step(config)
    try:
        result = step.train_step()
        base = from_plain(
            cast("object", step.optimizer.param_groups[0]["initial_lr"]),
            float,
        )
    finally:
        step.close()
    assert "metrics" in result
    assert result["metrics"]["learning_rate"] == base / 2


def test_each_epoch_refills_the_one_rate_tensor_the_optimizer_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Epoch k's learner reads the schedule's rate after k - 1 epochs, written in place.

    A captured learner graph reads the tensor it was captured with, so a new
    tensor each epoch would replay every captured epoch at its capture's rate.
    The schedule is exp000's fp32 cosine over the budget, whose rate at every
    epoch the rates golden freezes.
    """
    config = tiny_train_step()
    config.train_budget_steps = 6662
    step = _train_step(config)
    try:
        (group,) = step.optimizer.param_groups
        read: list[Tensor] = []

        def learn(storage: RolloutStorage) -> tuple[Tensor, dict[str, Tensor]]:
            del storage
            read.append(from_plain(cast("object", group["lr"]), Tensor))
            return torch.zeros(len(TorchPPO.Config.LOSS_NAMES)), {}

        monkeypatch.setattr(step, "_learn_epoch", learn)
        for epoch in (1, 200):
            while step.global_step < epoch - 1:
                with step.timer_step:
                    pass
            _, _, rate = step._train_epoch(step.ready)
            assert rate == cosine_annealing_fp32(
                from_plain(cast("object", group["initial_lr"]), float),
                0.0,
                epoch - 1,
                6662,
            )
            assert float(read[-1]) == rate
    finally:
        step.close()
    assert read[0] is read[1]


def test_the_default_precision_learns_from_fp32_storage_and_an_fp32_carry() -> None:
    """The class defaults' precision end to end: fp32 carries, values and rollout rows."""
    config = _quick(tiny_train_step())
    model = config.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.state_dtype = model.output_dtype = config.rollout.dtype = torch.float32
    step = _train_step(config)
    try:
        result = step.train_step()
        storage = step.rollout.slots[step.ready]
    finally:
        step.close()
    for value in (storage.initial_states, storage.logprobs, storage.rewards):
        assert value.dtype == torch.float32
    # The fp32 decoder's values, stored unrounded.
    assert not torch.equal(storage.values, storage.values.bfloat16().float())
    assert storage.observations.dtype == torch.bfloat16
    assert bool(torch.isfinite(result["model"]).all())


def test_the_step_trains_with_an_optimizer_other_than_fused_muon() -> None:
    """The optimizer slot takes any optimizer the learner can drive (CFG-8)."""
    config = _quick(tiny_train_step())
    config.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
    step = _train_step(config)
    try:
        before = [p.detach().clone() for p in step.model.parameters()]
        step.train_step()
        after = list(step.model.parameters())
    finally:
        step.close()
    assert isinstance(step.optimizer, torch.optim.SGD)
    assert not all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


class _LinearPolicy(nn.Module):
    """A stateless policy: one projection from the observation to the fused row."""

    class Config(Fig["_LinearPolicy"]):
        """The observation's width."""

        observation_size: int = OBS_SIZE

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.head = nn.Linear(config.observation_size, ATN_DIM + 1, bias=False)

    def initial_state(
        self,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Return an empty carry: nothing is threaded between steps."""
        # No layers and no width, as a feed-forward policy's carry.
        return torch.zeros(0, num_envs, 0, device=device)

    def forward_fused(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor | None,
        *,
        carry: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Score one step."""
        del episode_start, features
        return self(observations), state if carry is None else carry

    def forward_sequence(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor,
        *,
        actions: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Score a window; the auxiliary loss is the mean of the head's weight."""
        del episode_start, actions, features
        return self(observations), state, self.head.weight.float().mean()

    @override
    def forward(self, observations: Tensor) -> Tensor:
        return self.head(observations.to(self.dtype))


def test_the_model_slot_takes_any_policy_the_actor_and_learner_can_call() -> None:
    """A policy that is not a MinGRU, without a carry, trains through the pipeline."""
    config = tiny_train_step()
    config.model = _LinearPolicy.Config()
    step = _train_step(config)
    try:
        assert isinstance(step.model, _LinearPolicy)
        before = [p.detach().clone() for p in step.model.parameters()]
        result = step.train_step()
        after = list(step.model.parameters())
    finally:
        step.close()
    assert bool(torch.isfinite(result["model"]).all())
    assert not all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


class _FreshnessLearner:
    """PufferLib's windows, recording first whether the epoch's data is its weights'."""

    class Config(Fig["_FreshnessLearner"]):
        """Nothing to configure."""

    def __init__(self, config: Config) -> None:
        del config
        windows = AgentWindows.Config()
        windows.objective = TorchPPO.Config()
        windows.minibatch_size = 8
        self.windows = AgentWindows(windows)
        self.fresh: list[bool] = []

    def prepare(self, config: CraftaxTrainStep.Config) -> None:
        """Place PufferLib's windows."""
        self.windows.prepare(config)

    def begin_epoch(self, step: CraftaxTrainStep, epoch: int) -> None:
        """Nothing to prepare."""
        del step, epoch

    def loss(
        self,
        step: CraftaxTrainStep,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, Tensor]:
        """Score as PufferLib's windows do."""
        return self.windows.loss(step, rollout)

    def state_dict(self) -> dict[str, Tensor]:
        """Return PufferLib's windows' state."""
        return self.windows.state_dict()

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore PufferLib's windows' state."""
        self.windows.load_state_dict(state)

    def __call__(
        self,
        step: CraftaxTrainStep,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compare the rollout's values with the policy's now, then learn."""
        with torch.no_grad():
            decoded, _, _ = step.model.forward_sequence(
                rollout.observations,
                rollout.initial_states,
                rollout.terminals,
            )
        self.fresh.append(
            bool(
                torch.allclose(decoded[..., -1], rollout.values, rtol=1e-5, atol=1e-6),
            ),
        )
        return self.windows(step, rollout)


@pytest.mark.parametrize(("num_slots", "fresh"), [(1, True), (2, False)])
def test_one_slot_learns_from_a_rollout_of_the_weights_it_updates(
    num_slots: int,
    fresh: bool,
) -> None:
    """One slot collects with the current weights; two learn one epoch stale.

    From the second epoch on, a prefetching pipeline's data is the previous
    weights'; a single slot's is always the weights the epoch starts from, and
    every epoch times its rollout.
    """
    config = _quick(tiny_train_step())
    config.model = _LinearPolicy.Config()
    config.learner = _FreshnessLearner.Config()
    # The fp32 policy's values, stored unrounded, so a fresh rollout's match it.
    config.rollout.dtype = torch.float32
    config.rollout.num_slots = num_slots
    step = _train_step(config)
    try:
        results = [step.train_step() for _ in range(3)]
        learner = step.learner
        assert isinstance(learner, _FreshnessLearner)
        assert learner.fresh == [True, fresh, fresh]
        assert len(step.rollout.slots) == num_slots
        # Three rollouts either way: a single slot boots none, a pipeline's last
        # epoch prefetches none.
        horizon = config.rollout.horizon
        assert all(
            int(graph.draws[0]) == 3 * horizon for graph in step.rollout.graphs[0]
        )
        if num_slots == 1:
            assert step.ready == 0
            for result in results:
                assert "metrics" in result
                assert float(result["metrics"]["rollout_seconds"]) > 0.0
    finally:
        step.close()


def test_the_default_learner_is_pufferlibs_windows() -> None:
    assert isinstance(CraftaxTrainStep.Config().learner, AgentWindows.Config)


def test_an_infinite_reward_clip_leaves_the_rewards_as_they_are() -> None:
    config = tiny_policy()
    buffers = _rollout(config, agents=4, horizon=5)
    rollout = LearnerRollout.from_time_major(
        *buffers,
        reward_scale=1.0,
        reward_clip=math.inf,
    )
    assert torch.equal(rollout.rewards, buffers[3].transpose(0, 1))
    assert rollout.rewards.abs().max() > 1


def test_the_default_schedule_is_primls_cosine_of_the_progress() -> None:
    schedule = CraftaxTrainStep.Config().schedule.make()
    assert schedule(0.5, step=0, total_steps=8) == 0.5
    assert schedule(0.5, step=4, total_steps=8) == pytest.approx(0.25)
    assert schedule(0.5, step=8, total_steps=8) == 0.0


def test_the_fp32_cosine_runs_from_the_fp32_base_to_the_minimum() -> None:
    base = 0.00472887093
    single = _single(base)
    assert cosine_annealing_fp32(base, 0.0, 0, 6662) == single
    # cos(pi / 2) is 6e-17, which fp32 rounds away: exactly half the base.
    assert cosine_annealing_fp32(base, 0.0, 3331, 6662) == single / 2
    assert cosine_annealing_fp32(base, 0.0, 6662, 6662) == 0.0
    assert cosine_annealing_fp32(1.0, 0.25, 8, 8) == 0.25


def test_the_fp32_cosine_is_not_the_default_schedule_rounded() -> None:
    # Over exp000's 6,662 epochs the two differ by an fp32 ulp at 2,836 of the
    # 6,663 boundaries (measured on macOS libm), the first at epoch 5.
    base = 0.00472887093
    standard = ProgressSchedule(ProgressSchedule.Config())
    ours = cosine_annealing_fp32(base, 0.0, 5, 6662)
    assert ours != _single(standard(base, step=5, total_steps=6662))


def _single(value: float) -> float:
    """Round to the nearest fp32."""
    return float(np.float32(value))


def _half_rate(base: float, *, step: int, total_steps: int) -> float:
    """Return half the configured rate at every epoch: a schedule unlike the cosine."""
    del step, total_steps
    return base / 2


def test_the_tiny_steps_epoch_matches_its_golden(tmp_path: Path) -> None:
    """The tiny step's epoch from portable seed-73 masters, frozen on every host.

    The whole pipeline in its torch forms: the boot rollout, the learner's
    window of every agent, Muon, and exp000's fp32 cosine at its base rate:
    the epoch's rate, mean losses and masters. One epoch of two-step rollouts,
    as host-agnostic numerics cost the pipeline 0.1 s an epoch on x86; the
    rollouts the epochs after it learn from are ``rollout_test``'s golden, and
    a resume and a reload are the tests above and below.
    """
    config = _quick(tiny_train_step())
    config.train_budget_steps = 1
    with host_agnostic_pipeline():
        lines = _train_out(_from_portable_masters(config, tmp_path))
    assert_golden(test_file=__file__, name="train_step_tiny", lines=lines)


def test_a_checkpoint_loaded_into_a_step_that_moved_on_trains_as_it_did() -> None:
    """Epoch 1's checkpoint, loaded into the step after epoch 3: epoch 2 runs again.

    The load lands on a step whose optimizer state, slots, carries and
    environments have all moved on; the epoch after it must be epoch 2 as it
    ran: its losses, Muon's state, and the rollout it prefetched. On the GPU
    the load also drops captured learner epochs, which the GPU test below pins.
    """
    config = _quick(tiny_train_step())
    config.train_budget_steps = 4
    step = _train_step(config)
    try:
        step.train_step()
        saved = io.BytesIO()
        torch.save(step.state_dict(), saved)
        expected: list[Tensor] = [step.train_step()["model"]]
        expected += _masters_and_momentum(step)
        prefetched = step.ready
        # The third epoch learns from the second's rollout, so it is whole by now.
        step.train_step()
        expected += _slot(step, prefetched)
        step.load_state_dict(_read(saved))
        actual: list[Tensor] = [step.train_step()["model"]]
        actual += _masters_and_momentum(step)
    finally:
        step.close()
    actual += _prefetched(step)
    assert len(actual) == len(expected)
    for index, (ours, theirs) in enumerate(zip(actual, expected, strict=True)):
        assert torch.equal(ours, theirs), index


@pytest.mark.gpu_triton
def test_exp000s_kernels_match_their_tiny_golden_on_the_gpu(
    tmp_path: Path,
) -> None:
    """exp000's step at test size from portable seed-73 masters, frozen on the GPU.

    exp000's CUDA kernels throughout: the exact scan, sampler, learning rule
    and Muon, the rollout's step graphs and the env's ``nogil`` loop. Epoch 1
    learns eagerly, epochs 2 and 3 capture each slot's learner epoch, and
    epoch 4 replays one.
    """
    host = gpu_key()
    require_golden(
        test_file=__file__,
        name="train_step_exp000_tiny",
        host=host,
    )
    lines = _train_out(_from_portable_masters(tiny_exp000_step(), tmp_path))
    assert_golden(
        test_file=__file__,
        name="train_step_exp000_tiny",
        lines=lines,
        host=host,
    )


@pytest.mark.gpu_triton
def test_a_checkpoint_loaded_after_the_capture_trains_as_the_golden(
    tmp_path: Path,
) -> None:
    """Epoch 2's checkpoint, loaded into the step that went on to epoch 4, then 3 and 4.

    By epoch 4 both slots' learner epochs are captured graphs, and the load
    puts new tensors in the optimizer's state; epochs 3 and 4 must still be
    the uninterrupted golden's.
    """
    golden = read_golden(
        test_file=__file__,
        name="train_step_exp000_tiny",
        host=gpu_key(),
    )
    step = _from_portable_masters(tiny_exp000_step(), tmp_path)
    lines = _reloaded_epochs(step, tmp_path / "checkpoint.pt")
    assert lines == _after_epoch_2(golden)


def test_exp002s_epoch_matches_its_golden() -> None:
    """exp002 at test size from the policy's own init under torch seed 0, frozen.

    From scratch, so the golden moves with the init's draws: their order,
    their distributions, or torch's generator. The env plays exp002's rules
    in fresh worlds and writes the dense symbolic view. One epoch of two-step
    rollouts, as the tiny step's golden above.
    """
    config = _quick(tiny_exp002_step())
    config.train_budget_steps = 1
    with host_agnostic_pipeline():
        step = _train_step(config)
        lines = ["# from-scratch: the policy's own init under torch seed 0"]
        lines += _train_out(step)
    assert_golden(
        test_file=__file__,
        name="train_step_exp002_tiny",
        lines=lines,
    )


@pytest.mark.gpu_triton
def test_exp000s_muon_matches_its_tiny_golden_on_the_gpu(
    tmp_path: Path,
) -> None:
    """exp000's optimizer over its policy at test size, from portable seed-73 masters.

    Three steps of the fused kernels with the rate as a device tensor, as the
    learner runs them; the first and last are clipped, the second is not.
    """
    host = gpu_key()
    require_golden(test_file=__file__, name="muon_exp000_tiny", host=host)
    config = tiny_exp000_step()
    assert isinstance(config.model, MinGRUPolicy.Config)
    model = config.model.make().cuda()
    optimizer = config.optimizer.make()(model.parameters())
    assert isinstance(optimizer, FusedMuon)
    load_masters(
        model,
        optimizer,
        portable_checkpoint(config.model, tmp_path / "masters.pt", seed=73),
    )
    for group in optimizer.param_groups:
        group["lr"] = torch.tensor(group["lr"], dtype=torch.float32, device="cuda")
    gradients = _gradients(model, scales=(2**-5, 2**-9, 2**-5))
    lines = _muon_entries(model, optimizer, gradients)
    assert_golden(
        test_file=__file__,
        name="muon_exp000_tiny",
        lines=lines,
        host=host,
    )


def test_exp000s_learning_rates_match_their_golden() -> None:
    """The fp32 cosine at every one of exp000's 6,663 epoch boundaries, frozen.

    ``cos`` comes from the platform's libm in double, rounded once to fp32:
    macOS's and glibc 2.35's and 2.39's land on the same rates, so one golden
    holds on each.
    """
    rates = np.array(
        [
            cosine_annealing_fp32(0.00472887093, 0.0, epoch, 6662)
            for epoch in range(6663)
        ],
        dtype=np.float32,
    )
    lines = [f"rates {digest(rates)}"]
    lines += [
        f"epoch {epoch} {fp32(rates.item(epoch))}" for epoch in (0, 1, 5, 3331, 6661)
    ]
    assert_golden(test_file=__file__, name="muon_exp000_rates", lines=lines)


# The step loads them as exp000 does, from its ``checkpoint``.
def _from_portable_masters(
    config: CraftaxTrainStep.Config,
    tmp_path: Path,
) -> CraftaxTrainStep:
    """Build ``config``'s step, started from portable seed-73 masters."""
    assert isinstance(config.model, MinGRUPolicy.Config)
    config.checkpoint = str(
        portable_checkpoint(config.model, tmp_path / "masters.pt", seed=73),
    )
    return _train_step(config)


def _train_out(step: CraftaxTrainStep, *, checkpoint: Path | None = None) -> list[str]:
    """Load ``checkpoint`` if given, train to the budget and close; digest each epoch."""
    try:
        if checkpoint is not None:
            step.load_state_dict(
                from_plain(
                    cast("object", torch.load(checkpoint, weights_only=True)),
                    dict[str, object],
                ),
            )
        return _epochs(step)
    finally:
        step.close()


def _reloaded_epochs(step: CraftaxTrainStep, checkpoint: Path) -> list[str]:
    """Train 2 epochs, save, train to the budget, load the save, train again; close."""
    try:
        for _ in range(2):
            step.train_step()
        torch.save(step.state_dict(), checkpoint)
        while step.global_step < step.total_steps:
            step.train_step()
        step.load_state_dict(
            from_plain(
                cast("object", torch.load(checkpoint, weights_only=True)),
                dict[str, object],
            ),
        )
        return _epochs(step)
    finally:
        step.close()


def _epochs(step: CraftaxTrainStep) -> list[str]:
    """Train to the budget; digest each epoch."""
    lines: list[str] = []
    while step.global_step < step.total_steps:
        lines += _epoch_entries(step, step.train_step())
    return lines


def _after_epoch_2(golden: list[str]) -> list[str]:
    """Return the golden's lines of the epochs after the second."""
    return [line for line in golden if int(line.split()[1]) > 2]


def _epoch_entries(step: CraftaxTrainStep, result: TrainStepOutput) -> list[str]:
    """Digest one epoch: its rate, its mean loss terms, then every master."""
    prefix = f"epoch {step.global_step:04d}"
    metrics = result.get("metrics", {})
    lines = [f"{prefix} learning_rate {fp32(metrics['learning_rate'])}"]
    lines += [
        f"{prefix} {name} {fp32(value)}"
        for name, value in zip(TorchPPO.Config.LOSS_NAMES, result["model"], strict=True)
    ]
    return lines + [
        f"{prefix} master {name} {digest(value)}"
        for name, value in optimizer_state(
            step.model,
            step.optimizer,
            "master_weight",
        ).items()
    ]


def _gradients(
    model: nn.Module,
    *,
    scales: tuple[float, ...],
) -> list[dict[str, Tensor]]:
    """Draw one portable bf16 gradient per parameter per step, ``U(+-scale)``."""
    generator = torch.Generator().manual_seed(1)
    return [
        {
            name: portable_uniform(
                *parameter.shape,
                bound=scale,
                generator=generator,
            ).bfloat16()
            for name, parameter in forward_parameters(model)
        }
        for scale in scales
    ]


def _muon_entries(
    model: MinGRUPolicy,
    optimizer: FusedMuon,
    gradients: list[dict[str, Tensor]],
) -> list[str]:
    """Step once per gradient set; digest the clip, masters, momentum and parameters."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    # The optimizer's order, which exp000's global norm reduces over.
    order = [
        names[id(parameter)]
        for group in optimizer.param_groups
        for parameter in cast("list[Tensor]", group["params"])
    ]
    lines: list[str] = []
    for index, step in enumerate(gradients):
        for name, parameter in model.named_parameters():
            parameter.grad = step[name].to(parameter.device)
        device = next(model.parameters()).device
        clip = clip_coefficient(
            optimizer.norm([step[name].to(device) for name in order]),
            optimizer.max_grad_norm,
        )
        lines += [f"step {index} clip {fp32(clip)}"]
        optimizer.step()
        for key in ("master_weight", "momentum_buffer"):
            lines += [
                f"step {index} {key} {name} {digest(value)}"
                for name, value in optimizer_state(model, optimizer, key).items()
            ]
        lines += [
            f"step {index} parameter {name} {digest(parameter)}"
            for name, parameter in forward_parameters(model)
        ]
    return lines


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
