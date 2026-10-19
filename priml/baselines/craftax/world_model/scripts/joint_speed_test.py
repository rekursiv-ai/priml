"""Check the speed script's synthetic contexts, its window accounting and its report."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import sys
import warnings

from configgle import Fig, PartialConfig
from treekle import json_loads

import pytest
import torch

from priml.baselines.craftax.experiments import exp003
from priml.baselines.craftax.testing import tiny_policy
from priml.baselines.craftax.train_step import AgentWindows, learn_joint_minibatch
from priml.baselines.craftax.world_model.batch import Kind
from priml.baselines.craftax.world_model.context import (
    ContextReplay,
    JointWorldModel,
    plan_replay,
)
from priml.baselines.craftax.world_model.model import (
    DecoderBlock,
    FrameEncoder,
    WorldModel,
)
from priml.baselines.craftax.world_model.scripts import joint_speed
from priml.baselines.craftax.world_model.scripts.joint_speed import (
    _rollout,
    synthetic_contexts,
)
from priml.baselines.craftax.world_model.scripts.testing import Clock, CpuTorch
from priml.baselines.craftax.world_model.testing import (
    random_contexts,
    small_schema,
    tiny_model,
)
from priml.loss.policy_gradient import TorchPPO
from priml.model.attention.kernel import SdpaVarlen
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.craftax.experiments import CraftaxTrainLoop
    from priml.baselines.craftax.model import MinGRUPolicy
    from priml.baselines.craftax.train_step import LearnerRollout
    from priml.baselines.craftax.world_model.context import Contexts
    from priml.loss.policy_gradient import PPO


def test_refill_contexts_cycle_from_256_kept_decisions_to_512() -> None:
    contexts = synthetic_contexts(
        "refill",
        agents=3,
        horizon=256,
        generator=torch.Generator().manual_seed(0),
    )
    lengths = contexts.lengths
    assert int(lengths.min()) == 257
    assert int(lengths.max()) == 512
    assert float(lengths.float().mean()) == (257 + 512) / 2
    assert not bool(contexts.anchored.any())
    # Each row restarts once in 256 steps: two passes.
    plan = plan_replay(contexts, bin_tokens=4_096, pass_tokens=16_384)
    assert plan.passes == 2 * 3


def test_sliding_contexts_begin_their_episodes_and_never_fill() -> None:
    contexts = synthetic_contexts(
        "sliding",
        agents=5,
        horizon=256,
        generator=torch.Generator().manual_seed(1),
    )
    assert bool(contexts.anchored.all())
    assert int(contexts.lengths.max()) <= 512
    assert int(contexts.prefix_counts.max()) < 511
    # Rows 0 and 4 restart their episodes once; the others are one pass each.
    plan = plan_replay(contexts, bin_tokens=4_096, pass_tokens=16_384)
    assert plan.passes == 5 + 2


@pytest.mark.parametrize("shape", ["refill", "sliding"])
def test_every_frame_token_is_in_its_fields_range(shape: str) -> None:
    contexts = synthetic_contexts(
        shape,
        agents=2,
        horizon=256,
        generator=torch.Generator().manual_seed(2),
    )
    assert contexts.cells.dtype == torch.uint8
    assert contexts.aux.dtype == torch.int16
    assert contexts.prefix_cells.shape == (2, 511, 99, 8)


def test_the_synthetic_rollout_is_in_the_joint_learners_layout() -> None:
    """Every field the joint learner reads, the stored features and contexts included."""
    config = tiny_policy(dtype=torch.float32)
    policy = config.make()
    contexts = _contexts(horizon=3)
    rollout = _rollout(
        policy,
        observation_size=config.observation_size,
        contexts=contexts,
        horizon=3,
        width=36,
        generator=torch.Generator().manual_seed(2),
    )
    # ``LearnerRollout``'s layout: agent-major, the policy's dtype, 43 actions.
    observations, features = rollout.observations, rollout.features
    assert observations.shape == (2, 3, config.observation_size)
    assert features is not None
    assert features.shape == (2, 3, 36)
    assert observations.dtype == features.dtype == policy.dtype
    for steps in (rollout.logprobs, rollout.rewards, rollout.terminals, rollout.values):
        assert steps.shape == (2, 3)
    assert rollout.action_mask.shape == (2, 3, 43)
    actions = rollout.actions
    assert torch.equal(actions, actions.floor())
    assert 0 <= int(actions.min()) <= int(actions.max()) < 43
    assert bool((rollout.logprobs <= 0).all())
    assert torch.equal(
        rollout.initial_states,
        policy.initial_state(2, device=torch.device("cpu")),
    )
    assert (rollout.branch_starts.dtype, rollout.branch_starts.shape) == (
        torch.uint8,
        (2,),
    )
    assert rollout.contexts is contexts


def test_planning_refuses_a_window_without_contexts() -> None:
    config = tiny_policy(dtype=torch.float32)
    rollout = _rollout(
        config.make(),
        observation_size=config.observation_size,
        contexts=_contexts(horizon=3),
        horizon=3,
        width=36,
        generator=torch.Generator().manual_seed(2),
    )
    with pytest.raises(ValueError, match="carries its contexts"):
        joint_speed._plan(replace(rollout, contexts=None), _flags())


def test_window_stats_count_the_plans_passes_tokens_and_flops() -> None:
    model = tiny_model(small_schema(), global_layers=2)
    # Bins of 4 tokens pad the passes, so read and padded tokens differ.
    plan = plan_replay(_contexts(horizon=3), bin_tokens=4, pass_tokens=26)
    real = sum(int((batch.kind != Kind.PAD).sum()) for batch in plan.batches)
    padded = sum(batch.kind.numel() for batch in plan.batches)
    assert 0 < real < padded
    stats = joint_speed._window_stats(model, plan, layers=2)
    assert stats == {
        "passes": plan.passes,
        "frames": len(plan.frame_cells),
        "tokens": real,
        "padded_tokens": padded,
        "micro_batches": len(plan.batches),
        "dense_flops": joint_speed._dense_flops(model, plan, layers=2),
    }
    # The second block adds its matrices on every padded token and its queries
    # against their keys, four times over: two forwards and the two backwards.
    second = list(model.transformer.blocks)[1]
    matrices = sum(p.numel() for p in second.parameters() if p.ndim >= 2)
    keys = sum(
        float((batch.positions + 1)[batch.kind != Kind.PAD].sum())
        for batch in plan.batches
    )
    width = model.start.shape[-1]
    per_layer = 4.0 * (2 * matrices * padded + 4 * width * keys)
    one = joint_speed._dense_flops(model, plan, layers=1)
    assert stats["dense_flops"] - one == pytest.approx(per_layer)
    # The first block's share plus the frame encoder's, which is not zero.
    first = sum(
        p.numel()
        for p in next(iter(model.transformer.blocks)).parameters()
        if p.ndim >= 2
    )
    assert one > 4.0 * (2 * first * padded + 4 * width * keys)


@pytest.mark.parametrize("compiled", [False, True])
def test_the_joint_world_model_copies_the_feature_source_for_the_replay(
    monkeypatch: pytest.MonkeyPatch,
    *,
    compiled: bool,
) -> None:
    fake = CpuTorch()
    monkeypatch.setattr(joint_speed, "torch", fake)
    monkeypatch.setattr(joint_speed, "InitialWeights", _TwentyLayers)
    monkeypatch.setattr(joint_speed, "Flash4Varlen", SdpaVarlen)
    flags = _flags(compile=compiled)
    joint = joint_speed._joint_world_model(flags, device=torch.device("cpu"))
    assert joint.layers == 20
    replay = joint.replay
    assert (replay.bin_tokens, replay.pass_tokens, replay.frames_per_batch) == (
        flags.bin_tokens,
        flags.pass_tokens,
        flags.frames_per_batch,
    )
    assert isinstance(replay.attention, SdpaVarlen)
    # The engine put the source in the actor's dtype; the replay trains a copy.
    assert joint.source.start.dtype == torch.bfloat16
    assert joint.model is not joint.source
    options = {"fullgraph": True, "dynamic": False}
    assert fake.compiled == ([options] * 3 if compiled else [])


def test_main_times_each_epoch_and_reports_its_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = CpuTorch()
    monkeypatch.setattr(joint_speed, "torch", fake)
    # Epochs of 1.0 s and 0.5 s: the rate is the faster one's.
    monkeypatch.setattr(joint_speed, "time", Clock([0.0, 1.0, 2.0, 2.5]))
    monkeypatch.setattr(joint_speed, "exp003", _tiny_exp003)
    drawn: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        joint_speed,
        "synthetic_contexts",
        partial(_drawn_contexts, drawn=drawn),
    )
    built: list[tuple[JointWorldModel, list[Tensor], torch.device]] = []
    monkeypatch.setattr(
        joint_speed,
        "_joint_world_model",
        partial(_wide_joint, built=built),
    )
    learned: list[int] = []
    monkeypatch.setattr(
        joint_speed,
        "learn_joint_minibatch",
        partial(_counted_learn, learned=learned),
    )
    output = tmp_path / "report.json"
    argv = ["joint_speed.py", str(output), "--contexts", "sliding", "--epochs", "2"]
    replay = ["--bin-tokens", "1", "--pass-tokens", "26", "--frames-per-batch", "5"]
    monkeypatch.setattr(sys, "argv", [*argv, *replay])

    assert joint_speed.main() == 0

    # Two agents of horizon 2 in minibatches of one agent: exp003's replay
    # ratio of 1.17 makes two windows an epoch.
    assert drawn == [("sliding", 2, 2)]
    [(joint, initial, device)] = built
    assert device == torch.device("cpu")
    # Two warmup windows, then two epochs of two.
    assert learned == list(range(1, 7))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial, joint.parameters(), strict=True)
    )
    plans = [
        plan_replay(_contexts(horizon=2).rows(offset, 1), bin_tokens=1, pass_tokens=26)
        for offset in (0, 1)
    ]
    stats = [joint_speed._window_stats(joint.model, plan, layers=1) for plan in plans]
    flops = sum(window["dense_flops"] for window in stats)
    report = json_loads(output.read_text(), dict[str, object])
    assert report == {
        "contexts": "sliding",
        "replay": {
            "bin_tokens": 1,
            "pass_tokens": 26,
            "frames_per_batch": 5,
            "compile": False,
        },
        "device": "Fake H200",
        "windows": 2,
        "transitions_per_epoch": 4,
        "epoch_seconds": [1.0, 0.5],
        "transitions_per_second": 8.0,
        "resident_gib": 3.0,
        "peak_allocated_gib": 5.0,
        "peak_reserved_gib": 6.0,
        "window_mean": {
            name: (stats[0][name] + stats[1][name]) / 2 for name in stats[0]
        },
        "dense_gflop_per_transition": pytest.approx(flops / 4 / 1e9),
        "achieved_tflops": pytest.approx(flops / 0.5 / 1e12),
        "warmup_warnings": ["learned window 1", "learned window 2"],
    }
    assert capsys.readouterr().out == output.read_text()
    # The peak is reset once, after the warmup; each epoch synchronizes before
    # its clock stops.
    assert fake.cuda.calls == [
        "synchronize",
        "memory_allocated",
        "reset_peak_memory_stats",
        "synchronize",
        "synchronize",
    ]


@dataclass(kw_only=True, slots=True)
class _Flags:
    output: Path
    contexts: str
    epochs: int
    bin_tokens: int
    pass_tokens: int
    frames_per_batch: int
    seed: int
    compile: bool


def _flags(*, compile: bool = False) -> _Flags:
    return _Flags(
        output=Path("unused.json"),
        contexts="refill",
        epochs=1,
        bin_tokens=7,
        pass_tokens=30,
        frames_per_batch=3,
        seed=0,
        compile=compile,
    )


def _contexts(*, horizon: int) -> Contexts:
    """Two anchored rows over the small schema, one filled prefix slot in the second."""
    return random_contexts(
        torch.tensor(((1, 2, 3), (2, 3, 4)))[:, :horizon],
        torch.ones(2, horizon, dtype=torch.bool),
        schema=small_schema(),
        slots=4,
        counts=torch.tensor((0, 1)),
        seed=1,
    )


class _TwentyLayers:
    """Twenty global blocks of the tiny world model: exp001's depth, at test width."""

    class Config(Fig["_TwentyLayers"]):
        """Nothing to configure."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self) -> WorldModel:
        """Return the model."""
        return tiny_model(small_schema(), global_layers=20)


def _tiny_exp003() -> CraftaxTrainLoop:
    """Return exp003 at test size on the CPU: two agents, horizon 2, SGD."""
    config = exp003()
    step = config.step
    step.model = tiny_policy(dtype=torch.float32)
    step.env.num_envs = 2
    step.rollout.horizon = 2
    sgd = step.optimizer = PartialConfig(torch.optim.SGD)
    sgd.lr = 0.1
    windows = step.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.objective = TorchPPO.Config().update(windows.objective, skip_missing=True)
    windows.minibatch_size = 2
    return config


def _drawn_contexts(
    shape: str,
    *,
    agents: int,
    horizon: int,
    generator: torch.Generator,
    drawn: list[tuple[str, int, int]],
) -> Contexts:
    """Stand in for ``synthetic_contexts``: two short rows over the small schema."""
    assert generator.device == torch.device("cpu")
    drawn.append((shape, agents, horizon))
    return _contexts(horizon=horizon)


def _wide_joint(
    flags: joint_speed.Flags,
    *,
    device: torch.device,
    built: list[tuple[JointWorldModel, list[Tensor], torch.device]],
) -> JointWorldModel:
    """Stand in for ``_joint_world_model``: one narrow block, float32."""
    replay = ContextReplay.Config()
    replay.bin_tokens = flags.bin_tokens
    replay.pass_tokens = flags.pass_tokens
    replay.frames_per_batch = flags.frames_per_batch
    joint = JointWorldModel(_wide_model(), layers=1, replay=replay.make())
    built.append((joint, [p.detach().clone() for p in joint.parameters()], device))
    return joint


def _wide_model() -> WorldModel:
    """Return the tiny world model with one global block wider than its frames."""
    config = WorldModel.Config(schema=small_schema())
    config.encoder.channels_in = config.decoder.channels_in = 16
    assert isinstance(config.encoder, FrameEncoder.Config)
    for stack in (config.encoder.stack, config.decoder.stack):
        stack.num_layers = 1
        assert isinstance(stack.block, TransformerBlock.Config | DecoderBlock.Config)
        assert isinstance(stack.block.ffn, SwiGLU.Config)
        stack.block.ffn.channels_hidden = 32
    config.transformer.channels_in = 24
    config.transformer.num_layers = 1
    block = config.transformer.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.ffn.channels_hidden = 48
    torch.manual_seed(0)
    return config.make().eval()


def _counted_learn(
    policy: MinGRUPolicy,
    objective: PPO,
    minibatch: LearnerRollout,
    world_model: JointWorldModel,
    *,
    learned: list[int],
) -> tuple[Tensor, Tensor, Tensor]:
    """Learn the window, counting it; the warmup's two each warn once."""
    learned.append(len(learned) + 1)
    if len(learned) <= 2:
        warnings.warn(f"learned window {len(learned)}", UserWarning, stacklevel=2)
    return learn_joint_minibatch(policy, objective, minibatch, world_model)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
