#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Time one joint learner epoch at exp103's geometry with a 20-layer world model.

exp103's policy, objective and optimizer, the policy reading a projection of
the feature, and exp001's world model at its seed-0 init in bfloat16 on CUDA,
trained with the policy (``world_model.context``): 18 windows of 128 agents
by 256 steps, each recomputing its features from synthetic contexts, the
policy's backward, the world model's two-phase backward and a Muon step. The
contexts are one of two shapes, every row with 511 prefix slots:

  refill   steady refill (t_max 1,024, keep 256): every context restarts from
           its last 256 decisions once it holds 512, so lengths cycle
           257..512 (mean 384.5) from a random phase; none anchored.
  sliding  exact sliding windows of 512 decisions that never fill: each row's
           context begins its episode, 1..256 decisions long at step 0 and one
           longer per step; every fourth row's episode restarts once.

The frames are drawn from a pool of valid frames, so the encoder computes what
it would on real ones. After a warmup of two windows, ``--epochs`` epochs are
timed between device synchronizations, a recompile among them an error;
peak memory is the allocator's over them. With --compile the per-block and
frame-encoder functions are compiled, in the warmup. OUTPUT is one JSON
report: seconds and transitions per second per epoch, peak memory, each
window's planned passes, tokens and frames, the dense FLOPs those imply, and
the warmup's warnings.

Examples:
  priml/baselines/craftax/world_model/scripts/joint_speed.py /opt/scratch/artifacts/craftax/sole-wm-joint/joint-speed-refill.json --contexts refill

'''
# fmt: on

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import json
import time
import warnings

from configgle import PartialConfig
from torch import Tensor, nn

import torch

from priml.baselines.craftax.experiments import exp103
from priml.baselines.craftax.model import MinGRUPolicy
from priml.baselines.craftax.train_step import (
    AgentWindows,
    LearnerRollout,
    learn_joint_minibatch,
    minibatch_count,
    minibatch_offsets,
)
from priml.baselines.craftax.world_model.batch import Kind
from priml.baselines.craftax.world_model.context import (
    ContextReplay,
    Contexts,
    JointWorldModel,
    ReplayPlan,
    plan_replay,
)
from priml.baselines.craftax.world_model.feature import (
    InitialWeights,
    WorldModelFeature,
)
from priml.baselines.craftax.world_model.model import (
    FrameEncoder,
    WorldModel,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.loss.policy_gradient import PPO
from priml.model.attention.flash4 import Flash4Varlen
from priml.model.linear import Linear
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from priml.lib.codec import PlainTree


def main() -> int:
    """Time the epochs and write the report.

    Returns:
      status: 0.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output)
    device = torch.device("cuda")
    step = exp103().step
    model = step.model
    assert isinstance(model, MinGRUPolicy.Config)
    proj = model.proj_feature = Linear.Config()
    proj.channels_in = 1_152  # The world model's width.
    proj.init_weight = nn.init.zeros_
    windows = step.learner
    assert isinstance(windows, AgentWindows.Config)
    policy = model.make().to(device)
    objective = windows.objective.make()
    joint = _joint_world_model(flags, device=device)
    optimizer = step.optimizer.make()([*policy.parameters(), *joint.parameters()])
    agents, horizon = step.env.num_envs, step.rollout.horizon
    rows = windows.minibatch_size // horizon
    offsets = minibatch_offsets(
        agents=agents,
        rows=rows,
        count=minibatch_count(
            agents=agents,
            horizon=horizon,
            minibatch_size=windows.minibatch_size,
            replay_ratio=windows.replay_ratio,
        ),
    )
    generator = torch.Generator(device=device).manual_seed(flags.seed)
    rollout = _rollout(
        policy,
        observation_size=model.observation_size,
        contexts=synthetic_contexts(
            flags.contexts,
            agents=agents,
            horizon=horizon,
            generator=generator,
        ),
        horizon=horizon,
        width=joint.model.start.shape[-1],
        generator=generator,
    )
    # Numbers only: each window's plan holds its frames on the device.
    stats = [
        _window_stats(
            joint.model,
            _plan(rollout.minibatch(offset, rows), flags),
            layers=joint.layers,
        )
        for offset in offsets
    ]
    learn = partial(
        _learn,
        policy=policy,
        objective=objective,
        optimizer=optimizer,
        world_model=joint,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for offset in offsets[:2]:
            learn(rollout.minibatch(offset, rows))
    warned = sorted({str(warning.message)[:200] for warning in caught})
    torch.cuda.synchronize(device)
    resident = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    seconds: list[float] = []
    # A compiled kernel's every shape compiled in the warmup: a recompile now
    # would time the compiler, or the eager fallback past dynamo's limit.
    torch.compiler.set_stance("fail_on_recompile")
    try:
        for _ in range(flags.epochs):
            started = time.perf_counter()
            for offset in offsets:
                learn(rollout.minibatch(offset, rows))
            torch.cuda.synchronize(device)
            seconds.append(time.perf_counter() - started)
    finally:
        torch.compiler.set_stance("default")
    flops = sum(window["dense_flops"] for window in stats)
    transitions = len(offsets) * rows * horizon
    report: dict[str, PlainTree] = {
        "contexts": flags.contexts,
        "replay": {
            "bin_tokens": flags.bin_tokens,
            "pass_tokens": flags.pass_tokens,
            "frames_per_batch": flags.frames_per_batch,
            "compile": flags.compile,
        },
        "device": torch.cuda.get_device_name(device),
        "windows": len(offsets),
        "transitions_per_epoch": transitions,
        "epoch_seconds": cast("PlainTree", seconds),
        "transitions_per_second": transitions / min(seconds),
        "resident_gib": resident / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "window_mean": {
            name: sum(window[name] for window in stats) / len(stats)
            for name in stats[0]
        },
        "dense_gflop_per_transition": flops / transitions / 1e9,
        "achieved_tflops": flops / min(seconds) / 1e12,
        "warmup_warnings": cast("PlainTree", warned),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    output: Path
    contexts: str
    epochs: int
    bin_tokens: int
    pass_tokens: int
    frames_per_batch: int
    seed: int
    compile: bool


def synthetic_contexts(
    shape: str,
    *,
    agents: int,
    horizon: int,
    generator: torch.Generator,
    slots: int = 511,
) -> Contexts:
    """Draw an epoch's contexts of ``shape``, agent-major, over a pool of valid frames.

    Args:
      shape: ``refill`` or ``sliding``, as the module docstring describes.
      agents: Rows.
      horizon: Steps per row; a refill row restarts once in 256.
      generator: The draws' generator, on the device the contexts go to.
      slots: Prefix slots per row: the C512 contract's 511.

    Returns:
      contexts: The epoch's contexts, every frame token in range.

    """
    device = generator.device
    steps = torch.arange(horizon, device=device)
    if shape == "refill":
        phase = torch.randint(256, (agents, 1), generator=generator, device=device)
        lengths = 257 + (phase + steps) % 256
        anchored = torch.zeros_like(lengths, dtype=torch.bool)
        counts = torch.full((agents,), slots, device=device)
    else:
        first = torch.randint(1, 257, (agents, 1), generator=generator, device=device)
        lengths = first + steps
        restart = torch.randint(
            horizon,
            (agents, 1),
            generator=generator,
            device=device,
        )
        restarts = (torch.arange(agents, device=device) % 4 == 0)[:, None]
        lengths = torch.where(
            restarts & (steps >= restart),
            steps - restart + 1,
            lengths,
        )
        anchored = torch.ones_like(lengths, dtype=torch.bool)
        counts = first[:, 0] - 1
    cells, aux = _frame_pool(4_096, generator=generator)
    frames = torch.randint(
        len(cells),
        (agents, slots + horizon),
        generator=generator,
        device=device,
    )
    actions = torch.randint(
        43,
        (agents, slots + horizon),
        generator=generator,
        device=device,
    ).float()
    return Contexts(
        cells=cells[frames[:, slots:]],
        aux=aux[frames[:, slots:]],
        previous_actions=actions[:, slots:],
        lengths=lengths,
        anchored=anchored,
        prefix_cells=cells[frames[:, :slots]],
        prefix_aux=aux[frames[:, :slots]],
        prefix_previous_actions=actions[:, :slots],
        prefix_counts=counts,
    )


def _frame_pool(count: int, *, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Draw ``count`` frames whose every token is in its field's range."""
    schema = craftax_schema()
    device = generator.device
    cells = torch.stack(
        [
            torch.randint(
                field.valid,
                (count, schema.cell_slots),
                generator=generator,
                device=device,
            )
            for field in schema.cell_fields
        ],
        dim=-1,
    )
    aux = torch.stack(
        [
            torch.randint(
                low - 155,
                high - 154,
                (count,),
                generator=generator,
                device=device,
            )
            for low, high in schema.scalar_ranges
        ],
        dim=-1,
    )
    return cells.to(torch.uint8), aux.to(torch.int16)


# Its stored features, ``width`` floats per step, are random too: the joint learner only
# measures its replay against them.
def _rollout(
    policy: MinGRUPolicy,
    *,
    observation_size: int,
    contexts: Contexts,
    horizon: int,
    width: int,
    generator: torch.Generator,
) -> LearnerRollout:
    """Return an agent-major epoch of random transitions over ``contexts``."""
    agents = contexts.lengths.shape[0]
    device = generator.device

    def draw(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=generator, device=device)

    return LearnerRollout(
        observations=torch.zeros(
            agents,
            horizon,
            observation_size,
            dtype=policy.dtype,
            device=device,
        ),
        actions=(draw(agents, horizon) * 43).floor(),
        logprobs=-3 * draw(agents, horizon),
        rewards=draw(agents, horizon) - 0.5,
        terminals=torch.zeros(agents, horizon, device=device),
        values=draw(agents, horizon),
        action_mask=torch.ones(agents, horizon, 43, device=device),
        initial_states=policy.initial_state(agents, device=device),
        branch_starts=torch.zeros(agents, dtype=torch.uint8, device=device),
        features=torch.randn(
            agents,
            horizon,
            width,
            generator=generator,
            device=device,
        ).to(policy.dtype),
        contexts=contexts,
    )


def _learn(
    minibatch: LearnerRollout,
    *,
    policy: MinGRUPolicy,
    objective: PPO,
    optimizer: torch.optim.Optimizer,
    world_model: JointWorldModel,
) -> None:
    """Learn one window and step, as the joint learner's windows do."""
    optimizer.zero_grad(set_to_none=True)
    learn_joint_minibatch(policy, objective, minibatch, world_model)
    optimizer.step()


def _plan(minibatch: LearnerRollout, flags: Flags) -> ReplayPlan:
    """Plan one window's passes as the replay does."""
    if minibatch.contexts is None:
        raise ValueError("The synthetic rollout carries its contexts.")
    return plan_replay(
        minibatch.contexts,
        bin_tokens=flags.bin_tokens,
        pass_tokens=flags.pass_tokens,
    )


def _window_stats(
    model: WorldModel,
    plan: ReplayPlan,
    *,
    layers: int,
) -> dict[str, float]:
    """Return a window's passes, frames, read and padded tokens, micro-batches and FLOPs."""
    tokens, padded = _tokens(plan)
    return {
        "passes": plan.passes,
        "frames": len(plan.frame_cells),
        "tokens": tokens,
        "padded_tokens": padded,
        "micro_batches": len(plan.batches),
        "dense_flops": _dense_flops(model, plan, layers=layers),
    }


def _tokens(plan: ReplayPlan) -> tuple[int, int]:
    """Return a plan's tokens of passes, and of bins with their padding."""
    real = sum(int((batch.kind != Kind.PAD).sum()) for batch in plan.batches)
    return real, sum(batch.kind.numel() for batch in plan.batches)


# The model's dense FLOPs, times four for the forward without a graph, the forward
# with one and the backward's two: every matrix on every token it runs on, padding
# included, and each query against the keys it attends to (two products of the
# width per key). The encoder is counted at every position of every block, though
# its last block runs the pooling position alone. Norms, rotations, embeddings and
# the optimizer are left out.
def _dense_flops(model: WorldModel, plan: ReplayPlan, *, layers: int) -> float:
    """Return the dense FLOPs one window's replay and backward spend."""
    blocks = list(model.transformer.blocks)[:layers]
    width = model.start.shape[-1]
    matrices = sum(
        p.numel() for block in blocks for p in block.parameters() if p.ndim >= 2
    )
    keys = sum(
        float((batch.positions + 1)[batch.kind != Kind.PAD].sum())
        for batch in plan.batches
    )
    encoder = model.encoder
    assert isinstance(encoder, FrameEncoder)
    positions = len(encoder.slot_embedding) + 1
    attended = 4 * len(encoder.pool) * positions**2 * len(encoder.stack.blocks)
    frame = 2 * encoder.frame_macs() + attended
    forward = (
        2 * matrices * _tokens(plan)[1]
        + 4 * width * keys * layers
        + frame * len(plan.frame_cells)
    )
    return 4.0 * float(forward)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    defaults = ContextReplay.Config()
    parser.add_argument("output", type=Path, help="Where to write the JSON report.")
    parser.add_argument(
        "--contexts",
        choices=("refill", "sliding"),
        default="refill",
        help="The shape of the contexts (see above).",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epochs to time.")
    parser.add_argument("--bin-tokens", type=int, default=defaults.bin_tokens)
    parser.add_argument("--pass-tokens", type=int, default=defaults.pass_tokens)
    parser.add_argument(
        "--frames-per-batch",
        type=int,
        default=defaults.frames_per_batch,
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of the draws.")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the per-block and frame-encoder functions (fullgraph, static).",
    )


def _joint_world_model(flags: Flags, *, device: torch.device) -> JointWorldModel:
    """Build exp001's world model at its seed-0 init where the actor runs it, and its copy."""
    feature = WorldModelFeature.Config()
    feature.weights = InitialWeights.Config()
    source = feature.make()
    # An engine puts the model on the device in the actor's dtype, as a rollout's does.
    source.make_engine(rows=1, device=device).release()
    replay = ContextReplay.Config()
    replay.attention = Flash4Varlen.Config()
    replay.bin_tokens = flags.bin_tokens
    replay.pass_tokens = flags.pass_tokens
    replay.frames_per_batch = flags.frames_per_batch
    if flags.compile:
        replay.compile = PartialConfig(torch.compile, fullgraph=True, dynamic=False)
    return JointWorldModel(source.model, layers=feature.layers, replay=replay.make())


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
