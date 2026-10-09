"""Test support shared by the port's tests: tiny configs, a fake env, observations.

Nothing here trains. A test that needs a small policy, a small pipeline, an
env without a game, valid packed observations, a masked random action stream,
or the bit-for-bit goldens' inputs and text files imports it from here rather
than from another test module.

A golden is a text file under the test's ``testdata/``: one line per frozen
value, a tensor as its dtype, shape and sha256, a scalar as its fp32 bits.
``--regenerate-golden`` mints it. A golden of a tiny CPU run computes inside
:func:`host_agnostic_pipeline`, so one file holds on every host. A golden
whose bits depend on the host -- a GPU's kernels, or the env's libm over
enough worlds and steps to reach an input two libms round apart -- carries a
host key in its name and skips on a host nobody minted it for,
before the work when the test calls :func:`require_golden` first. The weights
the goldens start from are drawn by :func:`fill_portable`, the same bits on
every host, so no golden reads a checkpoint from outside the repository.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import difflib
import hashlib
import re

from torch import Tensor, nn

import numpy as np
import pytest
import torch

from priml.baselines.craftax.env import CraftaxEnv, WorldPool
from priml.baselines.craftax.experiments import exp000, exp002
from priml.baselines.craftax.game.jit import jit, platform_key
from priml.baselines.craftax.game.state import ATN_DIM, OBS_SIZE
from priml.baselines.craftax.model import (
    FeasibilityLoss,
    MinGRUPolicy,
    NoEncoder,
    Policy,
)
from priml.baselines.craftax.policies.encoder import MLP, BoardEncoder
from priml.baselines.craftax.rollout import StepGraph, TorchPhiloxSampler
from priml.baselines.craftax.train_step import AgentWindows
from priml.baselines.craftax.world_model.feature import (
    InitialWeights,
    Refill,
    WorldModelFeature,
)
from priml.loss.policy_gradient import TorchPPO
from priml.model.embedding import MultiHotEmbedding
from priml.model.linear import Linear
from priml.model.min_gru import TorchScan
from priml.testing import regenerate
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_text_golden


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from priml.baselines.craftax.game.state import Array1, Array2
    from priml.baselines.craftax.train_step import CraftaxTrainStep


@jit
def masked_random_actions_numba(
    streams: Array1[np.uint32],
    masks: Array2[int],
    actions: Array2[np.float32],
) -> None:
    """Advance each xorshift32 stream and pick ``valid[x % nvalid]`` from its mask.

    The oracle traces' action stream: environment ``i``'s stream starts at
    ``0x9E3779B9 ^ (i + 1)``, and each step reads the previous step's mask.

    Args:
      streams: ``uint32 [num_envs]``, advanced in place.
      masks: ``uint8 [num_envs, 43]``.
      actions: ``float32 [num_envs, 1]``, overwritten.

    """
    # A Numba kernel over the env's numpy buffers, one environment at a time, as
    # the trace generator's C loop is; the uint32 constants and masks keep the
    # xorshift's 32-bit wraparound, which Python ints would not.
    for i in range(streams.shape[0]):
        x = streams[i]
        x ^= (x << np.uint32(13)) & np.uint32(0xFFFF_FFFF)
        x ^= x >> np.uint32(17)
        x ^= (x << np.uint32(5)) & np.uint32(0xFFFF_FFFF)
        streams[i] = np.uint32(x)
        nvalid = 0
        for a in range(masks.shape[1]):
            if masks[i, a]:
                nvalid += 1
        pick = np.uint32(x) % np.uint32(nvalid)
        seen = np.uint32(0)
        for a in range(masks.shape[1]):
            if masks[i, a]:
                if seen == pick:
                    actions[i, 0] = np.float32(a)
                    break
                seen += np.uint32(1)


def multi_hot_embedding(config: MinGRUPolicy.Config) -> MultiHotEmbedding.Config:
    """Return the policy's first stage, which must be the default multi-hot embedding.

    Args:
      config: A policy config.

    Returns:
      embedding: Its ``embedding``, narrowed from the slot's protocol.

    """
    embedding = config.embedding
    assert isinstance(embedding, MultiHotEmbedding.Config)
    return embedding


def tiny_policy(*, dtype: torch.dtype = torch.bfloat16) -> MinGRUPolicy.Config:
    """Return exp000's policy at test size: 2 layers of 8, embeddings of 2.

    Args:
      dtype: The model dtype, and the carry's and the decoder output's, as
        exp000 keeps them alike.

    Returns:
      config: The policy config.

    """
    config = MinGRUPolicy.Config()
    multi_hot_embedding(config).channels_out = 2
    config.channels_hidden = 8
    config.num_layers = 2
    config.dtype = config.state_dtype = config.output_dtype = dtype
    return config


def tiny_board_policy(*, dtype: torch.dtype = torch.bfloat16) -> MinGRUPolicy.Config:
    """Return the board policy at test size: :func:`tiny_policy`, a tiny encoder, both slots.

    The encoder's widths are each the smallest that stays distinct from the
    rest: cells of 2, an action embedding of 3, status blocks of 4, a FiLM
    branch of 5 and a multiscale mix of 7 (its input is 3 x 2); each block's
    injection is 3 wide and zero at init; the feasibility loss weighs 0.1.
    The layout's own sizes stay: 99 cells on a 9 x 11 board, 51 scalars, 44
    action ids.

    Args:
      dtype: As :func:`tiny_policy` takes it.

    Returns:
      config: The policy config.

    """
    config = tiny_policy(dtype=dtype)
    encoder = config.embedding = BoardEncoder.Config()
    encoder.cells.channels_out = 2
    encoder.status.embedding.channels_out = 3
    encoder.status.block.channels_hidden = 4
    encoder.board.spatial.channels_hidden = 5
    encoder.board.multiscale.channels_hidden = 7
    injection = config.injection = MLP.Config()
    injection.channels_hidden = 3
    injection.proj_out.init_weight = nn.init.zeros_
    auxiliary = config.auxiliary = FeasibilityLoss.Config()
    auxiliary.coefficient = 0.1
    return config


def sole_feature_policy(*, feature_width: int) -> MinGRUPolicy.Config:
    """Return :func:`tiny_board_policy`'s trunk and loss reading a feature alone, no encoder.

    Args:
      feature_width: Floats per step of the feature ``proj_feature`` reads.

    Returns:
      config: No board encoder, ``proj_in`` or injection; the observation's
        width stays the board policy's, for the feasibility loss's targets.

    """
    config = tiny_board_policy()
    stage = config.embedding = NoEncoder.Config()
    stage.observation_size = tiny_board_policy().observation_size
    config.injection = None
    proj = config.proj_feature = Linear.Config()
    proj.channels_in = feature_width
    return config


def smoke_feature() -> WorldModelFeature.Config:
    """Return a float32 masked feature over the smoke world model, hooked every 2 steps.

    Returns:
      config: 16 positions per row, re-prefilled from the last 4 decisions,
        tapping the model's one global block.

    """
    config = WorldModelFeature.Config()
    weights = config.weights = InitialWeights.Config()
    weights.experiment = "priml.baselines.craftax.world_model.experiments.exp_smoke"
    refill = config.history = Refill.Config()
    refill.t_max = 16
    refill.keep = 4
    config.layers = 1
    config.hook_interval = 2
    config.dtype = torch.float32
    return config


def packed_observations(
    config: MinGRUPolicy.Config,
    *,
    batch: int,
    time: int = 1,
    seed: int = 0,
) -> Tensor:
    """Return random packed observations: in-range ids per field, then scalars.

    For a :class:`~priml.baselines.craftax.policies.encoder.BoardEncoder`, a previous
    action id follows, drawn from the ids its table holds.

    Args:
      config: The policy whose first stage reads them.
      batch: Rows.
      time: Steps per row; a single step is squeezed out.
      seed: The draws' seed.

    Returns:
      observations: ``[batch, time, observation_size]``, or ``[batch,
        observation_size]`` for one step.

    """
    generator = torch.Generator().manual_seed(seed)
    encoder = config.embedding
    if isinstance(encoder, BoardEncoder.Config):
        packed = _draw_packed(encoder.cells, generator, batch=batch, time=time)
        ids = encoder.status.embedding.channels_in
        actions = torch.randint(ids, (batch, time, 1), generator=generator).float()
        return torch.cat((packed, actions), dim=-1).squeeze(1)
    embedding = multi_hot_embedding(config)
    return _draw_packed(embedding, generator, batch=batch, time=time).squeeze(1)


def tiny_env() -> CraftaxEnv.Config:
    """Return 4 real environments in 2 buffers, one thread each, 8 pool worlds.

    Returns:
      config: The env config.

    """
    config = CraftaxEnv.Config()
    config.num_envs = 4
    config.num_buffers = 2
    config.threads_per_buffer = 1
    pool = config.restart = WorldPool.Config()
    pool.num_worlds = 8
    return config


def tiny_train_step() -> CraftaxTrainStep.Config:
    """Return exp000's pipeline at test size on the CPU: its recipe, a tiny geometry.

    The scan, the learning rule and the sampler are exp000's in their torch
    forms, since their kernels run only on CUDA: the same recurrence,
    coefficients and Philox streams, with torch's precise ``exp``.

    Returns:
      config: Three epochs of 4 environments, horizon 4, minibatches of 2
        agents, from the policy's own init.

    """
    config = exp000().step
    config.env = tiny_env()
    return _shrink_pipeline(config)


def tiny_exp000_step() -> CraftaxTrainStep.Config:
    """Return exp000's step at test size on the GPU: its kernels and recipe, a tiny geometry.

    The policy is 2 layers of 8, in :func:`tiny_env`'s 4 environments; the
    horizon is 8, the least the learning rule's kernel takes, and the
    minibatches 2 agents.

    Returns:
      config: Four epochs, from the policy's own init.

    """
    config = exp000().step
    config.checkpoint = None
    model = config.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.channels_hidden = 8
    model.num_layers = 2
    config.env = tiny_env()
    config.parallelism.device = "cuda"
    windows = config.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.minibatch_size = 16
    config.rollout.horizon = 8
    config.train_budget_steps = 4
    return config


def tiny_exp002_step() -> CraftaxTrainStep.Config:
    """Return exp002's pipeline at :func:`tiny_train_step`'s size and in its torch forms.

    exp002's rules, fresh worlds and dense symbolic view, in 4 environments of
    2 buffers.

    Returns:
      config: Three epochs from the policy's own init.

    """
    config = exp002().step
    env = tiny_env()
    env.rules = config.env.rules
    env.restart = config.env.restart
    config.env = env
    return _shrink_pipeline(config)


@contextmanager
def host_agnostic_pipeline() -> Generator[None]:
    """Run ``host_agnostic_numerics`` here and in every rollout step, on any thread.

    Torch keeps a dispatch mode per thread, and each buffer of a rollout steps
    on a worker thread, so the mode entered here alone would leave the actor's
    forward and the sampler on the host's native kernels. Each
    :meth:`StepGraph.replay` enters it too; with the learner on this thread, a
    CPU train step's or evaluation's every float op is host-agnostic. The env
    steps in Numba, outside torch, on the platform's libm: the tiny
    pipeline's worlds and steps measured the same bits on glibc and macOS.

    Yields:
      item: Nothing; used as a context manager.

    """
    replay = StepGraph.replay
    with (
        host_agnostic_numerics(),
        patch.object(StepGraph, "replay", _host_agnostic(replay)),
    ):
        yield


def portable_uniform(
    *shape: int,
    bound: float,
    generator: torch.Generator,
) -> Tensor:
    """Draw fp32 ``U(-bound, bound)`` whose bits are the same on every host.

    ``torch.rand`` fills multiples of 2^-24 from integer draws, and ``2u - 1``
    is exact, so only the correctly rounded product with ``bound`` rounds.
    ``randn`` is not portable: its Box-Muller transform runs SLEEF's
    ``log``/``cos`` under AVX2 and libm's elsewhere (up to 6 ULP apart).

    Args:
      *shape: The tensor's shape.
      bound: Half the interval's width.
      generator: The draws' CPU generator, advanced.

    Returns:
      values: fp32 on the CPU.

    """
    return (torch.rand(shape, generator=generator) * 2 - 1) * bound


def forward_parameters(policy: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Name a policy's weights, a MinGRU's in forward order, the order the goldens drew them in.

    The first stage's, ``proj_in``'s, each block's, then ``proj_out``'s, then
    the optional slots' (each block's injection, the auxiliary loss): not
    ``parameters()``, which puts ``proj_out`` before the blocks, as the optimizer
    takes them. Every other weight follows in ``named_parameters()`` order: a
    MinGRU's ``proj_feature``, and all of another policy's (the GRU's cell and
    heads, the actor-critic's towers), so none keeps its init.

    Args:
      policy: Any policy.

    Returns:
      parameters: ``(name, parameter)`` pairs, every weight once.

    """
    named = list(policy.named_parameters())
    stages = (
        "embedding.",
        "proj_in.",
        "blocks.",
        "proj_out.",
        "injections.",
        "auxiliary.",
    )
    staged = [
        (name, parameter)
        for stage in stages
        for name, parameter in named
        if name.startswith(stage)
    ]
    seen = {name for name, _ in staged}
    return [*staged, *((name, p) for name, p in named if name not in seen)]


# Every parameter, not a MinGRU's stages alone: an init left in place draws ``randn``
# (an orthogonal init's QR input), whose bits differ by host, so the GRU's golden
# differed between macOS and x86 while its cell and heads kept theirs.
def fill_portable(model: nn.Module, *, seed: int) -> None:
    """Fill synthetic uniform test weights with :func:`portable_uniform`.

    Each parameter uses its fan-in bound. This portable fixture does not
    reproduce the policy's layer-specific initialization. Draws are rounded
    to the parameter dtype, in :func:`forward_parameters` order.

    Args:
      model: The policy whose every parameter is overwritten.
      seed: The draws' seed.

    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, parameter in forward_parameters(model):
            draws = portable_uniform(
                *parameter.shape,
                bound=parameter.shape[-1] ** -0.5,
                generator=generator,
            )
            parameter.copy_(draws)


def portable_checkpoint(config: MinGRUPolicy.Config, path: Path, *, seed: int) -> Path:
    """Write :func:`fill_portable`'s weights as a checkpoint of fp32 masters.

    The masters are the weights' own values, rounded to the policy's dtype as
    :func:`fill_portable` leaves them, so a step started from the file holds
    exactly the policy :func:`fill_portable` fills.

    Args:
      config: The policy's config.
      path: The file to write, a ``state_dict`` by parameter name.
      seed: The draws' seed.

    Returns:
      path: ``path``, written.

    """
    model = config.make()
    fill_portable(model, seed=seed)
    masters = {
        name: weight.detach().float() for name, weight in model.named_parameters()
    }
    torch.save(masters, path)
    return path


def optimizer_state(
    model: Policy,
    optimizer: torch.optim.Optimizer,
    key: str,
) -> dict[str, Tensor]:
    """Return one of an optimizer's per-parameter states, keyed by parameter name.

    Args:
      model: The policy.
      optimizer: Its optimizer, after a step or ``train_step.load_masters``.
      key: The state, e.g. Muon's ``master_weight`` or ``momentum_buffer``.

    Returns:
      state: The live tensors, by parameter name.

    """
    state = cast("dict[int, dict[str, Tensor]]", optimizer.state_dict()["state"])
    return {
        name: state[index][key]
        for index, name in enumerate(_optimizer_names(model, optimizer))
    }


def digest(value: Tensor | np.ndarray) -> str:
    """Return a golden entry for a tensor or array: its dtype, shape and sha256.

    Args:
      value: Any tensor, on any device, or a numpy array.

    Returns:
      entry: ``"<dtype> <shape> <sha256 of the bytes>"``.

    """
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return f"{array.dtype.str} {array.shape} {_sha256(array.tobytes())}"
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return f"{value.dtype} {tuple(value.shape)} {_sha256(raw.numpy().tobytes())}"


def env_digests(env: CraftaxEnv) -> list[str]:
    """Return golden entries for everything an env holds between steps.

    Args:
      env: The environments.

    Returns:
      lines: The worlds, ``rand_r`` streams and episode stats as bytes, then
        the five buffers, one entry each.

    """
    return [
        f"states {digest(env.states.view(np.uint8))}",
        f"rngs {digest(env.rngs)}",
        f"stats {digest(env.stats.view(np.uint8))}",
        f"observations {digest(env.observations)}",
        f"action_mask {digest(env.action_mask)}",
        f"rewards {digest(env.rewards)}",
        f"terminals {digest(env.terminals)}",
        f"actions {digest(env.actions)}",
    ]


def fp32(value: float | Tensor) -> str:
    """Return a golden entry for one fp32 value: its bits, then its decimal.

    Args:
      value: A float holding an fp32 value, or a one-element fp32 tensor.

    Returns:
      entry: ``"0x3f800000 1.0"``.

    """
    single = np.float32(float(value))
    return f"0x{int(single.view(np.uint32)):08x} {float(single)!r}"


def gpu_key() -> str:
    """Return the host key of a GPU golden: the device's model, then the platform.

    The platform's libm is part of it because a golden that steps the env on
    the CPU holds only where that libm runs (``game.jit.platform_key``).
    Without a CUDA device there is no model to name, and the test skips.

    Returns:
      key: For example ``nvidia-h200_linux-x86_64-glibc2.35``.

    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    model = re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name().lower())
    return f"{model.strip('-')}_{platform_key()}"


def assert_golden(
    *,
    test_file: str,
    name: str,
    lines: Sequence[str],
    host: str = "",
) -> None:
    """Assert ``lines`` equal ``testdata/<name>[_<host>].txt`` beside ``test_file``.

    A missing golden is minted and fails, for review, as ``assert_text_golden``
    does; one keyed by ``host`` instead skips, since a golden holds only on
    the host class it was minted on. ``--regenerate-golden`` rewrites it.

    Args:
      test_file: ``__file__`` of the test module.
      name: The golden's name.
      lines: The rendered values, one per line.
      host: The host key, for a golden whose bits depend on the host.

    Raises:
      AssertionError: The golden differs; the message holds the diff.

    """
    path = _golden_path(test_file=test_file, name=name, host=host)
    overwrite = regenerate.golden()
    if not overwrite and path.exists():
        # pragma: no mutate start -- "UTF-8" names the same codec, and the
        # default is UTF-8 on every host the suite runs on.
        expected = path.read_text(encoding="utf-8").splitlines()
        # pragma: no mutate end
        if expected != list(lines):
            diff = difflib.unified_diff(expected, lines, "golden", "now", lineterm="")
            raise AssertionError(f"{path.stem} changed:\n" + "\n".join(diff))
        return
    if not overwrite and host:
        pytest.skip(_missing_golden(path))
    assert_text_golden(
        test_file=test_file,
        name=path.stem,
        rendered="\n".join(lines),
    )


def read_golden(*, test_file: str, name: str, host: str = "") -> list[str]:
    """Return a golden's lines, as :func:`assert_golden` names it; skip without it.

    Args:
      test_file: ``__file__`` of the test module that owns it.
      name: The golden's name.
      host: Its host key, if it has one.

    Returns:
      lines: The golden's lines.

    """
    path = _golden_path(test_file=test_file, name=name, host=host)
    if not path.exists():
        pytest.skip(_missing_golden(path))
    return path.read_text(encoding="utf-8").splitlines()


def require_golden(
    *,
    test_file: str,
    name: str,
    host: str,
) -> None:
    """Skip now, before the work, on a host that has no golden to compare it with.

    :func:`assert_golden` skips such a host too, but only after the work:
    the full-size golden of exp000's 200th epoch, since retired, ran 198 s on
    an RTX 5090 before that skip.
    ``--regenerate-golden`` runs on, to mint it.

    Args:
      test_file: ``__file__`` of the test module.
      name: The golden's name.
      host: Its host key.

    """
    if not regenerate.golden():
        read_golden(test_file=test_file, name=name, host=host)


@dataclass(slots=True, kw_only=True)
class FakeEnv:
    """An env of random transitions with the buffer contract.

    Each buffer draws from its own generator, so buffers stepped concurrently
    produce the same transitions in any interleaving.

    Attributes:
      observations: As ``CraftaxEnv``'s.
      action_mask: As ``CraftaxEnv``'s.
      rewards: As ``CraftaxEnv``'s.
      terminals: As ``CraftaxEnv``'s.
      actions: As ``CraftaxEnv``'s.
      num_envs: Environments.
      num_buffers: Buffers.
      generators: Each buffer's transitions' randomness.
      stepped: Every action row ``step_buffer`` consumed, per buffer.

    """

    observations: Tensor
    action_mask: Tensor
    rewards: Tensor
    terminals: Tensor
    actions: Tensor
    num_envs: int
    num_buffers: int
    generators: list[torch.Generator]
    stepped: list[list[Tensor]] = field(default_factory=list)

    @classmethod
    def make(cls, *, num_envs: int, num_buffers: int, seed: int) -> FakeEnv:
        """Build one with the env's widths and a first random observation.

        Args:
          num_envs: Environments.
          num_buffers: Buffers.
          seed: The transitions' seed; buffer ``b`` draws from ``seed + b``.

        Returns:
          env: With ``observations`` filled and everything else zero.

        """
        # Pinned where CUDA is, as ``CraftaxEnv``'s: a captured step graph
        # copies to and from these rows asynchronously.
        # pragma: no mutate start -- pinning exists only with CUDA; on a CPU host
        # every variant allocates the same rows, and the GPU tests copy from them.
        zeros = partial(torch.zeros, pin_memory=torch.cuda.is_available())
        ones = partial(torch.ones, pin_memory=torch.cuda.is_available())
        # pragma: no mutate end
        env = cls(
            observations=zeros(num_envs, OBS_SIZE),
            action_mask=ones(num_envs, ATN_DIM, dtype=torch.uint8),
            rewards=zeros(num_envs),
            terminals=zeros(num_envs),
            actions=zeros(num_envs, 1),
            num_envs=num_envs,
            num_buffers=num_buffers,
            generators=[
                torch.Generator().manual_seed(seed + buffer)
                for buffer in range(num_buffers)
            ],
            stepped=[[] for _ in range(num_buffers)],
        )
        for buffer in range(num_buffers):
            env._observe(buffer)
        return env

    def buffer_slice(self, buffer: int) -> slice:
        """Return the rows of every buffer attribute that belong to ``buffer``.

        Args:
          buffer: Which buffer.

        Returns:
          rows: Its rows.

        """
        rows = self.num_envs // self.num_buffers
        return slice(buffer * rows, (buffer + 1) * rows)

    def step_buffer(self, buffer: int) -> None:
        """Record the actions, then write a random transition.

        Args:
          buffer: Which buffer to step.

        """
        rows = self.buffer_slice(buffer)
        generator = self.generators[buffer]
        self.stepped[buffer].append(self.actions[rows].clone())
        self._observe(buffer)
        count = len(range(*rows.indices(self.num_envs)))
        self.rewards[rows] = torch.randn(count, generator=generator)
        self.terminals[rows] = (torch.rand(count, generator=generator) < 0.1).float()

    def _observe(self, buffer: int) -> None:
        rows = self.buffer_slice(buffer)
        generator = self.generators[buffer]
        count = len(range(*rows.indices(self.num_envs)))
        embedding = multi_hot_embedding(MinGRUPolicy.Config())
        self.observations[rows] = _draw_packed(
            embedding,
            generator,
            batch=count,
        ).squeeze(1)
        self.action_mask[rows] = (
            torch.rand(count, ATN_DIM, generator=generator) > 0.2
        ).to(torch.uint8)
        self.action_mask[rows, 0] = 1


def _shrink_pipeline(config: CraftaxTrainStep.Config) -> CraftaxTrainStep.Config:
    """Shrink a recipe's step in place to the tiny CPU pipeline; its env is already tiny."""
    config.checkpoint = None
    model = config.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.channels_hidden = 8
    model.num_layers = 2
    model.block.scan = TorchScan.Config()
    config.parallelism.device = "cpu"
    windows = config.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.objective = TorchPPO.Config().update(
        windows.objective,
        skip_missing=True,
    )
    windows.minibatch_size = 8
    config.sampler = TorchPhiloxSampler.Config().update(
        config.sampler,
        skip_missing=True,
    )
    config.rollout.horizon = 4
    config.train_budget_steps = 3
    return config


def _host_agnostic(replay: Callable[[StepGraph], None]) -> Callable[[StepGraph], None]:
    """Return ``replay`` run inside ``host_agnostic_numerics`` on the calling thread."""

    def replay_host_agnostic(graph: StepGraph) -> None:
        with host_agnostic_numerics():
            replay(graph)

    return replay_host_agnostic


def _draw_packed(
    embedding: MultiHotEmbedding.Config,
    generator: torch.Generator,
    *,
    batch: int,
    time: int = 1,
) -> Tensor:
    """Draw ``[batch, time, size]`` packed observations: in-range ids, then scalars."""
    offsets = torch.tensor(embedding.offsets)
    widths = torch.diff(torch.cat((offsets, torch.tensor([embedding.channels_in]))))
    ids = torch.floor(
        torch.rand(
            batch,
            time,
            embedding.num_cells,
            len(embedding.offsets),
            generator=generator,
        )
        * widths,
    )
    scalars = torch.rand(batch, time, embedding.num_scalars, generator=generator)
    return torch.cat((ids.flatten(-2), scalars), dim=-1)


def _optimizer_names(model: Policy, optimizer: torch.optim.Optimizer) -> list[str]:
    """Name the optimizer's parameters in its own order, as its state dict indexes them."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        names[id(parameter)]
        for group in optimizer.param_groups
        for parameter in cast("list[Tensor]", group["params"])
    ]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _golden_path(*, test_file: str, name: str, host: str) -> Path:
    """Return ``testdata/<name>[_<host>].txt`` beside ``test_file``."""
    stem = f"{name}_{host}" if host else name
    return Path(test_file).resolve().parent / "testdata" / f"{stem}.txt"


def _missing_golden(path: Path) -> str:
    """Return the skip reason for a host that has no ``path`` golden."""
    return f"no {path.stem} golden; mint it on this host with --regenerate-golden"
