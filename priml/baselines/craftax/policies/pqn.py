"""A recurrent Q-network and its exploration: value learning without a replay buffer.

Deep Q-learning normally needs two crutches. A replay buffer decorrelates
consecutive samples, and a target network holds the regression target still
while the online network chases it -- without them, the network bootstraps
from itself and diverges.

This has neither, and the claim is that at enough parallelism it does not need
them. A thousand simultaneous workers already supply decorrelated samples, so
the buffer is redundant; and normalization plus a multi-step Q(lambda) target
keeps the regression stable enough to drop the target network. What remains is
one network trained on its own fresh experience.

Two pieces do that work. Batch renormalization on the raw observation absorbs
the distribution shift that comes from the policy changing under its own
training, and an LSTM carries the history the environment does not show. The
previous ACTION is fed in alongside the observation, which a Q-learner needs
and a policy-gradient method does not: the value of a state depends on what
the agent just tried, and epsilon-greedy exploration makes that unpredictable
from the observation alone.

The port's actor (``rollout.Rollout``) runs the network as it runs any policy.
Its fused row is every action's value, then their maximum, the greedy value,
which the rollout stores as each step's value. Three pieces fit it to that
actor:

- :class:`PreviousAction` is the rollout's per-step feature: the action each
  environment last played, read from the env's action buffer, which the
  network one-hot encodes. It carries over an episode's end, as the
  reference's does; the carry, which the actor zeroes there, could not hold it.
- :class:`EpsilonGreedy` samples the epsilon-greedy distribution by the
  wrapped sampler's inverse CDF: one Philox draw per step, its exploration
  rate a function of the draw count, so a captured step graph decays it
  without the host.
- :class:`GreedySampler` takes the greedy action, as an evaluation does.

References:
    https://arxiv.org/abs/2407.04811
        Gallici et al. 2024. Simplifying deep temporal difference learning.
    https://github.com/mttga/purejaxql
        The reference implementation, ``pqn_rnn_craftax.py``.

"""

from __future__ import annotations

from dataclasses import field, replace
from typing import TYPE_CHECKING

import math
import sys

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.baselines.craftax.rollout import PhiloxSampler, Sampled, Sampler
from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    resolve_dtype,
    traffic,
)
from priml.loss.policy_gradient import TorchPPO
from priml.model.norm import BatchRenorm, LayerNorm


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.baselines.craftax.world_model.feature import RecentDecisions


class RecurrentQNetwork(nn.Module):
    """An LSTM Q-network over renormalized observations and the previous action.

    The carry is ``[2, envs, channels_hidden]``: the hidden state, then the
    cell state. The feature is the previous action's id, ``[..., 1]``.
    """

    class Config(Fig["RecurrentQNetwork"]):
        """Configure the Q-network."""

        observation_size: int = 8_268
        """Width of one observation; the environment's own width."""

        num_actions: int = 43
        """Size of the discrete action space."""

        channels_hidden: int = 512
        """Width of the encoder and of the recurrent state."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one recurrent step of one worker.

            A token is one environment step: the observation and the previous
            action in, both carried tensors updated, a value per action and
            their maximum out. The step runs the whole cell once whatever the
            history, so ``bytes_state`` is the hidden and cell vectors carried
            to the next step, and both are differentiable (every step but a
            window's first), so the state gates' adjoint is counted in full.

            The renormalization and the layer norm cost themselves. The one-hot
            previous action is written, not computed: one element moved per
            action, no gradient. The cell is a biased ``[width + actions] -> [4
            width]`` matmul and a biased ``[width] -> [4 width]`` matmul, then
            per unit: four gates (an add and a sigmoid or tanh each), the cell
            update (two products, an add), and the output (a tanh, a product):
            thirteen operations. The adjoint reuses the saved gates:
            twenty-two. Scalar-region I/O reads eight projected gates and the
            cell, writing two states; backward reads four gates, two cell
            values, and two gradients, writing eight projected-gate gradients
            and the old-cell gradient. The episode reset selects each carried
            unit each way. The greedy value is a max over the actions, and the
            fused row a concatenation.

            This is the model root: every parameter is shared by the concrete
            ``batch_size * seq_len`` rows in this invocation.

            Args:
              seq_len: Rows per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            batch = kwargs
            rows = seq_len * batch_size
            dt = dtype
            width, actions = self.channels_hidden, self.num_actions
            normalize = cost(
                _renorm_config(self.observation_size),
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **batch,
            )
            encoder = matmul_cost(
                channels_in=self.observation_size,
                channels_out=width,
                bias=True,
                rows=rows,
                dtype=dt,
            )
            encoder_norm = cost(
                LayerNorm.Config(width, elementwise_affine=True),
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **batch,
            ) + elementwise_cost(
                primal=rows * width,
                adjoint=rows * width,
                channels=width,
                rows=rows,
                dtype=dt,
            )
            one_hot = traffic("primal", "selection", elements=rows * actions, dtype=dt)
            concatenate = traffic(
                "primal",
                "elementwise",
                elements=2 * rows * (width + actions),
                dtype=dt,
            )
            reset = elementwise_cost(
                primal=2 * rows * width,
                adjoint=2 * rows * width,
                channels=2 * width,
                inputs=2,
                rows=rows,
                dtype=dt,
            )
            gates = matmul_cost(
                channels_in=width + actions,
                channels_out=4 * width,
                bias=True,
                rows=rows,
                dtype=dt,
            ) + matmul_cost(
                channels_in=width,
                channels_out=4 * width,
                bias=True,
                rows=rows,
                dtype=dt,
            )
            cell = elementwise_cost(
                primal=13 * rows * width,
                adjoint=22 * rows * width,
                channels=width,
                inputs=9,
                outputs=2,
                adjoint_inputs=8,
                adjoint_outputs=9,
                rows=rows,
                dtype=dt,
            )
            head = matmul_cost(
                channels_in=width,
                channels_out=actions,
                bias=True,
                rows=rows,
                dtype=dt,
            )
            greedy = reduction_cost(
                input_elements=rows * actions,
                output_groups=rows,
                dtype=dt,
            )
            fused = traffic(
                "primal",
                "elementwise",
                elements=2 * rows * (actions + 1),
                dtype=dt,
            )
            return replace(
                normalize
                + encoder
                + encoder_norm
                + one_hot
                + concatenate
                + reset
                + gates
                + cell
                + head
                + greedy
                + fused,
                bytes_state=resolve_dtype(dtype).itemsize * batch_size * 2 * width,
            )

    def __init__(self, config: Config) -> None:
        """Build the encoder, the recurrent cell, and the value head.

        Args:
          config: Geometry of the network.

        Raises:
          ValueError: A dimension is not positive.

        """
        super().__init__()
        if (
            min(config.observation_size, config.num_actions, config.channels_hidden)
            <= 0
        ):
            raise ValueError("RecurrentQNetwork dimensions must be positive")

        self.channels_hidden = config.channels_hidden
        self.num_actions = config.num_actions

        self.normalize = _renorm_config(config.observation_size).make()

        self.encoder = nn.Linear(config.observation_size, config.channels_hidden)
        self.encoder_norm = nn.LayerNorm(config.channels_hidden)
        # The previous action joins the encoding, not the observation: it is
        # one-hot and would otherwise pass through renormalization, whose
        # running statistics have no business tracking an action histogram.
        self.cell = nn.LSTMCell(
            config.channels_hidden + config.num_actions,
            config.channels_hidden,
        )
        self.head = nn.Linear(config.channels_hidden, config.num_actions)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the weights, and of the rollout's stored observations."""
        return self.head.weight.dtype

    def initial_state(
        self,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Return the zeroed carry for a fresh set of workers.

        Args:
          num_envs: Parallel workers the carry covers.
          device: Device the carry lives on.

        Returns:
          state: ``[2, num_envs, channels_hidden]``: hidden, then cell.

        """
        return torch.zeros(2, num_envs, self.channels_hidden, device=device)

    def forward_fused(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor | None,
        *,
        carry: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Advance the recurrence one step and value every action.

        Args:
          observations: ``[envs, observation_size]``.
          state: The carry, ``[2, envs, channels_hidden]``.
          episode_start: Nonzero where the carry resets first, ``[envs]``;
            None when the caller has already zeroed those rows.
          carry: Where the next carry is written; a new tensor if None.
          features: The previous action's id, ``[envs, 1]``.

        Returns:
          decoded: ``[envs, num_actions + 1]``: every action's value, then
            their maximum.
          state: The next carry; ``carry`` if given.

        """
        previous = _previous_action(features)
        if episode_start is not None:
            state = _reset(state, episode_start)
        hidden, cell, values = self._step(
            state,
            self.normalize(observations.to(self.dtype)),
            previous,
        )
        decoded = torch.cat((values, values.amax(dim=-1, keepdim=True)), dim=-1)
        if carry is None:
            return decoded, torch.stack((hidden, cell))
        carry[0].copy_(hidden)
        carry[1].copy_(cell)
        return decoded, carry

    def forward_sequence(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor,
        *,
        actions: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run a window, for the learner to differentiate.

        The renormalization takes the whole window as one batch: in training,
        the statistics over every environment and step normalize it and move
        the running ones once. In inference it reads the running ones, as
        :meth:`forward_fused` does, so a window scores what stepping it would.

        Args:
          observations: ``[batch, time, observation_size]``.
          state: The carry before the first step, ``[2, batch, channels_hidden]``.
          episode_start: Nonzero where the carry resets, ``[batch, time]``.
          actions: Unused: there is no auxiliary loss.
          features: The previous action's id, ``[batch, time, 1]``.

        Returns:
          decoded: ``[batch, time, num_actions + 1]``: every action's value,
            then their maximum.
          final: The carry after the last step.
          auxiliary_loss: Zero, 0-dim fp32.

        """
        del actions
        # One batch for the whole window, as purejaxql renormalizes its network's
        # ``[time, batch]`` input (``pqn_rnn_craftax.py`` line 84, reducing every
        # axis but the features, ``utils/batch_renorm.py`` line 46). Step by
        # step, the running statistics would move once per step rather than once
        # per window, so their momentum and their 1,000-call warmup would run a
        # window's length of times faster.
        normalized = self.normalize(observations.to(self.dtype))
        previous = _previous_action(features)
        values: list[Tensor] = []
        # ``unbind``, not an index per step: each indexed step's backward writes a
        # zeroed gradient of the whole window, which took the learner from 1.7 s to
        # 2.9 s an update at 128 steps (measured on an H200).
        for step, step_normalized in enumerate(normalized.unbind(1)):
            state = _reset(state, episode_start[:, step])
            hidden, cell, step_values = self._step(
                state,
                step_normalized,
                previous[:, step],
            )
            state = torch.stack((hidden, cell))
            values.append(step_values)
        stacked = torch.stack(values, dim=1)
        decoded = torch.cat((stacked, stacked.amax(dim=-1, keepdim=True)), dim=-1)
        return decoded, state, stacked.new_zeros((), dtype=torch.float32)

    # A step at a time after the window's renormalization, at the actor's shapes:
    # one product over the whole window rounded differently (measured), and the
    # rescore then no longer equaled the actor's values bit for bit.
    def _step(
        self,
        state: Tensor,
        normalized: Tensor,
        previous: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode, advance the cell and value every action; return hidden, cell, values."""
        encoded = self.encoder_norm(self.encoder(normalized)).relu()
        one_hot = nn.functional.one_hot(previous.long(), self.num_actions)
        hidden, cell = self.cell(
            torch.cat((encoded, one_hot.to(encoded.dtype)), dim=-1),
            (state[0], state[1]),
        )
        return hidden, cell, self.head(hidden)


def epsilon_at(
    update: int,
    *,
    total_updates: int,
    start: float = 1.0,
    finish: float = 0.005,
    decay_fraction: float = 0.1,
) -> float:
    """Return the exploration rate for one update, decayed linearly.

    Exploration is front-loaded: the rate falls from ``start`` to ``finish``
    over the first ``decay_fraction`` of the run and stays there. A Q-learner
    has no entropy bonus keeping it curious, so this schedule is the whole of
    its exploration, and a run that decayed across its full length would still
    be acting half-randomly at the end.

    Args:
      update: Zero-based update index.
      total_updates: Updates in the whole run.
      start: Initial probability of a random action.
      finish: Floor the probability decays to.
      decay_fraction: Fraction of the run spent decaying.

    Returns:
      epsilon: Probability of taking a random action.

    Raises:
      ValueError: The schedule geometry is invalid.

    """
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    if decay_fraction <= 0.0 or decay_fraction > 1.0:
        raise ValueError("decay_fraction must be in (0, 1]")
    horizon = max(1.0, decay_fraction * total_updates)
    progress = min(1.0, update / horizon)
    return start + (finish - start) * progress


class EpsilonGreedy:
    """Take a uniformly random action at rate epsilon, else the greedy one.

    Each step draws once from the epsilon-greedy distribution -- ``epsilon /
    legal`` on every legal action and the rest on the greedy one -- by the
    wrapped sampler's inverse CDF over its log-probabilities: the reference's
    two-draw rule's distribution, from one Philox draw. The rate is :func:`epsilon_at`
    of the update, which the draw count gives: every agent draws once a step,
    ``steps_per_update`` steps an update. The decoder's value column, the
    greedy value, passes through as the step's value.
    """

    class Config(Fig["EpsilonGreedy"]):
        """The exploration schedule and the sampler that draws from it."""

        start: float = 1.0
        """Initial probability of taking a random action."""

        finish: float = 0.005
        """Floor the exploration rate decays to."""

        decay_fraction: float = 0.1
        """Fraction of the run spent decaying the exploration rate."""

        steps_per_update: int = -1
        """Draws each agent takes per update, the rollout's horizon; -1 takes
        the train step's."""

        total_updates: int = -1
        """Updates in the run, the schedule's horizon; -1 takes the train
        step's budget."""

        sampler: Makeable[Sampler] = field(default_factory=PhiloxSampler.Config)
        """Draws each action from the epsilon-greedy distribution's
        log-probabilities, and owns the streams."""

    def __init__(self, config: Config) -> None:
        """Validate the schedule and build the sampler.

        Args:
          config: The schedule and the sampler.

        Raises:
          ValueError: A rate is outside ``[0, 1]``, or a count is not positive,
            or ``decay_fraction`` is outside ``(0, 1]``.

        """
        for name, rate in (("start", config.start), ("finish", config.finish)):
            if math.isnan(rate) or rate < 0.0 or rate > 1.0:
                raise ValueError(f"{name} must be in [0, 1], not {rate}")
        if config.steps_per_update <= 0 or config.total_updates <= 0:
            raise ValueError("steps_per_update and total_updates must be positive")
        fraction = config.decay_fraction
        if math.isnan(fraction) or fraction <= 0.0 or fraction > 1.0:
            raise ValueError(f"decay_fraction must be in (0, 1], not {fraction}")
        self.config = config
        self.sampler = config.sampler.make()

    def draws(self, num_agents: int, *, device: torch.device | str) -> Tensor:
        """Return the wrapped sampler's fresh per-agent draw count."""
        return self.sampler.draws(num_agents, device=device)

    def epsilon(self, update: int) -> float:
        """Return the exploration rate of an update, as each of its steps draws it.

        Args:
          update: Zero-based update index.

        Returns:
          epsilon: Probability of taking a random action.

        """
        config = self.config
        return epsilon_at(
            update,
            total_updates=config.total_updates,
            start=config.start,
            finish=config.finish,
            decay_fraction=config.decay_fraction,
        )

    def __call__(
        self,
        decoded: Tensor,
        action_mask: Tensor,
        draws: Tensor,
        *,
        buffer: int,
        dtype: torch.dtype | None = None,
    ) -> Sampled:
        """Draw one epsilon-greedy action per agent and advance every agent's stream.

        Args:
          decoded: ``[agents, num_actions + 1]``: the action values, then the
            greedy value.
          action_mask: ``[agents, num_actions]``, nonzero where legal.
          draws: The buffer's draw counts from :meth:`draws`; incremented.
          buffer: Which buffer these agents are, for the stream seed.
          dtype: The log-probabilities' and values' dtype; ``decoded``'s if
            None.

        Returns:
          sampled: Actions, their log-probabilities under the epsilon-greedy
            distribution, and the greedy values.

        """
        num_actions = decoded.shape[1] - 1
        legal = action_mask != 0
        greedy = torch.where(legal, decoded[:, :num_actions].float(), -math.inf).argmax(
            dim=1,
            keepdim=True,
        )
        epsilon = self._epsilons(draws)[:, None]
        # The illegal actions take a share too, which nothing reads: the wrapped
        # sampler masks them, as it masks any decoder's.
        spread = (epsilon / legal.sum(dim=1, keepdim=True)).expand(-1, num_actions)
        probability = spread.scatter_add(1, greedy, 1.0 - epsilon)
        # An action of probability zero gets the masked logit, not -inf, which
        # the samplers' logsumexp would subtract from itself.
        logits = probability.log().clamp_min(TorchPPO.Config.MASKED_LOGIT)
        fused = torch.cat((logits, decoded[:, num_actions:].float()), dim=1)
        return self.sampler(fused, action_mask, draws, buffer=buffer, dtype=dtype)

    # The host's arithmetic, in float64 on the device: the rates are
    # :func:`epsilon_at`'s, rounded once to fp32.
    def _epsilons(self, draws: Tensor) -> Tensor:
        """Return each agent's exploration rate from its draw count, fp32 ``[agents]``."""
        config = self.config
        horizon = max(1.0, config.decay_fraction * config.total_updates)
        updates = torch.div(draws, config.steps_per_update, rounding_mode="floor")
        progress = (updates.double() / horizon).clamp(max=1.0)
        return (config.start + (config.finish - config.start) * progress).float()


class GreedySampler:
    """Take every agent's greedy action: the highest-valued legal one."""

    class Config(Fig["GreedySampler"]):
        """Nothing to configure: the greedy action draws nothing."""

    def __init__(self, config: Config) -> None:
        del config

    def draws(self, num_agents: int, *, device: torch.device | str) -> Tensor:
        """Return a per-agent draw count, which nothing advances.

        Args:
          num_agents: Agents in the buffer.
          device: Where the counter lives.

        Returns:
          draws: ``[num_agents]`` int64 zeros.

        """
        return torch.zeros(num_agents, dtype=torch.int64, device=device)

    def __call__(
        self,
        decoded: Tensor,
        action_mask: Tensor,
        draws: Tensor,
        *,
        buffer: int,
        dtype: torch.dtype | None = None,
    ) -> Sampled:
        """Take each agent's greedy action.

        Args:
          decoded: ``[agents, num_actions + 1]``: the action values, then the
            greedy value.
          action_mask: ``[agents, num_actions]``, nonzero where legal.
          draws: Unused: nothing is drawn.
          buffer: Unused: nothing is drawn.
          dtype: The log-probabilities' and values' dtype; ``decoded``'s if
            None.

        Returns:
          sampled: The greedy actions, log-probability zero, and the greedy
            values.

        """
        del draws, buffer
        num_actions = decoded.shape[1] - 1
        actions = torch.where(
            action_mask != 0,
            decoded[:, :num_actions],
            -math.inf,
        ).argmax(dim=1)
        dtype = dtype or decoded.dtype
        return Sampled(
            actions=actions.float(),
            logprobs=torch.zeros(actions.shape, dtype=dtype, device=decoded.device),
            values=decoded[:, num_actions].to(dtype, copy=True),
        )


class PreviousAction:
    """The action each environment last played: the per-step feature the network reads.

    The rollout uploads each step's previous action from the env's action
    buffer; this feature passes it through as ``[agents, 1]``, so the rollout
    stores it beside the observation for the learner. It keeps no history, so
    every hook is free and a step needs no upkeep.

    Attributes:
      width: One float, the action's id.
      hook_interval: Unbounded: the feature never needs room.
      joint: False: nothing trains it.
      context_decisions: Zero: it keeps no context.

    """

    class Config(Fig["PreviousAction"]):
        """Nothing to configure: the env's action buffer is the feature."""

    def __init__(self, config: Config) -> None:
        del config
        self.width = 1
        self.hook_interval = sys.maxsize
        self.joint = False
        self.context_decisions = 0

    def make_engine(self, *, rows: int, device: torch.device) -> PreviousActionStep:
        """Return the stateless step; it serves any rows on any device."""
        del rows, device
        return PreviousActionStep()

    def history_archive(
        self,
        entries: int,
        *,
        device: torch.device,
    ) -> dict[str, Tensor] | None:
        """Return None: there is no history to archive."""
        del entries, device
        return None


class PreviousActionStep:
    """One buffer's :class:`PreviousAction`: the uploaded action, as a column."""

    @property
    def plan(self) -> int:
        """Zero: the step's shape never changes."""
        return 0

    def __call__(
        self,
        observation: Tensor,
        terminals: Tensor,
        previous_action: Tensor,
    ) -> Tensor:
        """Return each row's previous action as ``[agents, 1]``.

        Args:
          observation: Unused: the feature is the action alone.
          terminals: Unused: the action carries over an episode's end.
          previous_action: The action that led to the observation, fp32
            ``[agents]``.

        Returns:
          feature: ``previous_action[:, None]``.

        """
        del observation, terminals
        return previous_action[:, None]

    def capture_state(self) -> list[Tensor]:
        """Return nothing: a warmup changes no state of this step."""
        return []

    def ensure_room(self, steps: int) -> dict[str, float]:
        """Report nothing: there is no room to make."""
        del steps
        return {}

    def reset(self) -> None:
        """Do nothing: there is no episode to begin."""

    def begin_window(self, rows: Tensor) -> None:
        """Do nothing: there is no history to window."""
        del rows

    def context(self) -> tuple[Tensor, Tensor]:
        """Refuse: only a joint feature's rollout stores contexts.

        Returns:
          context: Never; a stateless feature has none.

        Raises:
          ValueError: Always.

        """
        raise ValueError("the previous action keeps no context")

    def last_decisions(self, n: int) -> RecentDecisions:
        """Refuse: only a joint feature's rollout stores decisions.

        Args:
          n: Decisions asked for.

        Returns:
          decisions: Never; a stateless feature keeps none.

        Raises:
          ValueError: Always.

        """
        del n
        raise ValueError("the previous action keeps no decisions")

    def rebuild(self) -> None:
        """Refuse: only a joint feature's history is rebuilt.

        Raises:
          ValueError: Always.

        """
        raise ValueError("the previous action keeps no history to rebuild")

    def save_history(
        self,
        archive: Mapping[str, Tensor],
        entries: Tensor,
        rows: slice,
        *,
        previous_action: Tensor,
        fresh: Tensor,
    ) -> None:
        """Refuse: :meth:`PreviousAction.history_archive` keeps no archive to write.

        Args:
          archive: The archive to write.
          entries: The archive row of each saving row.
          rows: The saving rows.
          previous_action: The action each took last.
          fresh: Where the next observation begins an episode.

        Raises:
          ValueError: Always.

        """
        del archive, entries, rows, previous_action, fresh
        raise ValueError("the previous action keeps no history to save")

    def restore_history(
        self,
        archive: Mapping[str, Tensor] | None,
        entries: Tensor,
    ) -> Tensor | None:
        """Return None: a restored row reads the env's previous action."""
        del archive, entries
        return None

    def release(self) -> None:
        """Do nothing: the step holds no memory."""


def _previous_action(features: Tensor | None) -> Tensor:
    """Return the previous action's ids from the feature, ``[...]``."""
    if features is None:
        raise ValueError("RecurrentQNetwork reads the previous action as its feature")
    return features.squeeze(-1)


def _reset(state: Tensor, episode_start: Tensor) -> Tensor:
    """Zero the carry of every environment whose episode starts here."""
    return torch.where(episode_start[None, :, None] != 0, 0.0, state)


def _renorm_config(observation_size: int) -> BatchRenorm.Config:
    """Configure the observation renormalization; one source for build and cost."""
    normalize = BatchRenorm.Config()
    normalize.channels_in = observation_size
    return normalize
