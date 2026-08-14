"""A recurrent Q-network: value learning without a replay buffer.

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

References:
    https://arxiv.org/abs/2407.04811
        Gallici et al. 2024. Simplifying deep temporal difference learning.
    https://github.com/mttga/purejaxql
        The reference implementation this ports.

"""

from __future__ import annotations

from typing import override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.model.norm import BatchRenorm


class RecurrentQNetwork(nn.Module):
    """An LSTM Q-network over renormalized observations.

    Attributes:
      hidden_size: Width of the encoder and of each recurrent state.
      num_actions: Size of the discrete action space.

    """

    class Config(Fig["RecurrentQNetwork"]):
        """Configure the Q-network."""

        observation_size: int = 8_268
        """Width of one observation; the environment's own width."""

        num_actions: int = 43
        """Size of the discrete action space."""

        hidden_size: int = 512
        """Width of the encoder and of the recurrent state."""

    def __init__(self, config: Config) -> None:
        """Build the encoder, the recurrent cell, and the value head.

        Args:
          config: Geometry of the network.

        Raises:
          ValueError: A dimension is not positive.

        """
        super().__init__()
        if min(config.observation_size, config.num_actions, config.hidden_size) <= 0:
            raise ValueError("RecurrentQNetwork dimensions must be positive")

        self.hidden_size = config.hidden_size
        self.num_actions = config.num_actions

        normalize = BatchRenorm.Config()
        normalize.channels_in = config.observation_size
        self.normalize = normalize.make()

        self.encoder = nn.Linear(config.observation_size, config.hidden_size)
        self.encoder_norm = nn.LayerNorm(config.hidden_size)
        # The previous action joins the encoding, not the observation: it is
        # one-hot and would otherwise pass through renormalization, whose
        # running statistics have no business tracking an action histogram.
        self.cell = nn.LSTMCell(
            config.hidden_size + config.num_actions,
            config.hidden_size,
        )
        self.head = nn.Linear(config.hidden_size, config.num_actions)

    def initial_state(
        self,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Return the zeroed recurrent state for a fresh set of workers.

        Args:
          num_envs: Parallel workers the state covers.
          device: Device the state lives on.

        Returns:
          hidden: Zeroed hidden state, ``[envs, hidden_size]``.
          cell: Zeroed cell state, ``[envs, hidden_size]``.

        """
        zeros = torch.zeros(num_envs, self.hidden_size, device=device)
        return zeros, zeros.clone()

    @override
    def forward(self, observation: Tensor) -> Tensor:
        """Score every action from a fresh state, with no previous action.

        The stateless surface, so a probe works unchanged. A real rollout uses
        :meth:`step`.

        Args:
          observation: Batched observations, ``[batch, observation_size]``.

        Returns:
          q_values: Value of each action, ``[batch, num_actions]``.

        """
        _, q_values = self.step(
            self.initial_state(observation.shape[0], device=observation.device),
            observation,
            torch.zeros(
                observation.shape[0],
                dtype=torch.int64,
                device=observation.device,
            ),
            torch.zeros(
                observation.shape[0],
                dtype=torch.bool,
                device=observation.device,
            ),
        )
        return q_values

    def step(
        self,
        state: tuple[Tensor, Tensor],
        observation: Tensor,
        previous_action: Tensor,
        previous_done: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """Advance the recurrence one step and value every action.

        Args:
          state: Hidden and cell state, each ``[envs, hidden_size]``.
          observation: Current observations, ``[envs, observation_size]``.
          previous_action: The action taken into this state, ``[envs]``.
          previous_done: Whether the PRECEDING transition ended an episode.

        Returns:
          state: The updated recurrent state.
          q_values: Value of each action, ``[envs, num_actions]``.

        """
        encoded = self.encoder_norm(self.encoder(self.normalize(observation))).relu()
        encoded = torch.cat(
            (
                encoded,
                nn.functional.one_hot(previous_action, self.num_actions).to(
                    encoded.dtype,
                ),
            ),
            dim=-1,
        )
        keep = ~previous_done[:, None]
        hidden, cell = self.cell(encoded, (state[0] * keep, state[1] * keep))
        return (hidden, cell), self.head(hidden)

    def sequence(
        self,
        state: tuple[Tensor, Tensor],
        observation: Tensor,
        previous_action: Tensor,
        previous_done: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """Run a whole time window, for the gradient step.

        Args:
          state: Recurrent state at the window's first step.
          observation: Time-major observations, ``[time, envs, obs]``.
          previous_action: Time-major preceding actions, ``[time, envs]``.
          previous_done: Time-major preceding-transition terminal flags.

        Returns:
          state: The recurrent state after the last step.
          q_values: Time-major action values, ``[time, envs, num_actions]``.

        """
        values: list[Tensor] = []
        for index in range(observation.shape[0]):
            state, step_values = self.step(
                state,
                observation[index],
                previous_action[index],
                previous_done[index],
            )
            values.append(step_values)
        return state, torch.stack(values)


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
    if not 0.0 < decay_fraction <= 1.0:
        raise ValueError("decay_fraction must be in (0, 1]")
    horizon = max(1.0, decay_fraction * total_updates)
    progress = min(1.0, update / horizon)
    return start + (finish - start) * progress
