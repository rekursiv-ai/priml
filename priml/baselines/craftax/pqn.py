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

from dataclasses import replace
from typing import override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.model.cost import (
    Bytes,
    Compute,
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
)
from priml.model.norm import BatchRenorm, LayerNorm


class RecurrentQNetwork(nn.Module):
    """An LSTM Q-network over renormalized observations.

    Attributes:
      channels_in: Width of the encoder and of each recurrent state.
      num_actions: Size of the discrete action space.

    """

    class Config(Fig["RecurrentQNetwork"]):
        """Configure the Q-network."""

        observation_size: int = 8_268
        """Width of one observation; the environment's own width."""

        num_actions: int = 43
        """Size of the discrete action space."""

        channels_in: int = 512
        """Width of the encoder and of the recurrent state."""

        def cost(
            self,
            *,
            seq_len: int = 1,
            batch_size: int = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Price one recurrent step of one worker.

            A token is one environment step: the observation and the previous
            action in, both carried tensors updated, a Q-value per action
            out. The step runs the whole cell once whatever the history, so
            ``bytes_state`` is the hidden and cell vectors carried to the
            next step, and both are differentiable (every step but a
            window's first), so the state gates' adjoint is counted in full.

            The renormalization and the layer norm price themselves. The
            one-hot previous action is written, not computed: one element
            moved per action, no gradient. The cell is a biased
            ``[width + actions] -> [4 width]`` matmul and a biased
            ``[width] -> [4 width]`` matmul, then per unit: four gates (an add
            and a sigmoid or tanh each), the cell update (two products, an
            add), and the output (a tanh, a product): thirteen operations.
            The adjoint reuses the saved gates: twenty-two. Scalar-region I/O
            reads eight projected gates and the cell, writing two states;
            backward reads four gates, two cell values, and two gradients,
            writing eight projected-gate gradients and the old-cell gradient.
            The episode reset is one product per carried unit each way.

            This is the model root: it states the batch geometry itself, so
            a ``rows`` already on the bus is discarded and every parameter is
            shared by ``batch_size * seq_len`` tokens.

            Args:
              seq_len: Steps per worker in one pass.
              batch_size: Workers stepped together.
              itemsize: Uniform bytes per tensor element.
              **kwargs: The rest of the open message bus, forwarded to every
                child.

            Returns:
              cost: Per-token cost of this module.

            """
            kwargs.pop("rows", None)
            rows = seq_len * batch_size
            width, actions = self.channels_in, self.num_actions
            normalize = cost(
                _renorm_config(self.observation_size),
                rows=rows,
                itemsize=itemsize,
                **kwargs,
            )
            encoder = matmul_cost(
                channels_in=self.observation_size,
                channels_out=width,
                bias=True,
                rows=rows,
                itemsize=itemsize,
            )
            encoder_norm = cost(
                LayerNorm.Config(width, elementwise_affine=True),
                rows=rows,
                itemsize=itemsize,
                **kwargs,
            ) + elementwise_cost(
                primal=width,
                adjoint=width,
                channels=width,
                itemsize=itemsize,
            )
            one_hot = Cost(primal=Compute(bytes=Bytes(selection=itemsize * actions)))
            concatenate = Cost(
                primal=Compute(
                    bytes=Bytes(elementwise=2 * itemsize * (width + actions)),
                ),
            )
            reset = elementwise_cost(
                primal=2 * width,
                adjoint=2 * width,
                channels=2 * width,
                inputs=2,
                itemsize=itemsize,
            )
            gates = matmul_cost(
                channels_in=width + actions,
                channels_out=4 * width,
                bias=True,
                rows=rows,
                itemsize=itemsize,
            ) + matmul_cost(
                channels_in=width,
                channels_out=4 * width,
                bias=True,
                rows=rows,
                itemsize=itemsize,
            )
            cell = elementwise_cost(
                primal=13 * width,
                adjoint=22 * width,
                channels=width,
                inputs=9,
                outputs=2,
                adjoint_inputs=8,
                adjoint_outputs=9,
                itemsize=itemsize,
            )
            head = matmul_cost(
                channels_in=width,
                channels_out=actions,
                bias=True,
                rows=rows,
                itemsize=itemsize,
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
                + head,
                bytes_state=itemsize * 2 * width,
            )

    def __init__(self, config: Config) -> None:
        """Build the encoder, the recurrent cell, and the value head.

        Args:
          config: Geometry of the network.

        Raises:
          ValueError: A dimension is not positive.

        """
        super().__init__()
        if min(config.observation_size, config.num_actions, config.channels_in) <= 0:
            raise ValueError("RecurrentQNetwork dimensions must be positive")

        self.channels_in = config.channels_in
        self.num_actions = config.num_actions

        self.normalize = _renorm_config(config.observation_size).make()

        self.encoder = nn.Linear(config.observation_size, config.channels_in)
        self.encoder_norm = nn.LayerNorm(config.channels_in)
        # The previous action joins the encoding, not the observation: it is
        # one-hot and would otherwise pass through renormalization, whose
        # running statistics have no business tracking an action histogram.
        self.cell = nn.LSTMCell(
            config.channels_in + config.num_actions,
            config.channels_in,
        )
        self.head = nn.Linear(config.channels_in, config.num_actions)

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
          hidden: Zeroed hidden state, ``[envs, channels_in]``.
          cell: Zeroed cell state, ``[envs, channels_in]``.

        """
        zeros = torch.zeros(num_envs, self.channels_in, device=device)
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
          state: Hidden and cell state, each ``[envs, channels_in]``.
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
    if decay_fraction <= 0.0 or decay_fraction > 1.0:
        raise ValueError("decay_fraction must be in (0, 1]")
    horizon = max(1.0, decay_fraction * total_updates)
    progress = min(1.0, update / horizon)
    return start + (finish - start) * progress


def _renorm_config(observation_size: int) -> BatchRenorm.Config:
    """Configure the observation renormalization; one source for build and cost."""
    normalize = BatchRenorm.Config()
    normalize.channels_in = observation_size
    return normalize
