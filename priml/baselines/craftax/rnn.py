"""A recurrent actor-critic: memory as a single running vector.

The cheapest way to give a policy a past. A GRU carries one hidden vector
forward, updating it at each step, so the network's view of history costs the
same whatever the horizon -- unlike the transformer in :mod:`gtrxl`, whose
attention grows with the window it remembers.

That is the trade this model exists to measure. A vector must overwrite to
learn, so it forgets gradually and cannot look back at a specific earlier
moment; attention keeps every remembered step addressable and pays for it in
memory and compute. Craftax rewards long sequential plans, so which one wins
is an empirical question, not an obvious one.

The recurrence is RESET-AWARE: the hidden state is zeroed wherever the
previous transition ended an episode. Without that, a policy would begin each
new world remembering the one it just died in, which is worse than beginning
with no memory at all.

References:
    https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py
        The official Craftax PPO-RNN baseline this ports.
    https://arxiv.org/abs/1406.1078
        Cho et al. 2014. Learning phrase representations using RNN
        encoder-decoder.

"""

from __future__ import annotations

from dataclasses import replace
from typing import override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    elementwise_cost,
    matmul_cost,
    resolve_dtype,
)


class ActorCriticRNN(nn.Module):
    """A GRU actor-critic whose recurrent state clears at episode boundaries.

    Attributes:
      channels_in: Width of the embedding, the recurrent state, and both
        heads. One number, as the reference uses.

    """

    class Config(Fig["ActorCriticRNN"]):
        """Configure the recurrent network."""

        observation_size: int = 8_268
        """Width of one observation; the environment's own width."""

        num_actions: int = 43
        """Size of the discrete action space."""

        channels_in: int = 512
        """Width of the embedding, the GRU state, and each head."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one recurrent step of one worker.

            A token is one environment step: the observation in, the carried
            state updated, logits and a value out. The step runs the whole
            cell once whatever the history, so ``bytes_state`` is the one
            hidden vector carried to the next step, and the carried state is
            differentiable (every step but a window's first), so the state
            gates' adjoint is counted in full.

            The cell is two biased ``[width] -> [3 width]`` matmuls -- input
            gates and state gates -- then per unit: two sigmoid gates (an add
            and a sigmoid each), the candidate (a product, an add, a tanh),
            and the interpolation (a subtraction, two products, an add):
            eleven operations. The adjoint reuses the saved gates: seventeen.
            Scalar-region I/O reads six projected gates plus state and writes
            state; backward reads saved gates/state and the gradient, writing
            six projected-gate gradients and the state gradient.
            The episode reset is one select per state unit each way.

            This is the model root: every parameter is shared by
            ``batch_size * seq_len`` tokens, so a caller's own ``rows`` is
            replaced by the batch's tokens.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            rows = seq_len * batch_size
            dt = dtype
            width = self.channels_in
            embed = matmul_cost(
                channels_in=self.observation_size,
                channels_out=width,
                bias=True,
                rows=rows,
                dtype=dt,
            ) + elementwise_cost(
                primal=width,
                adjoint=width,
                channels=width,
                dtype=dt,
            )
            reset = elementwise_cost(
                primal=width,
                adjoint=width,
                channels=width,
                inputs=2,
                dtype=dt,
            )
            gates = matmul_cost(
                channels_in=width,
                channels_out=3 * width,
                bias=True,
                rows=rows,
                dtype=dt,
            ).tile(2, copies=2)
            cell = elementwise_cost(
                primal=11 * width,
                adjoint=17 * width,
                channels=width,
                inputs=7,
                adjoint_inputs=6,
                adjoint_outputs=7,
                dtype=dt,
            )
            heads = sum(
                (
                    _head_cost(
                        channels_in=width,
                        output_size=output_size,
                        rows=rows,
                        dt=dt,
                    )
                    for output_size in (self.num_actions, 1)
                ),
                Cost(),
            )
            return replace(
                embed + reset + gates + cell + heads,
                bytes_state=resolve_dtype(dtype).itemsize * width,
            )

    def __init__(self, config: Config) -> None:
        """Build the embedding, the recurrent cell, and both heads.

        Args:
          config: Geometry of the network.

        Raises:
          ValueError: A dimension is not positive.

        """
        super().__init__()
        if min(config.observation_size, config.num_actions, config.channels_in) <= 0:
            raise ValueError("ActorCriticRNN dimensions must be positive")

        self.channels_in = config.channels_in
        self.embed = nn.Sequential(
            _dense(config.observation_size, config.channels_in, gain=2.0**0.5),
            nn.ReLU(),
        )
        self.cell = nn.GRUCell(config.channels_in, config.channels_in)
        self.actor = _head(
            channels_in=config.channels_in,
            output_size=config.num_actions,
            output_gain=0.01,
        )
        self.critic = _head(
            channels_in=config.channels_in,
            output_size=1,
            output_gain=1.0,
        )

    def initial_state(
        self,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Return the zeroed recurrent state for a fresh set of workers.

        Args:
          num_envs: Parallel workers the state covers.
          device: Device the state lives on.

        Returns:
          state: Zeros, ``[envs, channels_in]``.

        """
        return torch.zeros(num_envs, self.channels_in, device=device)

    @override
    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Score a batch of observations with no remembered context.

        The feed-forward surface, so a probe or a smoke test works unchanged.
        A real rollout uses :meth:`step`, which is what gives the memory its
        value.

        Args:
          observation: Batched observations, ``[batch, observation_size]``.

        Returns:
          logits: Unnormalized action scores, ``[batch, num_actions]``.
          value: Estimated return from here, ``[batch]``.

        """
        _, logits, value = self.step(
            self.initial_state(observation.shape[0], device=observation.device),
            observation,
            torch.zeros(
                observation.shape[0],
                dtype=torch.bool,
                device=observation.device,
            ),
        )
        return logits, value

    def step(
        self,
        state: Tensor,
        observation: Tensor,
        previous_done: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance the recurrence one step.

        Args:
          state: Recurrent state, ``[envs, channels_in]``.
          observation: Current observations, ``[envs, observation_size]``.
          previous_done: Whether the PRECEDING transition ended an episode.

        Returns:
          state: The updated recurrent state.
          logits: Unnormalized action scores, ``[envs, num_actions]``.
          value: Estimated return from here, ``[envs]``.

        """
        # Cleared here rather than by the caller, so the step path and the
        # sequence path cannot disagree about where an episode begins.
        state = torch.where(previous_done[:, None], 0.0, state)
        state = self.cell(self.embed(observation), state)
        return state, self.actor(state), self.critic(state).squeeze(-1)

    def sequence(
        self,
        state: Tensor,
        observation: Tensor,
        previous_done: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run a whole time window, for the gradient step.

        Identical to calling :meth:`step` down the window -- a recurrence has
        no parallel form, which is the cost of the constant-size state. What
        this buys is one backward pass over the whole window instead of one
        per step.

        Args:
          state: Recurrent state at the window's first step.
          observation: Time-major observations, ``[time, envs, obs]``.
          previous_done: Time-major preceding-transition terminal flags.

        Returns:
          state: The recurrent state after the last step.
          logits: Time-major action scores, ``[time, envs, num_actions]``.
          value: Time-major value estimates, ``[time, envs]``.

        """
        logits: list[Tensor] = []
        values: list[Tensor] = []
        for index in range(observation.shape[0]):
            state, step_logits, step_value = self.step(
                state,
                observation[index],
                previous_done[index],
            )
            logits.append(step_logits)
            values.append(step_value)
        return state, torch.stack(logits), torch.stack(values)


def _head(*, channels_in: int, output_size: int, output_gain: float) -> nn.Sequential:
    """Build one two-layer ReLU head with an orthogonally-scaled output."""
    return nn.Sequential(
        _dense(channels_in, channels_in, gain=2.0**0.5),
        nn.ReLU(),
        _dense(channels_in, channels_in, gain=2.0**0.5),
        nn.ReLU(),
        _dense(channels_in, output_size, gain=output_gain),
    )


def _head_cost(
    *,
    channels_in: int,
    output_size: int,
    rows: int,
    dt: torch.dtype | None,
) -> Cost:
    """Cost one head: two biased ReLU layers, then a biased readout."""
    hidden = matmul_cost(
        channels_in=channels_in,
        channels_out=channels_in,
        bias=True,
        rows=rows,
        dtype=dt,
    ) + elementwise_cost(
        primal=channels_in,
        adjoint=channels_in,
        channels=channels_in,
        dtype=dt,
    )
    return hidden.tile(2, copies=2) + matmul_cost(
        channels_in=channels_in,
        channels_out=output_size,
        bias=True,
        rows=rows,
        dtype=dt,
    )


def _dense(in_features: int, out_features: int, *, gain: float) -> nn.Linear:
    """Build a linear layer with orthogonal weights and no initial bias."""
    layer = nn.Linear(in_features, out_features)
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)
    return layer
