"""The policy and value network.

Two separate towers, one choosing actions and one estimating returns, over the
flat symbolic observation. Sharing a trunk would be cheaper, but the two heads
want different features -- the critic must value a state the policy is already
confident about -- and the published baseline this reproduces keeps them apart.

Initialization follows the same recipe: orthogonal weights scaled by the gain
that keeps activations at unit variance through tanh, except at the heads. The
policy head is scaled down by a hundred so the initial policy is nearly
uniform, which is what stops the first few updates from committing to an
arbitrary action before any reward has been seen.
"""

from __future__ import annotations

from typing import override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    elementwise_cost,
    matmul_cost,
)


class ActorCritic(nn.Module):
    """Separate policy and value towers over a flat observation."""

    class Config(Fig["ActorCritic"]):
        """Configure the two towers."""

        observation_size: int = 8_268
        """Width of one observation; the environment's own width."""

        num_actions: int = 43
        """Size of the discrete action space."""

        channels_in: int = 512
        """Width of each hidden layer, in both towers."""

        num_layers: int = 3
        """Hidden layers per tower."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price one observation through both towers.

            A token is one environment step of one worker: the observation
            vector in, action logits and a value out. Nothing is carried
            between steps, so ``bytes_state`` is zero. Each hidden layer is a
            biased matmul and a tanh; the tanh's adjoint is ``g * (1 - t**2)``
            on the saved output, three operations per unit.

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
            return sum(
                (
                    _tower_cost(
                        observation_size=self.observation_size,
                        channels_in=self.channels_in,
                        num_layers=self.num_layers,
                        output_size=output_size,
                        rows=rows,
                        dt=dt,
                    )
                    for output_size in (self.num_actions, 1)
                ),
                Cost(),
            )

    def __init__(self, config: Config) -> None:
        """Build both towers.

        Args:
          config: Geometry of the network.

        Raises:
          ValueError: A dimension is not positive.

        """
        super().__init__()
        if (
            min(
                config.observation_size,
                config.num_actions,
                config.channels_in,
                config.num_layers,
            )
            <= 0
        ):
            raise ValueError("ActorCritic dimensions must be positive")
        self.policy = _tower(
            observation_size=config.observation_size,
            channels_in=config.channels_in,
            num_layers=config.num_layers,
            output_size=config.num_actions,
            output_gain=0.01,
        )
        self.value = _tower(
            observation_size=config.observation_size,
            channels_in=config.channels_in,
            num_layers=config.num_layers,
            output_size=1,
            output_gain=1.0,
        )

    @override
    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Score every action and estimate the state's value.

        Args:
          observation: Batched observations, ``[batch, observation_size]``.

        Returns:
          logits: Unnormalized action scores, ``[batch, num_actions]``.
          value: Estimated return from here, ``[batch]``.

        """
        return self.policy(observation), self.value(observation).squeeze(-1)


def _tower(
    *,
    observation_size: int,
    channels_in: int,
    num_layers: int,
    output_size: int,
    output_gain: float,
) -> nn.Sequential:
    """Build one tanh tower with an orthogonally-initialized output."""
    layers: list[nn.Module] = []
    width = observation_size
    for _ in range(num_layers):
        layers.append(_linear(width, channels_in, gain=2.0**0.5))
        layers.append(nn.Tanh())
        width = channels_in
    layers.append(_linear(width, output_size, gain=output_gain))
    return nn.Sequential(*layers)


def _tower_cost(
    *,
    observation_size: int,
    channels_in: int,
    num_layers: int,
    output_size: int,
    rows: int,
    dt: torch.dtype | None,
) -> Cost:
    """Price one tower: ``num_layers`` biased tanh layers, then a biased readout."""
    total = Cost()
    width = observation_size
    for _ in range(num_layers):
        total += matmul_cost(
            channels_in=width,
            channels_out=channels_in,
            bias=True,
            rows=rows,
            dtype=dt,
        )
        total += elementwise_cost(
            primal=channels_in,
            adjoint=3 * channels_in,
            channels=channels_in,
            dtype=dt,
        )
        width = channels_in
    return total + matmul_cost(
        channels_in=width,
        channels_out=output_size,
        bias=True,
        rows=rows,
        dtype=dt,
    )


def _linear(in_features: int, out_features: int, *, gain: float) -> nn.Linear:
    """Build a linear layer with orthogonal weights and no initial bias."""
    layer = nn.Linear(in_features, out_features)
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    # ``bias`` is optional on the module but always present here, since the
    # layer is constructed with the default ``bias=True``.
    if layer.bias is None:
        raise ValueError("Expected layer.bias is not None.")
    torch.nn.init.zeros_(layer.bias)
    return layer
