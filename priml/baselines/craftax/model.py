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

import math

from configgle import Fig
from torch import Tensor, nn

import torch


class ActorCritic(nn.Module):
    """Separate policy and value towers over a flat observation."""

    class Config(Fig["ActorCritic"]):
        """Configure the two towers."""

        observation_size: int = 8_268
        """Width of one observation; the environment's own width."""

        num_actions: int = 43
        """Size of the discrete action space."""

        hidden_size: int = 512
        """Width of each hidden layer, in both towers."""

        num_layers: int = 3
        """Hidden layers per tower."""

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
                config.hidden_size,
                config.num_layers,
            )
            <= 0
        ):
            raise ValueError("ActorCritic dimensions must be positive")
        self.policy = _tower(
            observation_size=config.observation_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.num_actions,
            output_gain=0.01,
        )
        self.value = _tower(
            observation_size=config.observation_size,
            hidden_size=config.hidden_size,
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
    hidden_size: int,
    num_layers: int,
    output_size: int,
    output_gain: float,
) -> nn.Sequential:
    """Build one tanh tower with an orthogonally-initialized output."""
    layers: list[nn.Module] = []
    width = observation_size
    for _ in range(num_layers):
        layers.append(_linear(width, hidden_size, gain=math.sqrt(2.0)))
        layers.append(nn.Tanh())
        width = hidden_size
    layers.append(_linear(width, output_size, gain=output_gain))
    return nn.Sequential(*layers)


def _linear(in_features: int, out_features: int, *, gain: float) -> nn.Linear:
    """Build a linear layer with orthogonal weights and no initial bias."""
    layer = nn.Linear(in_features, out_features)
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    # ``bias`` is optional on the module but always present here, since the
    # layer is constructed with the default ``bias=True``.
    assert layer.bias is not None
    torch.nn.init.zeros_(layer.bias)
    return layer
