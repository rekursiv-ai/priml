"""Craftax_Baselines' actor-critic: two tanh MLPs over the symbolic observation.

The actor and the critic are separate towers over the same observation, each
``num_layers`` biased layers of ``channels_hidden`` with ``activation`` after
each, then a biased head: the action logits for the actor, one value for the
critic. Weights are orthogonal (:func:`torch.nn.init.orthogonal_`: a QR of a
Gaussian matrix, the algorithm Flax's ``orthogonal`` runs), scaled by sqrt(2)
in the hidden layers, 0.01 at the policy head and 1 at the value head; biases
start at zero.

The policy carries no state between steps, so its carry is empty,
``[0, envs, 0]``: the actor's resets and snapshots of it touch nothing. It
emits the step's output as one fused row, the logits then the value, as the
sampler and the learning rules read it.

References:
    https://github.com/MichaelTMatthews/Craftax_Baselines
        Matthews et al. Craftax_Baselines (MIT license), ``ActorCritic`` in
        ``models/actor_critic.py``, commit ``7ce36fa``.

"""

from __future__ import annotations

from dataclasses import field
from functools import partial
from typing import Self, override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.baselines.craftax.lib.costs import (
    activation_cost,
    concat_cost,
    weight_gradient_only,
)
from priml.cost import Cost
from priml.math.custom_types import TensorFn
from priml.model.linear import Linear


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs over a dense observation, with no carry."""

    class Config(Fig["ActorCritic"]):
        """The towers' geometry; every projection's widths derive from it."""

        observation_size: int = 8_268
        """Floats per observation: original Craftax's symbolic view by default."""

        num_actions: int = 43
        """Logits per observation; the fused row adds one for the value."""

        channels_hidden: int = 512
        """Width of every hidden layer, in both towers."""

        num_layers: int = 3
        """Hidden layers per tower."""

        activation: TensorFn = torch.tanh
        """Applied after every hidden layer."""

        proj_in: Linear.Config = field(
            default_factory=lambda: _orthogonal_linear(gain=2**0.5),
        )
        """Each tower's first layer, ``observation_size -> channels_hidden``."""

        proj_hidden: Linear.Config = field(
            default_factory=lambda: _orthogonal_linear(gain=2**0.5),
        )
        """Each tower's later hidden layers, ``channels_hidden -> channels_hidden``."""

        proj_policy: Linear.Config = field(
            default_factory=lambda: _orthogonal_linear(gain=0.01),
        )
        """The actor's head, ``channels_hidden -> num_actions``."""

        proj_value: Linear.Config = field(
            default_factory=lambda: _orthogonal_linear(gain=1.0),
        )
        """The critic's head, ``channels_hidden -> 1``."""

        dtype: torch.dtype = torch.float32
        """Parameter and activation dtype, and so the rollout's storage dtype. The
        weights are drawn in fp32 and cast once: drawn in bf16, the init would
        keep 8 bits per draw."""

        @override
        def finalize(self) -> Self:
            self.proj_in.channels_in = self.observation_size
            self.proj_in.channels_out = self.channels_hidden
            self.proj_hidden.channels_in = self.channels_hidden
            self.proj_hidden.channels_out = self.channels_hidden
            self.proj_policy.channels_in = self.channels_hidden
            self.proj_policy.channels_out = self.num_actions
            self.proj_value.channels_in = self.channels_hidden
            self.proj_value.channels_out = 1
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost both towers and the fused row's concatenation.

            Each tower is ``num_layers`` projections, each followed by the
            activation, then its head; every stage runs at the policy's
            ``dtype``, as the forward casts the observations to it. The
            observations take no gradient, so back each tower's first
            projection forms its weight's alone.

            Args:
              seq_len: Steps per window.
              batch_size: Windows in this invocation.
              dtype: Unread; the policy's own ``dtype`` prices every stage.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            del dtype
            dt, rows = self.dtype, seq_len * batch_size
            hidden = self.proj_hidden.cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dt,
                **kwargs,
            ).tile(self.num_layers - 1, copies=self.num_layers - 1)
            activations = activation_cost(
                self.activation,
                elements=rows * self.channels_hidden,
                dtype=dt,
            ).tile(self.num_layers)
            first = self.proj_in.cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dt,
                **kwargs,
            )
            towers = (weight_gradient_only(first) + hidden + activations).tile(
                2,
                copies=2,
            )
            heads = sum(
                (
                    head.cost(
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dt,
                        **kwargs,
                    )
                    for head in (self.proj_policy, self.proj_value)
                ),
                Cost(),
            )
            return (
                towers
                + heads
                + concat_cost(elements=rows * (self.num_actions + 1), dtype=dt)
            )

    def __init__(self, config: Config) -> None:
        """Build both towers.

        Args:
          config: The geometry.

        Raises:
          ValueError: ``num_layers`` is not positive.

        """
        if config.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, not {config.num_layers}")
        super().__init__()
        self.dtype = config.dtype
        self.activation = config.activation
        self.actor = _tower(config, head=config.proj_policy)
        self.critic = _tower(config, head=config.proj_value)
        self.to(config.dtype)

    def initial_state(
        self,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Return the empty carry: nothing is threaded between steps.

        Args:
          num_envs: Environments the carry covers.
          device: Where it lives.

        Returns:
          state: ``[0, num_envs, 0]`` in the model dtype.

        """
        return torch.zeros(0, num_envs, 0, device=device, dtype=self.dtype)

    def forward_fused(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor | None,
        *,
        carry: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Score one step of every environment.

        Args:
          observations: ``[batch, observation_size]``.
          state: The empty carry.
          episode_start: Unused: there is no carry to reset.
          carry: Returned in place of ``state`` when given, as the rollout
            passes its own.
          features: Must be None: the towers read no feature.

        Returns:
          decoded: ``[batch, num_actions + 1]``: the logits, then the value.
          state: The carry, unchanged.

        """
        del episode_start
        _refuse_feature(features)
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
        """Score a window of steps; each is scored alone.

        Args:
          observations: ``[batch, time, observation_size]``.
          state: The empty carry.
          episode_start: Unused: there is no carry to reset.
          actions: Unused: there is no auxiliary loss.
          features: Must be None: the towers read no feature.

        Returns:
          decoded: ``[batch, time, num_actions + 1]``: the logits, then the
            value.
          final: The carry, unchanged.
          auxiliary_loss: Zero, 0-dim fp32.

        """
        del episode_start, actions
        _refuse_feature(features)
        decoded = self(observations)
        return decoded, state, decoded.new_zeros((), dtype=torch.float32)

    @override
    def forward(self, observations: Tensor) -> Tensor:
        """Return the fused row of logits and value for each observation."""
        features = observations.to(self.dtype)
        return torch.cat(
            (self._run(self.actor, features), self._run(self.critic, features)),
            dim=-1,
        )

    def _run(self, tower: nn.ModuleList, features: Tensor) -> Tensor:
        """Apply a tower: the activation after every layer but the head."""
        *hidden, head = _linears(tower)
        for layer in hidden:
            features = self.activation(layer(features))
        return head(features)


def _refuse_feature(features: Tensor | None) -> None:
    """Raise when a feature reaches the towers, which have nothing to read it with."""
    if features is not None:
        raise ValueError("ActorCritic reads no feature")


def _orthogonal_linear(*, gain: float) -> Linear.Config:
    """Return a biased projection: orthogonal weights of ``gain``, zero biases."""
    config = Linear.Config()
    config.bias = True
    config.init_weight = partial(nn.init.orthogonal_, gain=gain)
    return config


def _linears(tower: nn.ModuleList) -> list[Linear]:
    """Return a tower's layers at their class."""
    layers: list[Linear] = []
    for layer in tower:
        assert isinstance(layer, Linear)
        layers.append(layer)
    return layers


def _tower(config: ActorCritic.Config, *, head: Linear.Config) -> nn.ModuleList:
    """Build ``proj_in``, ``num_layers - 1`` copies of ``proj_hidden``, ``head``."""
    layers = [config.proj_in.copy_tree()]
    layers += [config.proj_hidden.copy_tree() for _ in range(config.num_layers - 1)]
    layers.append(head.copy_tree())
    return nn.ModuleList(layer.make() for layer in layers)
