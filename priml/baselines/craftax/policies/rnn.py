"""Craftax_Baselines' recurrent actor-critic: one GRU state carried step to step.

An embedding of the observation, ``channels_hidden`` wide and activated, feeds
a GRU cell whose state is the policy's whole memory; separate actor and critic
heads, ``num_layers`` activated layers each, read that state, then the logits
and the value, emitted as one fused row as the sampler and the learning rules
read it. Every projection is orthogonal (:func:`torch.nn.init.orthogonal_`),
scaled by sqrt(2) in the embedding and the heads' hidden layers, 0.01 at the
policy's output and 1 at the value's, with zero biases; the cell keeps
torch's own init.

The recurrence is reset-aware: the state is zeroed wherever an episode starts,
before the cell reads it, so a policy beginning a new world does not remember
the one it just left. The carry is ``[1, envs, channels_hidden]``, one layer of
state. A rollout step advances it once (:meth:`ActorCriticRNN.forward_fused`);
the learner replays a window of steps from the carry its rows started with
(:meth:`ActorCriticRNN.forward_sequence`), the embedding and the heads over
every step at once and the cell step by step, since a recurrence has no
parallel form.

References:
    https://github.com/MichaelTMatthews/Craftax_Baselines
        Matthews et al. Craftax_Baselines (MIT license), ``ActorCriticRNN`` and
        ``ScannedRNN`` in ``ppo_rnn.py``.
    https://arxiv.org/abs/1406.1078
        Cho et al. 2014. Learning phrase representations using RNN
        encoder-decoder for statistical machine translation.

"""

from __future__ import annotations

from dataclasses import field, replace
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
from priml.cost import (
    Cost,
    elementwise_cost,
    matmul_cost,
    resolve_dtype,
)
from priml.math.custom_types import TensorFn
from priml.model.linear import Linear
from priml.model.swiglu import relu


class ActorCriticRNN(nn.Module):
    """A GRU actor-critic over a dense observation, its carry reset where episodes start."""

    class Config(Fig["ActorCriticRNN"]):
        """The geometry; every projection's widths derive from it."""

        observation_size: int = 8_268
        """Floats per observation: original Craftax's symbolic view by default."""

        num_actions: int = 43
        """Logits per observation; the fused row adds one for the value."""

        channels_hidden: int = 512
        """Width of the embedding, the GRU's state and every hidden head layer."""

        num_layers: int = 2
        """Hidden layers in each head, before its output."""

        activation: TensorFn = relu
        """Applied after the embedding and after every hidden head layer."""

        proj_in: Linear.Config = field(
            default_factory=lambda: _orthogonal_linear(gain=2**0.5),
        )
        """The embedding, ``observation_size -> channels_hidden``."""

        proj_hidden: Linear.Config = field(
            default_factory=lambda: _orthogonal_linear(gain=2**0.5),
        )
        """Each head's hidden layers, ``channels_hidden -> channels_hidden``."""

        proj_policy: Linear.Config = field(
            default_factory=lambda: _orthogonal_linear(gain=0.01),
        )
        """The actor's output, ``channels_hidden -> num_actions``."""

        proj_value: Linear.Config = field(
            default_factory=lambda: _orthogonal_linear(gain=1.0),
        )
        """The critic's output, ``channels_hidden -> 1``."""

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
            """Cost the learner's window forward: the embedding, the cell per step, the heads.

            The embedding and both heads each run once over the window's
            ``seq_len * batch_size`` rows; the observations take no gradient,
            so back the embedding forms its weight's alone. The cell runs once
            per step on ``batch_size`` rows, re-reading its weights each time:
            two biased ``width -> 3 width`` products, the input gates' and the
            state's, then per unit two sigmoid gates (an add and a sigmoid
            each), the candidate (a product, an add, a tanh) and the
            interpolation (a subtraction, two products, an add), eleven
            operations, seventeen in their adjoint. The first step's state is
            the window's carry, which takes no gradient, so back its product
            forms its weight's alone. The reset is one select per state unit
            each way, and the stacked states and the fused row are copies.
            ``bytes_state`` is the one state each window carries out.

            Args:
              seq_len: Steps per window.
              batch_size: Windows in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            rows, width = seq_len * batch_size, self.channels_hidden
            embed = weight_gradient_only(
                self.proj_in.cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                ),
            ) + activation_cost(self.activation, elements=rows * width, dtype=dtype)
            gates = matmul_cost(
                channels_in=width,
                channels_out=3 * width,
                bias=True,
                rows=batch_size,
                dtype=dtype,
            )
            cell = (
                gates.tile(seq_len)
                + gates.tile(seq_len - 1)
                + weight_gradient_only(gates).tile(1, copies=0)
                + elementwise_cost(
                    primal=rows * 11 * width,
                    adjoint=rows * 17 * width,
                    channels=width,
                    rows=rows,
                    inputs=7,
                    adjoint_inputs=6,
                    adjoint_outputs=7,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=rows * width,
                    adjoint=rows * width,
                    channels=width,
                    rows=rows,
                    inputs=2,
                    dtype=dtype,
                )
                + concat_cost(elements=rows * width, dtype=dtype)
            )
            hidden = (
                self.proj_hidden.cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + activation_cost(self.activation, elements=rows * width, dtype=dtype)
            ).tile(2 * self.num_layers, copies=2 * self.num_layers)
            outputs = sum(
                (
                    head.cost(
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                    for head in (self.proj_policy, self.proj_value)
                ),
                Cost(),
            )
            fused = concat_cost(elements=rows * (self.num_actions + 1), dtype=dtype)
            return replace(
                embed + cell + hidden + outputs + fused,
                bytes_state=resolve_dtype(dtype).itemsize * batch_size * width,
            )

    def __init__(self, config: Config) -> None:
        """Build the embedding, the cell and both heads, in that order.

        Args:
          config: The geometry.

        Raises:
          ValueError: ``num_layers`` is not positive.

        """
        if config.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, not {config.num_layers}")
        super().__init__()
        self.activation = config.activation
        self.proj_in = config.proj_in.make()
        self.cell = nn.GRUCell(config.channels_hidden, config.channels_hidden)
        self.actor = _head(config, output=config.proj_policy)
        self.critic = _head(config, output=config.proj_value)

    @property
    def dtype(self) -> torch.dtype:
        """The weights' dtype, and so the rollout's stored observations'."""
        return self.cell.weight_ih.dtype

    def initial_state(
        self,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Return the zero carry every environment starts from.

        Args:
          num_envs: Environments the carry covers.
          device: Where it lives.

        Returns:
          state: Zeros, ``[1, num_envs, channels_hidden]``, in the weights'
            dtype.

        """
        return torch.zeros(
            1,
            num_envs,
            self.cell.hidden_size,
            device=device,
            dtype=self.dtype,
        )

    @override
    def forward(self, observations: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Advance the cell one step from ``state`` and score the step.

        Args:
          observations: ``[batch, observation_size]``.
          state: The GRU state ``[batch, channels_hidden]``, already reset
            where an episode starts.

        Returns:
          decoded: ``[batch, num_actions + 1]``: the logits, then the value.
          state: The next GRU state.

        """
        state = self.cell(self._embed(observations), state)
        return self._decode(state), state

    def forward_fused(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor | None,
        *,
        carry: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Score one step of every environment, as the rollout does.

        Args:
          observations: ``[envs, observation_size]``.
          state: The carry, ``[1, envs, channels_hidden]``.
          episode_start: Nonzero where the carry resets first, ``[envs]``;
            None when the caller has already zeroed those rows.
          carry: Where the next carry is written; a new tensor if None. The
            rollout passes ``state`` to advance it in place.
          features: Must be None: the policy reads no feature.

        Returns:
          decoded: ``[envs, num_actions + 1]``: the logits, then the value.
          state: The next carry; ``carry`` if given.

        """
        _refuse_feature(features)
        hidden = state[0]
        if episode_start is not None:
            hidden = _reset(hidden, episode_start)
        decoded, hidden = self(observations, hidden)
        if carry is None:
            return decoded, hidden[None]
        carry[0].copy_(hidden)
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
        """Score a window of steps, as the learner does; autograd differentiates it.

        Args:
          observations: ``[batch, time, observation_size]``.
          state: The carry before the first step, ``[1, batch,
            channels_hidden]``.
          episode_start: Nonzero where the carry resets before that step,
            ``[batch, time]``.
          actions: Unused: there is no auxiliary loss.
          features: Must be None: the policy reads no feature.

        Returns:
          decoded: ``[batch, time, num_actions + 1]``: the logits, then the
            value.
          final: The carry after the last step.
          auxiliary_loss: Zero, 0-dim fp32.

        """
        del actions
        _refuse_feature(features)
        embedded = self._embed(observations)
        hidden = state[0]
        states: list[Tensor] = []
        for step in range(embedded.shape[1]):
            hidden = _reset(hidden, episode_start[:, step])
            hidden = self.cell(embedded[:, step], hidden)
            states.append(hidden)
        decoded = self._decode(torch.stack(states, dim=1))
        return decoded, hidden[None], decoded.new_zeros((), dtype=torch.float32)

    def _embed(self, observations: Tensor) -> Tensor:
        """Return the activated embedding of observations in any leading shape."""
        return self.activation(self.proj_in(observations.to(self.dtype)))

    def _decode(self, state: Tensor) -> Tensor:
        """Return the fused row of logits and value for GRU states in any leading shape."""
        logits = _run(self.actor, state, self.activation)
        return torch.cat((logits, _run(self.critic, state, self.activation)), dim=-1)


def _reset(state: Tensor, episode_start: Tensor) -> Tensor:
    """Zero the rows of ``state`` ``[batch, width]`` where ``episode_start`` is nonzero."""
    return torch.where(episode_start[:, None] != 0, 0.0, state)


def _refuse_feature(features: Tensor | None) -> None:
    """Raise when a feature reaches the policy, which has nothing to read it with."""
    if features is not None:
        raise ValueError("ActorCriticRNN reads no feature")


def _orthogonal_linear(*, gain: float) -> Linear.Config:
    """Return a biased projection: orthogonal weights of ``gain``, zero biases."""
    config = Linear.Config()
    config.bias = True
    config.init_weight = partial(nn.init.orthogonal_, gain=gain)
    return config


def _head(config: ActorCriticRNN.Config, *, output: Linear.Config) -> nn.ModuleList:
    """Build ``num_layers`` copies of ``proj_hidden``, then ``output``."""
    layers = [config.proj_hidden.copy_tree() for _ in range(config.num_layers)]
    layers.append(output.copy_tree())
    return nn.ModuleList(layer.make() for layer in layers)


def _run(head: nn.ModuleList, state: Tensor, activation: TensorFn) -> Tensor:
    """Apply a head: ``activation`` after every layer but its output."""
    layers: list[Linear] = []
    for layer in head:
        assert isinstance(layer, Linear)
        layers.append(layer)
    *hidden, output = layers
    for layer in hidden:
        state = activation(layer(state))
    return output(state)
