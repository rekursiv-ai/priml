"""Gated Transformer-XL actor-critic: a policy that remembers.

Craftax rewards long plans -- mine coal, then iron, then find a furnace -- and
a feed-forward policy sees only the current tile. This network carries a
window of its own past layer inputs as attention memory, so the policy that
decides now can condition on what it did a hundred steps ago.

Two mechanisms make that memory trainable. Attention is RELATIVE: a key is
scored by how far back it is rather than where it sits in an absolute
sequence, so a memory window that slides every step keeps meaning the same
thing. And each residual connection is GATED by a GRU-style update gate
initialized closed, so an untrained layer passes its input through unchanged
and the transformer starts out behaving like the identity -- which is what
stops the early, high-variance policy gradients from destroying it.

Memory is caller-owned. The module never stores it, because a rollout, a
gradient window, and an evaluation all thread different memory through the
same weights, and hiding it in the module would make those three paths
disagree silently.

References:
    https://arxiv.org/abs/1910.06764
        Parisotto et al. 2020. Stabilizing transformers for reinforcement
        learning.
    https://arxiv.org/abs/1901.02860
        Dai et al. 2019. Transformer-XL: attentive language models beyond a
        fixed-length context.
    https://github.com/Reytuag/transformerXL_PPO_JAX
        The Craftax scoreboard implementation this ports.

"""

from __future__ import annotations

from typing import override

import math

from configgle import Fig
from torch import Tensor, nn

import torch


class ActorCriticGTrXL(nn.Module):
    """A recurrent actor-critic over a window of remembered layer inputs.

    Attributes:
      memory_length: Steps of layer-input memory the module attends over.
      num_layers: Transformer layers, and therefore memory rows per step.
      embed_dim: Width of the embedding each layer reads and writes.

    """

    class Config(Fig["ActorCriticGTrXL"]):
        """Configure the transformer and its heads."""

        observation_size: int = 8_268
        """Width of one observation; the environment's own width."""

        num_actions: int = 43
        """Size of the discrete action space."""

        embed_dim: int = 256
        """Width of the embedding carried between layers."""

        num_heads: int = 8
        """Attention heads per layer; must divide ``qkv_dim``."""

        num_layers: int = 2
        """Transformer layers."""

        qkv_dim: int = 256
        """Combined width of the query, key, and value projections."""

        channels_in: int = 256
        """Width of each hidden layer in the actor and critic heads."""

        memory_length: int = 128
        """Steps of layer-input memory attended over."""

        gating_bias: float = 2.0
        """Initial bias subtracted inside every update gate.

        Positive values close the gate at initialization, so an untrained
        layer is the identity and the memory survives the first updates."""

    def __init__(self, config: Config) -> None:
        """Build the transformer stack and both heads.

        Args:
          config: Geometry of the network.

        Raises:
          ValueError: A dimension is not positive, or the heads do not divide
            the query-key-value width.

        """
        super().__init__()
        if (
            min(
                config.observation_size,
                config.num_actions,
                config.embed_dim,
                config.num_heads,
                config.num_layers,
                config.qkv_dim,
                config.channels_in,
                config.memory_length,
            )
            <= 0
        ):
            raise ValueError("GTrXL dimensions must be positive")
        if config.qkv_dim % config.num_heads:
            raise ValueError("num_heads must divide qkv_dim")

        self.memory_length = config.memory_length
        self.num_layers = config.num_layers
        self.embed_dim = config.embed_dim

        self.encoder = _dense(config.observation_size, config.embed_dim)
        self.layers = nn.ModuleList(
            _GTrXLLayer(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                qkv_dim=config.qkv_dim,
                gating_bias=config.gating_bias,
            )
            for _ in range(config.num_layers)
        )
        self.actor = _head(
            embed_dim=config.embed_dim,
            channels_in=config.channels_in,
            output_size=config.num_actions,
            output_gain=0.01,
        )
        self.critic = _head(
            embed_dim=config.embed_dim,
            channels_in=config.channels_in,
            output_size=1,
            output_gain=1.0,
        )

    def initial_state(
        self,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Return empty memory for a fresh set of workers.

        Args:
          num_envs: Parallel workers the memory covers.
          device: Device the memory lives on.

        Returns:
          memory: Zeroed layer-input cache,
            ``[envs, memory_length, layers, embed]``.
          valid_length: How many rows of that cache are real, ``[envs]``.

        """
        return (
            torch.zeros(
                num_envs,
                self.memory_length,
                self.num_layers,
                self.embed_dim,
                device=device,
            ),
            torch.zeros(num_envs, dtype=torch.int64, device=device),
        )

    @override
    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Score a batch of observations with no remembered context.

        The feed-forward surface, so anything that only wants "what would this
        policy do here" -- a probe, a smoke test -- works unchanged. A real
        rollout uses :meth:`step`, which is what gives the memory its value.

        Args:
          observation: Batched observations, ``[batch, observation_size]``.

        Returns:
          logits: Unnormalized action scores, ``[batch, num_actions]``.
          value: Estimated return from here, ``[batch]``.

        """
        memory, valid_length = self.initial_state(
            observation.shape[0],
            device=observation.device,
        )
        _, _, logits, value = self.step(
            memory,
            valid_length,
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
        memory: Tensor,
        valid_length: Tensor,
        observation: Tensor,
        previous_done: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Take one recurrent step, appending this step's layer inputs.

        A worker whose previous transition ended its episode has its memory
        cleared here rather than by the caller, so the sequence path and the
        step path cannot disagree about where an episode begins.

        Args:
          memory: Layer-input cache, ``[envs, memory_length, layers, embed]``.
          valid_length: Real rows in that cache, ``[envs]``.
          observation: Current observations, ``[envs, observation_size]``.
          previous_done: Whether the PRECEDING transition ended an episode.

        Returns:
          memory: The cache with this step's layer inputs appended.
          valid_length: Updated count of real rows.
          logits: Unnormalized action scores, ``[envs, num_actions]``.
          value: Estimated return from here, ``[envs]``.

        """
        memory = torch.where(previous_done[:, None, None, None], 0.0, memory)
        valid_length = torch.where(previous_done, 0, valid_length)

        positions = torch.arange(self.memory_length + 1, device=memory.device)
        # A row is attendable once it is inside the filled tail of the window;
        # the window fills from the right, so the frontier moves left.
        mask = positions[None, :] >= (self.memory_length - valid_length)[:, None]
        mask = mask[:, None, None, :]
        positional = _sinusoidal_positions(
            self.memory_length + 1,
            embed_dim=self.embed_dim,
            device=memory.device,
            dtype=memory.dtype,
        )

        hidden = self.encoder(observation)
        layer_inputs: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            layer_inputs.append(hidden)
            keys = torch.cat((memory[:, :, index], hidden[:, None]), dim=1)
            hidden = layer(keys, hidden[:, None], positional, mask)[:, 0]

        appended = torch.stack(layer_inputs, dim=1)
        memory = torch.cat((memory[:, 1:], appended[:, None]), dim=1)
        valid_length = (valid_length + 1).clamp_max(self.memory_length)
        logits, value = self._heads(hidden)
        return memory, valid_length, logits, value

    def sequence(
        self,
        memory: Tensor,
        valid_length: Tensor,
        observation: Tensor,
        previous_done: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Rescore a whole time window in one pass, for the gradient step.

        Equivalent to calling :meth:`step` down the window, but computed as
        one masked attention so the whole window backpropagates at once. The
        mask does the work the recurrence did: a query attends only to earlier
        steps, and only to steps in its own episode.

        Args:
          memory: Cache at the window's first step,
            ``[envs, memory_length, layers, embed]``.
          valid_length: Real rows in that cache, ``[envs]``.
          observation: Time-major observations, ``[time, envs, obs]``.
          previous_done: Time-major preceding-transition terminal flags.

        Returns:
          logits: Time-major action scores, ``[time, envs, num_actions]``.
          value: Time-major value estimates, ``[time, envs]``.

        """
        observations = observation.transpose(0, 1)
        dones = previous_done.transpose(0, 1)
        num_steps = observations.shape[1]

        # Each terminal flag starts a new segment, so two steps belong to the
        # same episode exactly when their running counts agree.
        segments = dones.to(torch.int64).cumsum(dim=1)
        steps = torch.arange(num_steps, device=observations.device)
        causal = (steps[None, :, None] >= steps[None, None, :]) & (
            segments[:, :, None] == segments[:, None, :]
        )
        rows = torch.arange(self.memory_length, device=observations.device)
        filled = rows[None, :] >= (self.memory_length - valid_length)[:, None]
        # Memory predates the window, so only steps still in the window's
        # first episode may look at it.
        remembered = (segments == 0)[:, :, None] & filled[:, None, :]
        mask = torch.cat((remembered, causal), dim=-1)[:, None]

        positional = _sinusoidal_positions(
            self.memory_length + num_steps,
            embed_dim=self.embed_dim,
            device=observations.device,
            dtype=observations.dtype,
        )
        hidden = self.encoder(observations)
        for index, layer in enumerate(self.layers):
            keys = torch.cat((memory[:, :, index], hidden), dim=1)
            hidden = layer(keys, hidden, positional, mask)
        logits, value = self._heads(hidden)
        return logits.transpose(0, 1), value.transpose(0, 1)

    def _heads(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the separate actor and critic towers."""
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class _GTrXLLayer(nn.Module):
    """One pre-norm layer: gated relative attention, then a gated MLP."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        qkv_dim: int,
        gating_bias: float,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = _RelativeAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_dim=qkv_dim,
        )
        self.gate_attention = _Gate(embed_dim=embed_dim, bias=gating_bias)
        self.norm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_in = _dense(embed_dim, embed_dim)
        self.mlp_out = _dense(embed_dim, embed_dim)
        self.gate_mlp = _Gate(embed_dim=embed_dim, bias=gating_bias)

    @override
    def forward(
        self,
        keys: Tensor,
        queries: Tensor,
        positional: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Attend, gate, transform, gate again.

        Args:
          keys: Memory rows followed by this window's own inputs.
          queries: The steps being scored.
          positional: Relative-position encodings, one per key slot.
          mask: Which key each query may attend to.

        Returns:
          hidden: The layer's output, same shape as ``queries``.

        """
        # One normalization instance for both sides: keys and queries are the
        # same representation at different times, so normalizing them with
        # different statistics would make a remembered step incomparable to
        # the step querying it.
        attended = self.attention(
            self.norm_attention(queries),
            self.norm_attention(keys),
            positional,
            mask,
        )
        attended = self.gate_attention(queries, attended.relu())
        hidden = self.mlp_out(nn.functional.gelu(self.mlp_in(self.norm_mlp(attended))))
        return self.gate_mlp(attended, hidden.relu())


class _Gate(nn.Module):
    """A GRU-style gated residual, initialized closed."""

    def __init__(self, *, embed_dim: int, bias: float) -> None:
        super().__init__()
        self.gating_bias = nn.Parameter(torch.full((embed_dim,), bias))
        self.reset_y = _dense(embed_dim, embed_dim, bias=False)
        self.reset_x = _dense(embed_dim, embed_dim, bias=False)
        self.update_y = _dense(embed_dim, embed_dim, bias=False)
        self.update_x = _dense(embed_dim, embed_dim, bias=False)
        self.candidate_y = _dense(embed_dim, embed_dim, bias=False)
        self.candidate_x = _dense(embed_dim, embed_dim, bias=False)

    @override
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Interpolate the residual ``x`` toward a candidate built from ``y``.

        Args:
          x: The residual input, passed through when the gate is closed.
          y: The sublayer's proposal.

        Returns:
          output: The gated combination.

        """
        reset = torch.sigmoid(self.reset_y(y) + self.reset_x(x))
        update = torch.sigmoid(
            self.update_y(y) + self.update_x(x) - self.gating_bias,
        )
        candidate = torch.tanh(self.candidate_y(y) + self.candidate_x(reset * x))
        return (1.0 - update) * x + update * candidate


class _RelativeAttention(nn.Module):
    """Multi-head attention scored by relative rather than absolute position."""

    def __init__(self, *, embed_dim: int, num_heads: int, qkv_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = qkv_dim // num_heads
        self.qkv_dim = qkv_dim
        self.query = _dense(embed_dim, qkv_dim)
        self.key = _dense(embed_dim, qkv_dim)
        self.value = _dense(embed_dim, qkv_dim)
        self.relative_position = _dense(embed_dim, qkv_dim, bias=False)
        self.bias_content = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.bias_position = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.out = _dense(qkv_dim, embed_dim)

    @override
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        positional: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Score every query against every key it is allowed to see.

        Args:
          queries: Normalized query steps, ``[envs, queries, embed]``.
          keys: Normalized key steps, ``[envs, keys, embed]``.
          positional: Relative encodings, ``[keys, embed]``.
          mask: Broadcastable boolean mask over ``[envs, heads, queries,
            keys]``.

        Returns:
          attended: The attention output, ``[envs, queries, embed]``.

        """
        num_queries = queries.shape[1]
        query = self._split(self.query(queries))
        key = self._split(self.key(keys))
        value = self._split(self.value(keys))
        relative = self.relative_position(positional).reshape(
            positional.shape[0],
            self.num_heads,
            self.head_dim,
        )

        content = (query + self.bias_content).transpose(1, 2)
        weights = content @ key.transpose(1, 2).transpose(-2, -1)
        position = (query + self.bias_position).transpose(1, 2)
        relative_weights = position @ relative.transpose(0, 1).transpose(-2, -1)

        # The relative encodings are shared across queries, so each query row
        # reads them at its own offset: the last query sees lag zero at the
        # last slot, the one before it one slot earlier, and so on.
        shifts = torch.arange(num_queries, device=queries.device) - (num_queries - 1)
        source = (
            torch.arange(relative_weights.shape[-1], device=queries.device)[None, :]
            - shifts[:, None]
        ) % relative_weights.shape[-1]
        relative_weights = relative_weights.gather(
            -1,
            source[None, None].expand(relative_weights.shape),
        )

        weights = (weights + relative_weights) / math.sqrt(self.head_dim)
        weights = torch.where(mask, weights, -1e30).softmax(-1)
        attended = weights @ value.transpose(1, 2)
        attended = attended.transpose(1, 2).reshape(
            queries.shape[0],
            num_queries,
            self.qkv_dim,
        )
        return self.out(attended)

    def _split(self, projected: Tensor) -> Tensor:
        """Reshape a flat projection into per-head vectors."""
        return projected.reshape(
            *projected.shape[:-1],
            self.num_heads,
            self.head_dim,
        )


def _head(
    *,
    embed_dim: int,
    channels_in: int,
    output_size: int,
    output_gain: float,
) -> nn.Sequential:
    """Build one two-layer ReLU head with an orthogonal output."""
    return nn.Sequential(
        _dense(embed_dim, channels_in, gain=math.sqrt(2.0)),
        nn.ReLU(),
        _dense(channels_in, channels_in, gain=math.sqrt(2.0)),
        nn.ReLU(),
        _dense(channels_in, output_size, gain=output_gain),
    )


def _dense(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    gain: float | None = None,
    truncation: float = 0.87962566103423978,
) -> nn.Linear:
    """Build a linear layer initialized the way the reference does.

    Without an explicit ``gain`` the weights are drawn from a two-sigma
    truncated normal scaled by the fan-in, which is the reference framework's
    default and differs from this one's uniform default by enough to change
    where training starts.

    Args:
      in_features: Input width.
      out_features: Output width.
      bias: Whether the layer has a bias, always initialized to zero.
      gain: Orthogonal-initialization gain; omit for the fan-in default.
      truncation: Standard deviation of a standard normal truncated to two
        sigma. Dividing by it makes the truncated sample's spread equal the
        nominal one, as the reference initializer does; omitting it would
        shrink every weight in the network by twelve percent.

    Returns:
      layer: The initialized layer.

    """
    layer = nn.Linear(in_features, out_features, bias=bias)
    if gain is None:
        deviation = math.sqrt(1.0 / in_features) / truncation
        torch.nn.init.trunc_normal_(
            layer.weight,
            std=deviation,
            a=-2.0 * deviation,
            b=2.0 * deviation,
        )
    else:
        torch.nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)
    return layer


def _sinusoidal_positions(
    length: int,
    *,
    embed_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Encode each key slot by how far in the past it is.

    Slots are numbered from ``length`` down to one, so the encoding of "one
    step ago" is the same vector whatever the window's absolute position --
    which is the property that lets a sliding memory be attended over at all.

    Args:
      length: Key slots to encode.
      embed_dim: Width of the encoding.
      device: Device to build on.
      dtype: Floating type of the result.

    Returns:
      positions: Encodings, ``[length, embed_dim]``.

    """
    frequency = 1.0 / (
        10_000
        ** (
            torch.arange(0.0, embed_dim, 2.0, device=device, dtype=torch.float32)
            / embed_dim
        )
    )
    distance = torch.arange(length, 0, -1, device=device, dtype=torch.float32)
    angles = torch.outer(distance, frequency)
    return torch.cat((angles.sin(), angles.cos()), dim=-1).to(dtype)
