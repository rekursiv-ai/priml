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

from dataclasses import replace
from typing import override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    resolve_dtype,
    traffic,
)
from priml.model.norm import LayerNorm


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

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one environment step of one worker.

            A token is one step: the observation in, logits and a value out,
            attending over the whole memory. ``seq_len`` is the steps scored
            together -- one for :meth:`step`, the gradient window for
            :meth:`sequence` -- and every query in a window shares its keys:
            the ``memory_length + seq_len`` key rows are normalized and
            projected once per window, so a token pays ``keys / seq_len`` of
            them. The relative-position table is one constant shared by the
            whole batch, so its projection is spread over ``rows`` and,
            having no input gradient, its adjoint is the weight gradient
            alone. Attention is counted over every key with no mask discount.

            ``bytes_state`` is what one step adds to the memory: one layer
            input per layer. Maintaining that memory -- the reset, the
            append -- is cache bookkeeping, uncosted like a KV-cache update.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            rows = seq_len * batch_size
            keys = self.memory_length + seq_len
            layer = _layer_cost(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                qkv_dim=self.qkv_dim,
                keys=keys,
                key_rows=rows * keys // seq_len,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            encoder = matmul_cost(
                channels_in=self.observation_size,
                channels_out=self.embed_dim,
                bias=True,
                rows=rows,
                dtype=dtype,
            )
            heads = sum(
                (
                    _head_cost(
                        embed_dim=self.embed_dim,
                        channels_in=self.channels_in,
                        output_size=output_size,
                        rows=rows,
                        dtype=dtype,
                    )
                    for output_size in (self.num_actions, 1)
                ),
                Cost(),
            )
            return replace(
                encoder + layer.tile(self.num_layers, copies=self.num_layers) + heads,
                bytes_state=resolve_dtype(dtype).itemsize
                * self.num_layers
                * self.embed_dim,
            )

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

        weights = (weights + relative_weights) / self.head_dim**0.5
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
        _dense(embed_dim, channels_in, gain=2.0**0.5),
        nn.ReLU(),
        _dense(channels_in, channels_in, gain=2.0**0.5),
        nn.ReLU(),
        _dense(channels_in, output_size, gain=output_gain),
    )


def _head_cost(
    *,
    embed_dim: int,
    channels_in: int,
    output_size: int,
    rows: float,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost one head: two biased ReLU layers, then a biased readout."""
    dt = dtype
    relu = elementwise_cost(
        primal=channels_in,
        adjoint=channels_in,
        channels=channels_in,
        dtype=dt,
    )
    return (
        matmul_cost(
            channels_in=embed_dim,
            channels_out=channels_in,
            bias=True,
            rows=rows,
            dtype=dt,
        )
        + relu
        + matmul_cost(
            channels_in=channels_in,
            channels_out=channels_in,
            bias=True,
            rows=rows,
            dtype=dt,
        )
        + relu
        + matmul_cost(
            channels_in=channels_in,
            channels_out=output_size,
            bias=True,
            rows=rows,
            dtype=dt,
        )
    )


# ``key_rows`` is how many key rows the whole batch normalizes and projects; a token
# pays ``key_rows / rows`` of them, plus its own query row. The shared
# normalization's parameters see both.
def _layer_cost(
    *,
    embed_dim: int,
    num_heads: int,
    qkv_dim: int,
    keys: int,
    key_rows: int,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype | None,
    **kwargs: object,
) -> Cost:
    """Cost one layer for one token that attends over ``keys`` rows."""
    rows = seq_len * batch_size
    dt = dtype
    per_token_key_rows = key_rows / rows
    norm = LayerNorm.Config(embed_dim, elementwise_affine=True).finalize()
    norm_attention = cost(
        norm,
        seq_len=seq_len + key_rows // batch_size,
        batch_size=batch_size,
        dtype=dtype,
        **kwargs,
    ).tile(1 + per_token_key_rows)
    projection = matmul_cost(
        channels_in=embed_dim,
        channels_out=qkv_dim,
        bias=True,
        rows=rows,
        dtype=dt,
    )
    key_value = (
        matmul_cost(
            channels_in=embed_dim,
            channels_out=qkv_dim,
            bias=True,
            rows=key_rows,
            dtype=dt,
        )
        .tile(per_token_key_rows)
        .tile(2, copies=2)
    )
    # The content and position biases are each added to the query; the
    # adjoint sums the two score paths' gradients back into it.
    query_biases = elementwise_cost(
        primal=2 * qkv_dim,
        adjoint=qkv_dim,
        channels=qkv_dim,
        inputs=2,
        outputs=2,
        adjoint_inputs=2,
        params=2 * qkv_dim,
        rows=rows,
        dtype=dt,
    )
    attention = (
        norm_attention
        + projection
        + query_biases
        + key_value
        + _relative_table_cost(
            embed_dim=embed_dim,
            qkv_dim=qkv_dim,
            keys=keys,
            rows=rows,
            dtype=dt,
        )
        + _relative_scores_cost(
            num_heads=num_heads,
            qkv_dim=qkv_dim,
            keys=keys,
            seq_len=seq_len,
            dtype=dt,
        )
        + matmul_cost(
            channels_in=qkv_dim,
            channels_out=embed_dim,
            bias=True,
            rows=rows,
            dtype=dt,
        )
        + elementwise_cost(
            primal=embed_dim,
            adjoint=embed_dim,
            channels=embed_dim,
            dtype=dt,
        )
    )
    # GELU: scale, erf, add, halve, multiply; the adjoint adds the density.
    mlp = (
        cost(norm, seq_len=seq_len, batch_size=batch_size, dtype=dtype, **kwargs)
        + matmul_cost(
            channels_in=embed_dim,
            channels_out=embed_dim,
            bias=True,
            rows=rows,
            dtype=dt,
        ).tile(2, copies=2)
        + elementwise_cost(
            primal=5 * embed_dim,
            adjoint=6 * embed_dim,
            channels=embed_dim,
            dtype=dt,
        )
        + elementwise_cost(
            primal=embed_dim,
            adjoint=embed_dim,
            channels=embed_dim,
            dtype=dt,
        )
    )
    gate = _gate_cost(embed_dim=embed_dim, rows=rows, dtype=dt)
    return attention + gate + mlp + gate


# Content and position scores are the same product on two biased copies of the query;
# the position scores are then gathered into place, one element moved per score and one
# scatter-add back. The combined score is added, scaled, masked, then softmaxed: subtract
# the max, exponentiate, divide, with the max and the sum as the reductions and the
# adjoint's ``sum(g * p)`` as one more.
# The position gather's index is one ``int64`` per score each way.
def _relative_scores_cost(
    *,
    num_heads: int,
    qkv_dim: int,
    keys: int,
    seq_len: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost unfused relative attention, sharing K/V within each query sequence."""
    dt = dtype
    channels_head = qkv_dim // num_heads
    scores = matmul_cost(
        channels_in=channels_head,
        channels_out=keys,
        weight=False,
        rows=seq_len,
        dtype=dt,
    )
    values = matmul_cost(
        channels_in=keys,
        channels_out=channels_head,
        weight=False,
        rows=seq_len,
        dtype=dt,
    )
    gather = (
        traffic("primal", "selection", elements=keys, dtype=dt)
        + traffic("primal", "selection", elements=keys, dtype=torch.int64)
        + traffic("adjoint", "selection", elements=2 * keys, flops=keys, dtype=dt)
        + traffic("adjoint", "selection", elements=keys, dtype=torch.int64)
    )
    # Add, scale, mask, subtract, exp, divide; backward includes the two score paths.
    softmax = elementwise_cost(
        primal=6 * keys,
        adjoint=6 * keys,
        channels=keys,
        inputs=9,
        outputs=6,
        adjoint_inputs=10,
        adjoint_outputs=6,
        dtype=dt,
    ) + (
        reduction_cost(input_elements=keys, dtype=dt).tile(2, copies=2)
        + reduction_cost(input_elements=keys, dtype=dt, phase="adjoint")
    )
    return (scores.tile(2, copies=2) + gather + softmax + values).tile(
        num_heads,
        copies=num_heads,
    )


def _relative_table_cost(
    *,
    embed_dim: int,
    qkv_dim: int,
    keys: int,
    rows: float,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost one shared constant table's projection and weight gradient."""
    dt = dtype
    products = 2 * embed_dim * qkv_dim * (keys / rows)
    params = embed_dim * qkv_dim
    moved = (keys * (embed_dim + qkv_dim) + params) / rows
    return (
        traffic("primal", "matmul", elements=moved, flops=products, dtype=dt)
        + traffic("adjoint", "matmul", elements=moved, flops=products, dtype=dt)
        + Cost(params=params, params_active=params)
    )


# Per unit forward: the reset gate is an add and a sigmoid; the update gate two adds
# and a sigmoid; the candidate a product, an add, and a tanh; the interpolation a
# subtraction, two products, and an add. The adjoint reuses the saved gates: five for the
# interpolation, three for each of the three nonlinearities, two through the reset
# product, and three accumulating the residual's four gradient paths.
def _gate_cost(*, embed_dim: int, rows: float, dtype: torch.dtype | None) -> Cost:
    """Cost one gated residual's projections and scalar-region tensor boundary."""
    dt = dtype
    return matmul_cost(
        channels_in=embed_dim,
        channels_out=embed_dim,
        rows=rows,
        dtype=dt,
    ).tile(6, copies=6) + elementwise_cost(
        primal=12 * embed_dim,
        adjoint=19 * embed_dim,
        channels=embed_dim,
        inputs=7,
        outputs=2,
        adjoint_inputs=7,
        adjoint_outputs=8,
        params=embed_dim,
        rows=rows,
        dtype=dt,
    )


# Without an explicit ``gain`` the weights are drawn from a two-sigma truncated normal
# scaled by the fan-in, which is the reference framework's default and differs from this
# one's uniform default by enough to change where training starts.
def _dense(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    gain: float | None = None,
    truncation: float = 0.87962566103423978,
) -> nn.Linear:
    """Build a linear layer initialized the way the reference does."""
    layer = nn.Linear(in_features, out_features, bias=bias)
    if gain is None:
        deviation = (1.0 / in_features) ** 0.5 / truncation
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


# Slots are numbered from ``length`` down to one, so the encoding of "one step ago" is
# the same vector whatever the window's absolute position -- which is the property that
# lets a sliding memory be attended over at all.
def _sinusoidal_positions(
    length: int,
    *,
    embed_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Encode each key slot by how far in the past it is."""
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
