"""Gated Transformer-XL actor-critic: a policy that attends over its own past.

Craftax rewards long plans -- mine coal, then iron, then find a furnace -- and
a feed-forward policy sees only the current view. This network keeps the last
``memory_length`` steps' layer inputs as attention memory, so the policy that
decides now can condition on what it did a hundred steps ago, each step
individually addressable.

Two mechanisms make that memory trainable. Attention is RELATIVE: a key is
scored by how far back it is, not by its absolute place, so a memory window
that slides every step keeps meaning the same thing. And each residual
connection is GATED by a GRU-style update gate initialized closed, so an
untrained layer passes its input through unchanged and the transformer starts
as the identity -- which stops the early, high-variance policy gradients from
destroying it.

The memory is the policy's carry, as the rollout threads every policy's:
``[memory_length, envs, num_layers * channels_hidden + 1]``, oldest row
first. A row holds one step's input to every layer, then a 1 marking it
written. The rollout zeroes an environment's rows where its episode ends,
which empties the memory, flags included; each step appends a row and drops
the oldest. A window of steps (:meth:`GTrXLPolicy.forward`) attends over the
memory it starts from and its own earlier steps, never across an episode
start; a rollout step is a window of one.

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

from dataclasses import field, replace
from typing import Self, override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.craftax.lib.costs import concat_cost
from priml.baselines.craftax.policies.actor_critic import ActorCritic
from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    resolve_dtype,
    traffic,
)
from priml.model.init import corrected_fan_in_normal
from priml.model.linear import Linear
from priml.model.norm import LayerNorm


class GTrXLBlock(nn.Module):
    """One pre-norm layer: gated relative attention, then a gated MLP."""

    class Config(Fig["GTrXLBlock"]):
        """The layer's widths and its gates' initial bias."""

        channels_hidden: int = -1
        """Width of the stream the layer reads and writes; the policy's."""

        heads: int = 8
        """Attention heads."""

        channels_head: int = 32
        """Width of each head's query, key and value."""

        gating_bias: float = 2.0
        """Bias subtracted inside every update gate at initialization.

        Positive closes the gate, so an untrained layer is the identity and the
        memory survives the first updates."""

        eps: float = 1e-6
        """The two layer norms' floor."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            memory_len: int = 0,
            **kwargs: object,
        ) -> Cost:
            """Cost one block over windows of queries and the memory rows before them.

            Each of ``batch_size`` windows scores its ``seq_len`` queries against
            ``memory_len + seq_len`` keys. Every key row is normalized and
            projected once, and the relative-position table, which the windows
            share, once. The table is a constant, so its projection forms only
            its weight's gradient.

            Args:
              seq_len: Queries per window.
              batch_size: Windows in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              memory_len: Remembered rows ahead of each window's keys.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            rows, keys = seq_len * batch_size, memory_len + seq_len
            width, qkv = self.channels_hidden, self.heads * self.channels_head
            norm = LayerNorm.Config(width, elementwise_affine=True).finalize()
            # One norm reads the keys and the queries, so it owns its weights once.
            norms = cost(
                norm,
                seq_len=keys,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + cost(
                norm,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ).tile(1, copies=0)
            projection = matmul_cost(
                channels_in=width,
                channels_out=qkv,
                bias=True,
                rows=rows,
                dtype=dtype,
            )
            key_value = matmul_cost(
                channels_in=width,
                channels_out=qkv,
                bias=True,
                rows=batch_size * keys,
                dtype=dtype,
            ).tile(2, copies=2)
            # The content and position biases are each added to the query; the
            # adjoint sums the two score paths' gradients back into it.
            query_biases = elementwise_cost(
                primal=2 * rows * qkv,
                adjoint=rows * qkv,
                channels=qkv,
                inputs=2,
                outputs=2,
                adjoint_inputs=2,
                params=2 * qkv,
                rows=rows,
                dtype=dtype,
            )
            relu = elementwise_cost(
                primal=rows * width,
                adjoint=rows * width,
                channels=width,
                rows=rows,
                dtype=dtype,
            )
            attention = (
                norms
                + projection
                + query_biases
                + key_value
                + _weight_only_matmul_cost(
                    channels_in=width,
                    channels_out=qkv,
                    rows=keys,
                    dtype=dtype,
                )
                + _relative_scores_cost(
                    heads=self.heads,
                    channels_head=self.channels_head,
                    keys=keys,
                    rows=rows,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + matmul_cost(
                    channels_in=qkv,
                    channels_out=width,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                )
                + relu
            )
            # GELU: scale, erf, add, halve, multiply; the adjoint adds the density.
            mlp = (
                cost(
                    norm,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + matmul_cost(
                    channels_in=width,
                    channels_out=width,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                ).tile(2, copies=2)
                + elementwise_cost(
                    primal=rows * 5 * width,
                    adjoint=rows * 6 * width,
                    channels=width,
                    rows=rows,
                    dtype=dtype,
                )
                + relu
            )
            gate = _gate_cost(channels_hidden=width, rows=rows, dtype=dtype)
            return attention + gate + mlp + gate

    def __init__(self, config: Config) -> None:
        """Build the norms, the attention, the MLP and their gates.

        Args:
          config: The layer's geometry.

        Raises:
          ValueError: A width or the head count is not positive.

        """
        if min(config.channels_hidden, config.heads, config.channels_head) <= 0:
            raise ValueError("GTrXL block widths and heads must be positive")
        super().__init__()
        width = config.channels_hidden
        self.norm_attention = nn.LayerNorm(width, eps=config.eps)
        self.attention = _RelativeAttention(
            channels_hidden=width,
            heads=config.heads,
            channels_head=config.channels_head,
        )
        self.gate_attention = _Gate(channels_hidden=width, bias=config.gating_bias)
        self.norm_mlp = nn.LayerNorm(width, eps=config.eps)
        self.mlp_in = _dense(width, width, bias=True)
        self.mlp_out = _dense(width, width, bias=True)
        self.gate_mlp = _Gate(channels_hidden=width, bias=config.gating_bias)

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
          keys: The memory's rows, then the window's own inputs, ``[batch, keys,
            width]``.
          queries: The window's inputs, ``[batch, time, width]``.
          positional: One relative encoding per key slot, ``[keys, width]``.
          mask: Which keys each query may read, ``[batch, 1, time, keys]``.

        Returns:
          hidden: The layer's output, ``[batch, time, width]``.

        """
        # One norm for both sides: keys and queries are one representation at
        # different times, and normalizing them apart would make a remembered
        # step incomparable to the step that queries it.
        attended = self.attention(
            self.norm_attention(queries),
            self.norm_attention(keys),
            positional,
            mask,
        )
        attended = self.gate_attention(queries, attended.relu())
        hidden = self.mlp_out(functional.gelu(self.mlp_in(self.norm_mlp(attended))))
        return self.gate_mlp(attended, hidden.relu())


class GTrXLPolicy(nn.Module):
    """A gated Transformer-XL over the observation, then separate actor and critic towers.

    The encoder projects each observation to ``channels_hidden``; ``num_layers``
    blocks attend over the memory and the window; the decoder, Craftax_Baselines'
    actor-critic towers over the last block's output, emits the fused row, the
    action logits then the value.
    """

    class Config(Fig["GTrXLPolicy"]):
        """The geometry; the encoder's, blocks' and decoder's widths derive from it."""

        observation_size: int = 8_268
        """Floats per observation: original Craftax's symbolic view by default."""

        num_actions: int = 43
        """Logits per observation; the fused row adds one for the value."""

        channels_hidden: int = 256
        """Width of the stream every block reads and writes, and of each memory
        row's entry per layer."""

        num_layers: int = 2
        """Blocks, broadcast from ``block``."""

        memory_length: int = 128
        """Steps of layer inputs the memory holds, each attended individually."""

        proj_in: Linear.Config = field(
            default_factory=lambda: _fan_in_linear(bias=True),
        )
        """The encoder, ``observation_size -> channels_hidden``."""

        block: GTrXLBlock.Config = field(default_factory=GTrXLBlock.Config)
        """Block template, broadcast ``num_layers`` times."""

        decoder: ActorCritic.Config = field(
            default_factory=lambda: _towers(channels_hidden=256, num_layers=2),
        )
        """The actor and critic towers over the last block's output."""

        dtype: torch.dtype = torch.float32
        """Parameter, activation and memory dtype, and so the rollout's stored
        observations'."""

        @override
        def finalize(self) -> Self:
            self.proj_in.channels_in = self.observation_size
            self.proj_in.channels_out = self.channels_hidden
            self.block.channels_hidden = self.channels_hidden
            self.decoder.observation_size = self.channels_hidden
            self.decoder.num_actions = self.num_actions
            self.decoder.dtype = self.dtype
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one window of every environment over a full memory.

            ``seq_len`` steps of each of ``batch_size`` environments are scored
            together, each attending over every key of its memory and window;
            a rollout step is a window of one. Every stage runs at the policy's
            ``dtype``. ``bytes_state`` is what one step adds to the memory: one
            row of every layer's input and its flag. Keeping the memory -- the
            reset, the append -- is cache bookkeeping, uncosted like a KV-cache
            update.

            Args:
              seq_len: Steps per window.
              batch_size: Windows in this invocation.
              dtype: Unread; the policy's own ``dtype`` prices every stage.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            del dtype
            blocks = self.block.cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=self.dtype,
                memory_len=self.memory_length,
                **kwargs,
            ).tile(self.num_layers, copies=self.num_layers)
            total = (
                self.proj_in.cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=self.dtype,
                    **kwargs,
                )
                + blocks
                + _decoder_cost(
                    self.decoder,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=self.dtype,
                    **kwargs,
                )
            )
            row = self.num_layers * self.channels_hidden + 1
            return replace(
                total,
                bytes_state=self.dtype.itemsize * batch_size * row,
            )

    def __init__(self, config: Config) -> None:
        """Build the encoder, the blocks and the decoder.

        Args:
          config: The geometry.

        Raises:
          ValueError: ``num_layers`` or ``memory_length`` is not positive, or a
            child refuses its widths.

        """
        if config.num_layers <= 0 or config.memory_length <= 0:
            raise ValueError("num_layers and memory_length must be positive")
        super().__init__()
        self.memory_length = config.memory_length
        self.channels_hidden = config.channels_hidden
        self.dtype = config.dtype
        self.proj_in = config.proj_in.make()
        self.blocks: nn.ModuleList[GTrXLBlock] = nn.ModuleList(
            config.block.copy_tree().make() for _ in range(config.num_layers)
        )
        self.decoder = config.decoder.make()
        self.to(config.dtype)

    def initial_state(
        self,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Return the empty memory every environment starts from.

        Args:
          num_envs: Environments the memory covers.
          device: Where it lives.

        Returns:
          state: Zeros, ``[memory_length, num_envs, num_layers *
            channels_hidden + 1]``: no row written.

        """
        return torch.zeros(
            self.memory_length,
            num_envs,
            len(self.blocks) * self.channels_hidden + 1,
            device=device,
            dtype=self.dtype,
        )

    @override
    def forward(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Score a window of steps over the memory it starts from.

        Step ``t`` attends over the memory, unless an episode starts at or
        before ``t`` in the window, and over the window's steps from the last
        such start through ``t``.

        Args:
          observations: ``[batch, time, observation_size]``.
          state: The memory before the first step, ``[memory_length, batch,
            num_layers * channels_hidden + 1]``.
          episode_start: Nonzero where an episode starts at that step,
            ``[batch, time]``.

        Returns:
          decoded: ``[batch, time, num_actions + 1]``: the logits, then the value.
          state: The memory after the last step: the last ``memory_length``
            rows of the memory and the window's own layer inputs, those before
            the window's last episode start emptied; not differentiated.

        """
        batch, time = episode_start.shape
        memory = state.transpose(0, 1)
        # Each episode start opens a segment: two steps share an episode exactly
        # when their running counts agree, and the memory predates the window,
        # so only the window's first segment may read it.
        segments = (episode_start != 0).to(torch.int64).cumsum(dim=-1)
        steps = torch.arange(time, device=state.device)
        causal = (steps[:, None] >= steps[None, :]) & (
            segments[:, :, None] == segments[:, None, :]
        )
        remembered = (segments == 0)[:, :, None] & (memory[:, None, :, -1] != 0)
        mask = torch.cat((remembered, causal), dim=-1)[:, None]
        positional = _sinusoidal_positions(
            self.memory_length + time,
            channels=self.channels_hidden,
            device=state.device,
            dtype=self.dtype,
        )
        hidden = self.proj_in(observations.to(self.dtype))
        inputs: list[Tensor] = []
        width = self.channels_hidden
        for index, block in enumerate(self.blocks):
            inputs.append(hidden)
            rows = memory[..., index * width : (index + 1) * width]
            keys = torch.cat((rows, hidden), dim=1)
            hidden = block(keys, hidden, positional, mask)
        decoded = self.decoder(hidden)
        last = segments[:, -1:]
        written = torch.cat(
            (*inputs, hidden.new_ones(batch, time, 1)),
            dim=-1,
        ).detach()
        written = torch.where((segments == last)[..., None], written, 0.0)
        kept = torch.where((last == 0)[..., None], memory, 0.0)
        carry = torch.cat((kept, written), dim=1)[:, -self.memory_length :]
        return decoded, carry.transpose(0, 1)

    def forward_fused(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor | None,
        *,
        carry: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Score one step of every environment, as the rollout does: a window of one.

        Args:
          observations: ``[batch, observation_size]``.
          state: The memory, ``[memory_length, batch, width]``.
          episode_start: Nonzero where the memory empties first, ``[batch]``;
            None when the caller has already emptied those rows.
          carry: Where the next memory is written; a new tensor if None. The
            rollout passes ``state`` to advance it in place.
          features: Must be None: the policy reads no feature.

        Returns:
          decoded: ``[batch, num_actions + 1]``: the logits, then the value.
          state: The memory with this step's row appended; ``carry`` if given.

        """
        _refuse_feature(features)
        starts = (
            torch.zeros(observations.shape[0], 1, device=state.device)
            if episode_start is None
            else episode_start[:, None]
        )
        decoded, following = self(observations[:, None], state, starts)
        if carry is None:
            return decoded[:, 0], following.contiguous()
        # ``following`` is a new tensor, so it never aliases ``carry``.
        return decoded[:, 0], carry.copy_(following)

    def forward_sequence(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor,
        *,
        actions: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Score a window for the learner to differentiate; see :meth:`forward`.

        Args:
          observations: ``[batch, time, observation_size]``.
          state: The memory before the first step.
          episode_start: Nonzero where an episode starts, ``[batch, time]``.
          actions: Unused: the policy has no auxiliary loss.
          features: Must be None: the policy reads no feature.

        Returns:
          decoded: ``[batch, time, num_actions + 1]``.
          final: The memory after the last step; not differentiated.
          auxiliary_loss: Zero, 0-dim fp32.

        """
        del actions
        _refuse_feature(features)
        decoded, final = self(observations, state, episode_start)
        return decoded, final, decoded.new_zeros((), dtype=torch.float32)


class _RelativeAttention(nn.Module):
    """Multi-head attention scored by each key's lag rather than its absolute place."""

    def __init__(self, *, channels_hidden: int, heads: int, channels_head: int) -> None:
        super().__init__()
        self.heads = heads
        self.channels_head = channels_head
        width = heads * channels_head
        self.query = _dense(channels_hidden, width, bias=True)
        self.key = _dense(channels_hidden, width, bias=True)
        self.value = _dense(channels_hidden, width, bias=True)
        self.relative_position = _dense(channels_hidden, width, bias=False)
        self.bias_content = nn.Parameter(torch.zeros(heads, channels_head))
        self.bias_position = nn.Parameter(torch.zeros(heads, channels_head))
        self.out = _dense(width, channels_hidden, bias=True)

    @override
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        positional: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Score every query against every key it may read.

        Args:
          queries: Normalized window steps, ``[batch, time, width]``.
          keys: Normalized memory rows and window steps, ``[batch, keys, width]``.
          positional: Relative encodings, ``[keys, width]``.
          mask: Broadcastable bool over ``[batch, heads, time, keys]``.

        Returns:
          attended: ``[batch, time, width]``.

        """
        time = queries.shape[1]
        query = self._split(self.query(queries))
        key = self._split(self.key(keys))
        value = self._split(self.value(keys))
        relative = self._split(self.relative_position(positional))
        content = (query + self.bias_content).transpose(1, 2)
        weights = content @ key.transpose(1, 2).transpose(-2, -1)
        position = (query + self.bias_position).transpose(1, 2)
        relative_weights = position @ relative.transpose(0, 1).transpose(-2, -1)
        # The encodings are shared across queries, so each query row reads them at
        # its own offset: its own key, the last of its row, at lag zero, which is
        # slot ``keys - 1``'s encoding, and each step back one slot earlier.
        shifts = torch.arange(time, device=queries.device) - (time - 1)
        source = (
            torch.arange(relative_weights.shape[-1], device=queries.device)[None, :]
            - shifts[:, None]
        ) % relative_weights.shape[-1]
        relative_weights = relative_weights.gather(
            -1,
            source[None, None].expand(relative_weights.shape),
        )
        weights = (weights + relative_weights) / self.channels_head**0.5
        weights = torch.where(mask, weights, -1e30).softmax(-1)
        attended = (weights @ value.transpose(1, 2)).transpose(1, 2)
        return self.out(attended.flatten(-2))

    def _split(self, projected: Tensor) -> Tensor:
        """Reshape a flat projection into per-head vectors."""
        return projected.unflatten(-1, (self.heads, self.channels_head))


class _Gate(nn.Module):
    """A GRU-style gated residual, initialized closed."""

    def __init__(self, *, channels_hidden: int, bias: float) -> None:
        super().__init__()
        self.gating_bias = nn.Parameter(torch.full((channels_hidden,), bias))
        self.reset_y = _dense(channels_hidden, channels_hidden, bias=False)
        self.reset_x = _dense(channels_hidden, channels_hidden, bias=False)
        self.update_y = _dense(channels_hidden, channels_hidden, bias=False)
        self.update_x = _dense(channels_hidden, channels_hidden, bias=False)
        self.candidate_y = _dense(channels_hidden, channels_hidden, bias=False)
        self.candidate_x = _dense(channels_hidden, channels_hidden, bias=False)

    @override
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Interpolate the residual ``x`` toward a candidate built from ``y``.

        Args:
          x: The residual, passed through when the gate is closed.
          y: The sublayer's proposal.

        Returns:
          output: The gated combination.

        """
        reset = torch.sigmoid(self.reset_y(y) + self.reset_x(x))
        update = torch.sigmoid(self.update_y(y) + self.update_x(x) - self.gating_bias)
        candidate = torch.tanh(self.candidate_y(y) + self.candidate_x(reset * x))
        return (1.0 - update) * x + update * candidate


def _refuse_feature(features: Tensor | None) -> None:
    """Raise when a feature reaches the policy, which has nothing to read it with."""
    if features is not None:
        raise ValueError("GTrXLPolicy reads no feature")


# The reference framework's default dense init: a fan-in truncated normal whose
# realized deviation is 1/sqrt(fan_in), not torch's uniform, which starts training
# elsewhere.
def _fan_in_linear(*, bias: bool) -> Linear.Config:
    """Return a projection drawn as the reference draws its dense layers, biases zero."""
    config = Linear.Config()
    config.bias = bias
    config.init_weight = corrected_fan_in_normal
    return config


def _dense(channels_in: int, channels_out: int, *, bias: bool) -> Linear:
    """Build one of a block's projections."""
    config = _fan_in_linear(bias=bias)
    config.channels_in = channels_in
    config.channels_out = channels_out
    return config.make()


def _towers(*, channels_hidden: int, num_layers: int) -> ActorCritic.Config:
    """Return Craftax_Baselines' towers with ReLU between the layers, as the reference's."""
    config = ActorCritic.Config()
    config.channels_hidden = channels_hidden
    config.num_layers = num_layers
    config.activation = torch.relu
    return config


# Each tower's input is the last block's output, which takes a gradient, so unlike
# exp003's towers over the observation the first projection forms its input's
# gradient too. ReLU is one compare-and-select each way.
def _decoder_cost(
    decoder: ActorCritic.Config,
    *,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype | None,
    **kwargs: object,
) -> Cost:
    """Cost both towers, their ReLUs and the fused row's concatenation."""
    rows, width = seq_len * batch_size, decoder.channels_hidden
    first, later, policy, value = (
        projection.cost(seq_len=seq_len, batch_size=batch_size, dtype=dtype, **kwargs)
        for projection in (
            decoder.proj_in,
            decoder.proj_hidden,
            decoder.proj_policy,
            decoder.proj_value,
        )
    )
    relu = elementwise_cost(
        primal=rows * width,
        adjoint=rows * width,
        channels=width,
        rows=rows,
        dtype=dtype,
    )
    hidden = decoder.num_layers - 1
    tower = first + later.tile(hidden, copies=hidden) + relu.tile(decoder.num_layers)
    fused = concat_cost(elements=rows * (decoder.num_actions + 1), dtype=dtype)
    return tower.tile(2, copies=2) + policy + value + fused


def _weight_only_matmul_cost(
    *,
    channels_in: int,
    channels_out: int,
    rows: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost a projection whose input is constant and takes no gradient."""
    full = matmul_cost(
        channels_in=channels_in,
        channels_out=channels_out,
        rows=rows,
        dtype=dtype,
    )
    dt = resolve_dtype(dtype)
    cells = dict(full.cells)
    cells["flops", "adjoint", "matmul", dt] //= 2
    cells["bytes", "adjoint", "matmul", dt] //= 2
    return replace(full, cells=cells)


def _activation_rhs_matmul_cost(
    *,
    channels_in: int,
    channels_out: int,
    rows: int,
    batch_size: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost a product whose right operand is a distinct activation per window."""
    full = matmul_cost(
        channels_in=channels_in,
        channels_out=channels_out,
        weight=False,
        rows=rows,
        dtype=dtype,
    )
    dt = resolve_dtype(dtype)
    extra = (batch_size - 1) * channels_in * channels_out * dt.itemsize
    cells = dict(full.cells)
    cells["bytes", "primal", "matmul", dt] += extra
    cells["bytes", "adjoint", "matmul", dt] += 2 * extra
    return replace(full, cells=cells)


# Content and position scores are the same product on two biased copies of the query;
# the position scores are then gathered into place, one element moved per score and one
# scatter-add back. The combined score is added, scaled, masked, then softmaxed: subtract
# the max, exponentiate, divide, with the max and the sum as the reductions and the
# adjoint's ``sum(g * p)`` as one more. The gather's index is one ``int64`` per score
# each way.
def _relative_scores_cost(
    *,
    heads: int,
    channels_head: int,
    keys: int,
    rows: int,
    batch_size: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost unfused relative attention for every query row and head."""
    scores = _activation_rhs_matmul_cost(
        channels_in=channels_head,
        channels_out=keys,
        rows=rows,
        batch_size=batch_size,
        dtype=dtype,
    )
    values = _activation_rhs_matmul_cost(
        channels_in=keys,
        channels_out=channels_head,
        rows=rows,
        batch_size=batch_size,
        dtype=dtype,
    )
    gather = (
        traffic("primal", "selection", elements=rows * keys, dtype=dtype)
        + traffic("primal", "selection", elements=rows * keys, dtype=torch.int64)
        + traffic(
            "adjoint",
            "selection",
            elements=2 * rows * keys,
            flops=rows * keys,
            dtype=dtype,
        )
        + traffic("adjoint", "selection", elements=rows * keys, dtype=torch.int64)
    )
    # Add, scale, mask, subtract, exp, divide; back, the two score paths too.
    softmax = elementwise_cost(
        primal=rows * 6 * keys,
        adjoint=rows * 6 * keys,
        channels=keys,
        inputs=9,
        outputs=6,
        adjoint_inputs=10,
        adjoint_outputs=6,
        rows=rows,
        dtype=dtype,
    ) + (
        reduction_cost(
            input_elements=rows * keys,
            output_groups=rows,
            dtype=dtype,
        ).tile(2, copies=2)
        + reduction_cost(
            input_elements=rows * keys,
            output_groups=rows,
            dtype=dtype,
            phase="adjoint",
        )
    )
    return (scores.tile(2, copies=2) + gather + softmax + values).tile(
        heads,
        copies=heads,
    )


# Per unit forward: the reset gate is an add and a sigmoid; the update gate two adds
# and a sigmoid; the candidate a product, an add, and a tanh; the interpolation a
# subtraction, two products, and an add. The adjoint reuses the saved gates: five for
# the interpolation, three for each of the three nonlinearities, two through the reset
# product, and three accumulating the residual's four gradient paths.
def _gate_cost(*, channels_hidden: int, rows: int, dtype: torch.dtype | None) -> Cost:
    """Cost one gated residual's six projections and its pointwise arithmetic."""
    return matmul_cost(
        channels_in=channels_hidden,
        channels_out=channels_hidden,
        rows=rows,
        dtype=dtype,
    ).tile(6, copies=6) + elementwise_cost(
        primal=rows * 12 * channels_hidden,
        adjoint=rows * 19 * channels_hidden,
        channels=channels_hidden,
        inputs=7,
        outputs=2,
        adjoint_inputs=7,
        adjoint_outputs=8,
        params=channels_hidden,
        rows=rows,
        dtype=dtype,
    )


# Slots are numbered from ``length`` down to one, so the encoding of "one step ago"
# is the same vector wherever the window sits -- the property that lets a sliding
# memory be attended over at all.
def _sinusoidal_positions(
    length: int,
    *,
    channels: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Encode each key slot by how far in the past it is, ``[length, channels]``."""
    frequency = 1.0 / (
        10_000
        ** (
            torch.arange(0.0, channels, 2.0, device=device, dtype=torch.float32)
            / channels
        )
    )
    distance = torch.arange(length, 0, -1, device=device, dtype=torch.float32)
    angles = torch.outer(distance, frequency)
    return torch.cat((angles.sin(), angles.cos()), dim=-1).to(dtype)
