"""The MinGRU actor-critic over Craftax's observation, as PufferLib builds and runs it.

An encoder, a recurrent trunk and a decoder, each a bf16 GEMM without bias:

- ``embedding``: the 99 view cells of the packed observation each carry eight
  small integers (block, item, visibility, five mob channels), embedded by one
  :class:`~priml.model.embedding.MultiHotEmbedding` over a 154-row table;
  the 51 status scalars are appended, giving 1,635 features. For a dense
  layout -- original Craftax's 8,268-float symbolic view --
  :class:`DenseObservation` takes the embedding's place: the observation is the
  features, and ``proj_in`` reads it directly. :class:`NoEncoder` encodes no
  observation at all: the policy then builds no ``proj_in``, and its trunk
  reads ``proj_feature``'s projection of the feature alone.
- ``proj_in``: 1,635 (or the dense width) -> 1,024.
- ``num_layers`` blocks of :class:`~priml.model.min_gru.MinGRUBlock`,
  their carry kept in ``state_dtype``: fp32, or PufferLib's bf16 for exp000.
- ``proj_out``: one fused 44-row decoder, 43 action logits and the value,
  output in ``output_dtype``: fp32, or PufferLib's bf16 for exp000. An fp32
  output's gradient is rounded once to bf16, into the bf16 decoder's GEMMs.

Three optional slots add to that trunk; None leaves it PufferLib's exactly:

- ``injection``: one module per block, each adding a function of
  ``proj_in``'s output to its block's input, so every layer reads the frame
  again.
- ``auxiliary``: a loss on the last block's output beside the learning
  rule's, e.g. :class:`FeasibilityLoss`; the learner's window forward returns
  it, and the rollout's step never computes it.
- ``proj_feature``: a projection of an external per-step feature (the
  rollout's ``feature``, e.g. a frozen world model's state) added to
  ``proj_in``'s output, so the frame encoding reads it too; after a
  :class:`NoEncoder` it is the trunk's only input.

The learner differentiates the policy with autograd. The projections are
plain GEMMs; each block's scan and the table pass their gradients through
their own backward kernels, the table's summed in fixed point so that scatter
order cannot move a bit. Those are the GEMMs and kernels PufferLib's
hand-written backward issues, so the gradients are its bits.

Weights are drawn as PufferLib draws them -- the table from N(0, 1), every
projection from U(+-1/sqrt(fan_in)), in fp32 and rounded once to the model
dtype -- but by torch's generator, not PufferLib's XORWOW stream; a run that
must reproduce PufferLib's bits starts from PufferLib's own weights, converted
once to a ``state_dict``.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``src/algo.cu`` and
        ``ocean/craftax/craftax.cu``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, Protocol, Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.craftax.lib.costs import residual_cost, weight_gradient_only
from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    reduction_cost,
    traffic,
)
from priml.loss.simple_loss import bce_with_logits
from priml.model.custom_types import ChannelsInOut
from priml.model.embedding import MultiHotEmbedding
from priml.model.linear import Linear
from priml.model.min_gru import MinGRUBlock


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class Policy(Protocol):
    """What the actor and the learner call on a policy.

    Every step's output is one fused row, the action logits then the value, as
    the sampler and the learning rule read it. The carry is ``[layers, envs,
    width]``: the actor zeroes an environment's rows where its episode ended.
    """

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the rollout's stored observations, masks and terminals."""
        ...

    def initial_state(
        self,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Return the zero carry every environment starts from, in the carry's dtype."""
        ...

    def forward_fused(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor | None,
        *,
        carry: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Score one step: the fused rows ``[envs, num_actions + 1]`` and the carry.

        Args:
          observations: ``[envs, observation_size]``.
          state: The carry.
          episode_start: Nonzero where the carry resets first, ``[envs]``;
            None when the caller has already zeroed those rows.
          carry: Where the next carry is written; a new tensor if None.
          features: The rollout's per-step feature ``[envs, width]``; None
            without one. A policy refuses a feature it does not read, and the
            lack of one it does.

        Returns:
          decoded: ``[envs, num_actions + 1]``, the logits then the value.
          state: The next carry; ``carry`` if given.

        """
        ...

    def forward_sequence(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor,
        *,
        actions: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Score a window for the learner to differentiate.

        Args:
          observations: ``[batch, time, observation_size]``.
          state: The carry before the first step.
          episode_start: Nonzero where the carry resets, ``[batch, time]``.
          actions: The actions taken, ``[batch, time]``; an auxiliary loss
            that scores them needs them, and None skips it.
          features: The feature each step's actor read, ``[batch, time,
            width]``; None without a feature.

        Returns:
          decoded: ``[batch, time, num_actions + 1]``, the logits then the value.
          final: The carry after the last step.
          auxiliary_loss: 0-dim fp32, the policy's own loss beside the
            learning rule's; zero for a policy without one.

        """
        ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Yield the weights in the order an optimizer takes them."""
        ...

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield the weights with their ``state_dict`` names."""
        ...

    def state_dict(self) -> dict[str, Tensor]:
        """Return the weights by name."""
        ...

    def load_state_dict(self, state_dict: Mapping[str, Tensor]) -> object:
        """Copy weights in by name."""
        ...

    def to(self, device: torch.device | str) -> Self:
        """Move the weights to ``device``."""
        ...


class PolicyConfig(Makeable[Policy], Protocol):
    """A policy's config: it builds the policy and states the observations it reads."""

    @property
    def observation_size(self) -> int:
        """Floats per observation the policy reads."""
        ...


class ObservationFeatures(Protocol):
    """The policy's first stage: observations in, ``proj_in``'s input out."""

    def __call__(self, input: Tensor) -> Tensor:
        """Map observations ``[..., observation_size]`` to ``[..., channels_concat]``."""
        ...


class FeatureStage(Makeable[ObservationFeatures], Protocol):
    """A first stage's config: it builds the stage and states both its widths."""

    @property
    def observation_size(self) -> int:
        """Floats per observation the stage reads."""
        ...

    @property
    def channels_concat(self) -> int:
        """Width of the features the stage produces."""
        ...


class InjectionStage(Makeable[nn.Module], ChannelsInOut, Protocol):
    """A block injection's config: a ``channels_in -> channels_out`` tensor module."""


class AuxiliaryStage(Makeable[nn.Module], ChannelsInOut, Protocol):
    """An auxiliary loss's config: ``channels_in`` features in, ``channels_out`` actions scored.

    The module it builds maps a window's last-block features ``[batch, time,
    channels_in]``, its observations in their stored dtype, its episode starts
    and its actions taken (both ``[batch, time]``) to a 0-dim fp32 loss, as
    :meth:`FeasibilityLoss.forward` does.
    """


def packed_observation_embedding() -> MultiHotEmbedding.Config:
    """Return the embedding of Craftax's packed observation.

    Each of the 99 view cells holds eight ids: the block, the item plus one, a
    0/1 light flag, and one plus the species of each of five creature classes.
    Each id indexes its own rows of one 154-row table; the 51 status scalars
    follow the cells.

    Returns:
      config: The table's geometry and the observation's layout.

    """
    embedding = MultiHotEmbedding.Config()
    embedding.channels_in = 154
    embedding.channels_out = 16
    embedding.offsets = (0, 64, 72, 74, 90, 106, 122, 138)
    embedding.num_cells = 99
    embedding.num_scalars = 51
    return embedding


class DenseObservation(nn.Module):
    """Pass a dense observation through as the features.

    Original Craftax's symbolic observation -- one-hot map channels and scaled
    scalars -- is already a feature vector, so ``proj_in`` reads it directly
    and this stage has no weights. The policy rounds observations to its dtype
    before its first stage, so the identity is the whole forward.
    """

    class Config(Fig["DenseObservation"]):
        """The observation's width, which is the features'."""

        observation_size: int = 8268
        """Floats per observation, and so the features' width; original
        Craftax's symbolic view by default (``constants.SYMBOLIC_OBS_SIZE``)."""

        @property
        def channels_concat(self) -> int:
            """Width of the features this module produces."""
            return self.observation_size

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost nothing: the observations are the features.

            Args:
              seq_len: Rows per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; unused.

            Returns:
              cost: An empty cost, owning no parameters.

            """
            del seq_len, batch_size, dtype, kwargs
            return Cost()

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Return the observations: they are the features."""
        return input


class NoEncoder(nn.Module):
    """Encode no observation: a first stage of no features, for a policy that reads only its feature.

    It states the width of the observations the policy is handed, which the
    train step checks against the env's and an auxiliary loss reads its
    targets from. With no features there is nothing for ``proj_in`` to
    project, so :class:`MinGRUPolicy` builds none, and its trunk's input is
    ``proj_feature``'s alone. It has no weights.
    """

    class Config(Fig["NoEncoder"]):
        """The width of the observations the policy is handed."""

        observation_size: int = 843
        """Floats per observation; the packed row by default
        (``game.state.OBS_SIZE``), one more with the previous action."""

        @property
        def channels_concat(self) -> int:
            """No features."""
            return 0

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost nothing: no features are made.

            Args:
              seq_len: Rows per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; unused.

            Returns:
              cost: An empty cost, owning no parameters.

            """
            del seq_len, batch_size, dtype, kwargs
            return Cost()

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Return no features, ``[..., 0]``."""
        return input[..., :0]


class MinGRUPolicy(nn.Module):
    """PufferLib's four-layer MinGRU actor-critic over the packed observation."""

    class Config(Fig["MinGRUPolicy"]):
        """The geometry; every width below derives from ``channels_hidden``."""

        embedding: FeatureStage = field(default_factory=packed_observation_embedding)
        """The first stage: the multi-hot embedding of the packed layout, or
        :class:`DenseObservation` for a dense one; its features feed ``proj_in``.
        :class:`NoEncoder` produces none, and the trunk reads ``proj_feature``
        alone."""

        proj_in: Linear.Config = field(default_factory=Linear.Config)
        """Encoder projection, ``channels_concat -> channels_hidden``; not built
        when the first stage has no features."""

        block: MinGRUBlock.Config = field(default_factory=MinGRUBlock.Config)
        """Block template, broadcast ``num_layers`` times."""

        num_layers: int = 4
        """Recurrent blocks."""

        channels_hidden: int = 1_024
        """Width of every carry and of the trunk."""

        num_actions: int = 43
        """Logits per observation; the decoder adds one row for the value."""

        proj_out: Linear.Config = field(default_factory=Linear.Config)
        """Fused decoder, ``channels_hidden -> num_actions + 1``."""

        dtype: torch.dtype = torch.bfloat16
        """Parameter and activation dtype at every GEMM boundary. The children
        build in fp32, so each draws its init there, and the policy rounds
        every weight to this dtype once."""

        state_dtype: torch.dtype = torch.float32
        """The carry's dtype, which each scan rounds it to between steps. A bf16
        carry drops every update below half its ulp, so a unit whose gate is
        under 2^-9 can neither integrate nor decay; fp32 keeps them."""

        output_dtype: torch.dtype = torch.float32
        """The decoder's output dtype, the logits' and the value's: its GEMM sums
        the bf16 products in fp32 and keeps the sums in this dtype, and their
        gradient is rounded once to the weights' dtype on the way back. A bf16
        output rounds a trained critic's values near 16 to 0.125, under most of
        their step-to-step changes."""

        injection: InjectionStage | None = None
        """Template of each block's injection, ``channels_hidden ->
        channels_hidden``, built once per block: block ``l`` reads ``v +
        injection_l(x)``, ``v`` its usual input and ``x`` ``proj_in``'s output.
        None injects nothing."""

        auxiliary: AuxiliaryStage | None = None
        """A loss on the last block's output that the learner adds to its rule's;
        None adds nothing."""

        proj_feature: Linear.Config | None = None
        """Projection of the rollout's per-step feature, ``channels_in ->
        channels_hidden``, added to ``proj_in``'s output: set ``channels_in`` to
        the feature's width. Zero-initialized (``init_weight =
        nn.init.zeros_``), it draws no random number and the policy starts as
        it would without it. After a :class:`NoEncoder` it is the trunk's only
        input, which needs it, and draws as every projection does. None reads
        no feature."""

        @property
        def observation_size(self) -> int:
            """Floats per observation, as the first stage reads them."""
            return self.embedding.observation_size

        @override
        def finalize(self) -> Self:
            self.proj_in.channels_in = self.embedding.channels_concat
            self.proj_in.channels_out = self.channels_hidden
            self.block.channels_hidden = self.channels_hidden
            self.proj_out.channels_in = self.channels_hidden
            self.proj_out.channels_out = self.num_actions + 1
            if self.injection is not None:
                self.injection.channels_in = self.channels_hidden
                self.injection.channels_out = self.channels_hidden
            if self.auxiliary is not None:
                self.auxiliary.channels_in = self.channels_hidden
                self.auxiliary.channels_out = self.num_actions
            if self.proj_feature is not None:
                self.proj_feature.channels_out = self.channels_hidden
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the learner's window forward: every stage, then the auxiliary loss.

            Each stage runs at the policy's ``dtype``, to which it rounds the
            observations and every weight; ``dtype`` from the bus is unread.
            The auxiliary loss is the window's, so a rollout step, which never
            computes it, costs less.

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
            encodes = self.embedding.channels_concat > 0
            once: list[object] = [self.embedding, self.proj_out]
            once += [] if self.auxiliary is None else [self.auxiliary]
            each: list[object] = [self.block]
            each += [] if self.injection is None else [self.injection]
            total, layer = (
                sum(
                    (
                        cost(
                            stage,
                            seq_len=seq_len,
                            batch_size=batch_size,
                            dtype=dt,
                            **kwargs,
                        )
                        for stage in stages
                    ),
                    Cost(),
                )
                for stages in (once, each)
            )
            # The dense stage hands ``proj_in`` the observations themselves, and
            # the rollout's feature reaches ``proj_feature`` as data: neither
            # input takes a gradient, so back such a projection forms its
            # weight's alone. Every other stage's features are its own outputs.
            projections: list[tuple[object, bool]] = []
            if encodes:
                dense = isinstance(self.embedding, DenseObservation.Config)
                projections.append((self.proj_in, dense))
            if self.proj_feature is not None:
                projections.append((self.proj_feature, True))
            for projection, reads_data in projections:
                priced = cost(
                    projection,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dt,
                    **kwargs,
                )
                total += weight_gradient_only(priced) if reads_data else priced
            # An injection adds to its block's input; a feature, to the encoding.
            add = residual_cost(channels=self.channels_hidden, rows=rows, dtype=dt)
            if self.injection is not None:
                layer += add
            if encodes and self.proj_feature is not None:
                total += add
            return total + layer.tile(self.num_layers, copies=self.num_layers)

    def __init__(self, config: Config) -> None:
        """Build the stages in PufferLib's allocator order, then the optional slots.

        Args:
          config: The geometry and the slots.

        Raises:
          ValueError: The first stage has no features and no ``proj_feature``
            gives the trunk an input instead.

        """
        encodes = config.embedding.channels_concat > 0
        if not encodes and config.proj_feature is None:
            raise ValueError(
                "a first stage of no features leaves the trunk only proj_feature "
                "to read, and it is None",
            )
        super().__init__()
        self.channels_hidden = config.channels_hidden
        self.num_actions = config.num_actions
        self.dtype = config.dtype
        self.state_dtype = config.state_dtype
        self.output_dtype = config.output_dtype
        self.embedding: ObservationFeatures = config.embedding.make()
        # A zero-width ``proj_in`` would draw its init from U(+-1/sqrt(0)).
        self.proj_in = config.proj_in.make() if encodes else None
        # Registered before the blocks: ``parameters()`` then yields the first
        # stage's, ``proj_in``, ``proj_out`` and each block's gates, PufferLib's
        # allocator order. FusedMuon's global norm reduces the gradients in the
        # order it holds them, so exp000's bits depend on this one.
        self.proj_out = config.proj_out.make()
        self.blocks = nn.ModuleList(
            config.block.copy_tree().make() for _ in range(config.num_layers)
        )
        # After the blocks, so the parameters above keep exp000's order. None
        # registers no module at all.
        injection, auxiliary = config.injection, config.auxiliary
        self.injections = (
            None
            if injection is None
            else nn.ModuleList(
                injection.copy_tree().make() for _ in range(config.num_layers)
            )
        )
        self.auxiliary = None if auxiliary is None else auxiliary.make()
        # Last, so every parameter above keeps its place and its init draws.
        self.proj_feature = (
            None if config.proj_feature is None else config.proj_feature.make()
        )
        # Built in fp32 and rounded once here, as PufferLib draws its init. A
        # weight built in bf16 would draw in bf16, where torch's CPU generator
        # keeps 8 bits of each uniform (256 values per layer) and runs the
        # normal's Box-Muller in bf16.
        self.to(config.dtype)

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
          state: Zeros, ``[layers, num_envs, channels_hidden]``, in the
            carry's dtype.

        """
        return torch.zeros(
            len(self.blocks),
            num_envs,
            self.channels_hidden,
            device=device,
            dtype=self.state_dtype,
        )

    @override
    def forward(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Score one step of every environment, as the rollout does.

        Args:
          observations: Packed observations, ``[batch, observation_size]``.
          state: The carry, ``[layers, batch, channels_hidden]``.
          episode_start: Nonzero where the carry resets first, ``[batch]``.

        Returns:
          logits: ``[batch, num_actions]`` in the output dtype.
          values: ``[batch]`` in the output dtype.
          state: The next carry.

        """
        decoded, state = self.forward_fused(observations, state, episode_start)
        return decoded[:, :-1], decoded[:, -1], state

    def forward_fused(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor | None,
        *,
        carry: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Score one step, returning the decoder's fused output as the sampler reads it.

        Args:
          observations: Packed observations, ``[batch, observation_size]``.
          state: The carry, ``[layers, batch, channels_hidden]``.
          episode_start: Nonzero where the carry resets first, ``[batch]``;
            None when the caller has already zeroed those rows.
          carry: Where the next carry is written, layer by layer; a new
            tensor if None. The rollout passes ``state`` to advance it in
            place, saving the stack and the copy back.
          features: The step's feature ``[batch, width]``, which
            ``proj_feature`` reads; None without one.

        Returns:
          decoded: ``[batch, num_actions + 1]``: the logits, then the value,
            in the output dtype.
          state: The next carry; ``carry`` if given.

        """
        encoded = value = self._encode(
            self.embedding(observations.to(self.dtype)),
            features,
        )
        # ``zero_term_state`` runs before the forward, on the whole carry.
        if episode_start is not None:
            state = torch.where(episode_start[None, :, None] != 0, 0.0, state)
        carries: list[Tensor] = []
        for index, block in enumerate(self._blocks()):
            value, layer = block.step(
                self._inject(value, encoded, index),
                state[index],
                carry=None if carry is None else carry[index],
            )
            carries.append(layer)
        return self._decode(value), torch.stack(carries) if carry is None else carry

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
          observations: Packed observations, ``[batch, time, observation_size]``.
          state: The carry before the first step, ``[layers, batch,
            channels_hidden]``.
          episode_start: Nonzero where the carry resets before that step,
            ``[batch, time]``.
          actions: The actions taken, ``[batch, time]``, which the auxiliary
            loss scores; None skips it.
          features: The feature each step's actor read, ``[batch, time,
            width]``, which ``proj_feature`` reads; None without one.

        Returns:
          decoded: ``[batch, time, num_actions + 1]``: the logits, then the
            value, in the output dtype.
          final: The carry after the last step; not differentiated.
          auxiliary_loss: 0-dim fp32: the auxiliary loss, or zero without the
            slot or the actions.

        """
        batch, time = episode_start.shape
        embedded = self.embedding(observations.to(self.dtype)).flatten(0, 1)
        encoded = value = self._encode(
            embedded,
            None if features is None else features.flatten(0, 1),
        ).reshape(batch, time, -1)
        finals: list[Tensor] = []
        for index, block in enumerate(self._blocks()):
            value, final = block(
                self._inject(value, encoded, index),
                state[index],
                episode_start,
            )
            finals.append(final)
        decoded = self._decode(value.flatten(0, 1)).reshape(batch, time, -1)
        if self.auxiliary is None or actions is None:
            auxiliary_loss = decoded.new_zeros((), dtype=torch.float32)
        else:
            auxiliary_loss = cast(
                "Tensor",
                self.auxiliary(value, observations, episode_start, actions),
            )
        return decoded, torch.stack(finals), auxiliary_loss

    # A feature the policy has no ``proj_feature`` for is refused, as is the lack of one
    # it has.
    def _encode(self, embedded: Tensor, features: Tensor | None) -> Tensor:
        """Return ``proj_in``'s projection plus the feature's, either alone if the other is absent."""
        encoded = None if self.proj_in is None else self.proj_in(embedded)
        if self.proj_feature is None:
            if features is not None:
                raise ValueError("this policy has no proj_feature to read a feature")
            # The constructor builds ``proj_in`` wherever it builds no ``proj_feature``.
            assert isinstance(encoded, Tensor)
            return encoded
        if features is None:
            raise ValueError("this policy's proj_feature reads a feature; none came")
        projected = self.proj_feature(features.to(self.dtype))
        return projected if encoded is None else encoded + projected

    # Called per block, never hoisted into one batch before the loop: ``encoded``
    # feeds every injection and block 0's add, and autograd sums those gradients in
    # bf16 in the reverse of their creation order, which the reference's bits follow.
    def _inject(self, value: Tensor, encoded: Tensor, index: int) -> Tensor:
        """Return block ``index``'s input: ``value``, plus its injection of ``encoded``."""
        if self.injections is None:
            return value
        return value + cast("Tensor", self.injections[index](encoded))

    def _decode(self, features: Tensor) -> Tensor:
        """Return the fused rows, the logits then the value, in ``output_dtype``."""
        weight = self.proj_out.weight
        if weight.dtype == self.output_dtype:
            return self.proj_out(features)
        decoded = _WideLinear.apply(features, weight, self.output_dtype)
        bias = self.proj_out.bias
        return decoded if bias is None else decoded + bias.to(self.output_dtype)

    def _blocks(self) -> Iterator[MinGRUBlock]:
        yield from self.blocks


class FeasibilityLoss(nn.Module):
    """Predict whether each action taken changed what the agent sees next.

    A head scores every action from the last block's output; the taken
    action's logit ``phi_t`` is fit by binary cross-entropy, in fp32, to
    ``chi_t = 1[o_{t+1} != o_t]`` over the observation's first
    ``observation_size`` floats, compared in their stored dtype. A step
    counts where the next observation is in the same episode, ``t < T - 1``
    and no start at ``t + 1``:

        L = coefficient * sum_t valid_t BCE(phi_t, chi_t) / max(1, sum_t valid_t)

    summed over the whole window, every row.
    """

    class Config(Fig["FeasibilityLoss"], kw_only=False):
        """The head's widths, the compared fields and the loss's weight."""

        channels_in: int = -1
        """Width of the features the head reads: the policy's trunk."""

        channels_out: int = -1
        """Actions the head scores: the policy's action count."""

        _: KW_ONLY

        proj_out: Linear.Config = field(default_factory=Linear.Config)
        """The head, ``channels_in -> channels_out``."""

        observation_size: int = 843
        """Leading floats of an observation the target compares: PufferLib's
        packed row, and not the previous action id the env writes after it,
        which changes with every action."""

        coefficient: float = 1.0
        """Weight of the loss in the learner's total."""

        @override
        def finalize(self) -> Self:
            self.proj_out.channels_in = self.channels_in
            self.proj_out.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the head, the targets, the taken logit's pick, and the weighted BCE.

            The targets compare each step's ``observation_size`` floats with
            the next step's and reduce them to one flag, with no gradient. The
            pick reads one logit per step and scatters its gradient back into
            the head's row; the BCE runs in fp32, then a weighted mean.

            Args:
              seq_len: Steps per window.
              batch_size: Windows in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            rows = seq_len * batch_size
            compared = rows * self.observation_size
            fp32 = torch.float32
            targets = traffic(
                "primal",
                "elementwise",
                elements=3 * compared,
                flops=compared,
                dtype=dtype,
            ) + reduction_cost(input_elements=compared, output_groups=rows, dtype=dtype)
            pick = traffic(
                "primal",
                "selection",
                elements=2 * rows,
                dtype=dtype,
            ) + traffic(
                "adjoint",
                "selection",
                elements=rows * (1 + self.channels_out),
                flops=rows,
                dtype=dtype,
            )
            bce = cost(
                bce_with_logits,
                dtype=fp32,
                channels_out=1,
                weighted=False,
                rescale=0,
            ).tile(rows)
            mean = elementwise_cost(
                primal=rows + 1,
                adjoint=rows,
                channels=1,
                rows=rows,
                inputs=2,
                dtype=fp32,
            ) + reduction_cost(input_elements=2 * rows, output_groups=2, dtype=fp32)
            return (
                self.proj_out.cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + targets
                + pick
                + bce
                + mean
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.proj_out = config.proj_out.make()
        self.observation_size = config.observation_size
        self.coefficient = config.coefficient

    def targets(
        self,
        observations: Tensor,
        episode_start: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return each step's target and whether it counts.

        Args:
          observations: ``[batch, time, >= observation_size]``, stored dtype.
          episode_start: Nonzero where an episode starts, ``[batch, time]``.

        Returns:
          changed: bool ``[batch, time]``, ``chi_t``; False at the last step.
          valid: bool ``[batch, time]``: the next observation exists and is in
            the same episode.

        """
        compared = observations[..., : self.observation_size]
        changed = torch.zeros_like(episode_start, dtype=torch.bool)
        valid = torch.zeros_like(episode_start, dtype=torch.bool)
        changed[:, :-1] = (compared[:, 1:] != compared[:, :-1]).any(dim=-1)
        valid[:, :-1] = episode_start[:, 1:] == 0
        return changed, valid

    @override
    def forward(
        self,
        features: Tensor,
        observations: Tensor,
        episode_start: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Return the weighted loss, as :class:`AuxiliaryStage` states it."""
        changed, valid = self.targets(observations, episode_start)
        logits = self.proj_out(features).gather(-1, actions.long()[..., None])[..., 0]
        per_step = functional.binary_cross_entropy_with_logits(
            logits.float(),
            changed.float(),
            reduction="none",
        )
        weight = valid.to(per_step.dtype)
        return self.coefficient * (
            (per_step * weight).sum() / weight.sum().clamp_min(1)
        )


class _WideLinearContext(Protocol):
    """What :class:`_WideLinear` keeps between its forward and its backward."""

    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


# The gradient is rounded to the operands' dtype first, so both products are the bf16
# decoder's GEMMs. Differentiating in fp32 instead cost 2.2% of an epoch (H200,
# measured), for gradients the bf16 weights round away anyway.
def _wide_linear_backward(
    ctx: _WideLinearContext,
    /,
    *grad_outputs: Tensor,
) -> tuple[Tensor, Tensor, None]:
    """Round the output's gradient once to the operands' dtype; differentiate there."""
    (grad,) = grad_outputs
    features, weight = ctx.saved_tensors
    low = grad.to(weight.dtype)
    return low @ weight, low.T @ features, None


# One GEMM, bf16 operands and wide sums: on CUDA cuBLAS writes its fp32 accumulators
# out unrounded (``mm``'s ``out_dtype``, which torch implements only there and without
# a derivative). A separate fp32 GEMM over widened operands would be exact too, but its
# casts and launches cost the rollout's step 14 us, about 4% of an epoch; this one costs
# the bf16 GEMM's 10 us. Elsewhere the widened operands' GEMM is the same sum.
class _WideLinear(torch.autograd.Function):
    """``features @ weight.T`` over low-precision operands, summed and output in ``dtype``."""

    @classmethod
    @override
    def forward(
        cls,
        ctx: _WideLinearContext,
        /,
        features: Tensor,
        weight: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        ctx.save_for_backward(features, weight)
        if features.is_cuda:
            return torch.mm(features, weight.T, out_dtype=dtype)
        return functional.linear(features.to(dtype), weight.to(dtype))

    # Torch declares ``backward`` a staticmethod, and an override must stay one.
    backward = staticmethod(_wide_linear_backward)
