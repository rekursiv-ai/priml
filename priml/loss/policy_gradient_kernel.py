"""The clipped policy-gradient rule as three fused Triton kernels.

:class:`TritonPPO` implements :class:`~priml.loss.policy_gradient.PPO`.
Per minibatch of ``[rows, horizon]`` transitions, one row per lane:

1. the scoring kernel reads the fused ``[logits, value]`` row once and writes
   the masked log-softmax, the sampled action's log-probability and the live
   value;
2. the advantage kernel walks
   :func:`~priml.math.advantage.observation_aligned_advantage` backwards
   along each row in registers, eight steps per load and store;
3. the loss kernel writes both gradients and each block's eight partial loss
   sums, which are then added over the blocks.

The kernels use Triton's standard arithmetic: its ``exp`` and ``log``, a
block's ``sum``, and its default contraction of a multiply into the add that
follows. A subclass reproduces another implementation's bits by replacing
device functions by name (:attr:`TritonPPO.helpers`), the launch options and
the sum over blocks (:meth:`TritonPPO.sum_blocks`).

:class:`TritonPPO` refuses tensors off a CUDA device rather than running
:class:`~priml.loss.policy_gradient.TorchPPO` there: that rule's precise
``exp`` and ``log`` land a few ulp away, so a silent fallback would give one
config different bits on different hosts. A CPU run selects ``TorchPPO``.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``cache_imp_and_v``,
        ``puff_advantage``, ``ppo_loss_compute`` and ``ppo_loss_reduce`` in
        ``src/algo.cu``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, override

from configgle import Makes
from torch import Tensor

import torch

from priml.kernel import jit_kernel, require_power_of_two
from priml.loss.policy_gradient import LogProbs, Loss, TorchPPO


if TYPE_CHECKING:
    from collections.abc import Callable

    from triton import language

    import triton
else:
    from wrapt import lazy_import

    triton = lazy_import("triton")
    language = lazy_import("triton.language")


class TritonPPO(TorchPPO):
    """The rule's fused kernels, on a CUDA device only.

    It takes :class:`TorchPPO`'s coefficients and the scoring and loss
    kernels' launch geometry: every stage is its own kernel.
    """

    class Config(Makes["TritonPPO"], TorchPPO.Config):
        """The coefficients and the launch geometry; the arithmetic is fixed."""

        ADVANTAGE_WIDTH: ClassVar[int] = 8
        """Steps the advantage kernel reads and writes per row at a time, one
        16-byte vector in bf16 (two in fp32), so a horizon must be a multiple of
        it. The kernel's column split asserts it, so it is a class constant, not
        a knob."""

        block: int = 256
        """Rows per program of the scoring and loss kernels; the loss kernel
        writes one row of partial sums per block."""

        num_warps: int = 8
        """Warps per program: one row per thread at the default block."""

    helpers: ClassVar[dict[str, Callable[..., object]]] = {}
    """Device functions the kernels call, by name, in place of this module's
    own (:func:`add_log_triton`, ``_store_block_sums_triton``); empty for
    Triton's standard arithmetic."""

    launch_options: ClassVar[dict[str, bool]] = {}
    """Keywords every launch adds to Triton's defaults."""

    def __init__(self, config: Config) -> None:
        """Keep the coefficients and the launch geometry.

        Args:
          config: The coefficients and the geometry.

        """
        super().__init__(config)
        require_power_of_two(block=config.block, num_warps=config.num_warps)
        self.block = config.block
        self.num_warps = config.num_warps

    @override
    def check_horizon(self, horizon: int) -> None:
        """Refuse a horizon the advantage kernel cannot tile.

        Args:
          horizon: Steps per rollout.

        Raises:
          ValueError: ``horizon`` is not a positive multiple of
            :attr:`Config.ADVANTAGE_WIDTH`.

        """
        width = TritonPPO.Config.ADVANTAGE_WIDTH
        if horizon <= 0 or horizon % width:
            msg = (
                f"horizon must be a positive multiple of {width}, the "
                f"advantage kernel's width, not {horizon}"
            )
            raise ValueError(msg)

    @override
    def log_probs(
        self,
        decoded: Tensor,
        actions: Tensor,
        action_mask: Tensor,
    ) -> LogProbs:
        """Score the sampled actions under the live policy.

        Args:
          decoded: ``[rows, horizon, num_actions + 1]`` bf16, logits then
            value.
          actions: ``[rows, horizon]`` fp32, the sampled action ids.
          action_mask: ``[rows, horizon, num_actions]`` bf16, zero where
            illegal.

        Returns:
          result: The live values and log-probabilities.

        """
        _require_cuda(decoded)
        rows, horizon, fused = decoded.shape
        num_actions = fused - 1
        count = rows * horizon
        values = torch.empty(rows, horizon, dtype=decoded.dtype, device=decoded.device)
        logps = torch.empty(
            rows,
            horizon,
            num_actions,
            dtype=torch.float32,
            device=decoded.device,
        )
        new_lp = torch.empty(rows, horizon, dtype=torch.float32, device=decoded.device)
        _kernels(**self.helpers).cache[(triton.cdiv(count, self.block),)](
            (
                decoded.contiguous(),
                actions.contiguous(),
                action_mask.contiguous(),
                values,
                logps,
                new_lp,
            ),
            count,
            num_actions=num_actions,
            block=self.block,
            num_warps=self.num_warps,
            **self.launch_options,
        )
        return LogProbs(values=values, logps=logps, new_lp=new_lp)

    @override
    def advantage(
        self,
        values: Tensor,
        rewards: Tensor,
        terminals: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Estimate every row's advantages, newest step first.

        Args:
          values: The live values, ``[rows, horizon]``.
          rewards: ``[rows, horizon]``, already clamped; step ``t``'s
            arrived with observation ``t``.
          terminals: ``[rows, horizon]``, likewise.

        Returns:
          advantages: ``[rows, horizon]`` in the dtype ``values`` and
            ``rewards`` promote to, zero at the last step.
          returns: ``values + advantages``, likewise.

        """
        _require_cuda(values)
        rows, horizon = values.shape
        self.check_horizon(horizon)
        dtype = torch.promote_types(values.dtype, rewards.dtype)
        advantages = torch.empty_like(values, dtype=dtype)
        returns = torch.empty_like(values, dtype=dtype)
        _kernels(**self.helpers).advantage[(triton.cdiv(rows, 64),)](
            (values.contiguous(), rewards.contiguous(), terminals.contiguous()),
            advantages,
            returns,
            rows,
            horizon,
            self.discount,
            self.trace_decay,
            block=64,
            width=TritonPPO.Config.ADVANTAGE_WIDTH,
            num_warps=2,
            **self.launch_options,
        )
        return advantages, returns

    @override
    def loss(
        self,
        logprobs: LogProbs,
        *,
        decoded: Tensor,
        actions: Tensor,
        old_logprobs: Tensor,
        advantages: Tensor,
        values: Tensor,
        returns: Tensor,
    ) -> Loss:
        """Evaluate the clipped objective and its gradients.

        Args:
          logprobs: What :meth:`log_probs` returned.
          decoded: As passed to :meth:`log_probs`.
          actions: As passed to :meth:`log_probs`.
          old_logprobs: ``[rows, horizon]``, from the rollout.
          advantages: From :meth:`advantage`.
          values: The ROLLOUT's values, ``[rows, horizon]`` -- the
            value clip's anchor.
          returns: From :meth:`advantage`.

        Returns:
          result: The gradients and the summed loss terms.

        """
        _require_cuda(decoded)
        rows, horizon = actions.shape
        num_actions = logprobs.logps.shape[-1]
        count = rows * horizon
        blocks = triton.cdiv(count, self.block)
        grad_logits = torch.empty_like(logprobs.logps)
        grad_values = torch.empty_like(logprobs.new_lp)
        partials = torch.empty(
            blocks,
            len(TorchPPO.Config.LOSS_NAMES),
            dtype=torch.float32,
            device=decoded.device,
        )
        _kernels(**self.helpers).loss[(blocks,)](
            (
                logprobs.logps.contiguous(),
                logprobs.new_lp.contiguous(),
                decoded.contiguous(),
                actions.contiguous(),
                old_logprobs.contiguous(),
                advantages.contiguous(),
                values.contiguous(),
                returns.contiguous(),
            ),
            (grad_logits, grad_values, partials),
            count,
            self.clip_epsilon,
            self.value_clip_epsilon,
            self.value_coefficient,
            self.entropy_coefficient,
            num_actions=num_actions,
            block=self.block,
            num_warps=self.num_warps,
            **self.launch_options,
        )
        return Loss(
            grad_logits=grad_logits,
            grad_values=grad_values,
            losses=self.sum_blocks(partials),
        )

    def sum_blocks(self, partials: Tensor) -> Tensor:
        """Add the loss kernel's per-block sums over the blocks.

        Args:
          partials: ``[blocks, 8]`` fp32, one row per :attr:`block` rows.

        Returns:
          losses: ``[8]`` fp32; a subclass may add in a fixed order.

        """
        return partials.sum(0)


def add_log_triton(offset: language.tensor, x: language.tensor) -> language.tensor:
    """Return ``offset + log(x)``, the logsumexp's last step.

    :attr:`TritonPPO.helpers` replaces it by name for another implementation's
    rounding of the two.

    Args:
      offset: fp32, the running maximum.
      x: fp32, positive: the rescaled sum.

    Returns:
      result: fp32.

    """
    return offset + language.log(x)


def masked_logsumexp_triton(
    logits_ptr: language.tensor,
    mask_ptr: language.tensor,
    lanes: language.tensor,
    num_actions: language.constexpr,
) -> language.tensor:
    """Compute each lane's masked logsumexp in one pass over its row.

    The running sum is rescaled whenever the maximum moves. An illegal logit
    is ``-1e4``. The last step is :func:`add_log_triton`, which a caller binds.

    Args:
      logits_ptr: Each lane's first logit; the row is contiguous.
      mask_ptr: Each lane's first mask value, nonzero where legal.
      lanes: Which lanes are live.
      num_actions: Logits per row.

    Returns:
      logsumexp: One fp32 per lane.

    """
    maximum = language.full(lanes.shape, float("-inf"), dtype=language.float32)
    total = language.zeros(lanes.shape, dtype=language.float32)
    for action in language.static_range(num_actions):
        logit = language.load(logits_ptr + action, mask=lanes, other=0.0)
        legal = language.load(mask_ptr + action, mask=lanes, other=0.0)
        logit = language.where(
            legal.to(language.float32) == 0.0,
            -1e4,
            logit.to(language.float32),
        )
        bigger = logit > maximum
        total = language.where(bigger, total * language.exp(maximum - logit), total)
        maximum = language.where(bigger, logit, maximum)
        total = total + language.exp(logit - maximum)
    return add_log_triton(maximum, total)


def _require_cuda(tensor: Tensor) -> None:
    """Refuse a tensor off a CUDA device: the kernels have no bit-equal fallback."""
    if not tensor.is_cuda:
        msg = (
            f"TritonPPO runs on a CUDA device, not {tensor.device}; select "
            "TorchPPO to run the rule elsewhere"
        )
        raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Kernels:
    cache: triton.JITFunction[..., object]
    advantage: triton.JITFunction[..., object]
    loss: triton.JITFunction[..., object]


@lru_cache(maxsize=2)
def _kernels(**helpers: Callable[..., object]) -> _Kernels:
    """Jit the kernels once per helper set, on first use; importing needs no Triton."""
    device = {
        "add_log_triton": add_log_triton,
        "_store_block_sums_triton": _store_block_sums_triton,
    } | helpers
    logsumexp = jit_kernel(
        masked_logsumexp_triton,
        add_log_triton=jit_kernel(device["add_log_triton"]),
    )
    return _Kernels(
        cache=jit_kernel(_log_probs_triton, masked_logsumexp_triton=logsumexp),
        advantage=jit_kernel(
            _advantage_triton,
            _load_tile_triton=jit_kernel(_load_tile_triton),
            _walk_segment_triton=jit_kernel(
                _walk_segment_triton,
                _columns_triton=jit_kernel(_columns_triton),
            ),
        ),
        loss=jit_kernel(
            _loss_triton,
            _store_block_sums_triton=jit_kernel(device["_store_block_sums_triton"]),
        ),
    )


def _log_probs_triton(
    buffers: language.tuple,
    count: int,
    num_actions: language.constexpr,
    block: language.constexpr,
) -> None:
    """Score one row per lane: log-softmax, new log-probability, value."""
    (
        decoded_ptr,
        actions_ptr,
        mask_ptr,
        values_ptr,
        logps_ptr,
        new_lp_ptr,
    ) = buffers
    idx = language.program_id(0) * block + language.arange(0, block)
    mask = idx < count
    fused = idx * (num_actions + 1)
    flat = idx * num_actions
    language.store(
        values_ptr + idx,
        language.load(decoded_ptr + fused + num_actions, mask=mask),
        mask=mask,
    )
    # The one-pass logsumexp: rescale the running sum whenever the maximum moves.
    lse = masked_logsumexp_triton(
        decoded_ptr + fused,
        mask_ptr + flat,
        mask,
        num_actions,
    )
    act = language.load(actions_ptr + idx, mask=mask, other=0.0).to(language.int32)
    new_lp = language.zeros([block], dtype=language.float32)
    for action in language.static_range(num_actions):
        logit = language.load(decoded_ptr + fused + action, mask=mask, other=0.0)
        legal = language.load(mask_ptr + flat + action, mask=mask, other=0.0)
        logit = language.where(
            legal.to(language.float32) == 0.0,
            -1e4,
            logit.to(language.float32),
        )
        logp = logit - lse
        language.store(logps_ptr + flat + action, logp, mask=mask)
        new_lp = language.where(act == action, logp, new_lp)
    language.store(new_lp_ptr + idx, new_lp, mask=mask)


# Each row is read and written ``width`` steps at a time from the end and walked
# backwards in registers; a load and two stores per step, strided by a row between
# lanes, left the walk waiting on memory.
def _advantage_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    buffers: language.tuple,
    advantages_ptr: language.tensor,
    returns_ptr: language.tensor,
    rows: int,
    horizon: int,
    gamma: float,
    trace_decay: float,
    block: language.constexpr,
    width: language.constexpr,
) -> None:
    """Walk :func:`observation_aligned_advantage` along each lane's row."""
    values_ptr, rewards_ptr, terminals_ptr = buffers
    row = language.program_id(0) * block + language.arange(0, block)
    mask = row < rows
    last = row * horizon + horizon - 1
    next_v = language.load(values_ptr + last, mask=mask, other=0.0).to(language.float32)
    next_d = language.load(terminals_ptr + last, mask=mask, other=0.0).to(
        language.float32,
    )
    next_r = language.load(rewards_ptr + last, mask=mask, other=0.0).to(
        language.float32,
    )
    decay = gamma * trace_decay
    lastlam = language.zeros([block], dtype=language.float32)
    column = language.arange(0, width)[None, :]
    segments = horizon // width
    # The newest segment starts one step in: ``A[horizon - 1]`` is left at
    # zero and its return is ``V``.
    advantage = language.zeros([block, width], dtype=language.float32)
    returned = language.where(column == width - 1, next_v[:, None], advantage)
    first = width - 2
    for index in range(segments):
        base = row * horizon + (segments - 1 - index) * width
        v = _load_tile_triton(values_ptr, base, mask, width)
        d = _load_tile_triton(terminals_ptr, base, mask, width)
        r = _load_tile_triton(rewards_ptr, base, mask, width)
        lastlam, next_v, next_d, next_r, advantage, returned = _walk_segment_triton(
            v,
            d,
            r,
            lastlam,
            next_v,
            next_d,
            next_r,
            advantage,
            returned,
            first,
            gamma,
            decay,
            block,
            width,
        )
        # Each store rounds to its tensor's dtype, the values' and rewards' promoted.
        cells = base[:, None] + column
        language.store(advantages_ptr + cells, advantage, mask=mask[:, None])
        language.store(returns_ptr + cells, returned, mask=mask[:, None])
        first = width - 1


def _walk_segment_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    v_tile: language.tensor,
    d_tile: language.tensor,
    r_tile: language.tensor,
    lastlam: language.tensor,
    next_v: language.tensor,
    next_d: language.tensor,
    next_r: language.tensor,
    advantage: language.tensor,
    returned: language.tensor,
    first: int,
    gamma: float,
    decay: float,
    rows: language.constexpr,
    width: language.constexpr,
) -> tuple[language.tensor, ...]:
    """Walk one segment's steps from column ``first`` down to 0."""
    language.static_assert(width == 8, "the columns are split for a width of 8")
    v = _columns_triton(v_tile, rows)
    d = _columns_triton(d_tile, rows)
    r = _columns_triton(r_tile, rows)
    lane = language.arange(0, width)[None, :]
    for column in language.static_range(width - 1, -1, -1):
        if column <= first:
            continuing = 1.0 - next_d
            delta = language.fma(gamma * next_v, continuing, next_r) - v[column]
            lastlam = delta + (decay * lastlam) * continuing
            advantage = language.where(lane == column, lastlam[:, None], advantage)
            returned = language.where(
                lane == column,
                (v[column] + lastlam)[:, None],
                returned,
            )
            next_v = v[column]
            next_d = d[column]
            next_r = r[column]
    return lastlam, next_v, next_d, next_r, advantage, returned


def _load_tile_triton(
    pointer: language.tensor,
    base: language.tensor,
    mask: language.tensor,
    width: language.constexpr,
) -> language.tensor:
    """Load ``width`` consecutive steps per row as fp32, ``[rows, width]``."""
    return language.load(
        pointer + base[:, None] + language.arange(0, width)[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(language.float32)


def _columns_triton(
    tile: language.tensor,
    rows: language.constexpr,
) -> tuple[language.tensor, ...]:
    """Return a ``[rows, 8]`` tile's columns in order."""
    # ``split`` halves the last axis: each reshape pairs columns ``k`` and
    # ``k + 4``, then ``k`` and ``k + 2``, then neighbours.
    even, odd = language.split(language.reshape(tile, rows, 4, 2))
    first, third = language.split(language.reshape(even, rows, 2, 2))
    second, fourth = language.split(language.reshape(odd, rows, 2, 2))
    c0, c4 = language.split(first)
    c2, c6 = language.split(third)
    c1, c5 = language.split(second)
    c3, c7 = language.split(fourth)
    return c0, c1, c2, c3, c4, c5, c6, c7


def _loss_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    inputs: language.tuple,
    outputs: language.tuple,
    count: int,
    clip_coef: float,
    vf_clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    num_actions: language.constexpr,
    block: language.constexpr,
) -> None:
    """Write one row per lane's gradients, and the block's eight loss sums."""
    (
        logps_ptr,
        new_lp_ptr,
        decoded_ptr,
        actions_ptr,
        old_ptr,
        adv_ptr,
        val_ptr,
        ret_ptr,
    ) = inputs
    grad_logits_ptr, grad_values_ptr, partials_ptr = outputs
    pid = language.program_id(0)
    idx = pid * block + language.arange(0, block)
    mask = idx < count
    flat = idx * num_actions
    inv_nt = language.div_rn(1.0, count * 1.0)
    adv = language.load(adv_ptr + idx, mask=mask, other=0.0).to(language.float32)
    val = language.load(val_ptr + idx, mask=mask, other=0.0).to(language.float32)
    ret = language.load(ret_ptr + idx, mask=mask, other=0.0).to(language.float32)
    val_pred = language.load(
        decoded_ptr + idx * (num_actions + 1) + num_actions,
        mask=mask,
        other=0.0,
    )
    val_pred = val_pred.to(language.float32)
    d_entropy_term = inv_nt * (-ent_coef)
    new_lp = language.load(new_lp_ptr + idx, mask=mask, other=0.0)
    old = language.load(old_ptr + idx, mask=mask, other=0.0).to(language.float32)
    logratio = new_lp - old
    ratio = language.exp(logratio)

    v_error = val_pred - val
    v_clipped = val + language.maximum(
        -vf_clip_coef,
        language.minimum(vf_clip_coef, v_error),
    )
    v_loss_unclipped = (val_pred - ret) * (val_pred - ret)
    v_loss_clipped = (v_clipped - ret) * (v_clipped - ret)
    v_loss = 0.5 * language.maximum(v_loss_unclipped, v_loss_clipped)
    d_val_pred = language.where(v_loss_clipped > v_loss_unclipped, 0.0, val_pred - ret)
    language.store(grad_values_ptr + idx, (inv_nt * vf_coef) * d_val_pred, mask=mask)

    clip_lo = 1.0 - clip_coef
    clip_hi = 1.0 + clip_coef
    ratio_clipped = language.maximum(clip_lo, language.minimum(clip_hi, ratio))
    wa = -adv
    pg_loss1 = wa * ratio
    pg_loss2 = wa * ratio_clipped
    pg_loss = language.maximum(pg_loss1, pg_loss2)
    clipped = (pg_loss2 > pg_loss1) & ((ratio <= clip_lo) | (ratio >= clip_hi))
    d_ratio = language.where(clipped, 0.0, wa * inv_nt)
    d_new_logp = d_ratio * ratio

    act = language.load(actions_ptr + idx, mask=mask, other=0.0).to(language.int32)
    entropy = language.zeros([block], dtype=language.float32)
    for action in language.static_range(num_actions):
        logp = language.load(logps_ptr + flat + action, mask=mask, other=0.0)
        entropy = language.fma(-language.exp(logp), logp, entropy)
    for action in language.static_range(num_actions):
        logp = language.load(logps_ptr + flat + action, mask=mask, other=0.0)
        p = language.exp(logp)
        indicator = language.where(act == action, 1.0, 0.0)
        grad = language.fma(
            indicator - p,
            d_new_logp,
            (d_entropy_term * p) * (-entropy - logp),
        )
        language.store(grad_logits_ptr + flat + action, grad, mask=mask)

    # Both products fuse into their adds under any launch options.
    thread_loss = language.fma(
        -ent_coef,
        entropy,
        language.fma(vf_coef, v_loss, pg_loss),
    )
    thread_loss = thread_loss * inv_nt
    zero = language.zeros([block], dtype=language.float32)
    _store_block_sums_triton(
        partials_ptr + pid * 8,
        language.where(mask, pg_loss * inv_nt, zero),
        language.where(mask, v_loss * inv_nt, zero),
        language.where(mask, entropy * inv_nt, zero),
        language.where(mask, thread_loss, zero),
        language.where(mask, (-logratio) * inv_nt, zero),
        language.where(mask, ((ratio - 1.0) - logratio) * inv_nt, zero),
        language.where(
            mask,
            language.where(language.abs(ratio - 1.0) > clip_coef, 1.0, 0.0) * inv_nt,
            zero,
        ),
        language.where(mask, ratio * inv_nt, zero),
    )


# :attr:`TritonPPO.helpers` replaces this by name to sum the lanes in a fixed tree.
def _store_block_sums_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    partials_ptr: language.tensor,
    policy: language.tensor,
    value: language.tensor,
    entropy: language.tensor,
    total: language.tensor,
    old_approx_kl: language.tensor,
    approx_kl: language.tensor,
    clipfrac: language.tensor,
    importance: language.tensor,
) -> None:
    """Store the eight terms, each summed over the lanes, in ``LOSS_NAMES`` order."""
    terms = (
        policy,
        value,
        entropy,
        total,
        old_approx_kl,
        approx_kl,
        clipfrac,
        importance,
    )
    for index in language.static_range(8):
        language.store(partials_ptr + index, language.sum(terms[index], axis=0))
