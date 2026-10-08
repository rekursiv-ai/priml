"""PufferLib's kernel arithmetic, swapped into the kernel classes for exp000.

priml's kernels and the sampler compile with Triton's defaults and use its
standard math: ``exp``, ``log``, ``sigmoid``, ``/``, ``a + w * (b - a)``, a
sum's own reduction order, and a multiply contracted into the add that
follows. PufferLib's CUDA kernels, compiled by nvcc with ``--fmad=false``,
round differently, so each class here is a kernel class with PufferLib's
arithmetic in place of the default. exp000 selects them, which keeps its run
bit for bit PufferLib's; a run from its own init has no reason to.

- :class:`ExactScan`: the sigmoid with the precise ``expf`` and ``div_rn``,
  ``lerp`` as one fma on either branch, and the highway output's ``s * h``
  fused into its add.
- :class:`ExactPPO`: ``__logf`` as ``lg2.approx`` with its ln 2 scaling fused
  into the logsumexp's add, and the loss terms summed over a halving tree of
  each block's 256 rows, then over the blocks in order.
- :class:`ExactPhiloxSampler`: the logsumexp as :class:`ExactPPO`'s, and the
  inverse CDF adding the precise ``expf`` as nvcc inlined it, its last
  multiply fused into the running sum.
- :class:`ExactMuon`: every norm over one fixed reduction tree
  (:func:`sum_squares`), and ``div_rn`` in the normalization.

Every class launches its kernels with :data:`LAUNCH_OPTIONS`. Triton's fp32
``exp`` already lowers to ``__expf``'s instruction and constant, so the fast
exponential needs no replacement.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``src/algo.cu`` and ``src/pufferl.cu``,
        pin ``6ffa5b10``.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING, ClassVar, Final, override

from configgle import Makes

from priml.baselines.craftax.rollout import PhiloxSampler
from priml.kernel import jit_kernel
from priml.loss.policy_gradient import TorchPPO
from priml.loss.policy_gradient_kernel import TritonPPO
from priml.model.min_gru import TritonScan
from priml.optimizers.fused_muon import FusedMuon


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor
    from triton import language
    from triton.language.extra.cuda import libdevice

    import torch
    import triton
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")
    triton = lazy_import("triton")
    language = lazy_import("triton.language")
    libdevice = lazy_import("triton.language.extra.cuda.libdevice")


LAUNCH_OPTIONS: Final = {"enable_fp_fusion": False, "enable_reflect_ftz": False}
"""Launch keywords that compile a kernel as ``nvcc --fmad=false`` compiles C.

Every product or sum not written as ``fma`` rounds on its own, and libdevice
keeps to its paths without flush-to-zero (nvcc's default ``-ftz=false``).
Triton's defaults contract a multiply into the add that follows, which moves
the last bit."""


# The device functions come before the classes whose ``helpers`` bind them. Each is
# a leaf: a kernel class jits and binds it by name, which reaches no callee of its own.
def sigmoid_rn_triton(x: language.tensor) -> language.tensor:
    """Return the overflow-safe sigmoid with libdevice's ``expf`` and ``div_rn``.

    Args:
      x: fp32.

    Returns:
      sigmoid: fp32.

    """
    e = libdevice.exp(-language.abs(x))
    return language.where(
        x >= 0,
        language.div_rn(1.0, 1.0 + e),
        language.div_rn(e, 1.0 + e),
    )


def lerp_fma_triton(
    a: language.tensor,
    b: language.tensor,
    w: language.tensor,
) -> language.tensor:
    """Return torch's CUDA ``lerp`` as nvcc compiles it: one fma on either branch.

    Args:
      a: fp32, the start.
      b: fp32, the end.
      w: fp32, the weight.

    Returns:
      lerp: fp32, ``a + w * (b - a)``.

    """
    diff = b - a
    return language.where(
        language.abs(w) < 0.5,
        language.fma(w, diff, a),
        language.fma(-diff, 1.0 - w, b),
    )


def highway_fma_triton(
    s: language.tensor,
    h: language.tensor,
    x: language.tensor,
) -> language.tensor:
    """Return ``s * h + (1 - s) * x`` with ``(1 - s) * x`` rounded and ``s * h`` fused."""
    return language.fma(s, h, (1.0 - s) * x)


def add_log_fma_triton(offset: language.tensor, x: language.tensor) -> language.tensor:
    """Return ``offset + __logf(x)``: ``lg2.approx`` with ln 2 fused into the add.

    Args:
      offset: fp32, the running maximum.
      x: fp32, positive: the rescaled sum.

    Returns:
      result: fp32.

    """
    log2 = language.inline_asm_elementwise(
        "lg2.approx.f32 $0, $1;",
        "=f,f",
        [x],
        language.float32,
        True,
        1,
    )
    return language.fma(log2, 0.6931471824645996, offset)


# One halving per level over a ``[rows, 8]`` tile adds each term's lanes in exactly
# the pairs its own tree would: a two-row ``sum`` is one addition per lane, whatever
# Triton's layout, where a ``sum`` over all the rows would reassociate.
def halving_block_sums_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    partials_ptr: language.tensor,
    c0: language.tensor,
    c1: language.tensor,
    c2: language.tensor,
    c3: language.tensor,
    c4: language.tensor,
    c5: language.tensor,
    c6: language.tensor,
    c7: language.tensor,
) -> None:
    """Store the eight terms, each summed over the block's 256 lanes by halving.

    Args:
      partials_ptr: Where the block's eight sums go, in order.
      c0: The first term, ``[256]`` fp32, zero on dead lanes.
      c1: The second term, likewise.
      c2: The third term, likewise.
      c3: The fourth term, likewise.
      c4: The fifth term, likewise.
      c5: The sixth term, likewise.
      c6: The seventh term, likewise.
      c7: The eighth term, likewise.

    """
    rows: language.constexpr = c0.shape[0]
    language.static_assert(rows == 256, "the tree halves 256 lanes")
    # Each join pairs columns ``k`` and ``k + 4``, then ``k`` and ``k + 2``, then
    # neighbours: the reshapes lay the eight terms out as columns in order.
    first = language.join(c0, c4)
    third = language.join(c2, c6)
    second = language.join(c1, c5)
    fourth = language.join(c3, c7)
    even = language.reshape(language.join(first, third), rows, 4)
    odd = language.reshape(language.join(second, fourth), rows, 4)
    tile = language.reshape(language.join(even, odd), rows, 8)
    for level in language.static_range(8):
        tile = language.sum(language.reshape(tile, 2, rows >> (level + 1), 8), axis=0)
    language.store(partials_ptr + language.arange(0, 8), language.reshape(tile, 8))


# ``expf(a) == e * scale``, with ``e`` the ``2^fraction`` of ``ex2.approx.ftz.f32`` and
# ``scale`` the ``2^integer`` as float bits. The sampler fuses that product with its
# running sum (``sample_logits`` in ``puffer.ptx``: ``fma.rn.f32 cumsum, e, scale,
# cumsum``). The constants are the PTX's.
def accumulate_expf_triton(
    total: language.tensor,
    a: language.tensor,
) -> language.tensor:
    """Return ``total + expf(a)``, the precise ``expf`` as nvcc inlines it.

    Args:
      total: fp32, the running sum.
      a: fp32, the exponent.

    Returns:
      total: fp32, with the product's rounding fused into the add.

    """
    t = language.clamp(language.fma(a, 0.005724980030208826, 0.5), 0.0, 1.0)
    j = language.inline_asm_elementwise(
        "fma.rm.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        [t, 252.0, 12582913.0],
        language.float32,
        True,
        1,
    )
    remainder = -(j + -12583039.0)
    fraction = language.fma(a, 1.4426950216293334961, remainder)
    fraction = language.fma(a, 1.925963033500011079e-08, fraction)
    scale = (j.to(language.int32, bitcast=True) << 23).to(
        language.float32,
        bitcast=True,
    )
    e = language.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        [fraction],
        language.float32,
        True,
        1,
    )
    return language.fma(e, scale, total)


def reciprocal_rn_triton(x: language.tensor) -> language.tensor:
    """Return ``1 / x`` correctly rounded, as nvcc's default ``-prec-div`` divides."""
    return language.div_rn(1.0, x)


class ExactScan(TritonScan):
    """TritonScan with the arithmetic of PufferLib's per-element scan kernels.

    PufferLib's ``_row`` kernels, which it selects at small shapes, fuse the
    highway the other way and are not transcribed.
    """

    class Config(Makes["ExactScan"], TritonScan.Config):
        """Launch geometry; the arithmetic is PufferLib's."""

    helpers: ClassVar[dict[str, Callable[..., object]]] = {
        "_sigmoid_triton": sigmoid_rn_triton,
        "_lerp_triton": lerp_fma_triton,
        "_highway_triton": highway_fma_triton,
    }
    launch_options: ClassVar[dict[str, bool]] = LAUNCH_OPTIONS


class ExactPPO(TritonPPO):
    """TritonPPO with the arithmetic and the loss reduction of PufferLib's kernels."""

    class Config(TritonPPO.Config):
        """The coefficients; the arithmetic is PufferLib's."""

    helpers: ClassVar[dict[str, Callable[..., object]]] = {
        "add_log_triton": add_log_fma_triton,
        "_store_block_sums_triton": halving_block_sums_triton,
    }
    launch_options: ClassVar[dict[str, bool]] = LAUNCH_OPTIONS

    @override
    def sum_blocks(self, partials: Tensor) -> Tensor:
        """Add the per-block sums in block order, as ``ppo_loss_reduce`` does.

        Args:
          partials: ``[blocks, 8]`` fp32.

        Returns:
          losses: ``[8]`` fp32.

        """
        losses = torch.empty(
            len(TorchPPO.Config.LOSS_NAMES),
            dtype=torch.float32,
            device=partials.device,
        )
        _ordered_sum_kernel()[(1,)](
            partials,
            losses,
            partials.shape[0],
            num_losses=len(TorchPPO.Config.LOSS_NAMES),
            num_warps=1,
            **LAUNCH_OPTIONS,
        )
        return losses


class ExactPhiloxSampler(PhiloxSampler):
    """PhiloxSampler with the arithmetic of PufferLib's ``sample_logits``."""

    class Config(PhiloxSampler.Config):
        """The stream seed and the launch geometry; the arithmetic is PufferLib's."""

    helpers: ClassVar[dict[str, Callable[..., object]]] = {
        "add_log_triton": add_log_fma_triton,
        "_accumulate_exp_triton": accumulate_expf_triton,
    }
    launch_options: ClassVar[dict[str, bool]] = LAUNCH_OPTIONS


class ExactMuon(FusedMuon):
    """FusedMuon with the reduction order and the arithmetic of PufferLib's ``muon_step``.

    Every norm is :func:`sum_squares`'s fixed tree over the tensors
    concatenated in order, then an IEEE square root, so the order the
    parameters arrive in is part of the result. The normalization divides with
    ``div_rn``.
    """

    class Config(Makes["Callable[..., ExactMuon]"], FusedMuon.Config):
        """FusedMuon's hyperparameters; the arithmetic is PufferLib's."""

        @override
        def make(self) -> Callable[..., ExactMuon]:
            """Return a constructor awaiting the parameters to optimize."""
            constructor = super().make()
            assert isinstance(constructor, partial)
            return partial(ExactMuon, **constructor.keywords)

    helpers: ClassVar[dict[str, Callable[..., object]]] = {
        "_reciprocal_triton": reciprocal_rn_triton,
    }
    launch_options: ClassVar[dict[str, bool]] = LAUNCH_OPTIONS

    @override
    def norm(self, tensors: Sequence[Tensor]) -> Tensor:
        """Return the norm over :func:`sum_squares`'s tree of the concatenation.

        Args:
          tensors: Any shapes and float dtypes, on one device, in the order
            the tree reads them.

        Returns:
          norm: A 0-dim fp32 tensor on their device.

        """
        values = (
            tensors[0]
            if len(tensors) == 1
            else torch.cat([tensor.reshape(-1) for tensor in tensors])
        )
        return sum_squares(values).sqrt()


def sum_squares(values: Tensor) -> Tensor:
    """Sum the squares of a flat vector in a fixed reduction tree.

    Block ``b`` of at most 256, lane ``t`` of 256, accumulates the elements
    ``b * 256 + t + k * stride`` for ``k = 0, 1, ...`` in order as an fma
    chain; each block's 256 partials then halve down to one, and the block
    totals, padded with zeros to 256, halve down to the sum.

    On CUDA this is two Triton launches of that shape. Elsewhere the same tree
    runs as torch ops, one ``addcmul_`` per grid stride, which is the reference
    the kernels are held to.

    Args:
      values: Any shape and float dtype; read flat, in fp32.

    Returns:
      total: A 0-dim fp32 tensor.

    """
    if values.is_cuda:
        return _sum_squares_cuda(values)
    values = values.reshape(-1).float()
    blocks = min((values.numel() + 255) // 256, 256)
    stride = blocks * 256
    padding = -values.numel() % stride
    stripes = torch.cat((values, values.new_zeros(padding))).reshape(-1, blocks, 256)
    partials = values.new_zeros((blocks, 256))
    for stripe in stripes.unbind():
        partials.addcmul_(stripe, stripe)
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        partials = partials[:, :width] + partials[:, width : 2 * width]
    totals = torch.cat((partials[:, 0], values.new_zeros(256 - blocks)))
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        totals = totals[:width] + totals[width : 2 * width]
    return totals[0]


def _sum_squares_cuda(values: Tensor, *, block: int = 256) -> Tensor:
    """Run the two sum-of-squares kernels; ``values`` may be any float dtype."""
    flat = values.reshape(-1)
    count = flat.numel()
    blocks = min((count + block - 1) // block, block)
    stride = blocks * block
    partials = torch.empty(blocks, dtype=torch.float32, device=flat.device)
    total = torch.empty(1, dtype=torch.float32, device=flat.device)
    kernels = _sum_squares_kernels()
    kernels.partials[(blocks,)](
        flat,
        partials,
        count,
        stride,
        (count + stride - 1) // stride,
        block=block,
        num_warps=block // 32,
        **LAUNCH_OPTIONS,
    )
    kernels.total[(1,)](
        partials,
        total,
        blocks,
        block=block,
        num_warps=block // 32,
        **LAUNCH_OPTIONS,
    )
    return total[0]


@dataclass(frozen=True, slots=True, kw_only=True)
class _SumSquaresKernels:
    partials: triton.JITFunction[..., object]
    total: triton.JITFunction[..., object]


@lru_cache(maxsize=1)
def _sum_squares_kernels() -> _SumSquaresKernels:
    """Jit the kernels once, on first use, so importing needs no Triton."""
    tree = jit_kernel(_halving_tree_triton)
    return _SumSquaresKernels(
        partials=jit_kernel(_sum_squares_partials_triton, _halving_tree_triton=tree),
        total=jit_kernel(_sum_squares_total_triton, _halving_tree_triton=tree),
    )


@lru_cache(maxsize=1)
def _ordered_sum_kernel() -> triton.JITFunction[..., object]:
    """Jit the block-order sum once, on first use."""
    return jit_kernel(_ordered_sum_triton)


# At each width from ``block / 2`` down to 1, lane ``t`` becomes ``s[t] + s[t +
# width]``: a ``[2, width]`` view summed over its first axis is one fadd per pair, the
# addition the tree makes at that level, where a ``sum`` over the whole vector would
# reassociate.
def _halving_tree_triton(
    values: language.tensor,
    block: language.constexpr,
) -> language.tensor:
    """Halve ``values`` down to one element, one pairwise add per level."""
    language.static_assert(block == 256, "both sum-of-squares kernels run 256 lanes")
    for level in language.static_range(8):
        values = language.sum(
            language.reshape(values, 2, block // (2 << level)),
            axis=0,
        )
    return values


def _sum_squares_partials_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    values_ptr: language.tensor,
    partials_ptr: language.tensor,
    count: int,
    stride: int,
    passes: int,
    block: language.constexpr,
) -> None:
    """Accumulate an fma chain per lane over its grid stride, then halve the block."""
    lane = language.arange(0, block)
    index = language.program_id(0) * block + lane
    total = language.zeros([block], dtype=language.float32)
    for _ in range(passes):
        value = language.load(values_ptr + index, mask=index < count, other=0.0).to(
            language.float32,
        )
        total = language.fma(value, value, total)
        index += stride
    language.store(
        partials_ptr + language.program_id(0) + language.arange(0, 1),
        _halving_tree_triton(total, block),
    )


def _sum_squares_total_triton(
    partials_ptr: language.tensor,
    total_ptr: language.tensor,
    blocks: int,
    block: language.constexpr,
) -> None:
    """Halve the block partials, zero past ``blocks``, down the same tree."""
    lane = language.arange(0, block)
    values = language.load(partials_ptr + lane, mask=lane < blocks, other=0.0)
    language.store(
        total_ptr + language.arange(0, 1),
        _halving_tree_triton(values, block),
    )


def _ordered_sum_triton(
    partials_ptr: language.tensor,
    losses_ptr: language.tensor,
    blocks: int,
    num_losses: language.constexpr,
) -> None:
    """Add the blocks' loss sums in block order."""
    lane = language.arange(0, num_losses)
    total = language.zeros([num_losses], dtype=language.float32)
    for block in range(blocks):
        total = total + language.load(partials_ptr + block * num_losses + lane)
    language.store(losses_ptr + lane, total)
