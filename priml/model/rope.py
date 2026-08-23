"""Rotary Position Embeddings.

Ported from ~/projects/sic/code/model.py (RoPE, RoPEMixed).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any, Literal, Self, cast, override

import math

from configgle import Fig, Makes
from torch import Tensor, nn

import torch

from priml.math.basic import broadcast_sequences, floor_multiple


@dataclass(frozen=True, slots=True, kw_only=True)
class YarnScaling:
    """YaRN (Yet another RoPE extensioN) scaling parameters.

    NTK-aware frequency interpolation + log-scaled attention
    correction. Reference: Peng et al., 2023
    (https://arxiv.org/abs/2309.00071). Convention matches HuggingFace
    ``rope_scaling={"type": "yarn", ...}`` on DeepSeek-V3 / Kimi-K2::

        {"factor": 32.0, "original_max_position_embeddings": 4096,
         "beta_fast": 1.0, "beta_slow": 1.0,
         "mscale": 1.0, "mscale_all_dim": 1.0}

    Given base frequencies ``theta_i = base ** (2 i / d)``, rotations
    per token at frequency i are ``L_orig / (2 pi theta_i)``. YaRN
    ramps between pure NTK interpolation (low freq,
    ``r <= beta_slow``) and pure extrapolation (high freq,
    ``r >= beta_fast``), dividing the inverse frequency by the scale
    factor on the interpolated side.

    The ``mscale`` correction scales cos/sin by ``m`` where
    ``m = (0.1 * log(factor) + 1) * mscale * mscale_all_dim``. Baking
    it into the rotation tables is equivalent to multiplying
    ``softmax_scale`` by ``m**2``.
    """

    factor: float
    """Context-length extension factor (new_max / original_max)."""

    original_max_position_embeddings: int
    """Pretraining context length used to define "base" positions."""

    beta_fast: float = 32.0
    """High-frequency cutoff (rotations/token) -- no interpolation above."""

    beta_slow: float = 1.0
    """Low-frequency cutoff -- full interpolation below."""

    mscale: float = 1.0
    """Attention-scale correction factor."""

    mscale_all_dim: float = 0.0
    """Second factor combined with ``mscale``. Kimi-K2 / DSV3 set both to 1.0."""


class RoPE(nn.Module):
    """N-dimensional Rotary Position Embedding with fixed frequencies.

    Number of axes inferred from ``dim``: scalar for 1D, iterable for N-D.
    Per-axis channel counts can be unequal (e.g. ``dim=[44, 42, 42]``
    allocates 44 frequencies to the first axis and 42 each to the other
    two). Each nonzero count must be even and >= 2; use 0 to skip an
    axis (e.g. ``dim=[128, 0]`` encodes only the first axis).

    Returns half-dim cos/sin (shape ``[..., S, 1, dim//2]``). Pair
    with ``RoPE.rotate`` which splits q/k, applies the 2D rotation, and
    recombines -- no replication needed. The ``interleave`` flag on
    ``RoPE.rotate`` selects the dimension pairing convention:
    GPT-NeoX / HuggingFace half-split (LLaMA, Mistral, Gemma, Qwen)
    vs RoFormer / Meta LLaMA interleave. See ``RoPE.rotate`` docstring
    for details.

    ``smallest_recommended_base(dim, max_positions)`` reference::

                  len      c=16    c=32    c=64   c=128   c=256
            ----------------------------------------------------
                   16       3       3       2       2       2
                   32       6       5       5       5       5
                   64      14      12      11      10      10
                  128      31      25      22      21      21
                  256      69      52      46      43      42
                  512     152     109      94      87      84
                1,024     337     229     192     177     169
                2,048     745     479     393     357     341
                4,096   1,645   1,004     803     722     686
                8,192   3,632   2,103   1,643   1,461   1,379
               16,384   8,021   4,405   3,361   2,954   2,774
               32,768  17,713   9,227   6,873   5,974   5,579
               65,536  39,114  19,328  14,058  12,080  11,218
              131,072  86,372  40,484  28,751  24,428  22,560

    Args:
      dim: Channel count. Scalar + multi-element ``base`` auto-splits:
          cat mode splits evenly (e.g. 128 across 3 axes -> [44, 42, 42]),
          sum mode replicates (e.g. 64 across 2 axes -> [64, 64]).
          Iterable for explicit per-axis allocation. Each nonzero value
          must be even and >= 2. Use 0 to skip an axis.
      base: Frequency base(s). Scalar (shared across axes) or iterable
          (one per axis). Controls the longest wavelength: the lowest
          frequency has period ``2*pi*base^((c-2)/c)`` where c is the
          per-axis channel count. Use ``smallest_recommended_base`` to
          compute from max position counts.
      reduction_mode: "cat" (axial; default) or "sum" (RoPE-Mixed).

    """

    class Config(Fig["RoPE"], kw_only=False):
        channels_head: int | Iterable[int] = 64
        """Channel count per axis (scalar auto-splits across axes)."""

        _: KW_ONLY

        base: float | Iterable[float] = 10e3
        """Frequency base(s) controlling longest wavelength."""

        reduction_mode: Literal["cat", "sum"] = "cat"
        """How to combine axes: "cat" (axial) or "sum" (RoPE-Mixed)."""

        yarn: YarnScaling | None = None
        """Optional YaRN scaling (NTK-aware interp + mscale correction).

        Applied per-axis when set. For N-axis RoPE, the same config is
        applied to each axis (typical for single-axis LMs anyway).
        """

        hf_inv_freq: bool = False
        """Use the HuggingFace inv_freq formula for bit-for-bit parity.

        Default (False): ``exp(linspace(0, log(b)*(-1+2/c), c//2))`` in
        float64 (higher precision). When True:
        ``1 / (base ** (arange(0, c, 2, int64).float() / c))`` — matches
        HF transformers exactly.
        """

        dtype: torch.dtype | None = None
        """Width the cos/sin factors are rounded to; None keeps float32.

        Not a memory choice -- the table is two vectors -- but an arithmetic
        one: rounding the factors makes every product inside the rotation
        accumulate at that width, where float32 factors promote the whole
        rotation and round once at the end. The two differ in the last bits of
        every query and key, so a port has to hold the factors at the width the
        reference held them.
        """

    def __init__(self, config: Config) -> None:
        super().__init__()
        c = config.channels_head
        base = config.base
        reduction_mode = config.reduction_mode
        self.reduction_mode = reduction_mode
        if (
            reduction_mode == "cat"
            and isinstance(c, int)
            and isinstance(base, Sequence)
            and len(base) > 1
        ):
            c = self._split_dim(c, len(base))
        c, base = broadcast_sequences(c, base)
        self.channels_head = tuple(int(d) for d in c)
        self.base = tuple(float(b) for b in base)
        self._hf_inv_freq = config.hf_inv_freq
        self._yarn = config.yarn
        # YaRN mscale applied as a scalar multiplier on cos/sin outputs.
        # When YaRN is off, this stays 1.0 and the math is a no-op.
        self._mscale = 1.0
        self._inv_freqs: list[Tensor] = []
        self._build_inv_freqs(torch.device("cpu"))
        if all(not f.numel() for f in self._inv_freqs):
            raise ValueError(
                f"At least one dim must be nonzero, got {self.channels_head}.",
            )
        self._dtype = nn.Buffer(
            torch.empty(0, dtype=config.dtype),
            persistent=False,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype.dtype

    @property
    def device(self) -> torch.device:
        return self._dtype.device

    @override
    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Encode positions to half-dim (cos, sin) embeddings.

        Args:
          positions: [..., num_axes].

        Returns:
          cos: [..., H, dim // 2].
          sin: [..., H, dim // 2].

        """
        if positions.ndim == 1:
            positions = positions.unsqueeze(-1)
        pos = positions.to(dtype=torch.float32).split(1, dim=-1)
        frequencies = self._inv_freqs
        parts = [
            p.unsqueeze(-1) * f.to(dtype=torch.float32, device=p.device)
            for p, f in zip(pos, frequencies, strict=True)
            if f.numel()
        ]
        if self.reduction_mode == "cat":
            emb = torch.cat(parts, dim=-1)
        elif self.reduction_mode == "sum":
            emb = cast(Tensor, sum(parts))
        else:
            raise ValueError(f"Unknown {self.reduction_mode=}.")
        cos = emb.cos() * self._mscale
        sin = emb.sin() * self._mscale
        return (
            cos.to(dtype=self.dtype, device=self.device),
            sin.to(dtype=self.dtype, device=self.device),
        )

    @classmethod
    def rotate(
        cls,
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
        interleave: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Apply rotary position embedding to query and key tensors.

        cos/sin are half-dim (D//2). Splits q/k into pairs, applies a 2D
        rotation, and recombines. No replication needed -- half the memory
        vs full-dim cos/sin approaches.

        The ``interleave`` flag selects the dimension pairing convention:

        - ``False`` (default): GPT-NeoX / HuggingFace half-split. Pairs
          dim i with dim i+D/2. This is the convention in all HuggingFace
          Transformers models (LLaMA, Mistral, Gemma, Qwen, etc.). Default
          because most pretrained checkpoints use HF.
        - ``True``: RoFormer / Meta LLaMA interleave. Pairs dim 2i with
          dim 2i+1 (consecutive). This matches the original paper's math
          and Meta's official LLaMA code.

        Both conventions are mathematically equivalent up to a permutation
        of embedding dimensions. HuggingFace applies a weight permutation
        during checkpoint conversion (``convert_llama_weights_to_hf.py``)
        to reconcile the two. For bit-for-bit reproduction, match the
        convention used during training.

        When q/k have more sequence positions than cos/sin (e.g. from
        padding), cos/sin are right-padded with identity values (cos=1,
        sin=0) so extra positions are unrotated.

        When cos/sin cover fewer channels than q/k (i.e.
        ``cos.shape[-1] * 2 < D``), only the first ``cos.shape[-1] * 2``
        channels are rotated; the remaining channels pass through unchanged.

        Args:
          q: [..., S, H, D].
          k: [..., S, H, D].
          cos: [..., S', H, D_rot//2] where H=1 (fixed) or heads
              (learnable). S' <= S; right-padded to S if smaller.
              D_rot <= D; when D_rot < D only the first D_rot channels
              are rotated.
          sin: Same shape as cos.
          interleave: Pairing convention (see above).

        Returns:
          q_embed: Rotated queries, same shape as q.
          k_embed: Rotated keys, same shape as k.

        """
        seq_dim = -3
        seq_len = q.shape[seq_dim]
        rope_len = cos.shape[seq_dim]
        if rope_len < seq_len:
            pad_shape = list(cos.shape)
            pad_shape[seq_dim] = seq_len - rope_len
            cos = torch.cat(
                [cos, torch.ones(pad_shape, dtype=cos.dtype, device=cos.device)],
                dim=seq_dim,
            )
            sin = torch.cat(
                [sin, torch.zeros(pad_shape, dtype=sin.dtype, device=sin.device)],
                dim=seq_dim,
            )
        rot_dim = cos.shape[-1] * 2
        D = q.shape[-1]
        if rot_dim < D:
            q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
            k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
            q_rot = cls._rotate(q_rot, cos, sin, interleave)
            k_rot = cls._rotate(k_rot, cos, sin, interleave)
            return (
                torch.cat([q_rot, q_pass], dim=-1),
                torch.cat([k_rot, k_pass], dim=-1),
            )
        return (
            cls._rotate(q, cos, sin, interleave),
            cls._rotate(k, cos, sin, interleave),
        )

    @classmethod
    def smallest_recommended_base(
        cls,
        dim: int | Iterable[int],
        max_positions: int | Iterable[int],
    ) -> float | tuple[float, ...]:
        """Return the smallest reasonable base for given position range(s).

        The lowest frequency has period 2pi*base^((c-2)/c). Setting this
        >= max_positions gives::

          base = ((max_pos - 1) / (2pi))^(c / (c - 2))

        Args:
          dim: Per-axis channel count(s) (same as __init__).
          max_positions: Max position count per axis. Scalar or iterable.
              Broadcast against dim.

        Returns:
          base: Float (1D) or tuple of floats (N-D).

        """
        dims, maxpos = broadcast_sequences(dim, max_positions)
        bases: list[float] = []
        for c, m in zip(dims, maxpos, strict=True):
            if c == 0:
                bases.append(0.0)
                continue
            cls._validate_c(c)
            bases.append(((m - 1) / (2 * math.pi)) ** (c / (c - 2)))
        return bases[0] if len(bases) == 1 else tuple(bases)

    @classmethod
    def _rotate(cls, x: Tensor, cos: Tensor, sin: Tensor, interleave: bool) -> Tensor:
        """Apply 2D rotation to a single tensor using half-dim cos/sin.

        Builds the output OUT-OF-PLACE (no ``torch.empty_like`` + strided
        in-place writes). The earlier in-place form
        (``out = empty_like(x); out[..., 0::2] = ...``) hung a compiled,
        ``torch.inference_mode`` eval on CUDA: a strided in-place write into
        uninitialized ``empty_like`` memory under ``torch.compile`` +
        ``inference_mode`` produced a wedged kernel (observed: an eval-only resume
        froze mid-pass, GPUs pinned, the next launch stuck at the eager cos/sin
        build). The out-of-place ``stack``/``cat`` assembly is numerically
        bit-identical (guarded by ``test_rotate_matches_stack_reference_*``) and
        still avoids the input ``aten.reshape`` that torchao fp8 axiswise scaling
        cannot trace -- the reason the reshape+stack original was replaced.
        """
        dtype = x.dtype
        x = x.float()
        if interleave:
            # Consecutive pairs (2i, 2i+1). Write the rotated even/odd lanes into a
            # ZEROS-initialized output (not ``empty_like``): the interleaved 0::2 /
            # 1::2 strides do not concatenate cleanly, so the output is assembled
            # by strided assignment -- but into initialized memory, so even if a
            # cell were ever missed it is a defined 0, not the uninitialized
            # ``empty_like`` garbage that wedged the compiled CUDA eval. No
            # ``torch.stack`` (the fp8-trace guard) and no input ``reshape``.
            x0 = x[..., 0::2]
            x1 = x[..., 1::2]
            out = torch.zeros_like(x)
            out[..., 0::2] = x0 * cos - x1 * sin
            out[..., 1::2] = x1 * cos + x0 * sin
        else:
            # Half-split (i, i+D/2) -- the production convention (HuggingFace; every
            # call site uses the default ``interleave=False``). The two rotated
            # halves CONCATENATE to the full output out-of-place: no uninitialized
            # memory, no strided in-place write, no ``stack``, no ``reshape``.
            half = x.shape[-1] // 2
            x0 = x[..., :half]
            x1 = x[..., half:]
            out = torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)
        return out.to(dtype)

    @override
    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> Self:
        super()._apply(fn, recurse)
        self._build_inv_freqs(self._dtype.device)
        return self

    def _build_inv_freqs(self, device: torch.device) -> None:
        """Build the inverse frequencies on ``device``.

        Rebuilt on a move rather than copied there: ``base ** x`` is a
        transcendental whose last bit differs between CPU and CUDA (measured, 4
        of 64 frequencies at head_dim 128), so a reference that constructs its
        table on the accelerator cannot be matched by moving a CPU-built one.
        """
        self._inv_freqs = [torch.empty(0, device=device) for _ in self.channels_head]
        for i, (b, c) in enumerate(zip(self.base, self.channels_head, strict=True)):
            if c == 0:
                continue
            self._validate_c(c)
            inv_freq = self._make_inv_freqs(b, c, hf=self._hf_inv_freq, device=device)
            if self._yarn is not None:
                inv_freq, self._mscale = _yarn_apply(inv_freq, b, c, self._yarn)
            self._inv_freqs[i] = inv_freq.unsqueeze(-2)

    @classmethod
    def _make_inv_freqs(
        cls,
        b: float,
        c: int,
        *,
        hf: bool = False,
        device: torch.device | None = None,
    ) -> Tensor:
        """Inverse frequencies: b^linspace(0, -1+2/c, c//2)."""
        if hf:
            channels = torch.arange(0, c, 2, dtype=torch.int64, device=device).float()
            return 1.0 / (b ** (channels / c))
        return (
            torch.linspace(
                0,
                math.log(b) * (-1 + 2 / c),
                c // 2,
                dtype=torch.float64,
                device=device,
            )
            .exp()
            .float()
        )

    @classmethod
    def _validate_c(cls, c: int) -> None:
        """Validate a single per-axis channel count."""
        if c < 2:
            raise ValueError(f"Dim {c} must be at least 2.")
        if c % 2 == 1:
            raise ValueError(f"Dim {c} must be even.")

    @classmethod
    def _split_dim(cls, total: int, naxes: int) -> list[int]:
        """Split total channels across axes, each even, front-loaded.

        E.g. _split_dim(128, 3) -> [44, 42, 42].
        """
        if total % 2:
            raise ValueError(f"Total dim={total} must be even.")
        if total < 2 * naxes:
            raise ValueError(
                f"Cannot split dim={total} across {naxes} axes"
                f" (need at least {2 * naxes}).",
            )
        per = int(floor_multiple(total // naxes, 2))
        remainder = total - per * naxes
        dims = [per] * naxes
        for i in range(remainder // 2):
            dims[i] += 2
        return dims


def rotate_conjugate(x: Tensor, *, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x``'s channel pairs by ``-theta``, at the input's own precision.

    Two deliberate differences from :meth:`RoPE.rotate`, both numerics rather
    than taste:

    * **Direction.** ``RoPE.rotate`` is HuggingFace's ``+theta``; this is its
      conjugate, with the sine entering the second half negated. The two are
      the same model under a channel permutation and DIFFERENT tensors, so a
      port has to pick the one its weights were trained under.
    * **Precision.** ``RoPE.rotate`` upcasts to float32, accumulates there, and
      rounds once. This accumulates at whatever width the factors and the input
      arrive in -- half, under autocast -- which is what a fused kernel does,
      and differs from the upcast form in the last bits.

    Args:
      x: ``[..., S, heads, channels_head]`` queries or keys.
      cos: Cosine factors, broadcastable over ``x``'s first half.
      sin: Sine factors, same shape as ``cos``.

    Returns:
      rotated: ``x`` with each ``(i, i + half)`` channel pair rotated.

    """
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat(
        [first * cos + second * sin, first * (-sin) + second * cos],
        dim=-1,
    )


def _yarn_mscale(scale: float, mscale: float) -> float:
    """YaRN attention scale: m = 0.1 * ln(factor) * mscale + 1 (for factor > 1)."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * math.log(scale) * mscale + 1.0


def _yarn_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position: int,
) -> tuple[float, float]:
    """Return the per-channel indices marking the YaRN ramp region."""

    def _find_correction_dim(
        num_rot: float,
        dim: int,
        base: float,
        max_position: int,
    ) -> float:
        return (
            dim
            * math.log(max_position / (num_rot * 2 * math.pi))
            / (2 * math.log(base))
        )

    low = math.floor(_find_correction_dim(low_rot, dim, base, max_position))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_position))
    return max(low, 0), min(high, dim - 1)


def _yarn_apply(
    inv_freq: Tensor,
    base: float,
    dim: int,
    yarn: YarnScaling,
) -> tuple[Tensor, float]:
    """Apply YaRN frequency scaling and return (scaled_inv_freq, mscale).

    Matches HF's DeepSeek-V3 convention. Low-rotation (low-frequency,
    high index) channels get linear interpolation (``inv_freq /
    factor``); high-rotation (high-frequency, low index) channels stay
    on their original base (extrapolation). A linear ramp on channel
    indices between ``low`` and ``high`` blends the two.
    """
    low_idx, high_idx = _yarn_correction_range(
        yarn.beta_fast,
        yarn.beta_slow,
        dim,
        base,
        yarn.original_max_position_embeddings,
    )
    low_f = float(low_idx)
    high_f = float(high_idx)
    if high_f == low_f:
        # beta_fast == beta_slow collapses the ramp to a step function.
        high_f = low_f + 1e-3
    ramp = torch.arange(dim // 2, dtype=torch.float32, device=inv_freq.device)
    ramp = ((ramp - low_f) / (high_f - low_f)).clamp(0.0, 1.0)
    # HF's ``inv_freq_mask = 1 - ramp``; so at low index (high freq,
    # ramp=0, mask=1) we use original (extrapolation), and at high
    # index (low freq, ramp=1, mask=0) we use ``inv_freq / factor``
    # (interpolation). inv_freq = extra * mask + inter * (1 - mask).
    inv_freq_mask = 1.0 - ramp
    inv_freq_interp = inv_freq / yarn.factor
    inv_freq_yarn = inv_freq * inv_freq_mask + inv_freq_interp * (1.0 - inv_freq_mask)
    # DSV3 mscale: if mscale_all_dim is 0, the correction is just
    # ``0.1 log(factor) * mscale + 1``. If nonzero, DSV3 uses the
    # ratio of two mscale formulas, one per ``mscale`` and one per
    # ``mscale_all_dim``. Kimi-K2 sets both to 1.0 → ratio = 1.0.
    if yarn.mscale_all_dim != 0:
        m = _yarn_mscale(yarn.factor, yarn.mscale) / _yarn_mscale(
            yarn.factor,
            yarn.mscale_all_dim,
        )
    else:
        m = _yarn_mscale(yarn.factor, yarn.mscale)
    return inv_freq_yarn, m


class RoPEMixed(RoPE):
    """N-dimensional RoPE with per-head frequencies.

    Extends ``RoPE`` with per-head frequency scaling via random directions on
    the N-sphere.

    When ``reduction_mode="cat"`` (default), per-axis channels stay separate
    (learned axial RoPE). When ``reduction_mode="sum"``, all axes share
    channels (summed) -- this and ``learnable=True`` is RoPE-Mixed from [Heo et
    al., ECCV 2024](https://arxiv.org/abs/2403.13298).

    Returns half-dim cos/sin (shape ``[..., S, H, dim//2]``). Pair with
    ``RoPE.rotate``.

    Args:
      dim: Channel count per axis (same semantics as ``RoPE``).
      heads: Number of attention heads.
      base: Frequency base(s) (same semantics as ``RoPE``).
      reduction_mode: "cat" (axial; default) or "sum" (RoPE-Mixed).
      learnable: If True, frequencies are learnable. Default False.

    """

    class Config(Makes["RoPEMixed"], RoPE.Config):
        heads: int = 1
        """Number of attention heads for per-head frequency scaling."""

        learnable: bool = False
        """Whether per-head frequencies are learnable parameters."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.heads = config.heads
        self.learnable = config.learnable
        if config.reduction_mode == "sum":
            active_dims = {f.shape[-1] for f in self._inv_freqs if f.numel()}
            if len(active_dims) > 1:
                raise ValueError(f"Sum mode requires uniform dims, got {active_dims}.")
        # Capture the base (deterministic) per-axis frequencies; reset_parameters
        # is the sole source of the per-head scaling, so it can rebuild the
        # learnable _inv_freqs from these without re-deriving them. The scaled
        # shape is the base broadcast against a per-head column, computed once
        # here so the empty parameters are allocated at the right shape.
        self._base_inv_freqs: list[Tensor] = [
            f.detach().clone() for f in self._inv_freqs
        ]
        head_col = torch.empty(self.heads, 1) if self.heads > 1 else None
        self._inv_freqs: nn.ParameterList = nn.ParameterList(  # pyright: ignore[reportIncompatibleVariableOverride]
            nn.Parameter(
                torch.empty_like(f * head_col if head_col is not None else f),
                requires_grad=self.learnable and f.numel() > 0,
            )
            if f.numel()
            else nn.Parameter(torch.empty(0), requires_grad=False)
            for f in self._base_inv_freqs
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Sole source of the per-head frequency init: draw fresh random
        # directions on the N-sphere and scale the base frequencies by them.
        directions = None
        active_count = sum(1 for f in self._base_inv_freqs if f.numel())
        if self.heads > 1:
            directions = nn.functional.normalize(
                nn.init.trunc_normal_(
                    torch.empty(active_count, self.heads, 1),
                    std=1.0,
                ),
                dim=0,
            )
        with torch.no_grad():
            j = 0
            for param, base in zip(self._inv_freqs, self._base_inv_freqs, strict=True):
                if base.numel():
                    scaled = base * directions[j] if directions is not None else base
                    param.copy_(scaled.to(param))
                    j += 1

    @override
    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> Self:
        freqs = [f.data.clone() for f in self._inv_freqs]
        nn.Module._apply(self, fn, recurse)  # noqa: SLF001
        for i, f in enumerate(freqs):
            self._inv_freqs[i].data = f.to(device=self._dtype.device)
        return self
