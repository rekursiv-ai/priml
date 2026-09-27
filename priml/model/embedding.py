"""Embedding layers.

:class:`MultiHotEmbedding`'s Triton kernels and its fixed-point gradient
transcribe PufferLib's embedding bag.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``ocean/craftax/craftax.cu``, pin
        ``6ffa5b10``.

"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Protocol, override

import itertools

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import (
    Cost,
    resolve_dtype,
)
from priml.kernel import jit_kernel
from priml.model.custom_types import DepthIndex, ShardStyle
from priml.model.init import InitFn, call_init, normal, truncated_normal


if TYPE_CHECKING:
    from triton import language
    from triton.language.extra.cuda import libdevice

    import triton
else:
    from wrapt import lazy_import

    triton = lazy_import("triton")
    language = lazy_import("triton.language")
    libdevice = lazy_import("triton.language.extra.cuda.libdevice")


class Embedding(nn.Embedding):
    """Embedding with truncated normal init."""

    class Config(Fig["Embedding"], kw_only=False):
        channels_in: int = -1
        """Vocabulary size: the input token range the table indexes."""

        channels_out: int = -1
        """Dimensionality of each embedding vector."""

        _: KW_ONLY

        padding_idx: int | None = None
        """Index whose embedding is zeroed out (e.g. for padding tokens)."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

        shard: ShardStyle | None = None
        """Tensor-parallel shard style over the mesh tp dim; ``None`` replicates."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init (-1 = no scaling).

        Present, and forwarded, for the same reason ``Linear`` and ``Conv``
        carry one: every initializer in :mod:`priml.model.init` divides by
        ``sqrt(depth + 1)`` and DEFAULTS that depth to 1, so a table that never
        states one is drawn at 0.707 of the spread it asked for. A lookup table
        has no residual branch to scale down, hence -1 rather than a depth."""

        init_weight: InitFn = partial(truncated_normal, std=0.02)
        """Draws the table.

        A slot rather than a fixed rule, because the right spread is a property
        of what READS the table: one feeding an RMS norm has its scale divided
        out and wants unit variance, while one summed into a residual stream
        does not."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost a gather and its dense scatter-add adjoint.

            Every gradient element is added to a zero-initialized table, so
            repeated indices do not change the count. Padding rows skip it, so
            this is an upper bound when padding is present. Include the index,
            row read/write, and dense gradient-table zeroing shared over rows.
            The index is one ``int64`` per pass; the table and its gradient are
            at this layer's dtype.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete invocation.

            """
            del kwargs
            rows = seq_len * batch_size
            dt = self.dtype if self.dtype is not None else resolve_dtype(dtype)
            index = torch.int64
            row = self.channels_out
            return Cost(
                cells={
                    ("flops", "adjoint", "selection", dt): rows * row,
                    ("bytes", "primal", "selection", index): rows * index.itemsize,
                    ("bytes", "primal", "selection", dt): dt.itemsize * 2 * rows * row,
                    ("bytes", "adjoint", "selection", index): rows * index.itemsize,
                    ("bytes", "adjoint", "selection", dt): dt.itemsize
                    * (3 * rows * row + self.channels_in * row),
                },
                params=self.channels_in * row,
                params_active=row,
            )

    def __init__(self, config: Config) -> None:
        self.shard = config.shard
        self.depth_index = config.depth_index
        self._init_weight = config.init_weight
        super().__init__(
            num_embeddings=config.channels_in,
            embedding_dim=config.channels_out,
            padding_idx=config.padding_idx,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def reset_parameters(self) -> None:
        # Depth is PASSED, as every other parameterized module here passes it.
        # Omitting it does not mean "no scaling": it takes the initializer's own
        # default of 1, which divides by sqrt(2) -- a table 0.707 as wide as the
        # one requested, invisible to every shape, name, and dtype check.
        call_init(self._init_weight, self.weight, depth_index=self.depth_index)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class MultiHotEmbedding(nn.Module):
    """Embed rows of multi-hot cells: sum each cell's rows, then append the scalars.

    A row packs ``num_cells`` cells of ``len(offsets)`` categorical ids, then
    ``num_scalars`` continuous values. Every field has its own vocabulary: the
    rows of one shared table from its offset on. A cell's embedding is the sum
    of its fields' rows, added in fp32 in field order and rounded once to the
    table's dtype; the scalars pass through. So a cell is a multi-hot vector
    over the table, one hot per field, and its embedding is that vector times
    the table.

    A module of its own rather than an ``nn.Embedding``: it reads packed rows,
    not token ids, so code that special-cases embeddings (tensor-parallel
    sharding, for one) must not take it for one. Autograd reaches the table
    through :meth:`backward`, which is exact: each contribution is rounded to a
    multiple of ``2^-24`` and summed as an integer, so no scatter order can
    move a bit.
    """

    class Config(Fig["MultiHotEmbedding"], kw_only=False):
        """The table, and the packed row's layout."""

        channels_in: int = -1
        """Rows: every field's vocabulary laid end to end."""

        channels_out: int = -1
        """Width of one embedding, and of one summed cell."""

        _: KW_ONLY

        offsets: tuple[int, ...] = (0,)
        """Each field's first row in the table; the tuple's length is the
        field count."""

        num_cells: int = 1
        """Cells per row, each ``len(offsets)`` ids wide."""

        num_scalars: int = 0
        """Continuous values after the cells, copied through unchanged."""

        dtype: torch.dtype | None = None
        """The table's dtype; ``None`` is torch's default unless a parent sets it."""

        init_weight: InitFn = partial(normal, std=1.0)
        """Draws the table; N(0, 1), as ``nn.Embedding`` draws its own."""

        @property
        def observation_size(self) -> int:
            """Width of one packed row: the cells' ids, then the scalars."""
            return self.num_cells * len(self.offsets) + self.num_scalars

        @property
        def channels_concat(self) -> int:
            """Width of the features: every cell's embedding, then the scalars."""
            return self.num_cells * self.channels_out + self.num_scalars

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the cells' gathers and sums and their scatter-add adjoint.

            Each cell gathers one table row per field and sums them; the
            scalars are copied. The adjoint scatter-adds each cell's gradient
            into every field's row of a zeroed table gradient. Ids are read in
            the row's dtype, as the packed row carries them.

            Args:
              seq_len: Rows per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; unused.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete invocation.

            """
            del kwargs
            rows = seq_len * batch_size
            dt = self.dtype if self.dtype is not None else resolve_dtype(dtype)
            fields = len(self.offsets)
            gathered = rows * self.num_cells * fields * self.channels_out
            summed = rows * self.num_cells * self.channels_out
            ids = rows * self.num_cells * fields
            scalars = rows * self.num_scalars
            table = self.channels_in * self.channels_out
            return Cost(
                cells={
                    ("flops", "primal", "elementwise", dt): gathered - summed,
                    ("bytes", "primal", "selection", dt): dt.itemsize
                    * (ids + gathered + 2 * scalars + summed),
                    ("flops", "adjoint", "selection", dt): gathered,
                    ("bytes", "adjoint", "selection", dt): dt.itemsize
                    * (ids + summed + 2 * gathered + table),
                },
            )

    def __init__(self, config: Config) -> None:
        """Allocate the table and draw it.

        Args:
          config: The table's geometry, dtype and initializer.

        """
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.channels_in, config.channels_out, dtype=config.dtype),
        )
        self._init_weight = config.init_weight
        self.num_cells = config.num_cells
        self.num_scalars = config.num_scalars
        self.field_offsets = config.offsets
        """The offsets on the host: reading the device buffer would sync."""
        # Annotated as well as registered: ``register_buffer`` types its
        # result as ``Tensor | Module | None``.
        self.offsets: Tensor
        self.register_buffer("offsets", torch.tensor(config.offsets), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Draw the table with the config's initializer, undivided by any depth."""
        call_init(self._init_weight, self.weight, depth_index=())

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Embed one packed row per leading index.

        On a CUDA device with a bf16 table this is one Triton launch per batch;
        elsewhere the torch reference. Either way autograd differentiates the
        table through :meth:`backward`.

        Args:
          input: Packed rows, ``[..., observation_size]``, in any float dtype
            whose values are the cells' ids and the scalars.

        Returns:
          features: ``[..., channels_concat]`` in the table's dtype.

        """
        return _MultiHotLookup.apply(self, input, self.weight)

    def forward_torch(self, input: Tensor) -> Tensor:
        """Embed with torch ops, in field order: the reference, and the CPU path.

        Args:
          input: As for :meth:`forward`.

        Returns:
          features: As for :meth:`forward`.

        """
        cells = self.num_cells * self.offsets.shape[0]
        ids = input[..., :cells].reshape(*input.shape[:-1], self.num_cells, -1).long()
        rows = functional.embedding(ids + self.offsets, self.weight).float()
        # Sequential in field order: a reduction would add in another order.
        summed = rows[..., 0, :]
        for field in range(1, rows.shape[-2]):
            summed = summed + rows[..., field, :]
        dtype = self.weight.dtype
        return torch.cat(
            (summed.to(dtype).flatten(-2), input[..., cells:].to(dtype)),
            dim=-1,
        )

    def forward_triton(self, input: Tensor) -> Tensor:
        """Embed with one Triton launch per batch (CUDA, bf16 table).

        Args:
          input: As for :meth:`forward`, on the table's device.

        Returns:
          features: As for :meth:`forward`.

        Raises:
          TypeError: The table is not bf16.

        """
        if self.weight.dtype != torch.bfloat16:
            msg = f"the Triton embedding stores bf16, not {self.weight.dtype}"
            raise TypeError(msg)
        width = self.weight.shape[1]
        fields = self.offsets.shape[0]
        # The input's own dtype, as the backward and the torch path read the ids: a
        # cast to bf16 would round every id above 256 (fp32 holds them exactly up
        # to 2^24), and the forward would read another row than the backward writes.
        packed = input.reshape(-1, input.shape[-1]).contiguous()
        output = torch.empty(
            packed.shape[0],
            self.num_cells * width + self.num_scalars,
            dtype=self.weight.dtype,
            device=packed.device,
        )
        _embed_kernel()[(packed.shape[0],)](
            packed,
            self.weight,
            self.offsets,
            output,
            self.num_cells,
            self.num_scalars,
            num_fields=fields,
            width=width,
            width_block=_power_of_two(width),
            cell_block=_power_of_two(self.num_cells),
            scalar_block=_power_of_two(self.num_scalars),
            num_warps=4,
        )
        return output.reshape(*input.shape[:-1], output.shape[-1])

    def backward(self, input: Tensor, grad_output: Tensor) -> Tensor:
        """Return the table's gradient, accumulated in 2^-24 fixed point.

        Each contribution is rounded to a multiple of 2^-24, the integers are
        summed in int64 -- exact whatever the scatter order, and wrapping as
        unsigned atomics do -- and the sums are decoded through double and
        float before rounding to the table's dtype. On a CUDA device with a
        bf16 table in the layout its kernels unroll (eight fields, rows a power
        of two of at least 16 wide) this is :meth:`backward_triton`, two
        launches and no host read, so a captured graph can hold it; otherwise
        :meth:`backward_torch`, whose sums are the same integers.

        Args:
          input: The forward's packed rows.
          grad_output: Gradient of the forward's features; the scalar
            columns carry none.

        Returns:
          grad_weight: The table's, ``[channels_in, channels_out]`` in its dtype.

        """
        width = self.weight.shape[1]
        if (
            input.is_cuda
            and self.weight.dtype == torch.bfloat16
            and len(self.field_offsets) == 8
            and width >= 16
            and _power_of_two(width) == width
        ):
            return self.backward_triton(input, grad_output)
        return self.backward_torch(input, grad_output)

    def backward_torch(self, input: Tensor, grad_output: Tensor) -> Tensor:
        """Return the table's gradient with torch ops: the reference and the CPU path.

        Args:
          input: As for :meth:`backward`.
          grad_output: As for :meth:`backward`.

        Returns:
          grad_weight: ``[channels_in, channels_out]`` in the table's dtype.

        """
        fields = self.offsets.shape[0]
        width = self.weight.shape[1]
        ids = input[..., : self.num_cells * fields].reshape(-1, fields).long()
        fixed = (
            (grad_output[..., : self.num_cells * width].float() * 2.0**24)
            .round()
            .long()
            .reshape(-1, width)
        )
        total = torch.zeros(
            self.weight.shape,
            dtype=torch.int64,
            device=input.device,
        )
        for field in range(fields):
            total.index_add_(0, ids[:, field] + self.offsets[field], fixed)
        return (total.double() * 2.0**-24).float().to(self.weight.dtype)

    def backward_triton(self, input: Tensor, grad_output: Tensor) -> Tensor:
        """Return the table's gradient as exact int8 tensor-core products.

        Every program owns a share of the ``(row, cell)`` pairs. Per tile of
        pairs and per field, ``hits[value, pair]`` is 1 where the pair's id is
        ``value``, and each pair's fixed-point contributions are split into a
        balanced low digit in ``[-128, 127]`` and the rest; ``hits @ digit`` is
        an int8 GEMM whose int32 sums are exact. The rest, and any id outside
        its field's vocabulary (added to row ``offset + id`` wherever that
        is), are zero on a normal tile; a tile where they are not adds its
        higher digits' products, and those rows, into the program's int64
        partials directly. A second launch adds the programs' partials and
        decodes them. Integer sums modulo 2^64 in any order are
        :meth:`backward_torch`'s sums, so the result is its to the bit.

        Args:
          input: As for :meth:`backward`, on the table's device.
          grad_output: As for :meth:`backward`.

        Returns:
          grad_weight: ``[channels_in, channels_out]`` in the table's dtype.

        Raises:
          ValueError: The layout is not the eight fields the kernel unrolls,
            or a row is not a power of two of at least 16 lanes, as its int8
            products need.

        """
        rows, width = self.weight.shape
        starts = [*self.field_offsets, rows]
        sizes = [end - start for start, end in itertools.pairwise(starts)]
        if len(sizes) != 8:
            msg = f"the Triton embedding backward unrolls 8 fields, not {len(sizes)}"
            raise ValueError(msg)
        if width < 16 or _power_of_two(width) != width:
            msg = f"the Triton embedding backward needs 2^k >= 16 lanes, not {width}"
            raise ValueError(msg)
        observations = input.reshape(-1, input.shape[-1]).contiguous()
        grad = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        pairs = observations.shape[0] * self.num_cells
        tile = 128
        # A program's int32 low-digit sums stay exact below 2^31 / 128 pairs.
        programs = max(
            2 * torch.cuda.get_device_properties(grad.device).multi_processor_count,
            -(-pairs // (2**31 // 128)),
        )
        partials = torch.zeros(
            programs,
            rows * width,
            dtype=torch.int64,
            device=grad.device,
        )
        kernels = _backward_kernels()
        kernels.accumulate[(programs,)](
            (observations, grad, self.offsets, partials),
            pairs,
            -(-pairs // tile),
            observations.shape[1],
            grad.shape[1],
            rows,
            num_cells=self.num_cells,
            width=width,
            tile=tile,
            vocab_first=_power_of_two(max(sizes[0], 16)),
            vocab_rest=_power_of_two(max(*sizes[1:], 16)),
            num_warps=4,
        )
        output = torch.empty_like(self.weight)
        block = 128
        kernels.decode[(-(-output.numel() // block),)](
            partials,
            output,
            programs,
            output.numel(),
            partials.shape[1],
            program_block=64,
            block=block,
            num_warps=4,
        )
        return output


class _LookupContext(Protocol):
    """What :class:`_MultiHotLookup` keeps between its forward and its backward."""

    embedding: MultiHotEmbedding
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _lookup_backward(
    ctx: _LookupContext,
    /,
    *grad_outputs: Tensor,
) -> tuple[None, None, Tensor]:
    """Return the table's fixed-point gradient; the ids and the module take none."""
    (grad,) = grad_outputs
    (input,) = ctx.saved_tensors
    return None, None, ctx.embedding.backward(input, grad)


# The table enters as an argument, not only through the module, so that autograd
# records the lookup as a function of it; the packed ids carry no gradient.
class _MultiHotLookup(torch.autograd.Function):
    """The lookup as an autograd node whose backward is the fixed-point one."""

    @classmethod
    @override
    def forward(
        cls,
        ctx: _LookupContext,
        /,
        embedding: MultiHotEmbedding,
        input: Tensor,
        weight: Tensor,
    ) -> Tensor:
        ctx.embedding = embedding
        ctx.save_for_backward(input)
        if input.is_cuda and weight.dtype == torch.bfloat16:
            return embedding.forward_triton(input)
        return embedding.forward_torch(input)

    # Torch declares ``backward`` a staticmethod, and an override must stay one.
    backward = staticmethod(_lookup_backward)


def _power_of_two(n: int) -> int:
    """Return the smallest power of two that is at least ``n``."""
    return 1 << (n - 1).bit_length()


@lru_cache(maxsize=1)
def _embed_kernel() -> triton.JITFunction[..., object]:
    """Jit the embedding kernel once, on first use, so importing needs no Triton."""
    return jit_kernel(_embed_triton)


@dataclass(frozen=True, slots=True, kw_only=True)
class _BackwardKernels:
    accumulate: triton.JITFunction[..., object]
    decode: triton.JITFunction[..., object]


@lru_cache(maxsize=1)
def _backward_kernels() -> _BackwardKernels:
    """Jit the backward's kernels once, on first use."""
    rows = jit_kernel(_field_rows_triton)
    add_rows = jit_kernel(_add_rows_triton, _field_rows_triton=rows)
    hits_dot = jit_kernel(_hits_dot_triton)
    return _BackwardKernels(
        accumulate=jit_kernel(
            _embed_backward_triton,
            _hits_dot_triton=hits_dot,
            _add_rows_triton=add_rows,
            _spill_triton=jit_kernel(
                _spill_triton,
                _field_rows_triton=rows,
                _higher_rows_triton=jit_kernel(
                    _higher_rows_triton,
                    _hits_dot_triton=hits_dot,
                    _add_rows_triton=add_rows,
                ),
            ),
        ),
        decode=jit_kernel(_embed_decode_triton),
    )


# Each cell's table rows are added in field order from ``0.0``, in fp32, then rounded
# once to bf16; the scalars are copied through, rounded to bf16 by the store as
# torch's cast rounds them.
def _embed_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    input_ptr: language.tensor,
    table_ptr: language.tensor,
    offsets_ptr: language.tensor,
    output_ptr: language.tensor,
    num_cells: int,
    num_scalars: int,
    num_fields: language.constexpr,
    width: language.constexpr,
    width_block: language.constexpr,
    cell_block: language.constexpr,
    scalar_block: language.constexpr,
) -> None:
    """Embed one packed row per program: sum each cell's rows, copy the scalars."""
    row = language.program_id(0)
    observation = input_ptr + row * (num_cells * num_fields + num_scalars)
    features = output_ptr + row * (num_cells * width + num_scalars)
    cell = language.arange(0, cell_block)
    lane = language.arange(0, width_block)
    live = cell < num_cells
    lanes = live[:, None] & (lane < width)[None, :]
    total = language.zeros([cell_block, width_block], dtype=language.float32)
    for field in language.static_range(num_fields):
        ids = language.load(
            observation + cell * num_fields + field,
            mask=live,
            other=0.0,
        )
        rows = language.load(offsets_ptr + field) + ids.to(language.float32).to(
            language.int32,
        )
        entry = language.load(
            table_ptr + rows[:, None] * width + lane[None, :],
            mask=lanes,
            other=0.0,
        )
        total = total + entry.to(language.float32)
    language.store(
        features + cell[:, None] * width + lane[None, :],
        total.to(language.bfloat16),
        mask=lanes,
    )
    scalar = language.arange(0, scalar_block)
    live_scalar = scalar < num_scalars
    language.store(
        features + num_cells * width + scalar,
        language.load(
            observation + num_cells * num_fields + scalar,
            mask=live_scalar,
        ),
        mask=live_scalar,
    )


# A pair is one (observation row, cell). Per tile and per field, ``hits[value, pair]``
# is 1 where the pair's id is ``value``, and ``digit[pair, lane]`` is the balanced low
# digit of the lane's fixed-point gradient, so ``hits @ digit`` sums the field's rows
# exactly in int32. The eight fields are unrolled by hand: each keeps its own sums.
def _embed_backward_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    tensors: language.tuple,
    pairs: int,
    tiles: int,
    observation_size: int,
    feature_size: int,
    table_rows: int,
    num_cells: language.constexpr,
    width: language.constexpr,
    tile: language.constexpr,
    vocab_first: language.constexpr,
    vocab_rest: language.constexpr,
) -> None:
    """Sum the fixed-point contributions of one program's tiles as integers."""
    observations_ptr, grad_ptr, offsets_ptr, partials_ptr = tensors
    partials = partials_ptr + language.program_id(0) * (table_rows * width)
    lane = language.arange(0, width)
    sums0 = language.zeros([vocab_first, width], dtype=language.int32)
    sums1 = language.zeros([vocab_rest, width], dtype=language.int32)
    sums2 = language.zeros([vocab_rest, width], dtype=language.int32)
    sums3 = language.zeros([vocab_rest, width], dtype=language.int32)
    sums4 = language.zeros([vocab_rest, width], dtype=language.int32)
    sums5 = language.zeros([vocab_rest, width], dtype=language.int32)
    sums6 = language.zeros([vocab_rest, width], dtype=language.int32)
    sums7 = language.zeros([vocab_rest, width], dtype=language.int32)
    for index in range(language.program_id(0), tiles, language.num_programs(0)):
        pair = index * tile + language.arange(0, tile)
        live = pair < pairs
        row = pair // num_cells
        cell = pair - row * num_cells
        ids = observations_ptr + row * observation_size + cell * 8
        gradient = language.load(
            grad_ptr
            + row[:, None] * feature_size
            + cell[:, None] * width
            + lane[None, :],
            mask=live[:, None],
            other=0.0,
        )
        fixed = libdevice.float2ll_rn(gradient.to(language.float32) * 16777216.0)
        low = ((fixed + 128) & 255) - 128
        digit = low.to(language.int8)
        sums0 = _hits_dot_triton(ids, live, digit, sums0, 0, vocab_first)
        sums1 = _hits_dot_triton(ids, live, digit, sums1, 1, vocab_rest)
        sums2 = _hits_dot_triton(ids, live, digit, sums2, 2, vocab_rest)
        sums3 = _hits_dot_triton(ids, live, digit, sums3, 3, vocab_rest)
        sums4 = _hits_dot_triton(ids, live, digit, sums4, 4, vocab_rest)
        sums5 = _hits_dot_triton(ids, live, digit, sums5, 5, vocab_rest)
        sums6 = _hits_dot_triton(ids, live, digit, sums6, 6, vocab_rest)
        sums7 = _hits_dot_triton(ids, live, digit, sums7, 7, vocab_rest)
        _spill_triton(
            partials,
            ids,
            offsets_ptr,
            live,
            fixed,
            (fixed - low) >> 8,
            table_rows,
            width,
            vocab_first,
            vocab_rest,
        )
    _add_rows_triton(partials, sums0, offsets_ptr, table_rows, 0, vocab_first, width)
    _add_rows_triton(partials, sums1, offsets_ptr, table_rows, 1, vocab_rest, width)
    _add_rows_triton(partials, sums2, offsets_ptr, table_rows, 2, vocab_rest, width)
    _add_rows_triton(partials, sums3, offsets_ptr, table_rows, 3, vocab_rest, width)
    _add_rows_triton(partials, sums4, offsets_ptr, table_rows, 4, vocab_rest, width)
    _add_rows_triton(partials, sums5, offsets_ptr, table_rows, 5, vocab_rest, width)
    _add_rows_triton(partials, sums6, offsets_ptr, table_rows, 6, vocab_rest, width)
    _add_rows_triton(partials, sums7, offsets_ptr, table_rows, 7, vocab_rest, width)


def _hits_dot_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    ids: language.tensor,
    live: language.tensor,
    digit: language.tensor,
    sums: language.tensor,
    field: language.constexpr,
    vocab: language.constexpr,
) -> language.tensor:
    """Add ``hits @ digit`` for one field to its int32 sums."""
    identity = language.load(ids + field, mask=live, other=-1.0)
    hits = (
        language.arange(0, vocab)[:, None]
        == (identity.to(language.float32).to(language.int32)[None, :])
    )
    return language.dot(
        hits.to(language.int8),
        digit,
        acc=sums,
        out_dtype=language.int32,
    )


# What the low digits leave: the higher digits, and ids outside their field's
# vocabulary, which land in row ``offset + id`` wherever that falls. Both are zero
# on a normal tile, so one reduction skips them; where they are not, this adds
# them to the program's int64 partials with atomics, which keep repeated rows exact.
def _spill_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    partials: language.tensor,
    ids: language.tensor,
    offsets_ptr: language.tensor,
    live: language.tensor,
    fixed: language.tensor,
    rest: language.tensor,
    table_rows: int,
    width: language.constexpr,
    vocab_first: language.constexpr,
    vocab_rest: language.constexpr,
) -> None:
    """Add a tile's higher digits and out-of-vocabulary rows, if it has any."""
    lane = language.arange(0, width)
    flags = language.max((rest != 0).to(language.int32), axis=1)
    for field in language.static_range(8):
        first, size = _field_rows_triton(offsets_ptr, table_rows, field)
        identity = language.load(ids + field, mask=live, other=0.0)
        identity = identity.to(language.float32).to(language.int32)
        flags |= (live & ((identity < 0) | (identity >= size))).to(language.int32)
    if language.max(flags, axis=0) > 0:
        for plane in language.range(1, 8):
            if (
                language.max(
                    language.max((rest != 0).to(language.int32), axis=1),
                    axis=0,
                )
                > 0
            ):
                digit = ((rest + 128) & 255) - 128
                rest = (rest - digit) >> 8
                _higher_rows_triton(
                    partials,
                    ids,
                    offsets_ptr,
                    live,
                    digit.to(language.int8),
                    8 * plane,
                    table_rows,
                    0,
                    vocab_first,
                    width,
                )
                for field in language.static_range(1, 8):
                    _higher_rows_triton(
                        partials,
                        ids,
                        offsets_ptr,
                        live,
                        digit.to(language.int8),
                        8 * plane,
                        table_rows,
                        field,
                        vocab_rest,
                        width,
                    )
        for field in language.static_range(8):
            first, size = _field_rows_triton(offsets_ptr, table_rows, field)
            identity = language.load(ids + field, mask=live, other=0.0)
            identity = identity.to(language.float32).to(language.int32)
            target = first + identity
            outside = (
                live
                & ((identity < 0) | (identity >= size))
                & (target >= 0)
                & (target < table_rows)
            )
            language.atomic_add(
                partials + target[:, None] * width + lane[None, :],
                fixed,
                mask=outside[:, None],
                sem="relaxed",
            )


def _higher_rows_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    partials: language.tensor,
    ids: language.tensor,
    offsets_ptr: language.tensor,
    live: language.tensor,
    digit: language.tensor,
    shift: language.tensor,
    table_rows: int,
    field: language.constexpr,
    vocab: language.constexpr,
    width: language.constexpr,
) -> None:
    """Add one field's ``hits @ digit``, weighted by ``2^shift``, to its rows."""
    sums = language.zeros([vocab, width], dtype=language.int32)
    sums = _hits_dot_triton(ids, live, digit, sums, field, vocab)
    _add_rows_triton(
        partials,
        sums.to(language.int64) << shift,
        offsets_ptr,
        table_rows,
        field,
        vocab,
        width,
    )


def _add_rows_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    partials: language.tensor,
    sums: language.tensor,
    offsets_ptr: language.tensor,
    table_rows: int,
    field: language.constexpr,
    vocab: language.constexpr,
    width: language.constexpr,
) -> None:
    """Add a field's per-value sums, as int64, to its rows of the partials."""
    first, size = _field_rows_triton(offsets_ptr, table_rows, field)
    value = language.arange(0, vocab)
    language.atomic_add(
        partials
        + (first + value)[:, None] * width
        + language.arange(0, width)[None, :],
        sums.to(language.int64),
        mask=(value < size)[:, None],
        sem="relaxed",
    )


def _field_rows_triton(
    offsets_ptr: language.tensor,
    table_rows: int,
    field: language.constexpr,
) -> tuple[language.tensor, language.tensor]:
    """Return a field's first table row and its vocabulary size."""
    first = language.load(offsets_ptr + field)
    if field < 7:
        size = language.load(offsets_ptr + field + 1) - first
    else:
        size = table_rows - first
    return first, size


def _embed_decode_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    partials_ptr: language.tensor,
    output_ptr: language.tensor,
    programs: int,
    entries: int,
    stride: int,
    program_block: language.constexpr,
    block: language.constexpr,
) -> None:
    """Add the programs' int64 partials, then decode them through double to bf16."""
    entry = language.program_id(0) * block + language.arange(0, block)
    live = entry < entries
    total = language.zeros([block], dtype=language.int64)
    for start in range(0, programs, program_block):
        program = start + language.arange(0, program_block)
        total += language.sum(
            language.load(
                partials_ptr + program[:, None] * stride + entry[None, :],
                mask=(program[:, None] < programs) & live[None, :],
                other=0,
            ),
            axis=0,
        )
    value = (total.to(language.float64) * (1.0 / 16777216.0)).to(language.float32)
    language.store(output_ptr + entry, value.to(language.bfloat16), mask=live)
