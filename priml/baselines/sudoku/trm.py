"""Tiny Recursive Model (TRM) for sudoku.

Architecture: iterative reasoning with slow-cycles (outer) and fast-cycles
(inner). Two latent states z_slow and z_fast are updated by a shared stack of
post-norm transformer blocks (fused-QKV attention with a shared affine-free
qk-norm + modified SwiGLU under 1D RoPE), with three sudoku-specific input
channels added to the grid-token embeddings:

* a 16-token per-puzzle prefix from a sparse puzzle embedding,
* factored row/column/box positional tables (``embed_pos_row`` /
  ``embed_pos_col`` / ``embed_pos_box``),
* a decoded-grid feedback embedding (``embed_feedback``) re-injecting the
  previous ACT step's own prediction.

Core computation per slow-cycle::

    c = z_slow + input_emb
    for _ in range(fast_cycles):
        z_fast = reasoning(z_fast + c, cos_sin=cos_sin)
    z_slow = reasoning(z_slow + z_fast, cos_sin=cos_sin)
    logits = head(z_slow)
    q_halt = q_head(z_slow[:, 0])[..., 0]

:meth:`TRM.forward` runs ``slow_cycles`` of ``core()``, gradient only on the
last. :meth:`TRM.act_step` is the eval unit: one full ``slow_cycles`` forward
per ACT step, exactly the unit training carries latents across.

Precision and compile contract (the trainer's side of the seam):

* Master weights stay fp32; the trainer wraps forward passes in bf16
  autocast. The model never casts its own weights.
* When ``config.compile`` is True the constructor binds
  ``torch.compile(self.reasoning.forward)`` -- a compiled callable, not a
  submodule -- so the state_dict never grows ``_orig_mod`` keys and the
  recurrence driver stays eager. The trainer is expected to set
  ``torch._inductor.config.emulate_precision_casts = True`` BEFORE
  constructing a compiled model (it keeps compiled bf16 train and eval
  numerics sound); the model does not set process-global flags itself.
* The trainer may flip ``config.compile`` off at runtime to force eager
  evaluation; :meth:`TRM.core` re-reads the flag on every call.

Parameter names are a stable contract: the trainer routes parameters to
optimizers by name substring (``embed``/``head`` -> AdamW, everything else ->
Muon), so the attribute names here (``embed_tokens``, ``head``, ``q_head``,
``puzzle_emb``, ``reasoning``, ``embed_pos_row/col/box``, ``embed_feedback``)
must not be renamed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol, Self, cast, override

import copy
import functools
import logging
import math

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.baselines.sudoku import prefix
from priml.baselines.sudoku.embedding import FactoredPositions, PredictionFeedback
from priml.cost import Cost, cost, elementwise_cost, traffic
from priml.model.attention.rope import RoPE
from priml.model.attention.self_attention import SelfAttention
from priml.model.embedding import Embedding
from priml.model.init import truncated_normal
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock


logger = logging.getLogger(__name__)

InitFn = Callable[[Tensor], object]
"""Weight-init callable: mutates the tensor in place."""


def recipe_block() -> TransformerBlock.Config:
    """Build the TRM reasoning block, with every priml default pinned.

    priml's own defaults differ on four axes -- it is prenorm, uses eps-1e-6
    block norms, attaches no qk-norm or ffn norm, and initializes with
    kaiming_uniform. Each divergence changes forward numerics while leaving
    the state_dict byte-identical, so a checkpoint still loads and silently
    computes something else. Build every block from this factory rather than
    from a bare ``TransformerBlock.Config()``.

    Returns:
      config: A recipe-faithful block config (post-norm, eps-1e-5 block
        norms, shared affine-free eps-1e-6 qk-norm, modified SwiGLU).

    """
    return TransformerBlock.Config(
        prenorm=False,
        attn=SelfAttention.Config(
            # -1 means "inherit TRM.Config.num_heads" -- priml defaults heads
            # to 8, which would silently override the model's own width.
            num_heads=-1,
            norm_qk=RMSNorm.Config(),
            init_weight=trm_truncated_normal_corrected,
        ),
        ffn=SwiGLU.Config(
            expansion=8 / 3,
            round_to=256,
            gate=True,
            norm=RMSNorm.Config(),
            init_weight=trm_truncated_normal_corrected,
            init_weight_out=trm_truncated_normal_corrected,
        ),
        norm1=RMSNorm.Config(eps=1e-5),
        norm2=RMSNorm.Config(eps=1e-5),
    )


class TRM(nn.Module):
    """Tiny Recursive Model with pos2d position tables and feedback embedding.

    See the module docstring for the architecture and for the precision /
    compile / parameter-name contracts shared with the trainer.
    """

    class Config(Fig["TRM"]):
        vocab_size: int = -1
        """Tokens: 0=pad, 1=blank, 2-10=digits 1-9."""

        puzzle_grid_shape: tuple[int, ...] = ()
        """Grid token layout per puzzle (sudoku: a flat 81-token sequence)."""

        channels_in: int = 512
        """Token embedding and hidden state dimensionality."""

        num_layers: int = 2
        """Reasoning blocks per application."""

        num_heads: int = 8
        """Attention heads; also sets the RoPE dim (channels_in // num_heads)."""

        slow_cycles: int = 6
        """Outer iterations per forward pass."""

        fast_cycles: int = 9
        """Inner iterations per slow-cycle."""

        q_head_outputs: int = 2
        """Q-head logits. Index 0 is q_halt; index 1 (q_continue) is unused
        but kept for reference-TRM weight-shape parity."""

        num_puzzle_identifiers: int = 1
        """Vocab size of the per-puzzle embedding (0 disables it)."""

        puzzle_emb_len: int = 16
        """Number of prefix tokens contributed by the puzzle embedding."""

        puzzle_emb_ndim: int = -1
        """Per-identifier embedding dim. -1 = channels_in (reference default)."""

        puzzle_emb_batch_size: int = 384
        """Train batch size; sizes the sparse puzzle embedding's local
        gradient buffer."""

        puzzle_emb_init_std: float = 0.0
        """Stddev for the sparse puzzle embedding init. Reference uses 0."""

        pos2d_grid_shape: tuple[int, int] | None = (0, 0)
        """(rows, cols) factorization of the flat grid for the row/col/box
        positional tables. None disables the tables (the plain-TRM baseline)."""

        pos2d_box_shape: tuple[int, int] = (0, 0)
        """(rows, cols) of one constraint box tiling the grid (sudoku 3x3)."""

        pos2d_init_std: float = 1.0
        """Pre-rescale trunc-normal std for the positional tables (effective
        std 1.0 after the embed_scale multiply at runtime)."""

        feedback_init_std: float = 0.0
        """Pre-rescale trunc-normal std for the feedback table. The default
        0.0 zero-initializes it, making the initial forward bit-identical to
        a model without the feedback channel regardless of ``feedback_ids``."""

        compile: bool = True
        """Bind ``torch.compile(reasoning.forward, fullgraph=True)`` at
        construction. The trainer must set ``torch._inductor.config.
        emulate_precision_casts = True`` before construction, and may flip
        this flag off at runtime to force eager evaluation."""

        dtype: torch.dtype | None = torch.bfloat16
        """Autocast compute dtype. Master weights stay fp32; this only casts
        the sparse puzzle embedding's forward output (reference behavior)."""

        block: TransformerBlock.Config | None = None
        """Reasoning block config. None builds the recipe block in
        ``finalize()``: post-norm, eps-1e-5 block norms, modified SwiGLU, and
        a shared affine-free eps-1e-6 qk-norm. Baseline (pre-qk-norm) rungs
        must set ``block.attn.norm_qk = None`` explicitly."""

        @property
        def num_puzzle_grid_tokens(self) -> int:
            """Return the number of puzzle grid tokens."""
            return math.prod(self.puzzle_grid_shape)

        @property
        def num_prefix_tokens(self) -> int:
            """Return the number of prefix tokens."""
            return self.puzzle_emb_len if self.num_puzzle_identifiers > 0 else 0

        @property
        def total_seq_len(self) -> int:
            """Return the total sequence length."""
            return self.num_prefix_tokens + self.num_puzzle_grid_tokens

        def cost(self, *, batch_size: int, dtype: torch.dtype | None) -> Cost:
            """Cost one forward (``slow_cycles`` core applications) and its backward.

            Mirrors :meth:`SudokuNet.Config.cost`: every slow cycle runs the core
            forward, only the last runs backward, and one core runs the block
            stack ``fast_cycles + 1`` times over weights owned once.

            Args:
              batch_size: Puzzles in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            if self.block is None:
                raise ValueError("cost() needs a finalized config.")
            width = self.channels_in
            latent = self.total_seq_len
            rows = latent * batch_size
            stack = cost(
                self.block,
                seq_len=latent,
                batch_size=batch_size,
                dtype=dtype,
            ).tile(self.num_layers, copies=self.num_layers)
            adds = elementwise_cost(
                primal=(self.fast_cycles + 2) * width * rows,
                adjoint=(self.fast_cycles + 2) * width * rows,
                channels=(self.fast_cycles + 2) * width,
                inputs=2,
                rows=rows,
                dtype=dtype,
            )
            head = Linear.Config(channels_in=width, channels_out=self.vocab_size)
            q_head = Linear.Config(
                channels_in=width,
                channels_out=self.q_head_outputs,
                bias=True,
            )
            core = (
                stack.tile(self.fast_cycles + 1)
                + adds
                + cost(head, seq_len=latent, batch_size=batch_size, dtype=dtype)
                + cost(q_head, seq_len=1, batch_size=batch_size, dtype=dtype)
            )
            total = (
                self._embedding_cost(batch_size=batch_size, dtype=dtype)
                + core
                + core.only("primal").tile(self.slow_cycles - 1)
            )
            if self.num_prefix_tokens > 0:
                total += cost(
                    prefix.SparsePuzzleEmbedding.Config(
                        channels_in=width,
                        channels_out=self.puzzle_emb_ndim,
                        num_puzzles=self.num_puzzle_identifiers,
                        num_tokens=self.puzzle_emb_len,
                        batch_size=self.puzzle_emb_batch_size,
                    ),
                    seq_len=1,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            return total

        def _embedding_cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
        ) -> Cost:
            """Token table and scale, plus the pos2d and feedback channels."""
            width = self.channels_in
            grid_len = self.num_puzzle_grid_tokens
            rows = grid_len * batch_size
            bus = {"seq_len": grid_len, "batch_size": batch_size, "dtype": dtype}
            add = traffic(
                "primal",
                "elementwise",
                elements=3 * width * rows,
                flops=width * rows,
                dtype=dtype,
            )
            total = (
                cost(
                    Embedding.Config(channels_in=self.vocab_size, channels_out=width),
                    **bus,
                )
                + elementwise_cost(
                    primal=width * rows,
                    adjoint=width * rows,
                    channels=width,
                    adjoint_inputs=1,
                    rows=rows,
                    dtype=dtype,
                )
                + cost(
                    PredictionFeedback.Config(
                        channels_in=self.vocab_size,
                        channels_out=width,
                    ),
                    **bus,
                )
                + add
            )
            if self.pos2d_grid_shape is not None:
                total += (
                    cost(
                        FactoredPositions.Config(
                            grid_shape=self.pos2d_grid_shape,
                            box_shape=self.pos2d_box_shape,
                            channels_out=width,
                        ),
                        **bus,
                    )
                    + add
                )
            return total

        @override
        def finalize(self) -> Self:
            if self.puzzle_emb_ndim == -1:
                self.puzzle_emb_ndim = self.channels_in
            if self.block is None:
                self.block = recipe_block()
            if self.block.channels_in == -1:
                self.block.channels_in = self.channels_in
            # Priml types ``attn`` loosely as Makeable[nn.Module]; narrow to
            # the config whose head dims this model owns.
            attn = self.block.attn
            assert isinstance(attn, SelfAttention.Config)
            if attn.num_heads == -1:
                attn.num_heads = self.num_heads
            if attn.channels_head == -1:
                attn.channels_head = self.channels_in // self.num_heads
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.vocab_size < 1 or not config.puzzle_grid_shape:
            raise ValueError("TRM requires vocabulary and grid shape from the dataset.")
        if config.pos2d_grid_shape is not None and (
            min(*config.pos2d_grid_shape, *config.pos2d_box_shape) < 1
        ):
            raise ValueError("TRM requires pos2d grid and box shapes from the dataset.")
        self.config = config
        c = config.channels_in
        # Embedding rescale trick: init tables with std=1/sqrt(C), multiply
        # by sqrt(C) at runtime.
        self.embed_scale = c**0.5

        # NOTE: construction order below is a checkpoint-parity contract --
        # every global-RNG draw must happen in exactly this order so that a
        # seeded init reproduces the reference weights bit-for-bit:
        # embed_tokens -> head -> block[0..num_layers-1] internals (attn
        # QKV/out, ffn up/down) -> slow_init -> fast_init -> embed_pos_row ->
        # embed_pos_col -> embed_pos_box -> embed_feedback (no draw at std 0).
        # q_head and puzzle_emb draw nothing (zero/constant init).
        self.embed_tokens = Embedding.Config(
            channels_out=c,
            channels_in=config.vocab_size,
        ).make()
        truncated_normal(
            self.embed_tokens.weight,
            std=1.0 / self.embed_scale,
            depth_index=(),
            variance_correction=True,
        )

        self.head = Linear.Config(
            channels_in=c,
            channels_out=config.vocab_size,
            bias=False,
            init_weight=trm_truncated_normal_corrected,
        ).make()

        self.q_head = Linear.Config(
            channels_in=c,
            channels_out=config.q_head_outputs,
            bias=True,
            init_weight=nn.init.zeros_,
            init_bias=functools.partial(nn.init.constant_, val=-5.0),
        ).make()

        if config.num_puzzle_identifiers > 0:
            self.puzzle_emb: SparsePuzzleEmbedding | None = SparsePuzzleEmbedding(
                num_embeddings=config.num_puzzle_identifiers,
                embedding_dim=config.puzzle_emb_ndim,
                batch_size=config.puzzle_emb_batch_size,
                init_std=config.puzzle_emb_init_std,
                cast_to=config.dtype,
            )
        else:
            self.puzzle_emb = None

        block = config.block
        if block is None:
            raise ValueError("finalize() fills the default block")
        # repeat= builds num_layers independent copies, replacing an explicit
        # deepcopy loop. Each copy is finalized separately, so the per-block
        # init draw order (the checkpoint-parity contract) is unchanged.
        self.reasoning = Sequential.Config(
            elements=copy.deepcopy(block),
            repeat=config.num_layers,
        ).make()

        self.rope = RoPE.Config(
            channels_head=c // config.num_heads,
        ).make()

        self.slow_init: Tensor = nn.Buffer(
            _corrected(torch.empty(1, c), std=1.0),
            persistent=True,
        )
        self.fast_init: Tensor = nn.Buffer(
            _corrected(torch.empty(1, c), std=1.0),
            persistent=True,
        )

        # Bind the compiled reasoning sub-unit eagerly. ``torch.compile``
        # wraps the BOUND METHOD ``self.reasoning.forward`` -- a compiled
        # callable, not an ``nn.Module`` -- so it is not registered as a
        # submodule and adds no ``_orig_mod`` state_dict keys; the eager core
        # drives it. Compilation only TRACES on first invocation, so binding
        # here is cheap.
        self._reasoning_compiled: _ReasoningFn | None = None
        if config.compile:
            self._reasoning_compiled = cast(
                _ReasoningFn,
                torch.compile(self.reasoning.forward, fullgraph=True),
            )

        self._dummy = nn.Buffer(torch.empty(0), persistent=True)

        if config.pos2d_grid_shape is not None:
            rows, cols = config.pos2d_grid_shape
            box_rows, box_cols = config.pos2d_box_shape
            if rows * cols != config.num_puzzle_grid_tokens:
                raise ValueError(
                    f"pos2d_grid_shape {config.pos2d_grid_shape} does not factor "
                    f"the {config.num_puzzle_grid_tokens}-token puzzle grid.",
                )
            if rows % box_rows or cols % box_cols:
                raise ValueError(
                    f"pos2d_box_shape {config.pos2d_box_shape} does not tile "
                    f"pos2d_grid_shape {config.pos2d_grid_shape}.",
                )
            cell = torch.arange(rows * cols)
            row = cell // cols
            col = cell % cols
            box = (row // box_rows) * (cols // box_cols) + col // box_cols
            self.row_index = nn.Buffer(row, persistent=False)
            self.col_index = nn.Buffer(col, persistent=False)
            self.box_index = nn.Buffer(box, persistent=False)
            num_boxes = (rows // box_rows) * (cols // box_cols)
            self.embed_pos_row: nn.Parameter | None = _pos_table(rows, config)
            self.embed_pos_col: nn.Parameter | None = _pos_table(cols, config)
            self.embed_pos_box: nn.Parameter | None = _pos_table(num_boxes, config)
        else:
            self.embed_pos_row = None
            self.embed_pos_col = None
            self.embed_pos_box = None

        self._feedback_ids: Tensor | None = None
        self.embed_feedback = _feedback_table(config)

        self._log_parameter_counts()

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return self._dummy.device

    def set_feedback(self, feedback_ids: Tensor | None) -> None:
        """Stash the decoded-grid tokens consumed by the NEXT forward.

        Escape hatch for callers that drive the base-signature ``forward``
        indirectly (the train step calls ``self.model(...)`` without feedback
        kwargs). The stash is consumed exactly once by the next embedding
        build, so a stale grid can never leak into a later forward.

        Args:
          feedback_ids: ``[B, grid_len]`` token ids of the previous ACT
            step's decoded grid (givens already clamped by the caller), or
            None for no feedback contribution.

        """
        self._feedback_ids = feedback_ids

    @override
    def forward(
        self,
        input_ids: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        puzzle_identifiers: Tensor | None = None,
        feedback_ids: Tensor | None = None,
    ) -> dict[str, Tensor | list[Tensor]]:
        """Multi H-cycle forward.

        Args:
          input_ids: ``[B, grid_len]`` token ids.
          z_slow: ``[B, total_seq_len, hidden]`` slow latent state.
          z_fast: ``[B, total_seq_len, hidden]`` fast latent state.
          puzzle_identifiers: ``[B]`` puzzle ids; required when
            ``num_puzzle_identifiers > 0``.
          feedback_ids: ``[B, grid_len]`` decoded-grid tokens; when None the
            stash set via :meth:`set_feedback` (if any) is consumed instead,
            so both calling conventions compose.

        Returns:
          out: Dict with logits, all_logits, q_halt, z_slow, z_fast,
            all_z_slow (grid logits prefix-stripped; latents detached).

        """
        if feedback_ids is not None:
            self.set_feedback(feedback_ids)
        input_emb, cos_sin = self._embed_and_prepare(input_ids, puzzle_identifiers)
        hc = self.run_h_cycles(
            input_emb,
            z_slow,
            z_fast,
            cos_sin,
            collect_intermediates=True,
        )
        logits = hc.logits
        all_logits = list(hc.all_logits)
        n_prefix = self.config.num_prefix_tokens
        if n_prefix > 0:
            logits = logits[:, n_prefix:]
            all_logits = [lg[:, n_prefix:] for lg in all_logits]
        return {
            "logits": logits,
            "all_logits": all_logits,
            "q_halt": hc.q_halt,
            "z_slow": hc.z_slow.detach(),
            "z_fast": hc.z_fast.detach(),
            "all_z_slow": list(hc.all_z_slow),
        }

    def act_step(
        self,
        input_ids: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        puzzle_identifiers: Tensor | None = None,
        feedback_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """One ACT step for eval: the full ``slow_cycles`` forward.

        This is the unit the model is trained on -- :meth:`forward` runs
        ``slow_cycles`` core applications per call, and training carries
        ``z_slow``/``z_fast`` across ACT steps exactly this way. Eval must
        iterate this same forward (reference TRM evals identically: the whole
        ``H_cycles x L_cycles`` recurrence per ACT step); an eval loop built
        on a single core application drives the latents into a regime the
        model never saw in training and collapses accuracy to chance.

        Args:
          input_ids: ``[B, grid_len]`` token ids.
          z_slow: ``[B, total_seq_len, hidden]`` slow latent.
          z_fast: ``[B, total_seq_len, hidden]`` fast latent.
          puzzle_identifiers: ``[B]`` puzzle ids; required when
            ``num_puzzle_identifiers > 0``.
          feedback_ids: ``[B, grid_len]`` decoded-grid tokens (see
            :meth:`forward`).

        Returns:
          out: Dict with logits, q_halt, z_slow, z_fast (latents detached).

        """
        out = self.forward(input_ids, z_slow, z_fast, puzzle_identifiers, feedback_ids)
        logits = out["logits"]
        q_halt = out["q_halt"]
        z_slow_out = out["z_slow"]
        z_fast_out = out["z_fast"]
        assert isinstance(logits, Tensor)
        assert isinstance(q_halt, Tensor)
        assert isinstance(z_slow_out, Tensor)
        assert isinstance(z_fast_out, Tensor)
        return {
            "logits": logits,
            "q_halt": q_halt,
            "z_slow": z_slow_out,
            "z_fast": z_fast_out,
        }

    def init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Initialize z_slow, z_fast for a batch from the init vectors.

        Args:
          batch_size: Number of samples in the batch.

        Returns:
          z_slow: ``[B, total_seq_len, hidden]`` slow latent init.
          z_fast: ``[B, total_seq_len, hidden]`` fast latent init.

        """
        s = self.config.total_seq_len
        z_slow = self.slow_init[0].expand(batch_size, s, -1).contiguous()
        z_fast = self.fast_init[0].expand(batch_size, s, -1).contiguous()
        return z_slow, z_fast

    def run_h_cycles(
        self,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
        *,
        collect_intermediates: bool = False,
    ) -> HCycleResult:
        """Run ``slow_cycles - 1`` cycles under no_grad, the final with grad.

        Args:
          input_emb: ``[B, total_seq_len, hidden]`` prefix-prepended input.
          z_slow: ``[B, total_seq_len, hidden]`` slow latent state.
          z_fast: ``[B, total_seq_len, hidden]`` fast latent state.
          cos_sin: Optional RoPE ``(cos, sin)`` pair.
          collect_intermediates: Also return per-cycle logits and z_slow.

        Returns:
          result: The :class:`HCycleResult` bundle for the final cycle
            (plus per-cycle intermediates when requested).

        """
        cos_sin_det = (
            (cos_sin[0].detach(), cos_sin[1].detach()) if cos_sin is not None else None
        )
        all_logits: list[Tensor] = []
        all_z_slow: list[Tensor] = []
        with torch.no_grad():
            for _ in range(self.config.slow_cycles - 1):
                logits_i, _, z_slow, z_fast = self.core(
                    input_emb.detach(),
                    z_slow.detach(),
                    z_fast.detach(),
                    cos_sin_det,
                )
                if collect_intermediates:
                    all_logits.append(logits_i)
                    all_z_slow.append(z_slow)
        logits, q_halt, z_slow, z_fast = self.core(input_emb, z_slow, z_fast, cos_sin)
        if collect_intermediates:
            all_logits.append(logits.detach())
            all_z_slow.append(z_slow.detach())
        return HCycleResult(
            logits,
            q_halt,
            z_slow,
            z_fast,
            tuple(all_logits),
            tuple(all_z_slow),
        )

    def core(
        self,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One slow-cycle: fast_cycles of z_fast updates, then one z_slow update.

        The reasoning sub-unit dispatches on ``config.compile`` EVERY call, so
        the trainer can flip the flag off at runtime to force eager eval
        (compile + bf16 corrupts the repeated-carry eval loop unless
        ``emulate_precision_casts`` is set).

        Args:
          input_emb: ``[B, S, C]`` prefix-prepended input embedding.
          z_slow: ``[B, S, C]`` slow latent state.
          z_fast: ``[B, S, C]`` fast latent state.
          cos_sin: Optional RoPE ``(cos, sin)`` pair.

        Returns:
          logits: ``[B, S, V]`` token logits over the full sequence.
          q_halt: ``[B]`` halt logit at the readout position.
          z_slow: ``[B, S, C]`` updated slow latent.
          z_fast: ``[B, S, C]`` updated fast latent.

        """
        reasoning: _ReasoningFn = (
            self._reasoning_compiled
            if self._reasoning_compiled is not None and self.config.compile
            else self.reasoning
        )
        c = z_slow + input_emb
        for _ in range(self.config.fast_cycles):
            z_fast = reasoning(z_fast + c, cos_sin=cos_sin)
        z_slow = reasoning(z_slow + z_fast, cos_sin=cos_sin)
        logits = self.head(z_slow)
        # Q readout at sequence position 0 (the first puzzle-emb prefix
        # token). Index 0 of the q-head output is q_halt; the second output
        # (q_continue) exists only for reference weight-shape parity.
        q_logits = self.q_head(z_slow[:, 0]).to(torch.float32)
        if self.config.q_head_outputs == 1:
            q_halt = q_logits.squeeze(-1)
        else:
            q_halt = q_logits[..., 0]
        return logits, q_halt, z_slow, z_fast

    # Composition order is a numerics contract: (base embedding + scaled puzzle prefix
    # concat) -> + pos2d on grid tokens -> + feedback embedding. The fp addition order
    # is ``(emb + pos) + fb``.
    def _embed_and_prepare(
        self,
        input_ids: Tensor,
        puzzle_identifiers: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Embed tokens, prepend the puzzle prefix, add pos2d + feedback."""
        cfg = self.config
        input_emb = self.embed_scale * self.embed_tokens(input_ids)
        if self.puzzle_emb is not None:
            if puzzle_identifiers is None:
                raise ValueError(
                    "puzzle_identifiers is required when num_puzzle_identifiers > 0.",
                )
            B = input_emb.shape[0]
            puzzle_vec = self.puzzle_emb(puzzle_identifiers)
            # Pad puzzle_emb_ndim out to puzzle_emb_len * channels_in, then
            # reshape into [B, puzzle_emb_len, channels_in]. Matches reference
            # TRM's prepend behavior.
            total = cfg.puzzle_emb_len * cfg.channels_in
            if puzzle_vec.shape[-1] < total:
                pad = total - puzzle_vec.shape[-1]
                puzzle_vec = nn.functional.pad(puzzle_vec, (0, pad))
            puzzle_prefix = puzzle_vec.reshape(B, cfg.puzzle_emb_len, cfg.channels_in)
            # Reference TRM concatenates the raw puzzle embedding with the raw
            # token embedding and scales the whole sequence by embed_scale
            # afterwards. The token embedding here is already pre-scaled, so
            # the puzzle prefix must be scaled too; otherwise it enters
            # embed_scale (= sqrt(hidden)) too weak.
            puzzle_prefix = self.embed_scale * puzzle_prefix.to(dtype=input_emb.dtype)
            input_emb = torch.cat([puzzle_prefix, input_emb], dim=1)
        if self.embed_pos_row is not None:
            pos = self.embed_scale * self._pos2d().to(dtype=input_emb.dtype)
            n_grid = pos.shape[0]
            grid_emb = input_emb[:, -n_grid:] + pos
            input_emb = torch.cat([input_emb[:, :-n_grid], grid_emb], dim=1)
        feedback = self._feedback_ids
        self._feedback_ids = None  # Consume-once: stale grids never leak.
        if feedback is not None:
            fb_emb = self.embed_scale * self.embed_feedback[feedback].to(
                dtype=input_emb.dtype,
            )
            n_grid = fb_emb.shape[1]
            grid_emb = input_emb[:, -n_grid:] + fb_emb
            input_emb = torch.cat([input_emb[:, :-n_grid], grid_emb], dim=1)
        cos_sin = self.rope(
            torch.arange(cfg.total_seq_len, device=input_emb.device),
        )
        return cast(tuple[Tensor, tuple[Tensor, Tensor] | None], (input_emb, cos_sin))

    def _pos2d(self) -> Tensor:
        """Factored ``[grid_len, hidden]`` positional embedding: row+col+box."""
        if self.embed_pos_row is None:
            raise ValueError("Expected self.embed_pos_row is not None.")
        if self.embed_pos_col is None:
            raise ValueError("Expected self.embed_pos_col is not None.")
        if self.embed_pos_box is None:
            raise ValueError("Expected self.embed_pos_box is not None.")
        return (
            self.embed_pos_row[self.row_index]
            + self.embed_pos_col[self.col_index]
            + self.embed_pos_box[self.box_index]
        )

    def _log_parameter_counts(self) -> None:
        """Log body / puzzle-table parameter counts after construction."""
        body = sum(p.numel() for p in self.parameters())
        if self.puzzle_emb is not None:
            emb = self.puzzle_emb.weights.numel()
            logger.info(
                "model parameters: %.2fM total (%.2fM body + %.2fM puzzle-emb)",
                (body + emb) / 1e6,
                body / 1e6,
                emb / 1e6,
            )
        else:
            logger.info("model parameters: %.2fM (body)", body / 1e6)


class HCycleResult(NamedTuple):
    """Bundle of one H-cycle forward's outputs."""

    logits: Tensor
    q_halt: Tensor
    z_slow: Tensor
    z_fast: Tensor
    all_logits: tuple[Tensor, ...] = ()
    all_z_slow: tuple[Tensor, ...] = ()


# Only the two buffer constructions need a value back; every other call site uses
# ``truncated_normal`` directly.
def _corrected(tensor: Tensor, *, std: float) -> Tensor:
    """Initialize in place with the JAX-corrected truncated normal and return it."""
    truncated_normal(tensor, std=std, depth_index=(), variance_correction=True)
    return tensor


def trm_truncated_normal_corrected(w: Tensor, *, depth: int = -1) -> None:
    """TRM init with the JAX correction: std = 1/sqrt(fan_in).

    The default ``init_weight`` for every projection this model builds. The
    ``depth`` keyword exists because priml layers pass it through
    ``call_init``; TRM does not depth-scale, so it is discarded.

    Args:
      w: Tensor to initialize in place.
      depth: Ignored; present for the priml ``init_weight`` protocol.

    """
    del depth
    truncated_normal(
        w,
        std=w.shape[-1] ** -0.5,
        depth_index=(),
        variance_correction=True,
    )


class SparsePuzzleEmbedding(nn.Module):
    """Reference-style sparse puzzle embedding with local gradient buffer.

    Stores a flat ``[num_embeddings, embedding_dim]`` master table as a
    persistent fp32 buffer. During training, the forward pass copies
    looked-up rows into a non-persistent ``local_weights`` buffer with
    ``requires_grad=True`` and returns that, so backprop accumulates
    gradients only against the rows touched this batch. The trainer's
    SignSGD sparse-embedding step scatters those gradients back into the
    master table. During eval the master table is read directly.

    Args:
      num_embeddings: Vocabulary size (number of distinct puzzle IDs).
      embedding_dim: Per-ID embedding dimension.
      batch_size: Train batch size; sizes the local gradient buffer.
      init_std: Stddev for truncated-normal init of the master table.
        Reference uses 0 (all zeros) for stable initial training.
      cast_to: Optional dtype to cast outputs into for autocast forward.

    """

    def __init__(
        self,
        num_embeddings: int,
        *,
        embedding_dim: int,
        batch_size: int,
        init_std: float,
        cast_to: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.cast_to = cast_to
        self.weights = nn.Buffer(
            _corrected(torch.empty(num_embeddings, embedding_dim), std=init_std),
            persistent=True,
        )
        self.local_weights = nn.Buffer(
            torch.zeros(batch_size, embedding_dim, requires_grad=True),
            persistent=False,
        )
        self.local_ids = nn.Buffer(
            torch.zeros(batch_size, dtype=torch.int32),
            persistent=False,
        )

    @override
    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training:
            out = self.weights[inputs.to(torch.long)]
            return out if self.cast_to is None else out.to(self.cast_to)
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs.to(torch.long)])
            self.local_ids.copy_(inputs.to(torch.int32))
        return (
            self.local_weights
            if self.cast_to is None
            else self.local_weights.to(self.cast_to)
        )

    @override
    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        module = super()._apply(fn, recurse=recurse)
        self.local_weights = nn.Buffer(
            self.local_weights.detach().requires_grad_(True),
            persistent=False,
        )
        self.local_ids = nn.Buffer(self.local_ids.detach(), persistent=False)
        return module


class _ReasoningFn(Protocol):
    """Signature of ``TRM.reasoning`` forward / its compiled wrapper.

    Declared as ``__call__``, not ``forward``: the compiled arm is
    ``torch.compile(self.reasoning.forward)``, a plain function with no
    ``forward`` attribute, so a ``forward`` member would type-check and then
    raise ``AttributeError`` on every compiled run.
    """

    def __call__(
        self,
        z: Tensor,
        /,
        *,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor: ...


def _pos_table(num_embeddings: int, config: TRM.Config) -> nn.Parameter:
    """Positional table with pre-rescaled init (effective std pos2d_init_std)."""
    w = torch.empty(num_embeddings, config.channels_in)
    truncated_normal(
        w,
        std=config.pos2d_init_std / config.channels_in**0.5,
        depth_index=(),
        variance_correction=True,
    )
    return nn.Parameter(w)


def _feedback_table(config: TRM.Config) -> nn.Parameter:
    """``[vocab_size, hidden]`` feedback table (zeros at the default std 0)."""
    w = torch.zeros(config.vocab_size, config.channels_in)
    if config.feedback_init_std > 0:
        truncated_normal(
            w,
            std=config.feedback_init_std / config.channels_in**0.5,
            depth_index=(),
            variance_correction=True,
        )
    return nn.Parameter(w)
