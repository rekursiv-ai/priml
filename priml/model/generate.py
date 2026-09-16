"""Autoregressive text generation with KV caching.

Works with any model exposing the standard interface:
  proj_in(tokens) -> hidden
  blocks: Iterable[TransformerBlock]
  project_to_logits(hidden) -> logits

Example::

    tokens = generate(
        model,
        prompt_ids,
        max_new_tokens=128,
        temperature=0.8,
        top_k=40,
    )
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, cast, runtime_checkable

import math

from torch import Tensor, nn

import torch

from priml.model.custom_types import TensorModule, has_weight


@runtime_checkable
class AttentionLike(Protocol):
    """The attention member ``generate`` reaches for on each block."""

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> object:
        """Alloc kv cache."""
        ...


class BlockLike(Protocol):
    """One transformer block with an explicit cached path."""

    attn: AttentionLike

    def forward_cached[CacheT](
        self,
        x: Tensor,
        /,
        *,
        cache: CacheT,
    ) -> tuple[Tensor, CacheT]:
        """Forward cached."""
        ...


class TransformerLike(Protocol):
    """The model surface ``generate`` uses -- nothing more.

    Declared structurally rather than against ``nn.Module``: a ``Protocol``
    cannot inherit a non-protocol class, and ``Module.__call__`` erases the
    per-model signature anyway.
    """

    @property
    def proj_in(self) -> TensorModule | None:
        """In proj."""
        ...

    @property
    def blocks(self) -> Iterable[nn.Module]:
        """Blocks."""
        ...

    def project_to_logits(self, hidden: Tensor, /) -> Tensor:
        """Project to logits."""
        ...


@torch.inference_mode()
def generate(
    model: TransformerLike,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    max_seq_len: int = 1024,
) -> Tensor:
    """Generate tokens autoregressively with KV caching.

    Args:
      model: A Transformer-compatible model.
      prompt_ids: (B, S) prompt token ids.
      max_new_tokens: Maximum tokens to generate.
      temperature: Sampling temperature (0 = greedy).
      top_k: Top-k filtering (0 = disabled).
      top_p: Nucleus sampling threshold (1.0 = disabled).
      eos_token_id: Stop on this token (None = don't stop early).
      max_seq_len: Maximum total sequence length for KV cache.

    Returns:
      tokens: (B, S + generated) full sequence including prompt.

    """
    if prompt_ids.ndim != 2 or 0 in prompt_ids.shape:
        raise ValueError("prompt_ids must have shape (B, S) with B and S > 0.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative.")
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")
    if not math.isfinite(top_p) or top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be finite and in (0, 1].")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")

    device = prompt_ids.device
    B = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[-1]
    if prompt_len > max_seq_len:
        raise ValueError(
            f"prompt length {prompt_len} exceeds max_seq_len={max_seq_len}.",
        )
    if prompt_len + max_new_tokens > max_seq_len:
        raise ValueError(
            f"prompt length {prompt_len} plus max_new_tokens={max_new_tokens} "
            f"exceeds max_seq_len={max_seq_len}.",
        )

    proj_in = model.proj_in
    if proj_in is None or not has_weight(proj_in):
        raise TypeError("Token generation requires an proj_in with embedding weights.")
    forward = proj_in
    dtype = proj_in.weight.dtype

    # Delegate cache alloc to block; keeps generate arch-agnostic
    # (MLA caches compressed latent).
    blocks: list[BlockLike] = []
    for block in model.blocks:
        if not isinstance(getattr(block, "attn", None), AttentionLike):
            raise TypeError(
                "Token generation requires blocks with an attn attribute "
                "implementing alloc_kv_cache."
            )
        if not isinstance(block, _HasForwardCached):
            raise TypeError(
                "Token generation requires blocks with a forward_cached method."
            )
        blocks.append(cast(BlockLike, block))
    caches = [
        block.attn.alloc_kv_cache(
            batch=B,
            max_seq=max_seq_len,
            device=device,
            dtype=dtype,
        )
        for block in blocks
    ]

    x: Tensor = forward(prompt_ids)
    for i, block in enumerate(blocks):
        x, caches[i] = block.forward_cached(x, cache=caches[i])
    logits: Tensor = model.project_to_logits(x[:, -1:, :])

    generated: list[Tensor] = []
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        next_token = _sample(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if eos_token_id is not None:
            next_token = torch.where(
                finished[:, None],
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
        generated.append(next_token)

        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        x = forward(next_token)
        for i, block in enumerate(blocks):
            x, caches[i] = block.forward_cached(x, cache=caches[i])
        logits = model.project_to_logits(x)

    if not generated:
        return prompt_ids
    return torch.cat([prompt_ids, *generated], dim=-1)


# `isinstance` against a runtime-checkable Protocol resolves a data member
# (BlockLike.attn) via inspect.getattr_static, which never sees an nn.Module
# submodule -- submodules surface only through nn.Module's own __getattr__,
# which getattr_static deliberately bypasses. So isinstance(block, BlockLike)
# is never True for a real block; this method-only half of it is what CAN be
# checked, and `attn` is checked separately above with a plain getattr
# (which, unlike getattr_static, does trigger __getattr__).
@runtime_checkable
class _HasForwardCached(Protocol):
    """The half of ``BlockLike`` an isinstance check can prove."""

    def forward_cached[CacheT](
        self,
        x: Tensor,
        /,
        *,
        cache: CacheT,
    ) -> tuple[Tensor, CacheT]:
        """Forward cached."""
        ...


def _sample(
    logits: Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tensor:
    """Sample a token from logits with temperature, top-k, and top-p."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.where(
            logits >= kth_val,
            torch.full_like(logits, float("-inf")),
        )

    if top_p < 1.0:
        logits = _topp_filter(logits, top_p=top_p)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)


# Sets tokens whose exclusive cumulative probability exceeds ``top_p`` to ``-inf``
# (which softmaxes to 0). Strict ``>`` keeps a boundary token when the preceding
# cumulative mass equals ``top_p`` (the Hugging Face convention).
def _topp_filter(logits: Tensor, top_p: float) -> Tensor:
    """Mask logits outside the top-p nucleus, in original vocab order."""
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    mask = probs.cumsum(dim=-1) - probs > top_p
    mask_value = float("-inf")
    sorted_logits[mask] = mask_value
    # Scatter back to original vocab order over a fully-masked base so filtered
    # positions stay filtered regardless of scatter coverage.
    return torch.full_like(logits, mask_value).scatter(-1, sorted_idx, sorted_logits)
