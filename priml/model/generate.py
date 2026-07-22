"""Autoregressive text generation with KV caching.

Works with any model exposing the standard interface:
  embed(tokens) -> hidden
  blocks: Iterable[TransformerBlock]
  final_norm(hidden) -> hidden
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

from typing import Any

from torch import Tensor

import torch

from priml.model.kvcache import KVCache


# CausalLM duck-type interface:
#   embed: nn.Module           (B, S) -> (B, S, D)
#   blocks: Iterable[Module]   each has ``attn`` with ``alloc_kv_cache``
#                              and accepts ``(x, cache=...)``.
#   final_norm: nn.Module      (B, S, D) -> (B, S, D)
#   project_to_logits(hidden) -> logits


@torch.inference_mode()
def generate(
    model: Any,
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
      model: A CausalLM-compatible model.
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
    device = prompt_ids.device
    B = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[-1]
    if prompt_len > max_seq_len:
        raise ValueError(
            f"prompt length {prompt_len} exceeds max_seq_len={max_seq_len}.",
        )

    # Infer dtype from model parameters.
    param: Tensor = next(iter(model.embed.parameters()))
    dtype = param.dtype

    # Delegate cache alloc to block; keeps generate arch-agnostic
    # (MLA caches compressed latent).
    blocks = list(model.blocks)
    caches = [
        block.attn.alloc_kv_cache(
            batch=B,
            max_seq=max_seq_len,
            device=device,
            dtype=dtype,
        )
        for block in blocks
    ]

    x: Tensor = model.embed(prompt_ids)
    for i, block in enumerate(blocks):
        result: Tensor | tuple[Tensor, KVCache] = block(x, cache=caches[i])
        if isinstance(result, tuple):
            x, caches[i] = result
        else:
            x = result
    x = model.final_norm(x)
    logits: Tensor = model.project_to_logits(x[:, -1:, :])

    generated: list[Tensor] = []
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        next_token = _sample(logits[:, -1, :], temperature, top_k, top_p)
        generated.append(next_token)

        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        x = model.embed(next_token)
        for i, block in enumerate(blocks):
            step_result: Tensor | tuple[Tensor, KVCache] = block(x, cache=caches[i])
            if isinstance(step_result, tuple):
                x, caches[i] = step_result
            else:
                x = step_result
        x = model.final_norm(x)
        logits = model.project_to_logits(x)

    if not generated:
        return prompt_ids
    return torch.cat([prompt_ids, *generated], dim=-1)


def _sample(
    logits: Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tensor:
    """Sample a token from logits with temperature, top-k, and top-p.

    Args:
      logits: (B, V) unnormalized logits.
      temperature: Sampling temperature. 0 = greedy.
      top_k: Keep only top-k logits (0 = disabled).
      top_p: Nucleus sampling threshold (1.0 = disabled).

    Returns:
      token: (B, 1) sampled token ids.

    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.where(logits >= kth_val, torch.full_like(logits, -1e10))

    if top_p < 1.0:
        logits = _topp_filter(logits, top_p)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _topp_filter(logits: Tensor, top_p: float) -> Tensor:
    """Mask logits outside the top-p nucleus, in original vocab order.

    Keeps the smallest token set whose cumulative mass reaches ``top_p`` and
    sets the rest to ``-1e10`` (which softmaxes to 0). Removes tokens whose
    EXCLUSIVE cumulative probability already exceeds ``top_p`` (HF convention;
    strict ``>`` keeps the boundary token that brings the running mass exactly
    to ``top_p``).

    Args:
      logits: (B, V) logits, already temperature- and top-k-adjusted.
      top_p: Nucleus threshold in (0, 1).

    Returns:
      filtered: (B, V) logits with out-of-nucleus positions set to ``-1e10``,
        in original vocab order.

    """
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    mask = probs.cumsum(dim=-1) - probs > top_p
    sorted_logits[mask] = -1e10
    # Scatter back to original vocab order over a fully-masked base so filtered
    # positions stay filtered regardless of scatter coverage.
    return torch.full_like(logits, -1e10).scatter(-1, sorted_idx, sorted_logits)
