"""Attention windows and masks."""

from __future__ import annotations

from torch import Tensor

import torch

from priml.model.custom_types import DepthIndex, flatten_depth_index


def layer_window(*, depth_index: DepthIndex, max_seq_len: int, pattern: str) -> int:
    """One layer's attention window, from a cycled short/long pattern.

    Everything a LAYER needs to know: the pattern cycles over depth, so a
    module holding its own index can answer without asking the stack.

    Args:
      depth_index: This layer's global-to-local stack position.
      max_seq_len: Full context length, and the long window.
      pattern: Cycled ``S`` (half context) and ``L`` (full context) symbols.

    Returns:
      window: Keys this layer's queries may look back over, itself included.

    Raises:
      ValueError: The position is unspecified, or ``pattern`` is invalid.

    """
    pattern = pattern.upper()
    if not pattern or set(pattern) - set("SL"):
        raise ValueError(f"pattern must hold only S and L; got {pattern!r}.")
    index = flatten_depth_index(depth_index)
    if index < 0:
        raise ValueError("depth_index must specify a stack position.")
    return max_seq_len if pattern[index % len(pattern)] == "L" else max_seq_len // 2


def window_sizes(*, num_layers: int, max_seq_len: int, pattern: str) -> list[int]:
    """Return every layer's attention window from a cycled short/long pattern.

    Args:
      num_layers: Blocks in the stack.
      max_seq_len: Full context length, and the long window.
      pattern: Cycled ``S`` (half context) and ``L`` (full context) symbols.

    Returns:
      windows: One window per layer; the last is always the full context, since
        it is the layer that has to see the whole sequence to predict the next
        token -- which is the one thing here a LAYER cannot decide, since it
        turns on being last.

    Raises:
      ValueError: ``pattern`` is empty or holds a symbol other than S and L.

    """
    windows = [
        layer_window(
            depth_index=((layer, num_layers),),
            max_seq_len=max_seq_len,
            pattern=pattern,
        )
        for layer in range(num_layers)
    ]
    windows[-1] = max_seq_len
    return windows


def window_mask(q: Tensor, k: Tensor, *, window: int) -> Tensor | None:
    """Additive mask admitting the last ``window`` keys, causally.

    Args:
      q: ``[..., S, num_heads, channels_head]`` queries.
      k: Keys, same layout.
      window: Keys each query may look back over, ITSELF INCLUDED. -1, or a
        value reaching the whole context, needs no mask.

    Returns:
      mask: Additive ``[S, T]`` mask, or None when every key is admissible.

    """
    s, t = q.shape[-3], k.shape[-3]
    if window < 0 or window >= t:
        return None
    offset = torch.arange(t, device=q.device)
    offset = offset[t - s :, None] - offset[None, :]
    # ``<=``, not ``<``: a fused kernel's ``window_size=(w, 0)`` admits w keys
    # of history IN ADDITION to the query's own position, so the exclusive form
    # attends to one key fewer per row and is a different model.
    admissible = (offset >= 0) & (offset <= window)
    return torch.zeros(s, t, dtype=q.dtype, device=q.device).masked_fill(
        ~admissible,
        float("-inf"),
    )


def causal_chunk_mask(q: Tensor, k: Tensor) -> Tensor | None:
    """Additive causal mask for a multi-token chunk against a longer cache.

    Returns ``None`` when query and key lengths match (the square case
    that ``is_causal`` already handles). Otherwise builds a ``[S, T]``
    bottom-right-aligned mask so query ``i`` (absolute position
    ``T - S + i``) attends to keys ``0..T - S + i`` -- never to later
    tokens in the same chunk.
    """
    s, t = q.shape[-3], k.shape[-3]
    if s == t:
        return None
    allowed = torch.ones(s, t, dtype=torch.bool, device=q.device).tril(diagonal=t - s)
    return torch.zeros(s, t, dtype=q.dtype, device=q.device).masked_fill(
        ~allowed,
        float("-inf"),
    )
