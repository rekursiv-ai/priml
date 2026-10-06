"""Safe persistence boundary for native Qwen3.5 mixed attention caches."""

from collections.abc import Mapping
from typing import TypeGuard

from torch import Tensor

import torch

from priml.model.attention.kvcache import KVCache
from priml.model.custom_types import DepthIndex


type Qwen35CacheState = dict[DepthIndex, dict[str, Tensor | int | str]]


def cache_state_dict(cache: Mapping[DepthIndex, object]) -> Qwen35CacheState:
    """Convert a live Qwen3.5 cache to tensors and primitive metadata.

    The returned tensors and primitive metadata can be passed to ``torch.save``
    and restored with ``torch.load(weights_only=True)``.

    Args:
      cache: Live attention caches returned by ``Qwen35.alloc_cache``.

    Returns:
      state: Cache tensors plus primitive progress metadata for safe serialization.

    Raises:
      TypeError: A layer cache is not native Qwen3.5 attention state.
      ValueError: A cache layer has incompatible tensors or progress metadata.

    """
    state: Qwen35CacheState = {}
    for key, layer in cache.items():
        if not key:
            raise ValueError("A Qwen3.5 cache key must be nonempty.")
        if isinstance(layer, KVCache):
            _validate_kv_cache(
                k=layer.k,
                v=layer.v,
                length=layer.length,
                seen=layer.seen,
            )
            state[key] = {
                "kind": "full_attention",
                "k": layer.k.detach().clone(),
                "v": layer.v.detach().clone(),
                "length": layer.length,
                "seen": layer.seen,
            }
        elif (delta_cache := _delta_cache(layer)) is not None:
            _validate_delta_cache(delta_cache)
            state[key] = {"kind": "linear_attention", **_tensor_dict(delta_cache)}
        else:
            raise TypeError(
                "A Qwen3.5 cache layer must be KV or delta attention state.",
            )
    return state


def cache_from_state_dict(state: object) -> dict[DepthIndex, object]:
    """Restore an independent mutable Qwen3.5 inference snapshot.

    Restored caches do not preserve behavioral cache subclasses. Their tensors
    do not alias the caller-owned serialized state or another restoration.

    Args:
      state: Cache tensors plus primitive metadata from ``cache_state_dict``.

    Returns:
      cache: Live caches accepted by ``Qwen35.forward`` with ``cache=``.

    Raises:
      TypeError: State containers, keys, or values have unsupported types.
      ValueError: A cache layer kind or field set is invalid, or its tensors or
        progress metadata are incompatible.

    """
    if not _is_object_dict(state):
        raise TypeError("Qwen3.5 cache state must be a dictionary.")
    state_dict: dict[object, object] = state
    cache: dict[DepthIndex, object] = {}
    for key, raw_layer in state_dict.items():
        if not isinstance(key, tuple) or not key:
            raise TypeError("Qwen3.5 cache keys must be nonempty depth indices.")
        layer = _string_keyed_dict(raw_layer)
        kind = layer.get("kind")
        if not isinstance(kind, str):
            raise TypeError("kind must be a str.")
        if kind == "full_attention":
            if set(layer) != {"kind", "k", "v", "length", "seen"}:
                raise ValueError("A full-attention cache has an invalid field set.")
            k = _tensor(layer, name="k")
            v = _tensor(layer, name="v")
            length = _integer(layer, name="length")
            seen = _integer(layer, name="seen")
            _validate_kv_cache(k=k, v=v, length=length, seen=seen)
            restored = KVCache(
                k=k.detach().clone(),
                v=v.detach().clone(),
                length=length,
            )
            restored.seen = seen
            cache[key] = restored
        elif kind == "linear_attention":
            tensors: dict[str, Tensor] = {}
            for name, value in layer.items():
                if name == "kind":
                    continue
                if not isinstance(value, Tensor):
                    raise TypeError(
                        "A delta-attention cache must map strings to tensors.",
                    )
                tensors[name] = value
            if set(tensors) not in (set(), {"conv_state", "recurrent_state"}):
                raise ValueError("A delta-attention cache has an invalid field set.")
            _validate_delta_cache(tensors)
            cache[key] = _tensor_dict(tensors)
        else:
            raise ValueError("A Qwen3.5 cache layer has an unknown kind.")
    return cache


def _validate_kv_cache(
    *,
    k: Tensor,
    v: Tensor,
    length: int,
    seen: int,
) -> None:
    """Validate native full-attention cache tensors and progress metadata."""
    if isinstance(length, bool) or isinstance(seen, bool):
        raise TypeError("Full-attention cache progress metadata must be integers.")
    if (
        k.ndim < 3
        or k.shape[:-1] != v.shape[:-1]
        or k.dtype != v.dtype
        or k.device != v.device
    ):
        raise ValueError(
            "Full-attention key and value caches have incompatible shapes, "
            "dtypes, or devices.",
        )
    if length < 0 or length > k.shape[-2] or seen < length:
        raise ValueError("Full-attention cache progress metadata is invalid.")


# Checks only what the snapshot alone determines. Widths and the conv dtype depend on
# the layer it is resumed into, which ``Qwen35GatedDeltaNet`` checks on first use.
def _validate_delta_cache(state: dict[str, Tensor]) -> None:
    """Validate native delta-attention cache tensors against each other."""
    if not state:
        return
    conv_state = state["conv_state"]
    recurrent_state = state["recurrent_state"]
    if (
        conv_state.ndim != 3
        or recurrent_state.ndim != 4
        or conv_state.shape[0] != recurrent_state.shape[0]
        or not conv_state.is_floating_point()
        or recurrent_state.dtype != torch.float32
        or conv_state.device != recurrent_state.device
        or conv_state.layout != torch.strided
        or recurrent_state.layout != torch.strided
    ):
        raise ValueError(
            "Delta-attention conv and recurrent states have incompatible shapes, "
            "dtypes, or devices.",
        )


def _delta_cache(value: object) -> dict[str, Tensor] | None:
    """Return one validated native delta-attention cache, when present."""
    if not _is_object_dict(value):
        return None
    if not value:
        return {}
    if value.keys() != {"conv_state", "recurrent_state"}:
        return None
    conv_state = value["conv_state"]
    recurrent_state = value["recurrent_state"]
    if not isinstance(conv_state, Tensor) or not isinstance(recurrent_state, Tensor):
        return None
    return {
        "conv_state": conv_state,
        "recurrent_state": recurrent_state,
    }


def _tensor_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Clone a delta-attention cache snapshot."""
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def _string_keyed_dict(value: object) -> dict[str, object]:
    """Validate one serialized cache-layer dictionary."""
    if not _is_object_dict(value):
        raise TypeError("Qwen3.5 cache layer must be a dictionary.")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Qwen3.5 cache layer keys must be strings.")
        result[key] = item
    return result


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    """Return whether a value is a dictionary with runtime keys and values."""
    return isinstance(value, dict)


def _tensor(state: dict[str, object], *, name: str) -> Tensor:
    """Return one required tensor field."""
    value = state[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    return value


def _integer(state: dict[str, object], *, name: str) -> int:
    """Return one required integer field."""
    value = state[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    return value
