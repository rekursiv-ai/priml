"""Safe persistence boundary for native Qwen3.5 mixed attention caches."""

from collections.abc import Sequence
from typing import TypeGuard

from torch import Tensor

from priml.model.attention.kvcache import KVCache


type Qwen35CacheState = list[dict[str, Tensor | int | str]]


def cache_state_dict(cache: Sequence[object]) -> Qwen35CacheState:
    """Convert a live Qwen3.5 cache to tensors and primitive metadata.

    The returned tensors and primitive metadata can be passed to ``torch.save``
    and restored with ``torch.load(weights_only=True)``.

    Args:
      cache: Live attention caches returned by ``Qwen35.alloc_cache``.

    Returns:
      state: Cache tensors plus primitive progress metadata for safe serialization.

    Raises:
      TypeError: A layer cache is not native Qwen3.5 attention state.
      ValueError: A full-attention cache has incompatible tensors or progress
        metadata.

    """
    state: Qwen35CacheState = []
    for layer in cache:
        if isinstance(layer, KVCache):
            _validate_kv_cache(
                k=layer.k,
                v=layer.v,
                length=layer.length,
                seen=layer.seen,
            )
            state.append(
                {
                    "kind": "full_attention",
                    "k": layer.k.detach().clone(),
                    "v": layer.v.detach().clone(),
                    "length": layer.length,
                    "seen": layer.seen,
                }
            )
        elif (delta_cache := _delta_cache(layer)) is not None:
            state.append({"kind": "linear_attention", **_tensor_dict(delta_cache)})
        else:
            raise TypeError(
                "A Qwen3.5 cache layer must be KV or delta attention state."
            )
    return state


def cache_from_state_dict(state: object) -> list[object]:
    """Restore an independent mutable Qwen3.5 inference snapshot.

    Restored caches do not preserve behavioral cache subclasses. Their tensors
    do not alias the caller-owned serialized state or another restoration.

    Args:
      state: Cache tensors plus primitive metadata from ``cache_state_dict``.

    Returns:
      cache: Live caches accepted by ``Qwen35.forward_cached``.

    Raises:
      TypeError: State containers, keys, or values have unsupported types.
      ValueError: A cache layer kind or field set is invalid, or full-attention
        tensors or progress metadata are incompatible.

    """
    if not _is_object_list(state):
        raise TypeError("Qwen3.5 cache state must be a list.")
    cache: list[object] = []
    for raw_layer in state:
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
            cache.append(restored)
        elif kind == "linear_attention":
            tensors: dict[str, Tensor] = {}
            for name, value in layer.items():
                if name == "kind":
                    continue
                if not isinstance(value, Tensor):
                    raise TypeError(
                        "A delta-attention cache must map strings to tensors."
                    )
                tensors[name] = value
            if set(tensors) not in (set(), {"conv_state", "recurrent_state"}):
                raise ValueError("A delta-attention cache has an invalid field set.")
            cache.append(_tensor_dict(tensors))
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
        or v.ndim < 3
        or k.shape[:-1] != v.shape[:-1]
        or k.dtype != v.dtype
        or k.device != v.device
    ):
        raise ValueError(
            "Full-attention key and value caches have incompatible shapes, "
            "dtypes, or devices."
        )
    if length < 0 or length > k.shape[-2] or seen < length:
        raise ValueError("Full-attention cache progress metadata is invalid.")


def _delta_cache(value: object) -> dict[str, Tensor] | None:
    """Return one validated native delta-attention cache, when present."""
    if not isinstance(value, dict):
        return None
    match value:
        case {} if not value:
            return {}
        case {
            "conv_state": Tensor() as conv_state,
            "recurrent_state": Tensor() as recurrent_state,
        } if value.keys() == {"conv_state", "recurrent_state"}:
            return {
                "conv_state": conv_state,
                "recurrent_state": recurrent_state,
            }
        case dict():
            return None


def _tensor_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Clone a delta-attention cache snapshot."""
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    """Return whether a value is a list whose members are runtime objects."""
    return isinstance(value, list)


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
