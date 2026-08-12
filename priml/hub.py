"""Model hub utilities for loading models with consistent caching.

Provides unified interfaces for loading models from HuggingFace, torch.hub, etc.
with consistent caching behavior across all users.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import logging

import torch

from priml.lib.userdirs import cache_dir


if TYPE_CHECKING:
    # safetensors ships partially-unknown stubs for this symbol.
    from safetensors.torch import (
        load_file,  # pyright: ignore[reportUnknownVariableType]
    )
else:
    from wrapt import lazy_import

    load_file = lazy_import("safetensors.torch", "load_file")


def get_cache_dir() -> Path:
    """Return the per-user cache directory for downloaded models.

    Model weights are regenerable downloads, not experiment state, so they
    live under ``cache_dir("rekursiv-ai") / "models"`` rather than the
    runtime scratch tree -- the hub is a shared client utility, not a
    ``runtime.py`` derivative. The single overridable cache seam (tests patch
    this function).

    Returns:
      models_dir: Path to the model cache directory (created if absent).

    """
    models_dir = cache_dir("rekursiv-ai") / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def load_transformers_model(
    model_id: str,
    model_class: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    force_redownload: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a model from HuggingFace with consistent caching.

    By default, attempts to load from cache first (offline mode), then falls back
    to downloading if not cached. This avoids network requests when models are
    already cached.

    Args:
      model_id: HuggingFace model identifier (e.g., "facebook/opt-125m")
      model_class: Transformers model class name string (e.g., "AutoModel",
                   "AutoImageProcessor"). Using string form delays transformers
                   import to avoid CUDA initialization before fork.
      device: Device to load model on (e.g., "cuda", "cpu")
      dtype: Data type to load model in (e.g., torch.float16). Converted to
             torch_dtype parameter for from_pretrained().
      revision: Git revision to use (default: None = latest)
      trust_remote_code: Whether to trust remote code (default: False)
      force_redownload: Force download from internet, skip cache (default: False)
      **kwargs: Additional arguments passed to from_pretrained

    Returns:
      model: Loaded model

    Example:
      model = load_transformers_model(
          "facebook/opt-125m",
          "AutoModel",
          device="cuda",
          dtype=torch.float16,
      )

    """
    # Import transformers lazily to avoid CUDA initialization at module import time.
    # AutoImageProcessor, AutoProcessor, and SiglipVisionModel all initialize CUDA
    # as a side effect of import, which breaks fork-safe multiprocessing.
    # We use inline not `lazy_import` because `transformers` replaces its own
    # sys.modules entry during init, which is incompatible with LazyLoader.
    import transformers  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    model_class_type = getattr(transformers, model_class)

    cache_dir = get_cache_dir() / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    # ``local_files_only`` controls offline/online behavior per call, so we
    # never mutate ``os.environ["HF_HUB_OFFLINE"]`` -- that env var is
    # process-global, and concurrent loads would interleave and clobber each
    # other's offline state.
    if force_redownload:
        logger.info(f"Force redownloading {model_id}")
        model = model_class_type.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_dir=str(cache_dir),
            local_files_only=False,
            force_download=True,
            **kwargs,
        )
    else:
        try:
            logger.debug(f"Attempting to load {model_id} from cache (offline)")
            model = model_class_type.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=trust_remote_code,
                cache_dir=str(cache_dir),
                local_files_only=True,
                force_download=False,
                **kwargs,
            )
            logger.debug(f"Loaded {model_id} from cache")
        except (OSError, ValueError) as e:
            logger.info(f"Cache miss for {model_id}, downloading from HuggingFace")
            logger.debug(f"Cache miss reason: {e}")
            model = model_class_type.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=trust_remote_code,
                cache_dir=str(cache_dir),
                local_files_only=False,
                force_download=False,
                **kwargs,
            )

    if device is not None:
        model = model.to(device)

    return model


def resolve_hf_dtype(name: str) -> torch.dtype:
    """Map an HF ``torch_dtype`` string to a ``torch.dtype``.

    Args:
      name: Dtype name from HF ``config.json`` (e.g. ``"bfloat16"``).

    Returns:
      dtype: Corresponding ``torch.dtype``; defaults to ``float32``.

    """
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.float32)


def load_local_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load sharded safetensors / pytorch_model.bin from a local dir.

    Args:
      path: Directory containing weight shards.

    Returns:
      state_dict: Merged state dict from all shards.

    """
    safetensors = sorted(path.glob("*.safetensors"))
    if safetensors:
        sd: dict[str, torch.Tensor] = {}
        for shard in safetensors:
            sd.update(load_file(str(shard)))
        return sd
    pt = path / "pytorch_model.bin"
    if pt.exists():
        # torch.load is annotated `-> Any`; weights_only=True guarantees tensors.
        loaded: dict[str, torch.Tensor] = torch.load(
            str(pt), map_location="cpu", weights_only=True
        )
        return loaded
    shards = sorted(path.glob("pytorch_model-*.bin"))
    if shards:
        pt_sd: dict[str, torch.Tensor] = {}
        for shard in shards:
            pt_sd.update(
                torch.load(str(shard), map_location="cpu", weights_only=True),
            )
        return pt_sd
    raise FileNotFoundError(
        f"No safetensors or pytorch_model weights found in {path}.",
    )
