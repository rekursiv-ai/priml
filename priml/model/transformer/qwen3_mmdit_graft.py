"""Qwen3-backed joint-attention grafts with native checkpoint loading.

Example::

    model = Qwen3MMDiTGraft.load("Qwen/Qwen3-0.6B", device="cuda")
    model.freeze_backbone()
    logits, streams = model(tokens, [modality], attn_mask=[language_mask, None])

The language mask must retain causal language-only visibility to preserve logits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

from configgle import Makes

from priml import hub
from priml.model.custom_types import TransformerConfig
from priml.model.transformer.mmdit_graft import MMDiTGraft
from priml.model.transformer.qwen3 import (
    Qwen3,
    _load_hf_checkpoint,
    remap_hf_state_dict,
)


if TYPE_CHECKING:
    import torch


class Qwen3MMDiTGraft(MMDiTGraft):
    """Add independently trainable modality streams to dense Qwen3."""

    class Config(Makes["Qwen3MMDiTGraft"], MMDiTGraft.Config):
        backbone: TransformerConfig = field(
            default_factory=cast(Callable[[], TransformerConfig], Qwen3.Config)
        )
        """Dense Qwen3 language architecture; the checkpoint supplies this on load."""

    @classmethod
    def load(
        cls,
        path_or_repo: Path | str,
        *,
        config: Config | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Load a dense Qwen3 checkpoint and initialize additional streams.

        Args:
          path_or_repo: Local HF checkpoint directory or HuggingFace repository ID.
          config: Unfinalized graft configuration for the added streams. The
              checkpoint replaces its backbone; the caller's config is unchanged.
          device: Target device; defaults to CPU.
          dtype: Parameter dtype; defaults to the checkpoint's declared dtype.

        Returns:
          model: Graft with strictly loaded language weights and fresh modalities.

        """
        hf_config, hf_state = _load_hf_checkpoint(path_or_repo, dtype=dtype)
        graft = config.copy_tree() if config is not None else cls.Config()
        graft.backbone = Qwen3.Config.from_hf(hf_config)
        graft.finalize()
        assert isinstance(graft.backbone, Qwen3.Config)
        # ``make``, not ``cls(graft)``: a late-bound head (a tied ``lm_head``)
        # is resolved by ``make`` after the whole tree exists.
        model = graft.make()
        assert isinstance(model, cls)
        model.load_backbone_state(remap_hf_state_dict(hf_state, graft.backbone))
        model = model.to(
            dtype=dtype
            or hub.resolve_hf_dtype(str(hf_config.get("torch_dtype", "bfloat16")))
        )
        if device is not None:
            model = model.to(device=device)
        return model
