"""Frozen Qwen features built from Priml's native transformer components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import gc

from configgle import Fig, Makeable, Makes
from torch import Tensor
from treekle import from_plain

import numpy as np
import torch

from priml import hub
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaFused
from priml.model.attention.rope import RoPE, rotate_activation_dtype
from priml.model.custom_types import AttentionKernel, TensorModule
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3 import Qwen3, remap_hf_state_dict


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers.models.auto import tokenization_auto
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
else:
    from wrapt import lazy_import

    tokenization_auto = lazy_import("transformers.models.auto.tokenization_auto")


class QwenRMSNorm(RMSNorm):
    """Preserve Qwen's float32 operation boundaries with Priml's RMS parameters."""

    class Config(Makes["QwenRMSNorm"], RMSNorm.Config):
        """The existing RMSNorm fields and parameter layout."""

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        if input.shape[-1:] != self.normalized_shape:
            raise RuntimeError(f"Expected normalized_shape={self.normalized_shape}.")
        value = input.float()
        eps = torch.finfo(input.dtype).eps if self.eps is None else self.eps
        inverse_rms = torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps)
        normalized = (value * inverse_rms).to(input.dtype)
        return normalized if self.weight is None else self.weight * normalized


class FrozenQwen:
    """Extract all four layers using native Priml models and external weights."""

    class Config(Fig["FrozenQwen"]):
        """Pinned model location and tokenization/extraction settings."""

        directory: Path | str = "/datasets/name-that-llm/qwen-3-8b-base-v1"
        """Unmodified qwen-lm/qwen-3/Transformers/8b-base/1 snapshot."""

        layers: list[int] = field(default_factory=list[int])
        """Saved hidden-state indices; embedding state has index zero."""

        batch_size: int = 1
        """Longest-first extraction batch size."""

        max_length: int = 512
        """Right truncation length without added special tokens."""

        channels_hidden: int = 4096
        """Expected hidden width of the pinned model."""

        dtype: torch.dtype = torch.float32
        """Initial frozen-weight precision."""

        retry_dtype: torch.dtype = torch.bfloat16
        """Precision used only to recompute nonfinite rows."""

        device: str = "cpu"
        """Single device holding the frozen native model."""

        norm: Makeable[TensorModule] = field(default_factory=QwenRMSNorm.Config)
        """Block and final normalization; the checkpoint sets width and epsilon."""

        rotate: Callable[[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]] = (
            rotate_activation_dtype
        )
        """Rotation applied to queries and keys in every block."""

        split_qkv_projection: bool = True
        """Keep the query, key and value projections as the checkpoint's three."""

        split_gate_projection: bool = True
        """Keep the SwiGLU gate and up projections as the checkpoint's two."""

        attention_kernel: Makeable[AttentionKernel] = field(
            default_factory=SdpaFused.Config,
        )
        """Native attention kernel; the CPU golden injects unfused attention."""

        release_model: Callable[[], object] = gc.collect
        """Collection policy after releasing each frozen-weight instance."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Extract embeddings and retry nonfinite fp16 rows in bf16.

        Args:
          texts: Training passages followed by prediction passages.

        Returns:
          embeddings: Float32 rows by saved layers by hidden width.

        Raises:
          FileNotFoundError: The model snapshot has not been downloaded.
          RuntimeError: Retry rows remain nonfinite.

        """
        config = self.config
        directory = Path(config.directory)
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"Pinned Qwen snapshot missing at {directory}.")
        tokenizer = tokenization_auto.AutoTokenizer.from_pretrained(
            directory,
            local_files_only=True,
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        lengths = np.array(
            [
                len(ids)
                for ids in from_plain(
                    tokenizer(texts, add_special_tokens=False)["input_ids"],
                    target=list[list[int]],
                )
            ],
        )
        order = np.argsort(-lengths, kind="stable")

        embeddings = np.zeros(
            (len(texts), len(config.layers), config.channels_hidden),
            dtype=np.float32,
        )
        embeddings[order] = self._encode_rows(
            order,
            dtype=config.dtype,
            texts=texts,
            tokenizer=tokenizer,
        )
        bad = np.where(~np.isfinite(embeddings).all(axis=(1, 2)))[0]
        if len(bad):
            embeddings[bad] = self._encode_rows(
                bad,
                dtype=config.retry_dtype,
                texts=texts,
                tokenizer=tokenizer,
            )
        finite = np.isfinite(embeddings).all()
        if not finite:
            raise RuntimeError("Qwen embeddings remain nonfinite after bf16 retry.")
        return embeddings

    def _encode_rows(
        self,
        rows: NDArray[np.int64],
        dtype: torch.dtype,
        *,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
    ) -> NDArray[np.float32]:
        config = self.config
        model = self._load_model(dtype)
        first = next(model.parameters()).device
        output = np.zeros(
            (len(rows), len(config.layers), config.channels_hidden),
            dtype=np.float32,
        )
        for start in range(0, len(rows), config.batch_size):
            indices = rows[start : start + config.batch_size]
            encoded = tokenizer(
                [texts[index] for index in cast("list[int]", indices.tolist())],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
                add_special_tokens=False,
            ).to(first)
            tensors = from_plain(encoded.data, target=dict[str, Tensor])
            with torch.inference_mode():
                states = selected_hidden_states(
                    model,
                    input_ids=tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                    layers=config.layers,
                )
                mask = tensors["attention_mask"]
                for position, hidden in enumerate(states):
                    output[start : start + len(indices), position] = (
                        mean_pool(hidden, attention_mask=mask).cpu().numpy()
                    )
        del model
        config.release_model()
        torch.cuda.empty_cache()
        return output

    def _load_model(self, dtype: torch.dtype) -> Qwen3:
        """Materialize external tensors directly into Priml's native model."""
        metadata, state = hub.load_hf_checkpoint(
            Path(self.config.directory),
            dtype=dtype,
        )
        config = Qwen3.Config.from_hf(metadata).finalize()
        if config.channels_in != self.config.channels_hidden:
            raise ValueError("Qwen checkpoint width differs from channels_hidden.")
        if not self.config.layers or any(
            layer < 0 or layer > config.num_layers for layer in self.config.layers
        ):
            raise ValueError("Saved hidden-state indices exceed the Qwen stack.")
        assert isinstance(config.block, list)
        for block in config.block:
            assert isinstance(block, TransformerBlock.Config)
            assert isinstance(block.attn, Attention.Config)
            assert isinstance(block.ffn, SwiGLU.Config)
            assert isinstance(block.attn.rope, RoPE.Config)
            block.attn.rotate = self.config.rotate
            block.attn.split_qkv_projection = self.config.split_qkv_projection
            block.attn.attn_kernel = self.config.attention_kernel
            block.ffn.split_gate_projection = self.config.split_gate_projection
            block.norm1 = self.config.norm.copy_tree().update(block.norm1)
            block.norm2 = self.config.norm.copy_tree().update(block.norm2)
            block.attn.norm_qk = self.config.norm.copy_tree().update(
                block.attn.norm_qk,
            )
        assert isinstance(config.proj_out, Sequential.Config)
        assert isinstance(config.proj_out.elements, list)
        config.proj_out.elements[0] = self.config.norm.copy_tree().update(
            config.proj_out.elements[0],
        )
        # Allocating on meta avoids allocating and initializing eight billion
        # unused random parameters before assigning the acquired checkpoint.
        with torch.device("meta"):
            model = config.make()
        for module in model.modules():
            if isinstance(module, RoPE):
                module.to_empty(device="cpu")
        model.load_state_dict(remap_hf_state_dict(state, config), assign=True)
        return (
            model.to(device=self.config.device, dtype=dtype)
            .eval()
            .requires_grad_(False)
        )


def selected_hidden_states(
    model: Qwen3,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    layers: list[int],
) -> tuple[Tensor, ...]:
    """Return embedding/block states, including final normalization at stack end.

    Args:
      model: Native Priml transformer holding the acquired weights.
      input_ids: Right-padded token identifiers, batch by sequence.
      attention_mask: One for real tokens, zero for padding.
      layers: Saved-state indices; zero denotes the token embeddings.

    Returns:
      states: Complete selected tensors in the requested order.

    """
    if model.proj_in is None:
        raise ValueError("Expected model.proj_in is not None.")
    assert isinstance(model.proj_out, Sequential)
    final_norm = model.proj_out[0]
    assert isinstance(final_norm, RMSNorm)
    hidden = model.proj_in(input_ids)
    length = input_ids.shape[1]
    causal = torch.ones(length, length, device=hidden.device, dtype=torch.bool).tril()
    allowed = causal[None, None] & attention_mask[:, None, None].bool()
    mask = torch.zeros(allowed.shape, device=hidden.device, dtype=hidden.dtype)
    mask = mask.masked_fill(~allowed, torch.finfo(hidden.dtype).min)
    selected: dict[int, Tensor] = {0: hidden} if 0 in layers else {}
    for index, block in enumerate(model.blocks, start=1):
        assert isinstance(block, TransformerBlock)
        hidden = block(hidden, attn_mask=mask, is_causal=False)
        if index == model.num_layers:
            hidden = final_norm(hidden)
        if index in layers:
            selected[index] = hidden
    return tuple(selected[layer] for layer in layers)


def mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool in float32, preserving the reference's multiply/sum/divide order."""
    hidden = hidden.float()
    mask = attention_mask.unsqueeze(-1).to(hidden.device).float()
    return (hidden * mask).sum(1) / mask.sum(1)
