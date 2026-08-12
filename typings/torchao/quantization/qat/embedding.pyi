from typing import Any

from torchao.quantization.unified import TwoStepQuantizer

import torch

from .fake_quantize_config import FakeQuantizeConfigBase

class FakeQuantizedEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = ...,
        max_norm: float | None = ...,
        norm_type: float = ...,
        scale_grad_by_freq: bool = ...,
        sparse: bool = ...,
        weight_config: FakeQuantizeConfigBase | None = ...,
        *args,
        **kwargs,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def to_embedding(self) -> torch.nn.Embedding: ...
    @classmethod
    def from_embedding(
        cls,
        mod: torch.nn.Embedding,
        weight_config: FakeQuantizeConfigBase | None = ...,
    ):  # -> FakeQuantizedEmbedding:
        ...

class Int4WeightOnlyEmbeddingQATQuantizer(TwoStepQuantizer):
    def __init__(
        self,
        group_size: int = ...,
        scale_precision: torch.dtype = ...,
        zero_point_precision: torch.dtype = ...,
    ) -> None: ...
    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...
    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...

class Int4WeightOnlyQATEmbedding(FakeQuantizedEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = ...,
        max_norm: float | None = ...,
        norm_type: float = ...,
        scale_grad_by_freq: bool = ...,
        sparse: bool = ...,
        group_size: int = ...,
        scale_precision: torch.dtype = ...,
        zero_point_precision: torch.dtype = ...,
        *args,
        **kwargs,
    ) -> None: ...
    def enable_fake_quant(self, enabled: bool = ...):  # -> None:
        ...
    def disable_fake_quant(self):  # -> None:
        ...

class Int4WeightOnlyEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = ...,
        max_norm: float | None = ...,
        norm_type: float = ...,
        scale_grad_by_freq: bool = ...,
        sparse: bool = ...,
        group_size: int = ...,
        scale_precision: torch.dtype = ...,
        zero_point_precision: torch.dtype = ...,
        device: torch.device = ...,
        output_dtype: torch.dtype = ...,
    ) -> None: ...
    def forward(self, x):  # -> Tensor:
        ...
