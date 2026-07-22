from typing import Any

from torch import nn
from torchao.float8.float8_training_tensor import LinearMMConfig

import torch

@torch.no_grad()
def precompute_float8_dynamic_scale_for_fsdp(module: nn.Module) -> None: ...

_ops_to_preserve_subclass = ...

class WeightWithDynamicFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        precomputed_scale: torch.Tensor | None = ...,
    ): ...
    def __init__(
        self,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        precomputed_scale: torch.Tensor | None = ...,
    ) -> None: ...
    @classmethod
    def __torch_dispatch__(
        cls, func, types, args, kwargs=...
    ):  # -> WeightWithDynamicFloat8CastTensor | PyTree:
        ...
    def __tensor_flatten__(
        self,
    ):  # -> tuple[list[str], dict[str, LinearMMConfig | dtype]]:
        ...
    @staticmethod
    def __tensor_unflatten__(
        inner_tensors, flatten_spec, outer_size, outer_stride
    ):  # -> WeightWithDynamicFloat8CastTensor:
        ...
    def __repr__(self):  # -> str:
        ...
    def fsdp_pre_all_gather(
        self, mesh
    ):  # -> tuple[tuple[Any | Tensor], tuple[Any | Tensor]]:
        ...
    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = ...,
    ):  # -> tuple[Float8TrainingTensor, tuple[Tensor]] | None:
        ...
