from typing import NamedTuple

import torch

"""
Defines an nn module designed to be used during inference
"""
Tensor = torch.Tensor

class Float8MMConfig(NamedTuple):
    emulate: bool = ...
    use_fast_accum: bool = ...
    pad_inner_dim: bool = ...

def preprocess_data(
    a_data: Tensor, b_data: Tensor, scaled_mm_config: Float8MMConfig
) -> tuple[Tensor, Tensor]: ...
def preprocess_scale(
    input_scale: torch.Tensor, input_shape: tuple[int, ...]
):  # -> Tensor:
    ...
def addmm_float8_unwrapped_inference(
    a_data: Tensor,
    a_scale: Tensor,
    b_data: Tensor,
    b_scale: Tensor,
    output_dtype: torch.dtype,
    output_scale: Tensor | None = ...,
    bias: Tensor | None = ...,
    use_fast_accum: bool = ...,
) -> Tensor: ...
