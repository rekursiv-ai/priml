from fla.utils import tensor_cache

import torch

_LAYER_IDX_REQUIRED_MSG = ...

class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices) -> Tensor: ...
    @staticmethod
    def backward(ctx, do) -> tuple[Tensor, None]: ...

index_first_axis = ...

class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, first_axis_dim) -> Tensor: ...
    @staticmethod
    def backward(ctx, do) -> tuple[Any, None, None]: ...

index_put_first_axis = ...

@tensor_cache
def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]: ...
def unpad_input(
    q: torch.Tensor,
    states: tuple[torch.Tensor],
    attention_mask: torch.Tensor,
    q_len: int,
    keepdim: bool = ...,
) -> tuple[
    Tensor,
    tuple[Any, ...] | tuple[Any | None, ...],
    Any | Tensor,
    tuple[Any | Tensor, Any],
    tuple[Any | int, Any],
]: ...
def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor: ...
def require_cache_layer_idx(module, past_key_values) -> Any | None: ...
def get_layer_cache(module, past_key_values) -> None: ...
def update_layer_cache(module, past_key_values, **kwargs) -> None: ...
