from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from fla.models.utils import Cache, FLAGenerationMixin
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg

import torch

logger = ...

class RWKV7FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: int | None = ...,
        intermediate_size: int | None = ...,
        hidden_act: str = ...,
        layer_idx: int = ...,
        num_hidden_layers: int = ...,
    ) -> RWKV7FeedForward: ...
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        state: Cache | None = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class RWKV7Block(GradientCheckpointingLayer):
    def __init__(self, config: RWKV7Config, layer_idx: int) -> RWKV7Block: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        v_first: torch.Tensor = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class RWKV7PreTrainedModel(PreTrainedModel):
    config_class = RWKV7Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_cache_class = ...
    _skip_keys_device_placement = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

class RWKV7Model(RWKV7PreTrainedModel):
    def __init__(self, config: RWKV7Config) -> None: ...
    def get_input_embeddings(self) -> Embedding | Module: ...
    def set_input_embeddings(self, value) -> None: ...
    def load_state_dict(self, state_dict, strict=..., assign=...) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple | BaseModelOutputWithPast: ...

class RWKV7ForCausalLM(RWKV7PreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self) -> Embedding | Module: ...
    def set_input_embeddings(self, value) -> None: ...
    def get_output_embeddings(self) -> Linear: ...
    def set_output_embeddings(self, new_embeddings) -> None: ...
    def set_decoder(self, decoder) -> None: ...
    def get_decoder(self) -> RWKV7Model: ...
    def generate(self, *args, **kwargs): ...
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        labels: torch.LongTensor | None = ...,
        shift_labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        logits_to_keep: int | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple | CausalLMOutputWithPast: ...
