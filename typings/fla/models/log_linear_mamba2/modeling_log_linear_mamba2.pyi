from fla.models.log_linear_mamba2.configuration_log_linear_mamba2 import (
    LogLinearMamba2Config,
)
from fla.models.utils import Cache, FLAGenerationMixin
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.deprecation import deprecate_kwarg

import torch

logger = ...

class LogLinearMamba2Block(nn.Module):
    def __init__(self, config: LogLinearMamba2Config, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[Tensor, Any, Cache | list[FloatTensor] | None]: ...

class LogLinearMamba2PreTrainedModel(PreTrainedModel, FLAGenerationMixin):
    config_class = LogLinearMamba2Config
    base_model_prefix = ...
    _no_split_modules = ...
    supports_gradient_checkpointing = ...
    _supports_cache_class = ...

class LogLinearMamba2Model(LogLinearMamba2PreTrainedModel):
    def __init__(self, config) -> None: ...
    def load_hook(self, state_dict, prefix, *args) -> None: ...
    def get_input_embeddings(self) -> Embedding: ...
    def set_input_embeddings(self, new_embeddings) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast: ...

class LogLinearMamba2ForCausalLM(LogLinearMamba2PreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self) -> Linear: ...
    def set_output_embeddings(self, new_embeddings) -> None: ...
    def get_input_embeddings(self) -> Embedding: ...
    def set_input_embeddings(self, new_embeddings) -> None: ...
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        logits_to_keep: int | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast: ...
