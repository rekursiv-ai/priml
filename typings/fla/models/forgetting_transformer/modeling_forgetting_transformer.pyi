from typing import Any

from fla.models.forgetting_transformer.configuration_forgetting_transformer import (
    ForgettingTransformerConfig,
)
from fla.models.utils import Cache, FLAGenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg

import torch

logger = ...

class ForgettingTransformerBlock(GradientCheckpointingLayer):
    def __init__(self, config: ForgettingTransformerConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: tuple[torch.Tensor] | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[Any],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class ForgettingTransformerPreTrainedModel(PreTrainedModel):
    config_class = ForgettingTransformerConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_cache_class = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

class ForgettingTransformerModel(ForgettingTransformerPreTrainedModel):
    def __init__(
        self, config: ForgettingTransformerConfig
    ) -> ForgettingTransformerModel: ...
    def get_input_embeddings(self) -> Embedding | Module: ...
    def set_input_embeddings(self, value) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | CausalLMOutputWithPast: ...

class ForgettingTransformerForCausalLM(
    ForgettingTransformerPreTrainedModel, FLAGenerationMixin
):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self) -> Embedding | Module: ...
    def set_input_embeddings(self, value) -> None: ...
    def get_output_embeddings(self) -> Linear: ...
    def set_output_embeddings(self, new_embeddings) -> None: ...
    def set_decoder(self, decoder) -> None: ...
    def get_decoder(self) -> ForgettingTransformerModel: ...
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        logits_to_keep: int | None = ...,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast: ...
