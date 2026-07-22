from fla.models.samba.configuration_samba import SambaConfig
from fla.models.utils import Cache, FLAGenerationMixin
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

class SambaBlock(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple[Tensor, Any, Cache | list[FloatTensor] | None]: ...

class SambaPreTrainedModel(PreTrainedModel):
    config_class = SambaConfig
    base_model_prefix = ...
    _no_split_modules = ...
    supports_gradient_checkpointing = ...

class SambaModel(SambaPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self) -> Embedding: ...
    def set_input_embeddings(self, new_embeddings) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple | BaseModelOutputWithPast: ...

class SambaForCausalLM(SambaPreTrainedModel, FLAGenerationMixin):
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
        **kwargs: Unpack[dict],
    ) -> tuple | CausalLMOutputWithPast: ...
