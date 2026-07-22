from fla.models.deltaformer.configuration_deltaformer import DeltaFormerConfig
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

class DeltaFormerBlock(GradientCheckpointingLayer):
    def __init__(self, config: DeltaFormerConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class DeltaFormerPreTrainedModel(PreTrainedModel):
    config_class = DeltaFormerConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_cache_class = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

class DeltaFormerModel(DeltaFormerPreTrainedModel):
    def __init__(self, config: DeltaFormerConfig) -> None: ...
    def get_input_embeddings(self) -> Embedding | Module: ...
    def set_input_embeddings(self, value) -> None: ...
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
        **kwargs: Unpack[dict],
    ) -> tuple | BaseModelOutputWithPast: ...

class DeltaFormerForCausalLM(DeltaFormerPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: DeltaFormerConfig) -> None: ...
    def get_input_embeddings(self) -> Embedding | Module: ...
    def set_input_embeddings(self, value) -> None: ...
    def tie_weights(self, *args, **kwargs) -> None: ...
    def get_output_embeddings(self) -> Linear: ...
    def set_output_embeddings(self, new_embeddings) -> None: ...
    def set_decoder(self, decoder) -> None: ...
    def get_decoder(self) -> DeltaFormerModel: ...
    def generate(self, *args, **kwargs): ...
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
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
        labels: torch.LongTensor | None = ...,
        logits_to_keep: int | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple | CausalLMOutputWithPast: ...
