from dataclasses import dataclass
from typing import Any

from fla.models.mom.configuration_mom import MomConfig
from fla.models.utils import Cache, FLAGenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack

import torch

logger = ...

def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = ...,
    top_k=...,
    attention_mask: torch.Tensor | None = ...,
) -> torch.Tensor | int: ...

class MomBlock(GradientCheckpointingLayer):
    def __init__(self, config: MomConfig, layer_idx: int) -> None: ...
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
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class MomPreTrainedModel(PreTrainedModel):
    config_class = MomConfig
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

@dataclass
class MomOutputWithPast(BaseModelOutputWithPast):
    router_logits: tuple[torch.FloatTensor, ...] | None = ...

class MomModel(MomPreTrainedModel):
    def __init__(self, config: MomConfig) -> None: ...
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
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | BaseModelOutputWithPast: ...

@dataclass
class MomCausalLMOutputWithPast(CausalLMOutputWithPast):
    aux_loss: torch.FloatTensor | None = ...
    router_logits: tuple[torch.FloatTensor, ...] | None = ...

class MomForCausalLM(MomPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self) -> Embedding | Module: ...
    def set_input_embeddings(self, value) -> None: ...
    def get_output_embeddings(self) -> Linear: ...
    def set_output_embeddings(self, new_embeddings) -> None: ...
    def set_decoder(self, decoder) -> None: ...
    def get_decoder(self) -> MomModel: ...
    def generate(self, *args, **kwargs): ...
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        num_logits_to_keep: int | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple | CausalLMOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | CausalLMOutputWithPast: ...
