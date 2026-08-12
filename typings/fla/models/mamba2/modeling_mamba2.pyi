from torch import Tensor
from typing import Any
from fla.models.mamba2.configuration_mamba2 import Mamba2Config
from fla.models.utils import Cache, FLAGenerationMixin
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import DeviceMesh
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.deprecation import deprecate_kwarg

import torch

logger = ...

def tensor_to_dtensor(
    tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    current_placement: Placement | list[Placement],
    desired_placement: Placement | list[Placement] | None = ...,
    run_check: bool = ...,
) -> DTensor: ...

class Mamba2Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[Tensor, Any, Cache | list[FloatTensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Tensor, Any, Cache | list[FloatTensor] | None]: ...

class Mamba2PreTrainedModel(PreTrainedModel):
    config_class = Mamba2Config
    base_model_prefix = ...
    _no_split_modules = ...
    supports_gradient_checkpointing = ...
    _supports_cache_class = ...

class Mamba2Model(Mamba2PreTrainedModel):
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
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | BaseModelOutputWithPast: ...

class Mamba2ForCausalLM(Mamba2PreTrainedModel, FLAGenerationMixin):
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
