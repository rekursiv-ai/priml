from collections.abc import Callable

from torchao.core.config import AOBaseConfig

import torch

_QUANTIZE_CONFIG_HANDLER: dict[
    type[AOBaseConfig],
    Callable[[torch.nn.Module, AOBaseConfig], torch.nn.Module],
] = ...

def register_quantize_module_handler(config_type):  # -> _Wrapped[..., Any, ..., Any]:
    ...
