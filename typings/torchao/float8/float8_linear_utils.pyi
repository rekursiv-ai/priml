from collections.abc import Callable

from torch import nn
from torchao.float8.config import Float8LinearConfig

log = ...

def swap_linear_layers(
    module: nn.Module,
    from_float_func: Callable[[nn.Linear], nn.Linear],
    *,
    module_filter_fn: Callable[[nn.Module, str], bool] | None = ...,
) -> nn.Module: ...
def convert_to_float8_training(
    module: nn.Module,
    *,
    module_filter_fn: Callable[[nn.Module, str], bool] | None = ...,
    config: Float8LinearConfig | None = ...,
) -> nn.Module: ...
