from torch import nn
from torch.ao.pruning import BaseSparsifier

__all__ = ["WandaSparsifier"]

class WandaSparsifier(BaseSparsifier):
    def __init__(
        self,
        sparsity_level: float = ...,
        semi_structured_block_size: int | None = ...,
    ) -> None: ...
    def prepare(self, model: nn.Module, config: list[dict]) -> None: ...
    def update_mask(
        self, module: nn.Module, tensor_name: str, sparsity_level: float, **kwargs
    ) -> None: ...
    def squash_mask(
        self,
        params_to_keep: tuple[str, ...] | None = ...,
        params_to_keep_per_layer: dict[str, tuple[str, ...]] | None = ...,
        *args,
        **kwargs,
    ):  # -> None:
        ...
