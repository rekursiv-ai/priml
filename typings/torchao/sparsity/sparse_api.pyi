from collections.abc import Callable
from dataclasses import dataclass

from torchao.core.config import AOBaseConfig

import torch

def apply_fake_sparsity(model, **kwargs):  # -> None:
    ...

@dataclass
class BlockSparseWeightConfig(AOBaseConfig):
    blocksize: int = ...
    def __post_init__(self):  # -> None:
        ...

block_sparse_weight = BlockSparseWeightConfig

class SemiSparseWeightConfig(AOBaseConfig):
    def __post_init__(self):  # -> None:
        ...

semi_sparse_weight = SemiSparseWeightConfig

def sparsify_(
    model: torch.nn.Module,
    config: AOBaseConfig,
    filter_fn: Callable[[torch.nn.Module, str], bool] | None = ...,
) -> torch.nn.Module: ...
