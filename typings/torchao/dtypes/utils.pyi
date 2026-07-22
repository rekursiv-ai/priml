from dataclasses import dataclass

from torchao.utils import TorchAOBaseTensor

import torch

@dataclass(frozen=True)
class Layout:
    def pre_process(self, input: torch.Tensor) -> torch.Tensor: ...
    def post_process(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def pre_process_static(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __repr__(self):  # -> str:
        ...
    def extra_repr(self) -> str: ...
    def __post_init__(self):  # -> None:
        ...

@dataclass(frozen=True)
class PlainLayout(Layout): ...

def is_device(target_device_str: str, device: str | torch.device):  # -> bool:
    ...
def get_out_shape(
    input_shape: tuple[int], weight_shape: tuple[int]
) -> tuple[int, int]: ...

class AQTTensorImpl(TorchAOBaseTensor):
    def get_plain(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
    def get_layout(self) -> Layout: ...
    @classmethod
    def from_plain(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        _layout: Layout,
    ):  # -> None:
        ...
    def __repr__(self):  # -> str:
        ...
