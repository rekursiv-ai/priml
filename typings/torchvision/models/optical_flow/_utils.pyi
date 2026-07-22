from torch import Tensor

def grid_sample(
    img: Tensor,
    absolute_grid: Tensor,
    mode: str = ...,
    align_corners: bool | None = ...,
):  # -> Tensor:
    ...
def make_coords_grid(batch_size: int, h: int, w: int, device: str = ...):  # -> Tensor:
    ...
def upsample_flow(flow, up_mask: Tensor | None = ..., factor: int = ...):  # -> Tensor:
    ...
