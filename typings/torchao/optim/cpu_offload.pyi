from torch.optim.optimizer import Optimizer, ParamsT

import torch

class CPUOffloadOptimizer(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        optimizer_class: type[Optimizer] = ...,
        *,
        offload_gradients: bool = ...,
        minimal_size: int = ...,
        **kwargs,
    ) -> None: ...
    @torch.no_grad()
    def step(self, closure=...):  # -> None:
        ...
    def zero_grad(self, set_to_none=...):  # -> None:
        ...
    @property
    def param_groups(self):  # -> list[Any]:
        ...
    def state_dict(self):  # -> dict[str, list[Any]]:
        ...
    def load_state_dict(self, state_dict):  # -> None:
        ...
