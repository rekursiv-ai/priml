from torch import Tensor
from torch.optim import Optimizer

import torch

class _AdamBase(Optimizer):
    def __init__(
        self,
        params,
        lr,
        betas,
        eps,
        weight_decay,
        amsgrad,
        *,
        block_size,
        bf16_stochastic_round,
        is_adamw,
    ) -> None: ...
    def add_param_group(self, param_group: dict) -> None: ...
    def __setstate__(self, state):  # -> None:
        ...
    @torch.no_grad()
    def step(self, closure=...):  # -> None:
        ...

def single_param_adam(
    p: Tensor,
    grad: Tensor,
    step: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Tensor | None,
    lr: Tensor,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    IS_ADAMW: bool,
    BF16_STOCHASTIC_ROUND: bool,
):  # -> None:
    ...

class Adam8bit(_AdamBase):
    def __init__(
        self,
        params,
        lr=...,
        betas=...,
        eps=...,
        weight_decay=...,
        amsgrad=...,
        *,
        block_size=...,
        bf16_stochastic_round=...,
    ) -> None: ...

class Adam4bit(_AdamBase):
    def __init__(
        self,
        params,
        lr=...,
        betas=...,
        eps=...,
        weight_decay=...,
        amsgrad=...,
        *,
        block_size=...,
        bf16_stochastic_round=...,
    ) -> None: ...

class AdamFp8(_AdamBase):
    def __init__(
        self,
        params,
        lr=...,
        betas=...,
        eps=...,
        weight_decay=...,
        amsgrad=...,
        *,
        block_size=...,
        bf16_stochastic_round=...,
    ) -> None: ...

class AdamW8bit(_AdamBase):
    def __init__(
        self,
        params,
        lr=...,
        betas=...,
        eps=...,
        weight_decay=...,
        amsgrad=...,
        *,
        block_size=...,
        bf16_stochastic_round=...,
    ) -> None: ...

class AdamW4bit(_AdamBase):
    def __init__(
        self,
        params,
        lr=...,
        betas=...,
        eps=...,
        weight_decay=...,
        amsgrad=...,
        *,
        block_size=...,
        bf16_stochastic_round=...,
    ) -> None: ...

class AdamWFp8(_AdamBase):
    def __init__(
        self,
        params,
        lr=...,
        betas=...,
        eps=...,
        weight_decay=...,
        amsgrad=...,
        *,
        block_size=...,
        bf16_stochastic_round=...,
    ) -> None: ...

class _AdamW(_AdamBase):
    def __init__(
        self,
        params,
        lr=...,
        betas=...,
        eps=...,
        weight_decay=...,
        amsgrad=...,
        *,
        bf16_stochastic_round=...,
    ) -> None: ...
