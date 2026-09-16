from typing import Any, Literal, overload

from torch import Tensor, nn

def spatial_average(in_tens, keepdim=...): ...
def upsample(in_tens, out_HW=...): ...

class LPIPS(nn.Module):
    L: int
    net: nn.Module
    lins: nn.ModuleList
    def __init__(
        self,
        pretrained=...,
        net=...,
        version=...,
        lpips=...,
        spatial=...,
        pnet_rand=...,
        pnet_tune=...,
        use_dropout=...,
        model_path=...,
        eval_mode=...,
        verbose=...,
    ) -> None: ...
    @overload
    def forward(
        self,
        in0: Tensor,
        in1: Tensor,
        retPerLayer: Literal[False] = ...,
        normalize: bool = ...,
    ) -> Tensor: ...
    @overload
    def forward(
        self,
        in0: Tensor,
        in1: Tensor,
        retPerLayer: Literal[True],
        normalize: bool = ...,
    ) -> tuple[Tensor, list[Tensor]]: ...
    @overload
    def __call__(
        self,
        in0: Tensor,
        in1: Tensor,
        retPerLayer: Literal[False] = ...,
        normalize: bool = ...,
    ) -> Tensor: ...
    @overload
    def __call__(
        self,
        in0: Tensor,
        in1: Tensor,
        retPerLayer: Literal[True],
        normalize: bool = ...,
    ) -> tuple[Tensor, list[Tensor]]: ...

class ScalingLayer(nn.Module):
    def __init__(self) -> None: ...
    def forward(self, inp): ...

class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=..., use_dropout=...) -> None: ...
    def forward(self, x): ...

class Dist2LogitLayer(nn.Module):
    def __init__(self, chn_mid=..., use_sigmoid=...) -> None: ...
    def forward(self, d0, d1, eps=...): ...

class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=...) -> None: ...
    def forward(self, d0, d1, judge): ...

class FakeNet(nn.Module):
    def __init__(self, use_gpu=..., colorspace=...) -> None: ...

class L2(FakeNet):
    def forward(self, in0, in1, retPerLayer=...) -> Tensor | Variable | None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | Variable | None: ...

class DSSIM(FakeNet):
    def forward(self, in0, in1, retPerLayer=...) -> Tensor | Variable: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | Variable: ...

def print_network(net) -> None: ...
