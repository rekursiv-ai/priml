from _typeshed import Incomplete
from jax import (
    lax as lax,
    random as random,
)
from jax.nn import (
    elu as elu,
    gelu as gelu,
    leaky_relu as leaky_relu,
    log_softmax as log_softmax,
    relu as relu,
    selu as selu,
    sigmoid as sigmoid,
    softmax as softmax,
    softplus as softplus,
    standardize as standardize,
)
from jax.nn.initializers import (
    glorot_normal as glorot_normal,
    normal as normal,
    ones as ones,
    zeros as zeros,
)

glorot = glorot_normal
randn = normal
logsoftmax = log_softmax

def Dense(out_dim, W_init=..., b_init=...): ...
def GeneralConv(
    dimension_numbers,
    out_chan,
    filter_shape,
    strides=None,
    padding: str = "VALID",
    W_init=None,
    b_init=...,
): ...

Conv: Incomplete

def GeneralConvTranspose(
    dimension_numbers,
    out_chan,
    filter_shape,
    strides=None,
    padding: str = "VALID",
    W_init=None,
    b_init=...,
): ...

Conv1DTranspose: Incomplete
ConvTranspose: Incomplete

def BatchNorm(
    axis=(0, 1, 2),
    epsilon: float = 1e-05,
    center: bool = True,
    scale: bool = True,
    beta_init=...,
    gamma_init=...,
): ...
def elementwise(fun, **fun_kwargs): ...

Tanh: Incomplete
Relu: Incomplete
Exp: Incomplete
LogSoftmax: Incomplete
Softmax: Incomplete
Softplus: Incomplete
Sigmoid: Incomplete
Elu: Incomplete
LeakyRelu: Incomplete
Selu: Incomplete
Gelu: Incomplete
MaxPool: Incomplete
SumPool: Incomplete
AvgPool: Incomplete

def Flatten(): ...

Flatten: Incomplete

def Identity(): ...

Identity: Incomplete

def FanOut(num): ...
def FanInSum(): ...

FanInSum: Incomplete

def FanInConcat(axis: int = -1): ...
def Dropout(rate, mode: str = "train"): ...
def serial(*layers): ...
def parallel(*layers): ...
def shape_dependent(make_layer): ...
