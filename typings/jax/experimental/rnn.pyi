from typing import Any

from _typeshed import Incomplete
from jax._src import (
    core as core,
    dispatch as dispatch,
)
from jax._src.custom_derivatives import custom_vjp as custom_vjp
from jax._src.lax import lax as lax
from jax._src.lib import gpu_rnn as gpu_rnn
from jax._src.typing import (
    Array as Array,
    Shape as Shape,
)
from jax.interpreters import mlir as mlir

type PRNGKeyArray = Any
sigmoid: Incomplete
tanh: Incomplete

def get_num_params_in_lstm(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
) -> int: ...
def init_lstm_weight(
    rng: PRNGKeyArray,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
): ...
def swap_lstm_gates(weights, input_size, hidden_size, num_layers, bidirectional): ...
def unpack_lstm_weights(
    weights: Array,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
) -> tuple[dict[int, Array], dict[int, Array], dict[int, Array], dict[int, Array]]: ...
def lstm(
    x: Array,
    h_0: Array,
    c_0: Array,
    weights: Array,
    seq_lengths: Array,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    precision: lax.PrecisionLike = None,
) -> tuple[Array, Array, Array]: ...
def lstm_ref(
    x: Array,
    h_0: Array,
    c_0: Array,
    W_ih: dict[int, Array],
    W_hh: dict[int, Array],
    b_ih: dict[int, Array],
    b_hh: dict[int, Array],
    seq_lengths: Array,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
) -> tuple[Array, Array, Array]: ...
def lstm_fwd(
    x: Array,
    h_0: Array,
    c_0: Array,
    w: Array,
    seq_lengths: Array,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    precision: lax.PrecisionLike,
): ...
def rnn_abstract_eval(
    x_aval,
    h_0_aval,
    c_0_aval,
    w_aval,
    seq_lengths_aval,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    cudnn_allow_tf32: bool,
): ...

rnn_fwd_p: Incomplete

def lstm_bwd(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    precision: lax.PrecisionLike,
    residuals,
    gradients,
): ...
def rnn_bwd_abstract_eval(
    dy_aval,
    dhn_aval,
    dcn_aval,
    x_aval,
    h0_aval,
    c0_aval,
    w_aval,
    y_aval,
    reserve_space_aval,
    seq_lengths_aval,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    cudnn_allow_tf32: bool,
): ...

rnn_bwd_p: Incomplete
