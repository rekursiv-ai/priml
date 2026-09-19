from jax import lax as lax

def rms_norm_forward_kernel(
    x_ref,
    weight_ref,
    bias_ref,
    o_ref,
    rstd_ref=None,
    *,
    eps: float,
    block_size: int,
): ...
def rms_norm_forward(
    x,
    weight,
    bias,
    num_warps: int | None = None,
    num_stages: int | None = 3,
    eps: float = 1e-05,
    backward_pass_impl: str = "triton",  # noqa: S107 -- default value mirrors upstream; stub is never executed
    interpret: bool = False,
): ...
def rms_norm_backward_kernel_dx(
    x_ref,
    weight_ref,
    bias_ref,
    do_ref,
    rstd_ref,
    dx_ref,
    *,
    eps: float,
    block_size: int,
): ...
def rms_norm_backward_kernel_dw_db(
    x_ref,
    weight_ref,
    bias_ref,
    do_ref,
    rstd_ref,
    dw_ref,
    db_ref,
    *,
    eps: float,
    block_m: int,
    block_n: int,
): ...
def rms_norm_backward(
    num_warps: int | None,
    num_stages: int | None,
    eps: float,
    backward_pass_impl: str,
    interpret: bool,
    res,
    do,
): ...
def rms_norm(
    x,
    weight,
    bias,
    num_warps: int | None = None,
    num_stages: int | None = 3,
    eps: float = 1e-05,
    backward_pass_impl: str = "triton",  # noqa: S107 -- default value mirrors upstream; stub is never executed
    interpret: bool = False,
): ...
def rms_norm_reference(x, weight, bias, *, eps: float = 1e-05): ...
