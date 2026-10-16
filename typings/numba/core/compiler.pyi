from collections.abc import Callable

from numba.core.ir import FunctionIR

def run_frontend(
    func: Callable[..., object],
    inline_closures: bool = False,
    emit_dels: bool = False,
) -> FunctionIR: ...
