from collections.abc import Sequence

from numba.core.ir import FunctionIR
from numba.core.types import Type

# Prunes, in place, each branch whose condition the argument types decide.
def dead_branch_prune(func_ir: FunctionIR, called_args: Sequence[Type]) -> None: ...
