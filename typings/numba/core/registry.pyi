from numba.core.base import BaseContext

class CPUTarget:
    @property
    def target_context(self) -> BaseContext: ...

cpu_target: CPUTarget
