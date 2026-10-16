from numba.core.types import Type

class Signature:
    return_type: Type
    args: tuple[Type, ...]
    def __init__(
        self,
        return_type: Type,
        args: tuple[Type, ...],
        recvr: Type | None,
        pysig: object = None,
    ) -> None: ...
