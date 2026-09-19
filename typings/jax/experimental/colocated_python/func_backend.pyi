from collections.abc import Sequence

from _typeshed import Incomplete

import jax

class _ResultStore:
    def __init__(self) -> None: ...
    def push(self, uid: int, out: Sequence[jax.Array]) -> None: ...
    def pop(self, uid: int) -> Sequence[jax.Array]: ...

SINGLETON_RESULT_STORE: Incomplete
