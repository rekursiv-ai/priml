from _typeshed import Incomplete

__all__ = ["MonoWorker"]

class MonoWorker:
    pool: Incomplete
    futures: Incomplete
    def __init__(self) -> None: ...
    def submit(self, func, *args, **kwargs): ...
