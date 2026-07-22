from _typeshed import Incomplete
from dask.callbacks import Callback

__all__ = ["TqdmCallback"]

class TqdmCallback(Callback):
    tqdm_class: Incomplete
    def __init__(
        self, start=None, pretask=None, tqdm_class=..., **tqdm_kwargs
    ) -> None: ...
    def display(self) -> None: ...
