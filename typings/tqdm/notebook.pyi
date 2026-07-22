from _typeshed import Incomplete
from IPython.html.widgets import ContainerWidget as HBox

from .std import tqdm as std_tqdm

__all__ = ["tnrange", "tqdm", "tqdm_notebook", "trange"]

HBox = object

class TqdmHBox(HBox): ...

class tqdm_notebook(std_tqdm):
    @staticmethod
    def status_printer(_, total=None, desc=None, ncols=None): ...
    displayed: bool
    def display(
        self,
        msg=None,
        pos=None,
        close: bool = False,
        bar_style=None,
        check_delay: bool = True,
    ) -> None: ...
    @property
    def colour(self): ...
    @colour.setter
    def colour(self, bar_color) -> None: ...
    disp: Incomplete
    ncols: Incomplete
    container: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def __iter__(self): ...
    def update(self, n: int = 1): ...
    def close(self) -> None: ...
    def clear(self, *_, **__) -> None: ...
    def reset(self, total=None): ...

def tnrange(*args, **kwargs): ...

tqdm = tqdm_notebook
trange = tnrange
