from _typeshed import Incomplete

from .utils_worker import MonoWorker
from ..auto import tqdm as tqdm_auto

__all__ = ["TelegramIO", "tqdm", "tqdm_telegram", "trange", "ttgrange"]

class TelegramIO(MonoWorker):
    API: str
    token: Incomplete
    chat_id: Incomplete
    session: Incomplete
    text: Incomplete
    def __init__(self, token, chat_id) -> None: ...
    @property
    def message_id(self): ...
    def write(self, s): ...
    def delete(self): ...

class tqdm_telegram(tqdm_auto):
    tgio: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def display(self, **kwargs) -> None: ...
    def clear(self, *args, **kwargs) -> None: ...
    def close(self) -> None: ...

def ttgrange(*args, **kwargs): ...

tqdm = tqdm_telegram
trange = ttgrange
