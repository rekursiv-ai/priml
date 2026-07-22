from _typeshed import Incomplete

from .utils_worker import MonoWorker
from ..auto import tqdm as tqdm_auto

__all__ = ["DiscordIO", "tdrange", "tqdm", "tqdm_discord", "trange"]

class DiscordIO(MonoWorker):
    API: str
    UA: Incomplete
    token: Incomplete
    channel_id: Incomplete
    session: Incomplete
    text: Incomplete
    def __init__(self, token, channel_id) -> None: ...
    @property
    def message_id(self): ...
    def write(self, s): ...
    def delete(self): ...

class tqdm_discord(tqdm_auto):
    dio: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def display(self, **kwargs) -> None: ...
    def clear(self, *args, **kwargs) -> None: ...
    def close(self) -> None: ...

def tdrange(*args, **kwargs): ...

tqdm = tqdm_discord
trange = tdrange
