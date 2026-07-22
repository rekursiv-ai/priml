from collections.abc import Generator

from _typeshed import Incomplete

from .std import tqdm as std_tqdm

__all__ = ["tarange", "tqdm", "tqdm_asyncio", "trange"]

class tqdm_asyncio(std_tqdm):
    iterable_awaitable: bool
    iterable_next: Incomplete
    iterable_iterator: Incomplete
    def __init__(self, iterable=None, *args, **kwargs) -> None: ...
    def __aiter__(self): ...
    async def __anext__(self): ...
    def send(self, *args, **kwargs): ...
    @classmethod
    def as_completed(
        cls, fs, *, loop=None, timeout=None, total=None, **tqdm_kwargs
    ) -> Generator[Incomplete, Incomplete]: ...
    @classmethod
    async def gather(cls, *fs, loop=None, timeout=None, total=None, **tqdm_kwargs): ...  # noqa: ASYNC109 -- mirrors upstream timeout parameter

def tarange(*args, **kwargs): ...

tqdm = tqdm_asyncio
trange = tarange
