from collections.abc import Iterator
from contextlib import contextmanager

import logging

from _typeshed import Incomplete

from ..std import tqdm as std_tqdm

class _TqdmLoggingHandler(logging.StreamHandler):
    tqdm_class: Incomplete
    def __init__(self, tqdm_class: type[std_tqdm] = ...) -> None: ...
    def emit(self, record) -> None: ...

@contextmanager
def logging_redirect_tqdm(
    loggers: tuple[list[logging.Logger] | None] | None = None,
    tqdm_class: type[std_tqdm] = ...,
) -> Iterator[None]: ...
@contextmanager
def tqdm_logging_redirect(*args, **kwargs) -> Iterator[None]: ...
