import logging

from _typeshed import Incomplete
from jax._src.lib import utils as utils

logging_formatter: Incomplete

def update_logging_level_global(logging_level: str | None) -> None: ...

class _DebugHandlerFilter(logging.Filter):
    def filter(self, record): ...

def update_debug_log_modules(module_names_str: str | None): ...
