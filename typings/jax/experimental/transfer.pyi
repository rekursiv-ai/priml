from typing import Any

from _typeshed import Incomplete
from jax._src.util import (
    use_cpp_class as use_cpp_class,
    use_cpp_method as use_cpp_method,
)

class TransferConnection:
    def pull(self, uuid: int, xs: Any) -> Any: ...

class TransferServer:
    def address(self) -> str: ...
    def connect(self, address: str) -> TransferConnection: ...
    def await_pull(self, uuid: int, arrays: Any) -> Any: ...

start_transfer_server: Incomplete

def make_error_array(aval, message): ...
