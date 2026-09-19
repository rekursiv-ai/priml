from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

def attach(
    package_name: str,
    submodules: Sequence[str],
) -> tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]: ...
