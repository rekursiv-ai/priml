from typing import Any

class RcParams(dict[str, Any]): ...

rcParams: RcParams

def use(backend: str, *, force: bool = True) -> None: ...
