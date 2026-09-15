from collections.abc import Callable

def do_bench(
    fn: Callable[[], object],
    warmup: int = ...,
    rep: int = ...,
    quantiles: list[float] | None = ...,
    return_mode: str = ...,
) -> float: ...
