from typing import Any

import os

from _typeshed import Incomplete

DEFAULT_NUM_TRACING_ATTEMPTS: int
parser: Incomplete

def collect_profile(
    port: int,
    duration_in_ms: int,
    host: str,
    log_dir: os.PathLike | str | None,
    no_perfetto_link: bool,
    xprof_options: dict[str, Any] | None = None,
): ...
def main(known_args, unknown_flags) -> None: ...
