from typing import Any

from jax._src.debugger import (
    cli_debugger as cli_debugger,
    core as debugger_core,
)

class WebDebugger(cli_debugger.CliDebugger):
    prompt: str
    use_rawinput: bool
    def __init__(
        self,
        frames: list[debugger_core.DebuggerFrame],
        thread_id,
        completekey: str = "tab",
        host: str = "",
        port: int = 5555,
    ) -> None: ...
    def get_current_frame_data(self): ...
    def get_globals(self): ...
    def get_locals(self): ...
    def run(self): ...

def run_debugger(
    frames: list[debugger_core.DebuggerFrame],
    thread_id: int | None,
    **kwargs: Any,
): ...
