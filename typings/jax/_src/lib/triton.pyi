from typing import Protocol

from jaxlib.triton import dialect as dialect

class CompilationResult(Protocol):
    asm: str
    hsaco_path: str
    smem_bytes: int

class CompilationHandler(Protocol):
    def __call__(
        self,
        module: bytes,
        arch_name: str,
        num_warps: int,
        num_ctas: int,
        num_stages: int,
    ) -> CompilationResult: ...

def register_compilation_handler(
    platform: str,
    handler: CompilationHandler,
) -> None: ...
def has_compilation_handler(platform: str) -> bool: ...
def compile(
    platform: str,
    module: bytes,
    arch_name: str,
    *,
    num_warps: int,
    num_ctas: int,
    num_stages: int,
) -> CompilationResult: ...
