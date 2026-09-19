from types import ModuleType

from _typeshed import Incomplete

msg: Incomplete

def check_jaxlib_version(
    jax_version: str,
    jaxlib_version: str,
    minimum_jaxlib_version: str,
) -> tuple[int, ...]: ...

version_str: Incomplete
version: Incomplete
jaxlib_extension_version: int
ifrt_version: int
has_cpu_sparse: bool
cuda_versions: ModuleType | None
cuda_path: Incomplete
