# Generated content -- partially. The structure and docstrings are produced by
# `python stub.py`. The following are hand-edited additions that must be
# re-applied after each regeneration:
#   - module-level imports (`os`, `typing`)
#   - `__version__: str`
#   - type annotations on `TensorSpec` / `serialize` / `serialize_file`
#
# TODO: once we upgrade pyo3 to >= 0.28, replace `stub.py` with a dedicated
# `tools/stub-gen` binary using `pyo3-introspection`,
# mirroring how `huggingface/tokenizers` does it (see PR #1928).
# That generator emits typed stubs directly from Rust
# signatures -- no hand-editing, no drift.

from collections.abc import Sequence

import os

__version__: str

@staticmethod
def deserialize(bytes): ...
@staticmethod
def serialize(
    tensor_dict: dict[str, TensorSpec],
    metadata: dict[str, str] | None = ...,
) -> bytes: ...
@staticmethod
def serialize_file(
    tensor_dict: dict[str, TensorSpec],
    filename: str | os.PathLike[str],
    metadata: dict[str, str] | None = ...,
) -> None: ...

class TensorSpec:
    def __init__(
        self,
        *,
        dtype: str,
        shape: Sequence[int],
        data_ptr: int,
        data_len: int,
    ) -> None: ...
    @property
    def data_len(self) -> int: ...
    @property
    def data_ptr(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> list[int]: ...

class safe_open:
    def __init__(
        self,
        filename,
        framework,
        device=...,
        *,
        backend: str = ...,
    ) -> None: ...
    def __enter__(self): ...
    def __exit__(self, _exc_type, _exc_value, _traceback): ...
    def get_slice(self, name): ...
    def get_tensor(self, name): ...
    def get_tensors(self): ...
    def keys(self): ...
    def metadata(self): ...
    def offset_keys(self): ...

class SafetensorError(Exception): ...
