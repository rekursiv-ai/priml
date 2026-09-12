from collections.abc import Iterator, Sequence
from os import PathLike
from typing import Protocol

from pyarrow import Array, Table

class _RecordBatch(Protocol):
    def column(self, i: int | str) -> Array: ...

class FileMetaData:
    num_rows: int
    created_by: str

class ParquetFile:
    num_row_groups: int

    def __init__(self, source: str | PathLike[str]) -> None: ...
    def read_row_group(self, i: int) -> Table: ...
    def iter_batches(
        self,
        batch_size: int = 65536,
        row_groups: Sequence[int] | None = None,
        columns: Sequence[str] | None = None,
        use_threads: bool = True,
        use_pandas_metadata: bool = False,
    ) -> Iterator[_RecordBatch]: ...

def read_table(source: str | PathLike[str]) -> Table: ...
def read_metadata(where: str | PathLike[str]) -> FileMetaData: ...
def write_table(
    table: Table,
    where: str | PathLike[str],
    row_group_size: int | None = None,
    *,
    compression: str | dict[str, str] = "snappy",
) -> None: ...
