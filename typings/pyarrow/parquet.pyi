from os import PathLike

from pyarrow import Table

class ParquetFile:
    num_row_groups: int

    def __init__(self, source: str | PathLike[str]) -> None: ...
    def read_row_group(self, i: int) -> Table: ...

def read_table(source: str | PathLike[str]) -> Table: ...
def write_table(
    table: Table,
    where: str | PathLike[str],
) -> None: ...
