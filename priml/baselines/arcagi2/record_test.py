"""Frozen ARC2 records, and the helpers that compare them.

The implementation this recipe was ported from was run once through each
test's own recorder, and what it produced is
kept in ``testdata/<module>.pt``: one flat name-to-tensor record per case. A
scalar is kept as a rank-0 tensor and any other leaf as its ``repr`` encoded to
bytes, so every leaf compares with ``torch.equal``. Nothing here imports the
source, so the proof outlives it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from torch import Tensor

import torch

from priml.testing.golden import mismatches, read_tensors, write_tensors


if TYPE_CHECKING:
    from collections.abc import Mapping


_CWD: Final = Path(__file__).resolve().parent

_CASE: Final = "::"
"""Separates a case name from a leaf path; neither contains it."""


def reduce(value: object, prefix: str = "") -> dict[str, Tensor]:
    """Flatten nested mappings and sequences to ``{path: tensor}``.

    Args:
      value: A tensor, scalar, or nested mapping/list/tuple of them.
      prefix: Path of ``value`` inside the enclosing record.

    Returns:
      record: One detached CPU tensor per leaf. Booleans, ints, and floats
        become rank-0 tensors; every other leaf is its ``repr`` as bytes.
        Floats are float32: a float64 result keeps its host's libm error,
        which only the round to float32 absorbs.

    """
    if isinstance(value, Tensor):
        leaf = value.detach().cpu()
        return {prefix: leaf.float() if leaf.dtype == torch.float64 else leaf.clone()}
    if isinstance(value, dict):
        out: dict[str, Tensor] = {}
        for key, item in cast("dict[object, object]", value).items():
            out.update(reduce(item, f"{prefix}/{key}" if prefix else str(key)))
        return out
    if isinstance(value, list | tuple):
        out = {}
        for index, item in enumerate(cast("list[object]", value)):
            out.update(reduce(item, f"{prefix}/{index}" if prefix else str(index)))
        return out
    if isinstance(value, bool):
        return {prefix: torch.tensor(value)}
    if isinstance(value, int):
        return {prefix: torch.tensor(value, dtype=torch.int64)}
    if isinstance(value, float):
        return {prefix: torch.tensor(value, dtype=torch.float32)}
    return {
        prefix: torch.frombuffer(bytearray(repr(value).encode()), dtype=torch.uint8),
    }


def golden_path(name: str) -> Path:
    """Where the frozen records of test module ``name`` live."""
    return _CWD / "testdata" / f"{name}.pt"


def load(name: str) -> dict[str, dict[str, Tensor]]:
    """Load the case-keyed source records of one test module."""
    table: dict[str, dict[str, Tensor]] = {}
    for key, value in read_tensors(golden_path(name)).items():
        case, _, leaf = key.partition(_CASE)
        table.setdefault(case, {})[leaf] = value
    return table


def save(name: str, table: Mapping[str, Mapping[str, Tensor]]) -> None:
    """Write one test module's case-keyed records as one flat tensor golden."""
    flat = {
        f"{case}{_CASE}{key}": value
        for case, record in table.items()
        for key, value in record.items()
    }
    write_tensors(golden_path(name), flat)


def assert_matches(name: str, case: str, record: Mapping[str, Tensor]) -> None:
    """Require ``record`` to equal the frozen source record of ``case``."""
    report = mismatches(load(name)[case], record)
    assert not report, f"{len(report)} mismatches:\n" + "\n".join(report)


def test_reduce_keys_every_leaf() -> None:
    record = reduce({"a": [torch.zeros(1), 2], "b": {"c": 0.5, "d": None}})
    assert record.keys() == {"a/0", "a/1", "b/c", "b/d"}
    assert record["a/1"].dtype == torch.int64
    assert record["b/c"].dtype == torch.float32
    assert record["b/c"].item() == 0.5
    assert record["b/d"].numpy().tobytes() == b"None"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
