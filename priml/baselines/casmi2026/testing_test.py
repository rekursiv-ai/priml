"""Checks for the helpers shared by two or more casmi2026 test files."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import json

from pyarrow import parquet
from torch import nn

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT
from priml.baselines.casmi2026.model import (
    DreamAttention,
    FragmentAttention,
    ReleasedModel,
)
from priml.baselines.casmi2026.testing import (
    decode,
    identity_rows,
    released_parameters,
    retrieval_inputs,
    set_heads,
    write_identity,
)
from priml.model.graphormer import GraphAttention
from priml.testing.golden import read_tensors


_CWD: Final = Path(__file__).resolve().parent

_BASELINE: Final = _CWD


class _Released(nn.Module):
    def __init__(self, *, names: dict[str, str]) -> None:
        super().__init__()
        self.first = nn.Linear(1, 1)
        self.second = nn.Linear(1, 1)
        self._names = names

    def checkpoint_names(self) -> dict[str, str]:
        return self._names


@pytest.mark.parametrize(
    "attn",
    [DreamAttention.Config(), GraphAttention.Config(), FragmentAttention.Config()],
)
def test_head_count_is_set_on_every_attention_configuration(attn: object) -> None:
    set_heads(attn, 3)
    assert cast("DreamAttention.Config", attn).attention.num_heads == 3


def test_head_count_rejects_a_foreign_configuration() -> None:
    with pytest.raises(AssertionError):
        set_heads(object(), 3)


def test_released_order_follows_the_schema_and_lists_each_parameter_once() -> None:
    model = _Released(
        names={
            "b": "second.weight",
            "a": "first.weight",
            "c": "second.bias",
            "d": "first.bias",
            "e": "absent",
        },
    )
    ordered = released_parameters(cast("ReleasedModel", model))
    assert ordered == [
        model.second.weight,
        model.first.weight,
        model.second.bias,
        model.first.bias,
    ]


def test_released_order_rejects_an_incomplete_schema() -> None:
    model = _Released(names={"a": "first.weight"})
    with pytest.raises(ValueError, match="does not cover"):
        released_parameters(cast("ReleasedModel", model))


def test_released_order_rejects_duplicate_that_masks_a_missing_parameter() -> None:
    model = _Released(
        names={
            "a": "first.weight",
            "b": "first.bias",
            "c": "second.bias",
            "d": "first.weight",
        },
    )
    with pytest.raises(ValueError, match="does not cover"):
        released_parameters(cast("ReleasedModel", model))


def test_released_order_rejects_duplicate_with_complete_coverage() -> None:
    model = _Released(
        names={
            "a": "first.weight",
            "b": "first.bias",
            "c": "second.weight",
            "d": "second.bias",
            "e": "first.weight",
        },
    )
    with pytest.raises(ValueError, match="does not cover"):
        released_parameters(cast("ReleasedModel", model))


def test_text_fields_decode_from_their_bytes() -> None:
    assert decode(torch.tensor(list(b"C(=O)"))) == "C(=O)"


def test_identity_records_stage_the_prepared_inputs(tmp_path: Path) -> None:
    expected = read_tensors(_BASELINE / "testdata/identity_preparation.pt")
    rows = identity_rows(expected)
    assert len(rows) == 8
    assert rows[0]["smiles"] == decode(expected["0/input"])
    write_identity(tmp_path, expected)
    table = parquet.read_table(tmp_path / "molecules.parquet")
    assert table.column("smiles").to_pylist() == [row["smiles"] for row in rows]
    fingerprints = cast("np.ndarray", np.load(tmp_path / "fingerprints.npy"))
    assert fingerprints.shape[0] == 8
    manifest = cast(
        "dict[str, object]",
        json.loads((tmp_path / "manifest.json").read_text()),
    )
    assert manifest == {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "raw_bits": 16_551,
    }


def test_retrieval_inputs_are_complete_and_deterministic() -> None:
    library, pool, spectra = retrieval_inputs()
    assert len(library.offsets) == len(library.peak_count) + 1
    assert len(library.keys) == 9
    assert len(pool.keys) == len(pool.smiles) == len(pool.mass) == 5
    assert [len(spectrum["mz"]) for spectrum in spectra] == [3, 5, 0]
    again = retrieval_inputs()
    assert np.array_equal(again[0].mz, library.mz)
    assert again[2] == spectra


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
