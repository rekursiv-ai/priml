"""Checks for the helpers shared by two or more casmi2026 test files."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import json

from pyarrow import parquet

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT
from priml.baselines.casmi2026.model import (
    DreamAttention,
    FragmentAttention,
)
from priml.baselines.casmi2026.testing import (
    float32_comparands,
    identity_fingerprints,
    identity_rows,
    retrieval_inputs,
    set_heads,
    write_identity,
)
from priml.model.graphormer import GraphAttention


if TYPE_CHECKING:
    from pathlib import Path


def test_float32_comparands_round_outputs_once_and_keep_matching_inputs() -> None:
    wide = torch.tensor([1 + 2.0**-30, 0.1, -0.0], dtype=torch.float64)
    narrow = torch.tensor([0.5], dtype=torch.float32)
    counts = torch.tensor([3, 4])
    record = {
        "a/input/mass": wide,
        "b/mass": wide,
        "counts": counts,
        "half": narrow,
        "config": wide,
    }
    actual = float32_comparands(record, inputs=("*/input/*", "config"))
    assert list(actual) == list(record)
    assert actual["a/input/mass"] is wide
    assert actual["config"] is wide
    assert actual["counts"] is counts
    assert actual["half"] is narrow
    assert actual["b/mass"].dtype == torch.float64
    assert actual["b/mass"].tolist() == [1.0, float(np.float32(0.1)), -0.0]
    assert torch.signbit(actual["b/mass"][2])
    unmatched = float32_comparands({"input/mass": wide})["input/mass"]
    assert unmatched.dtype == torch.float64
    assert unmatched.tolist() == [1.0, float(np.float32(0.1)), -0.0]


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


def test_identity_records_stage_the_prepared_inputs(tmp_path: Path) -> None:
    rows = identity_rows()
    assert [row["smiles"] for row in rows] == [
        "C/C=C/C",
        "CC(=O)C",
        "C[N+](C)(C)C",
        "CCO.[Na+].[Cl-]",
        "[13CH3]O",
        "*C",
        "invalid",
        "",
    ]
    assert rows[0]["metric"] == (
        '{"py/tuple":["IAQRGUVFOMOMEM",{"py/tuple":[{"py/tuple":["C",4]}]}]}'
    )
    assert (rows[3]["key"], rows[3]["standard_key"]) == (
        "LOHGGLZYTJNUAL",
        "LFQSCWFLJHTTHZ",
    )
    assert (rows[3]["formula"], rows[3]["standard_formula"]) == ("C2H6ClNaO", "C2H6O")
    assert [row["standardized"] is None for row in rows] == [
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    assert rows[2]["key"] == "QEMXHQIAXOOASZ"
    assert rows[2]["standard_metric"] == rows[5]["metric"] == "null"
    assert rows[5]["standard_formula_lookup"] == "CH3*"
    assert rows[6]["standard_formula_lookup"] == ""
    rows[0]["key"] = "edited"
    assert identity_rows()[0]["key"] == "IAQRGUVFOMOMEM"
    write_identity(tmp_path)
    table = parquet.read_table(tmp_path / "molecules.parquet")
    assert table.to_pylist() == [dict(row) for row in identity_rows()]
    fingerprints = cast("np.ndarray", np.load(tmp_path / "fingerprints.npy"))
    np.testing.assert_array_equal(fingerprints, identity_fingerprints())
    assert fingerprints.dtype == np.uint8
    assert fingerprints.shape == (8, 2069)
    assert fingerprints[1, :3].tolist() == [0, 5, 10]
    assert fingerprints[7, :3].tolist() == [0, 17, 34]
    assert not fingerprints[2].any()
    assert not fingerprints[6].any()
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
