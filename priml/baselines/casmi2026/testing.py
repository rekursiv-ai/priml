"""Helpers shared by two or more casmi2026 test files; the wheel does not ship them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

from treekle import from_plain, to_plain

import numpy as np

from priml.baselines.casmi2026.chemistry import CandidatePool, SpectralLibrary
from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT
from priml.baselines.casmi2026.model import DreamAttention, FragmentAttention
from priml.model.graphormer import GraphAttention


if TYPE_CHECKING:
    from pathlib import Path

    from pyarrow import Table, parquet
    from torch import Tensor, nn

    from priml.baselines.casmi2026.custom_types import IdentityRow, SpectrumRequest
    from priml.baselines.casmi2026.model import ReleasedModel
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")


def set_heads(attn: object, heads: int) -> None:
    """Set the query-head count of one of the baseline's attention configurations.

    Args:
      attn: A block's ``attn`` slot holding a graph, fragment or Dream attention.
      heads: Query heads of the attention.

    """
    assert isinstance(
        attn,
        (DreamAttention.Config, GraphAttention.Config, FragmentAttention.Config),
    )
    attn.attention.num_heads = heads


def released_parameters(model: ReleasedModel) -> list[nn.Parameter]:
    """Return every parameter in released registration order, shared ones once.

    Args:
      model: Released model whose checkpoint schema names the parameters.

    Returns:
      ordered: Frozen and trainable parameters, covering the whole model.

    Raises:
      ValueError: The schema leaves a parameter of the model unnamed.

    """
    parameters = dict(model.named_parameters())
    ordered = [
        parameters[native]
        for native in model.checkpoint_names().values()
        if native in parameters
    ]
    if len(ordered) != len(parameters) or {id(parameter) for parameter in ordered} != {
        id(parameter) for parameter in parameters.values()
    }:
        raise ValueError("Checkpoint order does not cover the model's parameters.")
    return ordered


def identity_rows(expected: dict[str, Tensor]) -> list[IdentityRow]:
    """Transport every complete source field into the declared native table schema.

    Args:
      expected: Source-produced identity record tensors of the eight reached cases.

    Returns:
      rows: One native identity row per case.

    """
    rows: list[IdentityRow] = []
    for index in range(8):
        prefix = str(index)
        rows.append(
            {
                "smiles": decode(expected[prefix + "/input"]),
                "standardized": decode(expected[prefix + "/standardized/text"])
                if bool(expected[prefix + "/standardized/valid"])
                else None,
                "key": decode(expected[prefix + "/key/text"])
                if bool(expected[prefix + "/key/valid"])
                else None,
                "standard_key": decode(expected[prefix + "/standard_key/text"])
                if bool(expected[prefix + "/standard_key/valid"])
                else None,
                "formula": decode(expected[prefix + "/formula"]),
                "metric": decode(expected[prefix + "/metric"]),
                "standard_metric": decode(expected[prefix + "/standard_metric"]),
                "standard_formula_lookup": decode(
                    expected[prefix + "/properties/formula"],
                ),
                "properties_valid": bool(expected[prefix + "/properties/valid"]),
                "standard_formula": decode(expected[prefix + "/properties/formula"]),
                "standard_mass": float(expected[prefix + "/properties/mass"]),
                "standard_heavy": int(expected[prefix + "/properties/heavy"]),
                "fingerprint_valid": bool(expected[prefix + "/fingerprint/valid"]),
            },
        )
    return rows


def write_identity(directory: Path, expected: dict[str, Tensor]) -> None:
    """Stage whole source-produced records as native prepared inputs.

    Args:
      directory: Existing directory receiving molecules, fingerprints and manifest.
      expected: Source-produced identity record tensors of the eight reached cases.

    """
    parquet.write_table(
        Table.from_pylist([dict(row) for row in identity_rows(expected)]),
        directory / "molecules.parquet",
    )
    np.save(
        directory / "fingerprints.npy",
        np.stack(
            [expected[f"{index}/fingerprint/packed"].numpy() for index in range(8)],
        ),
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            to_plain(
                {"format_version": 1, "rdkit": IDENTITY_RDKIT, "raw_bits": 16_551},
            ),
        ),
    )


def decode(value: Tensor) -> str:
    """Decode every byte of a source-produced text field."""
    return bytes(from_plain(value.tolist(), list[int])).decode()


def retrieval_inputs() -> tuple[SpectralLibrary, CandidatePool, list[SpectrumRequest]]:
    """Build complete deterministic raw inputs without computed expected outputs.

    Returns:
      inputs: Spectral library, candidate pool and the three query spectra.

    """
    counts = np.array([3, 4, 3, 5, 2, 4, 4, 5, 6, 3, 4, 7, 2], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    sid = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    structure_mass = np.array(
        [100, 100.0005, 100.002, 103, 90, 110, 90, 300, 100],
        dtype=np.float64,
    )
    mass = structure_mass[sid]
    order = np.argsort(mass, kind="mergesort")
    library = SpectralLibrary(
        offsets=offsets,
        mz=np.concatenate(
            [
                np.arange(counts.item(i), dtype=np.float32) * 10 + 20
                for i in range(len(counts))
            ],
        ),
        intensity=np.concatenate(
            [
                np.arange(counts.item(i), dtype=np.float32) % 3 + 1
                for i in range(len(counts))
            ],
        ),
        structure_mass=structure_mass,
        keys=tuple(f"key{index}" for index in range(9)),
        smiles=tuple("C" * (index + 1) for index in range(9)),
        sid=sid,
        library=np.array([2, 2, 4, 0, 3, 1, 10, 7, 6, 5, 4, 3, 2], dtype=np.int8),
        mode=np.array([1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1], dtype=np.int8),
        instrument=np.arange(13, dtype=np.int8) % 6,
        adduct=np.arange(13, dtype=np.int16) % 2,
        precursor=np.arange(13, dtype=np.float64) / 8 + 101,
        peak_count=counts,
        mass=mass,
        order=order,
        mass_sorted=mass[order],
    )
    pool = CandidatePool(
        mass=np.array([99.999, 100, 100.001, 100.002, 130], dtype=np.float64),
        keys=("a", "b", "a", "d", "e"),
        smiles=("C", "CC", "C", "O", "N"),
        formulas=("CH4", "C2H6", "CH4", "H2O", "H3N"),
        heavy_atoms=np.array([1, 2, 1, 1, 1], dtype=np.int32),
        source=np.arange(5, dtype=np.int8) % 3,
        train_sid=np.array([0, 1, 2, -1, -1], dtype=np.int64),
        packed=np.array([[17], [33], [17], [89], [63]], dtype=np.uint8),
        bits=np.arange(7, dtype=np.int64),
        fragment_offsets=np.arange(6, dtype=np.int64) * 2,
        fragment_mass=np.arange(10, dtype=np.float32) + 10,
        key_to_pid={"a": 2, "b": 1, "d": 3, "e": 4},
    )
    spectra: list[SpectrumRequest] = [
        {
            "mz": [20, 30, 40],
            "it": [1, 2, 3],
            "mode": 1,
            "prec": 101.0,
            "adduct": "[M+H]+",
        },
        {
            "mz": [20, 40, 50, 60, 70],
            "it": [3, 1, 2, 1, 0.001],
            "mode": -1,
            "prec": 99.0,
            "adduct": "[M-H]-",
        },
        {"mz": [], "it": [], "mode": 1, "prec": 101.0, "adduct": "[M+H]+"},
    ]
    return library, pool, spectra
