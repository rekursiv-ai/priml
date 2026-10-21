"""Helpers shared by two or more casmi2026 test files; the wheel does not ship them."""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

import json

from treekle import to_plain

import numpy as np

from priml.baselines.casmi2026.chemistry import CandidatePool, SpectralLibrary
from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT
from priml.baselines.casmi2026.model import DreamAttention, FragmentAttention
from priml.model.graphormer import GraphAttention


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from pyarrow import Table, parquet
    from torch import Tensor

    import torch

    from priml.baselines.casmi2026.custom_types import IdentityRow, SpectrumRequest
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    torch = lazy_import("torch")


def float32_comparands(
    record: Mapping[str, Tensor],
    *,
    inputs: Sequence[str] = (),
) -> dict[str, Tensor]:
    """Round every float64 output once to float32; stored replay inputs stay whole.

    Args:
      record: Golden record of computed outputs and the inputs its test replays.
      inputs: Shell-style key patterns of the replayed inputs, kept unrounded.

    Returns:
      record: The same keys in order, every other float64 tensor rounded to
        float32 values and kept in float64, so the golden still pins its dtype.

    """
    return {
        key: value.float().double()
        if value.dtype == torch.float64
        and not any(fnmatchcase(key, pattern) for pattern in inputs)
        else value
        for key, value in record.items()
    }


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


def identity_rows() -> list[IdentityRow]:
    """Return eight prepared identity rows covering every rejection and alias case.

    Returns:
      rows: Fresh rows: two aliases, a charged and a salt input, an isotope, a
        dummy atom, an unparsable input and the empty string.

    """
    return [
        _identity_row(
            "C/C=C/C",
            standardized="CC=CC",
            key="IAQRGUVFOMOMEM",
            formula="C4H8",
            composition=(("C", 4),),
            mass=56.062600255999996,
            heavy=4,
        ),
        _identity_row(
            "CC(=O)C",
            standardized="CC(C)=O",
            key="CSCPPACGZOOCGX",
            formula="C3H6O",
            composition=(("C", 3), ("O", 1)),
            mass=58.041864812,
            heavy=4,
        ),
        {
            **_identity_row("C[N+](C)(C)C", standardized=None, formula="C4H12N+"),
            "key": "QEMXHQIAXOOASZ",
            "metric": _metric("QEMXHQIAXOOASZ", (("C", 4), ("N", 1))),
        },
        {
            **_identity_row(
                "CCO.[Na+].[Cl-]",
                standardized="CCO",
                key="LFQSCWFLJHTTHZ",
                formula="C2H6O",
                composition=(("C", 2), ("O", 1)),
                mass=46.041864812,
                heavy=3,
            ),
            "key": "LOHGGLZYTJNUAL",
            "formula": "C2H6ClNaO",
            "metric": _metric(
                "LOHGGLZYTJNUAL",
                (("C", 2), ("Cl", 1), ("Na", 1), ("O", 1)),
            ),
        },
        _identity_row(
            "[13CH3]O",
            standardized="[13CH3]O",
            key="OKKJLVBELUTLKV",
            formula="CH4O",
            composition=(("C", 1), ("O", 1)),
            mass=33.029569588,
            heavy=2,
        ),
        _identity_row(
            "*C",
            standardized="*C",
            formula="CH3*",
            mass=15.023475096,
            heavy=1,
        ),
        _identity_row("invalid", standardized=None, formula="invalid"),
        _identity_row("", standardized="", formula=""),
    ]


def identity_fingerprints() -> NDArray[np.uint8]:
    """Return one distinct packed 16,551-bit row per identity row, zero when invalid."""
    valid = np.array([row["fingerprint_valid"] for row in identity_rows()])
    pattern = np.arange(2069)[None, :] * (2 * np.arange(8)[:, None] + 3) % 256
    return np.where(valid[:, None], pattern, 0).astype(np.uint8)


def write_identity(directory: Path) -> None:
    """Stage the identity rows as native prepared inputs.

    Args:
      directory: Existing directory receiving molecules, fingerprints and manifest.

    """
    parquet.write_table(
        Table.from_pylist([dict(row) for row in identity_rows()]),
        directory / "molecules.parquet",
    )
    np.save(directory / "fingerprints.npy", identity_fingerprints())
    (directory / "manifest.json").write_text(
        json.dumps(
            to_plain(
                {"format_version": 1, "rdkit": IDENTITY_RDKIT, "raw_bits": 16_551},
            ),
        ),
    )


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


def _identity_row(
    smiles: str,
    *,
    standardized: str | None,
    formula: str,
    key: str | None = None,
    composition: tuple[tuple[str, int], ...] = (),
    mass: float = 0.0,
    heavy: int = 0,
) -> IdentityRow:
    metric = _metric(key, composition) if key is not None else "null"
    valid = standardized is not None
    lookup = formula if valid else ""
    return {
        "smiles": smiles,
        "standardized": standardized,
        "key": key,
        "formula": formula,
        "metric": metric,
        "standard_key": key,
        "standard_metric": metric,
        "standard_formula_lookup": lookup,
        "properties_valid": valid,
        "standard_formula": lookup,
        "standard_mass": mass,
        "standard_heavy": heavy,
        "fingerprint_valid": valid,
    }


def _metric(key: str, composition: tuple[tuple[str, int], ...]) -> str:
    return json.dumps(to_plain((key, composition)), separators=(",", ":"))
