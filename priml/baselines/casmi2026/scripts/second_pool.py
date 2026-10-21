"""Prepare separate engine2 fingerprint data and assemble its provided candidate pool."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from treekle import from_plain, json_loads

import numpy as np

from priml.baselines.casmi2026.custom_types import PreparationHeader
from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, read_pool_metadata
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.baselines.casmi2026.scripts.rdkit_backend import Chemistry
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from pyarrow import Table, parquet
    from rdkit.Chem.Descriptors import ExactMolWt
    from rdkit.Chem.MACCSkeys import GenMACCSKeys
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    ExactMolWt = lazy_import("rdkit.Chem.Descriptors", "ExactMolWt")
    GenMACCSKeys = lazy_import("rdkit.Chem.MACCSkeys", "GenMACCSKeys")


_THIS: Final = Path(__file__).resolve()


def prepare_second_fingerprints(
    input_path: Path,
    *,
    bits: Path,
    output: Path,
    version: str,
) -> None:
    """Generate selected four-block fingerprints without an InChI validity gate.

    Args:
      input_path: Requested smiles or normalized_smiles Parquet, including duplicates.
      bits: Complete selected-bit vocabulary for Morgan2, Morgan3, RDK6 and MACCS.
      output: New external fingerprint-cache directory.
      version: Required isolated scientific-library release.

    """
    if version != IDENTITY_RDKIT:
        raise ValueError(f"Second-engine fingerprints require RDKit {IDENTITY_RDKIT}.")
    chemistry = Chemistry(version)
    selected = cast("NDArray[np.int64]", np.load(bits, allow_pickle=False))
    if (
        selected.ndim != 1
        or selected.dtype != np.int64
        or not len(selected)
        or np.any(selected < 0)
        or np.any(selected >= 10_407)
    ):
        raise ValueError(
            "Selected bits require an int64 vector within the four-block layout.",
        )
    source = parquet.ParquetFile(input_path)
    column = "smiles" if "smiles" in source.schema_arrow.names else "normalized_smiles"
    requested: dict[str, None] = {}
    for batch in source.iter_batches(batch_size=256, columns=[column]):
        for value in from_plain(
            cast(object, batch.column(0).to_pylist()),
            list[str | None],
        ):
            if value is not None:
                requested[value] = None
    destination = validated_output_path(output, protected=[input_path, bits])
    destination.mkdir(parents=True)
    packed = np.lib.format.open_memmap(
        destination / "fingerprints.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(requested), (len(selected) + 7) // 8),
    )
    valid = np.zeros(len(requested), dtype=np.bool_)
    mass = np.full(len(requested), np.nan)
    for row, value in enumerate(requested):
        packed[row] = 0
        try:
            molecule = chemistry.chem.MolFromSmiles(value)
            if molecule is None:
                continue
            parts = [
                chemistry.generators[index]
                .GetFingerprintAsNumPy(molecule)
                .astype(np.uint8)
                for index in (0, 1, 3)
            ]
            parts.append(
                np.array(list(GenMACCSKeys(molecule)), dtype=np.uint8),
            )
            packed[row] = np.packbits(np.concatenate(parts)[selected])
            mass[row] = ExactMolWt(molecule)
            valid[row] = True
        except (ValueError, RuntimeError, TypeError, OverflowError):
            packed[row] = 0
    packed.flush()
    np.save(destination / "fp_bits.npy", selected)
    parquet.write_table(
        Table.from_pydict({"smiles": list(requested), "valid": valid, "mass": mass}),
        destination / "fingerprints.parquet",
    )
    _write_manifest(
        destination,
        inputs=[input_path, bits],
        fields={
            "rdkit": version,
            "raw_bits": 10_407,
            "rows": len(requested),
            "selected_bits": len(selected),
        },
    )


def prepare_second_pool(
    input_path: Path,
    *,
    coconut: Path,
    bio: Path,
    fingerprints: Path,
    output: Path,
) -> None:
    """Assemble the exact provided-plus-training pool without molecular imports.

    Args:
      input_path: Original training Parquet, retaining key and SMILES encounter order.
      coconut: Provided coco_meta.pkl, coco_mass.npy and coco_fp.npy directory.
      bio: Provided bio_meta.pkl, bio_mass.npy and bio_fp.npy directory.
      fingerprints: Prepared four-block training fingerprint cache.
      output: New external directory with the stable mass-sorted complete pool.

    """
    paths = [
        input_path,
        *(coconut / f"coco_{name}" for name in ("meta.pkl", "mass.npy", "fp.npy")),
        *(bio / f"bio_{name}" for name in ("meta.pkl", "mass.npy", "fp.npy")),
        *(
            fingerprints / name
            for name in (
                "fingerprints.parquet",
                "fingerprints.npy",
                "fp_bits.npy",
                "manifest.json",
            )
        ),
    ]
    destination = validated_output_path(output, protected=paths)
    destination.mkdir(parents=True)
    manifest = json_loads(
        (fingerprints / "manifest.json").read_text(),
        PreparationHeader,
    )
    if manifest["format_version"] != 1 or manifest["rdkit"] != IDENTITY_RDKIT:
        raise ValueError(
            f"Fingerprint cache requires format 1 and RDKit {IDENTITY_RDKIT}.",
        )
    selected = cast(
        "NDArray[np.int64]",
        np.load(fingerprints / "fp_bits.npy"),
    )
    cache = parquet.read_table(fingerprints / "fingerprints.parquet")
    strings = from_plain(cast(object, cache.column("smiles").to_pylist()), list[str])
    positions = {value: row for row, value in enumerate(strings)}
    valid = from_plain(cast(object, cache.column("valid").to_pylist()), list[bool])
    cached_mass = np.array(
        from_plain(cast(object, cache.column("mass").to_pylist()), list[float]),
    )
    cached_fp = cast(
        "NDArray[np.uint8]",
        np.load(fingerprints / "fingerprints.npy", mmap_mode="r"),
    )
    width = (len(selected) + 7) // 8
    if cached_fp.dtype != np.uint8 or cached_fp.shape != (len(strings), width):
        raise ValueError("Fingerprint cache rows or packed width differ from metadata.")
    first = read_pool_metadata(coconut / "coco_meta.pkl")
    second = read_pool_metadata(bio / "bio_meta.pkl")
    included = set(first["keys"])
    bio_rows = [row for row, key in enumerate(second["keys"]) if key not in included]
    keys = [*first["keys"], *(second["keys"][row] for row in bio_rows)]
    smiles = [*first["smiles"], *(second["smiles"][row] for row in bio_rows)]
    origins = [0] * len(first["keys"]) + [1] * len(bio_rows)
    source_rows = [*range(len(first["keys"])), *bio_rows]
    first_mass = cast(
        "NDArray[np.float64]",
        np.load(coconut / "coco_mass.npy"),
    )
    second_mass = cast(
        "NDArray[np.float64]",
        np.load(bio / "bio_mass.npy"),
    )
    arrays = [
        cast(
            "NDArray[np.uint8]",
            np.load(coconut / "coco_fp.npy", mmap_mode="r"),
        ),
        cast(
            "NDArray[np.uint8]",
            np.load(bio / "bio_fp.npy", mmap_mode="r"),
        ),
        cached_fp,
    ]
    for index, metadata in enumerate((first, second)):
        if arrays[index].dtype != np.uint8 or arrays[index].shape != (
            len(metadata["keys"]),
            width,
        ):
            raise ValueError(
                "Provided fingerprint rows or packed width differ from metadata.",
            )
    if first_mass.shape != (len(first["keys"]),) or second_mass.shape != (
        len(second["keys"]),
    ):
        raise ValueError("Provided mass rows differ from metadata.")
    masses = [
        *from_plain(cast(object, first_mass.tolist()), list[float]),
        *(float(cast(np.float64, second_mass[row])) for row in bio_rows),
    ]
    seen = set(keys)
    training = parquet.ParquetFile(input_path)
    for batch in training.iter_batches(
        batch_size=256,
        columns=["inchikey14", "normalized_smiles"],
    ):
        key_rows = from_plain(
            cast(object, batch.column(0).to_pylist()),
            list[str | None],
        )
        smile_rows = from_plain(
            cast(object, batch.column(1).to_pylist()),
            list[str | None],
        )
        for row, key in enumerate(key_rows):
            value = smile_rows[row]
            if key is None or value is None or key in seen:
                continue
            seen.add(key)
            cached_row = positions[value]
            if valid[cached_row]:
                keys.append(key)
                smiles.append(value)
                masses.append(cast(float, cached_mass.item(cached_row)))
                origins.append(2)
                source_rows.append(cached_row)
    mass = np.array(masses)
    good = np.isfinite(mass)
    order = np.argsort(np.where(good, mass, 1e18), stable=True)[
        : int(cast(np.int64, good.sum()))
    ]
    indices = from_plain(cast(object, order.tolist()), list[int])
    output_fp = np.lib.format.open_memmap(
        destination / "pool_fp.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(order), width),
    )
    for row, index in enumerate(indices):
        output_fp[row] = arrays[origins[index]][source_rows[index]]
    output_fp.flush()
    np.save(destination / "pool_mass.npy", mass[order])
    np.save(destination / "fp_bits.npy", selected)
    parquet.write_table(
        Table.from_pydict(
            {
                "mass": mass[order],
                "key": [keys[index] for index in indices],
                "smiles": [smiles[index] for index in indices],
            },
        ),
        destination / "pool_meta.parquet",
    )
    _write_manifest(
        destination,
        inputs=paths,
        fields={
            "rows": len(order),
            "selected_bits": len(selected),
            "raw_bits": 10_407,
            "rdkit": IDENTITY_RDKIT,
        },
    )


def _write_manifest(
    destination: Path,
    *,
    inputs: Sequence[Path],
    fields: dict[str, object],
) -> None:
    fields.update(
        format_version=1,
        numpy=np.__version__,
        inputs=input_digests(inputs),
        generator=source_digests(
            (
                _THIS,
                _THIS.with_name("rdkit_backend.py"),
                _THIS.parent.parent / "chemistry.py",
            ),
        ),
    )
    write_manifest(destination, fields)
