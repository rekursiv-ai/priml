"""Numerical molecular tables, spectra and reaction declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override

import math

from configgle import Fig
from numpy.typing import NDArray
from treekle import from_plain, json_loads

import numpy as np

from priml.baselines.casmi2026.custom_types import (
    IdentityRow,
    MetricIdentity,
    PreparationHeader,
)
from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, formula_vector
from priml.data.arrays import load_array
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyarrow import Table, parquet

    from priml.baselines.casmi2026.custom_types import FloatArray
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")


# Ordered transfers retain the public table's floating-point grouping.


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ChemicalTable:
    """All generated molecular identities, properties and raw fingerprint bits."""

    input_smiles: tuple[str, ...]
    smiles: tuple[str, ...]
    valid: NDArray[np.bool_]
    formula: tuple[str, ...]
    mass: NDArray[np.float64]
    heavy_atoms: NDArray[np.int32]
    keys: tuple[str, ...]
    raw_keys: tuple[str, ...]
    metric_keys: tuple[str, ...]
    compositions: tuple[tuple[tuple[str, int], ...], ...]
    packed: NDArray[np.uint8]
    raw_bits: int
    _positions: dict[str, int] = field(repr=False)

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Read the complete generated Arrow table and packed raw fingerprints.

        Args:
          directory: Prepared structures.parquet, raw_fp.npy and fingerprint_bits.txt.

        Returns:
          table: All generated rows and the exact raw fingerprint layout.

        """
        metadata = parquet.read_table(directory / "structures.parquet")
        columns = {
            name: tuple(
                from_plain(cast(object, metadata.column(name).to_pylist()), list[str]),
            )
            for name in (
                "input_smiles",
                "smiles",
                "formula",
                "key",
                "raw_key",
                "metric_key",
                "composition",
            )
        }
        raw_bits = int((directory / "fingerprint_bits.txt").read_text())
        if raw_bits <= 0:
            raise ValueError("Generated fingerprint width must be positive.")
        packed = load_array(directory / "raw_fp.npy", np.uint8)
        if packed.shape != (metadata.num_rows, (raw_bits + 7) // 8):
            raise ValueError("Generated fingerprint shape differs from metadata.")
        return cls(
            input_smiles=columns["input_smiles"],
            smiles=columns["smiles"],
            valid=np.asarray(
                from_plain(
                    cast(object, metadata.column("valid").to_pylist()),
                    list[bool],
                ),
                dtype=np.bool_,
            ),
            formula=columns["formula"],
            mass=np.asarray(
                from_plain(
                    cast(object, metadata.column("mass").to_pylist()),
                    list[float],
                ),
            ),
            heavy_atoms=np.asarray(
                from_plain(
                    cast(object, metadata.column("heavy_atoms").to_pylist()),
                    list[int],
                ),
                dtype=np.int32,
            ),
            keys=columns["key"],
            raw_keys=columns["raw_key"],
            metric_keys=columns["metric_key"],
            compositions=tuple(
                json_loads(value, tuple[tuple[str, int], ...])
                for value in columns["composition"]
            ),
            packed=packed,
            raw_bits=raw_bits,
            _positions={
                value: index for index, value in enumerate(columns["input_smiles"])
            },
        )

    def fingerprints(
        self,
        rows: NDArray[np.int64],
        bits: NDArray[np.int64],
    ) -> NDArray[np.uint8]:
        """Select complete declared fingerprint columns in caller row order.

        Args:
          rows: Requested generated rows, including duplicates.
          bits: Complete selected raw-bit indices in the model's declared order.

        Returns:
          fingerprints: All requested binary fingerprint rows and columns.

        """
        if np.any(bits < 0) or np.any(bits >= self.raw_bits):
            raise ValueError("Selected fingerprint bits exceed the generated layout.")
        return np.unpackbits(self.packed[rows], axis=1)[:, bits]

    def metric_identity(
        self,
        smiles: str,
    ) -> tuple[str, tuple[tuple[str, int], ...]] | None:
        """Read official identity, rejecting structures absent from preparation.

        Args:
          smiles: Exact source candidate or truth structure from the generated table.

        Returns:
          identity: Official metric key and heavy composition, or None if invalid.

        """
        index = self._positions[smiles]
        if not self.valid.item(index):
            return None
        return self.metric_keys[index], self.compositions[index]


def instrument_family(value: object) -> int:
    """Map library instrument labels into the six main-engine model families.

    Args:
      value: Original instrument label, including missing values.

    Returns:
      token: timsTOF, Orbitrap, QTOF, ion trap, triple quadrupole, or other.

    """
    text = str(value).lower()
    families = (
        ("timstof",),
        ("orbitrap", "qft", "ftms", "hybrid ft", "itft", "exactive", "ft-icr", "fticr"),
        ("tof",),
        ("trap", "qit", "lit"),
        ("qq", "quattro", "triple"),
    )
    return next(
        (
            index
            for index, names in enumerate(families)
            if any(name in text for name in names)
        ),
        len(families),
    )


def clean_block(
    offsets: NDArray[np.int64],
    *,
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    precursor: NDArray[np.float64],
    floor: float,
    topk: int,
    over: float,
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]]:
    """Filter and normalize a complete CSR block in spectrum encounter order.

    Args:
      offsets: Raw CSR boundaries with one final sentinel.
      mz: Raw flattened mass-to-charge values.
      intensity: Aligned raw peak intensities.
      precursor: Measured precursor for each spectrum.
      floor: Minimum intensity relative to the eligible base peak.
      topk: Maximum retained peaks per spectrum.
      over: Allowed m/z above each precursor.

    Returns:
      boundaries: Cleaned int64 CSR boundaries.
      masses: Sorted float32 mass-to-charge values.
      values: Float32 intensities normalized to eligible base peak one.

    """
    boundaries = np.zeros(len(offsets), dtype=np.int64)
    masses = np.empty(len(mz), dtype=np.float32)
    values = np.empty(len(mz), dtype=np.float32)
    for row in range(len(precursor)):
        limit = precursor.item(row)
        start, stop = offsets.item(row), offsets.item(row + 1)
        raw_mz, raw_it = mz[start:stop], intensity[start:stop]
        allowed = raw_mz <= limit + over
        maximum = raw_it[allowed & (raw_it > 0)].max(initial=0.0).item()
        indices = np.flatnonzero(allowed & (raw_mz > 0) & (raw_it >= floor * maximum))
        if maximum <= 0:
            indices = indices[:0]
        if len(indices) > topk:
            indices = indices[np.argsort(raw_it[indices], kind="stable")[-topk:]]
        order = np.argsort(raw_mz[indices], kind="stable")
        begin = boundaries.item(row)
        end = begin + len(indices)
        masses[begin:end] = raw_mz[indices][order]
        values[begin:end] = raw_it[indices][order] / maximum
        boundaries[row + 1] = end
    return boundaries, masses[: boundaries.item(-1)], values[: boundaries.item(-1)]


class IdentityTable:
    """Require complete prepared rows; known rejection and missing input differ."""

    class Config(Fig["IdentityTable"]):
        base_dir: Path | str | None = None
        """Parent-supplied dataset root."""

        working_dir: Path | str = "native-chemistry/identities"
        """Complete independently prepared molecular rows."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        root = Path(config.working_dir)
        header = json_loads((root / "manifest.json").read_text(), _Header)
        if (
            header["format_version"] != 1
            or header["rdkit"] != IDENTITY_RDKIT
            or header["raw_bits"] != 16_551
        ):
            raise ValueError(
                f"Identity data requires format 1, RDKit {IDENTITY_RDKIT} and 16551 bits.",
            )
        self.metadata = tuple(
            from_plain(
                parquet.read_table(root / "molecules.parquet").to_pylist(),
                list[IdentityRow],
            ),
        )
        self.packed = load_array(root / "fingerprints.npy", np.uint8, mmap=True)
        if self.packed.shape != (len(self.metadata), 2069):
            raise ValueError(
                "Identity fingerprints require complete U8[rows,2069] layout.",
            )
        self._positions: dict[str, int] = {}
        self._products: dict[str, int] = {}
        self._keys: dict[str, str | None] = {}
        self._formulas: dict[str, str] = {}
        self._identities: dict[str, MetricIdentity | None] = {}
        for index, row in enumerate(self.metadata):
            value = row["smiles"]
            self._positions.setdefault(value, index)
            self._keys.setdefault(value, row["key"])
            self._formulas.setdefault(value, row["formula"])
            self._identities.setdefault(
                value,
                cast(  # pyrefly: ignore[redundant-cast] -- ty or basedpyright needs this cast; pyrefly already infers it.
                    "MetricIdentity | None",
                    json_loads(row["metric"], MetricIdentity | None),
                ),
            )
        for index, row in enumerate(self.metadata):
            value = row["standardized"]
            if value is not None:
                self._products.setdefault(value, index)
                self._keys.setdefault(value, row["standard_key"])
                self._formulas.setdefault(value, row["standard_formula_lookup"])
                self._identities.setdefault(
                    value,
                    cast(  # pyrefly: ignore[redundant-cast] -- ty or basedpyright needs this cast; pyrefly already infers it.
                        "MetricIdentity | None",
                        json_loads(row["standard_metric"], MetricIdentity | None),
                    ),
                )

    def for_smiles(self, smiles: str) -> IdentityRow:
        """Return the first exact original input row."""
        return self.metadata[self._positions[smiles]]

    def standardize(self, smiles: str) -> str | None:
        """Return the explicit prepared standardization decision for a raw product."""
        return self.metadata[self._positions[smiles]]["standardized"]

    def key(self, smiles: str) -> str | None:
        """Return the raw score key, also admitting explicitly standardized aliases."""
        return self._keys[smiles]

    def formula(self, smiles: str) -> str:
        """Return the raw formula or input fallback, including standardized aliases."""
        return self._formulas[smiles]

    def identity(self, smiles: str) -> MetricIdentity | None:
        """Return the official stereo-free identity independently of fingerprints."""
        return self._identities[smiles]

    def properties(self, standardized: str) -> tuple[str, float, int] | None:
        """Return explicitly standardized product properties, independently of keys."""
        row = self.metadata[self._products[standardized]]
        if not row["properties_valid"]:
            return None
        return row["standard_formula"], row["standard_mass"], row["standard_heavy"]

    def fingerprint(self, standardized: str) -> NDArray[np.uint8] | None:
        """Return every packed product bit, independently of key validity."""
        index = self._products[standardized]
        return self.packed[index] if self.metadata[index]["fingerprint_valid"] else None


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Representatives:
    """Aligned representative-spectrum rows, sorted by labelled neutral mass."""

    indices: NDArray[np.int64]
    sid: NDArray[np.int64]
    mass: NDArray[np.float64]
    mode: NDArray[np.int8]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SpectralLibrary:
    """Hold cleaned peak arrays and the public structure-labelled mass index."""

    offsets: NDArray[np.int64]
    mz: NDArray[np.float32]
    intensity: NDArray[np.float32]
    structure_mass: NDArray[np.float64]
    keys: tuple[str, ...]
    smiles: tuple[str, ...]
    sid: NDArray[np.int64]
    library: NDArray[np.int8]
    mode: NDArray[np.int8]
    instrument: NDArray[np.int8]
    adduct: NDArray[np.int16]
    precursor: NDArray[np.float64]
    peak_count: NDArray[np.int32]
    mass: NDArray[np.float64]
    order: NDArray[np.int64]
    mass_sorted: NDArray[np.float64]
    _representatives: dict[str, Representatives] = field(
        default_factory=dict,
        repr=False,
    )

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Read every public cleaned-cache and structure-index field.

        Args:
          directory: Prepared spec arrays, metadata and public train_structs.parquet.

        Returns:
          library: Complete indexed library retaining spectrum encounter order.

        """
        metadata = parquet.read_table(directory / "spec_meta.parquet")
        structures = parquet.read_table(
            directory / "train_structs.parquet",
            columns=["sid", "mass", "key", "smiles"],
        )
        structure_mass = _numeric(structures, "mass", np.float64)
        sid = _numeric(metadata, "sid", np.int64)
        mass = structure_mass[sid]
        order = np.argsort(mass, kind="mergesort")
        return cls(
            offsets=load_array(directory / "spec_off.npy", np.int64),
            mz=load_array(directory / "spec_mz.npy", np.float32),
            intensity=load_array(directory / "spec_it.npy", np.float32),
            structure_mass=structure_mass,
            keys=_strings(structures, "key"),
            smiles=_strings(structures, "smiles"),
            sid=sid,
            library=_numeric(metadata, "lib", np.int8),
            mode=_numeric(metadata, "mode", np.int8),
            instrument=_numeric(metadata, "instr", np.int8),
            adduct=_numeric(metadata, "adduct_ix", np.int16),
            precursor=_numeric(metadata, "precursor_mz", np.float64),
            peak_count=_numeric(metadata, "n_clean", np.int32),
            mass=mass,
            order=order,
            mass_sorted=mass[order],
        )

    def window(
        self,
        target: float,
        *,
        ppm: float,
        excluded: NDArray[np.bool_] | None = None,
        mode: int | None = None,
    ) -> NDArray[np.int64]:
        """Select an inclusive labelled-mass window, then polarity and exclusions.

        Args:
          target: Neutral query mass.
          ppm: Half-width in parts per million of the target.
          excluded: True for each spectrum withheld from retrieval.
          mode: Optional positive or negative polarity selector.

        Returns:
          indices: Spectrum indices in stable increasing mass order.

        """
        tolerance = target * ppm * 1e-6
        left = int(np.searchsorted(self.mass_sorted, target - tolerance, side="left"))
        right = int(np.searchsorted(self.mass_sorted, target + tolerance, side="right"))
        indices = self.order[left:right]
        if mode is not None:
            indices = indices[self.mode[indices] == mode]
        return indices if excluded is None else indices[~excluded[indices]]

    def representatives(
        self,
        *,
        excluded: NDArray[np.bool_] | None = None,
        key: str = "default",
    ) -> Representatives:
        """Cache one most-populated spectrum per structure and polarity.

        Args:
          excluded: True for spectra unavailable to this representative view.
          key: Public cache identity; repeated keys retain their first view.

        Returns:
          representatives: Stable mass order with encounter-order peak-count ties.

        """
        if key not in self._representatives:
            indices = (
                np.arange(len(self.sid))
                if excluded is None
                else np.flatnonzero(~excluded)
            )
            groups = self.sid[indices] * 2 + (self.mode[indices] > 0)
            order = np.lexsort((-self.peak_count[indices], groups))
            ordered_groups = groups[order]
            first = np.ones(len(order), dtype=np.bool_)
            first[1:] = ordered_groups[1:] != ordered_groups[:-1]
            selected = indices[order[first]]
            selected = selected[np.argsort(self.mass[selected], kind="mergesort")]
            self._representatives[key] = Representatives(
                indices=selected,
                sid=self.sid[selected],
                mass=self.mass[selected],
                mode=self.mode[selected],
            )
        return self._representatives[key]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CandidatePool:
    """Hold the complete provided candidate table and its precomputed fragments."""

    mass: NDArray[np.float64]
    keys: tuple[str, ...]
    smiles: tuple[str, ...]
    formulas: tuple[str, ...]
    heavy_atoms: NDArray[np.int32]
    source: NDArray[np.int8]
    train_sid: NDArray[np.int64]
    packed: NDArray[np.uint8]
    bits: NDArray[np.int64]
    fragment_offsets: NDArray[np.int64]
    fragment_mass: NDArray[np.float32]
    key_to_pid: dict[str, int]
    _formula_vectors: dict[str, NDArray[np.int64] | None] = field(
        default_factory=dict,
        repr=False,
    )

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Read declared public pool tables, selected fingerprints and fragments.

        Args:
          directory: The provided V2 pool data directory; both fragment arrays exist.

        Returns:
          pool: Complete public candidate data in declared mass order.

        """
        metadata = parquet.read_table(directory / "pool_meta.parquet")
        keys = _strings(metadata, "key")
        return cls(
            mass=_numeric(metadata, "mass", np.float64),
            keys=keys,
            smiles=_strings(metadata, "smiles"),
            formulas=_strings(metadata, "formula"),
            heavy_atoms=_numeric(metadata, "n_heavy", np.int32),
            source=_numeric(metadata, "src", np.int8),
            train_sid=_numeric(metadata, "train_sid", np.int64),
            packed=load_array(directory / "pool_fp.npy", np.uint8),
            bits=load_array(directory / "fp_bits.npy", np.int64),
            fragment_offsets=load_array(directory / "pool_frag_off.npy", np.int64),
            fragment_mass=load_array(directory / "pool_frag_mass.npy", np.float32),
            key_to_pid={key: index for index, key in enumerate(keys)},
        )

    def window(self, target: float, *, ppm: float) -> NDArray[np.int64]:
        """Return an inclusive neutral-mass window in the provided sorted pool."""
        tolerance = target * ppm * 1e-6
        return np.arange(
            np.searchsorted(self.mass, target - tolerance, side="left"),
            np.searchsorted(self.mass, target + tolerance, side="right"),
        )

    def fingerprints(self, indices: NDArray[np.int64]) -> NDArray[np.uint8]:
        """Unpack complete selected-bit rows in caller order, retaining duplicates."""
        return np.unpackbits(self.packed[indices], axis=1)[:, : len(self.bits)]

    def fragments(self, index: int) -> NDArray[np.float64]:
        """Return the complete provided fragment-mass row in double precision."""
        begin, end = (
            self.fragment_offsets.item(index),
            self.fragment_offsets.item(index + 1),
        )
        return self.fragment_mass[begin:end].astype(np.float64)

    def formula_vector(self, index: int) -> NDArray[np.int64] | None:
        """Cache C/H/N/O/P/S/F/Cl/Br/I counts, or None for unsupported formulas.

        Args:
          index: Candidate row in the provided pool.

        Returns:
          counts: Complete ten-element count vector, including explicit zero counts.

        """
        formula = self.formulas[index]
        if formula not in self._formula_vectors:
            counts = formula_vector(formula)
            self._formula_vectors[formula] = (
                None if counts.item(0) < 0 else counts.astype(np.int64)
            )
        return self._formula_vectors[formula]


class OfficialIdentity:
    """Read stereo-free canonical keys and heavy composition without chemistry imports."""

    class Config(Fig["OfficialIdentity"]):
        base_dir: Path | str | None = None
        """Parent-supplied resource root."""

        working_dir: Path | str = "/datasets/casmi2026/native-chemistry/main"
        """Complete independently generated identity/property/fingerprint cache."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        directory = Path(config.working_dir)
        manifest = json_loads(
            (directory / "manifest.json").read_text(),
            dict[str, object],
        )
        version = from_plain(manifest["rdkit"], str)
        if version != IDENTITY_RDKIT:
            raise ValueError(
                f"Official identity data requires RDKit {IDENTITY_RDKIT}; got {version}.",
            )
        self.table = ChemicalTable.load(directory)

    def __call__(self, smiles: str) -> tuple[str, tuple[tuple[str, int], ...]] | None:
        """Return a prepared metric identity and require explicit input coverage.

        Args:
          smiles: Exact structure string covered by the independent preparation.

        Returns:
          identity: Official key and heavy composition, or None for a known invalid row.

        """
        return self.table.metric_identity(smiles)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class PeakStore:
    """Carry complete CSR peaks, or the second engine's prepared peak weights."""

    offsets: NDArray[np.int64]
    mz: NDArray[np.float32]
    intensity: NDArray[np.float32]

    def row(self, index: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return both complete peak arrays of one CSR row."""
        begin, end = self.offsets.item(index), self.offsets.item(index + 1)
        return self.mz[begin:end], self.intensity[begin:end]


class MainSearch:
    """Score double-precision entropy, cosine and direct-plus-shifted hybrids."""

    class Config(Fig["MainSearch"]):
        floor: float = 0.002
        """Relative intensity threshold for matching."""

        max_peaks: int = 256
        """Maximum intensity-selected peaks in each query and candidate."""

        power: float = 1.0
        """Intensity exponent before normalization."""

        entropy_weight: bool = True
        """Apply low-entropy intensity weighting."""

        tolerance_da: float = 0.01
        """Minimum absolute peak matching tolerance."""

        tolerance_ppm: float = 20.0
        """Relative query-peak tolerance, taking the larger bound."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def score(
        self,
        query: tuple[FloatArray, FloatArray],
        *,
        store: PeakStore,
        candidates: NDArray[np.int64],
        shifts: NDArray[np.float64] | None = None,
        cosine: bool = False,
    ) -> NDArray[np.float32]:
        """Score every requested library row in its supplied order.

        Args:
          query: Raw mass and intensity arrays in sorted mass order.
          store: Raw cleaned public library CSR arrays.
          candidates: Requested spectrum rows, including duplicates.
          shifts: Optional candidate-to-query neutral-mass offsets.
          cosine: Use the source cosine channel for unshifted search.

        Returns:
          scores: Complete float32 score vector in candidate order.

        """
        prepared = self.prepare(query)
        result = np.zeros(len(candidates), dtype=np.float32)
        for position in range(len(candidates)):
            raw = store.row(candidates.item(position))
            if not len(raw[0]):
                continue
            candidate = self.prepare(raw)
            if not len(candidate[0]):
                continue
            matches = _matches(
                prepared[0],
                candidate[0],
                tolerance=self.config.tolerance_da,
                ppm=self.config.tolerance_ppm,
            )
            if shifts is not None:
                result[position] = self._hybrid(
                    prepared,
                    candidate,
                    matches=matches,
                    shift=shifts.item(position),
                )
            elif cosine:
                denominator = float(
                    np.sum(prepared[1] ** 2) * np.sum(candidate[1] ** 2),
                )
                dot = sum(
                    prepared[1].item(i) * candidate[1].item(j)
                    for i, j in matches
                    if i >= 0 and j >= 0
                )
                result[position] = dot / denominator**0.5 if denominator > 0 else 0.0
            else:
                result[position] = _main_similarity(
                    prepared[1],
                    candidate[1],
                    matches=matches,
                )
        return result

    def prepare(
        self,
        peaks: tuple[FloatArray, FloatArray],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Prepare complete double-precision masses and entropy-weighted intensities.

        Args:
          peaks: Sorted raw mass and intensity arrays.

        Returns:
          masses: Every retained mass in encounter order.
          weights: Corresponding complete normalized intensity weights.

        """
        masses, intensity = (np.asarray(array, dtype=np.float64) for array in peaks)
        selected = _selected(
            intensity,
            floor=self.config.floor,
            maximum=self.config.max_peaks,
        )
        weights = np.array(
            [
                intensity.item(selected.item(index)) ** self.config.power
                for index in range(len(selected))
            ],
            dtype=np.float64,
        )
        total = weights.sum()
        if total > 0:
            weights *= 1.0 / total
            entropy = _entropy64(weights)
            if self.config.entropy_weight and entropy < 3.0:
                weights = np.array(
                    [
                        weights.item(index) ** (0.25 + 0.25 * entropy)
                        for index in range(len(weights))
                    ],
                    dtype=np.float64,
                )
                total = weights.sum()
                if total > 0:
                    weights *= 1.0 / total
        return masses[selected], weights

    def similarity(
        self,
        query: tuple[NDArray[np.float64], NDArray[np.float64]],
        candidate: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> float:
        """Compare prepared spectra without narrowing the similarity to float32.

        Args:
          query: Prepared query masses and weights.
          candidate: Prepared prediction masses and weights.

        Returns:
          similarity: Complete double entropy similarity with greedy peak matching.

        """
        matches = _matches(
            query[0],
            candidate[0],
            tolerance=self.config.tolerance_da,
            ppm=self.config.tolerance_ppm,
        )
        return _main_similarity(query[1], candidate[1], matches=matches)

    def _hybrid(
        self,
        query: tuple[NDArray[np.float64], NDArray[np.float64]],
        candidate: tuple[NDArray[np.float64], NDArray[np.float64]],
        *,
        matches: list[tuple[int, int]],
        shift: float,
    ) -> float:
        if not len(query[0]) or not len(candidate[0]):
            return 0.0
        if -1e-4 < shift < 1e-4:
            return _main_similarity(query[1], candidate[1], matches=matches)
        used = np.zeros(len(query[0]), dtype=np.bool_)
        assigned = np.zeros(len(candidate[0]), dtype=np.float64)
        for i, j in matches:
            if i >= 0 and j >= 0:
                used[i] = True
                assigned[j] = query[1].item(i)
        shifted = _matches(
            query[0],
            candidate[0] + shift,
            tolerance=self.config.tolerance_da,
            ppm=self.config.tolerance_ppm,
        )
        for i, j in shifted:
            if i >= 0 and j >= 0 and not used.item(i) and assigned.item(j) == 0:
                used[i] = True
                assigned[j] = query[1].item(i)
        mixture = np.concatenate(
            [(candidate[1] + assigned) * 0.5, query[1][~used] * 0.5],
        )
        difference = (
            _entropy64(query[1]) + _entropy64(candidate[1]) - 2 * _entropy64(mixture)
        )
        value = difference * (1.0 / math.log(4.0)) + 1.0
        return 0.0 if value < 0 else 1.0 if value > 1 else value


class SecondSearch:
    """Retain float32 weights and maximum-of-direct-or-wholly-shifted entropy."""

    class Config(Fig["SecondSearch"]):
        floor: float = 0.002
        """Relative base-peak threshold."""

        max_peaks: int = 256
        """Maximum retained peaks in original encounter order."""

        power: float = 1.0
        """Intensity exponent before float32 weight storage."""

        entropy_weight: bool = True
        """Apply low-entropy intensity reweighting."""

        tolerance_da: float = 0.01
        """Absolute greedy peak matching tolerance."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def prepare(
        self,
        peaks: tuple[NDArray[np.float32], NDArray[np.float32]],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Prepare the complete float32 query or reusable representative spectrum.

        Args:
          peaks: Raw sorted float32 mass and intensity arrays.

        Returns:
          masses: Retained float32 masses in source encounter order.
          weights: Complete normalized and optionally entropy-weighted float32 values.

        """
        selected = _selected(
            peaks[1],
            floor=self.config.floor,
            maximum=self.config.max_peaks,
        )
        powered = np.array(
            [
                peaks[1].item(selected.item(index)) ** self.config.power
                for index in range(len(selected))
            ],
            dtype=np.float64,
        )
        weights = powered.astype(np.float32)
        total = powered.sum()
        if total > 0:
            weights = (weights.astype(np.float64) / total).astype(np.float32)
        entropy = _entropy32(weights)
        if self.config.entropy_weight and entropy < 3.0:
            weights = np.array(
                [
                    weights.item(index) ** (0.25 + 0.25 * entropy)
                    for index in range(len(weights))
                ],
                dtype=np.float32,
            )
            total = weights.sum(dtype=np.float64)
            if total > 0:
                weights = (weights.astype(np.float64) / total).astype(np.float32)
        return peaks[0][selected], weights

    def clean_store(self, store: PeakStore, *, indices: NDArray[np.int64]) -> PeakStore:
        """Prepare every selected representative and pack complete reusable arrays.

        Args:
          store: Raw public library peak arrays.
          indices: Representative rows in the declared representative order.

        Returns:
          prepared: Full cleaned CSR store, including empty representative rows.

        """
        rows = [
            self.prepare(store.row(indices.item(index)))
            for index in range(len(indices))
        ]
        offsets = np.concatenate(
            [[0], np.cumsum([len(row[0]) for row in rows])],
        ).astype(np.int64)
        masses = np.empty(offsets.item(-1), dtype=np.float32)
        weights = np.empty_like(masses)
        for index, (mass, intensity) in enumerate(rows):
            begin, end = offsets.item(index), offsets.item(index + 1)
            masses[begin:end] = mass
            weights[begin:end] = intensity
        return PeakStore(
            offsets=offsets,
            mz=masses,
            intensity=weights,
        )

    def score(
        self,
        query: tuple[NDArray[np.float32], NDArray[np.float32]],
        *,
        store: PeakStore,
        candidates: NDArray[np.int64],
        shifts: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Score raw library rows against a prepared query using this family's rule.

        Args:
          query: Prepared float32 masses and weights.
          store: Raw public library peak arrays.
          candidates: Complete requested library row order.
          shifts: Optional precursor offsets, applied to whole candidate spectra.

        Returns:
          scores: Unclipped float32 entropy scores in candidate order.

        """
        result = np.zeros(len(candidates), dtype=np.float32)
        for position in range(len(candidates)):
            candidate = self.prepare(store.row(candidates.item(position)))
            if len(candidate[0]):
                result[position] = self._similarity(
                    query,
                    candidate,
                    shift=shifts.item(position) if shifts is not None else 0.0,
                )
        return result

    def score_prepared(
        self,
        query: tuple[NDArray[np.float32], NDArray[np.float32]],
        *,
        store: PeakStore,
        start: int,
        stop: int,
        shifts: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Score a contiguous mass window of the prepared representative store.

        Args:
          query: Prepared query mass and weight vectors.
          store: Prepared representative CSR arrays.
          start: First included representative.
          stop: One past the final representative.
          shifts: Corresponding float32 neutral-mass offsets.

        Returns:
          scores: Full float32 window scores in representative order.

        """
        result = np.zeros(stop - start, dtype=np.float32)
        for position, index in enumerate(range(start, stop)):
            candidate = store.row(index)
            if len(candidate[0]):
                result[position] = self._similarity(
                    query,
                    candidate,
                    shift=shifts.item(position),
                )
        return result

    def _similarity(
        self,
        query: tuple[NDArray[np.float32], NDArray[np.float32]],
        candidate: tuple[NDArray[np.float32], NDArray[np.float32]],
        *,
        shift: float,
    ) -> float:
        direct = _second_similarity(
            query,
            candidate,
            tolerance=self.config.tolerance_da,
        )
        if -0.001 < shift < 0.001:
            return direct
        shifted = (
            (candidate[0].astype(np.float64) + shift).astype(np.float32),
            candidate[1],
        )
        return max(
            direct,
            _second_similarity(query, shifted, tolerance=self.config.tolerance_da),
        )


def clean_main(
    peaks: tuple[FloatArray, FloatArray],
    *,
    precursor: float,
    floor: float,
    max_peaks: int,
    over: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Clean query peaks with the first engine's precursor and base-peak rules.

    Args:
      peaks: Complete original masses and linear peak intensities.
      precursor: Original precursor mass, before the explicit overage allowance.
      floor: Relative threshold against every intensity below the precursor cap.
      max_peaks: Complete intensity-ranked peak cap; ties keep the later peak.
      over: Maximum mass above the precursor admitted to the query.

    Returns:
      masses: Retained float32 masses in stable mass order.
      intensity: Float32 linear intensities normalized by the original base peak.

    """
    masses, intensity = (np.asarray(array, dtype=np.float64) for array in peaks)
    _, cleaned_mass, cleaned_intensity = clean_block(
        np.array([0, len(masses)], dtype=np.int64),
        mz=masses,
        intensity=intensity,
        precursor=np.array([precursor], dtype=np.float64),
        floor=floor,
        topk=max_peaks,
        over=over,
    )
    return cleaned_mass, cleaned_intensity


def merge_main(
    spectra: Sequence[tuple[FloatArray, FloatArray]],
    *,
    tolerance: float,
    max_peaks: int | None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Merge normalized spectra by comparing each peak to the last survivor.

    Args:
      spectra: Complete input mass and intensity pairs.
      tolerance: Strict near-duplicate mass distance.
      max_peaks: Optional final intensity-ranked cap.

    Returns:
      masses: Sorted surviving float32 peak masses.
      intensity: Corresponding per-spectrum normalized float32 intensities.

    """
    present = [
        (np.asarray(mz, dtype=np.float64), np.asarray(values, dtype=np.float64))
        for mz, values in spectra
        if len(mz) and values.max() > 0
    ]
    if not present:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    masses = np.concatenate([mz for mz, _ in present])
    intensity = np.concatenate([values / values.max() for _, values in present])
    order = np.argsort(masses, kind="stable")
    masses, intensity = masses[order], intensity[order]
    keep = np.ones(len(masses), dtype=np.bool_)
    survivor = 0
    for index in range(1, len(masses)):
        if masses.item(index) - masses.item(survivor) < tolerance:
            rejected = (
                survivor if intensity.item(index) >= intensity.item(survivor) else index
            )
            keep[rejected] = False
            if rejected == survivor:
                survivor = index
        else:
            survivor = index
    masses, intensity = masses[keep], intensity[keep]
    if max_peaks is not None and len(masses) > max_peaks:
        selected = np.sort(np.argsort(-intensity, kind="stable")[:max_peaks])
        masses, intensity = masses[selected], intensity[selected]
    return masses.astype(np.float32), intensity.astype(np.float32)


def _selected(values: FloatArray, *, floor: float, maximum: int) -> NDArray[np.int64]:
    highest = values[values > 0].max(initial=0).item()
    selected = (
        np.flatnonzero(values >= floor * highest)
        if highest > 0
        else np.empty(0, dtype=np.int64)
    )
    if len(selected) > maximum:
        order = np.argsort(values[selected], kind="stable")
        selected = np.sort(selected[order[-maximum:]])
    return selected


def _entropy64(weights: NDArray[np.float64]) -> float:
    positive = weights[weights > 0]
    # `fsum` rounds once; `ndarray.sum()` misses the exact entropy by up to 4.2 ulp.
    return -math.fsum(value * math.log(value) for value in positive.tolist())


def _entropy32(weights: NDArray[np.float32]) -> float:
    positive = weights[weights > 0]
    return -float(
        (positive * np.log(positive.astype(np.float64)).astype(np.float32)).sum(
            dtype=np.float64,
        ),
    )


def _matches(
    query: FloatArray,
    candidate: FloatArray,
    *,
    tolerance: float,
    ppm: float = 0.0,
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    i = j = 0
    while i < len(query) and j < len(candidate):
        bound = max(tolerance, query.item(i) * ppm * 1e-6)
        difference = query.item(i) - candidate.item(j)
        if difference < -bound:
            matches.append((i, -1))
            i += 1
        elif difference > bound:
            matches.append((-1, j))
            j += 1
        else:
            matches.append((i, j))
            i += 1
            j += 1
    matches.extend((index, -1) for index in range(i, len(query)))
    matches.extend((-1, index) for index in range(j, len(candidate)))
    return matches


def _main_similarity(
    query: NDArray[np.float64],
    candidate: NDArray[np.float64],
    *,
    matches: list[tuple[int, int]],
) -> float:
    if not len(query) or not len(candidate):
        return 0.0
    mixture = np.array(
        [
            (query.item(i) if i >= 0 else 0.0) * 0.5
            + (candidate.item(j) if j >= 0 else 0.0) * 0.5
            for i, j in matches
        ],
        dtype=np.float64,
    )
    difference = _entropy64(query) + _entropy64(candidate) - 2 * _entropy64(mixture)
    value = difference * (1.0 / math.log(4.0)) + 1.0
    return 0.0 if value < 0 else 1.0 if value > 1 else value


def _second_similarity(
    query: tuple[NDArray[np.float32], NDArray[np.float32]],
    candidate: tuple[NDArray[np.float32], NDArray[np.float32]],
    *,
    tolerance: float,
) -> float:
    matches = _matches(query[0], candidate[0], tolerance=tolerance)
    mixture = np.array(
        [
            float(
                np.float32(
                    (query[1].item(i) if i >= 0 else 0.0)
                    + (candidate[1].item(j) if j >= 0 else 0.0),
                ),
            )
            for i, j in matches
        ],
        dtype=np.float64,
    )
    total = mixture.sum()
    return (
        (
            _entropy32(query[1])
            + _entropy32(candidate[1])
            - 2 * _entropy64(mixture * (1.0 / total))
        )
        * (1.0 / math.log(4.0))
        + 1.0
        if total > 0
        else 0.0
    )


class _Header(PreparationHeader):
    raw_bits: int


def _numeric[S: np.generic](table: Table, name: str, dtype: type[S]) -> NDArray[S]:
    values = from_plain(
        cast(object, table.column(name).to_pylist()),
        list[float | int | None],
    )
    return np.asarray(values, dtype=dtype)


def _strings(table: Table, name: str) -> tuple[str, ...]:
    return tuple(from_plain(cast(object, table.column(name).to_pylist()), list[str]))
