"""Native fingerprint fine-tuning data: the spectral cache and its preparation."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, TypedDict, cast, override

import hashlib
import json
import math
import tempfile

from configgle import Fig
from torch import Tensor
from treekle import ReadError, from_plain, json_loads, to_plain

import numpy as np
import torch

from priml.baselines.casmi2026.chemistry import (
    IdentityTable,
    clean_block,
    instrument_family,
)
from priml.baselines.casmi2026.isolated import adduct_index, neutral_mass
from priml.baselines.casmi2026.peaks import FPNetInputs, pack_inputs, prepare_peaks
from priml.data.arrays import load_array
from priml.data.passes import Passes
from priml.model.ae.checkpoint import sha256_file
from priml.paths import resolve_working_dir, validated_output_path
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from numpy.typing import NDArray
    from pyarrow import Table, parquet
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")


def build_spectral_cache(
    training: Path,
    *,
    structures: Path,
    output: Path,
    intensity_floor: float,
    max_peaks: int,
    precursor_margin: float,
) -> None:
    """Build the full public cleaned CSR cache from declared data inputs.

    Args:
      training: Public raw training parquet with ragged peak and metadata columns.
      structures: Provided public inchikey14-to-sid mapping table.
      output: Destination for spec_meta.parquet and the three numeric CSR arrays.
      intensity_floor: Minimum intensity relative to the eligible base peak.
      max_peaks: Maximum peaks retained per spectrum.
      precursor_margin: Allowed m/z above each precursor.

    """
    destination = validated_output_path(output, protected=(training, structures))
    destination.mkdir(parents=True, exist_ok=True)
    mapping = parquet.read_table(structures, columns=["inchikey14", "sid"])
    keys = from_plain(cast(object, mapping.column("inchikey14").to_pylist()), list[str])
    sids = from_plain(cast(object, mapping.column("sid").to_pylist()), list[int])
    key_to_sid = dict(zip(keys, sids, strict=True))
    source = parquet.ParquetFile(training)
    offsets: list[NDArray[np.int64]] = [np.zeros(1, np.int64)]
    masses: list[NDArray[np.float32]] = []
    intensities: list[NDArray[np.float32]] = []
    columns: dict[str, list[object]] = {}
    total = 0
    for group in range(source.num_row_groups):
        table = source.read_row_group(group)
        raw_mz = from_plain(
            cast(object, table.column("ms2_mzs").to_pylist()),
            list[list[float]],
        )
        raw_it = from_plain(
            cast(object, table.column("ms2_normalized_intensities").to_pylist()),
            list[list[float]],
        )
        boundaries = np.concatenate(
            [[0], np.cumsum([len(peaks) for peaks in raw_mz])],
        ).astype(np.int64)
        precursor = np.asarray(
            from_plain(
                cast(object, table.column("precursor_mz").to_pylist()),
                list[float],
            ),
            dtype=np.float64,
        )
        cleaned = clean_block(
            boundaries,
            mz=np.asarray(
                [mass for peaks in raw_mz for mass in peaks],
                dtype=np.float64,
            ),
            intensity=np.asarray(
                [value for peaks in raw_it for value in peaks],
                dtype=np.float64,
            ),
            precursor=precursor,
            floor=intensity_floor,
            topk=max_peaks,
            over=precursor_margin,
        )
        boundaries, mass, intensity = cleaned
        offsets.append(boundaries[1:] + total)
        masses.append(mass)
        intensities.append(intensity)
        total += len(mass)
        metadata = _metadata(table, counts=np.diff(boundaries), key_to_sid=key_to_sid)
        for name, values in metadata.items():
            columns.setdefault(name, []).extend(values)
    typed: dict[str, object] = dict(columns)
    for name, dtype in (
        ("sid", np.int64),
        ("lib", np.int8),
        ("instr", np.int8),
        ("adduct_ix", np.int16),
        ("mode", np.int8),
        ("n_clean", np.int32),
        ("ce_n", np.int64),
    ):
        typed[name] = np.asarray(columns[name], dtype=dtype)
    parquet.write_table(
        Table.from_pydict(typed),
        where=destination / "spec_meta.parquet",
    )
    np.save(destination / "spec_off.npy", np.concatenate(offsets))
    np.save(destination / "spec_mz.npy", np.concatenate(masses))
    np.save(destination / "spec_it.npy", np.concatenate(intensities))


def _metadata(
    table: Table,
    *,
    counts: NDArray[np.int64],
    key_to_sid: dict[str, int],
) -> dict[str, list[object]]:
    columns = {
        name: from_plain(cast(object, table.column(name).to_pylist()), list[object])
        for name in (
            "adduct",
            "precursor_mz",
            "precursor_error_ppm",
            "collision_energy_orig_units",
            "base_peak_intensity",
            "num_peaks",
        )
    }
    energies = from_plain(
        cast(object, table.column("collision_energy_ev").to_pylist()),
        list[list[float | None] | None],
    )
    for name, reduction in (("mean", np.mean), ("min", np.min), ("max", np.max)):
        values = [
            float(reduction(np.asarray(energy, dtype=np.float64)))
            if energy is not None and len(energy)
            else math.nan
            for energy in energies
        ]
        columns[f"ce_{name}"] = [
            None if math.isnan(value) else value for value in values
        ]
    columns["ce_n"] = [len(energy) if energy is not None else 0 for energy in energies]
    keys = from_plain(cast(object, table.column("inchikey14").to_pylist()), list[str])
    columns["sid"] = [key_to_sid.get(key, -1) for key in keys]
    libraries = (
        "enveda-180",
        "pluskal_ms2",
        "riken",
        "gnps",
        "massbank",
        "mona",
        "spectraverse",
        "msdial",
        "drug_plus",
        "enveda-np-examples",
        "masaryk",
    )
    columns["lib"] = [
        libraries.index(name)
        for name in from_plain(
            cast(object, table.column("ingest_lib").to_pylist()),
            list[str],
        )
    ]
    columns["instr"] = [
        instrument_family(value)
        for value in from_plain(
            cast(object, table.column("instrument_type").to_pylist()),
            list[object],
        )
    ]
    adducts = from_plain(
        cast(object, table.column("adduct").to_pylist()),
        list[str | None],
    )
    columns["adduct_ix"] = [adduct_index(name) for name in adducts]
    modes = from_plain(
        cast(object, table.column("ionization_mode").to_pylist()),
        list[str | None],
    )
    columns["mode"] = [1 if mode == "positive" else -1 for mode in modes]
    masses = neutral_mass(
        np.asarray(columns["precursor_mz"], dtype=np.float64),
        adducts,
    )
    columns["nm"] = [
        None if math.isnan(value) else value
        for value in from_plain(cast(object, masses.tolist()), list[float])
    ]
    columns["n_clean"] = from_plain(cast(object, counts.tolist()), list[object])
    return columns


class SpectrumData:
    """Read the public CSR cache and split every spectrum by molecular identity."""

    class Config(Fig["SpectrumData"]):
        base_dir: Path | str | None = None
        """Parent-supplied resource root."""

        working_dir: Path | str = "/datasets/casmi2026/training/work"
        """Prepared spec arrays, metadata, structure mapping, and selected fingerprints."""

        batch_size: int = 64
        """Spectra in one optimizer update."""

        validation_modulus: int = 10
        """Number of stable SHA-256 molecule buckets."""

        validation_bucket: int = 0
        """Bucket withheld entirely from fine-tuning."""

        max_peaks: int = 160
        """Intensity-ranked peaks retained by the reference's FPNet preprocessing."""

        intensity_floor: float = 0.001
        """Minimum intensity relative to the base peak."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    class Batch(FPNetInputs):
        """The public FPNet input order and the selected-bit target."""

        label: Tensor
        sid: Tensor
        neutral_mass: Tensor
        valid_count: Tensor

    def __init__(self, config: Config) -> None:
        if config.batch_size <= 0 or config.validation_modulus < 2:
            raise ValueError("batch_size must be positive and validation_modulus >= 2.")
        if (
            config.validation_bucket < 0
            or config.validation_bucket >= config.validation_modulus
        ):
            raise ValueError("validation_bucket must be in [0, validation_modulus).")
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        root = Path(config.working_dir)
        self.offsets = load_array(root / "spec_off.npy", np.int64)
        self.mzs = load_array(root / "spec_mz.npy", np.float32)
        self.intensities = load_array(root / "spec_it.npy", np.float32)
        metadata = parquet.read_table(root / "spec_meta.parquet")
        structures = parquet.read_table(root / "train_metric_structs.parquet")
        keys = dict(
            zip(
                from_plain(
                    cast(object, structures.column("sid").to_pylist()),
                    list[int],
                ),
                from_plain(
                    cast(object, structures.column("metric_identity").to_pylist()),
                    list[str],
                ),
                strict=True,
            ),
        )
        self.precursors = np.asarray(
            from_plain(
                cast(object, metadata.column("precursor_mz").to_pylist()),
                list[float],
            ),
            dtype=np.float64,
        )
        self.collision_energies = np.asarray(
            from_plain(
                cast(object, metadata.column("ce_mean").to_pylist()),
                list[float | None],
            ),
            dtype=np.float32,
        )
        self.adducts = np.asarray(
            from_plain(
                cast(object, metadata.column("adduct_ix").to_pylist()),
                list[int],
            ),
            dtype=np.int64,
        )
        self.merged_counts = np.asarray(
            from_plain(cast(object, metadata.column("ce_n").to_pylist()), list[int]),
            dtype=np.float32,
        )
        self.modes = np.asarray(
            from_plain(cast(object, metadata.column("mode").to_pylist()), list[int]),
            dtype=np.float32,
        )
        self.neutral_masses = np.asarray(
            from_plain(
                cast(object, metadata.column("nm").to_pylist()),
                list[float | None],
            ),
            dtype=np.float64,
        )
        self.fingerprints = torch.from_numpy(
            np.unpackbits(load_array(root / "train_fp_sel.npy", np.uint8), axis=1)[
                :,
                : len(load_array(root / "fp_bits.npy", np.int64)),
            ],
        ).float()
        self.sids = np.asarray(
            from_plain(cast(object, metadata.column("sid").to_pylist()), list[int]),
            dtype=np.int64,
        )
        valid = self.sids >= 0
        validation = np.array(
            [
                molecule_bucket(keys[sid], modulus=config.validation_modulus)
                == config.validation_bucket
                if sid >= 0
                else False
                for sid in from_plain(cast(object, self.sids.tolist()), list[int])
            ],
            dtype=np.bool_,
        )
        self.train_indices = torch.from_numpy(np.flatnonzero(valid & ~validation))
        self.eval_indices = torch.from_numpy(np.flatnonzero(valid & validation))
        if not len(self.train_indices) or not len(self.eval_indices):
            raise ValueError(
                "SpectrumData requires nonempty train and evaluation populations.",
            )
        self.order: Tensor | None = None
        self.cursor = 0

    def train_dataloader(self) -> Passes[Batch]:
        """Return repeatable passes that resume the saved in-pass cursor."""
        return Passes(draw=self._train_pass)

    def eval_dataloader(self) -> Iterator[Batch]:
        """Yield every withheld spectrum without shuffling or dropping a tail."""
        for start in range(0, len(self.eval_indices), self.config.batch_size):
            yield self.collate(
                self.eval_indices[start : start + self.config.batch_size],
            )

    def collate(self, indices: Tensor) -> Batch:
        """Apply source FPNet peak cleaning and the reference's empty-peak mask.

        Args:
          indices: CPU spectrum indices into the prepared CSR store.

        Returns:
          batch: FPNet input tensors and their selected-bit molecular targets.

        """
        ids = [int(index) for index in indices]
        peaks = [
            prepare_peaks(
                self.mzs[self.offsets[i] : self.offsets[i + 1]],
                self.intensities[self.offsets[i] : self.offsets[i + 1]],
                precursor=self.precursors.item(i),
                max_peaks=self.config.max_peaks,
                floor=self.config.intensity_floor,
            )
            for i in ids
        ]
        energy = self.collision_energies[ids]
        inputs = pack_inputs(
            [mass for mass, _ in peaks],
            [values for _, values in peaks],
            precursor=self.precursors[ids].astype(np.float32),
            adduct=self.adducts[ids],
            energy=np.nan_to_num(energy),
            energy_known=(~np.isnan(energy)).astype(np.float32),
            merged_count=self.merged_counts[ids].clip(1, 8),
            mode=self.modes[ids],
        )
        return {  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.
            **inputs,
            "label": self.fingerprints[self.sids[ids]],
            "sid": torch.from_numpy(self.sids[ids]),
            "neutral_mass": torch.from_numpy(self.neutral_masses[ids]),
            "valid_count": torch.tensor(len(ids)),
        }

    class StateDict(TypedDict):
        """One training permutation and the next unconsumed spectrum."""

        order: Tensor | None
        cursor: int
        timer_epoch: CheckpointableStepTimer.StateDict

    def state_dict(self) -> StateDict:
        """Capture the next batch position for an exact within-pass resume."""
        return {
            "order": self.order,
            "cursor": self.cursor,
            "timer_epoch": self.timer_epoch.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore a previously saved training permutation and cursor."""
        state = cast(SpectrumData.StateDict, state_dict)
        self.order, self.cursor = state["order"], state["cursor"]
        self.timer_epoch.load_state_dict(state["timer_epoch"])

    def _train_pass(self) -> Iterator[Batch]:
        if self.order is None:
            self.order = self.train_indices[torch.randperm(len(self.train_indices))]
            self.cursor = 0
        while self.cursor < len(self.order):
            indices = self.order[self.cursor : self.cursor + self.config.batch_size]
            self.cursor += len(indices)
            yield self.collate(indices)
        self.order = None
        self.cursor = 0


def molecule_bucket(key: str, *, modulus: int) -> int:
    """Assign all spectra of one molecular identity to one stable bucket."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest(), "big") % modulus


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent


_LINKED: Final = (
    "pool_meta.parquet",
    "pool_fp.npy",
    "pool_frag_off.npy",
    "pool_frag_mass.npy",
    "fp_bits.npy",
    "train_structs.parquet",
    "train_fp_sel.npy",
)
_GENERATED: Final = (
    "spec_off.npy",
    "spec_mz.npy",
    "spec_it.npy",
    "spec_meta.parquet",
    "train_metric_structs.parquet",
)


class TrainingPreparation:
    """Own complete cache outputs and verify every input and generated byte."""

    class Config(Fig["TrainingPreparation"]):
        base_dir: Path = Path("/opt/scratch/datasets/casmi2026")
        """Declared root for raw data, scientific identities and generated cache."""

        training: Path = Path(
            "inputs/enveda-CASMI26-molecule-id-mass-spectra/train.parquet",
        )
        """Complete original labelled raw spectral data."""

        pool: Path = Path(
            "inputs/datasets/ahmedberatozer/casmi26-v2-pool",
        )
        """Provided numeric pool and training structure/fingerprint data."""

        identities: IdentityTable.Config = field(default_factory=IdentityTable.Config)
        """Independent complete prepared official training identities."""

        working_dir: Path = Path("training")
        """Owned cache and receipt directory; model data is placed in work/."""

        intensity_floor: float = 0.001
        """Minimum cached intensity relative to the eligible base peak."""

        max_peaks: int = 512
        """Maximum peaks cached per spectrum."""

        precursor_margin: float = 2.0
        """Allowed cached m/z above each precursor."""

        @override
        def finalize(self) -> Self:
            self.training = resolve_working_dir(self.base_dir, self.training)
            self.pool = resolve_working_dir(self.base_dir, self.pool)
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if self.identities.base_dir is None:
                self.identities.base_dir = self.base_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> None:
        """Build a complete independent cache and receipt, preserving owned failures.

        Args:
          *args: No passthrough arguments are accepted.

        """
        if args:
            raise ValueError("TrainingPreparation accepts configuration fields only.")
        config = self.config
        inputs = self._inputs()
        output = validated_output_path(config.working_dir, protected=inputs)
        if any(path.resolve().is_relative_to(output.resolve()) for path in inputs):
            raise ValueError(
                "Preparation output must not contain protected input artifacts.",
            )
        owner = self._owner()
        if output.exists():
            try:
                previous = json_loads(
                    (output / ".preparation.json").read_text(),
                    _Owner,
                )
            except (FileNotFoundError, ReadError) as error:
                raise FileExistsError(
                    f"Unowned preparation directory: {output}",
                ) from error
            if previous != owner:
                raise FileExistsError(
                    f"Preparation belongs to different inputs: {output}",
                )
            preserved = Path(
                tempfile.mkdtemp(
                    prefix=f"{output.name}.incomplete-",
                    dir=output.parent,
                ),
            )
            output.rename(preserved / "previous")
        work = output / "work"
        work.mkdir(parents=True)
        (output / ".preparation.json").write_text(json.dumps(to_plain(owner)) + "\n")
        for name in _LINKED:
            (work / name).symlink_to((config.pool / name).resolve())
        identities = config.identities.make()
        records = from_plain(
            parquet.read_table(config.pool / "train_structs.parquet").to_pylist(),
            list[dict[str, object]],
        )
        for row in records:
            smiles = from_plain(row["smiles"], str)
            identity = identities.identity(smiles)
            if identity is None:
                raise ValueError(f"Invalid prepared training identity: {smiles}")
            row["metric_identity"] = json.dumps(
                to_plain(identity, dialect="python"),
                separators=(",", ":"),
            )
        parquet.write_table(
            Table.from_pylist(records),
            work / "train_metric_structs.parquet",
        )
        build_spectral_cache(
            config.training,
            structures=config.pool / "train_structs.parquet",
            output=work,
            intensity_floor=config.intensity_floor,
            max_peaks=config.max_peaks,
            precursor_margin=config.precursor_margin,
        )
        receipt: _Receipt = {
            "format_version": 1,
            "owner": owner,
            "sources": self._sources(),
            "inputs": {str(path.resolve()): sha256_file(path) for path in inputs},
            "outputs": {
                f"work/{name}": sha256_file(work / name)
                for name in (*_LINKED, *_GENERATED)
            },
        }
        (output / "cache_receipt.json").write_text(
            json.dumps(to_plain(receipt), indent=2) + "\n",
        )

    def cache_ready(self) -> bool:
        """Verify the complete source, input and output identities.

        Returns:
          ready: Whether every declared provenance entry and complete file matches.

        """
        output = self.config.working_dir
        try:
            receipt = json_loads((output / "cache_receipt.json").read_text(), _Receipt)
            if (
                receipt["format_version"] != 1
                or receipt["owner"] != self._owner()
                or receipt["sources"] != self._sources()
                or receipt["inputs"]
                != {str(path.resolve()): sha256_file(path) for path in self._inputs()}
                or set(receipt["outputs"])
                != {f"work/{name}" for name in (*_LINKED, *_GENERATED)}
            ):
                return False
            return all(
                sha256_file(output / name) == seal
                for name, seal in receipt["outputs"].items()
            )
        except (FileNotFoundError, ReadError):
            return False

    def _owner(self) -> _Owner:
        config = self.config
        return {
            "training": str(config.training.resolve()),
            "pool": str(config.pool.resolve()),
            "identities": str(Path(config.identities.working_dir).resolve()),
            "intensity_floor": config.intensity_floor,
            "max_peaks": config.max_peaks,
            "precursor_margin": config.precursor_margin,
        }

    def _inputs(self) -> tuple[Path, ...]:
        config = self.config
        return (
            config.training,
            *(config.pool / name for name in _LINKED),
            *(
                Path(config.identities.working_dir) / name
                for name in ("manifest.json", "molecules.parquet", "fingerprints.npy")
            ),
        )

    def _sources(self) -> dict[str, str]:
        return {
            name: sha256_file(_CWD / name)
            for name in (
                "data.py",
                "chemistry.py",
                "custom_types.py",
            )
        }


class _Owner(TypedDict):
    training: str
    pool: str
    identities: str
    intensity_floor: float
    max_peaks: int
    precursor_margin: float


class _Receipt(TypedDict):
    format_version: int
    owner: _Owner
    sources: dict[str, str]
    inputs: dict[str, str]
    outputs: dict[str, str]


def prepare_data() -> TrainingPreparation.Config:
    """Declare native cache construction from public data and prepared identities.

    Returns:
      config: Explicit staging inputs and their owned output directory.

    """
    return TrainingPreparation.Config()
