"""Glacier and Ice simulator data, backends, scheduling and scoring."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import accumulate
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    Self,
    cast,
    override,
    runtime_checkable,
)

import logging
import math
import re

from configgle import Fig, Makeable
from torch import Tensor, sparse_coo_tensor
from treekle import from_plain, json_loads

import numpy as np
import torch

from priml.baselines.casmi2026.chemistry import (
    MainSearch,
    clean_block,
    instrument_family,
)
from priml.baselines.casmi2026.custom_types import (
    AtomicRow,
    GlacierRow,
    IceRow,
    PreparationHeader,
    SimulatorAdmissionRow,
    SpectrumRequest,
)
from priml.baselines.casmi2026.isolated import (
    SIMULATOR_ELECTRON_MASS,
    SIMULATOR_RDKIT,
    SIMULATOR_SYMBOLS,
)
from priml.baselines.casmi2026.model import (
    GenerationBatch,
    Glacier,
    GlacierBatch,
    GraphGenerator,
    GraphInput,
    GraphIntensity,
    IntensityBatch,
    MolecularGraph,
    pad_atoms,
)
from priml.model.ggnn import GraphBatch
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray
    from pyarrow import parquet

    from priml.baselines.casmi2026.model import GlacierOutput, ReleasedModel
    from priml.train.budget import WallBudget
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")


class SimulatorAdmissionTable:
    """Require explicitly staged decisions, canonical aliases and rejection reasons."""

    def __init__(self, directory: Path) -> None:
        header = json_loads(
            (directory / "manifest.json").read_text(),
            PreparationHeader,
        )
        if header["format_version"] != 1 or header["rdkit"] != SIMULATOR_RDKIT:
            raise ValueError(
                f"Admission data requires format 1 and RDKit {SIMULATOR_RDKIT}.",
            )
        self.rows = tuple(
            from_plain(
                parquet.read_table(directory / "admission.parquet").to_pylist(),
                list[SimulatorAdmissionRow],
            ),
        )
        self.canonical: dict[str, tuple[str | None, int]] = {}
        self._positions: dict[str, int] = {}
        for index, row in enumerate(self.rows):
            if (
                row["glacier_accepted"] != (row["glacier_canonical"] is not None)
                or row["glacier_heavy_atoms"] < 0
                or (row["ice_accepted"] and row["ice_reason"])
                or (row["glacier_accepted"] and row["glacier_reason"])
            ):
                raise ValueError("Inconsistent simulator admission decision.")
            self._positions.setdefault(row["smiles"], index)
            self.canonical.setdefault(
                row["smiles"],
                (row["glacier_canonical"], row["glacier_heavy_atoms"]),
            )

    def row(self, index: int) -> SimulatorAdmissionRow:
        """Return the complete original nonnegative row."""
        if index < 0 or index >= len(self.rows):
            raise IndexError(index)
        return self.rows[index]

    def for_smiles(self, smiles: str) -> SimulatorAdmissionRow:
        """Return the first explicit prepared decision or raise for unstaged strings."""
        return self.rows[self._positions[smiles]]

    def ice(self, smiles: str) -> tuple[bool, str]:
        """Return the distinct ICE cheap admission and its original reason."""
        row = self.for_smiles(smiles)
        return row["ice_accepted"], row["ice_reason"]

    def glacier(self, smiles: str) -> tuple[bool, str]:
        """Return the distinct GLACIER cheap admission and its original reason."""
        row = self.for_smiles(smiles)
        return row["glacier_accepted"], row["glacier_reason"]


def load_released_checkpoint(model: ReleasedModel, path: Path) -> None:
    """Load the ``state_dict`` of a trusted-format checkpoint file into ``model``.

    Args:
      model: Released architecture whose key map and shapes validate the file.
      path: Checkpoint saved with a ``state_dict`` entry of plain tensors.

    """
    checkpoint = from_plain(
        cast(object, torch.load(path, map_location="cpu", weights_only=True)),
        dict[str, object],
    )
    model.load_released_state(from_plain(checkpoint["state_dict"], dict[str, Tensor]))


RESIDENT_GROUPS: Final = 16
"""Most recently used row groups that stay decoded."""


class RowGroups[Row]:
    """Decode the row group holding a requested row and evict the least recent."""

    def __init__(
        self,
        source: parquet.ParquetFile,
        decode: Callable[[object], list[Row]],
    ) -> None:
        self._source = source
        self._decode = decode
        self._offsets = list(
            accumulate(
                (
                    source.metadata.row_group(index).num_rows
                    for index in range(source.num_row_groups)
                ),
                initial=0,
            ),
        )
        self._groups: OrderedDict[int, list[Row]] = OrderedDict()

    @property
    def resident(self) -> int:
        """Return how many row groups are currently decoded."""
        return len(self._groups)

    def row(self, index: int) -> Row:
        """Return the decoded row at its absolute position in the file.

        Args:
          index: Zero-based row position across every row group.

        Returns:
          row: The decoded row.

        """
        group = bisect_right(self._offsets, index) - 1
        if group not in self._groups:
            self._groups[group] = self._decode(
                self._source.read_row_group(group).to_pylist(),
            )
            if len(self._groups) > RESIDENT_GROUPS:
                _ = self._groups.popitem(last=False)
        self._groups.move_to_end(group)
        return self._groups[group][index - self._offsets[group]]

    def release(self) -> None:
        """Forget every decoded row group."""
        self._groups.clear()


DENSE_NODE_LIMIT: Final = 2000
"""Largest graph whose transition matrix is powered densely."""


def landing_probabilities(
    source: Tensor,
    destination: Tensor,
    weights: Tensor,
    *,
    count: int,
    steps: int,
) -> Tensor:
    """Return weighted landing probabilities using dense or sparse powers.

    Args:
      source: Origin of every directed edge.
      destination: Target of every directed edge.
      weights: Float bond-order weight of every directed edge.
      count: Nodes in the graph, including isolated ones.
      steps: Positive number of transition-matrix powers.

    Returns:
      probabilities: Every node's return probability for each walk length.

    """
    sums = weights.new_zeros(count).index_add_(0, source, weights)
    weights = weights / (sums[source] + 1e-30)
    if count <= DENSE_NODE_LIMIT:
        transition = weights.new_zeros((count, count))
        transition[source, destination] = weights
    else:
        with torch.sparse.check_sparse_tensor_invariants():
            transition = sparse_coo_tensor(
                torch.stack((source, destination)),
                weights,
                (count, count),
                check_invariants=True,
            )
    power = transition
    result: list[Tensor] = []
    for index in range(steps):
        if index:
            power = power @ transition
        if power.is_sparse:
            coalesced = power.coalesce()
            rows, columns = coalesced.indices().unbind(0)
            diagonal = rows == columns
            values = weights.new_zeros(count)
            values[rows[diagonal]] = coalesced.values()[diagonal]
        else:
            values = power.diagonal()
        result.append(values)
    return torch.stack(result, dim=-1)


ATOM_FEATURES: Final = 74
"""Width of the prepared atom property vector."""

BOND_FEATURES: Final = 28
"""Width of the prepared bond property vector."""


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class AtomicGraph:
    """Numerical atom properties and one row per undirected chemical bond."""

    features: Tensor
    bonds: Tensor
    bond_features: Tensor
    weights: Tensor
    degrees: Tensor


class AtomicGraphTable:
    """Index canonical molecular graphs and read bounded row groups on demand."""

    def __init__(self, directory: Path) -> None:
        manifest = json_loads(
            (directory / "manifest.json").read_text(),
            dict[str, object],
        )
        if (
            manifest["rdkit"] != SIMULATOR_RDKIT
            or manifest["format_version"] != 1
            or manifest["raw"] is not False
        ):
            raise ValueError(
                f"Canonical graph data requires format 1 and RDKit {SIMULATOR_RDKIT}.",
            )
        self._source = parquet.ParquetFile(directory / "graphs.parquet")
        self.smiles = tuple(
            from_plain(
                cast(
                    object,
                    self._source.read(columns=["input_smiles"])
                    .column("input_smiles")
                    .to_pylist(),
                ),
                list[str],
            ),
        )
        self._indices = {value: index for index, value in enumerate(self.smiles)}
        self._rows = RowGroups(
            self._source,
            lambda rows: from_plain(rows, list[AtomicRow]),
        )

    def graph(self, smiles: str) -> AtomicGraph | None:
        """Return every graph field, or the explicit rejection for this input."""
        return _atomic_graph(self._rows.row(self._indices[smiles]))

    def identity(self, smiles: str) -> tuple[str | None, int]:
        """Return the prepared canonical string and its complete heavy-atom count."""
        row = self._rows.row(self._indices[smiles])
        return (row["smiles"], len(row["features"])) if row["valid"] else (None, 0)


def graphormer_input(
    graph: AtomicGraph,
    *,
    walk_steps: int,
    max_hops: int,
    spatial_limit: int,
) -> dict[str, Tensor]:
    """Create the complete Graphormer feature, distance and path tensors.

    Args:
      graph: Generated atom features and ordered undirected bonds.
      walk_steps: Number of weighted return-probability channels.
      max_hops: Number of retained bond-feature steps on each shortest path.
      spatial_limit: Distance vocabulary bound, including unreachable atoms.

    Returns:
      inputs: Complete unbatched model inputs and the scalar atom count.

    """
    count = graph.features.shape[0]
    adjacency = torch.zeros((count, count))
    edge_types = torch.zeros((count, count, graph.bond_features.shape[1]))
    left, right = graph.bonds.unbind(1)
    adjacency[left, right] = adjacency[right, left] = graph.weights
    edge_types[left, right] = edge_types[right, left] = graph.bond_features
    features = graph.features
    if walk_steps > 0:
        features = torch.cat(
            (features, weighted_walks(adjacency, steps=walk_steps)),
            1,
        )
    distances, parents = _shortest_paths(np.not_equal(adjacency.numpy(), 0))
    spatial = torch.from_numpy(np.minimum(distances, spatial_limit - 1)).long()
    edges = edge_types.numpy()
    paths = np.zeros((count, count, max_hops, edges.shape[-1]), dtype=np.int64)
    for source in range(count):
        for target in range(count):
            if source == target or parents[source, target] == -1:
                continue
            route = [target]
            while route[-1] != source:
                route.append(int(cast(np.int16, parents[source, route[-1]])))
            route.reverse()
            for position in range(min(max_hops, len(route) - 1)):
                paths[source, target, position] = edges[
                    route[position],
                    route[position + 1],
                ]
    return {
        "x": features,
        # The source clips distances before testing the vocabulary bound.
        "attn_bias": torch.zeros((count + 1, count + 1)),
        "attn_edge_type": edge_types,
        "spatial_pos": spatial,
        "degree": graph.degrees,
        "edge_input": torch.from_numpy(paths),
        "num_atoms": torch.tensor(count),
    }


def read_atomic_graphs(directory: Path) -> dict[str, AtomicGraph | None]:
    """Read complete generated graph rows without importing a chemistry library.

    Args:
      directory: Generated graph dataset containing graphs.parquet.

    Returns:
      graphs: Original input strings mapped to complete graphs or explicit rejection.

    """
    rows = from_plain(
        parquet.read_table(directory / "graphs.parquet").to_pylist(),
        list[AtomicRow],
    )
    return {row["input_smiles"]: _atomic_graph(row) for row in rows}


def weighted_walks(adjacency: Tensor, *, steps: int) -> Tensor:
    """Return weighted random-walk landing probabilities at every step.

    Args:
      adjacency: Dense float32 bond-order matrix, including isolated atoms.
      steps: Positive number of transition-matrix powers.

    Returns:
      probabilities: One diagonal return-probability vector per walk length.

    """
    rows, columns = adjacency.nonzero(as_tuple=True)
    return landing_probabilities(
        rows,
        columns,
        adjacency[rows, columns],
        count=adjacency.shape[0],
        steps=steps,
    )


type SimulationJob = tuple[str, str, str, float]
type PreparedPeaks = tuple[NDArray[np.float64], NDArray[np.float64]]
type PredictedPeaks = tuple[NDArray[np.float32], NDArray[np.float32]]

ADDUCTS: Final = ("[M+H]+", "[M+Na]+")
"""Covered positive adducts in model id order."""

INSTRUMENTS: Final = ("Orbitrap", "QTOF")
"""Covered instrument classes in model id order."""

HYDROGEN_ROW: Final = SIMULATOR_SYMBOLS.index("H")
"""Row of hydrogen in the simulator element tables."""

ADDUCT_ROWS: Final = (HYDROGEN_ROW, SIMULATOR_SYMBOLS.index("Na"))
"""Element-table row of the atom each covered adduct adds."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationSpectrum:
    """Carry prepared query peaks and their declared simulator condition."""

    query: PreparedPeaks
    adduct: str
    instrument: str
    energy: float


def collision_energy(spectrum: SpectrumRequest) -> float:
    """Resolve NCE text, eV arrays or the explicit scalar with field-presence rules.

    Args:
      spectrum: Original query fields, retaining absent versus present-null values.

    Returns:
      energy: Resolved value, including NaN for absent or unknown collision energy.

    """
    if "ce_units" in spectrum or "ce_ev" in spectrum:
        if spectrum.get("ce_units") == "NCE":
            tokens = from_plain(
                cast(
                    object,
                    re.findall(r"[0-9]+\.?[0-9]*", str(spectrum.get("ce_orig"))),
                ),
                list[str],
            )
            return (
                float(np.mean([float(token) for token in tokens]))
                if tokens
                else math.nan
            )
        values = spectrum.get("ce_ev")
        return (
            float(np.mean(values)) if values is not None and len(values) else math.nan
        )
    value = spectrum.get("ce")
    return float(value) if value is not None and math.isfinite(value) else math.nan


def energy_bucket(energy: float) -> float:
    """Round finite collision energy to five-unit buckets with a minimum of five."""
    return (
        max(5.0, float(cast(np.float64, np.round(energy / 5.0))) * 5.0)
        if math.isfinite(energy)
        else math.nan
    )


def covered(spectrum: SpectrumRequest) -> bool:
    """Admit positive protonated and sodiated spectra to the forward panels."""
    mode = spectrum["mode"]
    positive = mode == 1 or (
        isinstance(mode, str) and mode.lower() in ("positive", "pos", "+", "1")
    )
    return positive and spectrum["adduct"] in ADDUCTS


def instrument_token(spectrum: SpectrumRequest) -> str:
    """Use an explicit supported token, then the original instrument family."""
    explicit = spectrum.get("instr")
    if explicit in ("Orbitrap", "QTOF"):
        return explicit
    return "Orbitrap" if instrument_family(spectrum.get("instrument")) == 1 else "QTOF"


class QueryScoring:
    """Prepare covered queries and score predictions against them."""

    class Config(Fig["QueryScoring"]):
        search: MainSearch.Config = field(default_factory=MainSearch.Config)
        """Entropy similarity preparing queries and predictions."""

        floor: float = 0.001
        """Minimum query intensity relative to the eligible base peak."""

        max_peaks: int = 512
        """Maximum retained peaks of each cleaned query."""

        over: float = 2.0
        """Allowed query m/z above its precursor."""

        passes: int = 2
        """Successive applications of the released query cleaning."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.search = config.search.make()

    def plan(self, spectra: Sequence[SpectrumRequest]) -> list[SimulationSpectrum]:
        """Prepare every covered query and retain its complete simulator condition.

        Args:
          spectra: Complete original peak and condition fields in molecule order.

        Returns:
          plan: Every covered prepared query, including queries with no retained peaks.

        """
        config = self.config
        result: list[SimulationSpectrum] = []
        for spectrum in spectra:
            if not covered(spectrum):
                continue
            mz, intensity = (
                np.array(spectrum["mz"], dtype=np.float64),
                np.array(spectrum["it"], dtype=np.float64),
            )
            for _ in range(config.passes):
                _, mz, intensity = clean_block(
                    np.array([0, len(mz)], dtype=np.int64),
                    mz=mz.astype(np.float64),
                    intensity=intensity.astype(np.float64),
                    precursor=np.array([spectrum["prec"]], dtype=np.float64),
                    floor=config.floor,
                    topk=config.max_peaks,
                    over=config.over,
                )
            result.append(
                SimulationSpectrum(
                    query=self.search.prepare(
                        (mz.astype(np.float64), intensity.astype(np.float64)),
                    ),
                    adduct=spectrum["adduct"],
                    instrument=instrument_token(spectrum),
                    energy=energy_bucket(collision_energy(spectrum)),
                ),
            )
        return result

    def score(
        self,
        plan: Sequence[SimulationSpectrum],
        candidates: Sequence[str],
        cache: dict[SimulationJob, PreparedPeaks | None],
    ) -> dict[str, float | None]:
        """Average covered finite entropy similarities without partial summed-energy jobs.

        Args:
          plan: Every covered prepared query and its original condition.
          candidates: Complete candidate list in retrieval order.
          cache: Completed job predictions, explicit failures and absent unfinished jobs.

        Returns:
          scores: Mean finite similarities, or None if no complete prediction is available.

        """
        search = self.search
        merged: dict[tuple[str, str, str], PreparedPeaks | None] = {}
        result: dict[str, float | None] = {}
        for candidate in candidates:
            similarities: list[float] = []
            for spectrum in plan:
                key = (candidate, spectrum.adduct, spectrum.instrument)
                if math.isfinite(spectrum.energy):
                    prediction = cache.get((*key, spectrum.energy))
                else:
                    if key not in merged:
                        parts = [
                            cache.get((*key, energy)) for energy in (20.0, 40.0, 60.0)
                        ]
                        present = [part for part in parts if part is not None]
                        merged[key] = (
                            merge_predictions(present) if len(present) == 3 else None
                        )
                    prediction = merged[key]
                if prediction is None:
                    continue
                value = search.similarity(spectrum.query, search.prepare(prediction))
                if value < 1e-9:
                    value = 0.0
                if math.isfinite(value):
                    similarities.append(value)
            result[candidate] = float(np.mean(similarities)) if similarities else None
        return result


def condition_ids(jobs: Sequence[SimulationJob]) -> tuple[list[int], list[int]]:
    """Return each job's adduct and instrument model ids.

    Args:
      jobs: Requests whose adducts and instruments are covered.

    Returns:
      adducts: Index of each job's adduct in ``ADDUCTS``.
      instruments: Index of each job's instrument in ``INSTRUMENTS``.

    Raises:
      ValueError: A job names an adduct or instrument the models do not cover.

    """
    return (
        [ADDUCTS.index(job[1]) for job in jobs],
        [INSTRUMENTS.index(job[2]) for job in jobs],
    )


def required_jobs(
    plan: Sequence[SimulationSpectrum],
    candidates: Sequence[str],
) -> set[SimulationJob]:
    """Expand missing energy to all three required predictions and deduplicate jobs.

    Args:
      plan: Complete prepared molecule query conditions.
      candidates: Candidate strings in retrieval order, including duplicates.

    Returns:
      jobs: Every distinct requested molecule/adduct/instrument/energy tuple.

    """
    return {
        (candidate, spectrum.adduct, spectrum.instrument, energy)
        for spectrum in plan
        for candidate in candidates
        for energy in (
            (spectrum.energy,) if math.isfinite(spectrum.energy) else (20.0, 40.0, 60.0)
        )
    }


class PeakReduction(Protocol):
    """Reduce one family's raw model peaks to unique rounded masses."""

    def keep(self, spectrum: NDArray[np.float32]) -> NDArray[np.bool_]:
        """Select the raw rows that carry a peak."""
        ...

    def merge(
        self,
        inverse: NDArray[np.intp],
        weights: NDArray[np.float64],
        size: int,
    ) -> NDArray[np.float64]:
        """Combine the intensities sharing one rounded mass."""
        ...

    def top(self, intensity: NDArray[np.float64], count: int) -> NDArray[np.intp]:
        """Pick the indices of the most intense peaks."""
        ...


class SumReduction:
    """Sum duplicate masses; take the top peaks, earliest mass first among equals."""

    class Config(Fig["SumReduction"]):
        """ICE reduction; it has no tunables."""

    def __init__(self, config: Config) -> None:
        del config

    def keep(self, spectrum: NDArray[np.float32]) -> NDArray[np.bool_]:
        """Keep rows of positive intensity."""
        return spectrum[:, 1] > 0

    def merge(
        self,
        inverse: NDArray[np.intp],
        weights: NDArray[np.float64],
        size: int,
    ) -> NDArray[np.float64]:
        """Sum the intensities of equal rounded masses."""
        del size
        return np.bincount(inverse, weights=weights)

    def top(self, intensity: NDArray[np.float64], count: int) -> NDArray[np.intp]:
        """Take the strongest peaks, earliest mass first among equals."""
        return np.argsort(-intensity, kind="stable")[:count]


class MaxReduction:
    """Take duplicate maxima, exclude nonpositive mass and break ties stably."""

    class Config(Fig["MaxReduction"]):
        """GLACIER reduction; it has no tunables."""

    def __init__(self, config: Config) -> None:
        del config

    def keep(self, spectrum: NDArray[np.float32]) -> NDArray[np.bool_]:
        """Keep rows of positive intensity and positive mass."""
        return (spectrum[:, 1] > 0) & (spectrum[:, 0] > 0)

    def merge(
        self,
        inverse: NDArray[np.intp],
        weights: NDArray[np.float64],
        size: int,
    ) -> NDArray[np.float64]:
        """Take the maximum intensity of equal rounded masses."""
        intensity = np.zeros(size, dtype=np.float64)
        np.maximum.at(intensity, inverse, weights)
        return intensity

    def top(self, intensity: NDArray[np.float64], count: int) -> NDArray[np.intp]:
        """Take the strongest peaks, earliest mass first among equals."""
        return np.argsort(-intensity, kind="stable")[:count]


def post_prediction(
    spectrum: NDArray[np.float32],
    *,
    reduction: PeakReduction,
    max_peaks: int,
) -> PredictedPeaks:
    """Merge rounded mass duplicates and retain the family-specific top peaks.

    Args:
      spectrum: Complete float32 model mass/intensity output.
      reduction: The family's row selection, duplicate merge and peak ordering.
      max_peaks: Complete intensity-selected prediction cap.

    Returns:
      masses: Sorted float32 masses after duplicate reduction and peak selection.
      intensity: Corresponding float32 intensities normalized by their maximum.

    """
    values = spectrum[reduction.keep(spectrum)]
    if not len(values):
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    masses, inverse = np.unique(
        np.round(values[:, 0].astype(np.float64), 4),
        return_inverse=True,
    )
    intensity = reduction.merge(inverse, values[:, 1].astype(np.float64), len(masses))
    if len(masses) > max_peaks:
        selected = reduction.top(intensity, max_peaks)
        masses, intensity = masses[selected], intensity[selected]
    order = np.argsort(masses)
    masses, intensity = masses[order], intensity[order]
    return masses.astype(np.float32), (intensity / intensity.max()).astype(np.float32)


def merge_predictions(spectra: Sequence[PreparedPeaks]) -> PreparedPeaks:
    """Merge every complete collision-energy prediction by rounded mass and sum.

    Args:
      spectra: Complete cached double-precision predictions for all required energies.

    Returns:
      masses: Every rounded unique mass in sorted order.
      intensity: Summed normalized intensities, scaled to maximum one.

    """
    mz = np.concatenate([mass for mass, _ in spectra])
    intensity = np.concatenate([values / values.max() for _, values in spectra])
    masses, inverse = np.unique(np.round(mz, 4), return_inverse=True)
    sums = np.bincount(inverse, weights=intensity)
    return masses, sums / sums.max()


class GlacierGraphTable:
    """Own bounded canonical graph data and its complete physical element arrays."""

    def __init__(
        self,
        directory: Path,
        *,
        walk_steps: int,
        max_hops: int,
        spatial_limit: int,
        root_cache: int = 512,
    ) -> None:
        manifest = json_loads(
            (directory / "manifest.json").read_text(),
            dict[str, object],
        )
        if manifest["rdkit"] != SIMULATOR_RDKIT or manifest["format_version"] != 1:
            raise ValueError(
                f"GLACIER data requires format 1 and RDKit {SIMULATOR_RDKIT}.",
            )
        if walk_steps < 0 or max_hops <= 0:
            raise ValueError("Walk steps must be nonnegative and max_hops positive.")
        self._walk_steps, self._max_hops = walk_steps, max_hops
        self._spatial_limit = spatial_limit
        self._root_cache = root_cache
        self._graphs = AtomicGraphTable(directory / "graphs")
        self.element_masses = cast(
            "NDArray[np.float64]",
            np.load(
                directory / "element_masses.npy",
                mmap_mode="c",
                allow_pickle=False,
            ),
        )
        if self.element_masses.dtype != np.float64 or self.element_masses.shape != (
            18,
            1,
        ):
            raise ValueError("GLACIER element masses require a float64 18-by-1 matrix.")
        self._source = parquet.ParquetFile(directory / "physical.parquet")
        identifiers = from_plain(
            self._source.read(columns=["input_smiles", "smiles", "valid"]).to_pylist(),
            list[dict[str, str | bool]],
        )
        self.smiles = tuple(from_plain(row["input_smiles"], str) for row in identifiers)
        if self.smiles != self._graphs.smiles:
            raise ValueError(
                "GLACIER physical and atomic rows must have the same order.",
            )
        self._indices = {value: index for index, value in enumerate(self.smiles)}
        for index, row in enumerate(identifiers):
            if row["valid"]:
                _ = self._indices.setdefault(from_plain(row["smiles"], str), index)
        self._rows = RowGroups(
            self._source,
            lambda rows: from_plain(rows, list[GlacierRow]),
        )
        self._prepared: OrderedDict[int, dict[str, Tensor]] = OrderedDict()

    def identity(self, smiles: str) -> tuple[str | None, int]:
        """Return prepared canonical identity and heavy-atom count or explicit rejection."""
        return self._graphs.identity(self.smiles[self._indices[smiles]])

    def batch(
        self,
        jobs: Sequence[SimulationJob],
        *,
        device: torch.device,
    ) -> GlacierBatch:
        """Pack every ordered numerical root and physical condition for inference.

        Args:
          jobs: Complete ordered canonical roots and physical conditions.
          device: Target model device; preparation remains on CPU.

        Returns:
          batch: Complete padded graph, physical and condition tensors.

        Raises:
          KeyError: A root was not prepared.
          ValueError: A prepared root was rejected during graph admission.

        """
        roots = [self._root(smiles) for smiles, _, _, _ in jobs]
        counts = torch.tensor([len(root["x"]) for root in roots])
        max_nodes = max(len(root["x"]) for root in roots)
        fields = {
            "features": torch.zeros(len(roots), max_nodes, roots[0]["x"].shape[-1]),
            "bias": torch.full((len(roots), max_nodes + 1, max_nodes + 1), -99_999.0),
            "edges": torch.zeros(len(roots), max_nodes, max_nodes, BOND_FEATURES),
            "spatial": torch.zeros(len(roots), max_nodes, max_nodes, dtype=torch.int64),
            "degrees": torch.zeros(len(roots), max_nodes, dtype=torch.int64),
            "paths": torch.zeros(
                len(roots),
                max_nodes,
                max_nodes,
                self._max_hops,
                BOND_FEATURES,
            ),
        }
        adjacency = torch.zeros(len(roots), max_nodes, max_nodes)
        masses = torch.zeros(len(roots), max_nodes)
        hydrogens = torch.zeros(len(roots), max_nodes, dtype=torch.int64)
        for index, root in enumerate(roots):
            count = len(root["x"])
            fields["features"][index, :count] = root["x"]
            fields["bias"][index, : count + 1, : count + 1] = root["attn_bias"]
            fields["edges"][index, :count, :count] = root["attn_edge_type"]
            fields["spatial"][index, :count, :count] = root["spatial_pos"]
            fields["degrees"][index, :count] = root["degree"]
            fields["paths"][index, :count, :count] = root["edge_input"]
            adjacency[index, :count, :count] = root["adjacency"]
            masses[index, :count] = root["masses"]
            hydrogens[index, :count] = root["hydrogens"]
        adducts, instruments = condition_ids(jobs)
        shifts = [
            [
                float(cast(np.float64, self.element_masses[ADDUCT_ROWS[adduct], 0]))
                - SIMULATOR_ELECTRON_MASS,
                -SIMULATOR_ELECTRON_MASS,
            ]
            for adduct in adducts
        ]
        return GlacierBatch(
            graph=GraphInput(
                **{name: value.to(device) for name, value in fields.items()},
            ),
            atom_counts=counts.to(device),
            adduct=torch.tensor(adducts, dtype=torch.float32, device=device),
            collision_energy=torch.tensor([job[3] for job in jobs], device=device),
            instrument=torch.tensor(instruments, dtype=torch.float32, device=device),
            root_forms=torch.stack([root["formula"] for root in roots]).to(device),
            atom_forms=torch.cat([root["atom_formulas"] for root in roots]).to(device),
            masses=masses.to(device),
            adduct_shifts=torch.tensor(shifts, dtype=torch.float32, device=device),
            adjacency=adjacency.to(device),
            atom_hs=hydrogens.to(device),
            total_hs=torch.stack([root["total_hydrogens"] for root in roots]).to(
                device,
            ),
        )

    def release_cache(self) -> None:
        """Release prepared numerical roots and loaded physical row groups."""
        self._prepared.clear()
        self._rows.release()

    def _root(self, smiles: str) -> dict[str, Tensor]:
        index = self._indices[smiles]
        if index not in self._prepared:
            original = self.smiles[index]
            graph = self._graphs.graph(original)
            if graph is None:
                raise ValueError(f"GLACIER rejected prepared root: {smiles}")
            root = graphormer_input(
                graph,
                walk_steps=self._walk_steps,
                max_hops=self._max_hops,
                spatial_limit=self._spatial_limit,
            )
            adjacency = torch.zeros(len(graph.features), len(graph.features))
            left, right = graph.bonds.unbind(-1)
            adjacency[left, right] = adjacency[right, left] = graph.weights
            row = self._rows.row(index)
            root.update(
                adjacency=adjacency,
                formula=torch.tensor(row["formula"], dtype=torch.float64),
                atom_formulas=torch.tensor(row["atom_formulas"], dtype=torch.float64),
                masses=torch.tensor(row["hydrogenated_masses"], dtype=torch.float32),
                hydrogens=torch.tensor(row["hydrogens"], dtype=torch.int64),
                total_hydrogens=torch.tensor(row["total_hydrogens"]),
            )
            self._prepared[index] = root
            if len(self._prepared) > self._root_cache:
                _ = self._prepared.popitem(last=False)
        self._prepared.move_to_end(index)
        return self._prepared[index]


class GlacierBackend:
    """Own the complete released GLACIER model and independent prepared data."""

    class Config(Fig["GlacierBackend"]):
        base_dir: Path | str | None = None
        """Dataset root supplied by the owning stage; paths resolve beneath it."""

        root_dir: Path | str = "native-chemistry/glacier-roots"
        """Prepared atomic graphs and complete physical records."""

        model: Makeable[Glacier] = field(default_factory=Glacier.Config)
        """Native model architecture matching the complete released state."""

        checkpoint: Path | str = (
            "inputs/datasets/ahmedberatozer/casmi26-glacier/ckpt/glacier.pt"
        )
        """Complete released GLACIER checkpoint."""

        device: str = "cpu"
        """Model and packed inference device."""

        walk_steps: int = 14
        """Prepared root return-probability channels."""

        reduction: Makeable[PeakReduction] = field(default_factory=MaxReduction.Config)
        """Duplicate-mass merge and peak ordering of the public prediction."""

        max_peaks: int = 100
        """Intensity-selected peaks kept per prediction."""

        @override
        def finalize(self) -> Self:
            self.root_dir = resolve_working_dir(self.base_dir, self.root_dir)
            self.checkpoint = resolve_working_dir(self.base_dir, self.checkpoint)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = config.model.make()
        self.reduction = config.reduction.make()
        root = self.model.config.root
        self.roots = GlacierGraphTable(
            Path(config.root_dir),
            walk_steps=config.walk_steps,
            max_hops=root.max_hops,
            spatial_limit=root.spatial.channels_in - 1,
        )
        load_released_checkpoint(self.model, Path(config.checkpoint))
        _ = self.model.to(self.device).eval().requires_grad_(False)
        self.feature_failures = 0

    def predict_fragments(self, jobs: list[SimulationJob]) -> dict[str, list[Tensor]]:
        """Return complete sparse spectra and aligned fragment membership masks.

        Args:
          jobs: Complete ordered prepared positive-ion prediction requests.

        Returns:
          prediction: Every sparse spectrum row and its fragment membership mask.

        """
        batch = self.roots.batch(jobs, device=self.device)
        with torch.inference_mode():
            output = self.model(batch)
            return _assemble(output, shifts=2 * (2 * self.model.config.max_broken + 1))

    def predict(self, jobs: list[SimulationJob]) -> list[PredictedPeaks | None]:
        """Return complete reduced predictions in request order.

        Args:
          jobs: Ordered declared requests, including explicitly rejected roots.

        Returns:
          spectra: A complete reduced prediction or explicit feature failure per job.

        Raises:
          KeyError: A required root was not generated during data preparation.

        """
        selected: list[SimulationJob] = []
        positions: list[int] = []
        result: list[PredictedPeaks | None] = [None] * len(jobs)
        for index, job in enumerate(jobs):
            if self.roots.identity(job[0])[0] is None:
                self.feature_failures += 1
            else:
                selected.append(job)
                positions.append(index)
        if selected:
            raw = self.predict_fragments(selected)["spec"]
            for index, spectrum in zip(positions, raw, strict=True):
                result[index] = post_prediction(
                    spectrum.float().cpu().numpy(),
                    reduction=self.reduction,
                    max_peaks=self.config.max_peaks,
                )
        return result

    def release_cache(self) -> None:
        """Release prepared roots and device allocator cache after a failed batch."""
        if self.device.type == "cpu":
            return
        self.roots.release_cache()
        torch.cuda.empty_cache()


@dataclass(frozen=True, slots=True, kw_only=True)
class FragmentGraph(MolecularGraph):
    """Keep root atom identities through fragment slicing and component grouping."""

    node_ids: Tensor


def take_graphs(
    graph: FragmentGraph,
    indices: Tensor,
) -> tuple[FragmentGraph, Tensor, Tensor]:
    """Replicate complete graph segments in the requested order.

    Args:
      graph: Ordered disjoint fragments with root atom identities.
      indices: Requested graph segments, including repeated indices.

    Returns:
      selected: Complete graph, segment sizes and node offsets.

    """
    topology = graph.topology
    counts = topology.node_counts
    device = counts.device
    segments = torch.repeat_interleave(counts)
    edge_counts = torch.bincount(segments[topology.source], minlength=len(counts))
    old_start = counts.cumsum(0) - counts
    old_edge_start = edge_counts.cumsum(0) - edge_counts
    sizes = counts[indices]
    offsets = sizes.cumsum(0) - sizes
    edge_sizes = edge_counts[indices]
    edge_offsets = edge_sizes.cumsum(0) - edge_sizes
    nodes = (
        torch.arange(int(sizes.sum()), device=device)
        - offsets.repeat_interleave(sizes)
        + old_start[indices].repeat_interleave(sizes)
    )
    edges = (
        torch.arange(int(edge_sizes.sum()), device=device)
        - edge_offsets.repeat_interleave(edge_sizes)
        + old_edge_start[indices].repeat_interleave(edge_sizes)
    )
    jobs = torch.arange(len(indices), device=device).repeat_interleave(edge_sizes)
    shift = offsets[jobs] - old_start[indices[jobs]]
    return (
        FragmentGraph(
            topology=GraphBatch(
                source=topology.source[edges] + shift,
                destination=topology.destination[edges] + shift,
                edge_type=topology.edge_type[edges],
                node_counts=sizes,
            ),
            features=graph.features[nodes],
            node_ids=graph.node_ids[nodes],
        ),
        sizes,
        offsets,
    )


def mask_nodes(graph: FragmentGraph, mask: Tensor, counts: Tensor) -> FragmentGraph:
    """Induce ordered edges and preserve the declared graph segment sizes.

    Args:
      graph: Complete graph before removing nodes.
      mask: One retention flag per packed node.
      counts: Number of retained nodes in each output graph.

    Returns:
      retained: Induced topology, features and unchanged root atom identities.

    """
    if int(mask.sum()) != int(counts.sum()):
        raise ValueError("The retained nodes do not match their segment counts.")
    ids = mask.nonzero(as_tuple=True)[0]
    positions = mask.cumsum(0) - 1
    topology = graph.topology
    retained = mask[topology.source] & mask[topology.destination]
    return FragmentGraph(
        topology=GraphBatch(
            source=positions[topology.source[retained]],
            destination=positions[topology.destination[retained]],
            edge_type=topology.edge_type[retained],
            node_counts=counts,
        ),
        features=graph.features[ids],
        node_ids=graph.node_ids[ids],
    )


def remove_atoms(
    graph: FragmentGraph,
    graph_index: Tensor,
    atom_index: Tensor,
    information: dict[str, Tensor],
) -> tuple[FragmentGraph | None, Tensor | None, dict[str, Tensor] | None]:
    """Remove requested atoms and keep touched connected non-singleton fragments.

    Args:
      graph: Packed candidate fragments before atom removal.
      graph_index: Source graph segment for each independent removal request.
      atom_index: Local atom position removed in each requested graph.
      information: Per-request values mapped to each resulting component.

    Returns:
      fragments: Grouped graph, cut-bond sums and mapped values, or three Nones.

    """
    jobs, sizes, offsets = take_graphs(graph, graph_index)
    device = jobs.features.device
    removed = torch.zeros(len(jobs.features), dtype=torch.bool, device=device)
    removed[offsets + atom_index] = True
    source, destination = jobs.topology.source, jobs.topology.destination
    cuts = removed[source] ^ removed[destination]
    kept_ends = torch.where(removed[source[cuts]], destination[cuts], source[cuts])
    orders = jobs.topology.edge_type[cuts]
    retained_ids = (~removed).nonzero(as_tuple=True)[0]
    old_to_new = torch.full_like(removed, -1, dtype=torch.int64)
    old_to_new[retained_ids] = torch.arange(len(retained_ids), device=device)
    subgraph = mask_nodes(jobs, ~removed, sizes - 1)
    components = _components(subgraph.topology, len(subgraph.features))
    cut_components = components[old_to_new[kept_ends]]
    touched = (components > 0) & torch.isin(components, cut_components)
    labels = components[touched] - 1
    if not len(labels):
        return None, None, None
    values, cut_components = torch.unique(cut_components, return_inverse=True)
    if int(values.min()) > 0:
        cut_components += 1
    labels = torch.unique(labels, return_inverse=True)[1]
    group_counts = torch.bincount(labels)
    # This temporary one-segment graph is regrouped before the next graph operation.
    subgraph = mask_nodes(subgraph, touched, touched.sum().reshape(1))
    permutation = torch.argsort(labels, stable=True)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation), device=device)
    source = inverse[subgraph.topology.source]
    destination = inverse[subgraph.topology.destination]
    edge_permutation = torch.argsort(source, stable=True)
    grouped = FragmentGraph(
        topology=GraphBatch(
            source=source[edge_permutation],
            destination=destination[edge_permutation],
            edge_type=subgraph.topology.edge_type[edge_permutation],
            node_counts=group_counts,
        ),
        features=subgraph.features[permutation],
        node_ids=subgraph.node_ids[permutation],
    )

    broken = torch.floor(
        _aggregate(orders, cut_components, len(group_counts), "sum").float() / 2,
    )
    request = torch.arange(len(graph_index), device=device).repeat_interleave(sizes)
    mapped = {
        name: _aggregate(
            value[request[kept_ends]],
            cut_components,
            len(group_counts),
            "amax",
        )
        for name, value in information.items()
    }
    return grouped, broken, mapped


def spectrum_hash(graph: FragmentGraph) -> Tensor:
    """Sum one ordered weighted message pass into one fingerprint per fragment.

    Args:
      graph: Complete prepared atom channels and directed bond-order edges.

    Returns:
      hashes: One feature vector for each graph segment.

    """
    topology = graph.topology
    messages = graph.features[topology.source] * topology.edge_type.float()[:, None]
    updated = graph.features.new_zeros(graph.features.shape).index_add_(
        0,
        topology.destination,
        messages,
    )
    return _sum_nodes(graph, updated)


def first_unique(values: Tensor, *, dim: int = 0) -> tuple[Tensor, Tensor]:
    """Return sorted distinct values and their first encounter positions.

    Args:
      values: Packed feature rows or columns whose duplicates are equivalent.
      dim: Axis containing the compared rows or columns.

    Returns:
      unique: Distinct values and corresponding first input indices.

    """
    unique, inverse = torch.unique(values, return_inverse=True, dim=dim)
    positions = torch.arange(
        inverse.size(dim),
        dtype=inverse.dtype,
        device=values.device,
    )
    result = inverse.new_full((unique.size(dim),), int(positions.max()) + 1)
    _ = result.scatter_reduce_(
        dim,
        inverse.flip([dim]),
        positions.flip([dim]),
        "amin",
    )
    return unique, result


def formula_counts(graph: FragmentGraph) -> Tensor:
    """Sum the complete elemental and attached-hydrogen feature channels.

    Args:
      graph: Prepared 18-element, eight-group and six-H-channel atom features.

    Returns:
      formulas: Complete 18-element counts for every graph segment.

    """
    forms = _sum_nodes(graph, graph.features[:, :18])
    hydrogens = (
        graph.features[:, 26:32] * torch.arange(6, device=graph.features.device)
    ).sum(1, keepdim=True)
    forms[:, HYDROGEN_ROW] = _sum_nodes(graph, hydrogens).squeeze(1)
    return forms


def fragment_masses(forms: Tensor, element_masses: Tensor) -> Tensor:
    """Apply the prepared monoisotopic element vector in the formula dtype."""
    masses = element_masses.to(device=forms.device, dtype=forms.dtype)
    return (forms[:, : masses.shape[0]] @ masses).squeeze(-1)


def walk_features(graph: FragmentGraph, *, steps: int) -> Tensor:
    """Return weighted landing probabilities of the directed bond-order edges.

    Args:
      graph: Directed bond-order edges, including isolated atoms.
      steps: Positive number of transition-matrix powers.

    Returns:
      probabilities: Every node's return probability for each walk length.

    """
    topology = graph.topology
    return landing_probabilities(
        topology.source,
        topology.destination,
        topology.edge_type.float(),
        count=len(graph.features),
        steps=steps,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class IceRoot:
    """Carry the complete root graph, canonical formula and isotope-aware masses."""

    smiles: str
    graph: FragmentGraph
    formula: Tensor
    hydrogens: Tensor
    atom_masses: Tensor
    hydrogenated_masses: Tensor
    mass: Tensor


class IceGraphTable:
    """Index prepared ICE roots and read graph row groups on demand."""

    def __init__(self, directory: Path) -> None:
        manifest = json_loads(
            (directory / "manifest.json").read_text(),
            dict[str, object],
        )
        if manifest["rdkit"] != SIMULATOR_RDKIT or manifest["format_version"] != 1:
            raise ValueError(
                f"ICE root data requires format 1 and RDKit {SIMULATOR_RDKIT}.",
            )
        self.element_masses = cast(
            "NDArray[np.float64]",
            np.load(
                directory / "element_masses.npy",
                mmap_mode="c",
                allow_pickle=False,
            ),
        )
        if self.element_masses.dtype != np.float64 or self.element_masses.shape != (
            18,
            1,
        ):
            raise ValueError("ICE element masses require a float64 18-by-1 matrix.")
        self._source = parquet.ParquetFile(directory / "ice.parquet")
        self.smiles = tuple(
            from_plain(
                cast(
                    object,
                    self._source.read(columns=["input_smiles"])
                    .column("input_smiles")
                    .to_pylist(),
                ),
                list[str],
            ),
        )
        self._indices = {value: index for index, value in enumerate(self.smiles)}
        self._rows = RowGroups(
            self._source,
            lambda rows: from_plain(rows, list[IceRow]),
        )

    def root(self, smiles: str) -> IceRoot | None:
        """Return a complete prepared root or its explicit invalid marker.

        Args:
          smiles: Exact original string declared during preparation.

        Returns:
          root: Complete canonical numerical data, or None for rejected input.

        Raises:
          KeyError: The requested input was not prepared.

        """
        index = self._indices[smiles]
        row = self._rows.row(index)
        if not row["valid"]:
            return None
        features = torch.tensor(row["features"], dtype=torch.float32).reshape(-1, 32)
        left, right = (
            torch.tensor(row["bonds"], dtype=torch.int64).reshape(-1, 2).unbind(-1)
        )
        orders = torch.tensor(row["bond_orders"], dtype=torch.int64)
        return IceRoot(
            smiles=row["smiles"],
            graph=FragmentGraph(
                topology=GraphBatch(
                    source=torch.cat([left, right]),
                    destination=torch.cat([right, left]),
                    edge_type=torch.cat([orders, orders]),
                    node_counts=torch.tensor([len(features)]),
                ),
                features=features,
                node_ids=torch.arange(len(features)),
            ),
            formula=torch.tensor(row["formula"], dtype=torch.float32),
            hydrogens=torch.tensor(row["hydrogens"], dtype=torch.int64),
            atom_masses=torch.tensor(row["atom_masses"], dtype=torch.float64),
            hydrogenated_masses=torch.tensor(
                row["hydrogenated_masses"],
                dtype=torch.float64,
            ),
            mass=torch.tensor(row["mass"], dtype=torch.float64),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class IceBatch:
    """Carry prepared root graphs and physical conditions without chemistry imports."""

    roots: FragmentGraph
    root_forms: Tensor
    collision_energy: Tensor
    precursor: Tensor
    adduct: Tensor
    instrument: Tensor
    adduct_mass: Tensor
    electron_shift: Tensor
    element_masses: Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class IcePrediction:
    """Retain every generated fragment and its complete attributed spectrum."""

    generation: dict[str, Tensor]
    spectra: list[Tensor]
    fragments: list[Tensor]


def pack_ice_batch(
    roots: Sequence[FragmentGraph],
    *,
    root_forms: Tensor,
    jobs: Sequence[SimulationJob],
    precursor: Sequence[float],
    element_masses: Tensor,
    device: torch.device,
) -> IceBatch:
    """Pack prepared roots and the two covered positive-ion physical conditions.

    Args:
      roots: Individual canonical root graphs in job order.
      root_forms: Complete float32 elemental-count rows in the same order.
      jobs: SMILES, adduct, instrument and collision energy per prediction.
      precursor: Explicit precursor values; the released runner supplies zeros.
      element_masses: Prepared float64 18-by-1 monoisotopic mass table.
      device: Destination of every numerical model input.

    Returns:
      batch: Ordered disjoint roots and all physical context tensors.

    """
    counts = torch.cat([root.topology.node_counts for root in roots])
    offsets = counts.cumsum(0) - counts
    topology = GraphBatch(
        source=torch.cat(
            [root.topology.source + offsets[index] for index, root in enumerate(roots)],
        ).to(device),
        destination=torch.cat(
            [
                root.topology.destination + offsets[index]
                for index, root in enumerate(roots)
            ],
        ).to(device),
        edge_type=torch.cat([root.topology.edge_type for root in roots]).to(device),
        node_counts=counts.to(device),
    )
    adducts, instruments = condition_ids(jobs)
    masses = element_masses.to(device)
    adduct_mass = masses[[ADDUCT_ROWS[adduct] for adduct in adducts], 0]
    return IceBatch(
        roots=FragmentGraph(
            topology=topology,
            features=torch.cat([root.features for root in roots]).to(device),
            node_ids=torch.cat([root.node_ids for root in roots]).to(device),
        ),
        root_forms=root_forms.to(device),
        collision_energy=torch.tensor(
            [job[3] for job in jobs],
            dtype=torch.float32,
            device=device,
        ),
        precursor=torch.tensor(precursor, dtype=torch.float32, device=device),
        adduct=torch.tensor(adducts, dtype=torch.int64, device=device),
        instrument=torch.tensor(instruments, dtype=torch.int64, device=device),
        adduct_mass=adduct_mass - SIMULATOR_ELECTRON_MASS,
        electron_shift=torch.full(
            (len(jobs),),
            -SIMULATOR_ELECTRON_MASS,
            dtype=torch.float64,
            device=device,
        ),
        element_masses=masses,
    )


def predict_spectra(
    generator: GraphGenerator,
    intensity: GraphIntensity,
    batch: IceBatch,
    *,
    max_nodes: int,
    max_depth: int,
    walk_steps: int,
    threshold: float,
) -> IcePrediction:
    """Generate graph fragments and evaluate all shifted intensity contributions.

    Args:
      generator: Native leaving-atom network with strict released checkpoints.
      intensity: Native binned-intensity and fragment-attribution network.
      batch: Complete prepared molecules and physical input tensors.
      max_nodes: Number of retained candidates per molecule.
      max_depth: Maximum number of autoregressive atom-removal iterations.
      walk_steps: Weighted landing-probability channels for the generator.
      threshold: Strict lower bound on accepted accumulated probabilities.

    Returns:
      prediction: Complete generator fields and unfiltered mass/intensity pairs.

    """
    generator.eval()
    intensity.eval()
    generator.requires_grad_(False)
    intensity.requires_grad_(False)
    with torch.no_grad():
        generation = grow_fragments(
            generator,
            batch,
            max_nodes=max_nodes,
            max_depth=max_depth,
            walk_steps=walk_steps,
            threshold=threshold,
        )
        # Intensity clamps a view of the formula counts after this stage boundary.
        snapshot = {key: value.clone() for key, value in generation.items()}
        counts = generation["nfrags"]
        width = int(counts.max())
        root_index = _ids(len(counts), like=counts).repeat_interleave(counts)
        repeated, _, _ = take_graphs(batch.roots, root_index)
        valid = _prefix(counts, max_nodes)
        fragment_mask = generation["frags"][valid]
        root_sizes = generation["natoms"].repeat_interleave(counts)
        atom_valid = _prefix(root_sizes, fragment_mask.shape[1])
        fragments = mask_nodes(
            repeated,
            fragment_mask[atom_valid],
            fragment_mask.sum(1),
        )
        broken = generation["brokens"]
        forms = generation["frag_form_vecs"][:, :width]
        offsets = torch.stack([batch.adduct_mass, batch.electron_shift], dim=1)
        radius = intensity.config.max_hydrogen_shift
        shifts = _ids(2 * radius + 1, like=counts) - radius
        offsets = offsets[:, :, None] + shifts[None, None, :] * float(
            batch.element_masses[HYDROGEN_ROW],
        )
        masses = generation["masses_no_adduct"][:, :width, None, None]
        masses = masses + offsets[:, None, :, :]
        output = intensity.forward(
            IntensityBatch(
                roots=batch.roots,
                fragments=fragments,
                root_index=root_index,
                broken=broken,
                collision_energy=batch.collision_energy,
                precursor=batch.precursor,
                adduct=batch.adduct,
                instrument=batch.instrument,
                root_forms=generation["root_form_vec"],
                fragment_forms=forms,
                fragment_counts=counts,
                max_add_h=broken[:, :width],
                max_remove_h=torch.minimum(
                    broken[:, :width],
                    forms[:, :, HYDROGEN_ROW],
                ),
                masses=masses,
            ),
        )["output"][:, :, 0]
        sizes = counts.tolist()
        spectra = [
            torch.stack(
                [masses[row, :size].reshape(-1), output[row, :size].reshape(-1)],
                dim=1,
            )
            for row, size in enumerate(sizes)
        ]
        masks = [
            generation["frags"][row, :size].repeat_interleave(
                offsets.shape[1] * len(shifts),
                dim=0,
            )
            for row, size in enumerate(sizes)
        ]
    return IcePrediction(generation=snapshot, spectra=spectra, fragments=masks)


def grow_fragments(
    generator: GraphGenerator,
    batch: IceBatch,
    *,
    max_nodes: int,
    max_depth: int,
    walk_steps: int,
    threshold: float,
) -> dict[str, Tensor]:
    """Select unique fragments by accumulated leaving-atom probabilities.

    Args:
      generator: Native graph generator, evaluated without parameter updates.
      batch: Prepared roots and physical conditions; formula counts may be clamped.
      max_nodes: Complete retained candidate width, including the root.
      max_depth: Number of atom-removal iterations.
      walk_steps: Generator landing-probability channels.
      threshold: Strict accumulated-probability acceptance boundary.

    Returns:
      generation: Masks, probabilities, broken orders, formulas, masses and counts.

    """
    roots = batch.roots
    device = roots.features.device
    counts = roots.topology.node_counts
    batch_size, atoms = len(counts), int(counts.max())
    molecule = torch.arange(batch_size, device=device)
    forms = batch.root_forms
    broken = roots.features.new_zeros(batch_size)
    probability = torch.ones_like(broken)
    root_mask = torch.arange(atoms, device=device)[None, :] < counts[:, None]
    probabilities = roots.features.new_zeros((batch_size, max_nodes))
    broken_orders = torch.zeros_like(probabilities)
    masks = root_mask.new_zeros((batch_size, max_nodes, atoms))
    masses = torch.zeros_like(probabilities)
    all_forms = forms.new_zeros((batch_size, max_nodes, forms.shape[-1]))
    hashes = roots.features.new_zeros((batch_size, max_nodes, roots.features.shape[-1]))
    probabilities[:, 0] = probability
    masks[:, 0] = root_mask
    masses[:, 0] = fragment_masses(forms, batch.element_masses)
    all_forms[:, 0] = forms
    hashes[:, 0] = spectrum_hash(roots)
    fragments = roots
    root_features = _with_walks(roots, walk_steps)
    for _ in range(max_depth):
        leaving = generator.forward(
            GenerationBatch(
                roots=root_features,
                fragments=_with_walks(fragments, walk_steps),
                root_index=molecule,
                broken=broken,
                collision_energy=batch.collision_energy,
                precursor=batch.precursor,
                adduct=batch.adduct,
                instrument=batch.instrument,
                root_forms=batch.root_forms,
                fragment_forms=forms,
            ),
        )
        fragment_counts = torch.bincount(molecule, minlength=batch_size)
        candidates = pad_atoms(probability[:, None] * leaving, fragment_counts)
        candidates = candidates.view(batch_size, -1)
        combined = torch.cat([probabilities, candidates], dim=1)
        selected = combined.argsort(dim=1, descending=True, stable=True)[
            :,
            : 2 * max_nodes,
        ]
        rows = torch.arange(batch_size, device=device)[:, None]
        eligible = (selected >= max_nodes) & (combined[rows, selected] > threshold)
        selected_molecule, selected_position = eligible.nonzero(as_tuple=True)
        positions = selected[selected_molecule, selected_position] - max_nodes
        if not len(positions):
            break
        offset = fragment_counts.cumsum(0) - fragment_counts
        graph_index = (
            offset.repeat_interleave(
                torch.bincount(selected_molecule, minlength=batch_size),
            )
            + positions // leaving.shape[1]
        )
        new_graph, cuts, information = remove_atoms(
            fragments,
            graph_index,
            positions % leaving.shape[1],
            {
                "probability": candidates[selected_molecule, positions],
                "molecule": selected_molecule,
                "broken": broken[graph_index],
            },
        )
        if new_graph is None or cuts is None or information is None:
            break
        cuts += information["broken"]
        new_counts = torch.bincount(information["molecule"], minlength=batch_size)
        added_masks = _fragment_masks(
            new_graph,
            information["molecule"],
            new_counts,
            atoms,
        )
        combined_masks = torch.cat([masks, added_masks], dim=1)
        combined_probabilities = torch.cat(
            [probabilities, pad_atoms(information["probability"], new_counts)],
            dim=1,
        )
        new_hashes = spectrum_hash(new_graph)
        combined_hashes = torch.cat([hashes, pad_atoms(new_hashes, new_counts)], dim=1)
        width = combined_hashes.shape[1]
        molecule_ids = torch.arange(batch_size, device=device)[:, None].expand(
            -1,
            width,
        )
        _, first = first_unique(
            torch.cat([molecule_ids[:, :, None], combined_hashes], dim=-1).view(
                -1,
                combined_hashes.shape[-1] + 1,
            ),
        )
        unique = torch.zeros_like(combined_probabilities, dtype=torch.bool)
        unique[first // width, first % width] = True
        combined_probabilities[~unique & (combined_probabilities > 0)] = -1
        selected = combined_probabilities.argsort(
            dim=1,
            descending=True,
            stable=True,
        )[:, :max_nodes]
        probabilities = combined_probabilities.gather(1, selected)
        masks = combined_masks[rows, selected]
        combined_broken = torch.cat([broken_orders, pad_atoms(cuts, new_counts)], dim=1)
        broken_orders = combined_broken.gather(1, selected)
        new_kept = (selected >= max_nodes) & (probabilities > threshold)
        new_rows, new_columns = new_kept.nonzero(as_tuple=True)
        offset = new_counts.cumsum(0) - new_counts
        selected_graphs = (
            selected[new_kept]
            - max_nodes
            + offset.repeat_interleave(
                torch.bincount(new_rows, minlength=batch_size),
            )
        )
        if not len(selected_graphs):
            break
        fragments, _, _ = take_graphs(new_graph, selected_graphs)
        molecule = information["molecule"][selected_graphs]
        broken = cuts[selected_graphs]
        probability = information["probability"][selected_graphs]
        forms = formula_counts(fragments)
        old_kept = selected < max_nodes
        old_rows, _ = old_kept.nonzero(as_tuple=True)
        masses[old_kept] = masses[old_rows, selected[old_kept]]
        masses[new_rows, new_columns] = fragment_masses(forms, batch.element_masses)
        all_forms[old_kept] = all_forms[old_rows, selected[old_kept]]
        all_forms[new_rows, new_columns] = forms
        hashes[old_kept] = hashes[old_rows, selected[old_kept]]
        hashes[new_rows, new_columns] = new_hashes[selected_graphs]
    return {
        "frags": masks,
        "probs": probabilities,
        "brokens": broken_orders,
        "masses_no_adduct": masses,
        "frag_form_vecs": all_forms,
        "root_form_vec": batch.root_forms,
        "natoms": counts,
        "nfrags": (probabilities > threshold).sum(-1),
    }


class IceBackend:
    """Compose prepared root data, native models and complete ICE peak reduction."""

    class Config(Fig["IceBackend"]):
        base_dir: Path | str | None = None
        """Dataset root supplied by the owning stage; paths resolve beneath it."""

        root_dir: Path | str = "native-chemistry/ice-roots"
        """Independently prepared roots and complete monoisotopic mass table."""

        generator: GraphGenerator.Config = field(default_factory=GraphGenerator.Config)
        """Native leaving-atom architecture matching the released generator."""

        intensity: GraphIntensity.Config = field(default_factory=GraphIntensity.Config)
        """Native intensity architecture matching the released model."""

        generator_checkpoint: Path | str = (
            "inputs/datasets/ahmedberatozer/casmi26-iceberg/ckpt/gen.pt"
        )
        """Complete released generator checkpoint, loaded through its strict key map."""

        intensity_checkpoint: Path | str = (
            "inputs/datasets/ahmedberatozer/casmi26-iceberg/ckpt/inten.pt"
        )
        """Complete released intensity checkpoint, loaded through its strict key map."""

        device: str = "cpu"
        """Explicit prediction device for models and numerical inputs."""

        max_nodes: int = 100
        """Retained fragments per molecule, including the root."""

        max_depth: int = 3
        """Autoregressive removal steps from the pinned fragment-engine data."""

        threshold: float = 0.0
        """Strict lower bound on accepted accumulated generation probabilities."""

        reduction: Makeable[PeakReduction] = field(default_factory=SumReduction.Config)
        """Duplicate-mass merge and peak ordering of the public prediction."""

        max_peaks: int = 100
        """Intensity-selected peaks kept per prediction."""

        @override
        def finalize(self) -> Self:
            self.root_dir = resolve_working_dir(self.base_dir, self.root_dir)
            self.generator_checkpoint = resolve_working_dir(
                self.base_dir,
                self.generator_checkpoint,
            )
            self.intensity_checkpoint = resolve_working_dir(
                self.base_dir,
                self.intensity_checkpoint,
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.roots = IceGraphTable(Path(config.root_dir))
        self.generator = config.generator.make()
        self.intensity = config.intensity.make()
        self.reduction = config.reduction.make()
        self.walk_steps = (
            self.generator.config.channels_in - self.intensity.config.channels_in
        )
        load_released_checkpoint(self.generator, Path(config.generator_checkpoint))
        load_released_checkpoint(self.intensity, Path(config.intensity_checkpoint))
        _ = self.generator.to(self.device)
        _ = self.intensity.to(self.device)
        self.feature_failures = 0

    def predict(self, jobs: list[SimulationJob]) -> list[PredictedPeaks | None]:
        """Predict every requested prepared root in encounter order.

        Args:
          jobs: Nonempty covered positive-ion requests in scheduler batch order.

        Returns:
          spectra: Complete top-100 reduced peak arrays in the same order.

        Raises:
          KeyError: A molecule was not included in the declared prepared data.
          ValueError: The batch contains an explicitly rejected prepared root.

        """
        roots: list[IceRoot] = []
        for job in jobs:
            root = self.roots.root(job[0])
            if root is None:
                raise ValueError(f"Rejected prepared ICE root: {job[0]}.")
            roots.append(root)
        batch = pack_ice_batch(
            [root.graph for root in roots],
            root_forms=torch.stack([root.formula for root in roots]),
            jobs=jobs,
            precursor=[0.0] * len(jobs),
            element_masses=torch.from_numpy(self.roots.element_masses),
            device=self.device,
        )
        prediction = predict_spectra(
            self.generator,
            self.intensity,
            batch,
            max_nodes=self.config.max_nodes,
            max_depth=self.config.max_depth,
            walk_steps=self.walk_steps,
            threshold=self.config.threshold,
        )
        return [
            post_prediction(
                spectrum.float().cpu().numpy(),
                reduction=self.reduction,
                max_peaks=self.config.max_peaks,
            )
            for spectrum in prediction.spectra
        ]

    def release_cache(self) -> None:
        """Release the CUDA allocator after an unsuccessful device batch."""
        if self.device.type == "cpu":
            return
        torch.cuda.empty_cache()


logger = logging.getLogger(__name__)


@runtime_checkable
class DatasetRooted(Protocol):
    """A backend Config resolving its paths beneath a parent-supplied dataset root."""

    base_dir: Path | str | None


class PredictionBackend(Protocol):
    """Native neural/feature boundary returning complete postprocessed spectra."""

    feature_failures: int

    def predict(self, jobs: list[SimulationJob]) -> list[PredictedPeaks | None]:
        """Return each completed prediction or explicit per-item failure."""
        ...

    def release_cache(self) -> None:
        """Release device caches after a failed model batch; a no-op on the CPU."""
        ...


@dataclass(slots=True, kw_only=True, eq=False)
class SimulationJobState:
    """Retain all completed jobs, canonical reuse, counters and adaptive batch size."""

    batch_size: int
    cache: dict[SimulationJob, PreparedPeaks | None] = field(
        default_factory=dict,
    )
    canonical_cache: dict[SimulationJob, PreparedPeaks | None] = field(
        default_factory=dict,
    )
    predicted: int = 0
    failed: int = 0
    feature_failures: int = 0
    seconds: float = 0.0
    last_error: str = ""

    def remember_failure(self, error: Exception) -> None:
        """Retain the complete bounded exception representation for GLACIER fail-fast."""
        self.last_error = repr(error)[:300]


class JobScheduling(Protocol):
    """Order, batch and admit one simulator family's prediction jobs."""

    def check(
        self,
        admission: SimulatorAdmissionTable,
    ) -> Callable[[str], tuple[bool, str]]:
        """Return the family's cheap admission decision for one candidate string."""
        ...

    def predict(
        self,
        jobs: set[SimulationJob],
        *,
        canonical: Mapping[str, tuple[str | None, int]],
        state: SimulationJobState,
        backend: PredictionBackend,
        budget: WallBudget,
    ) -> bool:
        """Predict pending jobs; return whether all finished inside the budget."""
        ...


class IceScheduling:
    """Execute condition-grouped ICE jobs in the released length-dependent order."""

    class Config(Fig["IceScheduling"]):
        length_divisors: list[tuple[int, int]] = field(
            default_factory=lambda: [(60, 1), (90, 2), (140, 4)],
        )
        """Pairs of longest SMILES length and the divisor shrinking a batch up to it."""

        longest_divisor: int = 8
        """Batch divisor for SMILES longer than every listed length."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def check(
        self,
        admission: SimulatorAdmissionTable,
    ) -> Callable[[str], tuple[bool, str]]:
        """Return the distinct ICE cheap admission."""
        return admission.ice

    def predict(
        self,
        jobs: set[SimulationJob],
        *,
        canonical: Mapping[str, tuple[str | None, int]],
        state: SimulationJobState,
        backend: PredictionBackend,
        budget: WallBudget,
    ) -> bool:
        """Execute condition-grouped jobs, stopping once the stage budget is spent.

        Args:
          jobs: Complete requested prediction keys.
          canonical: Unused; ICE needs no canonical identities.
          state: Mutable completed-job and adaptive-batch state.
          backend: Native prepared-graph/neural prediction implementation.
          budget: Stage allowance, including earlier preparation/loading costs.

        Returns:
          complete: Whether every requested pending job finished before the deadline.

        """
        del canonical
        pending: dict[tuple[str, str, float], list[str]] = {}
        for job in sorted(jobs - state.cache.keys()):
            pending.setdefault(job[1:], []).append(job[0])
        started = budget.clock()
        for condition in sorted(pending):
            smiles = sorted(pending[condition], key=lambda value: (len(value), value))
            cursor = 0
            while cursor < len(smiles):
                if budget.remaining < 0:
                    state.seconds += budget.clock() - started
                    return False
                length = len(smiles[min(cursor + state.batch_size, len(smiles)) - 1])
                divisor = next(
                    (
                        divisor
                        for longest, divisor in self.config.length_divisors
                        if length <= longest
                    ),
                    self.config.longest_divisor,
                )
                selected = smiles[cursor : cursor + max(1, state.batch_size // divisor)]
                cursor += len(selected)
                selected_jobs = [(value, *condition) for value in selected]
                predictions = _safe_predictions(
                    selected_jobs,
                    state=state,
                    backend=backend,
                    on_failure=_ignore_failure,
                )
                for index, prediction in enumerate(predictions):
                    state.cache[selected_jobs[index]] = _completed_prediction(
                        prediction,
                        state,
                    )
        state.seconds += budget.clock() - started
        return True


class GlacierScheduling:
    """Execute canonical GLACIER jobs with atom-square batch limits and fail-fast."""

    class Config(Fig["GlacierScheduling"]):
        batch_atoms_squared: int = 1_600
        """Atom-square capacity of one released-size batch; larger molecules shrink it."""

        fail_fast_count: int = 64
        """Predictions after which all-failed results abort the stage."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def check(
        self,
        admission: SimulatorAdmissionTable,
    ) -> Callable[[str], tuple[bool, str]]:
        """Return the distinct GLACIER cheap admission."""
        return admission.glacier

    def predict(
        self,
        jobs: set[SimulationJob],
        *,
        canonical: Mapping[str, tuple[str | None, int]],
        state: SimulationJobState,
        backend: PredictionBackend,
        budget: WallBudget,
    ) -> bool:
        """Execute canonical jobs, stopping once the stage budget is spent.

        Args:
          jobs: Complete requested prediction keys.
          canonical: Prepared canonical identity and heavy-atom count per input string.
          state: Mutable completed-job, canonical-cache and adaptive-batch state.
          backend: Native prepared-graph/neural prediction implementation.
          budget: Stage allowance, including earlier preparation/loading costs.

        Returns:
          complete: Whether every requested pending job finished before the deadline.

        Raises:
          RuntimeError: Every prediction of the first configured count failed.

        """
        pending: dict[SimulationJob, list[SimulationJob]] = {}
        for job in sorted(jobs - state.cache.keys()):
            molecule, adduct, instrument, energy = job
            normalized = canonical.get(molecule, (None, 0))[0]
            if normalized is None:
                state.cache[job] = None
                continue
            key = (normalized, adduct, instrument, float(energy))
            if key in state.canonical_cache:
                state.cache[job] = state.canonical_cache[key]
                continue
            pending.setdefault(key, []).append(job)
        heavy = {key: canonical[raw[0][0]][1] for key, raw in pending.items()}
        ordered = sorted(pending, key=lambda key: (heavy[key], *key))
        started = budget.clock()
        cursor = 0
        while cursor < len(ordered):
            if budget.remaining < 0:
                state.seconds += budget.clock() - started
                return False
            atoms = max(
                heavy[key] for key in ordered[cursor : cursor + state.batch_size]
            )
            limit = max(
                1,
                min(
                    state.batch_size,
                    state.batch_size
                    * self.config.batch_atoms_squared
                    // max(1, atoms * atoms),
                ),
            )
            selected = ordered[cursor : cursor + limit]
            cursor += len(selected)
            predictions = _safe_predictions(
                selected,
                state=state,
                backend=backend,
                on_failure=state.remember_failure,
            )
            for index, prediction in enumerate(predictions):
                key = selected[index]
                value = _completed_prediction(prediction, state)
                state.canonical_cache[key] = value
                for raw in pending[key]:
                    state.cache[raw] = value
            if (
                state.predicted >= self.config.fail_fast_count
                and state.failed == state.predicted
            ):
                raise RuntimeError(
                    f"All first {state.predicted} simulator predictions failed: {state.last_error}",
                )
        state.seconds += budget.clock() - started
        return True


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SimulationMolecule:
    """Retain complete ordered molecule queries and candidate strings."""

    mid: str
    candidates: tuple[str, ...]
    spectra: tuple[SpectrumRequest, ...]


@dataclass(slots=True, kw_only=True, eq=False)
class SimulatorResult:
    """Retain every public score, completed prediction and controller counter."""

    scores: dict[str, dict[str, float | None]]
    jobs: SimulationJobState
    status: Literal["init", "ok", "budget", "error"] = "init"
    covered: int = 0
    scored_molecules: int = 0
    scored_candidates: int = 0
    bad_smiles: int = 0
    errors: list[str] = field(default_factory=list)


def run_simulator(
    molecules: Sequence[SimulationMolecule],
    *,
    scheduling: JobScheduling,
    scoring: QueryScoring,
    admission_factory: Callable[[], SimulatorAdmissionTable],
    backend_factory: Callable[[], PredictionBackend],
    publish: Callable[[dict[str, dict[str, float | None]]], None],
    budget: WallBudget.Config,
    batch_size: int,
    chunk_jobs: int,
) -> SimulatorResult:
    """Run admission, full-molecule chunks, native prediction and complete scoring.

    Args:
      molecules: Complete ordered query/candidate rows, retaining repeated IDs.
      scheduling: The family's admission decision and job ordering.
      scoring: Query preparation and prediction scoring.
      admission_factory: Load declared cheap decisions inside the stage budget.
      backend_factory: Load the complete native released model inside the budget.
      publish: Durably publish each complete score map, beginning with all nulls.
      budget: Stage allowance and its wall clock, started here and shared with
        prediction scheduling; infinite seconds leave the deadline unlimited.
      batch_size: Initial prediction geometry, 96 for ICE and 32 for GLACIER.
      chunk_jobs: Flush after a whole molecule reaches this many distinct jobs.

    Returns:
      result: Complete scores, retained prediction state, status and diagnostics.

    """
    if batch_size < 1 or chunk_jobs < 1:
        raise ValueError("Simulator batch and chunk sizes must be positive.")
    origin = budget.clock()
    wall = budget.make()
    result = SimulatorResult(
        scores={
            molecule.mid: dict.fromkeys(molecule.candidates) for molecule in molecules
        },
        jobs=SimulationJobState(batch_size=batch_size),
    )
    publish(result.scores)
    try:
        admission, backend = admission_factory(), backend_factory()
        logger.info(
            "Simulator loaded in %.2fs; elapsed %.2fs.",
            wall.elapsed,
            wall.clock() - origin,
        )
        plans = _plans(molecules, result=result, scoring=scoring)
        result.covered = sum(bool(plan) for plan in plans.values())
        accepted: dict[str, bool] = {}
        chunk: list[tuple[str, list[str]]] = []
        jobs: set[SimulationJob] = set()
        complete = True
        for molecule in molecules:
            if not plans[molecule.mid]:
                continue
            candidates = _admit(
                molecule.candidates,
                accepted=accepted,
                check=scheduling.check(admission),
                result=result,
            )
            if not candidates:
                continue
            chunk.append((molecule.mid, candidates))
            jobs.update(required_jobs(plans[molecule.mid], candidates))
            if len(jobs) < chunk_jobs:
                continue
            complete = _execute_chunk(
                chunk,
                jobs=jobs,
                plans=plans,
                result=result,
                backend=backend,
                admission=admission,
                scheduling=scheduling,
                scoring=scoring,
                budget=wall,
                publish=publish,
                origin=origin,
            )
            chunk.clear()
            jobs.clear()
            if not complete:
                break
        if complete:
            complete = _execute_chunk(
                chunk,
                jobs=jobs,
                plans=plans,
                result=result,
                backend=backend,
                admission=admission,
                scheduling=scheduling,
                scoring=scoring,
                budget=wall,
                publish=publish,
                origin=origin,
            )
        result.status = "ok" if complete else "budget"
    except BaseException as error:
        logger.exception(
            "Simulator stage failed; retaining all completed publications.",
        )
        result.status = "error"
        result.errors.append(repr(error)[:3000])
    publish(result.scores)
    return result


def _plans(
    molecules: Sequence[SimulationMolecule],
    *,
    result: SimulatorResult,
    scoring: QueryScoring,
) -> dict[str, list[SimulationSpectrum]]:
    plans: dict[str, list[SimulationSpectrum]] = {}
    for molecule in molecules:
        try:
            plans[molecule.mid] = scoring.plan(molecule.spectra)
        except Exception as error:
            logger.exception("Simulator query planning failed.")
            result.errors.append(f"{molecule.mid}: plan {repr(error)[:200]}")
            plans[molecule.mid] = []
    return plans


def _admit(
    candidates: Sequence[str],
    *,
    accepted: dict[str, bool],
    check: Callable[[str], tuple[bool, str]],
    result: SimulatorResult,
) -> list[str]:
    unique = list(dict.fromkeys(candidates))
    for candidate in unique:
        if candidate in accepted:
            continue
        allowed, reason = check(candidate)
        accepted[candidate] = allowed
        if allowed:
            continue
        result.bad_smiles += 1
        if result.bad_smiles <= 5:
            result.errors.append(f"smiles rejected ({reason}): {candidate[:100]}")
        if result.bad_smiles >= 50 and result.bad_smiles == len(accepted):
            raise RuntimeError(
                f"first {result.bad_smiles} SMILES all rejected, last: {reason}",
            )
    return [candidate for candidate in unique if accepted[candidate]]


def _execute_chunk(
    chunk: Sequence[tuple[str, list[str]]],
    *,
    jobs: set[SimulationJob],
    plans: dict[str, list[SimulationSpectrum]],
    result: SimulatorResult,
    backend: PredictionBackend,
    admission: SimulatorAdmissionTable,
    scheduling: JobScheduling,
    scoring: QueryScoring,
    budget: WallBudget,
    publish: Callable[[dict[str, dict[str, float | None]]], None],
    origin: float,
) -> bool:
    if not chunk:
        return True
    started = budget.clock()
    complete = scheduling.predict(
        jobs,
        canonical=admission.canonical,
        state=result.jobs,
        backend=backend,
        budget=budget,
    )
    if complete:
        for mid, candidates in chunk:
            try:
                scores = scoring.score(plans[mid], candidates, result.jobs.cache)
                result.scores[mid].update(scores)
                result.scored_molecules += 1
                result.scored_candidates += sum(
                    score is not None for score in scores.values()
                )
            except Exception as error:
                logger.exception("Simulator molecule scoring failed.")
                result.errors.append(f"{mid}: score {repr(error)[:200]}")
        publish(result.scores)
        elapsed = budget.clock() - started
        logger.info(
            "Published %s complete molecules in %.2fs; elapsed %.2fs.",
            len(chunk),
            elapsed,
            budget.clock() - origin,
        )
    return complete


def _shortest_paths(
    adjacency: NDArray[np.bool_],
) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
    """Choose the smallest predecessor among each complete breadth-first layer."""
    count = len(adjacency)
    distances = np.full((count, count), 32_767, dtype=np.int16)
    parents = np.full((count, count), -1, dtype=np.int16)
    for source in range(count):
        distances[source, source] = 0
        parents[source, source] = source
        visited = {source}
        frontier = [source]
        depth = 0
        while frontier:
            found: set[int] = set()
            for previous in frontier:
                neighbors = np.flatnonzero(
                    cast("NDArray[np.bool_]", adjacency[previous]),
                )
                for neighbor in neighbors:
                    target = int(neighbor)
                    if target in visited:
                        continue
                    if target not in found:
                        parents[source, target] = previous
                    found.add(target)
            depth += 1
            frontier = sorted(found)
            distances[source, frontier] = depth
            visited.update(found)
    return distances, parents


def _atomic_graph(row: AtomicRow) -> AtomicGraph | None:
    return (
        AtomicGraph(
            features=torch.tensor(row["features"], dtype=torch.float32).reshape(
                -1,
                ATOM_FEATURES,
            ),
            bonds=torch.tensor(row["bonds"], dtype=torch.int64).reshape(-1, 2),
            bond_features=torch.tensor(
                row["bond_features"],
                dtype=torch.float32,
            ).reshape(-1, BOND_FEATURES),
            weights=torch.tensor(row["weights"], dtype=torch.float32),
            degrees=torch.tensor(row["degrees"], dtype=torch.int64),
        )
        if row["valid"]
        else None
    )


def _assemble(output: GlacierOutput, *, shifts: int) -> dict[str, list[Tensor]]:
    fragments = output["frags_pred"]
    masks = pad_atoms(fragments["fragments"], fragments["fragment_count"])
    spectra = output["inten_pred_end_to_end"]["output"]
    result: dict[str, list[Tensor]] = {"spec": [], "frag": []}
    for index, count in enumerate(fragments["fragment_count"]):
        result["spec"].append(spectra[index, : int(count) * shifts])
        result["frag"].append(masks[index, :count].repeat_interleave(shifts, dim=0))
    return result


def _sum_nodes(graph: FragmentGraph, features: Tensor) -> Tensor:
    counts = graph.topology.node_counts
    return features.new_zeros((len(counts), *features.shape[1:])).index_add_(
        0,
        torch.repeat_interleave(counts),
        features,
    )


def _aggregate(values: Tensor, labels: Tensor, count: int, operation: str) -> Tensor:
    result = values.new_full((count + 1,), -1)
    _ = result.scatter_reduce_(0, labels, values, operation, include_self=False)
    return result[1:]


def _components(topology: GraphBatch, count: int) -> Tensor:
    """Label undirected paired edges contiguously, with zero for isolated atoms."""
    source, destination = topology.source, topology.destination
    labels = source.new_ones((count,)).cumsum(0)
    while True:
        left = torch.zeros_like(labels).scatter_reduce_(
            0,
            destination,
            labels[source],
            "amin",
            include_self=False,
        )
        right = torch.zeros_like(labels).scatter_reduce_(
            0,
            source,
            labels[destination],
            "amin",
            include_self=False,
        )
        updated = torch.minimum(labels, torch.minimum(left, right))
        if torch.equal(updated, labels):
            break
        labels = updated
    included = torch.bincount(source, minlength=count) != 0
    result = torch.zeros_like(labels)
    result[included] = torch.unique(labels[included], return_inverse=True)[1] + 1
    return result


def _with_walks(graph: FragmentGraph, steps: int) -> FragmentGraph:
    return FragmentGraph(
        topology=graph.topology,
        features=torch.cat([graph.features, walk_features(graph, steps=steps)], dim=-1),
        node_ids=graph.node_ids,
    )


def _fragment_masks(
    graph: FragmentGraph,
    molecules: Tensor,
    counts: Tensor,
    atoms: int,
) -> Tensor:
    offsets = counts.cumsum(0) - counts
    positions = molecules.new_ones(len(molecules)).cumsum(0) - 1
    positions -= offsets.repeat_interleave(counts)
    graph_index = torch.repeat_interleave(graph.topology.node_counts)
    result = counts.new_zeros(
        (len(counts), int(counts.max()), atoms),
        dtype=torch.bool,
    )
    result[molecules[graph_index], positions[graph_index], graph.node_ids] = True
    return result


def _ids(count: int, *, like: Tensor) -> Tensor:
    return torch.arange(count, device=like.device)


def _prefix(counts: Tensor, width: int) -> Tensor:
    return _ids(width, like=counts)[None, :] < counts[:, None]


def _safe_predictions(
    jobs: list[SimulationJob],
    *,
    state: SimulationJobState,
    backend: PredictionBackend,
    on_failure: Callable[[Exception], None],
) -> list[PredictedPeaks | None]:
    try:
        return backend.predict(jobs)
    except Exception as error:
        logger.exception("Simulator prediction failed; retrying smaller batches.")
        on_failure(error)
        backend.release_cache()
        if "out of memory" in str(error).lower() and len(jobs) > 4:
            state.batch_size = min(state.batch_size, max(4, len(jobs) // 2))
    finally:
        state.feature_failures = backend.feature_failures
    if len(jobs) == 1:
        return [None]
    width = max(1, len(jobs) // 4)
    result: list[PredictedPeaks | None] = []
    for start in range(0, len(jobs), width):
        result.extend(
            _safe_predictions(
                jobs[start : start + width],
                state=state,
                backend=backend,
                on_failure=on_failure,
            ),
        )
    return result


def _ignore_failure(error: Exception) -> None:
    del error


def _completed_prediction(
    prediction: PredictedPeaks | None,
    state: SimulationJobState,
) -> PreparedPeaks | None:
    state.predicted += 1
    state.failed += int(prediction is None)
    return (
        (prediction[0].astype(np.float64), prediction[1].astype(np.float64))
        if prediction is not None and len(prediction[0])
        else None
    )
