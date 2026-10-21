"""Native independent second engine: banks, features, boosting and ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, Self, TypedDict, cast, override

import heapq
import math

from configgle import Fig
from numpy.typing import NDArray
from torch import Tensor
from treekle import from_plain, json_loads

import numpy as np
import torch

from priml.baselines.casmi2026.chemistry import (
    IdentityTable,
    PeakStore,
    SecondSearch,
)
from priml.baselines.casmi2026.custom_types import (
    Analog,
    CandidateMatrix,
    CandidateSet,
    EngineDiagnostics,
    EngineIdentityRow,
    EngineResult,
    FloatArray,
    PoolIndices,
    PreparationHeader,
    RankingScores,
    SpectrumRequest,
)
from priml.baselines.casmi2026.isolated import (
    HYDROGEN_MASS,
    IDENTITY_RDKIT,
    polarity_sign,
    second_adduct_parameters,
    second_neutral_mass,
)
from priml.baselines.casmi2026.model import SecondFPNet
from priml.baselines.casmi2026.scores import ordinal_ranks, population_z
from priml.data.arrays import load_array
from priml.model.ae.checkpoint import LocalFile
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyarrow import parquet

    from priml.baselines.casmi2026.custom_types import PredictionMatrix
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")


SECOND_ADDUCTS: Final = (
    "[M+H]+",
    "[M+NH4]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M-2H2O+H]+",
    "[M]+",
    "[M-H]-",
    "[M-H2O-H]-",
    "[M+CH2O2-H]-",
    "[M+C2H4O2-H]-",
    "[M+Cl]-",
    "[M]-",
    "[M+2H]2+",
    "[M-2H]-",
    "[2M+H]+",
    "[2M+Na]+",
    "[2M+NH4]+",
    "[2M-H]-",
    "[2M+K]+",
    "[2M+CH2O2-H]-",
    "[2M+C2H4O2-H]-",
    "[2M+Na-2H]-",
    "[M+Na-2H]-",
    "[M-H2O]+",
    "<unk>",
)


@dataclass(frozen=True, kw_only=True, slots=True)
class QuerySpectrum:
    """Carry raw peaks and the metadata consumed by the second-engine bank."""

    mass: FloatArray
    intensity: FloatArray
    precursor: float
    adduct: str
    instrument: str | None
    collision_energy: float
    polarity: float


def instrument_family(instrument: str | None) -> int:
    """Map the public instrument spelling to its five-way model vocabulary.

    Args:
      instrument: Raw instrument label, or None for unknown.

    Returns:
      index: timsTOF, Orbitrap, QTOF, ion trap or other vocabulary position.

    """
    if instrument is None:
        return 4
    name = instrument.lower()
    if "timstof" in name:
        return 0
    if any(
        marker in name
        for marker in ("orbitrap", "qft", "ftms", "hybrid ft", "itft", "exactive")
    ):
        return 1
    if "tof" in name:
        return 2
    if "trap" in name or "qq" in name:
        return 3
    return 4


class SecondPeaks:
    """Prepare and merge raw peaks with the second engine's window-quota rules."""

    class Config(Fig["SecondPeaks"]):
        max_peaks: int = 128
        """Maximum retained peak count."""

        floor: float = 0.001
        """Relative base-peak cutoff."""

        window: float = 50.0
        """Mass interval used for quota balancing."""

        per_window: int = 8
        """First-pass peak quota per mass interval."""

        margin: float = 1.5
        """Mass above the precursor still admitted to the model."""

        merge_tolerance: float = 0.005
        """Adjacent merged masses closer than this keep only the stronger peak."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def prepare(
        self,
        mass: FloatArray,
        intensity: FloatArray,
        *,
        precursor: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Prepare window-balanced, square-root intensities for second FPNet.

        Args:
          mass: Raw peak masses, retaining nonpositive masses as the released engine does.
          intensity: Aligned raw linear intensities.
          precursor: Mass boundary, extended by the configured margin.

        Returns:
          mass: Sorted float32 selected masses.
          intensity: Float32 square-root intensities relative to the selected maximum.

        """
        config = self.config
        masses = np.asarray(mass, np.float64)
        values = np.asarray(intensity, np.float64)
        allowed = masses <= precursor + config.margin
        masses, values = masses[allowed], values[allowed]
        if not len(masses) or values.max() <= 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        retained = values >= config.floor * values.max()
        masses, values = masses[retained], values[retained]
        if len(masses) > config.max_peaks:
            selection = _balanced_indices(
                masses,
                values,
                config.max_peaks,
                config.window,
                config.per_window,
            )
            masses, values = masses[selection], values[selection]
        order = np.argsort(masses, kind="stable")
        return masses[order].astype(np.float32), np.sqrt(
            values[order] / values.max(),
        ).astype(np.float32)

    def merge(
        self,
        spectra: Sequence[tuple[FloatArray, FloatArray]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Merge raw spectra with the second engine's adjacent-peak comparison.

        Args:
          spectra: Raw peak pairs; each nonempty spectrum is base-peak normalized.

        Returns:
          mass: Surviving masses in sorted order.
          intensity: Corresponding normalized intensities in float64.

        """
        present = [
            (np.asarray(mass, np.float64), np.asarray(values, np.float64))
            for mass, values in spectra
            if len(mass)
        ]
        if not present:
            return np.zeros(0), np.zeros(0)
        masses = np.concatenate([mass for mass, _ in present])
        values = np.concatenate(
            [value / max(float(value.max()), 1e-9) for _, value in present],
        )
        order = np.argsort(masses, kind="stable")
        masses, values = masses[order], values[order]
        rejected_mask = np.zeros(len(masses), np.bool_)
        for index in range(1, len(masses)):
            if (
                masses.item(index) - masses.item(index - 1)
                < self.config.merge_tolerance
            ):
                rejected = (
                    index - 1 if values.item(index) >= values.item(index - 1) else index
                )
                rejected_mask[rejected] = True
        return masses[~rejected_mask], values[~rejected_mask]


def predict_second(
    spectra: Sequence[QuerySpectrum],
    *,
    single: Sequence[SecondFPNet],
    merged: Sequence[SecondFPNet],
    device: str,
    peaks: SecondPeaks,
    merged_energy: float,
) -> NDArray[np.float32] | None:
    """Average per-spectrum, per-model and single/merged fingerprint views.

    Args:
      spectra: Raw query spectra with metadata in source encounter order.
      single: Evaluation-mode single-spectrum models.
      merged: Evaluation-mode merged-spectrum models.
      device: Device receiving packed model inputs.
      peaks: Peak preparation and merging recipe.
      merged_energy: Collision energy assigned to the merged spectrum.

    Returns:
      logits: Mean of the available single and merged banks, or None without peaks.

    """
    outputs: list[NDArray[np.float32]] = []
    if single:
        prepared = [
            peaks.prepare(item.mass, item.intensity, precursor=item.precursor)
            for item in spectra
        ]
        result = _predict_batch(prepared, spectra, single, device)
        if result is not None:
            outputs.append(result)
    if merged:
        masses, values = peaks.merge(
            [(item.mass, item.intensity) for item in spectra],
        )
        precursor = from_plain(
            cast(object, np.median([item.precursor for item in spectra])),
            float,
        )
        merged_item = QuerySpectrum(
            mass=masses,
            intensity=values,
            precursor=precursor,
            adduct=spectra[0].adduct,
            instrument=spectra[0].instrument,
            collision_energy=merged_energy,
            polarity=np.asarray([item.polarity for item in spectra], np.float64)
            .mean()
            .item(),
        )
        result = _predict_batch(
            [peaks.prepare(masses, values, precursor=precursor)],
            [merged_item],
            merged,
            device,
        )
        if result is not None:
            outputs.append(result)
    return np.asarray(np.mean(outputs, axis=0), np.float32) if outputs else None


def _balanced_indices(
    masses: NDArray[np.float64],
    intensity: NDArray[np.float64],
    maximum: int,
    window: float,
    per_window: int,
) -> NDArray[np.int64]:
    order = np.argsort(-intensity, kind="stable")
    buckets = (masses // window).astype(np.int64)
    counts: dict[int, int] = {}
    selected: list[int] = []
    for position in range(len(order)):
        index = order.item(position)
        bucket = buckets.item(index)
        used = counts.get(bucket, 0)
        if used < per_window:
            counts[bucket] = used + 1
            selected.append(index)
    indices = np.asarray(selected, np.int64)
    if len(indices) > maximum:
        return indices[np.argsort(-intensity[indices], kind="stable")[:maximum]]
    if len(indices) < maximum:
        chosen = set(selected)
        remaining = [
            order.item(index)
            for index in range(len(order))
            if order.item(index) not in chosen
        ]
        indices = np.concatenate(
            (indices, np.asarray(remaining[: maximum - len(indices)], np.int64)),
        )
    return indices


@torch.no_grad()
def _predict_batch(
    prepared: Sequence[tuple[NDArray[np.float32], NDArray[np.float32]]],
    spectra: Sequence[QuerySpectrum],
    models: Sequence[SecondFPNet],
    device: str,
) -> NDArray[np.float32] | None:
    keep = [index for index, (mass, _) in enumerate(prepared) if len(mass)]
    if not keep:
        return None
    peak_count = max(len(prepared[index][0]) for index in keep)
    masses = np.zeros((len(keep), peak_count), np.float32)
    intensity = np.zeros_like(masses)
    padding = np.ones(masses.shape, np.bool_)
    for row, index in enumerate(keep):
        mass, values = prepared[index]
        masses[row, : len(mass)] = mass
        intensity[row, : len(values)] = values
        padding[row, : len(mass)] = False
    selected = [spectra[index] for index in keep]
    adduct_indices = {name: index for index, name in enumerate(SECOND_ADDUCTS)}
    arguments: tuple[Tensor, ...] = (
        torch.as_tensor(masses, device=device),
        torch.as_tensor(intensity, device=device),
        torch.as_tensor(padding, device=device),
        torch.tensor(
            [item.precursor for item in selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            [
                adduct_indices.get(item.adduct, adduct_indices["<unk>"])
                for item in selected
            ],
            device=device,
        ),
        torch.tensor(
            [instrument_family(item.instrument) for item in selected],
            device=device,
        ),
        torch.tensor(
            [item.collision_energy for item in selected],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            [item.polarity for item in selected],
            dtype=torch.float32,
            device=device,
        ),
    )
    outputs = [
        model.forward(*arguments).float().mean(0).cpu().numpy() for model in models
    ]
    return np.asarray(np.mean(outputs, axis=0), np.float32)


def analog_features(
    fingerprints: NDArray[np.float32],
    library: NDArray[np.float32],
    analogs: NDArray[np.float32] | None,
    similarity: NDArray[np.float32],
    *,
    logits: FloatArray | None = None,
    fragments: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Calculate all 31 public second-engine ranker columns.

    Args:
      fingerprints: Candidate selected binary fingerprints.
      library: Maximum spectral-library evidence per candidate.
      analogs: Selected fingerprints of analogs in similarity order.
      similarity: Analog spectral similarities; negative values are clipped to zero.
      logits: Optional predicted fingerprint logits for the query molecule.
      fragments: Optional fragment-explanation scores for each candidate.

    Returns:
      features: The complete ordered float32 matrix consumed by the released fits.

    """
    candidates = fingerprints.astype(np.float32)
    count = len(candidates)
    cardinality = candidates.sum(1)
    library = np.asarray(library, np.float32)
    library_max = float(library.max()) if count else 0.0
    propagation, linear, best, leading, mean, leading_sim = _analog_evidence(
        candidates,
        analogs,
        similarity,
    )
    maximum = float(propagation.max()) if count else 0.0
    model = _model_evidence(candidates, cardinality, logits)
    fragment: list[FloatArray] = [np.zeros(count, np.float32)] * 4
    if fragments is not None:
        fragments = np.asarray(fragments, np.float32)
        fragment = [
            fragments,
            ordinal_ranks(fragments),
            fragments - fragments.max() if count else fragments,
            population_z(fragments),
        ]
    interactions = _model_interactions(candidates, library, propagation, logits)
    columns = [
        library,
        ordinal_ranks(library),
        np.full(count, library_max),
        library - library_max,
        (library > 0).astype(float),
        propagation,
        ordinal_ranks(propagation),
        np.full(count, maximum),
        propagation - maximum,
        linear,
        best,
        leading,
        mean,
        np.full(count, leading_sim),
        np.full(count, from_plain(cast(object, np.log(max(count, 1))), float)),
        *model,
        *fragment,
        *interactions,
    ]
    return np.column_stack(columns).astype(np.float32)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class FragmentMassStore:
    """Hold full unrounded F64 engine2 fragments in its mass-sorted pool row order."""

    offsets: NDArray[np.int64]
    mass: NDArray[np.float64]

    def row(self, index: int) -> NDArray[np.float64]:
        """Return every neutral fragment mass of one engine2 candidate."""
        return self.mass[self.offsets.item(index) : self.offsets.item(index + 1)]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SecondFragments:
    """Keep all four source aggregations from the three full spectrum-score panels."""

    maximum: NDArray[np.float32]
    adduct: NDArray[np.float32]
    mean: NDArray[np.float32]
    polarity: NDArray[np.float32]


class FingerprintPool(Protocol):
    """Require selected fingerprints without fabricating chemistry-only pool columns."""

    def fingerprints(self, indices: NDArray[np.int64], /) -> NDArray[np.uint8]:
        """Return every selected bit for the original positional pool indices."""
        ...


def second_fragments(
    spectra: Sequence[SpectrumRequest],
    *,
    rows: NDArray[np.int64],
    fragments: FragmentMassStore,
    search: SecondSearch,
) -> SecondFragments:
    """Compute complete mixed-polarity, adduct and per-polarity fragment evidence.

    Args:
      spectra: Query peaks after the second engine's precursor-plus-two cut.
      rows: Engine2 pool rows in candidate order, retaining duplicates.
      fragments: Independently prepared F64 full neutral fragment masses.
      search: Peak preparation of the query spectra.

    Returns:
      evidence: All four source fragment summaries in original candidate order.

    """
    queries = [
        search.prepare(
            (
                np.asarray(spectrum["mz"], dtype=np.float32),
                np.asarray(spectrum["it"], dtype=np.float32),
            ),
        )
        for spectrum in spectra
    ]
    modes = np.array(
        [1 if spectrum["mode"] in (1, "positive") else -1 for spectrum in spectra],
        dtype=np.float64,
    )
    mean_mode = modes.mean().item() if len(modes) else float("nan")
    mixed = np.zeros((len(rows), max(len(queries), 1)), dtype=np.float32)
    by_adduct, by_polarity = np.zeros_like(mixed), np.zeros_like(mixed)
    for position in range(len(rows)):
        masses = fragments.row(rows.item(position))
        for qi, (mass, intensity) in enumerate(queries):
            query = mass.astype(np.float64), intensity.astype(np.float64)
            mixed[position, qi] = _explanation(
                masses,
                query=query,
                carriers=_proton_carrier(mean_mode),
            )
            by_adduct[position, qi] = _explanation(
                masses,
                query=query,
                carriers=_adduct_carriers(spectra[qi]["adduct"]),
            )
            by_polarity[position, qi] = _explanation(
                masses,
                query=query,
                carriers=_proton_carrier(modes.item(qi)),
            )
    return SecondFragments(
        maximum=mixed.max(1),
        adduct=by_adduct.max(1),
        mean=by_polarity.mean(1),
        polarity=by_polarity.max(1),
    )


def second_features(
    pool: FingerprintPool,
    *,
    rows: NDArray[np.int64],
    channels: SecondChannels,
    fragments: SecondFragments,
    logits: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    """Build every base31/libx8/anapol5/fragx7 column in released archive order.

    Args:
      pool: Engine2 selected-fingerprint pool; its vocabulary differs from the main engine's.
      rows: Selected candidate rows retaining positional duplicates.
      channels: Complete direct and analog retrieval output.
      fragments: Complete source fragment summaries aligned to the selected rows.
      logits: Predicted selected-bit logits; None disables fingerprint features.

    Returns:
      features: Complete ordered float32 candidate matrix with 51 columns.

    """
    fingerprints = pool.fingerprints(rows).astype(np.float32)
    aligned = [np.zeros(len(rows), dtype=np.float32) for _ in range(4)]
    positions = {channels.pool_rows.item(i): i for i in range(len(channels.pool_rows))}
    for i in range(len(rows)):
        if (position := positions.get(rows.item(i))) is not None:
            for output, source in zip(
                aligned,
                (channels.maximum, channels.mean, channels.shifted, channels.count),
                strict=True,
            ):
                output[i] = source[position]
    analog_rows = np.array([analog.sid for analog in channels.analogs], dtype=np.int64)
    analog_fp = (
        pool.fingerprints(analog_rows).astype(np.float32) if len(analog_rows) else None
    )
    similarity = np.array(
        [analog.similarity for analog in channels.analogs],
        dtype=np.float32,
    )
    base = analog_features(
        fingerprints,
        aligned[0],
        analog_fp,
        similarity,
        logits=logits,
        fragments=fragments.maximum,
    )
    count = np.log1p(aligned[3].astype(np.float64)).astype(np.float32)
    library = np.column_stack(
        [*_relative(aligned[1]), *_relative(aligned[2]), count, count - count.max()],
    )
    polar = _polarity_features(pool, fingerprints=fingerprints, channels=channels)
    fragment = np.column_stack(
        [*_relative(fragments.adduct), *_relative(fragments.mean), fragments.polarity],
    )
    return np.hstack([base, library, polar, fragment]).astype(np.float32)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SecondLibrary:
    """Carry raw second-engine spectra and its independent formula-derived mass index."""

    store: PeakStore
    mass: NDArray[np.float64]
    pool_rows: NDArray[np.int64]
    polarity: NDArray[np.int8]
    timstof: NDArray[np.bool_]
    precursor: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SecondRepresentatives:
    """Keep the timsTOF-preferred spectrum and its fully cleaned reusable peaks."""

    rows: NDArray[np.int64]
    pool_rows: NDArray[np.int64]
    mass: NDArray[np.float64]
    store: PeakStore


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SecondIndex:
    """Retain the full library plus separate all/positive/negative representatives."""

    rows: NDArray[np.int64]
    mass: NDArray[np.float64]
    all: SecondRepresentatives
    positive: SecondRepresentatives
    negative: SecondRepresentatives


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SecondChannels:
    """Retain every raw library channel and both ordered analog lists."""

    pool_rows: NDArray[np.int64]
    maximum: NDArray[np.float32]
    mean: NDArray[np.float32]
    shifted: NDArray[np.float32]
    count: NDArray[np.float32]
    analogs: tuple[Analog, ...]
    polarity_analogs: tuple[Analog, ...]


class SecondRetrieval:
    """Assemble the second engine's distinct representative and direct-search rules."""

    class Config(Fig["SecondRetrieval"]):
        search: SecondSearch.Config = field(default_factory=SecondSearch.Config)
        """Complete float32 spectral preparation and matching."""

        analog_window: float = 200.0
        """Inclusive neutral-mass analog half-window in daltons."""

        ppm: float = 10.0
        """Direct library mass-window half-width in parts per million."""

        n_analog: int = 200
        """Public BIO/AFIX notebook's selected distinct analog count."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.search = config.search.make()

    def index(self, library: SecondLibrary) -> SecondIndex:
        """Build all three timsTOF-first representative stores from complete raw rows.

        Args:
          library: Raw complete spectra and formula-derived second-engine metadata.

        Returns:
          index: Stable raw mass index and all three full cleaned representative stores.

        """
        rows = np.flatnonzero(np.isfinite(library.mass) & (library.pool_rows >= 0))
        ordered = rows[np.argsort(library.mass[rows], kind="mergesort")]
        return SecondIndex(
            rows=ordered,
            mass=library.mass[ordered],
            all=self._representatives(library, rows=rows),
            positive=self._representatives(
                library,
                rows=rows[library.polarity[rows] == 1],
            ),
            negative=self._representatives(
                library,
                rows=rows[library.polarity[rows] == -1],
            ),
        )

    def channels(
        self,
        spectra: Sequence[SpectrumRequest],
        *,
        library: SecondLibrary,
        index: SecondIndex,
        target: float,
    ) -> SecondChannels:
        """Aggregate direct and precursor-shifted library evidence and both analog lists.

        Args:
          spectra: Raw query spectra retaining polarity and precursor metadata.
          library: Complete raw second-engine spectral library.
          index: Corresponding complete representative and mass index.
          target: Query neutral mass.

        Returns:
          channels: All library fields and strict source-order analog evidence.

        """
        prepared = [
            (
                self.search.prepare(
                    (
                        np.asarray(item["mz"], dtype=np.float32),
                        np.asarray(item["it"], dtype=np.float32),
                    ),
                ),
                item,
            )
            for item in spectra
        ]
        queries = [(peaks, item) for peaks, item in prepared if len(peaks[0])]
        tolerance = target * self.config.ppm / 1e6
        begin = int(np.searchsorted(index.mass, target - tolerance))
        end = int(np.searchsorted(index.mass, target + tolerance, side="right"))
        rows = index.rows[begin:end]
        unique, inverse = np.unique(library.pool_rows[rows], return_inverse=True)
        direct = np.zeros((max(len(queries), 1), len(unique)), dtype=np.float32)
        shifted = np.zeros_like(direct)
        for position, (peaks, item) in enumerate(queries):
            if not len(rows):
                break
            scores = self.search.score(peaks, store=library.store, candidates=rows)
            np.maximum.at(direct[position, :], inverse, scores)
            shifts = (item["prec"] - library.precursor[rows]).astype(np.float32)
            scores = self.search.score(
                peaks,
                store=library.store,
                candidates=rows,
                shifts=shifts,
            )
            np.maximum.at(shifted[position, :], inverse, scores)
        peaks = [query for query, _ in queries]
        ids, scores = self._analog_search(index.all, queries=peaks, target=target)
        all_analogs = self._top_analogs(ids, scores=scores)
        polarity_ids: list[NDArray[np.int64]] = []
        polarity_scores: list[NDArray[np.float32]] = []
        for sign, representatives in ((1, index.positive), (-1, index.negative)):
            group = [
                query for query, item in queries if polarity_sign(item["mode"]) == sign
            ]
            if group:
                ids, scores = self._analog_search(
                    representatives,
                    queries=group,
                    target=target,
                )
                polarity_ids.append(ids)
                polarity_scores.append(scores)
        return SecondChannels(
            pool_rows=unique,
            maximum=direct.max(0),
            mean=direct.mean(0),
            shifted=shifted.max(0),
            count=np.bincount(inverse, minlength=len(unique)).astype(np.float32),
            analogs=all_analogs,
            polarity_analogs=self._top_analogs(
                np.concatenate(polarity_ids),
                scores=np.concatenate(polarity_scores),
            )
            if polarity_ids
            else (),
        )

    def _representatives(
        self,
        library: SecondLibrary,
        *,
        rows: NDArray[np.int64],
    ) -> SecondRepresentatives:
        counts = np.diff(library.store.offsets)[rows]
        ranked = rows[
            np.lexsort(
                (
                    rows,
                    -counts,
                    -library.timstof[rows].astype(np.int8),
                    library.pool_rows[rows],
                ),
            )
        ]
        boundaries = np.concatenate(
            [[True], np.not_equal(np.diff(library.pool_rows[ranked]), 0)],
        )
        chosen = ranked[boundaries] if len(ranked) else ranked
        chosen = chosen[np.argsort(library.mass[chosen], kind="mergesort")]
        return SecondRepresentatives(
            rows=chosen,
            pool_rows=library.pool_rows[chosen],
            mass=library.mass[chosen],
            store=self.search.clean_store(library.store, indices=chosen),
        )

    def _analog_search(
        self,
        representatives: SecondRepresentatives,
        *,
        queries: Sequence[tuple[NDArray[np.float32], NDArray[np.float32]]],
        target: float,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        begin = int(
            np.searchsorted(representatives.mass, target - self.config.analog_window),
        )
        end = int(
            np.searchsorted(
                representatives.mass,
                target + self.config.analog_window,
                side="right",
            ),
        )
        scores = np.zeros(end - begin, dtype=np.float32)
        shifts = (target - representatives.mass[begin:end]).astype(np.float32)
        if end > begin:
            for query in queries:
                scores = np.maximum(
                    scores,
                    self.search.score_prepared(
                        query,
                        store=representatives.store,
                        start=begin,
                        stop=end,
                        shifts=shifts,
                    ),
                )
        return representatives.pool_rows[begin:end], scores

    def _top_analogs(
        self,
        rows: NDArray[np.int64],
        *,
        scores: NDArray[np.float32],
    ) -> tuple[Analog, ...]:
        best: dict[int, float] = {}
        for position in range(len(rows)):
            row, value = rows.item(position), scores.item(position)
            if row not in best or value > best[row]:
                best[row] = value
        selected = sorted(best, key=lambda row: -best[row])[: self.config.n_analog]
        return tuple(Analog(sid=row, similarity=best[row]) for row in selected)


class PoolIdentity(TypedDict):
    """Actual engine2 prepared metadata without main-engine-only property fields."""

    key: str
    smiles: str


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SecondPool:
    """Retain only the actual source pool fields, including positional duplicates."""

    mass: NDArray[np.float64]
    keys: tuple[str, ...]
    smiles: tuple[str, ...]
    packed: NDArray[np.uint8]
    bits: NDArray[np.int64]

    @classmethod
    def load(cls, directory: Path) -> SecondPool:
        """Read the exact prepared minimal pool without inventing molecular metadata.

        Args:
          directory: Complete prepared engine2 pool directory.

        Returns:
          pool: Original aligned rows and readonly numeric arrays.

        """
        metadata = from_plain(
            parquet.read_table(
                directory / "pool_meta.parquet",
                columns=["key", "smiles"],
            ).to_pylist(),
            list[PoolIdentity],
        )
        mass = _load_array(directory / "pool_mass.npy", dtype=np.float64, dimensions=1)
        packed = _load_array(directory / "pool_fp.npy", dtype=np.uint8, dimensions=2)
        bits = _load_array(directory / "fp_bits.npy", dtype=np.int64, dimensions=1)
        if len(metadata) != len(mass) or len(packed) != len(mass):
            raise ValueError("Engine2 pool rows must be aligned.")
        return cls(
            mass=mass,
            keys=tuple(row["key"] for row in metadata),
            smiles=tuple(row["smiles"] for row in metadata),
            packed=packed,
            bits=bits,
        )

    def fingerprints(self, rows: NDArray[np.int64]) -> NDArray[np.uint8]:
        """Expand the source packed rows with its fixed6930-bit maximum."""
        return np.unpackbits(np.asarray(self.packed[rows]), axis=1)[:, :6930]

    def window(self, target: float, *, ppm: float) -> NDArray[np.int64]:
        """Preserve stable pool positions inside the source multiplicative mass bounds."""
        lower = np.searchsorted(self.mass, target * (1 - ppm / 1e6))
        upper = np.searchsorted(self.mass, target * (1 + ppm / 1e6), side="right")
        return np.arange(lower, upper)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SecondBase:
    """Retain all controller evidence even when the finite target has no candidates."""

    target: float | None
    rows: PoolIndices
    spectra: tuple[SpectrumRequest, ...]
    channels: SecondChannels | None
    fragments: SecondFragments | None
    logits: NDArray[np.float32] | None
    own_features: CandidateMatrix
    pv_features: CandidateMatrix


class SecondEngine:
    """Compute every numerical stage of both second-engine ranking arms."""

    class Config(Fig["SecondEngine"]):
        retrieval: SecondRetrieval.Config = field(
            default_factory=SecondRetrieval.Config,
        )
        """Complete raw-library and polarity representative search recipe."""

        ppm: float = 10.0
        """Initial source multiplicative candidate mass half-window."""

        fallback_ppm: float = 30.0
        """Half-window used only when the initial candidate list is empty."""

        precursor_margin: float = 2.0
        """Mass above the precursor retained when preparing query spectra."""

        default_energy: float = 25.0
        """Collision energy of a query with an empty or unparseable energy list."""

        fragment_search: SecondSearch.Config = field(
            default_factory=lambda: SecondSearch.Config(entropy_weight=False),
        )
        """Peak preparation of the fragment-explanation panels, unweighted by entropy."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.retrieval = config.retrieval.make()
        self.fragment_search = config.fragment_search.make()

    def base(
        self,
        spectra: Sequence[SpectrumRequest],
        *,
        pool: SecondPool,
        library: SecondLibrary,
        index: SecondIndex,
        fragments: FragmentMassStore,
        bank: SecondBank,
    ) -> SecondBase:
        """Compute the complete own51 and predicted-fingerprint31 feature matrices.

        Args:
          spectra: Every raw query spectrum and its original conditioning metadata.
          pool: Independently prepared second-engine pool.
          library: Every original labelled raw library spectrum.
          index: All mass-sorted and separately cleaned polarity representatives.
          fragments: Complete independent unrounded fragment rows in pool order.
          bank: Every released single and merged fingerprint model.

        Returns:
          base: All controller evidence, including finite targets with no candidates.

        """
        prepared = prepare_queries(spectra, margin=self.config.precursor_margin)
        masses = second_neutral_mass(
            np.array([spectrum["prec"] for spectrum in spectra], np.float64),
            [spectrum["adduct"] for spectrum in spectra],
        )
        masses = masses[np.isfinite(masses)]
        target = float(np.median(masses)) if len(masses) else None
        rows = np.zeros(0, np.int64)
        channels = fragment_values = logits = None
        own, pv = np.zeros((0, 51), np.float32), np.zeros((0, 31), np.float32)
        if target is not None:
            channels = self.retrieval.channels(
                prepared,
                library=library,
                index=index,
                target=target,
            )
            rows = pool.window(target, ppm=self.config.ppm)
            if not len(rows):
                rows = pool.window(target, ppm=self.config.fallback_ppm)
            logits = bank.predict(
                [
                    QuerySpectrum(
                        mass=np.asarray(spectrum["mz"]),
                        intensity=np.asarray(spectrum["it"]),
                        precursor=spectrum["prec"],
                        adduct=spectrum["adduct"],
                        instrument=str(spectrum.get("instrument")),
                        collision_energy=collision_energy(
                            spectrum.get("ce_ev"),
                            default=self.config.default_energy,
                        ),
                        polarity=1.0 if spectrum["mode"] in (1, "positive") else -1.0,
                    )
                    for spectrum in prepared
                ],
            )
            if len(rows):
                fragment_values = second_fragments(
                    prepared,
                    rows=rows,
                    fragments=fragments,
                    search=self.fragment_search,
                )
                own = second_features(
                    pool,
                    rows=rows,
                    channels=channels,
                    fragments=fragment_values,
                    logits=None,
                )
                pv = second_features(
                    pool,
                    rows=rows,
                    channels=channels,
                    fragments=fragment_values,
                    logits=logits,
                )[:, :31]
        return SecondBase(
            target=target,
            rows=rows,
            spectra=prepared,
            channels=channels,
            fragments=fragment_values,
            logits=logits,
            own_features=own,
            pv_features=pv,
        )

    def run(
        self,
        spectra: Sequence[SpectrumRequest],
        *,
        pool: SecondPool,
        library: SecondLibrary,
        index: SecondIndex,
        fragments: FragmentMassStore,
        bank: SecondBank,
        rankers: SecondRankerBank,
        formulas: Sequence[str],
    ) -> EngineResult:
        """Return all82 features and the actual four-own/eight-PV blended ranking.

        Args:
          spectra: Every raw query spectrum and its original conditioning metadata.
          pool: Complete second-engine pool, retaining duplicate positions.
          library: Original raw labelled spectra.
          index: Complete separate mass and polarity representative stores.
          fragments: Independent unrounded engine2 fragment rows.
          bank: Every single and merged fingerprint checkpoint.
          rankers: Both fully fitted source ensembles and their archive block order.
          formulas: Explicit prepared formula join aligned to every pool row.

        Returns:
          result: Every candidate identity, own51/PV31 feature and blended score.

        """
        if len(formulas) != len(pool.mass):
            raise ValueError("Engine2 formulas require an explicit aligned pool join.")
        blocks = rankers.blocks
        bounds = {
            "base": (0, 31),
            "libx": (31, 39),
            "anapol": (39, 44),
            "fragx": (44, 51),
        }
        if len(blocks) != len(bounds) or set(blocks) != set(bounds):
            raise ValueError(
                "Engine2 own rankers require all four complete feature blocks.",
            )
        base = self.base(
            spectra,
            pool=pool,
            library=library,
            index=index,
            fragments=fragments,
            bank=bank,
        )
        columns = [column for block in blocks for column in range(*bounds[block])]
        own = base.own_features[:, columns]
        names = tuple(
            f"own__{block}_{column:02d}"
            for block in blocks
            for column in range(bounds[block][1] - bounds[block][0])
        ) + tuple(f"pv__base_{column:02d}" for column in range(31))
        count = len(base.rows)
        scores = (
            rankers.score(own, pv_features=base.pv_features).blend
            if count
            else np.zeros(0, np.float64)
        )
        channels = base.channels
        return EngineResult(
            candidates=CandidateSet(
                smiles=tuple(pool.smiles[base.rows.item(i)] for i in range(count)),
                keys=tuple(pool.keys[base.rows.item(i)] for i in range(count)),
                formulas=tuple(formulas[base.rows.item(i)] for i in range(count)),
                pool_rows=base.rows,
                generated=np.zeros(count, np.bool_),
                sources=("pool",) * count,
            ),
            features=np.hstack([own, base.pv_features]),
            feature_names=names,
            scores=scores,
            diagnostics=EngineDiagnostics(
                n_cand=count,
                lib_max=base.own_features[:, 0].max().item() if count else 0.0,
                top_sim=channels.analogs[0].similarity
                if channels is not None and channels.analogs
                else 0.0,
                fzmax=float(
                    np.float32(
                        (
                            pool.fingerprints(base.rows)
                            @ base.logits.astype(np.float64)
                        ).max(),
                    ),
                )
                if count and base.logits is not None
                else 0.0,
            ),
        )


def prepare_queries(
    spectra: Sequence[SpectrumRequest],
    *,
    margin: float,
) -> tuple[SpectrumRequest, ...]:
    """Apply the source F32 precursor cut and basepeak normalization in row order.

    Args:
      spectra: Complete original molecule query spectra.
      margin: Mass above each precursor that the cut still retains.

    Returns:
      prepared: Every original query row, including empty and nonpositive spectra.

    """
    output: list[SpectrumRequest] = []
    for spectrum in spectra:
        mass = np.asarray(spectrum["mz"], np.float32)
        intensity = np.asarray(spectrum["it"], np.float32)
        keep = mass <= spectrum["prec"] + margin
        mass, intensity = mass[keep], intensity[keep]
        if len(intensity) and intensity.max() > 0:
            intensity = intensity / intensity.max()
        prepared = spectrum.copy()
        prepared["mz"] = [mass.item(i) for i in range(len(mass))]
        prepared["it"] = [intensity.item(i) for i in range(len(intensity))]
        output.append(prepared)
    return tuple(output)


def collision_energy(
    value: float | str | Sequence[float] | None,
    *,
    default: float,
) -> float:
    """Retain source absolute-mean energy parsing and its empty/error defaults.

    Args:
      value: Original scalar/list conditioning value, retaining null and invalid text.
      default: Energy of an empty or unparseable list.

    Returns:
      energy: Absolute mean, NaN for null, or the default for empty/unparseable input.

    """
    try:
        values = np.abs(np.atleast_1d(np.asarray(value, float)))
    except (TypeError, ValueError):
        return default
    return float(values.mean()) if len(values) else default


def _load_array[S: np.generic](
    path: Path,
    *,
    dtype: type[S],
    dimensions: int,
) -> NDArray[S]:
    value = load_array(path, dtype, mmap=True)
    if value.ndim != dimensions:
        raise ValueError(f"Engine2 array {path.name} has an invalid rank.")
    return value


type FeatureMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type BinMatrix = np.ndarray[tuple[int, int], np.dtype[np.uint8]]


@dataclass(slots=True, kw_only=True)
class Histogram:
    """Per-feature bin sums in double precision and exact sample counts."""

    gradient: FeatureMatrix
    hessian: FeatureMatrix
    count: np.ndarray[tuple[int, int], np.dtype[np.uint32]]


@dataclass(slots=True, kw_only=True)
class Split:
    """A proposed numeric split and its complete Newton statistics."""

    gain: float
    feature: int
    threshold: int
    missing_left: bool
    left_gradient: float
    left_hessian: float
    right_gradient: float
    right_hessian: float
    left_value: float
    right_value: float


@dataclass(slots=True, kw_only=True, eq=False)
class Node:
    """Mutable training node, ordered only by its proposed gain."""

    samples: NDArray[np.uint32]
    depth: int
    gradient: float
    hessian: float
    value: float
    histogram: Histogram | None = None
    split: Split | None = None
    left: Node | None = None
    right: Node | None = None
    gain: float = -1

    def __lt__(self, other: Node) -> bool:
        return self.gain > other.gain


@dataclass(frozen=True, slots=True, kw_only=True)
class NumericTree:
    """Complete fitted numeric predictor with a typed column for every state field."""

    value: NDArray[np.float64]
    count: NDArray[np.uint32]
    feature_idx: NDArray[np.int64]
    num_threshold: NDArray[np.float64]
    missing_go_to_left: NDArray[np.uint8]
    left: NDArray[np.uint32]
    right: NDArray[np.uint32]
    gain: NDArray[np.float64]
    depth: NDArray[np.uint32]
    is_leaf: NDArray[np.uint8]
    bin_threshold: NDArray[np.uint8]
    is_categorical: NDArray[np.uint8]
    bitset_idx: NDArray[np.uint32]
    binned_left_cat_bitsets: NDArray[np.uint32]
    raw_left_cat_bitsets: NDArray[np.uint32]

    def state(self) -> dict[str, NDArray[np.generic]]:
        """Return every fitted column, preserving its complete values and dtype.

        Returns:
          state: Complete fitted numeric and categorical predictor columns.

        """
        return {
            "value": self.value,
            "count": self.count,
            "feature_idx": self.feature_idx,
            "num_threshold": self.num_threshold,
            "missing_go_to_left": self.missing_go_to_left,
            "left": self.left,
            "right": self.right,
            "gain": self.gain,
            "depth": self.depth,
            "is_leaf": self.is_leaf,
            "bin_threshold": self.bin_threshold,
            "is_categorical": self.is_categorical,
            "bitset_idx": self.bitset_idx,
            "binned_left_cat_bitsets": self.binned_left_cat_bitsets,
            "raw_left_cat_bitsets": self.raw_left_cat_bitsets,
        }


class HistogramBoosting:
    """Fit numeric-feature Newton trees with explicit bin and missing-value state."""

    class Config(Fig["HistogramBoosting"]):
        max_depth: int = 6
        """Largest tree depth."""

        rounds: int = 500
        """Largest number of boosting updates."""

        learning_rate: float = 0.03
        """Shrinkage applied only to terminal tree values."""

        min_leaf: int = 80
        """Smallest admitted child sample count."""

        l2: float = 1.0
        """Leaf Hessian regularization."""

        max_bins: int = 255
        """Numeric bins, excluding the separate missing-value bin."""

        max_leaves: int = 31
        """Largest terminal-node count per tree."""

        seed: int = -1
        """Initialization of validation and feature-sampling random streams; required."""

        early_stopping_rows: int = 10_000
        """Enable validation when the input has more rows than this capacity."""

        binning_rows: int = 200_000
        """Weighted replacement sample size for fitting large bin vocabularies."""

        validation_fraction: float = 0.1
        """Stratified held-out fraction when automatic stopping is active."""

        patience: int = 10
        """Updates without improvement before validation stops fitting."""

        tolerance: float = 1e-7
        """Required improvement in the negative validation loss."""

    def __init__(self, config: Config) -> None:
        if config.seed < 0:
            raise ValueError("HistogramBoosting requires a nonnegative seed.")
        self.config = config
        self.baseline = np.zeros((1, 1), dtype=np.float64)
        random = np.random.RandomState(config.seed)
        self.random_seed = int(random.randint(np.iinfo(np.uint32).max, dtype="u8"))
        feature_seed = random.randint(np.iinfo(np.uint32).max, dtype="u8")
        self.generator = np.random.default_rng(feature_seed)
        self.thresholds: list[NDArray[np.float64]] = []
        self.bin_counts: NDArray[np.uint32] = np.zeros(0, dtype=np.uint32)
        self.trees: list[NumericTree] = []
        self.train_score: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self.validation_score: NDArray[np.float64] = np.zeros(0, dtype=np.float64)

    def fit(
        self,
        data: FeatureMatrix,
        labels: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> Self:
        """Fit the complete weighted binary ranker on numerical feature rows.

        Args:
          data: Numerical training feature matrix.
          labels: Binary targets in row order.
          weights: Per-row simulation-prior weights.

        Returns:
          fitted: This estimator with every fitted bin and predictor.

        """
        early = len(data) > self.config.early_stopping_rows
        validation = np.empty((0, data.shape[1]), dtype=np.float64)
        targets = validation_weights = np.empty(0, dtype=np.float64)
        if early:
            train, valid = self.validation_indices(labels)
            validation, targets, validation_weights = (
                data[valid, :],
                labels[valid],
                weights[valid],
            )
            selected = np.empty((len(train), data.shape[1]), dtype=np.float64)
            selected[:] = data[train, :]
            data, labels, weights = selected, labels[train], weights[train]
        bin_data = data
        bin_weights: NDArray[np.float64] | None = weights
        if len(data) > self.config.binning_rows:
            random = np.random.RandomState(self.random_seed)
            subset = random.choice(
                len(data),
                self.config.binning_rows,
                p=weights / np.sum(weights),
                replace=True,
            )
            bin_data = np.empty((len(subset), data.shape[1]), dtype=np.float64)
            bin_data[:] = data[subset, :]
            bin_weights = None
        self.thresholds = [
            self.numeric_thresholds(bin_data[:, column], bin_weights)
            for column in range(data.shape[1])
        ]
        self.bin_counts = np.array(
            [len(values) + 1 for values in self.thresholds],
            dtype=np.uint32,
        )
        epsilon = 10 * np.finfo(np.float64).eps
        probability = float(
            np.clip(np.average(labels, weights=weights), epsilon, 1 - epsilon),
        )
        self.baseline[:] = math.log(probability) - math.log1p(-probability)
        raw = np.full(len(data), self.baseline.item(), dtype=np.float64)
        binned = self.binned(data)
        missing = np.any(np.equal(binned, self.config.max_bins), axis=0)
        self.trees = []
        validation_raw = np.full(
            len(validation),
            self.baseline.item(),
            dtype=np.float64,
        )
        train_scores: list[float] = []
        validation_scores: list[float] = []
        if early:
            train_scores.append(self.loss_score(raw, labels, weights))
            validation_scores.append(
                self.loss_score(validation_raw, targets, validation_weights),
            )
        for _ in range(self.config.rounds):
            gradient, hessian = self.derivatives(raw, labels, weights)
            root = self.grow(binned, gradient, hessian, missing)
            tree = self.export(root)
            self.trees.append(tree)
            raw += self.tree_prediction(tree, data)
            if early:
                validation_raw += self.tree_prediction(tree, validation)
                train_scores.append(self.loss_score(raw, labels, weights))
                validation_scores.append(
                    self.loss_score(validation_raw, targets, validation_weights),
                )
                if self.should_stop(validation_scores):
                    break
        self.train_score = np.array(train_scores, dtype=np.float64)
        self.validation_score = np.array(validation_scores, dtype=np.float64)
        return self

    def validation_indices(
        self,
        labels: NDArray[np.float64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Allocate class-wise counts and permute a stratified held-out split.

        Args:
          labels: Binary targets in input row order.

        Returns:
          indices: Training and held-out row indices in their shuffled order.

        """
        classes = np.unique(labels)
        groups = [
            np.flatnonzero(np.equal(labels, classes.item(index)))
            for index in range(len(classes))
        ]
        counts = np.array([len(group) for group in groups], dtype=np.int64)
        n_valid = math.ceil(len(labels) * self.config.validation_fraction)
        n_train = len(labels) - n_valid
        if counts.min() < 2 or min(n_train, n_valid) < len(groups):
            raise ValueError("Each class needs two rows and a place in both splits.")
        random = np.random.RandomState(self.random_seed)
        allocated = self.class_allocation(counts, n_train, random)
        held = self.class_allocation(counts - allocated, n_valid, random)
        train: list[int] = []
        valid: list[int] = []
        for index, group in enumerate(groups):
            order = group.take(random.permutation(len(group)), mode="clip")
            split = allocated.item(index)
            train.extend(order.item(row) for row in range(split))
            valid.extend(
                order.item(row) for row in range(split, split + held.item(index))
            )
        return random.permutation(np.array(train, dtype=np.int64)), random.permutation(
            np.array(valid, dtype=np.int64),
        )

    @classmethod
    def class_allocation(
        cls,
        counts: NDArray[np.int64],
        draws: int,
        random: np.random.RandomState,
    ) -> NDArray[np.int64]:
        """Round expected class counts, breaking equal fractional remainders randomly.

        Args:
          counts: Available row count in each sorted class.
          draws: Total allocated rows.
          random: Split-specific random stream.

        Returns:
          allocation: Integer allocated count in each class.

        """
        expected = counts / counts.sum() * draws
        output: NDArray[np.float64] = np.floor(expected)
        remaining = draws - sum(int(output.item(index)) for index in range(len(output)))
        remainder = expected - output
        if remaining:
            positions = np.sort(np.unique(remainder))[::-1]
            for index in range(len(positions)):
                tied = np.flatnonzero(np.equal(remainder, positions.item(index)))
                extra = min(len(tied), remaining)
                chosen = random.choice(tied, size=extra, replace=False)
                output[chosen] += 1
                remaining -= extra
                if not remaining:
                    break
        return output.astype(np.int64)

    @classmethod
    def derivatives(
        cls,
        raw: NDArray[np.float64],
        labels: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Compute stable weighted binary derivatives before their float32 store.

        Args:
          raw: Accumulated raw binary logits.
          labels: Binary targets.
          weights: Per-row simulation-prior weights.

        Returns:
          derivatives: Weighted gradient and Hessian vectors.

        """
        gradient = np.empty(len(raw), dtype=np.float32)
        hessian = np.empty(len(raw), dtype=np.float32)
        for row in range(len(raw)):
            value, target, weight = raw.item(row), labels.item(row), weights.item(row)
            if value > -37:
                exponential = math.exp(-value)
                gradient[row] = (
                    ((1 - target) - target * exponential) / (1 + exponential)
                ) * weight
                hessian[row] = (exponential / (1 + exponential) ** 2) * weight
            else:
                exponential = math.exp(value)
                gradient[row] = (exponential - target) * weight
                hessian[row] = exponential * weight
        return gradient, hessian

    @classmethod
    def loss_score(
        cls,
        raw: NDArray[np.float64],
        labels: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Return the weighted negative binary loss with stable exponential tails.

        Args:
          raw: Accumulated raw binary logits.
          labels: Binary targets.
          weights: Per-row simulation-prior weights.

        Returns:
          score: Negative weighted mean of the unweighted binary row losses.

        """
        loss = np.empty(len(raw), dtype=np.float64)
        for row in range(len(raw)):
            value = raw.item(row)
            if value <= -37:
                normalizer = math.exp(value)
            elif value <= -2:
                normalizer = math.log1p(math.exp(value))
            elif value <= 18:
                normalizer = math.log(1 + math.exp(value))
            elif value <= 33.3:
                normalizer = value + math.exp(-value)
            else:
                normalizer = value
            loss[row] = normalizer - labels.item(row) * value
        return -float(np.average(loss, weights=weights))

    def should_stop(self, scores: list[float]) -> bool:
        """Stop after a full patience window has no significant improvement."""
        if len(scores) <= self.config.patience:
            return False
        reference = scores[-self.config.patience - 1] + self.config.tolerance
        return not any(value > reference for value in scores[-self.config.patience :])

    def numeric_thresholds(
        self,
        values: NDArray[np.float64],
        weights: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Choose weighted empirical quantiles, retaining exact-boundary averaging.

        Args:
          values: One numeric feature, with optional missing entries.
          weights: Per-row simulation-prior weights.

        Returns:
          thresholds: Distinct sorted split boundaries.

        """
        keep = ~np.isnan(values)
        if weights is not None:
            keep &= np.not_equal(weights, 0)
        order = np.argsort(values[keep], kind="stable")
        ordered = values[keep][order]
        distinct = np.unique(ordered)
        if len(distinct) <= self.config.max_bins:
            thresholds = (distinct[:-1] + distinct[1:]) / 2
        elif weights is None:
            thresholds = np.percentile(
                ordered,
                np.linspace(0, 100, self.config.max_bins + 1)[1:-1],
                method="averaged_inverted_cdf",
            )
        else:
            cdf = np.cumsum(weights[keep][order])
            thresholds = np.empty(self.config.max_bins - 1, dtype=np.float64)
            percentiles = np.linspace(0, 100, self.config.max_bins + 1)[1:-1]
            for index in range(len(percentiles)):
                position = percentiles.item(index) / 100 * cdf.item(-1)
                lower = min(int(np.searchsorted(cdf, position)), len(cdf) - 1)
                if cdf.item(lower) - position > np.finfo(np.float64).eps:
                    thresholds[index] = ordered.item(lower)
                else:
                    thresholds[index] = (
                        ordered.item(lower) + ordered.item(min(lower + 1, len(cdf) - 1))
                    ) / 2
            thresholds = np.unique(thresholds)
        return np.clip(thresholds, None, 1e300)

    def binned(self, data: FeatureMatrix) -> BinMatrix:
        """Map numerical rows into the fitted uint8 bin vocabulary.

        Args:
          data: Numeric training or prediction rows.

        Returns:
          binned: Numeric bin indices with the separate missing-value token.

        """
        output = np.zeros(data.shape, dtype=np.uint8, order="F")
        for feature, thresholds in enumerate(self.thresholds):
            output[:, feature] = np.searchsorted(
                thresholds,
                data[:, feature],
                side="left",
            )
        output[np.isnan(data)] = self.config.max_bins
        return output

    def histogram(
        self,
        data: BinMatrix,
        samples: NDArray[np.uint32],
        gradient: NDArray[np.float32],
        hessian: NDArray[np.float32],
    ) -> Histogram:
        """Accumulate float32 derivatives into double bin sums in sample order.

        Args:
          data: Binned training features.
          samples: Ordered row indices belonging to the node.
          gradient: Complete float32 gradient vector.
          hessian: Complete float32 Hessian vector.

        Returns:
          histogram: Per-feature bin derivatives and sample counts.

        """
        shape = (data.shape[1], self.config.max_bins + 1)
        result = Histogram(
            gradient=np.zeros(shape, dtype=np.float64),
            hessian=np.zeros(shape, dtype=np.float64),
            count=np.zeros(shape, dtype=np.uint32),
        )
        for feature in range(data.shape[1]):
            bins = data[:, feature][samples]
            np.add.at(result.gradient[feature, :], bins, gradient[samples])
            np.add.at(result.hessian[feature, :], bins, hessian[samples])
            np.add.at(result.count[feature, :], bins, 1)
        return result

    def best_split(self, node: Node, missing: NDArray[np.bool_]) -> Split | None:
        """Scan numeric bins in both missing-value directions, retaining first ties.

        Args:
          node: Current node and its cached bin histogram.
          missing: Which features contain missing values in training.

        Returns:
          split: Best feasible split, or no split when every gain is nonpositive.

        """
        histogram = node.histogram
        if histogram is None:
            raise ValueError("Expected histogram is not None.")
        best: Split | None = None
        for feature in range(len(self.bin_counts)):
            bin_count = self.bin_counts.item(feature)
            feature_best: Split | None = None
            for reverse in (False, True):
                if reverse and not missing[feature]:
                    continue
                count = 0
                gradient = hessian = 0.0
                bins = (
                    range(bin_count - 2, -1, -1)
                    if reverse
                    else range(bin_count - 1 + int(missing.item(feature)))
                )
                for threshold in bins:
                    index = threshold + int(reverse)
                    count += histogram.count.item(feature, index)
                    gradient += histogram.gradient.item(feature, index)
                    hessian += histogram.hessian.item(feature, index)
                    complement = len(node.samples) - count
                    if count < self.config.min_leaf:
                        continue
                    if complement < self.config.min_leaf:
                        break
                    if hessian < 1e-3:
                        continue
                    other_hessian = node.hessian - hessian
                    if other_hessian < 1e-3:
                        break
                    other_gradient = node.gradient - gradient
                    lg, lh, rg, rh = (
                        (other_gradient, other_hessian, gradient, hessian)
                        if reverse
                        else (gradient, hessian, other_gradient, other_hessian)
                    )
                    left = -lg / (lh + self.config.l2 + 1e-15)
                    right = -rg / (rh + self.config.l2 + 1e-15)
                    gain = node.gradient * node.value
                    gain -= lg * left
                    gain -= rg * right
                    previous = -1.0 if feature_best is None else feature_best.gain
                    if gain > previous and gain > 0:
                        # Store the final complement subtraction used by the
                        # accepted split, including right-to-left cancellation.
                        rg = node.gradient - lg
                        rh = node.hessian - lh
                        feature_best = Split(
                            gain=gain,
                            feature=feature,
                            threshold=threshold,
                            missing_left=reverse,
                            left_gradient=lg,
                            left_hessian=lh,
                            right_gradient=rg,
                            right_hessian=rh,
                            left_value=-lg / (lh + self.config.l2 + 1e-15),
                            right_value=-rg / (rh + self.config.l2 + 1e-15),
                        )
            if feature_best is not None and (
                best is None or feature_best.gain > best.gain
            ):
                best = feature_best
        return best

    def grow(
        self,
        data: BinMatrix,
        gradient: NDArray[np.float32],
        hessian: NDArray[np.float32],
        missing: NDArray[np.bool_],
    ) -> Node:
        """Grow a best-first Newton tree using smaller-child histogram subtraction.

        Args:
          data: Binned training features.
          gradient: Complete float32 gradient vector.
          hessian: Complete float32 Hessian vector.
          missing: Which features contain missing values in training.

        Returns:
          root: Complete tree with unshrunk Newton values.

        """
        samples = np.arange(len(data), dtype=np.uint32)
        histogram = self.histogram(data, samples, gradient, hessian)
        root = Node(
            samples=samples,
            depth=0,
            gradient=float(histogram.gradient[0, :].sum()),
            hessian=float(histogram.hessian[0, :].sum()),
            value=0,
            histogram=histogram,
        )
        pending: list[Node] = []
        self.enqueue(root, pending, missing)
        leaves = 1
        while pending:
            node = heapq.heappop(pending)
            split = node.split
            if split is None:
                raise ValueError("Expected split is not None.")
            bins = data[node.samples][:, split.feature]
            left_mask = (bins <= split.threshold) | (
                np.equal(bins, self.config.max_bins) & split.missing_left
            )
            left = node.left = Node(
                samples=node.samples[left_mask],
                depth=node.depth + 1,
                gradient=split.left_gradient,
                hessian=split.left_hessian,
                value=split.left_value,
            )
            right = node.right = Node(
                samples=node.samples[~left_mask],
                depth=node.depth + 1,
                gradient=split.right_gradient,
                hessian=split.right_hessian,
                value=split.right_value,
            )
            if not missing[split.feature]:
                split.missing_left = len(left.samples) > len(right.samples)
            leaves += 1
            if leaves == self.config.max_leaves:
                break
            if left.depth == self.config.max_depth:
                continue
            if (
                len(left.samples) >= 2 * self.config.min_leaf
                or len(right.samples) >= 2 * self.config.min_leaf
            ):
                small, large = (
                    (left, right)
                    if len(left.samples) < len(right.samples)
                    else (right, left)
                )
                small.histogram = self.histogram(data, small.samples, gradient, hessian)
                parent = node.histogram
                if parent is None:
                    raise ValueError("Expected parent is not None.")
                large.histogram = Histogram(
                    gradient=parent.gradient - small.histogram.gradient,
                    hessian=parent.hessian - small.histogram.hessian,
                    count=parent.count - small.histogram.count,
                )
                for child in (left, right):
                    self.enqueue(child, pending, missing)
            node.histogram = None
        return root

    def enqueue(
        self,
        node: Node,
        pending: list[Node],
        missing: NDArray[np.bool_],
    ) -> None:
        """Queue a feasible split while preserving unsplit root and leaf values.

        Args:
          node: Candidate training node.
          pending: Best-gain priority queue.
          missing: Which features contain missing values in training.

        """
        if len(node.samples) < 2 * self.config.min_leaf or node.hessian < 1e-3:
            return
        node.split = self.best_split(node, missing)
        if node.split is not None:
            node.gain = node.split.gain
            heapq.heappush(pending, node)

    def export(self, root: Node) -> NumericTree:
        """Return every predictor column in depth-first order with native dtypes.

        Args:
          root: Complete tree with unshrunk Newton values.

        Returns:
          state: Every predictor column and both empty categorical tables.

        """
        ordered: list[Node] = []
        self.flatten(root, ordered)
        indices = {node: index for index, node in enumerate(ordered)}
        records: dict[str, list[int | float]] = {
            name: []
            for name in (
                "value",
                "count",
                "feature_idx",
                "num_threshold",
                "missing_go_to_left",
                "left",
                "right",
                "gain",
                "depth",
                "is_leaf",
                "bin_threshold",
                "is_categorical",
                "bitset_idx",
            )
        }
        for node in ordered:
            split = node.split
            leaf = node.left is None
            records["value"].append(
                node.value * self.config.learning_rate if leaf else node.value,
            )
            records["count"].append(len(node.samples))
            records["gain"].append(node.gain)
            records["depth"].append(node.depth)
            records["is_leaf"].append(int(leaf))
            if leaf:
                for name in (
                    "feature_idx",
                    "num_threshold",
                    "missing_go_to_left",
                    "left",
                    "right",
                    "bin_threshold",
                    "is_categorical",
                    "bitset_idx",
                ):
                    records[name].append(0)
            else:
                if split is None or node.left is None or node.right is None:
                    raise ValueError(
                        "Expected split is not None and node.left is not None and node.right is not None.",
                    )
                records["feature_idx"].append(split.feature)
                threshold = (
                    float("inf")
                    if split.threshold == self.bin_counts.item(split.feature) - 1
                    else self.thresholds[split.feature].item(split.threshold)
                )
                records["num_threshold"].append(threshold)
                records["missing_go_to_left"].append(int(split.missing_left))
                records["left"].append(indices[node.left])
                records["right"].append(indices[node.right])
                records["bin_threshold"].append(split.threshold)
                records["is_categorical"].append(0)
                records["bitset_idx"].append(0)
        return NumericTree(
            value=np.array(records["value"], dtype=np.float64),
            count=np.array(records["count"], dtype=np.uint32),
            feature_idx=np.array(records["feature_idx"], dtype=np.int64),
            num_threshold=np.array(records["num_threshold"], dtype=np.float64),
            missing_go_to_left=np.array(records["missing_go_to_left"], dtype=np.uint8),
            left=np.array(records["left"], dtype=np.uint32),
            right=np.array(records["right"], dtype=np.uint32),
            gain=np.array(records["gain"], dtype=np.float64),
            depth=np.array(records["depth"], dtype=np.uint32),
            is_leaf=np.array(records["is_leaf"], dtype=np.uint8),
            bin_threshold=np.array(records["bin_threshold"], dtype=np.uint8),
            is_categorical=np.array(records["is_categorical"], dtype=np.uint8),
            bitset_idx=np.array(records["bitset_idx"], dtype=np.uint32),
            binned_left_cat_bitsets=np.zeros((0, 8), dtype=np.uint32),
            raw_left_cat_bitsets=np.zeros((0, 8), dtype=np.uint32),
        )

    @classmethod
    def flatten(cls, node: Node, output: list[Node]) -> None:
        """Visit the left subtree before the right subtree for checkpoint rows."""
        output.append(node)
        if node.left is not None and node.right is not None:
            cls.flatten(node.left, output)
            cls.flatten(node.right, output)

    @classmethod
    def tree_prediction(
        cls,
        tree: NumericTree,
        data: PredictionMatrix,
    ) -> NDArray[np.float64]:
        """Traverse numeric branches with the fitted missing-value directions.

        Args:
          tree: Complete fitted predictor columns.
          data: Numeric prediction rows.

        Returns:
          values: One shrunk terminal value per row.

        """
        positions = np.zeros(len(data), dtype=np.int64)
        while bool(np.any(np.equal(tree.is_leaf[positions], 0))):
            active = np.equal(tree.is_leaf[positions], 0)
            nodes = positions[active]
            values = data[np.nonzero(active)[0], tree.feature_idx[nodes]]
            turn_left = np.where(
                np.isnan(values),
                tree.missing_go_to_left[nodes],
                values <= tree.num_threshold[nodes],
            )
            positions[active] = np.where(turn_left, tree.left[nodes], tree.right[nodes])
        return tree.value[positions]

    def stages(self, data: PredictionMatrix) -> NDArray[np.float64]:
        """Return raw predictions after every fitted boosting update.

        Args:
          data: Numeric prediction rows.

        Returns:
          stages: Raw accumulated predictions after every fitted round.

        """
        raw = np.full(len(data), self.baseline.item(), dtype=np.float64)
        output: list[NDArray[np.float64]] = []
        for tree in self.trees:
            raw += self.tree_prediction(tree, data)
            output.append(raw.copy())
        return np.stack(output)

    def predict_proba(self, data: PredictionMatrix) -> NDArray[np.float64]:
        """Return negative/positive class probabilities in that order."""
        probability = np.array(
            [_sigmoid(value) for value in self.stages(data)[-1, :].tolist()],
            dtype=np.float64,
        )
        return np.column_stack([1 - probability, probability])


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationRows:
    """Aligned numeric training rows and their declared source feature-block order."""

    data: FeatureMatrix
    labels: NDArray[np.float64]
    groups: NDArray[np.float64]
    blocks: tuple[str, ...]

    @classmethod
    def load(cls, path: Path, group_column: str) -> Self:
        """Read public numerical NPZ arrays without allowing pickle objects.

        Args:
          path: Public rank-training archive.
          group_column: Simulation class column, S for own rows or M for fingerprint.

        Returns:
          rows: Aligned double feature, target and simulation-class arrays.

        """
        with cast(
            "np.lib.npyio.NpzFile[np.generic]",
            np.load(path),
        ) as arrays:
            data = np.asarray(arrays["X"], dtype=np.float64)
            labels = np.asarray(arrays["Y"], dtype=np.float64)
            groups = np.asarray(arrays[group_column], dtype=np.float64)
            blocks: tuple[str, ...] = ()
            if "blocks" in arrays:
                names = arrays["blocks"].astype(np.str_)
                blocks = tuple(names.item(index) for index in range(len(names)))
        if (
            data.ndim != 2
            or labels.shape != (len(data),)
            or groups.shape != (len(data),)
        ):
            raise ValueError(
                "Simulation features, labels and groups need aligned rows.",
            )
        return cls(
            data=cast(FeatureMatrix, data),
            labels=labels,
            groups=groups,
            blocks=blocks,
        )


class HistogramEnsemble:
    """Fit the public Cartesian product of simulation priors and random seeds."""

    class Config(Fig["HistogramEnsemble"]):
        priors: list[float] = field(default_factory=list[float])
        """Class-one simulation weights, traversed before seeds."""

        seeds: list[int] = field(default_factory=list[int])
        """Source ranker seed values in averaging order."""

        tree: HistogramBoosting.Config = field(default_factory=HistogramBoosting.Config)
        """Complete histogram-fitting geometry and stopping policy."""

    def __init__(self, config: Config) -> None:
        if not config.priors or not config.seeds:
            raise ValueError("Histogram ensemble requires priors and seeds.")
        self.config = config
        self.models: list[HistogramBoosting] = []

    def fit(self, rows: SimulationRows) -> Self:
        """Fit all selected prior/seed combinations on the declared simulation rows.

        Args:
          rows: Numeric features, labels and simulation-class flags.

        Returns:
          ensemble: This fitted ensemble in source model order.

        """
        self.models = []
        for prior in self.config.priors:
            weights = np.where(np.equal(rows.groups, 0), prior, 1.0 - prior)
            for seed in self.config.seeds:
                config = HistogramBoosting.Config()
                _ = config.update(self.config.tree)
                config.seed = seed
                self.models.append(config.make().fit(rows.data, rows.labels, weights))
        return self

    def predict(self, data: PredictionMatrix) -> NDArray[np.float64]:
        """Average positive-class probabilities in fitted model order.

        Args:
          data: Complete numeric inference features.

        Returns:
          probabilities: One averaged positive-class probability per candidate.

        """
        if not self.models:
            raise ValueError("Histogram ensemble must be fitted before prediction.")
        return np.mean(
            [model.predict_proba(data)[:, 1] for model in self.models],
            axis=0,
        )


def blend_histograms(
    fingerprint: NDArray[np.float64],
    own: NDArray[np.float64],
    *,
    weight_fingerprint: float,
) -> NDArray[np.float64]:
    """Combine ordinal ranks and the source's tiny fingerprint score tiebreak.

    Args:
      fingerprint: Eight-model fingerprint-based probabilities in candidate order.
      own: Four-model simulation probabilities in candidate order.
      weight_fingerprint: Fingerprint ensemble's contribution to the within-molecule rank mixture.

    Returns:
      scores: Blended ranking values in the original candidate order.

    """
    ranks = [1.0 - ordinal_ranks(values) for values in (fingerprint, own)]
    return (
        weight_fingerprint * ranks[0]
        + (1.0 - weight_fingerprint) * ranks[1]
        + 1e-6 * fingerprint
    )


class EngineIdentityTable:
    """Preserve first raw spellings, nullable failures and valid empty key strings."""

    def __init__(self, directory: Path) -> None:
        header = json_loads((directory / "manifest.json").read_text(), _Header)
        if header["format_version"] != 1 or header["rdkit"] != IDENTITY_RDKIT:
            raise ValueError(
                f"Engine identities require format 1 and RDKit {IDENTITY_RDKIT}.",
            )
        self.rows = tuple(
            from_plain(
                parquet.read_table(directory / "keys.parquet").to_pylist(),
                list[EngineIdentityRow],
            ),
        )
        if len(self.rows) != header["rows"]:
            raise ValueError("Engine identity row count differs from its manifest.")
        self._positions: dict[str, int] = {}
        for index, row in enumerate(self.rows):
            self._positions.setdefault(row["smiles"], index)

    def for_smiles(self, smiles: str) -> EngineIdentityRow:
        """Return the first complete declared raw row or raise for unstaged strings."""
        return self.rows[self._positions[smiles]]

    def key(self, smiles: str) -> str | None:
        """Return the engine key, preserving the empty string as a distinct outcome."""
        return self.rows[self._positions[smiles]]["key"]


class _Header(PreparationHeader):
    rows: int


class SecondBank:
    """Retain source encounter order in independently averaged single/merged arms."""

    class Config(Fig["SecondBank"]):
        base_dir: Path | str | None = None
        """Parent-supplied dataset root; every relative checkpoint resolves beneath it."""

        model: SecondFPNet.Config = field(default_factory=SecondFPNet.Config)
        """Architecture of every checkpoint; its widths are verified against each file."""

        single: list[LocalFile.Config] = field(
            default_factory=list[LocalFile.Config],
        )
        """Pinned single-spectrum checkpoints in source encounter order."""

        merged: list[LocalFile.Config] = field(
            default_factory=list[LocalFile.Config],
        )
        """Pinned merged-spectrum checkpoints in source encounter order."""

        peaks: SecondPeaks.Config = field(default_factory=SecondPeaks.Config)
        """Peak preparation and merging shared by both arms."""

        merged_energy: float = 25.0
        """Collision energy assigned to the merged spectrum."""

        device: str = "cpu"
        """Device for evaluation models and packed spectra."""

        @override
        def finalize(self) -> Self:
            for pinned in (*self.single, *self.merged):
                pinned.path = resolve_working_dir(self.base_dir, pinned.path)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.peaks = config.peaks.make()
        self.single = tuple(self._load(pinned) for pinned in config.single)
        self.merged = tuple(self._load(pinned) for pinned in config.merged)

    def predict(self, spectra: Sequence[QuerySpectrum]) -> NDArray[np.float32] | None:
        """Average every single and merged model exactly as the released bank.

        Args:
          spectra: Complete ordered raw spectra and their conditioning metadata.

        Returns:
          logits: Available-arm mean in F32, or None when neither arm has peaks.

        """
        return predict_second(
            spectra,
            single=self.single,
            merged=self.merged,
            device=self.config.device,
            peaks=self.peaks,
            merged_energy=self.config.merged_energy,
        )

    def _load(self, pinned: LocalFile.Config) -> SecondFPNet:
        path = pinned.make().path()
        checkpoint = from_plain(
            cast(object, torch.load(path, map_location="cpu", weights_only=True)),
            dict[str, object],
        )
        state = from_plain(checkpoint["model"], dict[str, torch.Tensor])
        template = self.config.model
        if (
            from_plain(checkpoint["nbits"], int) != template.channels_out
            or from_plain(checkpoint["d"], int) != template.channels_hidden
            or from_plain(checkpoint["layers"], int) != template.num_layers
        ):
            raise ValueError(
                f"Checkpoint {path.name} widths differ from the declared model.",
            )
        model = template.make()
        model.load_released_state(state)
        return model.to(self.config.device).eval()


class SecondRankerBank:
    """Fit the four own and eight fingerprint rankers before candidate scoring."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class RankerScores:
        """Complete own, fingerprint and blended scores in candidate row order."""

        own: RankingScores
        fingerprint: RankingScores
        blend: RankingScores

    class Config(Fig["SecondRankerBank"]):
        """Configure both fitted ensembles and their final ordinal mixture."""

        own: HistogramEnsemble.Config = field(default_factory=HistogramEnsemble.Config)
        """Four-model simulation ensemble with its declared feature-block order."""

        pv: HistogramEnsemble.Config = field(default_factory=HistogramEnsemble.Config)
        """Eight-model ensemble using the complete predicted-fingerprint features."""

        weight_fingerprint: float = 0.5
        """Fingerprint contribution to the final within-molecule ordinal mixture."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.own = config.own.make()
        self.pv = config.pv.make()
        self.blocks: tuple[str, ...] = ()

    def fit(self, own: SimulationRows, *, pv: SimulationRows) -> Self:
        """Fit both complete ensembles using their separate simulation tables.

        Args:
            own: Own simulation features, labels, groups and ordered block names.
            pv: Fingerprint simulation features, labels and groups.

        Returns:
            bank: This fitted bank with the own table's complete block order.

        """
        _ = self.own.fit(own)
        _ = self.pv.fit(pv)
        self.blocks = own.blocks
        return self

    def score(
        self,
        own_features: PredictionMatrix,
        *,
        pv_features: PredictionMatrix,
    ) -> RankerScores:
        """Score both feature families and blend their within-molecule ranks.

        Args:
            own_features: Complete own columns in the fitted block order.
            pv_features: Complete fingerprint columns for the same candidate rows.

        Returns:
            scores: Both mean probabilities and the final double rank mixture.

        """
        own = self.own.predict(own_features)
        fingerprint = self.pv.predict(pv_features)
        return self.RankerScores(
            own=own,
            fingerprint=fingerprint,
            blend=blend_histograms(
                fingerprint,
                own=own,
                weight_fingerprint=self.config.weight_fingerprint,
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SecondInputs:
    """Carry complete raw second-arm data and both fully fitted ranking families."""

    pool: SecondPool
    library: SecondLibrary
    fragments: FragmentMassStore
    bank: SecondBank
    rankers: SecondRankerBank
    formulas: tuple[str, ...]
    identities: EngineIdentityTable


class SecondAssetProvider(Protocol):
    """Load every independent second-arm field and retain staging failures."""

    def read(self) -> SecondInputs:
        """Return all prepared data and every model, without fallback fields."""
        ...


class SecondAssets:
    """Load raw library/fragments and fit the complete four-own/eight-PV bank."""

    class Config(Fig["SecondAssets"]):
        base_dir: Path | str | None = None
        """Parent-supplied dataset root; every relative path below resolves beneath it."""

        pool_dir: Path | str = "native-chemistry/engine2-pool"
        """Complete independently sorted second pool, preserving duplicate keys."""

        library_dir: Path | str = "native-chemistry/engine2-library"
        """Every complete raw training spectrum in original Parquet row order."""

        fragments_dir: Path | str = "native-chemistry/engine2-fragments"
        """Full unrounded F64 fragments in exact second-pool row order."""

        chemistry: IdentityTable.Config = field(default_factory=IdentityTable.Config)
        """Explicit raw formula join covering every second-pool SMILES."""

        identities_dir: Path | str = "native-chemistry/engine-identities"
        """The second engine's distinct raw key decisions, including empty keys."""

        own_training: Path | str = (
            "inputs/datasets/megayak/casmi26-simulated-ranker-rows/sim_rank_rows_nofp.npz"
        )
        """Complete own feature/label archive, using simulation class S."""

        pv_training: Path | str = (
            "inputs/datasets/prvsiyan/casmi26-ranker-features/rank_train.npz"
        )
        """Complete fingerprint feature/label archive, using simulation class M."""

        bank: SecondBank.Config = field(default_factory=SecondBank.Config)
        """Every actual released single and merged checkpoint."""

        rankers: SecondRankerBank.Config = field(
            default_factory=SecondRankerBank.Config,
        )
        """Complete native 500-round public training and twelve-model mixture."""

        @override
        def finalize(self) -> Self:
            self.pool_dir = resolve_working_dir(self.base_dir, self.pool_dir)
            self.library_dir = resolve_working_dir(self.base_dir, self.library_dir)
            self.fragments_dir = resolve_working_dir(self.base_dir, self.fragments_dir)
            self.identities_dir = resolve_working_dir(
                self.base_dir,
                self.identities_dir,
            )
            self.own_training = resolve_working_dir(self.base_dir, self.own_training)
            self.pv_training = resolve_working_dir(self.base_dir, self.pv_training)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def read(self) -> SecondInputs:
        """Read complete raw data, strict identities and all twelve fitted models.

        Returns:
          inputs: Every prepared row and all complete released/fitted model states.

        """
        config = self.config
        pool = SecondPool.load(Path(config.pool_dir))
        library = _raw_library(Path(config.library_dir))
        fragments = Path(config.fragments_dir)
        offsets = _array(fragments / "frag_off.npy", dtype=np.int64)
        mass = _array(fragments / "frag_mass.npy", dtype=np.float64)
        _offsets(offsets, rows=len(pool.mass), size=len(mass))
        chemistry = config.chemistry.make()
        identities = EngineIdentityTable(Path(config.identities_dir))
        formulas = tuple(chemistry.formula(value) for value in pool.smiles)
        for value in pool.smiles:
            _ = identities.key(value)
        return SecondInputs(
            pool=pool,
            library=library,
            fragments=FragmentMassStore(offsets=offsets, mass=mass),
            bank=config.bank.make(),
            rankers=config.rankers.make().fit(
                SimulationRows.load(Path(config.own_training), "S"),
                pv=SimulationRows.load(Path(config.pv_training), "M"),
            ),
            formulas=formulas,
            identities=identities,
        )


def _raw_library(directory: Path) -> SecondLibrary:
    mass = _array(directory / "nm.npy", dtype=np.float64)
    rows = _array(directory / "row_p.npy", dtype=np.int64)
    polarity = _array(directory / "pol.npy", dtype=np.int8)
    timstof = _array(directory / "tims.npy", dtype=np.bool_)
    precursor = _array(directory / "prec.npy", dtype=np.float64)
    offsets = _array(directory / "off.npy", dtype=np.int64)
    mz = _array(directory / "mz.npy", dtype=np.float32)
    intensity = _array(directory / "it.npy", dtype=np.float32)
    if any(len(value) != len(mass) for value in (rows, polarity, timstof, precursor)):
        raise ValueError("Second-arm library metadata rows must be aligned.")
    if len(mz) != len(intensity):
        raise ValueError("Second-arm raw peak masses and intensities must be aligned.")
    _offsets(offsets, rows=len(mass), size=len(mz))
    return SecondLibrary(
        store=PeakStore(offsets=offsets, mz=mz, intensity=intensity),
        mass=mass,
        pool_rows=rows,
        polarity=polarity,
        timstof=timstof,
        precursor=precursor,
    )


def _array[S: np.generic](path: Path, *, dtype: type[S]) -> NDArray[S]:
    values = load_array(path, dtype, mmap=True)
    if values.ndim != 1:
        raise ValueError(f"Second-arm array {path.name} has an invalid rank.")
    return values


def _offsets(offsets: NDArray[np.int64], *, rows: int, size: int) -> None:
    if (
        len(offsets) != rows + 1
        or offsets[0] != 0
        or offsets[-1] != size
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("Second-arm offsets must cover every complete ordered row.")


def _analog_evidence(
    candidates: NDArray[np.float32],
    analogs: NDArray[np.float32] | None,
    similarity: NDArray[np.float32],
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
]:
    if analogs is None or not len(similarity):
        empty = np.zeros(len(candidates), np.float32)
        return empty, empty, empty, empty, empty, 0.0
    analogs = analogs.astype(np.float32)
    intersection = candidates @ analogs.T
    union = candidates.sum(1)[:, None] + analogs.sum(1)[None, :] - intersection + 1e-9
    tanimoto = intersection / union
    weights = np.clip(np.asarray(similarity, np.float32), 0, None)
    cubic = (weights.astype(np.float64) ** 3).astype(np.float32)
    return (
        (tanimoto * cubic[None, :]).max(1),
        (tanimoto * weights[None, :]).max(1),
        tanimoto.max(1),
        tanimoto[:, 0],
        (tanimoto * cubic[None, :]).sum(1) / (cubic.sum() + 1e-9),
        weights.item(0),
    )


def _model_evidence(
    candidates: NDArray[np.float32],
    cardinality: NDArray[np.float32],
    logits: FloatArray | None,
) -> list[FloatArray]:
    if logits is None:
        return [np.zeros(len(candidates), np.float32)] * 6
    scores = (candidates @ np.asarray(logits, np.float64)).astype(np.float32)
    normalized = scores / np.sqrt(np.maximum(cardinality, 1.0))
    return [
        population_z(scores),
        ordinal_ranks(scores),
        scores - scores.max(),
        population_z(normalized),
        ordinal_ranks(normalized),
        np.equal(scores, scores.max()).astype(np.float32),
    ]


def _model_interactions(
    candidates: NDArray[np.float32],
    library: NDArray[np.float32],
    propagation: FloatArray,
    logits: FloatArray | None,
) -> list[FloatArray]:
    count = len(candidates)
    if logits is None or not count:
        return [np.zeros(count, np.float32)] * 6
    ranks = ordinal_ranks(
        (candidates @ np.asarray(logits, np.float64)).astype(np.float32),
    )
    library_agreement = (
        1.0 - ranks.item(int(np.argmax(library))) if library.max() > 0 else 0.0
    )
    analog_agreement = (
        1.0 - ranks.item(int(np.argmax(propagation))) if propagation.max() > 0 else 0.0
    )
    correlation = (
        np.corrcoef(library, -ranks).item(0, 1) if library.std() > 1e-9 else 0.0
    )
    return [
        library * (1.0 - ranks),
        propagation * (1.0 - ranks),
        np.full(count, library_agreement),
        np.full(count, analog_agreement),
        np.full(count, library_agreement * float(library.max())),
        np.full(count, correlation),
    ]


def _relative(values: FloatArray) -> list[NDArray[np.float32]]:
    values = np.asarray(values, dtype=np.float32)
    return [values, ordinal_ranks(values).astype(np.float32), values - values.max()]


def _polarity_features(
    pool: FingerprintPool,
    *,
    fingerprints: NDArray[np.float32],
    channels: SecondChannels,
) -> NDArray[np.float32]:
    if not channels.polarity_analogs:
        return np.zeros((len(fingerprints), 5), dtype=np.float32)
    rows = np.fromiter(
        (analog.sid for analog in channels.polarity_analogs),
        dtype=np.int64,
    )
    analog_fp = pool.fingerprints(rows).astype(np.float32)
    intersection = fingerprints @ analog_fp.T
    tanimoto = intersection / (
        fingerprints.sum(1)[:, None] + analog_fp.sum(1)[None, :] - intersection + 1e-9
    )
    weights = np.clip(
        np.array(
            [analog.similarity for analog in channels.polarity_analogs],
            dtype=np.float32,
        ),
        0,
        None,
    )
    propagation = (
        tanimoto * (weights.astype(np.float64) ** 3).astype(np.float32)[None, :]
    ).max(1)
    strong = (
        tanimoto * (weights.astype(np.float64) ** 6).astype(np.float32)[None, :]
    ).max(1)
    return np.column_stack(
        [*_relative(propagation), strong, np.full(len(fingerprints), weights.item(0))],
    ).astype(np.float32)


def _proton_carrier(mode: float) -> tuple[float, ...]:
    return (_proton(positive=mode > 0),)


def _proton(*, positive: bool) -> float:
    proton = second_adduct_parameters("[M+H]+")[2]
    return proton if positive else -proton


def _adduct_carriers(adduct: str) -> tuple[float, ...]:
    if adduct in ("[M+NH4]+", "[M+Na]+", "[M+K]+", "[M+CH2O2-H]-", "[M+Cl]-"):
        return (
            _proton(positive=adduct.endswith("+")),
            second_adduct_parameters(adduct)[2],
        )
    return (_proton(positive="+" in adduct[-2:]),)


def _explanation(
    masses: NDArray[np.float64],
    *,
    query: tuple[NDArray[np.float64], NDArray[np.float64]],
    carriers: tuple[float, ...],
) -> float:
    if not len(masses) or not len(query[0]):
        return 0.0
    ions = np.sort(
        np.concatenate(
            [
                masses + shift * HYDROGEN_MASS + carrier
                for shift in (-2, -1, 0, 1, 2)
                for carrier in carriers
            ],
        ),
    )
    weights = np.asarray(np.sqrt(query[1]), dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        return 0.0
    indices = np.searchsorted(ions, query[0])
    matched = np.zeros(len(query[0]), dtype=np.bool_)
    for offset in (-1, 0):
        nearest = np.clip(indices + offset, 0, len(ions) - 1)
        matched |= np.abs(ions[nearest] - query[0]) <= 0.01
    return float(weights[matched].sum() / total)


def _sigmoid(value: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return 0.0
