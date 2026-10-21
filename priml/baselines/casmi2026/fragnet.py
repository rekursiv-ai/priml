"""Native fragment-network data, records and ranking features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast
from zipfile import ZipFile

import hashlib
import io
import math

from numpy.typing import NDArray
from treekle import from_plain, json_loads

import numpy as np

from priml.baselines.casmi2026.chemistry import MainSearch
from priml.baselines.casmi2026.custom_types import (
    FragnetMetadata,
    PreparationHeader,
)
from priml.baselines.casmi2026.isolated import (
    HYDROGEN_MASS,
    IDENTITY_RDKIT,
    ScientificDataUnpickler,
    main_ion_deltas,
)
from priml.baselines.casmi2026.scores import average_ranks


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pyarrow import parquet

    from priml.baselines.casmi2026.boosting import LightGBMModel
    from priml.baselines.casmi2026.custom_types import (
        FragnetRecord,
        SpectrumRequest,
    )
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")


class FragnetTable:
    """Require an explicit prepared record for each requested structure."""

    def __init__(self, directory: Path) -> None:
        header = json_loads(
            (directory / "manifest.json").read_text(),
            PreparationHeader,
        )
        if header["format_version"] != 1 or header["rdkit"] != IDENTITY_RDKIT:
            raise ValueError(
                f"Fragnet data requires format 1 and RDKit {IDENTITY_RDKIT}.",
            )
        self.metadata = from_plain(
            parquet.read_table(
                directory / "molecules.parquet",
                columns=["smiles", "parsed", "valid"],
            ).to_pylist(),
            list[FragnetMetadata],
        )
        self.amass = _array(directory / "amass.npy", dtype=np.float64, dimensions=1)
        self.kind = _array(directory / "kind.npy", dtype=np.int8, dimensions=1)
        self.el = _array(directory / "el.npy", dtype=np.int8, dimensions=1)
        self.nh = _array(directory / "nh.npy", dtype=np.int8, dimensions=1)
        self.el10 = _array(directory / "el10.npy", dtype=np.uint8, dimensions=1)
        self.bonds = _array(directory / "bonds.npy", dtype=np.int32, dimensions=2)
        self.bring = _array(directory / "bring.npy", dtype=np.int8, dimensions=1)
        self.mass = _array(directory / "mass.npy", dtype=np.float64, dimensions=1)
        if any(
            len(values) != len(self.amass)
            for values in (self.kind, self.el, self.nh, self.el10)
        ):
            raise ValueError(
                "Fragnet atom arrays differ from the complete atom layout.",
            )
        if self.bonds.shape[1] != 2 or len(self.bring) != len(self.bonds):
            raise ValueError(
                "Fragnet bond arrays differ from the complete edge layout.",
            )
        if len(self.mass) != len(self.metadata):
            raise ValueError("Fragnet molecular masses differ from the metadata rows.")
        self.atom_offsets = _offsets(
            directory / "atom_off.npy",
            rows=len(self.metadata),
            length=len(self.amass),
        )
        self.bond_offsets = _offsets(
            directory / "bond_off.npy",
            rows=len(self.metadata),
            length=len(self.bonds),
        )
        self._positions: dict[str, int] = {}
        for index, row in enumerate(self.metadata):
            self._positions.setdefault(row["smiles"], index)

    def row(self, index: int) -> FragnetRecord | None:
        """Return every molecular field, or a declared invalid row.

        Args:
          index: Original nonnegative row position.

        Returns:
          record: Complete readonly numeric graph or an explicitly rejected row.

        """
        if index < 0 or index >= len(self.metadata):
            raise IndexError(index)
        if not self.metadata[index]["valid"]:
            return None
        atoms = slice(
            self.atom_offsets.item(index),
            self.atom_offsets.item(index + 1),
        )
        bonds = slice(
            self.bond_offsets.item(index),
            self.bond_offsets.item(index + 1),
        )
        return {
            "amass": self.amass[atoms],
            "kind": self.kind[atoms],
            "el": self.el[atoms],
            "nh": self.nh[atoms],
            "el10": self.el10[atoms],
            "bonds": self.bonds[bonds],
            "bring": self.bring[bonds],
            "mass": float(self.mass.item(index)),
        }

    def for_smiles(self, smiles: str) -> FragnetRecord | None:
        """Return the first prepared row; absent structures are staging errors."""
        return self.row(self._positions[smiles])


type Fragments = tuple[
    NDArray[np.float64],
    NDArray[np.int8],
    NDArray[np.int16],
    NDArray[np.int16],
    NDArray[np.int16],
    NDArray[np.int16],
    NDArray[np.int16],
]

FRAGMENT_LIMIT: Final = 30_000
"""Source enumeration capacity of fragments per candidate."""

CANDIDATE_LIMIT: Final = 400
"""Source tie-inclusive cap on candidates scored by the family."""

PEAK_SEARCH: Final = MainSearch.Config(max_peaks=128, entropy_weight=False)
"""Source peak preparation: 128 intensity-ranked peaks without entropy weighting."""


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Peaks:
    """Retain every family peak, spectrum context and observable mass window."""

    S: int
    pk_mz: NDArray[np.float64]
    pk_w: NDArray[np.float64]
    pk_off: NDArray[np.int64]
    pk_spec: NDArray[np.int64]
    sp_del: NDArray[np.float64]
    sp_nd: NDArray[np.int64]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    top: NDArray[np.int64]
    wtot: NDArray[np.float64]
    mode: NDArray[np.int64]
    ce: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Prepared:
    """Keep positional fragment annotations, best matches and every CSR match."""

    cat: Fragments
    coff: NDArray[np.int64]
    M: NDArray[np.float64]
    cls: NDArray[np.int8]
    pkix: NDArray[np.int32]
    inv: NDArray[np.int64]
    moff: NDArray[np.int64]
    mk: NDArray[np.int32]
    mc: NDArray[np.int8]
    n: int


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class StageA:
    """Back off complete categorical six-class tables across ordered key chains."""

    chains: tuple[tuple[tuple[str, ...], ...], ...]
    tabs: tuple[tuple[tuple[NDArray[np.int64], NDArray[np.float32]], ...], ...]
    alpha: float

    @classmethod
    def load(cls, path: Path, expected_sha256: str) -> StageA:
        """Read a pinned scientific-data table bank without importing original code.

        Args:
          path: Complete released StageA NumPy archive.
          expected_sha256: Digest of the complete archive.

        Returns:
          model: Every ordered chain and fitted six-class backoff table.

        """
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("Fragnet StageA asset digest differs.")
        with ZipFile(io.BytesIO(raw)) as archive:
            chains = _read_chains(archive.read("chains.npy"))
            tabs = tuple(
                tuple(
                    (
                        _table_array(
                            cast(
                                object,
                                np.load(
                                    io.BytesIO(archive.read(f"k_{ci}_{li}.npy")),
                                    allow_pickle=False,
                                ),
                            ),
                            dtype=np.int64,
                            ndim=1,
                        ),
                        _table_array(
                            cast(
                                object,
                                np.load(
                                    io.BytesIO(archive.read(f"l_{ci}_{li}.npy")),
                                    allow_pickle=False,
                                ),
                            ),
                            dtype=np.float32,
                            ndim=2,
                        ),
                    )
                    for li in range(len(chain))
                )
                for ci, chain in enumerate(chains)
            )
            alpha = float(
                np.asarray(
                    cast(
                        object,
                        np.load(
                            io.BytesIO(archive.read("alpha.npy")),
                            allow_pickle=False,
                        ),
                    ),
                    dtype=np.float64,
                ).item(),
            )
        if not chains or any(not chain for chain in chains):
            raise ValueError("StageA requires nonempty categorical chains.")
        for chain, tables in zip(chains, tabs, strict=True):
            for level, (keys, values) in zip(chain, tables, strict=True):
                if (
                    not level
                    or any(name not in _cards() for name in level)
                    or values.shape != (len(keys), 6)
                    or np.any(np.diff(keys) <= 0)
                ):
                    raise ValueError("StageA categorical table schema differs.")
            if chain[0] != chains[0][0]:
                raise ValueError("StageA chains require a shared first level.")
        return cls(chains=chains, tabs=tabs, alpha=alpha)

    def llr(self, fields: dict[str, NDArray[np.int64]]) -> NDArray[np.float64]:
        """Return all six log-likelihood ratios for every annotated fragment.

        Args:
          fields: Every binned field in the fitted categorical vocabulary.

        Returns:
          ratios: Complete fragment-by-six matrix after ordered chain backoff.

        """
        output = np.zeros((len(fields["ft"]), 6))
        for ci, (chain, tables) in enumerate(zip(self.chains, self.tabs, strict=True)):
            keys_by_level = _chain_keys(fields, chain)
            values = np.zeros_like(output)
            pending = np.ones(len(output), np.bool_)
            for query, (keys, table) in reversed(
                list(zip(keys_by_level, tables, strict=True)),
            ):
                if not len(keys) or not pending.any():
                    continue
                rows = np.flatnonzero(pending)
                indices = np.minimum(np.searchsorted(keys, query[rows]), len(keys) - 1)
                found = np.equal(keys[indices], query[rows])
                values[rows[found]] = table[indices[found]]
                pending[rows[found]] = False
            if ci == 0:
                output = values
            else:
                keys, table = self.tabs[0][0]
                query = _chain_keys(fields, self.chains[0])[0]
                indices = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
                base = np.where(
                    np.equal(keys[indices], query)[:, None],
                    table[indices],
                    0.0,
                )
                output = output + values - base
        return output


def fragments(
    record: FragnetRecord | None,
    *,
    maximum: int = FRAGMENT_LIMIT,
) -> Fragments:
    """Enumerate every annotated one- and two-bond cleavage in source order.

    Args:
      record: Complete prepared graph, or an explicitly invalid molecule.
      maximum: Source enumeration capacity, retaining the first complete rows.

    Returns:
      rows: Mass, cut kind, both cut classes, heavy-atom, nitrogen and oxygen counts.

    """
    rows: list[tuple[float, int, int, int, int, int, int]] = []
    if record is not None:
        bonds = record["bonds"]
        for first in range(len(bonds)):
            groups = _model_components(record, first=first, second=-1)
            if len(groups) > 1:
                for group in groups:
                    if len(rows) == maximum:
                        break
                    rows.append(
                        _fragment_row(record, group=group, first=first, second=-1),
                    )
            for second in range(first + 1, len(bonds)):
                if len(rows) == maximum:
                    break
                if (record["bring"].item(first) > 0) != (
                    record["bring"].item(second) > 0
                ):
                    continue
                for group in _model_components(record, first=first, second=second):
                    a, b = (bonds.item((first, side)) for side in range(2))
                    c, d = (bonds.item((second, side)) for side in range(2))
                    if (a in group) == (b in group) or (c in group) == (d in group):
                        continue
                    if len(rows) == maximum:
                        break
                    rows.append(
                        _fragment_row(record, group=group, first=first, second=second),
                    )
    return (
        np.array([row[0] for row in rows], np.float64),
        np.array([row[1] for row in rows], np.int8),
        np.array([row[2] for row in rows], np.int16),
        np.array([row[3] for row in rows], np.int16),
        np.array([row[4] for row in rows], np.int16),
        np.array([row[5] for row in rows], np.int16),
        np.array([row[6] for row in rows], np.int16),
    )


def categorical_fields(
    rows: Fragments,
    parent: NDArray[np.float64],
    *,
    mode: int,
    energy: float,
) -> dict[str, NDArray[np.int64]]:
    """Bin all nine fragment fields for the complete categorical table schema.

    Args:
      rows: Complete intrinsic fragment annotations.
      parent: Parent molecular mass repeated for each fragment.
      mode: Positive or negative polarity.
      energy: Collision energy, retaining nonfinite values as the missing bin.

    Returns:
      fields: All nine integer categorical columns.

    """
    mass, kind, first, second, heavy, nitrogen, oxygen = rows
    count = len(mass)
    return {
        "mode": np.full(count, int(mode > 0)),
        "ce": np.full(
            count,
            1 + int(np.searchsorted([15.0, 30.0, 50.0], energy))
            if math.isfinite(energy)
            else 0,
        ),
        "ft": kind.astype(np.int64),
        "c1": first.astype(np.int64),
        "c2": second.astype(np.int64) + 1,
        "size": np.searchsorted([3, 5, 7, 10, 14, 20, 30], heavy, side="right").astype(
            np.int64,
        ),
        "frac": np.searchsorted(
            np.arange(1, 10) / 10.0,
            mass / np.maximum(parent, 1e-9),
        ).astype(np.int64),
        "nN": np.minimum(nitrogen, 3).astype(np.int64),
        "nO": np.minimum(oxygen, 5).astype(np.int64),
    }


def prepare_peaks(spectra: Sequence[SpectrumRequest]) -> Peaks | None:
    """Prepare complete family peaks, dropping only empty query spectra.

    Args:
      spectra: Complete family-prepared query spectra in original order.

    Returns:
      peaks: Full peak and spectrum contexts, or None when every spectrum is empty.

    """
    search = PEAK_SEARCH.make()
    rows: list[tuple[NDArray[np.float64], NDArray[np.float64], SpectrumRequest]] = []
    for spectrum in spectra:
        mass, intensity = search.prepare(
            (
                np.asarray(spectrum["mz"], np.float64),
                np.asarray(spectrum["it"], np.float64),
            ),
        )
        if len(mass) and intensity.max() > 0:
            order = np.argsort(mass, kind="mergesort")
            rows.append(
                (mass[order], np.sqrt(intensity[order] / intensity.max()), spectrum),
            )
    if not rows:
        return None
    offsets = np.concatenate([[0], np.cumsum([len(row[0]) for row in rows])]).astype(
        np.int64,
    )
    mass = np.concatenate([row[0] for row in rows])
    weight = np.concatenate([row[1] for row in rows])
    deltas = np.zeros((len(rows), 4))
    delta_count = np.zeros(len(rows), np.int64)
    top = np.zeros(len(mass), np.int64)
    for index, (_, values, spectrum) in enumerate(rows):
        carriers = main_ion_deltas(spectrum["adduct"])[:4]
        deltas[index, : len(carriers)] = carriers
        delta_count[index] = len(carriers)
        top[offsets.item(index) + np.argsort(-values, kind="mergesort")[:5]] = 1
    return Peaks(
        S=len(rows),
        pk_mz=mass,
        pk_w=weight,
        pk_off=offsets,
        pk_spec=np.repeat(np.arange(len(rows)), np.diff(offsets)),
        sp_del=deltas,
        sp_nd=delta_count,
        lo=np.array([max(50.0, row[0].min() - 0.5) for row in rows]),
        hi=np.array([row[2]["prec"] - 2.0 for row in rows]),
        top=top,
        wtot=np.array([row[1].sum() for row in rows]),
        mode=np.array([int(row[2]["mode"]) for row in rows], np.int64),
        ce=np.array([_energy(row[2]) for row in rows]),
    )


def prepare_candidates(
    records: Sequence[FragnetRecord | None],
    peaks: Peaks,
) -> Prepared:
    """Match every unique rounded mass while preserving candidate multiplicity.

    Args:
      records: Every candidate record, including positional duplicates.
      peaks: Complete prepared peak and spectrum contexts.

    Returns:
      prepared: All annotations, best matches and every ordered CSR match.

    """
    rows = [fragments(record) for record in records]
    empty = fragments(None)
    cat: Fragments = (
        np.concatenate([row[0] for row in rows]) if rows else empty[0],
        np.concatenate([row[1] for row in rows]) if rows else empty[1],
        np.concatenate([row[2] for row in rows]) if rows else empty[2],
        np.concatenate([row[3] for row in rows]) if rows else empty[3],
        np.concatenate([row[4] for row in rows]) if rows else empty[4],
        np.concatenate([row[5] for row in rows]) if rows else empty[5],
        np.concatenate([row[6] for row in rows]) if rows else empty[6],
    )
    offsets = np.concatenate([[0], np.cumsum([len(row[0]) for row in rows])]).astype(
        np.int64,
    )
    masses, inverse = np.unique(np.round(cat[0], 6), return_inverse=True)
    classes, best, match_offsets, match_peak, match_class = _matches(masses, peaks)
    return Prepared(
        cat=cat,
        coff=offsets,
        M=np.repeat(
            np.array(
                [record["mass"] if record is not None else 1.0 for record in records],
            ),
            np.diff(offsets),
        ),
        cls=classes[inverse],
        pkix=best[inverse],
        inv=inverse.astype(np.int64),
        moff=match_offsets,
        mk=match_peak,
        mc=match_class,
        n=len(records),
    )


def aggregates(
    model: StageA,
    prepared: Prepared,
    peaks: Peaks,
    *,
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Accumulate all nineteen fragment- and peak-centric channels in fixed order.

    Args:
      model: Complete fitted categorical table bank.
      prepared: Every candidate fragment and matching CSR row.
      peaks: Complete peak and spectrum contexts.
      mask: Candidate rows used for discriminative explanation frequencies.

    Returns:
      values: All nineteen channels per candidate and spectrum.

    """
    llr = np.zeros((len(prepared.M), peaks.S, 6))
    for spectrum in range(peaks.S):
        llr[:, spectrum] = model.llr(
            categorical_fields(
                prepared.cat,
                prepared.M,
                mode=peaks.mode.item(spectrum),
                energy=peaks.ce.item(spectrum),
            ),
        )
    best = np.full((prepared.n, len(peaks.pk_mz)), -1e30, dtype=np.float64)
    for candidate in range(prepared.n):
        for fragment in range(
            prepared.coff.item(candidate),
            prepared.coff.item(candidate + 1),
        ):
            unique = prepared.inv.item(fragment)
            for match in range(
                prepared.moff.item(unique),
                prepared.moff.item(unique + 1),
            ):
                peak = prepared.mk.item(match)
                value = llr.item(
                    (fragment, peaks.pk_spec.item(peak), prepared.mc.item(match)),
                )
                best[candidate, peak] = max(best.item((candidate, peak)), value)
    if mask is None:
        mask = np.ones(prepared.n, np.bool_)
    frequency = (best[mask] > -1e29).sum(0) / max(int(mask.sum()), 1)
    output = np.zeros((prepared.n, peaks.S, 19))
    for candidate in range(prepared.n):
        for fragment in range(
            prepared.coff.item(candidate),
            prepared.coff.item(candidate + 1),
        ):
            for spectrum in range(peaks.S):
                category = prepared.cls.item((fragment, spectrum))
                if category >= 0:
                    value = llr.item((fragment, spectrum, category))
                    row = cast("NDArray[np.float64]", output[candidate, spectrum])
                    row[0] += value
                    row[3] += 1.0
                    kind = prepared.cat[1].item(fragment)
                    row[9 + kind] += value
                    if kind == 1:
                        row[13] += 1.0
                    row[1 if category > 0 else 2] += value
                    if category > 0:
                        row[4] += 1.0
                        if kind == 1:
                            row[14] += 1.0
        for peak in range(len(peaks.pk_w)):
            spectrum = peaks.pk_spec.item(peak)
            row = cast("NDArray[np.float64]", output[candidate, spectrum])
            value = best.item((candidate, peak))
            weight = peaks.pk_w.item(peak)
            if value > -1e29:
                row[5] += weight * value
                row[16] += weight * (1.0 - frequency.item(peak)) * value
                if value > 0:
                    row[6] += weight * value
                    row[17] += weight * (1.0 - frequency.item(peak)) * value
                row[7] += weight
                row[18] += weight * (1.0 - frequency.item(peak))
                if peaks.top.item(peak) > 0:
                    row[8] += value
                    row[9] += 1.0
            elif peaks.top.item(peak) > 0:
                row[15] += weight
    return output


def stage_b_features(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build all31 StageB aggregate, ratio, average-rank and gap columns.

    Args:
      values: Complete nineteen-column aggregate for one spectrum.

    Returns:
      matrix: All31 model inputs in released feature order.

    """
    if not len(values):
        return np.zeros((0, 31))
    columns = [values[:, index] for index in range(19)]
    columns.extend(
        (
            values[:, 0] / np.maximum(values[:, 3], 1.0),
            values[:, 4] / np.maximum(values[:, 3], 1.0),
            values[:, 5] + 0.02 * values[:, 2],
            values[:, 14] / np.maximum(values[:, 13], 1.0),
        ),
    )
    for index in (5, 0, 21, 7):
        column = columns[index]
        columns.extend((average_ranks(column), column - column.max()))
    return np.stack(columns, 1)


def stage_b_names() -> tuple[str, ...]:
    """Return the exact31-column schema declared by the released StageB booster.

    Returns:
      names: Every aggregate, ratio and relative feature in model order.

    """
    return (
        "fc",
        "fc_m",
        "fc_u",
        "n_obs",
        "n_match",
        "pk",
        "pk_pos",
        "expl",
        "top5",
        "top5_n",
        "fc_d1",
        "fc_ro",
        "fc_mid",
        "obs_d1",
        "match_d1",
        "unexpl_top",
        "pkd",
        "pkd_pos",
        "expl_d",
        "fc_n",
        "match_r",
        "pk_fcu",
        "cov1",
        "pk_rk",
        "pk_gap",
        "fc_rk",
        "fc_gap",
        "pk_fcu_rk",
        "pk_fcu_gap",
        "expl_rk",
        "expl_gap",
    )


def features(
    records: Sequence[FragnetRecord | None],
    spectra: Sequence[SpectrumRequest],
    *,
    formulas: Sequence[str],
    model_score: NDArray[np.float32],
    stage_a: StageA,
    stage_b: LightGBMModel,
    maximum: int = CANDIDATE_LIMIT,
) -> dict[str, NDArray[np.float32]]:
    """Compute all fourteen family outputs with the tie-inclusive candidate cap.

    Args:
      records: Every explicitly prepared candidate graph or known-invalid record.
      spectra: Complete family-prepared query spectra.
      formulas: Candidate formulas retaining original row order.
      model_score: Base Full1 fz scores used to select the candidate geometry.
      stage_a: Complete fitted categorical bank.
      stage_b: Complete fitted31-column native booster.
      maximum: Tie-inclusive source candidate cap.

    Returns:
      values: All fourteen source family columns in declared order.

    """
    names = (
        "fn_score",
        "fn_score_max",
        "fn_rank",
        "fn_gap",
        "fn_grp_rank",
        "fn_grp_gap",
        "fn_vs_lead",
        "fn_raw_rank",
        "fn_raw_gap",
        "fn_raw_grp_gap",
        "fn_raw_vs_lead",
        "fn_pk_rank",
        "fn_fc_rank",
        "fn_sp_rank",
    )
    count = len(records)
    output = {name: np.zeros(count) for name in names}
    peaks = prepare_peaks(spectra) if count else None
    if peaks is None:
        return {name: array.astype(np.float32) for name, array in output.items()}
    selected = np.arange(count)
    if count > maximum:
        selected = np.flatnonzero(
            model_score >= np.sort(model_score)[::-1].item(maximum - 1),
        )
    if stage_b.features != stage_b_names():
        raise ValueError("Fragnet StageB feature schema differs.")
    agg = aggregates(
        stage_a,
        prepare_candidates(
            [records[selected.item(i)] for i in range(len(selected))],
            peaks,
        ),
        peaks,
    )
    scores = np.zeros((len(selected), peaks.S))
    raw = np.zeros_like(scores)
    peak_score = np.zeros_like(scores)
    fragment_score = np.zeros_like(scores)
    ranks = np.zeros_like(scores)
    for spectrum in range(peaks.S):
        matrix = stage_b_features(agg[:, spectrum])
        scores[:, spectrum] = stage_b.predict(
            matrix.astype(np.float32).astype(np.float64),
        )
        total = max(peaks.wtot.item(spectrum), 1e-9)
        raw[:, spectrum] = (agg[:, spectrum, 5] + 0.02 * agg[:, spectrum, 2]) / total
        peak_score[:, spectrum] = agg[:, spectrum, 5] / total
        fragment_score[:, spectrum] = agg[:, spectrum, 0] / np.maximum(
            agg[:, spectrum, 3],
            1.0,
        )
        ranks[:, spectrum] = average_ranks(scores[:, spectrum])
    means = [
        _mean_spectra(array)
        for array in (scores, raw, peak_score, fragment_score, ranks)
    ]
    full: list[NDArray[np.float64]] = []
    for values in (means[0], scores.max(1), means[1], means[2], means[3]):
        expanded = np.full(count, values.min() - 1.0)
        expanded[selected] = values
        full.append(expanded)
    score, maximum_score, raw_score, pk_score, fc_score = full
    lead = np.flatnonzero(np.equal(model_score[selected], model_score[selected].max()))
    output.update(
        fn_score=score,
        fn_score_max=maximum_score,
        fn_rank=average_ranks(score),
        fn_gap=score - score.max(),
        fn_raw_rank=average_ranks(raw_score),
        fn_raw_gap=raw_score - raw_score.max(),
        fn_vs_lead=score - float(np.sort(means[0][lead]).mean()),
        fn_raw_vs_lead=raw_score - float(np.sort(means[1][lead]).mean()),
        fn_pk_rank=average_ranks(pk_score),
        fn_fc_rank=average_ranks(fc_score),
    )
    for formula in sorted(set(formulas)):
        group = np.array(
            [index for index, value in enumerate(formulas) if value == formula],
        )
        output["fn_grp_rank"][group] = average_ranks(score[group])
        output["fn_grp_gap"][group] = score[group] - score[group].max()
        output["fn_raw_grp_gap"][group] = raw_score[group] - raw_score[group].max()
    output["fn_sp_rank"][:] = 1.0
    output["fn_sp_rank"][selected] = means[4]
    return {
        name: np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        for name, array in output.items()
    }


def _array[S: np.generic](path: Path, *, dtype: type[S], dimensions: int) -> NDArray[S]:
    value = cast("NDArray[S]", np.load(path, mmap_mode="r"))
    if value.dtype.type is not dtype or value.ndim != dimensions:
        raise ValueError(f"Fragnet array {path.name} has an invalid dtype or rank.")
    return value


def _offsets(path: Path, *, rows: int, length: int) -> NDArray[np.int64]:
    values = _array(path, dtype=np.int64, dimensions=1)
    if (
        values.shape != (rows + 1,)
        or values.item(0) != 0
        or values.item(-1) != length
        or np.any(np.diff(values) < 0)
    ):
        raise ValueError(
            "Fragnet ragged offsets differ from the complete graph layout.",
        )
    return values


def _table_array[S: np.generic](
    value: object,
    *,
    dtype: type[S],
    ndim: int,
) -> NDArray[S]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype.type is not dtype
        or value.ndim != ndim
    ):
        raise ValueError("StageA table dtype or dimensions differ.")
    return cast("NDArray[S]", value)


def _cards() -> dict[str, int]:
    return {
        "mode": 2,
        "ce": 5,
        "ft": 4,
        "c1": 13 * 13 * 3,
        "c2": 13 * 13 * 3 + 1,
        "size": 8,
        "frac": 10,
        "nN": 4,
        "nO": 6,
    }


def _chain_keys(
    fields: dict[str, NDArray[np.int64]],
    chain: tuple[tuple[str, ...], ...],
) -> list[NDArray[np.int64]]:
    keys = np.zeros(len(fields["ft"]), np.int64)
    levels: list[NDArray[np.int64]] = []
    for level in chain:
        for column in level:
            keys = keys * _cards()[column] + fields[column]
        levels.append(keys.copy())
    return levels


def _model_components(
    record: FragnetRecord,
    *,
    first: int,
    second: int,
) -> list[set[int]]:
    adjacency: list[list[int]] = [[] for _ in range(len(record["amass"]))]
    for edge in range(len(record["bonds"])):
        if edge not in (first, second):
            left, right = (record["bonds"].item((edge, side)) for side in range(2))
            adjacency[left].append(right)
            adjacency[right].append(left)
    seen: set[int] = set()
    groups: list[set[int]] = []
    for root in range(len(adjacency)):
        if root in seen:
            continue
        group = {root}
        pending = [root]
        seen.add(root)
        while pending:
            for child in adjacency[pending.pop()]:
                if child not in seen:
                    seen.add(child)
                    group.add(child)
                    pending.append(child)
        groups.append(group)
    return groups


def _fragment_row(
    record: FragnetRecord,
    *,
    group: set[int],
    first: int,
    second: int,
) -> tuple[float, int, int, int, int, int, int]:
    mass = 0.0
    nitrogen = oxygen = 0
    for atom in range(len(record["amass"])):
        if atom in group:
            mass += record["amass"].item(atom)
            nitrogen += int(record["el"].item(atom) == 1)
            oxygen += int(record["el"].item(atom) == 2)
    cuts: list[int] = []
    for edge in (first, second) if second >= 0 else (first,):
        left, right = (record["bonds"].item((edge, side)) for side in range(2))
        inside, outside = (left, right) if left in group else (right, left)
        cuts.append(
            (record["kind"].item(inside) * 13 + record["kind"].item(outside)) * 3
            + record["bring"].item(edge),
        )
    kind = 1 if second < 0 else 2 if record["bring"].item(first) > 0 else 3
    return (
        mass,
        kind,
        min(cuts),
        max(cuts) if second >= 0 else -1,
        len(group),
        nitrogen,
        oxygen,
    )


def _matches(
    masses: NDArray[np.float64],
    peaks: Peaks,
) -> tuple[
    NDArray[np.int8],
    NDArray[np.int32],
    NDArray[np.int64],
    NDArray[np.int32],
    NDArray[np.int8],
]:
    classes = np.zeros((len(masses), peaks.S), np.int8)
    best = np.full((len(masses), peaks.S), -1, np.int32)
    offsets = [0]
    match_peak: list[int] = []
    match_class: list[int] = []
    for fragment in range(len(masses)):
        mass = masses.item(fragment)
        for spectrum in range(peaks.S):
            ion = mass + peaks.sp_del.item((spectrum, 0))
            if ion < peaks.lo.item(spectrum) or ion > peaks.hi.item(spectrum):
                classes[fragment, spectrum] = -1
                continue
            absolute_best, weight_best = 99, -1.0
            start, stop = peaks.pk_off.item(spectrum), peaks.pk_off.item(spectrum + 1)
            for shift, category in ((0, 1), (-1, 2), (1, 3), (-2, 4), (2, 5)):
                for delta in range(peaks.sp_nd.item(spectrum)):
                    ion = (
                        mass
                        + peaks.sp_del.item((spectrum, delta))
                        + shift * HYDROGEN_MASS
                    )
                    tolerance = max(0.01, ion * 15.0 * 1e-6)
                    peak = start + int(
                        np.searchsorted(peaks.pk_mz[start:stop], ion - tolerance),
                    )
                    while peak < stop and peaks.pk_mz.item(peak) <= ion + tolerance:
                        weight = peaks.pk_w.item(peak)
                        if abs(shift) < absolute_best or (
                            abs(shift) == absolute_best and weight > weight_best
                        ):
                            absolute_best, weight_best = abs(shift), weight
                            classes[fragment, spectrum], best[fragment, spectrum] = (
                                category,
                                peak,
                            )
                        match_peak.append(peak)
                        match_class.append(category)
                        peak += 1
        offsets.append(len(match_peak))
    return (
        classes,
        best,
        np.array(offsets, np.int64),
        np.array(match_peak, np.int32),
        np.array(match_class, np.int8),
    )


def _mean_spectra(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = values[:, 0].copy()
    for spectrum in range(1, len(values.T)):
        result = result + values[:, spectrum]
    return result / len(values.T)


def _read_chains(raw: bytes) -> tuple[tuple[tuple[str, ...], ...], ...]:
    handle = io.BytesIO(raw)
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        _, _, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        _, _, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError("Unsupported StageA NumPy format.")
    if not dtype.hasobject:
        raise ValueError("StageA chains require an object array.")
    value = cast(object, ScientificDataUnpickler(handle).load())
    if not isinstance(value, np.ndarray):
        raise TypeError("StageA chains require a NumPy array.")
    array = cast("NDArray[np.object_]", value)
    chains: list[tuple[tuple[str, ...], ...]] = []
    for ci in range(len(array)):
        chain = cast("NDArray[np.object_]", array[ci])
        levels: list[tuple[str, ...]] = []
        for li in range(len(chain)):
            level = cast(object, chain[li])
            levels.append(
                tuple(from_plain(cast(object, np.asarray(level).tolist()), list[str])),
            )
        chains.append(tuple(levels))
    return tuple(chains)


def _energy(spectrum: SpectrumRequest) -> float:
    value = spectrum.get("ce", math.nan)
    if value is None:
        raise TypeError("Fragnet collision energy must be numeric.")
    return value
