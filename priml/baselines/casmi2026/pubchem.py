"""PubChem-scale candidate search, assets and screening."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    cast,
    override,
)

import logging
import math
import time

from configgle import Fig
from numpy.typing import NDArray
from treekle import from_plain, json_loads

import numpy as np

from priml.baselines.casmi2026.chemistry import CandidatePool
from priml.baselines.casmi2026.custom_types import PreparationHeader
from priml.baselines.casmi2026.dreams import MainBank
from priml.baselines.casmi2026.fusion import PLACEHOLDER, guarded_z, merge, promote
from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, neutral_mass
from priml.data.arrays import load_array
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pyarrow import parquet

    from priml.baselines.casmi2026.custom_types import SpectrumRequest
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")


logger = logging.getLogger(__name__)


type PackedVector = NDArray[np.uint8]
type Logits = NDArray[np.float32]
type Popularity = NDArray[np.float16 | np.float32 | np.float64]


class PubChemMetadata(TypedDict):
    """Keep independent parse, raw-fingerprint and score-key validity."""

    smiles: str
    parsed: bool
    raw_valid: bool
    key: str | None


class PubChemHeader(PreparationHeader):
    """Pin the logical fingerprint layouts independently of their packed byte counts."""

    raw_bits: int
    ecfp4_bits: int


class PubChemDiagnostics(TypedDict):
    """Retain every source diagnostic, including presence-sensitive early outputs."""

    molecule_id: str
    n_window: int
    n_pass1: int
    n_pass2: int
    n_keyed: int
    n_in_pool: int
    n_listed: int
    fz_top: float | None
    secs: float
    S: NotRequired[float]
    top_pop: NotRequired[float]


class PubChemResult(TypedDict):
    """Expose the complete public pc_runner mapping consumed by outer policy."""

    pc: list[str]
    pc_fz: list[float]
    pc_keys: list[str]
    fz_top: float | None
    S: float | None
    top_pop: float
    best_pool_fz: float
    target: float


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class PubChemRecord:
    """Retain every complete independent primitive for one requested SMILES row."""

    smiles: str
    parsed: bool
    raw_valid: bool
    key: str | None
    ecfp4: PackedVector
    raw_fp: PackedVector


class PubChemTable:
    """Require explicit prepared chemistry for every reached PubChem structure."""

    def __init__(self, directory: Path) -> None:
        header = json_loads(
            (directory / "manifest.json").read_text(),
            PubChemHeader,
        )
        if header["format_version"] != 1 or header["rdkit"] != IDENTITY_RDKIT:
            raise ValueError(
                f"PubChem data requires format 1 and RDKit {IDENTITY_RDKIT}.",
            )
        if header["raw_bits"] != 16_551 or header["ecfp4_bits"] != 4096:
            raise ValueError(
                "PubChem data requires complete 16551-bit raw and 4096-bit ECFP4 layouts.",
            )
        self.metadata = from_plain(
            parquet.read_table(
                directory / "molecules.parquet",
            ).to_pylist(),
            list[PubChemMetadata],
        )
        self.ecfp4 = _array(directory / "ecfp4.npy", dtype=np.uint8, dimensions=2)
        self.raw_fp = _array(directory / "raw_fp.npy", dtype=np.uint8, dimensions=2)
        if self.ecfp4.shape != (len(self.metadata), 512) or self.raw_fp.shape != (
            len(self.metadata),
            2069,
        ):
            raise ValueError(
                "PubChem complete fingerprint widths or rows differ from metadata.",
            )
        self._positions: dict[str, int] = {}
        for index, row in enumerate(self.metadata):
            self._positions.setdefault(row["smiles"], index)

    def row(self, index: int) -> PubChemRecord:
        """Read the complete original positional row, including invalid structures.

        Args:
          index: Original nonnegative prepared row position.

        Returns:
          record: Complete readonly fingerprints and three independent validity fields.

        """
        if index < 0 or index >= len(self.metadata):
            raise IndexError(index)
        return PubChemRecord(
            **self.metadata[index],  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.
            ecfp4=self.ecfp4[index : index + 1].reshape(-1),
            raw_fp=self.raw_fp[index : index + 1].reshape(-1),
        )

    def for_smiles(self, smiles: str) -> PubChemRecord:
        """Read the first explicit prepared row; missing SMILES raise KeyError."""
        return self.row(self._positions[smiles])


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class PubChemStore:
    """Map complete original mass/ASCII stores and optional aligned popularity data."""

    mass: NDArray[np.float64]
    offsets: NDArray[np.int64]
    buffer: PackedVector
    ls: Popularity | None = None
    lp: Popularity | None = None

    @classmethod
    def load(cls, directory: Path, *, popularity: Path | None = None) -> PubChemStore:
        """Map the original immutable numeric assets without molecular imports.

        Args:
          directory: Original pc_mass/pc_off/pc_smiles NumPy asset directory.
          popularity: Optional original aligned log-SID/log-PMID array directory.

        Returns:
          store: Complete readonly memory maps in their original dtypes.

        """
        mass = _array(directory / "pc_mass.npy", dtype=np.float64, dimensions=1)
        offsets = _array(directory / "pc_off.npy", dtype=np.int64, dimensions=1)
        buffer = _array(directory / "pc_smiles.npy", dtype=np.uint8, dimensions=1)
        if (
            offsets.shape != (len(mass) + 1,)
            or offsets.item(0) != 0
            or offsets.item(-1) != len(buffer)
        ):
            raise ValueError(
                "PubChem offsets differ from the complete mass and ASCII layout.",
            )
        ls = lp = None
        if popularity is not None:
            ls, lp = (
                _popularity(popularity / filename, rows=len(mass))
                for filename in ("pc_lsid.npy", "pc_lpmid.npy")
            )
        return cls(mass=mass, offsets=offsets, buffer=buffer, ls=ls, lp=lp)

    def window(self, target: float, *, ppm: float) -> tuple[int, tuple[str, ...]]:
        """Decode the entire inclusive mass window, retaining duplicates.

        Args:
          target: Neutral mass before the source relative tolerance calculation.
          ppm: Window half-width in parts per million of the target.

        Returns:
          begin: First source row inside the window.
          smiles: Complete source ASCII structures in positional order.

        """
        tolerance = target * ppm * 1e-6
        begin = int(np.searchsorted(self.mass, target - tolerance, side="left"))
        end = int(np.searchsorted(self.mass, target + tolerance, side="right"))
        if end <= begin:
            return begin, ()
        offsets = np.asarray(self.offsets[begin : end + 1]).astype(np.int64)
        if (
            offsets[0] < 0
            or offsets[-1] > len(self.buffer)
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError("PubChem window offsets exceed the declared ASCII layout.")
        raw = bytes(self.buffer[offsets[0] : offsets[-1]])
        base = offsets.item(0)
        return begin, tuple(
            raw[offsets.item(index) - base : offsets.item(index + 1) - base].decode(
                "ascii",
            )
            for index in range(end - begin)
        )


class PubChemScreen:
    """Run complete partial/full fingerprint screening and the popularity prior."""

    class Config(Fig["PubChemScreen"]):
        pass_one: int = 5_000
        """Partial ECFP4 screening geometry before full fingerprint evaluation."""

        popularity_weight: float = 0.25
        """Source standardized-fingerprint plus popularity coefficient."""

        popularity_union: int = 200
        """Additional popularity leaders unioned into the first-stage rows."""

        limit: int = 25
        """Maximum distinct non-pool score keys, without padding."""

        max_spectra: int = 16
        """Original raw-spectrum cap, after target neutral-mass aggregation."""

        ppm: float = 10.0
        """Mass-window half-width of the store and of the first pool-fit window."""

        fallback_ppm: float = 30.0
        """Mass-window half-width of the pool fit when the first window is empty."""

    def __init__(self, config: Config) -> None:
        if config.pass_one < 1 or config.limit < 1 or config.max_spectra < 1:
            raise ValueError("PubChem screening counts must be positive.")
        if config.popularity_union < 0:
            raise ValueError("PubChem popularity union must be nonnegative.")
        self.config = config

    def probe(
        self,
        molecule_id: str,
        target: float,
        logits: Logits,
        *,
        store: PubChemStore,
        chemistry: PubChemTable,
        bits: NDArray[np.int64],
        pool_keys: set[str],
        clock: Callable[[], float] = time.perf_counter,
    ) -> tuple[str, list[str], list[float], list[str], PubChemDiagnostics]:
        """Evaluate every source stage without a molecular runtime dependency.

        Args:
          molecule_id: Source query identity retained in diagnostics.
          target: Target neutral mass for the full PubChem window.
          logits: Complete selected-bit float32 bank prediction.
          store: Original mass, ASCII and optional popularity arrays.
          chemistry: Independent prepared partial/raw/key observations.
          bits: All selected source fingerprint-bit positions in model order.
          pool_keys: Complete candidate-pool exclusion identities.
          clock: Monotonic elapsed-time primitive, injectable for complete replay.

        Returns:
          molecule_id: Original query identity.
          smiles: Distinct non-pool structures in actual score order.
          scores: Complete standardized fingerprint plus popularity scores.
          keys: Source score keys aligned with the returned structures.
          diagnostics: Every source count, score gate and elapsed time.

        """
        start = clock()
        begin, smiles = store.window(target, ppm=self.config.ppm)
        diagnostics: PubChemDiagnostics = {
            "molecule_id": molecule_id,
            "n_window": len(smiles),
            "n_pass1": 0,
            "n_pass2": 0,
            "n_keyed": 0,
            "n_in_pool": 0,
            "n_listed": 0,
            "fz_top": None,
            "secs": 0.0,
        }
        output: list[str] = []
        scores: list[float] = []
        keys: list[str] = []
        output_popularity: list[float] = []
        records = [chemistry.for_smiles(value) for value in smiles]
        positions = (
            _partial_rows(
                records,
                bits=bits,
                logits=logits,
                maximum=self.config.pass_one,
            )
            if records
            else np.zeros(0, np.int64)
        )
        diagnostics["n_pass1"] = min(
            self.config.pass_one,
            sum(row.parsed for row in records),
        )
        if len(positions):
            popularity = np.zeros(len(smiles), np.float32)
            use_popularity = store.ls is not None and self.config.popularity_weight > 0
            if use_popularity:
                if store.ls is None or store.lp is None:
                    raise ValueError(
                        "Expected store.ls is not None and store.lp is not None.",
                    )
                popularity = np.asarray(
                    store.ls[begin : begin + len(smiles)],
                    np.float32,
                ) + np.asarray(store.lp[begin : begin + len(smiles)], np.float32)
                if self.config.popularity_union > 0:
                    partial_valid = np.array([row.parsed for row in records], np.bool_)
                    population_scores = np.where(partial_valid, popularity, -np.inf)
                    count = min(self.config.popularity_union, int(partial_valid.sum()))
                    positions = np.union1d(
                        positions,
                        np.argsort(-population_scores, kind="stable")[:count],
                    )
            full_scores: list[float] = []
            full_positions: list[int] = []
            for index in positions:
                record = records[index]
                if record.raw_valid:
                    fingerprint = np.unpackbits(record.raw_fp)[:16_551]
                    full_scores.append(
                        float(
                            np.float32(fingerprint[bits] @ logits.astype(np.float64)),
                        ),
                    )
                    full_positions.append(int(index))
            diagnostics["n_pass2"] = len(full_positions)
            if full_positions:
                fingerprint_scores = np.array(full_scores, np.float64)
                indices = np.array(full_positions, np.int64)
                prepared_keys = tuple(records[index].key for index in full_positions)
                key_cache: dict[int, str | None] = {}
                for index in _order(fingerprint_scores):
                    key = _key(
                        index,
                        keys=prepared_keys,
                        cache=key_cache,
                        diagnostics=diagnostics,
                    )
                    if key is not None and key not in pool_keys:
                        diagnostics["fz_top"] = float(fingerprint_scores.item(index))
                        break
                ranked_scores = (
                    guarded_z(fingerprint_scores)
                    + self.config.popularity_weight * popularity[indices]
                )
                seen: set[str] = set()
                for index in _order(ranked_scores):
                    key = _key(
                        index,
                        keys=prepared_keys,
                        cache=key_cache,
                        diagnostics=diagnostics,
                    )
                    if key is None or key in seen:
                        continue
                    seen.add(key)
                    if key in pool_keys:
                        diagnostics["n_in_pool"] += 1
                        continue
                    source_row = indices.item(index)
                    output.append(smiles[source_row])
                    scores.append(float(ranked_scores.item(index)))
                    keys.append(key)
                    output_popularity.append(float(popularity.item(source_row)))
                    if len(output) >= self.config.limit:
                        break
                diagnostics["n_listed"] = len(output)
                diagnostics["S"] = scores[0] if scores else 0.0
                diagnostics["top_pop"] = (
                    output_popularity[0] if output_popularity else 0.0
                )
        diagnostics["secs"] = clock() - start
        return molecule_id, output, scores, keys, diagnostics

    def run(
        self,
        molecule_id: str,
        spectra: Sequence[SpectrumRequest],
        *,
        store: PubChemStore,
        chemistry: PubChemTable,
        pool: CandidatePool,
        bank: MainBank,
    ) -> PubChemResult | None:
        """Return the complete source channel result or a skipped invalid query.

        Args:
          molecule_id: Query identity.
          spectra: All raw query spectra with prepared scalar CE and ce_n metadata.
          store: Original complete PubChem mass/string/popularity arrays.
          chemistry: Explicit independent PubChem chemistry staging.
          pool: The complete provided main candidate pool and selected bit order.
          bank: The exact two released V3 A/B fingerprint models.

        Returns:
          result: Complete pc_runner fields, or None for skipped source queries.

        """
        if len(bank.models) != 2:
            raise ValueError("PubChem requires exactly two V3 fingerprint models.")
        masses = neutral_mass(
            np.array([row["prec"] for row in spectra], np.float64),
            [row["adduct"] for row in spectra],
        )
        masses = masses[np.isfinite(masses)]
        if not len(masses):
            return None
        target = float(np.median(masses))
        try:
            logits = molecule_logits(bank, spectra[: self.config.max_spectra])
        except Exception:
            logger.exception(
                "PubChem fingerprint prediction failed for %s.",
                molecule_id,
            )
            return None
        if logits is None:
            return None
        best = best_pool_fz(
            pool,
            target=target,
            logits=logits,
            ppm=self.config.ppm,
            fallback_ppm=self.config.fallback_ppm,
        )
        _, output, scores, keys, diagnostics = self.probe(
            molecule_id,
            target,
            logits,
            store=store,
            chemistry=chemistry,
            bits=pool.bits,
            pool_keys=set(pool.keys),
        )
        return {
            "pc": output,
            "pc_fz": scores,
            "pc_keys": keys,
            "fz_top": diagnostics["fz_top"],
            "S": diagnostics.get("S"),
            "top_pop": diagnostics.get("top_pop", 0.0),
            "best_pool_fz": best,
            "target": target,
        }


def first_modal_adduct(adducts: Sequence[str]) -> str:
    """Pick the most frequent label; ties keep the first one encountered, as V3 does."""
    return max(adducts, key=adducts.count)


def molecule_logits(
    bank: MainBank,
    spectra: Sequence[SpectrumRequest],
) -> Logits | None:
    """Average both source bank view populations with equal population weight.

    Args:
      bank: Complete source-compatible model bank in checkpoint order.
      spectra: Raw source spectra before fingerprint preparation.

    Returns:
      logits: Source float32 selected-bit prediction, or None for no spectra.

    """
    single, merged = bank.items(spectra)
    if not single:
        return None
    single_logits, _ = bank.predict(single)
    merged_logits, _ = bank.predict(merged)
    return (0.5 * (single_logits.mean(0) + merged_logits.mean(0))).astype(np.float32)


def best_pool_fz(
    pool: CandidatePool,
    *,
    target: float,
    logits: Logits,
    ppm: float,
    fallback_ppm: float,
) -> float:
    """Use complete F64 fingerprints/logits inside the ppm then fallback bounds.

    Args:
      pool: Complete original mass-sorted pool and packed fingerprints.
      target: Source median finite neutral mass.
      logits: Complete selected-bit fingerprint logits.
      ppm: Half-width of the first window in parts per million.
      fallback_ppm: Half-width of the window used when the first is empty.

    Returns:
      best: Maximum source f.z, or NaN when both windows are empty.

    """
    for window in (ppm, fallback_ppm):
        tolerance = target * (window / 1e6)
        begin = int(np.searchsorted(pool.mass, target - tolerance, side="left"))
        end = int(np.searchsorted(pool.mass, target + tolerance, side="right"))
        if end > begin:
            fingerprints = np.unpackbits(np.asarray(pool.packed[begin:end]), axis=1)[
                :,
                : len(pool.bits),
            ].astype(np.float64)
            return float((fingerprints @ logits.astype(np.float64)).max())
    return float("nan")


def _array[S: np.generic](path: Path, *, dtype: type[S], dimensions: int) -> NDArray[S]:
    value = load_array(path, dtype, mmap=True)
    if value.ndim != dimensions:
        raise ValueError(f"PubChem array {path.name} has an invalid rank.")
    return value


def _popularity(path: Path, *, rows: int) -> Popularity:
    value = load_array(path, np.floating, mmap=True)
    if value.shape != (rows,) or value.dtype.itemsize > 8:
        raise ValueError(
            "PubChem popularity shape or floating dtype differs from its mass rows.",
        )
    return cast("Popularity", value)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class PubChemInputs:
    """Carry the complete independent V3 channel inputs as one immutable binding."""

    pool: CandidatePool
    store: PubChemStore
    chemistry: PubChemTable
    bank: MainBank


class PubChemAssetProvider(Protocol):
    """Load every declared input without a supplier or chemistry runtime."""

    def read(self) -> PubChemInputs:
        """Return the complete strict input binding or propagate staging failures."""
        ...


class PubChemAssets:
    """Read the original PC store, prepared chemistry, pool and exact V3 bank."""

    class Config(Fig["PubChemAssets"]):
        base_dir: Path | str | None = None
        """Parent-supplied dataset root; every relative path below resolves beneath it."""

        pool_dir: Path | str = "inputs/datasets/ahmedberatozer/casmi26-v2-pool"
        """Complete provided main candidate pool, packed bits and fragments."""

        store_dir: Path | str = "inputs/datasets/ahmedberatozer/casmi26-pubchem-tier"
        """Original complete PC mass, offset and ASCII arrays."""

        popularity_dir: Path | str = (
            "inputs/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior"
        )
        """Original aligned F16 log-SID and log-PMID maps."""

        chemistry_dir: Path | str = "native-chemistry/pubchem"
        """Complete independently prepared partial/full fingerprints and raw keys."""

        bank: MainBank.Config = field(default_factory=MainBank.Config)
        """Exactly the two pinned released V3 fingerprint checkpoints."""

        @override
        def finalize(self) -> Self:
            self.pool_dir = resolve_working_dir(self.base_dir, self.pool_dir)
            self.store_dir = resolve_working_dir(self.base_dir, self.store_dir)
            self.popularity_dir = resolve_working_dir(
                self.base_dir,
                self.popularity_dir,
            )
            self.chemistry_dir = resolve_working_dir(self.base_dir, self.chemistry_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def read(self) -> PubChemInputs:
        """Read all complete native PC data and load every pinned model tensor.

        Returns:
          inputs: Complete original arrays, explicit chemistry and both native models.

        """
        config = self.config
        return PubChemInputs(
            pool=CandidatePool.load(Path(config.pool_dir)),
            store=PubChemStore.load(
                Path(config.store_dir),
                popularity=Path(config.popularity_dir),
            ),
            chemistry=PubChemTable(Path(config.chemistry_dir)),
            bank=config.bank.make(),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScreenedCandidates:
    """Retain the complete public shortlist and its merge counters."""

    candidates: tuple[str, ...]
    mode: Literal["no_pc", "untouched", "gentle", "aggressive"]
    promoted: bool = False


class PubChemGate:
    """Gate complete PubChem proposals against the library score and fit margins."""

    class Config(Fig["PubChemGate"]):
        library_gate: float = 0.9
        """Direct-library score at or above which PubChem proposals are not admitted."""

        margin_aggressive: float = 600.0
        """Fit margin over the best pool candidate above which the tight slots apply."""

        margin_promote: float = 250.0
        """Fit margin over the best pool candidate above which the first proposal leads."""

        outlier_promote: float = 6.0
        """Standardized fit plus popularity above which the first proposal leads."""

        popularity_promote: float = 5.0
        """Top-proposal popularity at or above which a fit outlier leads the list."""

        slots_aggressive: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 10])
        """One-based insertion slots for proposals when the margin is aggressive."""

        slots_gentle: list[int] = field(default_factory=lambda: [4, 8, 12, 16, 20])
        """One-based insertion slots for proposals when the margin is gentle."""

        limit: int = 25
        """Maximum length of the merged shortlist."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def screen(
        self,
        base: Sequence[str],
        base_keys: Sequence[str],
        *,
        library_max: float = 0.0,
        proposals: Sequence[str] = (),
        proposal_keys: Sequence[str] = (),
        proposal_scores: Sequence[float] = (),
        top_fit: float | None = None,
        pool_fit: float = math.nan,
        outlier: float | None = None,
        popularity: float | None = None,
    ) -> ScreenedCandidates:
        """Gate complete PubChem proposals and preserve the source's independent margins.

        Args:
          base: Ranked main-arm candidates before PubChem admission.
          base_keys: Aligned main-arm keys used to reject duplicate proposals.
          library_max: Maximum direct-library score, protecting values at least the gate.
          proposals: Complete ranked PubChem-only shortlist.
          proposal_keys: Aligned older tautomer keys.
          proposal_scores: Complete source pc_fz values, used only when top_fit is absent.
          top_fit: Original best non-pool raw fingerprint fit, including explicit zero.
          pool_fit: Best main-pool fit; nonfinite values select margin 1e9.
          outlier: Source standardized fit plus popularity, distinct from raw top_fit.
          popularity: Source top-proposal popularity; absent values behave as zero.

        Returns:
          result: Complete capped shortlist, gate mode and promotion counter.

        """
        config = self.config
        promoted = False
        if not proposals:
            mode, final = "no_pc", list(base)
        elif library_max >= config.library_gate:
            mode, final = "untouched", list(base)
        else:
            margin = (
                (top_fit if top_fit is not None else proposal_scores[0]) - pool_fit
                if math.isfinite(pool_fit)
                else 1e9
            )
            aggressive = margin > config.margin_aggressive
            mode = "aggressive" if aggressive else "gentle"
            promoted = (
                outlier is not None
                and outlier > config.outlier_promote
                and (popularity or 0.0) >= config.popularity_promote
            ) or margin > config.margin_promote
            final = (promote if promoted else merge)(
                base,
                base_keys,
                proposals,
                proposal_keys,
                slots=config.slots_aggressive if aggressive else config.slots_gentle,
                limit=config.limit,
            )
        return ScreenedCandidates(
            candidates=tuple(final[: config.limit]) or (PLACEHOLDER,),
            mode=mode,
            promoted=promoted,
        )


def _partial_rows(
    records: Sequence[PubChemRecord],
    *,
    bits: NDArray[np.int64],
    logits: Logits,
    maximum: int,
) -> NDArray[np.int64]:
    selected = np.where(bits < 4096)[0]
    raw = bits[selected]
    fingerprints = np.zeros((len(records), len(raw)), np.float32)
    valid = np.array([row.parsed for row in records], np.bool_)
    for index, row in enumerate(records):
        if row.parsed:
            fingerprints[index] = np.unpackbits(row.ecfp4)[raw]
    scores = (fingerprints @ logits[selected].astype(np.float64)).astype(np.float32)
    scores[~valid] = -np.inf
    count = min(maximum, int(valid.sum()))
    if not count:
        return np.zeros(0, np.int64)
    return (
        np.argsort(-scores, kind="stable")[:count]
        if count < len(records)
        else np.where(valid)[0]
    )


def _key(
    index: int,
    *,
    keys: Sequence[str | None],
    cache: dict[int, str | None],
    diagnostics: PubChemDiagnostics,
) -> str | None:
    if index not in cache:
        cache[index] = keys[index]
        diagnostics["n_keyed"] += 1
    return cache[index]


def _order(values: NDArray[np.float64]) -> list[int]:
    return from_plain(
        cast(object, np.argsort(-values, kind="stable").tolist()),
        list[int],
    )
