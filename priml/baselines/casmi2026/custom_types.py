"""Typed scientific-data records and aligned candidate records shared across packages.

Torch-free: the RDKit preparation interpreter imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict


if TYPE_CHECKING:
    from numpy.typing import NDArray

    import numpy as np


type PredictionMatrix = np.ndarray[tuple[int, int], np.dtype[np.float32 | np.float64]]
type QueryMasses = NDArray[np.float64]
type GapWeights = NDArray[np.float64]
type GapBonds = NDArray[np.int64]
type MetricIdentity = tuple[str, tuple[tuple[str, int], ...]]
type FloatArray = NDArray[np.float32] | NDArray[np.float64]
type PoolIndices = NDArray[np.int64]
type GenerationFlags = NDArray[np.bool_]
type CandidateMatrix = NDArray[np.float32]
type RankingScores = NDArray[np.float64]


class IdentityRow(TypedDict):
    """Retain independently reached raw and standardized molecular decisions."""

    smiles: str
    standardized: str | None
    key: str | None
    formula: str
    metric: str
    standard_key: str | None
    standard_metric: str
    standard_formula_lookup: str
    properties_valid: bool
    standard_formula: str
    standard_mass: float
    standard_heavy: int
    fingerprint_valid: bool


class PreparationHeader(TypedDict):
    """Required scientific format and library version in a preparation manifest."""

    format_version: int
    rdkit: str


class EngineIdentityRow(TypedDict):
    """Preserve parsed empty engine keys separately from missing or failed keys."""

    smiles: str
    key: str | None


class ReactionRequest(TypedDict):
    """One explicitly staged mass-conditioned plan or rule application."""

    smiles: str
    delta: float
    operation: NotRequired[Literal["match", "apply", "derive"]]
    rule: NotRequired[int]
    tolerance: NotRequired[float]
    max_steps: NotRequired[int]
    max_products: NotRequired[int]
    allow_isomerization: NotRequired[bool]
    application_budget: NotRequired[int]
    stage_parents: NotRequired[int]
    first_products: NotRequired[int]
    later_products: NotRequired[int]


class ReactionHeader(TypedDict):
    """Complete request values, including every default and enumeration budget."""

    smiles: str
    delta: float
    operation: Literal["match", "apply", "derive"]
    rule: int
    tolerance: float
    max_steps: int
    max_products: int
    allow_isomerization: bool
    application_budget: int
    stage_parents: int
    first_products: int
    later_products: int


class ReactionProduct(TypedDict):
    """One complete ordered raw product and its first matching rule combination."""

    raw: str
    combo: tuple[str, ...] | None


class SimulatorAdmissionRow(TypedDict):
    """Keep both distinct cheap decisions before either simulator featurizer."""

    smiles: str
    ice_accepted: bool
    ice_reason: str
    glacier_accepted: bool
    glacier_reason: str
    glacier_canonical: str | None
    glacier_heavy_atoms: int


class AnalogMetadata(TypedDict):
    """Ordered molecular identities accompanying the complete structural arrays."""

    smiles: str
    formula_input: str | None
    core_smiles: str
    element_smiles: str
    query_smiles: str


class FragnetRecord(TypedDict):
    """Every ordered native molecular field consumed by the Fragnet family."""

    amass: NDArray[np.float64]
    kind: NDArray[np.int8]
    el: NDArray[np.int8]
    nh: NDArray[np.int8]
    bonds: NDArray[np.int32]
    bring: NDArray[np.int8]
    mass: float
    el10: NDArray[np.uint8]


class FragnetMetadata(TypedDict):
    """Declared input, parsing and family admission for each prepared molecule."""

    smiles: str
    parsed: bool
    valid: bool


class ProposalPath(TypedDict):
    """One terminal chemical path before native site-prior deduplication."""

    raw: str | None
    standardized: str | None
    key: str | None
    skeleton: str | None
    environments: tuple[tuple[int, int, int], ...]


class ProposalRow(ProposalPath):
    """Ordered terminal path and its explicit request-table position."""

    request: int
    position: int


class ProposalHeader(TypedDict):
    """Complete requested pair, all enumeration limits and actual call provenance."""

    parent_smiles: str
    combo: tuple[str, ...]
    max_first: int
    max_second: int
    max_mols: int
    terminal_count: int
    applications: str


class MolecularIdentity(TypedDict):
    """Explicitly staged score key and element/connectivity skeleton."""

    smiles: str
    key: str | None
    skeleton: str | None


class AtomicRow(TypedDict):
    """Complete sparse heavy-atom graph before native neural preprocessing."""

    input_smiles: str
    smiles: str
    valid: bool
    features: list[list[float]]
    bonds: list[list[int]]
    bond_features: list[list[float]]
    weights: list[float]
    degrees: list[int]


class IceRow(TypedDict):
    """Complete canonical ICE root graph and isotope-aware constructor data."""

    input_smiles: str
    smiles: str
    valid: bool
    features: list[list[float]]
    bonds: list[list[int]]
    bond_orders: list[int]
    formula: list[float]
    hydrogens: list[int]
    atom_masses: list[float]
    hydrogenated_masses: list[float]
    mass: float


class GlacierRow(TypedDict):
    """Complete physical GLACIER data aligned with the canonical atomic graphs."""

    input_smiles: str
    smiles: str
    valid: bool
    formula: list[float]
    atom_formulas: list[list[float]]
    hydrogens: list[int]
    atom_masses: list[float]
    hydrogenated_masses: list[float]
    total_hydrogens: int


class FragmentMassRow(TypedDict):
    """Complete ordered atom-weight graph used by the second engine's fragments."""

    smiles: str
    valid: bool
    weights: list[float]
    bonds: list[list[int]]


class GapSpectrum(TypedDict):
    """Retain raw peak and carrier fields for the final fragment-gap fallback."""

    mz: list[float]
    it: list[float]
    adduct: str | None
    prec: float
    mode: int | None


class GapGraphRow(FragmentMassRow):
    """Retain intact component count for the distinct final-gap mass graph."""

    pieces: int


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class GapGraph:
    """Retain complete ordered gap weights, bonds and intact component count."""

    weights: GapWeights
    bonds: GapBonds
    pieces: int


class SpectrumRequest(TypedDict):
    """Original query fields needed by the two public forward-simulator panels."""

    mz: list[float]
    it: list[float]
    prec: float
    mode: int | str
    adduct: str
    instrument: NotRequired[str | float | None]
    instr: NotRequired[str | None]
    ce: NotRequired[float | None]
    ce_n: NotRequired[int]
    ce_units: NotRequired[str | None]
    ce_orig: NotRequired[str | None]
    ce_ev: NotRequired[list[float] | None]


class QueryRow(TypedDict):
    """Retain each competition spectrum's original columns before conditioning."""

    molecule_id: str | int
    spectrum_id: str | int
    precursor_mz: float
    adduct: str
    ionization_mode: str | None
    collision_energy_ev: list[float | None] | None
    collision_energy_orig_units: str | None
    collision_energy_orig: str | None
    instrument_type: str | None
    ms2_mzs: list[float]
    ms2_normalized_intensities: list[float]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CandidateSet:
    """Keep candidate positions, including repeated structures and identities."""

    smiles: tuple[str, ...]
    keys: tuple[str, ...]
    formulas: tuple[str, ...]
    pool_rows: PoolIndices
    generated: GenerationFlags
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineDiagnostics:
    """Retain the first engine's source diagnostics used by the final overlays."""

    n_cand: int
    lib_max: float
    top_sim: float
    fzmax: float
    n_gen: int = 0


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class EngineResult:
    """Bind complete candidate-aligned model features and final ranking scores."""

    candidates: CandidateSet
    features: CandidateMatrix
    feature_names: tuple[str, ...]
    scores: RankingScores
    diagnostics: EngineDiagnostics


@dataclass(frozen=True, slots=True, kw_only=True)
class Analog:
    """Preserve the strict-best spectrum's identity, score and mass-shift metadata."""

    sid: int
    similarity: float
    shift: float = 0.0
    natural_product: bool = False
