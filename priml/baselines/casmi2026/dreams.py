"""DreaMS embedding store and the complete released neural checkpoint bank."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, TypedDict, cast, override

import math

from configgle import Fig, Makeable
from numpy.typing import NDArray
from torch import Tensor
from treekle import from_plain

import numpy as np
import torch

from priml.baselines.casmi2026.chemistry import merge_main
from priml.baselines.casmi2026.custom_types import PoolIndices, RankingScores
from priml.baselines.casmi2026.fusion import safe_extend
from priml.baselines.casmi2026.isolated import (
    ScientificDataUnpickler,
    adduct_index,
)
from priml.baselines.casmi2026.model import (
    DreamAttention,
    DreamsFP,
    FPNet,
    PeakAttention,
    ReleasedModel,
)
from priml.baselines.casmi2026.peaks import pack_inputs, prepare_peaks
from priml.model.ae.checkpoint import verify_sha256
from priml.model.transformer.block import TransformerBlock
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from priml.baselines.casmi2026.custom_types import SpectrumRequest


type PeakVector = NDArray[np.float32]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class BankItem:
    """Retain one prepared spectrum and every Full1/A/B/D conditioning field."""

    mass: PeakVector
    intensity: PeakVector
    precursor: float
    adduct: int
    energy: float
    energy_known: float
    merged_count: int
    mode: float


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ModelViews:
    """Keep model, spectrum and output axes before any bank averaging."""

    single: PeakVector
    single_elements: PeakVector
    merged: PeakVector
    merged_elements: PeakVector
    modes: PeakVector


def merged_adduct(adducts: Sequence[str]) -> str:
    """Break majority-adduct ties by full training frequency, then lexicographically.

    Args:
      adducts: One or more source labels, retaining repeated observations.

    Returns:
      adduct: The v6 deterministic merged-view label.

    """
    frequencies = {
        "[M+H]+": 1_324_585,
        "[M-H]-": 465_688,
        "[2M+Na]+": 190_033,
        "[M+Na]+": 178_092,
        "[2M+H]+": 92_041,
        "[M+CH2O2-H]-": 83_188,
        "[2M-H]-": 56_172,
        "[M+NH4]+": 53_452,
        "[M+Cl]-": 27_511,
        "[M-H2O+H]+": 21_682,
        "[M+K]+": 18_955,
        "[M]+": 8_818,
        "[M+C2H4O2-H]-": 7_436,
        "[M-2H2O+H]+": 3_439,
        "[M+2H]2+": 1_717,
        "[2M+Na-2H]-": 1_431,
        "[2M+CH2O2-H]-": 665,
        "[M-H2O]+": 624,
        "[M-CH3]-": 619,
        "[M-2H]-": 454,
        "[M]-": 446,
        "[2M+NH4]+": 378,
        "[M-H2O-H]-": 285,
        "[M-H5O3]+": 230,
        "[2M+C2H4O2-H]-": 187,
        "[M+C2H4N]+": 158,
        "[M+3H]3+": 138,
        "[M+2Na-H]+": 110,
        "[2M+K]+": 109,
        "[M-H2]2-": 103,
        "[M+Ca]2+": 85,
        "[M+CH5O]+": 67,
        "[M-H2N]+": 61,
        "[M+HNa]2+": 44,
        "[M+Li]+": 6,
    }
    counts = {name: adducts.count(name) for name in adducts}
    maximum = max(counts.values())
    return min(
        (name for name, count in counts.items() if count == maximum),
        key=lambda name: (-frequencies.get(name, 0), name),
    )


def clamp_count(count: int) -> int:
    """Raise an unrecorded or zero merge count to one, as the v6 views do."""
    return max(1, count)


def keep_count(count: int) -> int:
    """Retain the recorded merge count, as the Full1 bank does."""
    return count


def modal_adduct(adducts: Sequence[str]) -> str:
    """Pick the most frequent label; ties follow set order, as the Full1 bank does."""
    return max(set(adducts), key=adducts.count)


class BankCheckpoint:
    """Load one released checkpoint into the architecture its config declares."""

    class Config(Fig["BankCheckpoint"]):
        base_dir: Path | str | None = None
        """Parent-supplied resource root."""

        path: Path | str = Path()
        """Complete public FPNet or DreamsFP checkpoint, beneath ``base_dir``."""

        sha256: str = ""
        """Pinned digest checked before loading tensors."""

        model: Makeable[ReleasedModel] = field(default_factory=FPNet.Config)
        """Released architecture; its widths must equal those the checkpoint records."""

        dtype_autocast: torch.dtype | None = None
        """CUDA autocast precision of the forward pass; None keeps the model's own."""

        @override
        def finalize(self) -> Self:
            self.path = resolve_working_dir(self.base_dir, self.path)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if not config.sha256:
            raise ValueError("A bank checkpoint requires its SHA256 pin.")
        verify_sha256(Path(config.path), config.sha256)
        checkpoint = from_plain(
            cast(
                object,
                torch.load(Path(config.path), map_location="cpu", weights_only=True),
            ),
            dict[str, object],
        )
        state = from_plain(checkpoint["model"], dict[str, Tensor])
        architecture = from_plain(checkpoint.get("arch", "fpnet"), str)
        declared = config.model
        if isinstance(declared, FPNet.Config) and architecture == "fpnet":
            assert isinstance(declared.block.attn, PeakAttention.Config)
            recorded = (
                from_plain(checkpoint["nbits"], int),
                from_plain(checkpoint["d"], int),
                from_plain(checkpoint["layers"], int),
                from_plain(checkpoint.get("heads", 8), int),
            )
            if recorded != (
                declared.channels_out,
                declared.channels_hidden,
                declared.num_layers,
                declared.block.attn.attention.num_heads,
            ):
                raise ValueError(
                    f"Checkpoint geometry (bits, width, layers, heads) {recorded} "
                    "differs from the declared FPNet.",
                )
        elif isinstance(declared, DreamsFP.Config) and architecture == "dreams_fp":
            _verify_dreams(
                declared,
                checkpoint,
                dtype=config.dtype_autocast or torch.float32,
            )
        else:
            raise ValueError(
                f"Unknown neural checkpoint architecture {architecture!r} for "
                f"{type(declared).__name__}.",
            )
        model = declared.make()
        assert isinstance(model, FPNet | DreamsFP)
        model.load_released_state(state)
        self.model = model.eval()
        self.dtype_autocast = config.dtype_autocast


class MainBank:
    """Load complete released checkpoints and retain per-model and mean predictions."""

    class Config(Fig["MainBank"]):
        checkpoints: list[BankCheckpoint.Config] = field(
            default_factory=list[BankCheckpoint.Config],
        )
        """Ordered complete Full1 or A/B/D checkpoints."""

        device: str = "cpu"
        """Torch model and packed-input device."""

        batch_size: int = 64
        """Item count per independent forward chunk."""

        max_peaks: int = 160
        """Intensity-ranked peaks kept per query spectrum."""

        floor: float = 0.001
        """Relative intensity cutoff of a query peak after precursor filtering."""

        merge_tolerance: float = 0.005
        """Strict mass distance below which merged peaks count as duplicates."""

        merged_adduct: Callable[[Sequence[str]], str] = merged_adduct
        """Choose a merged polarity item's adduct from its members' labels."""

        single_count: Callable[[int], int] = clamp_count
        """Map each single spectrum's recorded merge count to its model input."""

    def __init__(self, config: Config) -> None:
        if not config.checkpoints:
            raise ValueError("A neural bank requires at least one checkpoint.")
        if config.batch_size < 1:
            raise ValueError("The neural-bank batch size must be positive.")
        self.config = config
        checkpoints = [checkpoint.make() for checkpoint in config.checkpoints]
        self.models = tuple(
            checkpoint.model.to(config.device) for checkpoint in checkpoints
        )
        self.dtypes = tuple(checkpoint.dtype_autocast for checkpoint in checkpoints)
        self.nbits = self.models[0].released_state()["head.3.bias"].numel()
        if any(
            model.released_state()["head.3.bias"].numel() != self.nbits
            for model in self.models
        ):
            raise ValueError("Neural-bank checkpoints disagree on fingerprint width.")

    @torch.no_grad()
    def predict(
        self,
        items: Sequence[BankItem],
        *,
        each: bool = False,
    ) -> tuple[PeakVector, PeakVector]:
        """Return all logits and element predictions, averaging models only on request.

        Args:
          items: Prepared query items in their original encounter order.
          each: Retain the leading model axis instead of averaging it.

        Returns:
          logits: Float32 fingerprint predictions, with an optional model axis.
          elements: Float32 ten-element predictions with the same leading axes.

        """
        count = len(self.models)
        if not items:
            leading = (count, 0) if each else (0,)
            return np.zeros((*leading, self.nbits), np.float32), np.zeros(
                (*leading, 10),
                np.float32,
            )
        logits: list[PeakVector] = []
        elements: list[PeakVector] = []
        for start in range(0, len(items), self.config.batch_size):
            arguments = collate_main(
                items[start : start + self.config.batch_size],
                device=self.config.device,
            )
            outputs: list[tuple[Tensor, Tensor]] = []
            for model, dtype in zip(self.models, self.dtypes, strict=True):
                with torch.autocast(
                    "cuda",
                    dtype=dtype,
                    enabled=self.config.device == "cuda" and dtype is not None,
                ):
                    outputs.append(model.forward(*arguments))
            for position, destination in ((0, logits), (1, elements)):
                stacked = torch.stack([output[position].float() for output in outputs])
                destination.append((stacked if each else stacked.mean(0)).cpu().numpy())
        return np.concatenate(logits, axis=int(each)), np.concatenate(
            elements,
            axis=int(each),
        )

    def views(
        self,
        spectra: Sequence[SpectrumRequest],
        *,
        merged_counts: Sequence[int] = (),
    ) -> ModelViews:
        """Run every bank model on single and per-polarity merged source items.

        Args:
          spectra: Complete spectra after the owning engine's preparation.
          merged_counts: Recorded source ce_n values; absent values mean one.

        Returns:
          views: Both prediction heads with every model and spectrum axis retained.

        """
        single, merged = self.items(spectra, merged_counts=merged_counts)
        zs, es = self.predict(single, each=True)
        zm, em = self.predict(merged, each=True)
        return ModelViews(
            single=zs,
            single_elements=es,
            merged=zm,
            merged_elements=em,
            modes=np.array([item.mode for item in merged], np.float32),
        )

    def items(
        self,
        spectra: Sequence[SpectrumRequest],
        *,
        merged_counts: Sequence[int] = (),
    ) -> tuple[list[BankItem], list[BankItem]]:
        """Build complete Full1 or v6 items with their distinct merged-adduct rules.

        Args:
          spectra: Query spectra in encounter order.
          merged_counts: Source ce_n for each spectrum, or an empty sequence for ones.

        Returns:
          single: One item per spectrum, including empty spectra.
          merged: One merged item per present polarity, positive before negative.

        """
        config = self.config
        counts = (
            list(merged_counts)
            if merged_counts
            else [spectrum.get("ce_n", 1) for spectrum in spectra]
        )
        single = [
            self._item(spectrum, count=count)
            for spectrum, count in zip(spectra, counts, strict=True)
        ]
        merged: list[BankItem] = []
        for mode in (1, -1):
            indices = [i for i, item in enumerate(single) if item.mode == mode]
            if not indices:
                continue
            group = [spectra[i] for i in indices]
            mass, intensity = merge_main(
                [
                    (
                        np.asarray(item["mz"], np.float64),
                        np.asarray(item["it"], np.float64),
                    )
                    for item in group
                ],
                tolerance=config.merge_tolerance,
                max_peaks=None,
            )
            precursor = float(np.median([item["prec"] for item in group]))
            mass, intensity = prepare_peaks(
                mass,
                intensity,
                precursor=precursor,
                max_peaks=config.max_peaks,
                floor=config.floor,
            )
            energies = [single[i].energy for i in indices if single[i].energy_known]
            merged.append(
                BankItem(
                    mass=mass,
                    intensity=intensity,
                    precursor=precursor,
                    adduct=adduct_index(
                        config.merged_adduct([item["adduct"] for item in group]),
                    ),
                    energy=float(np.mean(energies)) if energies else 0.0,
                    energy_known=float(bool(energies)),
                    merged_count=min(sum(max(1, counts[i]) for i in indices), 8),
                    mode=float(mode),
                ),
            )
        return single, merged

    def _item(self, spectrum: SpectrumRequest, *, count: int) -> BankItem:
        mass, intensity = prepare_peaks(
            np.asarray(spectrum["mz"], np.float64),
            np.asarray(spectrum["it"], np.float64),
            precursor=spectrum["prec"],
            max_peaks=self.config.max_peaks,
            floor=self.config.floor,
        )
        energy = spectrum.get("ce")
        if energy is None or math.isnan(energy):
            energy, known = 0.0, False
        else:
            known = True
        return BankItem(
            mass=mass,
            intensity=intensity,
            precursor=spectrum["prec"],
            adduct=adduct_index(spectrum["adduct"]),
            energy=energy,
            energy_known=float(known),
            merged_count=min(self.config.single_count(count), 8),
            mode=float(spectrum["mode"] == "positive") * 2 - 1
            if isinstance(spectrum["mode"], str)
            else float(spectrum["mode"]),
        )


def dreams_attention_fp32(dtype: torch.dtype, *, trained: bool) -> bool:
    """Select the complete source attention recipe independently of Fourier fp32.

    Args:
      dtype: Actual bank autocast precision, or fp32 without autocast.
      trained: Attention arithmetic saved in the released backbone configuration.

    Returns:
      fp32: The source inference attention precision flag.

    """
    return trained if dtype == torch.bfloat16 else True


def collate_main(
    items: Sequence[BankItem],
    *,
    device: str = "cpu",
) -> tuple[Tensor, ...]:
    """Pack all nine source arguments, keeping one unmasked slot for empty items.

    Args:
      items: At least one prepared spectrum item.
      device: Torch destination device.

    Returns:
      arguments: Masses, intensities, padding, precursor, adduct, CE, CE-known,
        merged-count/4 and polarity tensors in source call order.

    """
    inputs = pack_inputs(
        [item.mass for item in items],
        [item.intensity for item in items],
        precursor=np.array([item.precursor for item in items], np.float32),
        adduct=np.array([item.adduct for item in items], np.int64),
        energy=np.array([item.energy for item in items], np.float32),
        energy_known=np.array([item.energy_known for item in items], np.float32),
        merged_count=np.array([item.merged_count for item in items], np.float32),
        mode=np.array([item.mode for item in items], np.float32),
    )
    return tuple(
        value.to(device) for value in cast("Mapping[str, Tensor]", inputs).values()
    )


type DreamsRows = NDArray[np.generic]

RELATIVE_WINDOW: Final = 10e-6
"""Half-width, relative to the target mass, of the pinned DREAMS neighbor window."""


class TailAudit(TypedDict):
    """Preserve every source audit field in its original insertion order."""

    molecule_id: str
    original_count: int
    original_unique: int
    invalid_count: int
    duplicate_count: int
    revised_count: int
    ranked_added: int
    dreams_added: int
    promoted_original: int
    changed: bool


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class DreamsIndex:
    """Retain all mass and string rows plus both complete stable index arrays."""

    mass: RankingScores
    smiles: DreamsRows
    order: PoolIndices
    sorted_mass: RankingScores


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class DreamsSelection:
    """Retain both source bounds, every ordered pool row and decoded string."""

    low: int
    high: int
    rows: PoolIndices
    smiles: tuple[str, ...]


class DreamsStore:
    """Materialize an explicitly selected optional store only when queried."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.index: DreamsIndex | None = None

    @classmethod
    def load(cls, directory: Path) -> DreamsStore | None:
        """Select a present mass path without reading optional array payloads."""
        path = directory / "dreams_np_mass.npy"
        return cls(directory) if path.exists() or path.is_symlink() else None

    def select(self, target: float) -> DreamsSelection:
        """Return complete source query bounds, rows and decoded candidates.

        Args:
          target: Query neutral mass; the tail policy owns its admission checks.

        Returns:
          selection: Both left insertion bounds and all distance-ordered rows/strings.

        """
        if self.index is None:
            mass = cast(
                "DreamsRows",
                np.load(self.directory / "dreams_np_mass.npy", allow_pickle=False),
            ).astype(np.float64)
            smiles = _smiles_array(self.directory / "dreams_np_smiles.npy")
            if len(mass) != len(smiles):
                raise AssertionError
            order = np.argsort(mass, kind="stable")
            self.index = DreamsIndex(
                mass=mass,
                smiles=smiles,
                order=order,
                sorted_mass=mass[order],
            )
        index = self.index
        bounds = np.searchsorted(
            index.sorted_mass,
            [target * (1 - RELATIVE_WINDOW), target * (1 + RELATIVE_WINDOW)],
        )
        low, high = bounds.item(0), bounds.item(1)
        rows = index.order[low:high]
        rows = rows[np.argsort(np.abs(index.mass[rows] - target), kind="stable")]
        candidates: list[str] = []
        for position in range(len(rows)):
            row = rows.item(position)
            value = cast(object, index.smiles[row])
            candidates.append(
                value.decode() if isinstance(value, bytes) else str(value),
            )
        return DreamsSelection(low=low, high=high, rows=rows, smiles=tuple(candidates))

    def candidates(self, target: float) -> tuple[str, ...]:
        """Return all left/left 10ppm rows in stable absolute-distance order."""
        return self.select(target).smiles


def safe_tail[Identity: Hashable](
    molecule_id: str,
    original: Sequence[str],
    extras: Sequence[str],
    *,
    target: float,
    identity: Callable[[str], Identity | None],
    limit: int,
    dreams: DreamsStore | None = None,
    processed: bool = True,
) -> tuple[list[str], TailAudit | None]:
    """Preserve valid original ranks, fill from ranked extras, then optional masses.

    Args:
      molecule_id: Source identifier retained in the complete audit.
      original: Original semicolon tokens; inactive tokens remain unchanged.
      extras: Complete overflow/base-tail/engine/PubChem sequence in encounter order.
      target: Source median neutral mass, including explicit unknown/nonfinite values.
      identity: Strict prepared official-identity lookup; missing strings must raise.
      limit: Candidates a list may hold; fewer admit the optional masses.
      dreams: Explicitly selected optional store, or None when unavailable.
      processed: Whether this row entered the source processing loop.

    Returns:
      candidates: Complete list retaining every valid original identity at no lower rank.
      audit: All ten ordered source fields, or None for an inactive row.

    """
    if not processed:
        return list(original), None
    original = [value.strip() for value in original if value.strip()]
    ranks = _first_ranks(original, identity=identity)
    clean = safe_extend(original, [], identity=identity, limit=limit)
    revised = safe_extend(clean, extras, identity=identity, limit=limit)
    after_ranked = len(revised)
    if (
        len(revised) < limit
        and math.isfinite(target)
        and target > 0
        and dreams is not None
    ):
        revised = safe_extend(
            revised,
            dreams.candidates(target),
            identity=identity,
            limit=limit,
        )
    if not revised:
        raise AssertionError(f"No valid candidates for {molecule_id}")
    new_ranks = _first_ranks(revised, identity=identity)
    if len(revised) > limit or len(new_ranks) != len(revised):
        raise AssertionError
    if not all(
        key in new_ranks and new_ranks[key] <= rank for key, rank in ranks.items()
    ):
        raise AssertionError
    invalid_count = sum(identity(value) is None for value in original)
    audit: TailAudit = {
        "molecule_id": molecule_id,
        "original_count": len(original),
        "original_unique": len(ranks),
        "invalid_count": invalid_count,
        "duplicate_count": sum(identity(value) is not None for value in original)
        - len(ranks),
        "revised_count": len(revised),
        "ranked_added": after_ranked - len(clean),
        "dreams_added": len(revised) - after_ranked,
        "promoted_original": sum(new_ranks[key] < rank for key, rank in ranks.items()),
        "changed": original != revised,
    }
    return revised, audit


def _first_ranks[Identity: Hashable](
    candidates: Sequence[str],
    *,
    identity: Callable[[str], Identity | None],
) -> dict[Identity, int]:
    ranks: dict[Identity, int] = {}
    for rank, candidate in enumerate(candidates, start=1):
        candidate_identity = identity(candidate)
        if candidate_identity is not None:
            ranks.setdefault(candidate_identity, rank)
    return ranks


def _smiles_array(path: Path) -> DreamsRows:
    with path.open("rb") as handle:
        try:
            values = cast(object, np.load(handle, allow_pickle=False))
        except ValueError as error:
            if error.args != (
                "Object arrays cannot be loaded when allow_pickle=False",
            ):
                raise
            # NumPy has consumed the complete NPY header and stopped before its pickle.
            values = cast(object, ScientificDataUnpickler(handle).load())
    if not isinstance(values, np.ndarray):
        raise TypeError("DREAMS SMILES payload must be a NumPy array.")
    return cast("DreamsRows", values)


# Tensor shapes are checked when the released state loads; the attention head count,
# peak slots, polarity shift and precision recipe change the arithmetic without changing
# a shape.
def _verify_dreams(
    declared: DreamsFP.Config,
    checkpoint: dict[str, object],
    *,
    dtype: torch.dtype,
) -> None:
    """Require the declared DreamsFP to share the checkpoint's non-shape settings."""
    fcfg = from_plain(checkpoint["fcfg"], dict[str, object])
    bcfg = from_plain(checkpoint["bcfg"], dict[str, object])
    if fcfg.get("cond_target", "prec") != "prec":
        raise ValueError("The released DreamsFP requires precursor conditioning.")
    if (
        bcfg.get("graphormer", "key") != "key"
        or bcfg.get("pad_mode", "faithful") != "faithful"
    ):
        raise ValueError(
            "The released DreamsFP requires faithful mass-biased attention.",
        )
    # Finalization expands the block template into one config per layer.
    blocks = cast(object, declared.backbone.encoder.block)
    assert isinstance(blocks, list)
    block = cast(object, blocks[0])
    assert isinstance(block, TransformerBlock.Config)
    attention = block.attn
    assert isinstance(attention, DreamAttention.Config)
    recorded = (
        from_plain(fcfg["nbits"], int),
        from_plain(fcfg["n_elem"], int),
        from_plain(fcfg["n_peaks"], int),
        2 * 1.007276467 if fcfg["neg_shift"] else 0.0,
        from_plain(bcfg["n_heads"], int),
        dreams_attention_fp32(
            dtype,
            trained=from_plain(bcfg.get("attn_fp32", True), bool),
        ),
    )
    if recorded != (
        declared.channels_out,
        declared.num_elements,
        declared.max_peaks,
        declared.negative_mass_shift,
        attention.attention.num_heads,
        attention.precision.dtype == torch.float32,
    ):
        raise ValueError(
            "Checkpoint DreamsFP bits, elements, peaks, polarity shift, heads "
            f"and attention precision {recorded} differ from the declared ones.",
        )
