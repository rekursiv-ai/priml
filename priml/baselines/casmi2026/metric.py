"""Molecule-level MRR@25 for the distinct fingerprint-only budget experiment."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypedDict, cast, override

import json

from configgle import Fig, Makeable
from torch import Tensor
from treekle import from_plain, to_plain

import numpy as np

from priml.baselines.casmi2026.chemistry import OfficialIdentity
from priml.custom_types import HasNormalizedWorkingDirPattern
from priml.data.arrays import load_array
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from pyarrow import parquet
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")


class MoleculeMRR:
    """Average spectrum logits within molecules, then rank the full mass window."""

    class Config(Fig["MoleculeMRR"]):
        base_dir: Path | str | None = None
        """Parent-supplied resource root."""

        working_dir: Path | str = "/datasets/casmi2026/training/work"
        """Prepared pool and structure table written by the training cache build."""

        ppm: float = 10.0
        """Neutral mass candidate window in parts per million, without subsampling."""

        limit: int = 25
        """Maximum unique official identities eligible for reciprocal-rank credit."""

        identity: Makeable[Callable[[str], Hashable | None]] = field(
            default_factory=OfficialIdentity.Config,
        )
        """Official metric identities from the independently generated data cache."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if (
                isinstance(self.identity, HasNormalizedWorkingDirPattern)
                and self.identity.base_dir is None
            ):
                self.identity.base_dir = self.base_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.limit <= 0:
            raise ValueError("limit must be positive.")
        self.config = config
        self.identity = config.identity.make()
        root = Path(config.working_dir)
        pool = parquet.read_table(root / "pool_meta.parquet")
        self.mass = np.asarray(
            from_plain(cast(object, pool.column("mass").to_pylist()), list[float]),
            dtype=np.float64,
        )
        self.smiles = from_plain(
            cast(object, pool.column("smiles").to_pylist()),
            list[str],
        )
        self.packed = load_array(root / "pool_fp.npy", np.uint8, mmap=True)
        structures = parquet.read_table(root / "train_metric_structs.parquet")
        self.truth = dict(
            zip(
                from_plain(
                    cast(object, structures.column("sid").to_pylist()),
                    list[int],
                ),
                from_plain(
                    cast(object, structures.column("smiles").to_pylist()),
                    list[str],
                ),
                strict=True,
            ),
        )
        representatives: dict[str, int] = {}
        self.groups = {
            sid: representatives.setdefault(key, sid)
            for sid, key in zip(
                self.truth,
                from_plain(
                    cast(object, structures.column("metric_identity").to_pylist()),
                    list[str],
                ),
                strict=True,
            )
        }
        self.reset()

    def reset(self) -> None:
        """Clear every accumulated molecule and spectrum count."""
        self.sums: dict[int, Tensor] = {}
        self.counts: dict[int, int] = {}
        self.masses: dict[int, list[float]] = {}

    def update(self, logits: Tensor | tuple[Tensor, Tensor], **batch: object) -> None:
        """Accumulate all spectra of each molecule, including across batch boundaries.

        Args:
          logits: Fingerprint logits before sigmoid, or the model's complete output,
            whose first entry holds them.
          **batch: Molecule ``sid`` and public precursor-derived ``neutral_mass``.

        """
        if not isinstance(logits, Tensor):
            logits = logits[0]
        sid, mass = batch["sid"], batch["neutral_mass"]
        assert isinstance(sid, Tensor)
        assert isinstance(mass, Tensor)
        ids = from_plain(cast(object, sid.tolist()), list[int])
        masses = from_plain(cast(object, mass.tolist()), list[float])
        for sid_value, value, target in zip(
            ids,
            logits.detach().float().cpu(),
            masses,
            strict=True,
        ):
            identity = self.groups[sid_value]
            self.sums[identity] = (
                self.sums[identity] + value if identity in self.sums else value.clone()
            )
            self.counts[identity] = self.counts.get(identity, 0) + 1
            self.masses.setdefault(identity, []).append(target)

    def compute(self) -> dict[str, float]:
        """Rank each whole mass window and report one reciprocal rank per molecule.

        Returns:
          metrics: MRR@25 and molecule count, including absent correct candidates.

        """
        total = 0.0
        for sid, logits_sum in self.sums.items():
            masses = np.asarray(self.masses[sid])
            finite = masses[np.isfinite(masses)]
            if not len(finite):
                continue
            target = float(np.median(finite))
            tolerance = target * self.config.ppm * 1e-6
            lower = int(np.searchsorted(self.mass, target - tolerance, side="left"))
            upper = int(np.searchsorted(self.mass, target + tolerance, side="right"))
            logits = (logits_sum / self.counts[sid]).numpy()
            fingerprints = np.unpackbits(self.packed[lower:upper], axis=1)[
                :,
                : len(logits),
            ].astype(np.float32)
            scores = from_plain(
                cast(object, (fingerprints @ logits).tolist()),
                list[float],
            )
            expected = self.identity(self.truth[sid])
            if expected is None:
                raise ValueError(f"Invalid truth for molecule {sid}.")
            seen: set[Hashable] = set()
            candidates = [self.identity(smiles) for smiles in self.smiles[lower:upper]]
            ordered = sorted(
                range(len(candidates)),
                key=lambda index: (
                    -scores[index],
                    json.dumps(to_plain(candidates[index]), sort_keys=True),
                ),
            )
            for index in ordered:
                candidate = candidates[index]
                if candidate is None or candidate in seen:
                    continue
                seen.add(candidate)
                if candidate == expected:
                    total += 1.0 / len(seen)
                    break
                if len(seen) == self.config.limit:
                    break
        return {
            f"mrr@{self.config.limit}": total / len(self.sums) if self.sums else 0.0,
            "molecules": float(len(self.sums)),
        }

    class StateDict(TypedDict):
        """Complete molecular sufficient statistics for checkpointing."""

        sums: dict[int, Tensor]
        counts: dict[int, int]
        masses: dict[int, list[float]]

    def state_dict(self) -> StateDict:
        """Return all accumulated logits, counts, and public query masses."""
        return {"sums": self.sums, "counts": self.counts, "masses": self.masses}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the complete molecular sufficient statistics.

        Args:
          state_dict: Previously saved logits, counts, and masses.

        """
        state = cast(MoleculeMRR.StateDict, state_dict)
        self.sums, self.counts, self.masses = (
            state["sums"],
            state["counts"],
            state["masses"],
        )
