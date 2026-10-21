"""Launch complete native request planning around isolated chemical data generation."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override

from configgle import Fig
from treekle import from_plain

from priml.baselines.casmi2026.chemistry import (
    CandidatePool,
    IdentityTable,
    SpectralLibrary,
)
from priml.baselines.casmi2026.dreams import DreamsStore
from priml.baselines.casmi2026.isolated import ReactionTable
from priml.baselines.casmi2026.main_engine import (
    MainRetrieval,
    ReactionScreen,
    training_smiles,
)
from priml.baselines.casmi2026.pubchem import PubChemStore
from priml.baselines.casmi2026.recipe import (
    WindowRequest,
    plan_requests,
    product_inputs,
    read_queries,
)
from priml.baselines.casmi2026.scripts.recipe_plan import (
    write_plan,
    write_products,
    write_reaction_inputs,
)
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from pyarrow import parquet

    import numpy as np
else:
    from wrapt import lazy_import

    np = lazy_import("numpy")
    parquet = lazy_import("pyarrow.parquet")


class PlanningPreparation:
    """Plan all potential reaction parents and external mass-window rows."""

    class Config(Fig["PlanningPreparation"]):
        base_dir: Path | str | None = "/opt/scratch/datasets/casmi2026"
        """Resource root beneath which every logical path below resolves."""

        queries: Path | str = (
            "inputs/enveda-CASMI26-molecule-id-mass-spectra/test.parquet"
        )
        """Complete competition query spectra."""

        library: Path | str = "training/work"
        """Complete native prepared spectral library."""

        pool: Path | str = "inputs/datasets/ahmedberatozer/casmi26-v2-pool"
        """Provided main pool and complete training structures."""

        train_formula: Path | str = (
            "inputs/datasets/ahmedberatozer/casmi26-v4b-models/code/fe_v4/res/analog_struct/train_fvec.npy"
        )
        """Provided complete training formula vectors in SID order."""

        pubchem: Path | str = "inputs/datasets/ahmedberatozer/casmi26-pubchem-tier"
        """Original complete PubChem neutral-mass and ASCII stores."""

        dreams: Path | str | None = None
        """Directory holding the optional DREAMS mass and string arrays; None skips it."""

        retrieval: MainRetrieval.Config = field(default_factory=MainRetrieval.Config)
        """Exact main-arm analog retrieval configuration."""

        generator: ReactionScreen.Config = field(default_factory=ReactionScreen.Config)
        """Exact main-arm parent and reaction enumeration budgets."""

        max_spectra: int = 16
        """Main query prefix before analog parent selection."""

        ppm: float = 10.0
        """PubChem mass-window half-width of the main arm, in parts per million."""

        output: Path | str = "native-chemistry/preparation/requests"
        """New scratch directory for every complete first-phase request."""

        @override
        def finalize(self) -> Self:
            self.queries = resolve_working_dir(self.base_dir, self.queries)
            self.library = resolve_working_dir(self.base_dir, self.library)
            self.pool = resolve_working_dir(self.base_dir, self.pool)
            self.train_formula = resolve_working_dir(self.base_dir, self.train_formula)
            self.pubchem = resolve_working_dir(self.base_dir, self.pubchem)
            if self.dreams is not None:
                self.dreams = resolve_working_dir(self.base_dir, self.dreams)
            self.output = resolve_working_dir(self.base_dir, self.output)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> None:
        """Publish complete first-phase requests from provided native data.

        Args:
          *args: No passthrough arguments are accepted.

        """
        if args:
            raise ValueError("PlanningPreparation accepts configuration fields only.")
        config = self.config
        pool = Path(config.pool)
        dreams = None if config.dreams is None else Path(config.dreams)
        plan = plan_requests(
            read_queries(Path(config.queries)),
            library=SpectralLibrary.load(Path(config.library)),
            retrieval=config.retrieval.make(),
            generator=config.generator.make(),
            pubchem=PubChemStore.load(Path(config.pubchem)),
            ppm=config.ppm,
            max_spectra=config.max_spectra,
            dreams=None if dreams is None else DreamsStore(dreams),
        )
        write_plan(
            plan,
            pool=CandidatePool.load(pool),
            train_smiles=training_smiles(pool / "train_structs.parquet"),
            train_formula=cast(
                "NDArray[np.int16]",
                np.load(
                    Path(config.train_formula),
                    mmap_mode="r",
                    allow_pickle=False,
                ),
            ),
            output=Path(config.output),
            inputs=_files(
                [
                    Path(config.queries),
                    Path(config.library),
                    pool,
                    Path(config.train_formula),
                    Path(config.pubchem),
                    *([] if dreams is None else [dreams]),
                ],
            ),
        )


class RawProductPreparation:
    """Expose every ordered raw reaction product for independent identity preparation."""

    class Config(Fig["RawProductPreparation"]):
        base_dir: Path | str | None = "/opt/scratch/datasets/casmi2026"
        """Resource root beneath which every logical path below resolves."""

        reactions: Path | str = "native-chemistry/reactions"
        """Complete independently generated original46-rule request/results data."""

        output: Path | str = "native-chemistry/preparation/raw-products"
        """New complete raw-product request directory."""

        @override
        def finalize(self) -> Self:
            self.reactions = resolve_working_dir(self.base_dir, self.reactions)
            self.output = resolve_working_dir(self.base_dir, self.output)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> None:
        """Publish every complete raw product and its original path provenance.

        Args:
          *args: No passthrough arguments are accepted.

        """
        if args:
            raise ValueError("RawProductPreparation accepts configuration fields only.")
        reactions = Path(self.config.reactions)
        write_reaction_inputs(
            ReactionTable(reactions),
            output=Path(self.config.output),
            inputs=_files([reactions]),
        )


class ProductPreparation:
    """Join all proposal aliases, prior paths and complete consumer identities."""

    class Config(Fig["ProductPreparation"]):
        base_dir: Path | str | None = "/opt/scratch/datasets/casmi2026"
        """Resource root beneath which every logical path below resolves."""

        plan: Path | str = field(
            default_factory=lambda: PlanningPreparation.Config().output,
        )
        """Complete first-phase external-window requests."""

        reactions: Path | str = field(
            default_factory=lambda: RawProductPreparation.Config().reactions,
        )
        """Complete independently generated original46-rule request/results data."""

        identities: Path | str = field(
            default_factory=lambda: IdentityTable.Config().working_dir,
        )
        """Independent raw-product identities and standardized aliases."""

        pool_metadata: Path | str = (
            "inputs/datasets/ahmedberatozer/casmi26-v2-pool/pool_meta.parquet"
        )
        """Every original main pool SMILES, preserving positional duplicates."""

        training: Path | str = (
            "inputs/datasets/ahmedberatozer/casmi26-v2-pool/train_structs.parquet"
        )
        """Every training structure in explicit contiguous SID order."""

        second_metadata: Path | str = "native-chemistry/engine2-pool/pool_meta.parquet"
        """Every independently prepared second-pool SMILES in original row order."""

        max_first: int = 32
        """First-reaction site limit of every derivation request."""

        max_second: int = 16
        """Second-reaction site limit of every derivation request."""

        max_mols: int = 24
        """Molecule limit of every derivation request."""

        output: Path | str = "native-chemistry/preparation/products"
        """New scratch directory for all final family/identity requests."""

        @override
        def finalize(self) -> Self:
            self.plan = resolve_working_dir(self.base_dir, self.plan)
            self.reactions = resolve_working_dir(self.base_dir, self.reactions)
            self.identities = resolve_working_dir(self.base_dir, self.identities)
            self.pool_metadata = resolve_working_dir(self.base_dir, self.pool_metadata)
            self.training = resolve_working_dir(self.base_dir, self.training)
            self.second_metadata = resolve_working_dir(
                self.base_dir,
                self.second_metadata,
            )
            self.output = resolve_working_dir(self.base_dir, self.output)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> None:
        """Publish all final candidate, generated-family and prior requests.

        Args:
          *args: No passthrough arguments are accepted.

        """
        if args:
            raise ValueError("ProductPreparation accepts configuration fields only.")
        config = self.config
        plan, reactions = Path(config.plan), Path(config.reactions)
        identities = IdentityTable.Config(working_dir=config.identities).make()
        windows: list[WindowRequest] = []
        for name in ("pubchem", "dreams"):
            windows.extend(
                from_plain(
                    parquet.read_table(
                        plan / f"{name}-requests.parquet",
                    ).to_pylist(),
                    list[WindowRequest],
                ),
            )
        records = product_inputs(
            ReactionTable(reactions),
            chemistry=identities,
            pool=from_plain(
                parquet.read_table(Path(config.pool_metadata), columns=["smiles"])
                .column("smiles")
                .to_pylist(),
                list[str],
            ),
            train=training_smiles(Path(config.training)),
            second=from_plain(
                parquet.read_table(Path(config.second_metadata), columns=["smiles"])
                .column("smiles")
                .to_pylist(),
                list[str],
            ),
            windows=windows,
            max_first=config.max_first,
            max_second=config.max_second,
            max_mols=config.max_mols,
        )
        write_products(
            records,
            chemistry=identities,
            output=Path(config.output),
            inputs=_files(
                [
                    plan,
                    reactions,
                    Path(config.identities),
                    Path(config.pool_metadata),
                    Path(config.training),
                    Path(config.second_metadata),
                ],
            ),
        )


def prepare_requests() -> PlanningPreparation.Config:
    """Return the complete first-phase planning job for the Priml launcher."""
    return PlanningPreparation.Config()


def prepare_raw_products() -> RawProductPreparation.Config:
    """Return raw-product staging before independent identity preparation."""
    return RawProductPreparation.Config()


def prepare_product_requests() -> ProductPreparation.Config:
    """Return final identity, generated-family and prior-request staging."""
    return ProductPreparation.Config()


def _files(paths: Sequence[Path]) -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                file
                for path in paths
                for file in (path.rglob("*") if path.is_dir() else (path,))
                if file.is_file()
            },
        ),
    )
