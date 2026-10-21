"""Generate reusable CASMI chemistry assets in an isolated RDKit interpreter.

Run this tool with the independently provisioned preparation interpreter. Native
training and inference read the generated Arrow/NumPy files and never import RDKit.
Input parquet files supply a SMILES column; released candidate metadata pickles
are read through a restricted scientific-data reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypedDict, cast

import argparse
import json

from treekle import from_plain, to_plain

from priml.baselines.casmi2026.isolated import (
    ADMISSION_DIR,
    DATASET_ROOT,
    FRAGMENT_GAP_DIR,
    GLACIER_ROOT_DIR,
    ICE_ROOT_DIR,
    IDENTITY_DIR,
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    read_pool_metadata,
    v6_reaction_rules,
)
from priml.baselines.casmi2026.scripts.admission import prepare_admission
from priml.baselines.casmi2026.scripts.analog_data import prepare_analog
from priml.baselines.casmi2026.scripts.atomic_data import prepare_graphs
from priml.baselines.casmi2026.scripts.derivation_data import prepare_derivations
from priml.baselines.casmi2026.scripts.engine_identity import (
    prepare_engine_identity,
)
from priml.baselines.casmi2026.scripts.fragment_gap_data import (
    prepare_fragment_gap,
)
from priml.baselines.casmi2026.scripts.fragnet_data import prepare_fragnet
from priml.baselines.casmi2026.scripts.glacier_data import prepare_glacier_graphs
from priml.baselines.casmi2026.scripts.ice_data import prepare_ice_graphs
from priml.baselines.casmi2026.scripts.identity_data import prepare_identity
from priml.baselines.casmi2026.scripts.intrinsic import prepare_intrinsic
from priml.baselines.casmi2026.scripts.main_fragments import (
    prepare_main_fragments,
)
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.baselines.casmi2026.scripts.pubchem_data import prepare_pubchem
from priml.baselines.casmi2026.scripts.rdkit_backend import (
    Chemistry,
    MolecularContext,
)
from priml.baselines.casmi2026.scripts.reaction_data import prepare_reactions
from priml.baselines.casmi2026.scripts.second_fragments import (
    prepare_second_fragments,
)
from priml.baselines.casmi2026.scripts.second_library import (
    prepare_second_library,
)
from priml.baselines.casmi2026.scripts.second_pool import (
    prepare_second_fingerprints,
    prepare_second_pool,
)
from priml.paths import validated_output_path


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.lib import format as numpy_format
    from numpy.typing import NDArray
    from pyarrow import Table, parquet

    import numpy as np

    from priml.baselines.casmi2026.scripts.rdkit_backend import StructureRecord
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    numpy_format = lazy_import("numpy.lib.format")
    np = lazy_import("numpy")


RAW_BITS: Final = 2 * 4096 + 4 * 2048 + 167

BATCH_ROWS: Final = 256
"""Complete records per Arrow row group of the batched preparers."""


class _Batched(Protocol):
    """A preparer writing complete records in bounded Arrow row groups."""

    def __call__(
        self,
        input_path: Path,
        *,
        output: Path,
        version: str,
        batch_rows: int,
    ) -> None:
        """Write one new dataset directory."""
        ...


class Encoder(Protocol):
    """Chemistry operations supplied only by the preparation process."""

    version: str

    def record(self, smiles: str) -> StructureRecord | None:
        """Return complete molecular data for a valid structure."""
        ...

    def standardize(self, smiles: str, *, keep_charged: bool = False) -> str | None:
        """Return the declared standardized molecule or reject the input."""
        ...


def prepare(
    inputs: Sequence[Path],
    *,
    output: Path,
    chemistry: Encoder,
    column: str | None = None,
    standardize: bool = False,
    keep_charged: bool = False,
) -> None:
    """Generate every requested molecular row and its complete raw fingerprint.

    Args:
      inputs: Provided parquet or restricted candidate-metadata pickle data files.
      output: New external dataset directory; existing outputs are never replaced.
      chemistry: Scientific primitives from the isolated preparation interpreter.
      column: Explicit parquet SMILES field, or infer smiles/normalized_smiles.
      standardize: Apply largest-fragment/uncharging/stereo preparation first.
      keep_charged: Preserve formal charges that survive uncharging.

    """
    destination = validated_output_path(output, protected=inputs)
    destination.mkdir(parents=True, exist_ok=False)
    requested: dict[str, None] = {}
    for path in inputs:
        if path.suffix == ".pkl":
            values: list[str | None] = list(read_pool_metadata(path)["smiles"])
        else:
            source = parquet.ParquetFile(path)
            field_name = column or next(
                (
                    name
                    for name in ("smiles", "normalized_smiles")
                    if name in source.schema_arrow.names
                ),
                None,
            )
            if field_name is None:
                raise ValueError("A SMILES column is required.")
            values = from_plain(
                cast(
                    "object",
                    source.read(columns=[field_name]).column(field_name).to_pylist(),
                ),
                list[str | None],
            )
        requested.update((value, None) for value in values if value is not None)
    packed = cast(
        "NDArray[np.uint8]",
        numpy_format.open_memmap(
            destination / "raw_fp.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(requested), (RAW_BITS + 7) // 8),
        ),
    )
    columns: dict[str, list[object]] = {
        name: []
        for name in (
            "input_smiles",
            "smiles",
            "standardized_smiles",
            "charged_smiles",
            "valid",
            "formula",
            "mass",
            "heavy_atoms",
            "key",
            "raw_key",
            "metric_key",
            "composition",
        )
    }
    valid_rows = 0
    for index, value in enumerate(requested):
        standardized = chemistry.standardize(value)
        charged = chemistry.standardize(value, keep_charged=True)
        prepared = (charged if keep_charged else standardized) if standardize else value
        record = chemistry.record(prepared) if prepared is not None else None
        row: dict[str, object] = {
            "input_smiles": value,
            "smiles": record.smiles if record is not None else "",
            "standardized_smiles": standardized,
            "charged_smiles": charged,
            "valid": record is not None,
            "formula": record.formula if record is not None else "",
            "mass": record.mass if record is not None else float("nan"),
            "heavy_atoms": record.heavy_atoms if record is not None else 0,
            "key": record.key if record is not None else "",
            "raw_key": record.raw_key if record is not None else "",
            "metric_key": record.metric_key if record is not None else "",
            "composition": json.dumps(
                to_plain(record.composition if record is not None else ()),
            ),
        }
        for name, item in row.items():
            columns[name].append(item)
        packed[index] = record.packed if record is not None else 0
        valid_rows += record is not None
    parquet.write_table(Table.from_pydict(columns), destination / "structures.parquet")
    _ = (destination / "fingerprint_bits.txt").write_text(f"{RAW_BITS}\n")
    generators = (_THIS, _THIS.with_name("rdkit_backend.py").resolve())
    manifest: Manifest = {
        "format_version": 1,
        "rdkit": chemistry.version,
        "numpy": np.__version__,
        "raw_bits": RAW_BITS,
        "rows": len(requested),
        "valid_rows": valid_rows,
        "standardize": standardize,
        "keep_charged": keep_charged,
        "inputs": input_digests(inputs),
        "generator": source_digests(generators),
    }
    write_manifest(destination, manifest)


def main() -> int:
    """Prepare declared input structures.

    Returns:
      status: Zero after all complete assets and provenance have been written.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    flags.run(flags)
    return 0


class Flags(Protocol):
    """The command-line parser's complete typed result."""

    run: Callable[[Flags], None]
    inputs: list[Path]
    source: Path
    output: Path
    version: str
    column: str | None
    standardize: bool
    keep_charged: bool
    raw: bool
    batch_rows: int
    identities: Path
    pool: Path
    bits: Path
    fingerprints: Path
    coconut: Path
    bio: Path


class Manifest(TypedDict):
    """Versioned generator, scientific environment and complete input provenance."""

    format_version: int
    rdkit: str
    numpy: str
    raw_bits: int
    rows: int
    valid_rows: int
    standardize: bool
    keep_charged: bool
    inputs: dict[str, str]
    generator: dict[str, str]


def _run_structures(flags: Flags) -> None:
    prepare(
        flags.inputs,
        output=flags.output,
        chemistry=Chemistry(flags.version),
        column=flags.column,
        standardize=flags.standardize,
        keep_charged=flags.keep_charged,
    )


def _run_reactions(flags: Flags) -> None:
    prepare_reactions(
        flags.source,
        output=flags.output,
        chemistry=MolecularContext(flags.version),
    )


def _run_reactions_v6(flags: Flags) -> None:
    prepare_reactions(
        flags.source,
        output=flags.output,
        chemistry=MolecularContext(flags.version),
        rules=v6_reaction_rules(),
        default_products=300,
    )


def _run_derivations(flags: Flags) -> None:
    prepare_derivations(
        flags.source,
        identities=flags.identities,
        output=flags.output,
        version=flags.version,
    )


def _run_graphs(flags: Flags) -> None:
    prepare_graphs(
        flags.source,
        output=flags.output,
        version=flags.version,
        raw=flags.raw,
        batch_rows=flags.batch_rows,
    )


def _run_descriptors(flags: Flags) -> None:
    prepare_intrinsic(flags.source, output=flags.output, version=flags.version)


def _run_analog(flags: Flags) -> None:
    prepare_analog(flags.source, output=flags.output, version=flags.version)


def _run_fragnet(flags: Flags) -> None:
    prepare_fragnet(flags.source, output=flags.output, version=flags.version)


def _run_pubchem(flags: Flags) -> None:
    prepare_pubchem(flags.source, output=flags.output, version=flags.version)


def _run_main_fragments(flags: Flags) -> None:
    prepare_main_fragments(flags.source, output=flags.output, version=flags.version)


def _run_second_fragments(flags: Flags) -> None:
    prepare_second_fragments(flags.source, output=flags.output, version=flags.version)


def _run_second_library(flags: Flags) -> None:
    prepare_second_library(flags.source, pool=flags.pool, output=flags.output)


def _run_second_fingerprints(flags: Flags) -> None:
    prepare_second_fingerprints(
        flags.source,
        bits=flags.bits,
        output=flags.output,
        version=flags.version,
    )


def _run_second_pool(flags: Flags) -> None:
    prepare_second_pool(
        flags.source,
        coconut=flags.coconut,
        bio=flags.bio,
        fingerprints=flags.fingerprints,
        output=flags.output,
    )


def _run_batched(prepare_batched: _Batched) -> Callable[[Flags], None]:
    def run(flags: Flags) -> None:
        prepare_batched(
            flags.source,
            output=flags.output,
            version=flags.version,
            batch_rows=flags.batch_rows,
        )

    return run


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register one command per asset family, each with its own inputs."""
    commands = parser.add_subparsers(dest="command", required=True)
    structures = _command(commands, "structures", _run_structures)
    structures.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Parquet or restricted candidate-metadata pickle files.",
    )
    structures.add_argument(
        "--column",
        help="SMILES column; infer smiles/normalized_smiles by default.",
    )
    structures.add_argument("--standardize", action="store_true")
    structures.add_argument("--keep-charged", action="store_true")
    for name, run in (
        ("reactions", _run_reactions),
        ("reactions-v6", _run_reactions_v6),
        ("descriptors", _run_descriptors),
        ("analog", _run_analog),
        ("fragnet", _run_fragnet),
        ("pubchem", _run_pubchem),
        ("main-fragments", _run_main_fragments),
        ("second-fragments", _run_second_fragments),
    ):
        _ = _source(_command(commands, name, run))
    derivations = _source(_command(commands, "derivations", _run_derivations))
    derivations.add_argument("--identities", type=Path, required=True)
    graphs = _source(_command(commands, "graphs", _run_graphs, simulator=True))
    graphs.add_argument("--raw", action="store_true")
    graphs.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    for name, prepare_batched, simulator, output in (
        ("admission", prepare_admission, True, ADMISSION_DIR),
        ("ice", prepare_ice_graphs, True, ICE_ROOT_DIR),
        ("glacier", prepare_glacier_graphs, True, GLACIER_ROOT_DIR),
        (
            "fragment-gap",
            prepare_fragment_gap,
            False,
            FRAGMENT_GAP_DIR,
        ),
        (
            "identity",
            prepare_identity,
            False,
            IDENTITY_DIR,
        ),
        ("engine-identity", prepare_engine_identity, False, None),
    ):
        batched = _command(
            commands,
            name,
            _run_batched(prepare_batched),
            simulator=simulator,
            output=output,
        )
        batched.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
        _ = _source(batched)
    library = _source(_command(commands, "second-library", _run_second_library))
    library.add_argument("--pool", type=Path, required=True)
    fingerprints = _source(
        _command(commands, "second-fingerprints", _run_second_fingerprints),
    )
    fingerprints.add_argument(
        "--bits",
        type=Path,
        default=_provided("prvsiyan/coconut-casmi26-candidates/fp_bits.npy"),
    )
    pool = _source(_command(commands, "second-pool", _run_second_pool))
    pool.add_argument("--fingerprints", type=Path, required=True)
    pool.add_argument(
        "--coconut",
        type=Path,
        default=_provided("prvsiyan/coconut-casmi26-candidates"),
    )
    pool.add_argument(
        "--bio",
        type=Path,
        default=_provided("prvsiyan/chebi-lipidmaps-casmi26"),
    )


def _command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    run: Callable[[Flags], None],
    *,
    simulator: bool = False,
    output: str | None = None,
) -> argparse.ArgumentParser:
    command = commands.add_parser(name)
    command.set_defaults(run=run)
    command.add_argument(
        "--output",
        type=Path,
        required=output is None,
        default=None if output is None else Path(DATASET_ROOT) / output,
        help="New asset directory; existing outputs are never replaced.",
    )
    command.add_argument(
        "--version",
        default=SIMULATOR_RDKIT if simulator else IDENTITY_RDKIT,
        help="Required isolated RDKit release.",
    )
    return command


def _source(command: argparse.ArgumentParser) -> argparse.ArgumentParser:
    command.add_argument("source", type=Path, help="Provided input table.")
    return command


def _provided(path: str) -> Path:
    return Path(DATASET_ROOT) / "inputs/datasets" / path
