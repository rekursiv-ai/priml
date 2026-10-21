"""Publish complete native preparation requests and their immutable provenance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    sha256_file,
    source_digests,
    write_manifest,
)
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from pyarrow import Table, parquet

    import numpy as np

    from priml.baselines.casmi2026.chemistry import CandidatePool
    from priml.baselines.casmi2026.isolated import ReactionTable
    from priml.baselines.casmi2026.recipe import RecipeInputs, RecipePlan
else:
    from wrapt import lazy_import

    np = lazy_import("numpy")
    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent.parent


class Formulas(Protocol):
    """Require every independently staged raw or standardized formula decision."""

    def formula(self, smiles: str) -> str:
        """Read the prepared source-compatible formula or raise for missing strings."""
        ...


def write_plan(
    plan: RecipePlan,
    *,
    pool: CandidatePool,
    train_smiles: Sequence[str],
    train_formula: NDArray[np.int16],
    output: Path,
    inputs: Sequence[Path],
) -> None:
    """Write every native first-phase request without a scientific runtime.

    Args:
      plan: Complete parent headers and both external mass-window records.
      pool: Every provided main pool row in original PID order.
      train_smiles: Every training structure in original SID order.
      train_formula: The provided complete SID-aligned I16 ten-element table.
      output: New scratch dataset directory, never an existing destination.
      inputs: Complete input files whose SHA256 digests identify this preparation.

    """
    if train_formula.dtype != np.int16 or train_formula.shape != (
        len(train_smiles),
        10,
    ):
        raise ValueError("Training formula vectors differ from complete SID rows.")
    destination = _destination(output, inputs=inputs)
    _table(
        destination / "reaction-requests.parquet",
        rows=[dict(row.header) for row in plan.reactions],
        fields=(
            "smiles",
            "delta",
            "operation",
            "rule",
            "tolerance",
            "max_steps",
            "max_products",
            "allow_isomerization",
            "application_budget",
            "stage_parents",
            "first_products",
            "later_products",
        ),
    )
    _table(
        destination / "generation.parquet",
        rows=[
            {
                "molecule_id": row.molecule_id,
                "request": index,
                "sid": row.sid,
                "similarity": row.similarity,
                "target": row.target,
            }
            for index, row in enumerate(plan.reactions)
        ],
        fields=("molecule_id", "request", "sid", "similarity", "target"),
    )
    _table(
        destination / "analog-pool.parquet",
        rows=[
            {"smiles": smiles, "formula": formula}
            for smiles, formula in zip(pool.smiles, pool.formulas, strict=True)
        ],
        fields=("smiles", "formula"),
    )
    _table(
        destination / "analog-train.parquet",
        rows=[
            {
                "sid": index,
                "smiles": smiles,
                "formula": _formula_text(
                    cast("NDArray[np.int16]", train_formula[index]),
                ),
            }
            for index, smiles in enumerate(train_smiles)
        ],
        fields=("sid", "smiles", "formula"),
    )
    for name, windows in (("pubchem", plan.pubchem), ("dreams", plan.dreams)):
        _table(
            destination / f"{name}-requests.parquet",
            rows=[
                {
                    "molecule_id": row.molecule_id,
                    "row": row.row,
                    "smiles": row.smiles,
                }
                for row in windows
            ],
            fields=("molecule_id", "row", "smiles"),
        )
    _finish(destination, inputs=inputs)


def write_reaction_inputs(
    reactions: ReactionTable,
    *,
    output: Path,
    inputs: Sequence[Path],
) -> None:
    """Publish every raw product before independent standardization preparation.

    Args:
      reactions: Complete independently prepared requests and their ordered results.
      output: New scratch directory for the raw standardization requests.
      inputs: Every original request/result/manifest data file.

    """
    destination = _destination(output, inputs=inputs)
    rows: list[dict[str, object]] = []
    for index, request in enumerate(reactions.requests):
        for position, product in enumerate(reactions.products(request)):
            combo = product["combo"]
            rows.append(
                {
                    "smiles": product["raw"],
                    "request": index,
                    "position": position,
                    "parent_smiles": request["smiles"],
                    "combo": "+".join(combo) if combo is not None else None,
                },
            )
    _table(
        destination / "raw-products.parquet",
        rows=rows,
        fields=("smiles", "request", "position", "parent_smiles", "combo"),
    )
    _finish(destination, inputs=inputs)


def write_products(
    records: RecipeInputs,
    *,
    chemistry: Formulas,
    output: Path,
    inputs: Sequence[Path],
) -> None:
    """Publish every generated, prior and full-consumer chemistry request.

    Args:
      records: Complete native expansion with every raw alias and repeated path.
      chemistry: Strict independently prepared formula lookup for generated rows.
      output: New scratch dataset directory.
      inputs: All request, pool, reaction and identity files used by the expansion.

    """
    destination = _destination(output, inputs=inputs)
    _table(
        destination / "generated-requests.parquet",
        rows=[
            {"smiles": value, "formula": chemistry.formula(value)}
            for value in records.generated
        ],
        fields=("smiles", "formula"),
    )
    _table(
        destination / "candidate-requests.parquet",
        rows=[{"smiles": value} for value in records.candidates],
        fields=("smiles",),
    )
    _table(
        destination / "derivation-requests.parquet",
        rows=[
            {
                "parent_smiles": row.parent_smiles,
                "combo": row.combo,
                "max_first": row.max_first,
                "max_second": row.max_second,
                "max_mols": row.max_mols,
            }
            for row in records.derivations
        ],
        fields=("parent_smiles", "combo", "max_first", "max_second", "max_mols"),
    )
    _finish(destination, inputs=inputs)


def _destination(output: Path, *, inputs: Sequence[Path]) -> Path:
    destination = validated_output_path(output, protected=inputs)
    if any(path.resolve().is_relative_to(destination.resolve()) for path in inputs):
        raise ValueError("Request output contains a protected input.")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _table(
    path: Path,
    *,
    rows: Sequence[dict[str, object]],
    fields: Sequence[str],
) -> None:
    parquet.write_table(
        Table.from_pydict({field: [row[field] for row in rows] for field in fields}),
        path,
    )


def _formula_text(vector: NDArray[np.int16]) -> str | None:
    if np.any(vector < 0):
        return None
    symbols = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
    return (
        "".join(
            symbol + (str(vector.item(index)) if vector.item(index) != 1 else "")
            for index, symbol in enumerate(symbols)
            if vector.item(index)
        )
        or "+"
    )


def _finish(destination: Path, *, inputs: Sequence[Path]) -> None:
    manifest = {
        "format_version": 1,
        "implementation": "independent Priml preparation requests",
        "inputs": input_digests(inputs),
        "files": source_digests(sorted(destination.glob("*.parquet"))),
        "generator": {
            str(path.relative_to(_CWD)): sha256_file(path)
            for path in sorted(_CWD.rglob("*.py"))
            if not path.name.endswith("_test.py")
        },
    }
    write_manifest(destination, manifest)
