"""Generate complete chemistry paths and molecular identities in isolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, NotRequired, TypedDict

from treekle import from_plain, to_plain

from priml.baselines.casmi2026.scripts.derivation_chemistry import (
    DerivationChemistry,
)
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from pyarrow import Table, parquet
    from rdkit import Chem

    from priml.baselines.casmi2026.custom_types import (
        MolecularIdentity,
        ProposalHeader,
        ProposalPath,
        ProposalRow,
    )
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")


_THIS: Final = Path(__file__).resolve()


class DerivationRequest(TypedDict):
    """One explicit parent/transform pair and its original enumeration budgets."""

    parent_smiles: str
    combo: str
    max_first: NotRequired[int]
    max_second: NotRequired[int]
    max_mols: NotRequired[int]


def prepare_derivations(
    input_path: Path,
    *,
    identities: Path | None,
    output: Path,
    version: str,
) -> None:
    """Prepare all requested terminal paths before native prior arithmetic.

    Args:
      input_path: Explicit parent/combo requests and optional enumeration limits.
      identities: Optional additional candidate and derivation-log SMILES table.
      output: New external data directory.
      version: Required isolated scientific release.

    """
    inputs = [input_path, *([identities] if identities is not None else [])]
    destination = validated_output_path(output, protected=inputs)
    requests = from_plain(
        [
            {key: value for key, value in row.items() if value is not None}
            for row in parquet.read_table(input_path).to_pylist()
        ],
        list[DerivationRequest],
    )
    chemistry = DerivationChemistry(version)
    headers: list[ProposalHeader] = []
    paths: list[ProposalRow] = []
    requested: dict[str, None] = {}
    for index, request in enumerate(requests):
        budgets = _budgets(request)
        found, applications = _paths(chemistry, request=request, budgets=budgets)
        combo = tuple(request["combo"].split("+"))
        headers.append(
            {
                "parent_smiles": request["parent_smiles"],
                "combo": combo,
                "max_first": budgets.first,
                "max_second": budgets.second,
                "max_mols": budgets.molecules,
                "terminal_count": len(found),
                "applications": "\n".join(applications),
            },
        )
        requested.setdefault(request["parent_smiles"], None)
        for position, path in enumerate(found):
            paths.append({**path, "request": index, "position": position})  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.
            for value in (path["raw"], path["standardized"]):
                if value is not None:
                    requested.setdefault(value, None)
    if identities is not None:
        for value in from_plain(
            parquet.read_table(identities, columns=["smiles"])
            .column("smiles")
            .to_pylist(),
            list[str],
        ):
            requested.setdefault(value, None)
    molecular: list[MolecularIdentity] = []
    for value in requested:
        key, skeleton = chemistry.identity(value)
        molecular.append({"smiles": value, "key": key, "skeleton": skeleton})
    destination.mkdir(parents=True, exist_ok=False)
    for name, rows, fields in (
        (
            "requests",
            headers,
            (
                "parent_smiles",
                "combo",
                "max_first",
                "max_second",
                "max_mols",
                "terminal_count",
                "applications",
            ),
        ),
        (
            "paths",
            paths,
            (
                "request",
                "position",
                "raw",
                "standardized",
                "key",
                "skeleton",
                "environments",
            ),
        ),
        ("identities", molecular, ("smiles", "key", "skeleton")),
    ):
        records = from_plain(to_plain(rows), list[dict[str, object]])
        parquet.write_table(
            Table.from_pydict(
                {field: [row[field] for row in records] for field in fields},
            ),
            destination / f"{name}.parquet",
        )
    manifest = {
        "format_version": 1,
        "rdkit": version,
        "requests": len(headers),
        "paths": len(paths),
        "identities": len(molecular),
        "inputs": input_digests(inputs),
        "generator": source_digests(
            (
                _THIS,
                _THIS.with_name("derivation_chemistry.py"),
                _THIS.with_name("rdkit_backend.py"),
                _THIS.parent.parent / "chemistry.py",
                _THIS.parent.parent / "v6_reactions.tsv",
            ),
        ),
    }
    write_manifest(destination, manifest)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Budgets:
    """Enumeration caps with the released defaults applied once."""

    first: int
    second: int
    molecules: int


def _budgets(request: DerivationRequest) -> _Budgets:
    return _Budgets(
        first=request.get("max_first", 32),
        second=request.get("max_second", 16),
        molecules=request.get("max_mols", 24),
    )


def _paths(
    chemistry: DerivationChemistry,
    *,
    request: DerivationRequest,
    budgets: _Budgets,
) -> tuple[list[ProposalPath], list[str]]:
    molecule = chemistry.chem.MolFromSmiles(request["parent_smiles"])
    front: list[tuple[Chem.Mol, tuple[tuple[int, int, int], ...]]] = (
        [(molecule, ())] if molecule is not None else []
    )
    applications: list[str] = []
    for depth, name in enumerate(request["combo"].split("+")):
        next_stage: list[tuple[Chem.Mol, tuple[tuple[int, int, int], ...]]] = []
        limit = budgets.first if depth == 0 else budgets.second
        for parent, history in front[: budgets.molecules]:
            environments = chemistry.environments(parent)
            products = chemistry.products(
                parent,
                rule=chemistry.rules[name],
                limit=limit,
            )
            applications.append(
                f"{chemistry.chem.MolToSmiles(parent)}\t{name}\t{limit}\t{len(products)}",
            )
            next_stage.extend(
                (product, (*history, environments.get(site, (-1, -1, -1))))
                for product, site in products
            )
        front = next_stage
        if not front:
            break
    result: list[ProposalPath] = []
    for product, history in front:
        try:
            raw = chemistry.chem.MolToSmiles(product)
        except Exception:  # noqa: BLE001 -- Preserve a known-invalid terminal when SDK serialization fails.
            raw = None
        standardized = chemistry.standard.standardize(raw) if raw is not None else None
        key, skeleton = (
            chemistry.identity(standardized)
            if standardized is not None
            else (None, None)
        )
        result.append(
            {
                "raw": raw,
                "standardized": standardized,
                "key": key,
                "skeleton": skeleton,
                "environments": history,
            },
        )
    return result, applications
