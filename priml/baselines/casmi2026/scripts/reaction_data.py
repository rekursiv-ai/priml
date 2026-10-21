"""Generate ordered reaction-product data from explicit scientific requests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from treekle import from_plain

from priml.baselines.casmi2026.custom_types import ReactionRequest
from priml.baselines.casmi2026.isolated import matching_plans, normalize_request
from priml.baselines.casmi2026.scripts.derivatives import (
    Derivative,
    DerivativeGenerator,
)
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    sha256_file,
    write_manifest,
)
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyarrow import Table, parquet
    from rdkit import Chem

    from priml.baselines.casmi2026.isolated import Reaction
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    Chem = lazy_import("rdkit.Chem")


_THIS: Final = Path(__file__).resolve()


class ReactionChemistry(Protocol):
    """Checked scientific-library release behind the molecular primitives."""

    @property
    def version(self) -> str:
        """The checked scientific-library release."""
        ...


def prepare_reactions(
    input_path: Path,
    *,
    output: Path,
    chemistry: ReactionChemistry,
    rules: Sequence[Reaction] | None = None,
    default_products: int = 60,
) -> None:
    """Write complete requested plans, raw products and first-plan provenance.

    Args:
      input_path: Arrow request table with explicit parent and mass difference.
      output: New external dataset directory.
      chemistry: Checked isolated scientific-library context.
      rules: Explicit generation vocabulary, or the first-generation rules.
      default_products: Final output cap when the request has no explicit cap.

    """
    destination = validated_output_path(output, protected=[input_path])
    requests = [
        from_plain(
            {key: value for key, value in row.items() if value is not None},
            ReactionRequest,
        )
        for row in parquet.read_table(input_path).to_pylist()
    ]
    applications: list[str] = []

    def observe(molecule: Chem.Mol, rule: Reaction, limit: int, count: int) -> None:
        applications.append(
            f"{Chem.MolToSmiles(molecule)}\t{rule.name}\t{limit}\t{count}",
        )

    generator = DerivativeGenerator(
        rules=rules,
        on_application=observe,
    )
    columns: dict[str, list[object]] = {
        name: []
        for name in ("input_smiles", "products", "plans", "matches", "applications")
    }
    products: dict[str, list[object]] = {
        name: [] for name in ("request", "position", "smiles", "plan")
    }
    headers = [
        normalize_request(request, default_products=default_products)
        for request in requests
    ]
    for row_index, header in enumerate(headers):
        applications.clear()
        operation = header["operation"]
        found: list[Derivative] = []
        matches: list[tuple[int, ...]] = []
        if operation == "match":
            matches = matching_plans(
                generator.rules,
                header["delta"],
                tolerance=header["tolerance"],
                max_reactions=header["max_steps"],
            )
        elif operation == "apply":
            rule_index = header["rule"]
            if rule_index < 0 or rule_index >= len(generator.rules):
                raise ValueError("Reaction rule index exceeds the vocabulary.")
            molecule = Chem.MolFromSmiles(header["smiles"])
            if molecule is not None:
                found = [
                    Derivative(
                        smiles=Chem.MolToSmiles(product),
                        plan=(rule_index,),
                    )
                    for product in generator.apply(
                        molecule,
                        rule=generator.rules[rule_index],
                        max_products=header["max_products"],
                    )
                ]
        else:
            found = generator.derive(
                header["smiles"],
                delta=header["delta"],
                tolerance=header["tolerance"],
                max_reactions=header["max_steps"],
                max_products=header["max_products"],
                allow_isomerization=header["allow_isomerization"],
                application_budget=header["application_budget"],
                stage_parents=header["stage_parents"],
                first_products=header["first_products"],
                later_products=header["later_products"],
            )
        columns["input_smiles"].append(header["smiles"])
        columns["products"].append([product.smiles for product in found])
        columns["plans"].append([list(product.plan) for product in found])
        columns["matches"].append([list(plan) for plan in matches])
        columns["applications"].append("\n".join(applications))
        for position, product in enumerate(found):
            for name, value in {
                "request": row_index,
                "position": position,
                "smiles": product.smiles,
                "plan": list(product.plan),
            }.items():
                products[name].append(value)
    destination.mkdir(parents=True, exist_ok=False)
    parquet.write_table(
        Table.from_pylist([dict(row) for row in headers]),
        destination / "requests.parquet",
    )
    parquet.write_table(Table.from_pydict(columns), destination / "results.parquet")
    parquet.write_table(Table.from_pydict(products), destination / "products.parquet")
    parquet.write_table(
        Table.from_pydict(
            {
                "name": [rule.name for rule in generator.rules],
                "smarts": [rule.smarts for rule in generator.rules],
                "mass": [rule.mass for rule in generator.rules],
            },
        ),
        destination / "rules.parquet",
    )
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": chemistry.version,
        "default_products": default_products,
        "requests": len(requests),
        "products": len(products["smiles"]),
        "input": input_digests([input_path]),
        "generator": {
            str(path.relative_to(_THIS.parent, walk_up=True)): sha256_file(path)
            for path in (
                _THIS,
                _THIS.with_name("derivatives.py"),
                _THIS.parent.parent / "chemistry.py",
                _THIS.with_name("rdkit_backend.py"),
                _THIS.parent.parent / "v6_reactions.tsv",
                _THIS.parent.parent / "main_engine.py",
            )
        },
    }
    write_manifest(destination, manifest)
