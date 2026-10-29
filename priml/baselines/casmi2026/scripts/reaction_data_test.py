"""Checks for the reaction data writer."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from pyarrow import Table, parquet
from treekle import from_plain, json_loads

import pytest

from priml.baselines.casmi2026.custom_types import ReactionRequest
from priml.baselines.casmi2026.isolated import (
    IDENTITY_RDKIT,
    normalize_request,
    reaction_rules,
)
from priml.baselines.casmi2026.scripts import derivatives, reaction_data
from priml.baselines.casmi2026.scripts.derivatives import DerivativeGenerator


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from rdkit import Chem

    from priml.baselines.casmi2026.isolated import Reaction


@dataclass(frozen=True, slots=True, kw_only=True)
class _Structure:
    name: str


def _mol_from_smiles(smiles: str) -> _Structure | None:
    return None if smiles == "invalid" else _Structure(name=smiles)


def _mol_to_smiles(molecule: Chem.Mol) -> str:
    return cast("_Structure", molecule).name.removeprefix("duplicate:")


def _sanitize(molecule: _Structure) -> None:
    if molecule.name == "bad":
        raise ValueError("Invalid reaction product.")


def _run_reactants(
    molecules: tuple[_Structure],
) -> tuple[tuple[_Structure, ...], ...]:
    molecule = molecules[0]
    return tuple(
        (_Structure(name=name), _Structure(name="unused coproduct"))
        for name in (
            "bad",
            molecule.name,
            "duplicate:" + molecule.name,
            molecule.name + "A",
            molecule.name + "B",
        )
    )


def _reaction_from_smarts(smarts: str) -> SimpleNamespace:
    del smarts
    return SimpleNamespace(RunReactants=_run_reactants)


def _install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the lazily imported scientific names by one fake namespace."""
    library = SimpleNamespace(
        MolFromSmiles=_mol_from_smiles,
        MolToSmiles=_mol_to_smiles,
        SanitizeMol=_sanitize,
    )
    monkeypatch.setattr(derivatives, "Chem", library)
    monkeypatch.setattr(derivatives, "ReactionFromSmarts", _reaction_from_smarts)
    monkeypatch.setattr(reaction_data, "Chem", library)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Preparation:
    version: str = IDENTITY_RDKIT


def test_reaction_asset_writer_keeps_all_requests_outputs_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    generator = DerivativeGenerator()
    generator.rules = reaction_rules()[:3]

    def construct(
        *,
        rules: Sequence[Reaction] | None,
        on_application: Callable[[Chem.Mol, Reaction, int, int], None],
    ) -> DerivativeGenerator:
        assert rules is None
        generator.on_application = on_application
        return generator

    source = tmp_path / "requests.parquet"
    requests = [
        {"smiles": "parent", "delta": 15.9949146196},
        {"smiles": "parent", "delta": 0.0, "operation": "apply", "rule": 0},
        {"smiles": "invalid", "delta": 0.0, "operation": "apply", "rule": 0},
        {"smiles": "parent", "delta": 1000.0, "operation": "match"},
    ]
    fields = {name for row in requests for name in row}
    parquet.write_table(
        Table.from_pydict(
            {name: [row.get(name) for row in requests] for name in fields},
        ),
        source,
    )
    output = tmp_path / "reactions"
    with patch.object(reaction_data, "DerivativeGenerator", side_effect=construct):
        reaction_data.prepare_reactions(
            source,
            output=output,
            chemistry=_Preparation(),
        )
    assert parquet.read_table(output / "requests.parquet").to_pylist() == [
        normalize_request(from_plain(row, ReactionRequest)) for row in requests
    ]
    assert parquet.read_table(output / "results.parquet").to_pylist() == [
        {
            "input_smiles": "parent",
            "products": ["parentA", "parentB"],
            "plans": [[0], [0]],
            "matches": [],
            "applications": "\n".join(
                f"parent\t{rule.name}\t32\t3" for rule in generator.rules
            ),
        },
        {
            "input_smiles": "parent",
            "products": ["parent", "parentA", "parentB"],
            "plans": [[0], [0], [0]],
            "matches": [],
            "applications": "parent\thydroxyl_arom\t64\t3",
        },
        *(
            {
                "input_smiles": value,
                "products": [],
                "plans": [],
                "matches": [],
                "applications": "",
            }
            for value in ("invalid", "parent")
        ),
    ]
    assert parquet.read_table(output / "products.parquet").to_pylist() == [
        {"request": request, "position": position, "smiles": smiles, "plan": [0]}
        for request, values in (
            (0, ["parentA", "parentB"]),
            (1, ["parent", "parentA", "parentB"]),
        )
        for position, smiles in enumerate(values)
    ]
    vocabulary = parquet.read_table(output / "rules.parquet")
    assert from_plain(vocabulary.column("name").to_pylist(), list[str]) == [
        rule.name for rule in generator.rules
    ]
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["requests"] == 4
    assert manifest["products"] == 5
    assert manifest["rdkit"] == IDENTITY_RDKIT
    assert manifest["format_version"] == 1
    assert manifest["default_products"] == 60
    assert set(from_plain(manifest["input"], dict[str, str])) == {str(source.resolve())}
    assert set(from_plain(manifest["generator"], dict[str, str])) == {
        "reaction_data.py",
        "derivatives.py",
        "../chemistry.py",
        "rdkit_backend.py",
        "../v6_reactions.tsv",
        "../main_engine.py",
    }
    with (
        patch.object(reaction_data, "DerivativeGenerator", side_effect=construct),
        pytest.raises(FileExistsError),
    ):
        reaction_data.prepare_reactions(
            source,
            output=output,
            chemistry=_Preparation(),
        )


def test_match_operation_records_plans_without_products(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    source = tmp_path / "requests.parquet"
    parquet.write_table(
        Table.from_pylist(
            [{"smiles": "parent", "delta": 15.9949146196, "operation": "match"}],
        ),
        source,
    )
    reaction_data.prepare_reactions(
        source,
        output=tmp_path / "out",
        chemistry=_Preparation(),
    )
    row = parquet.read_table(tmp_path / "out" / "results.parquet").to_pylist()[0]
    assert row["products"] == []
    assert row["plans"] == []
    assert [0] in from_plain(row["matches"], list[list[int]])
    assert parquet.read_table(tmp_path / "out" / "products.parquet").num_rows == 0


@pytest.mark.parametrize(
    "rule",
    [-1, len(reaction_rules()) + 5],
)
def test_apply_rule_outside_the_vocabulary_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rule: int,
) -> None:
    _install(monkeypatch)
    source = tmp_path / "requests.parquet"
    parquet.write_table(
        Table.from_pylist(
            [{"smiles": "parent", "delta": 0.0, "operation": "apply", "rule": rule}],
        ),
        source,
    )
    with pytest.raises(
        ValueError,
        match=r"^Reaction rule index exceeds the vocabulary\.$",
    ):
        reaction_data.prepare_reactions(
            source,
            output=tmp_path / "out",
            chemistry=_Preparation(),
        )
    assert not (tmp_path / "out").exists()


def test_empty_request_table_writes_empty_outputs(tmp_path: Path) -> None:
    source = tmp_path / "requests.parquet"
    parquet.write_table(Table.from_pydict({"smiles": [], "delta": []}), source)
    reaction_data.prepare_reactions(
        source,
        output=tmp_path / "empty",
        chemistry=_Preparation(),
    )
    assert parquet.read_table(tmp_path / "empty" / "results.parquet").num_rows == 0
    assert parquet.read_table(tmp_path / "empty" / "products.parquet").num_rows == 0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
