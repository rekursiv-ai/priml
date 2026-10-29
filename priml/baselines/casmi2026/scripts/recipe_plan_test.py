"""Checks for the recipe request writers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib
import sys

from pyarrow import Table, parquet
from treekle import from_plain, json_dumps, json_loads

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import (
    IDENTITY_RDKIT,
    ReactionTable,
    formula_vector,
    normalize_request,
)
from priml.baselines.casmi2026.recipe import (
    GenerationRequest,
    PriorRequest,
    RecipeInputs,
    RecipePlan,
    WindowRequest,
)
from priml.baselines.casmi2026.scripts.recipe_plan import (
    write_plan,
    write_products,
    write_reaction_inputs,
)
from priml.baselines.casmi2026.testing import retrieval_inputs


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import ReactionRequest


_PACKAGE: Final = _CWD.parent


def test_complete_tables_vectors_duplicate_windows_and_byte_receipt_without_rdkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    _, pool, _ = retrieval_inputs()
    request: ReactionRequest = {"smiles": "parent", "delta": 14.0}
    header = normalize_request(request)
    plan = RecipePlan(
        reactions=(
            GenerationRequest(
                molecule_id="q",
                sid=2,
                similarity=0.5,
                target=100.0,
                header=header,
            ),
        ),
        pubchem=(WindowRequest(molecule_id="q", row=5, smiles=" pc "),) * 2,
        dreams=(WindowRequest(molecule_id="r", row=4, smiles="dream"),),
    )
    original = tmp_path / "provided"
    original.write_bytes(b"complete input bytes")
    output = tmp_path / "requests"
    vectors = np.array([[2, 6, 0, 1, 0, 0, 0, 0, 0, 0], [-1] * 10, [0] * 10], np.int16)
    write_plan(
        plan,
        pool=pool,
        train_smiles=("train0", "bad", "zero"),
        train_formula=vectors,
        output=output,
        inputs=[original],
    )
    assert parquet.read_table(output / "reaction-requests.parquet").to_pylist() == [
        dict(header),
    ]
    assert parquet.read_table(output / "generation.parquet").to_pylist() == [
        {
            "molecule_id": "q",
            "request": 0,
            "sid": 2,
            "similarity": 0.5,
            "target": 100.0,
        },
    ]
    assert parquet.read_table(output / "analog-pool.parquet").to_pylist() == [
        {"smiles": smiles, "formula": formula}
        for smiles, formula in zip(pool.smiles, pool.formulas, strict=True)
    ]
    training = parquet.read_table(output / "analog-train.parquet")
    assert training.column("sid").to_pylist() == [0, 1, 2]
    assert training.column("smiles").to_pylist() == ["train0", "bad", "zero"]
    formulas = from_plain(training.column("formula").to_pylist(), list[str | None])
    assert formulas == ["C2H6O", None, "+"]
    np.testing.assert_array_equal(
        np.stack([formula_vector(formula) for formula in formulas]),
        vectors,
    )
    assert (
        parquet.read_table(output / "pubchem-requests.parquet").to_pylist()
        == [{"molecule_id": "q", "row": 5, "smiles": " pc "}] * 2
    )
    assert parquet.read_table(output / "dreams-requests.parquet").to_pylist() == [
        {"molecule_id": "r", "row": 4, "smiles": "dream"},
    ]
    _receipt(output, source=original, count=6)
    with pytest.raises(FileExistsError):
        write_plan(
            plan,
            pool=pool,
            train_smiles=("a", "b", "c"),
            train_formula=vectors,
            output=output,
            inputs=[original],
        )
    with pytest.raises(
        ValueError,
        match=r"^Training formula vectors differ from complete SID rows\.$",
    ):
        write_plan(
            plan,
            pool=pool,
            train_smiles=("a",),
            train_formula=vectors,
            output=tmp_path / "bad-shape",
            inputs=[original],
        )
    assert not (tmp_path / "bad-shape").exists()
    with pytest.raises(
        ValueError,
        match=r"^Training formula vectors differ from complete SID rows\.$",
    ):
        write_plan(
            plan,
            pool=pool,
            train_smiles=("a", "b", "c"),
            train_formula=cast("NDArray[np.int16]", vectors.astype(np.int32)),
            output=tmp_path / "bad-dtype",
            inputs=[original],
        )
    with pytest.raises(
        ValueError,
        match=r"^Request output contains a protected input\.$",
    ):
        write_plan(
            plan,
            pool=pool,
            train_smiles=("a", "b", "c"),
            train_formula=vectors,
            output=tmp_path,
            inputs=[original],
        )


class _Formulas:
    def formula(self, smiles: str) -> str:
        return {"raw": "raw", "standard": "C2H6O", "invalid": "invalid"}[smiles]


def test_every_raw_proposal_and_final_request_including_empty_tables(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    requests: list[ReactionRequest] = [
        {"smiles": "parent", "delta": 14.0},
        {"smiles": "empty", "delta": 0},
    ]
    parquet.write_table(
        Table.from_pylist([normalize_request(row) for row in requests]),
        source / "requests.parquet",
    )
    rows: list[dict[str, object]] = [
        {
            "input_smiles": "parent",
            "products": ["raw", "invalid", "raw"],
            "plans": [[0], [], [1, 0]],
            "matches": [],
            "applications": "all",
        },
        {
            "input_smiles": "empty",
            "products": [],
            "plans": [],
            "matches": [],
            "applications": "zero",
        },
    ]
    parquet.write_table(Table.from_pylist(rows), source / "results.parquet")
    parquet.write_table(
        Table.from_pydict({"name": ["methyl_O", "dehexose2"]}),
        source / "rules.parquet",
    )
    (source / "manifest.json").write_text(
        json_dumps(
            {"format_version": 1, "rdkit": IDENTITY_RDKIT, "default_products": 60},
        ),
    )
    inputs = sorted(source.iterdir())
    reactions = ReactionTable(source)
    output = tmp_path / "raw"
    write_reaction_inputs(reactions, output=output, inputs=inputs)
    assert parquet.read_table(output / "raw-products.parquet").to_pylist() == [
        {
            "smiles": "raw",
            "request": 0,
            "position": 0,
            "parent_smiles": "parent",
            "combo": "methyl_O",
        },
        {
            "smiles": "invalid",
            "request": 0,
            "position": 1,
            "parent_smiles": "parent",
            "combo": None,
        },
        {
            "smiles": "raw",
            "request": 0,
            "position": 2,
            "parent_smiles": "parent",
            "combo": "dehexose2+methyl_O",
        },
    ]
    _receipt(output, source=inputs[0], count=1)
    records = RecipeInputs(
        generated=("raw", "standard", "invalid", "raw"),
        candidates=("pool", "parent", " raw ", "raw"),
        derivations=(
            PriorRequest(
                parent_smiles="parent",
                combo="methyl_O",
                max_first=32,
                max_second=16,
                max_mols=24,
            ),
        )
        * 2,
    )
    final = tmp_path / "final"
    write_products(records, chemistry=_Formulas(), output=final, inputs=inputs)
    assert parquet.read_table(final / "generated-requests.parquet").to_pylist() == [
        {"smiles": value, "formula": _Formulas().formula(value)}
        for value in records.generated
    ]
    assert parquet.read_table(final / "candidate-requests.parquet").column(
        "smiles",
    ).to_pylist() == list(records.candidates)
    assert (
        parquet.read_table(final / "derivation-requests.parquet").to_pylist()
        == [
            {
                "parent_smiles": "parent",
                "combo": "methyl_O",
                "max_first": 32,
                "max_second": 16,
                "max_mols": 24,
            },
        ]
        * 2
    )
    _receipt(final, source=inputs[0], count=3)
    empty = tmp_path / "empty"
    write_products(
        RecipeInputs(generated=(), candidates=(), derivations=()),
        chemistry=_Formulas(),
        output=empty,
        inputs=[],
    )
    for name in ("generated", "candidate", "derivation"):
        assert parquet.read_table(empty / f"{name}-requests.parquet").num_rows == 0
    with pytest.raises(KeyError, match=r"^'missing'$"):
        write_products(
            RecipeInputs(generated=("missing",), candidates=(), derivations=()),
            chemistry=_Formulas(),
            output=tmp_path / "missing",
            inputs=[],
        )
    assert not (tmp_path / "missing/manifest.json").exists()


def _receipt(directory: Path, *, source: Path, count: int) -> None:
    manifest = json_loads((directory / "manifest.json").read_text(), dict[str, object])
    assert manifest["format_version"] == 1
    assert manifest["implementation"] == "independent Priml preparation requests"
    inputs = from_plain(manifest["inputs"], dict[str, str])
    assert (
        inputs[str(source.resolve())] == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    files = from_plain(manifest["files"], dict[str, str])
    assert len(files) == count
    for name, digest in files.items():
        assert digest == hashlib.sha256((directory / name).read_bytes()).hexdigest()
    generator = from_plain(manifest["generator"], dict[str, str])
    for name, digest in generator.items():
        assert digest == hashlib.sha256((_PACKAGE / name).read_bytes()).hexdigest()
    assert "recipe.py" in generator
    assert "scripts/recipe_plan.py" in generator


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
