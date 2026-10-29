"""Checks for the recipe preparation launchers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import sys

from pyarrow import Table, parquet
from treekle import from_plain, json_dumps, json_loads

import numpy as np
import pytest

from priml.baselines.casmi2026.chemistry import IdentityTable
from priml.baselines.casmi2026.isolated import (
    DATASET_ROOT,
    IDENTITY_RDKIT,
    normalize_request,
)
from priml.baselines.casmi2026.recipe import WindowRequest
from priml.baselines.casmi2026.scripts.recipe_preparation import (
    PlanningPreparation,
    ProductPreparation,
    RawProductPreparation,
    prepare_product_requests,
    prepare_raw_products,
    prepare_requests,
)
from priml.baselines.casmi2026.testing import (
    identity_rows,
    retrieval_inputs,
    write_identity,
)


if TYPE_CHECKING:
    from priml.baselines.casmi2026.custom_types import QueryRow, ReactionRequest


def test_launcher_factories_are_pure_and_preserve_complete_public_budgets() -> None:
    config = prepare_requests().copy_tree().finalize()
    assert config.max_spectra == 16
    assert config.retrieval.n_analog == 100
    assert config.generator.parents == 6
    assert config.generator.min_similarity == 0.3
    assert config.generator.total == 150
    assert config.generator.per_parent == 60
    assert config.generator.application_budget == 800
    assert config.generator.stage_parents == 24
    assert config.generator.first_products == 32
    assert config.generator.later_products == 16
    assert config.generator.max_steps == 2
    assert isinstance(prepare_raw_products(), RawProductPreparation.Config)
    assert isinstance(prepare_product_requests(), ProductPreparation.Config)


def test_job_directories_resolve_beneath_the_dataset_root_and_agree() -> None:
    root = Path(DATASET_ROOT)
    planning = PlanningPreparation.Config().copy_tree().finalize()
    raw = RawProductPreparation.Config().copy_tree().finalize()
    product = ProductPreparation.Config().copy_tree().finalize()
    assert planning.base_dir == raw.base_dir == product.base_dir == DATASET_ROOT
    assert planning.queries == root / (
        "inputs/enveda-CASMI26-molecule-id-mass-spectra/test.parquet"
    )
    assert planning.library == root / "training/work"
    assert planning.dreams is None
    assert planning.output == product.plan
    assert planning.output == root / "native-chemistry/preparation/requests"
    assert raw.reactions == product.reactions == root / "native-chemistry/reactions"
    assert product.identities == root / "native-chemistry/identities"
    assert product.output == root / "native-chemistry/preparation/products"
    explicit = PlanningPreparation.Config()
    explicit.dreams = "inputs/dreams"
    assert explicit.copy_tree().finalize().dreams == root / "inputs/dreams"


def test_planning_job_uses_native_library_and_all_declared_data_without_rdkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    config = _planning(tmp_path)
    job = config.make()
    job.run()
    headers = parquet.read_table(
        Path(config.output) / "reaction-requests.parquet",
    ).to_pylist()
    assert len(headers) == 1
    assert headers[0]["max_products"] == 60
    assert headers[0]["application_budget"] == 800
    windows = from_plain(
        parquet.read_table(Path(config.output) / "dreams-requests.parquet").to_pylist(),
        list[WindowRequest],
    )
    assert windows == [WindowRequest(molecule_id="q", row=0, smiles="first-dream")]
    pc = from_plain(
        parquet.read_table(Path(config.output) / "pubchem-requests.parquet")
        .column("smiles")
        .to_pylist(),
        list[str],
    )
    assert pc == [" CCO ", "CCO"]
    manifest = json_loads(
        (Path(config.output) / "manifest.json").read_text(),
        dict[str, object],
    )
    inputs = from_plain(manifest["inputs"], dict[str, str])
    assert str(Path(config.queries).resolve()) in inputs
    assert str(Path(config.train_formula).resolve()) in inputs
    assert config.dreams is not None
    assert str((Path(config.dreams) / "dreams_np_mass.npy").resolve()) in inputs
    with pytest.raises(FileExistsError):
        job.run()
    with pytest.raises(
        ValueError,
        match=r"^PlanningPreparation accepts configuration fields only\.$",
    ):
        job.run("--truncate")
    config.output = tmp_path / "no-optional"
    config.dreams = None
    config.make().run()
    assert (
        parquet.read_table(Path(config.output) / "dreams-requests.parquet").num_rows
        == 0
    )


def test_raw_product_job_retains_all_paths_and_refuses_passthrough(
    tmp_path: Path,
) -> None:
    source, _ = _reactions(tmp_path)
    config = RawProductPreparation.Config()
    config.base_dir = None
    config.reactions = source
    config.output = tmp_path / "raw"
    job = config.make()
    job.run()
    rows = parquet.read_table(Path(config.output) / "raw-products.parquet")
    assert rows.num_rows == 2
    assert rows.column("position").to_pylist() == [0, 1]
    assert rows.column("combo").to_pylist() == ["methyl_O", "methyl_O"]
    with pytest.raises(FileExistsError):
        job.run()
    with pytest.raises(
        ValueError,
        match=r"^RawProductPreparation accepts configuration fields only\.$",
    ):
        job.run("--truncate")


def test_product_job_joins_complete_sid_pool_proposal_and_window_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    source, raw = _reactions(tmp_path)
    config = ProductPreparation.Config()
    config.base_dir = None
    config.reactions = source
    config.identities = tmp_path / "identity"
    config.identities.mkdir()
    write_identity(config.identities)
    config.pool_metadata = tmp_path / "pool.parquet"
    config.training = tmp_path / "train.parquet"
    config.second_metadata = tmp_path / "second.parquet"
    config.plan = tmp_path / "plan"
    config.plan.mkdir()
    config.output = tmp_path / "products"
    parquet.write_table(
        Table.from_pydict({"smiles": ["pool", "pool"]}),
        config.pool_metadata,
    )
    parquet.write_table(
        Table.from_pydict({"sid": [1, 0], "smiles": ["sid1", "sid0"]}),
        config.training,
    )
    parquet.write_table(
        Table.from_pydict({"smiles": ["second"]}),
        config.second_metadata,
    )
    for name, values in (("pubchem", [" pc ", " pc "]), ("dreams", ["dream"])):
        parquet.write_table(
            Table.from_pydict(
                {
                    "molecule_id": ["q"] * len(values),
                    "row": list(range(len(values))),
                    "smiles": values,
                },
            ),
            config.plan / f"{name}-requests.parquet",
        )
    job = config.make()
    job.run()
    chemistry = IdentityTable.Config(working_dir=config.identities).make()
    generated = (raw, chemistry.standardize(raw)) * 2
    assert None not in generated
    assert parquet.read_table(
        Path(config.output) / "generated-requests.parquet",
    ).column(
        "smiles",
    ).to_pylist() == list(generated)
    requests = (
        parquet.read_table(Path(config.output) / "candidate-requests.parquet")
        .column("smiles")
        .to_pylist()
    )
    prefix = [
        "pool",
        "pool",
        "sid0",
        "sid1",
        "second",
        *generated,
        " pc ",
        " pc ",
        "dream",
        "parent",
        "CCO",
    ]
    assert requests == prefix + [str(value).strip() for value in prefix]
    assert (
        parquet.read_table(
            Path(config.output) / "derivation-requests.parquet",
        ).to_pylist()
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
    with pytest.raises(FileExistsError):
        job.run()
    with pytest.raises(
        ValueError,
        match=r"^ProductPreparation accepts configuration fields only\.$",
    ):
        job.run("--truncate")


def _planning(root: Path) -> PlanningPreparation.Config:
    config = PlanningPreparation.Config()
    config.base_dir = None
    config.queries = root / "test.parquet"
    config.library = root / "data"
    config.library.mkdir()
    config.pool = config.library
    config.pubchem = root / "pc"
    config.pubchem.mkdir()
    config.train_formula = root / "train_fvec.npy"
    config.output = root / "requests"
    config.retrieval.n_analog = 2
    config.generator.parents = 1
    library, pool, _ = retrieval_inputs()
    for name, values in (
        ("spec_off", library.offsets),
        ("spec_mz", library.mz),
        ("spec_it", library.intensity),
        ("pool_fp", pool.packed),
        ("fp_bits", pool.bits),
        ("pool_frag_off", pool.fragment_offsets),
        ("pool_frag_mass", pool.fragment_mass),
    ):
        np.save(config.library / f"{name}.npy", values)
    parquet.write_table(
        Table.from_pydict(
            {
                "sid": list(range(len(library.smiles))),
                "smiles": library.smiles,
                "key": library.keys,
                "mass": library.structure_mass,
            },
        ),
        config.library / "train_structs.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "sid": library.sid,
                "lib": library.library,
                "mode": library.mode,
                "instr": library.instrument,
                "adduct_ix": library.adduct,
                "precursor_mz": library.precursor,
                "n_clean": library.peak_count,
            },
        ),
        config.library / "spec_meta.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "mass": pool.mass,
                "key": pool.keys,
                "smiles": pool.smiles,
                "formula": pool.formulas,
                "n_heavy": pool.heavy_atoms,
                "src": pool.source,
                "train_sid": pool.train_sid,
            },
        ),
        config.pool / "pool_meta.parquet",
    )
    np.save(config.train_formula, np.ones((len(library.smiles), 10), np.int16))
    query: QueryRow = {
        "molecule_id": "q",
        "spectrum_id": "s",
        "precursor_mz": 101.00727645217,
        "adduct": "[M+H]+",
        "ionization_mode": "positive",
        "collision_energy_ev": None,
        "collision_energy_orig_units": None,
        "collision_energy_orig": None,
        "instrument_type": None,
        "ms2_mzs": [20, 30, 40],
        "ms2_normalized_intensities": [1, 2, 3],
    }
    parquet.write_table(Table.from_pylist([query]), config.queries)
    np.save(config.pubchem / "pc_mass.npy", np.array([100, 100], np.float64))
    text = [b" CCO ", b"CCO"]
    np.save(
        config.pubchem / "pc_off.npy",
        np.concatenate([[0], np.cumsum([len(value) for value in text])]).astype(
            np.int64,
        ),
    )
    np.save(config.pubchem / "pc_smiles.npy", np.frombuffer(b"".join(text), np.uint8))
    config.dreams = root / "optional"
    config.dreams.mkdir()
    np.save(config.dreams / "dreams_np_mass.npy", np.array([100], np.float64))
    np.save(config.dreams / "dreams_np_smiles.npy", np.array(["first-dream"]))
    return config


def _reactions(root: Path) -> tuple[Path, str]:
    source = root / "reactions"
    source.mkdir()
    raw = identity_rows()[0]["smiles"]
    request: ReactionRequest = {"smiles": "parent", "delta": 14.0}
    parquet.write_table(
        Table.from_pylist([normalize_request(request)]),
        source / "requests.parquet",
    )
    row: dict[str, object] = {
        "input_smiles": "parent",
        "products": [raw, raw],
        "plans": [[0], [0]],
        "matches": [],
        "applications": "all",
    }
    parquet.write_table(Table.from_pylist([row]), source / "results.parquet")
    parquet.write_table(
        Table.from_pydict({"name": ["methyl_O"]}),
        source / "rules.parquet",
    )
    (source / "manifest.json").write_text(
        json_dumps(
            {"format_version": 1, "rdkit": IDENTITY_RDKIT, "default_products": 60},
        ),
    )
    return source, raw


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
