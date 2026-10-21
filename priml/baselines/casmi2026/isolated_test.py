"""Checks for the isolated preparation surface."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast, get_type_hints

import json
import math
import pickle
import sys

from numpy.typing import NDArray
from pyarrow import Table, parquet
from torch import Tensor
from treekle import ReadError, from_plain, to_plain

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026.custom_types import ReactionHeader, ReactionRequest
from priml.baselines.casmi2026.isolated import (
    ELEMENT_MASSES,
    HYDROGEN_MASS,
    IDENTITY_RDKIT,
    SECOND_NAMES,
    SIMULATOR_RDKIT,
    AnalogTable,
    FragnetAtoms,
    IntrinsicTable,
    Reaction,
    ReactionTable,
    _labels,
    _latin1,
    _properties,
    adduct_index,
    adduct_names,
    adduct_parameters,
    formula_vector,
    main_ion_deltas,
    matching_plans,
    molecular_record,
    neutral_mass,
    normalize_request,
    precursor_mass,
    reaction_key,
    reaction_rules,
    read_pool_metadata,
    record_fields,
    second_adduct_parameters,
    second_neutral_mass,
    v6_reaction_rules,
)
from priml.baselines.casmi2026.main_engine_test import Prepared
from priml.baselines.casmi2026.testing import float32_comparands
from priml.testing.golden import assert_tensor_golden, read_tensors


_CWD: Final = Path(__file__).resolve().parent
_BASELINE: Final = _CWD


def test_element_masses_are_monoisotopic_and_lead_with_the_organic_elements() -> None:
    assert len(ELEMENT_MASSES) == 82
    assert list(ELEMENT_MASSES)[:4] == ["C", "H", "N", "O"]
    assert ELEMENT_MASSES["C"] == 12.0
    assert ELEMENT_MASSES["O"] == 15.9949146196
    assert HYDROGEN_MASS == ELEMENT_MASSES["H"] == 1.00782503207
    assert all(mass > 0 for mass in ELEMENT_MASSES.values())


def test_adduct_parameters_decode_every_vocabulary_name_in_token_order() -> None:
    assert [(name, adduct_parameters(name)) for name in adduct_names()] == [
        ("[M+H]+", (1, 1, 1.00727645217, 1)),
        ("[M+NH4]+", (1, 1, 18.03382555318, 1)),
        ("[M-H2O+H]+", (1, 1, -17.00328823157, 1)),
        ("[M-2H2O+H]+", (1, 1, -35.01385291531, 1)),
        ("[M+Na]+", (1, 1, 22.989220701, 1)),
        ("[M+K]+", (1, 1, 38.9631581001, 1)),
        ("[M-H]-", (1, 1, -1.00727645217, -1)),
        ("[M-H2O-H]-", (1, 1, -19.01784113591, -1)),
        ("[M+CH2O2-H]-", (1, 1, 44.99820285116999, -1)),
        ("[M+Cl]-", (1, 1, 34.969401259899996, -1)),
        ("[M+Li]+", (1, 1, 7.0154559701, 1)),
        ("[M-H2O]+", (1, 1, -18.01111326364, 1)),
        ("[M]+", (1, 1, -0.0005485799, 1)),
        ("[M+2H]2+", (1, 2, 2.01455290434, 1)),
        ("[M+3H]3+", (1, 3, 3.02182935651, 1)),
        ("[M+H+Na]2+", (1, 2, 23.996497153169997, 1)),
        ("[M+HNa]2+", (1, 2, 23.996497153169997, 1)),
        ("[M+2Na]2+", (1, 2, 45.978441402, 1)),
        ("[M+2Na-H]+", (1, 1, 44.97116494983, 1)),
        ("[M+Ca]2+", (1, 2, 39.961493820200005, 1)),
        ("[M+CH3OH+H]+", (1, 1, 33.03349120005, 1)),
        ("[M+CH3CN+H]+", (1, 1, 42.03382555318, 1)),
        ("[M+CH3CN+Na]+", (1, 1, 64.01576980201, 1)),
        ("[M+C2H4N]+", (1, 1, 42.03382555318, 1)),
        ("[M+CH5O]+", (1, 1, 33.03349120005, 1)),
        ("[M-H2N]+", (1, 1, -16.019272648839998, 1)),
        ("[M-NH3+H]+", (1, 1, -16.01927264884, 1)),
        ("[M-H5O3]+", (1, 1, -53.02441759905, 1)),
        ("[M-CH3]+", (1, 1, -15.02402367611, 1)),
        ("[M+H-CH3]+", (1, 1, -14.01619864404, 1)),
        ("[M-2H]-", (1, 1, -2.01510148424, -1)),
        ("[M-2H]2-", (1, 2, -2.01455290434, -1)),
        ("[M-H2]2-", (1, 2, -2.01455290434, -1)),
        ("[M]-", (1, 1, 0.0005485799, -1)),
        ("[M+HCOO]-", (1, 1, 44.99820285116999, -1)),
        ("[M+C2H4O2-H]-", (1, 1, 59.01385291531, -1)),
        ("[M+CH3COO]-", (1, 1, 59.01385291531, -1)),
        ("[M+Br]-", (1, 1, 78.9188856799, -1)),
        ("[M+Na-2H]-", (1, 1, 20.974667796659997, -1)),
        ("[M+K-2H]-", (1, 1, 36.94860519576, -1)),
        ("[M+Na-H2]-", (1, 1, 20.974667796659997, -1)),
        ("[M+K-H2]-", (1, 1, 36.94860519576, -1)),
        ("[M-CH3]-", (1, 1, -15.02292651631, -1)),
        ("[M-C3H6NO2]-", (1, 1, -88.03930485652, -1)),
        ("[M+C2F3O2]-", (1, 1, 112.98558747909999, -1)),
        ("[M+CF3COO]-", (1, 1, 112.98558747909999, -1)),
        ("[2M+H]+", (2, 1, 1.00727645217, 1)),
        ("[2M+Na]+", (2, 1, 22.989220701, 1)),
        ("[2M+K]+", (2, 1, 38.9631581001, 1)),
        ("[2M+NH4]+", (2, 1, 18.03382555318, 1)),
        ("[2M+Ca]2+", (2, 2, 39.961493820200005, 1)),
        ("[2M-H]-", (2, 1, -1.00727645217, -1)),
        ("[2M+CH2O2-H]-", (2, 1, 44.99820285116999, -1)),
        ("[2M+C2H4O2-H]-", (2, 1, 59.01385291531, -1)),
        ("[2M+Na-2H]-", (2, 1, 20.974667796659997, -1)),
        ("[2M+Cl]-", (2, 1, 34.969401259899996, -1)),
        ("[3M+H]+", (3, 1, 1.00727645217, 1)),
        ("[3M+Na]+", (3, 1, 22.989220701, 1)),
        ("[3M-H]-", (3, 1, -1.00727645217, -1)),
    ]
    assert [adduct_index(name) for name in ("[M+H]+", "[3M-H]-", "unknown", None)] == [
        0,
        58,
        59,
        59,
    ]


def test_second_adduct_parameters_keep_proton_first_offsets_in_table_order() -> None:
    assert [(name, second_adduct_parameters(name)) for name in SECOND_NAMES] == [
        ("[M+H]+", (1, 1, 1.00727645217)),
        ("[M+NH4]+", (1, 1, 18.03382555318)),
        ("[M+Na]+", (1, 1, 22.989220701)),
        ("[M+K]+", (1, 1, 38.9631581001)),
        ("[M-H2O+H]+", (1, 1, -17.00328823157)),
        ("[M-2H2O+H]+", (1, 1, -35.013852915310004)),
        ("[M+2H]2+", (1, 2, 2.01455290434)),
        ("[M]+", (1, 1, -0.0005485799)),
        ("[M-H2O]+", (1, 1, -18.01111326364)),
        ("[M+CH3OH+H]+", (1, 1, 33.033491200049994)),
        ("[M+CH3CN+H]+", (1, 1, 42.03382555318)),
        ("[M-H]-", (1, 1, -1.00727645217)),
        ("[M-H2O-H]-", (1, 1, -19.01784113591)),
        ("[M+CH2O2-H]-", (1, 1, 44.99820285117)),
        ("[M+C2H4O2-H]-", (1, 1, 59.013852915310004)),
        ("[M+Cl]-", (1, 1, 34.969401259899996)),
        ("[M]-", (1, 1, 0.0005485799)),
        ("[M-2H]-", (1, 2, -2.01455290434)),
        ("[M+Na-2H]-", (1, 1, 20.97521637656)),
        ("[2M+H]+", (2, 1, 1.00727645217)),
        ("[2M+Na]+", (2, 1, 22.989220701)),
        ("[2M+NH4]+", (2, 1, 18.03382555318)),
        ("[2M+K]+", (2, 1, 38.9631581001)),
        ("[2M-H]-", (2, 1, -1.00727645217)),
        ("[2M+CH2O2-H]-", (2, 1, 44.99820285117)),
        ("[2M+C2H4O2-H]-", (2, 1, 59.013852915310004)),
        ("[2M+Na-2H]-", (2, 1, 20.97521637656)),
        ("[3M+H]+", (3, 1, 1.00727645217)),
        ("[3M-H]-", (3, 1, -1.00727645217)),
    ]


def test_neutral_and_precursor_masses_invert_the_adduct_arithmetic() -> None:
    labels: list[str | None] = ["[M+H]+", "[2M+Na]+", "[M+2H]2+", "unknown", None]
    mz = np.array([101.0, 223.0, 51.0, 70.0, 70.0])
    actual = neutral_mass(mz, labels)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(
        actual,
        [
            101.0 - 1.00727645217,
            (223.0 - 22.989220701) / 2,
            102.0 - 2.01455290434,
            math.nan,
            math.nan,
        ],
    )
    labels = ["[M-2H]-", "[3M-H]-", "[M+Li]+"]
    np.testing.assert_array_equal(
        second_neutral_mass(mz[:3], labels),
        [202.0 + 2.01455290434, (223.0 + 1.00727645217) / 3, math.nan],
    )
    assert precursor_mass(100.0, adduct="[M+2H]2+") == (100.0 + 2.01455290434) / 2
    assert precursor_mass(100.0, adduct="[3M-H]-") == 300.0 - 1.00727645217


def test_main_engine_fragment_carriers_follow_polarity_and_adduct_markers() -> None:
    proton = adduct_parameters("[M+H]+")[2]
    deprotonated = adduct_parameters("[M-H]-")[2]
    hcoo = ELEMENT_MASSES["C"] + 2 * HYDROGEN_MASS + 2 * ELEMENT_MASSES["O"] - proton
    assert main_ion_deltas("[M+H]+") == (proton,)
    assert main_ion_deltas("[M+Na]+") == (proton, adduct_parameters("[M+Na]+")[2])
    assert main_ion_deltas("[M+K]+") == (proton, adduct_parameters("[M+K]+")[2])
    assert main_ion_deltas("[M+NH4]+") == (proton, adduct_parameters("[M+NH4]+")[2])
    assert main_ion_deltas("[M-H]-") == (deprotonated,)
    assert main_ion_deltas("[M+Cl]-") == (
        deprotonated,
        adduct_parameters("[M+Cl]-")[2],
    )
    for label in ("[M+CH2O2-H]-", "[M+HCOO]-"):
        assert main_ion_deltas(label) == (deprotonated, hcoo)


def test_formula_vectors_retain_repeated_tokens_charges_and_invalid_rows() -> None:
    for formula, expected in (
        ("H3C2H3NO", [2, 6, 1, 1, 0, 0, 0, 0, 0, 0]),
        ("H4N+", [0, 4, 1, 0, 0, 0, 0, 0, 0, 0]),
        ("+", [0] * 10),
        (None, [-1] * 10),
        ("", [-1] * 10),
        ("Na", [-1] * 10),
        ("C(OH)2", [-1] * 10),
        ("C01H0", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ):
        actual = formula_vector(formula)
        assert actual.dtype == np.int16
        np.testing.assert_array_equal(actual, expected)


def test_complete_reader_and_malformed_graph_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    directory = tmp_path / "graphs"
    _write_graphs(directory)
    table = AnalogTable(directory)
    row = table.row(1)
    assert row.formula_input is None
    assert row.core_smiles == row.query_smiles == "[N]"
    assert row.element_smiles == "[N]"
    assert row.graph is not None
    assert row.core is not None
    np.testing.assert_array_equal(row.graph.elements, [7])
    assert row.graph.bonds.shape == (0, 2)
    np.testing.assert_array_equal(row.core.elements, [7])
    np.testing.assert_array_equal(row.hashes, np.arange(8, 16))
    np.testing.assert_array_equal(row.numbers, np.arange(8, 16))
    np.testing.assert_array_equal(row.formula, np.arange(10, 20))
    np.testing.assert_array_equal(row.core_mask, [True])
    for array in (
        row.hashes,
        row.numbers,
        row.formula,
        row.core_mask,
        row.graph.elements,
        row.graph.bonds,
        row.core.elements,
        row.core.bonds,
    ):
        assert not array.flags.writeable
    assert table.row(0).core is None
    np.testing.assert_array_equal(table.for_smiles("same").hashes, np.arange(8))
    with pytest.raises(KeyError, match="absent"):
        _ = table.for_smiles("absent")
    with pytest.raises(IndexError):
        _ = table.row(2)
    path = directory / "numbers.npy"
    correct = cast("NDArray[np.generic]", np.load(path))
    np.save(path, np.zeros((2, 8), dtype=np.int32))
    with pytest.raises(TypeError, match=r"expected int16, got int32\.$"):
        _ = AnalogTable(directory)
    np.save(path, correct)
    for name, malformed, message in (
        ("hashes", np.zeros((2, 7), dtype=np.int64), "fixed arrays"),
        # AnalogTable rejects one fixed-array row for the two molecule rows.
        ("numbers", np.zeros((1, 8), dtype=np.int16), "fixed arrays"),
        ("formula", np.zeros((2, 9), dtype=np.int16), "fixed arrays"),
        # AnalogTable requires a flat graph-element vector, not a matrix.
        ("graph_elements", np.zeros((1, 3), dtype=np.int16), "invalid rank"),
        (
            "graph_bonds",
            # AnalogTable requires two atom indices per edge, not three.
            np.zeros((1, 3), dtype=np.int32),
            r"^Analog bonds require two atom indices per undirected edge\.$",
        ),
        (
            "graph_atom_off",
            np.array([0, 3]),
            r"^Analog ragged offsets differ from the complete graph layout\.$",
        ),
        (
            "graph_atom_off",
            np.array([1, 2, 3]),
            r"^Analog ragged offsets differ from the complete graph layout\.$",
        ),
        (
            "graph_atom_off",
            np.array([0, 2, 4]),
            r"^Analog ragged offsets differ from the complete graph layout\.$",
        ),
        (
            "graph_atom_off",
            np.array([0, 4, 3]),
            r"^Analog ragged offsets differ from the complete graph layout\.$",
        ),
        (
            "core_bond_off",
            np.array([0, 1, 0]),
            r"^Analog ragged offsets differ from the complete graph layout\.$",
        ),
        ("core_mask", np.zeros(1, dtype=np.bool_), "core mask"),
    ):
        path = directory / f"{name}.npy"
        correct = cast("NDArray[np.generic]", np.load(path))
        np.save(path, malformed)
        with pytest.raises(ValueError, match=message):
            _ = AnalogTable(directory)
        np.save(path, correct)
    for version, release in ((2, IDENTITY_RDKIT), (1, SIMULATOR_RDKIT)):
        _ = (directory / "manifest.json").write_text(
            json.dumps(to_plain({"format_version": version, "rdkit": release})),
        )
        with pytest.raises(ValueError, match="Analog data requires format 1"):
            _ = AnalogTable(directory)


def _write_graphs(directory: Path) -> None:
    directory.mkdir()
    parquet.write_table(
        Table.from_pydict(
            {
                "smiles": ["same", "same"],
                "formula_input": ["CO", None],
                "core_smiles": ["", "[N]"],
                "element_smiles": ["[C][O]", "[N]"],
                "query_smiles": ["", "[N]"],
            },
        ),
        directory / "molecules.parquet",
    )
    for name, values in (
        ("hashes", np.arange(16, dtype=np.int64).reshape(2, 8)),
        ("numbers", np.arange(16, dtype=np.int16).reshape(2, 8)),
        ("formula", np.arange(20, dtype=np.int16).reshape(2, 10)),
        ("graph_elements", np.array([6, 8, 7], dtype=np.int16)),
        ("graph_bonds", np.array([[0, 1]], dtype=np.int32)),
        ("graph_atom_off", np.array([0, 2, 3])),
        ("graph_bond_off", np.array([0, 1, 1])),
        ("core_elements", np.array([7], dtype=np.int16)),
        ("core_bonds", np.empty((0, 2), dtype=np.int32)),
        ("core_atom_off", np.array([0, 0, 1])),
        ("core_bond_off", np.array([0, 0, 0])),
        ("core_mask", np.array([False, False, True])),
    ):
        np.save(directory / f"{name}.npy", values)
    _ = (directory / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": IDENTITY_RDKIT})),
    )


def test_complete_reaction_vocabulary_and_every_mass_plan() -> None:
    rules = reaction_rules()
    deltas = [rule.mass for rule in rules] + [0.0, 0.003, -0.003, 1000.0]
    actual = {"names": _reactions_text("\0".join(rule.name for rule in rules))}
    for index, delta in enumerate(deltas):
        plans = matching_plans(rules, delta, tolerance=0.003, max_reactions=2)
        actual[f"match/{index}"] = _plans([list(plan) for plan in plans])
    assert_tensor_golden(_CWD / "testdata" / "derivatives.pt", actual)


def test_mass_plans_keep_single_before_pair_order_and_cancel_element_counts() -> None:
    rules = (
        Reaction(name="oxygen", smarts="unused", delta=(("O", 1),)),
        Reaction(name="minus_oxygen", smarts="unused", delta=(("O", -1),)),
        Reaction(name="water", smarts="unused", delta=(("H", 2), ("O", 1))),
        Reaction(name="hydrogen", smarts="unused", delta=(("H", 2),)),
    )
    assert matching_plans(rules, 0.0, tolerance=0.0, max_reactions=2) == [(0, 1)]
    assert matching_plans(rules, 15.9949146196, tolerance=0.0, max_reactions=2) == [
        (0,),
    ]
    assert matching_plans(rules, 2 * 1.00782503207, tolerance=0.0, max_reactions=2) == [
        (3,),
        (1, 2),
    ]
    assert matching_plans(rules, 0.0, tolerance=0.0, max_reactions=1) == []
    assert matching_plans(rules, 15.9949146196, tolerance=0.0, max_reactions=0) == []
    assert matching_plans(rules, 32.0, tolerance=0.0, max_reactions=2) == []
    assert matching_plans(rules, 0.5, tolerance=0.5, max_reactions=2) == [(0, 1)]
    assert matching_plans(rules, 0.5, tolerance=0.499, max_reactions=2) == []
    assert matching_plans([], 0.0, tolerance=0.003, max_reactions=2) == []


@pytest.mark.parametrize("element", ["H", "C", "N", "O", "S", "Cl"])
def test_reaction_element_table_covers_every_recipe_element(element: str) -> None:
    single = Reaction(
        name="cancel",
        smarts="unused",
        delta=((element, 1), (element, -1)),
    )
    assert matching_plans((single,), 0.0, tolerance=0.0, max_reactions=1) == [(0,)]


def test_complete_v6_grammar_and_all_mass_plan_matches_are_hermetic() -> None:
    deltas = (0.0, 0.003, -0.003, 15.9949146196, 14.01565006414, 1000.0)
    rules = v6_reaction_rules()
    actual = {
        "names": _reactions_text("\0".join(rule.name for rule in rules)),
        "smarts": _reactions_text("\0".join(rule.smarts for rule in rules)),
        "formula_deltas": _reactions_text(
            "\n".join(
                ";".join(f"{element}={count}" for element, count in rule.delta)
                for rule in rules
            ),
        ),
        "masses": torch.tensor([rule.mass for rule in rules], dtype=torch.float64),
    }
    for index, delta in enumerate(deltas):
        plans = matching_plans(rules, delta, tolerance=0.003, max_reactions=2)
        actual[f"match/{index}"] = _plans([list(plan) for plan in plans])
    assert_tensor_golden(
        _CWD / "testdata/v6_derivatives.pt",
        float32_comparands(actual),
    )


def _plans(values: list[list[int]]) -> Tensor:
    return torch.tensor(
        [[plan[0], plan[1] if len(plan) == 2 else -1] for plan in values],
        dtype=torch.int64,
    ).reshape(-1, 2)


def _reactions_text(value: str) -> Tensor:
    return torch.tensor(list(value.encode()), dtype=torch.uint8)


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_complete_candidate_metadata_preserves_duplicates_and_order(
    tmp_path: Path,
    protocol: int,
) -> None:
    path = tmp_path / "pool.pkl"
    metadata = {
        "keys": np.array(["third", "first", "third"], dtype=object),
        "smiles": np.array(["CCO", "C", "O"], dtype="U3"),
        "mass": np.array([46.0, 16.0, 18.0], dtype=np.float64),
    }
    _ = path.write_bytes(pickle.dumps(metadata, protocol=protocol))
    assert read_pool_metadata(path) == {
        "keys": ["third", "first", "third"],
        "smiles": ["CCO", "C", "O"],
    }


def test_primitive_empty_metadata_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "empty.pkl"
    _ = path.write_bytes(pickle.dumps({"keys": [], "smiles": []}))
    assert read_pool_metadata(path) == {"keys": [], "smiles": []}


def test_every_non_numpy_global_is_rejected_even_in_unused_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forbidden.pkl"
    _ = path.write_bytes(
        pickle.dumps({"keys": ["A"], "smiles": ["C"], "unused": range(3)}),
    )
    with pytest.raises(pickle.UnpicklingError, match=r"builtins\.range"):
        _ = read_pool_metadata(path)


@pytest.mark.parametrize(
    "metadata",
    [
        {"keys": [42], "smiles": ["C"]},
        {"keys": [["A"]], "smiles": ["C"]},
        {"keys": ["A"]},
    ],
)
def test_invalid_column_schema_is_rejected(
    tmp_path: Path,
    metadata: dict[str, object],
) -> None:
    path = tmp_path / "invalid.pkl"
    _ = path.write_bytes(pickle.dumps(metadata))
    with pytest.raises((ValueError, KeyError)):
        _ = read_pool_metadata(path)


def test_mismatched_column_lengths_identify_the_schema_error(tmp_path: Path) -> None:
    path = tmp_path / "different.pkl"
    _ = path.write_bytes(pickle.dumps({"keys": ["A", "B"], "smiles": ["C"]}))
    with pytest.raises(
        ValueError,
        match=r"^Candidate metadata columns have different lengths\.$",
    ):
        _ = read_pool_metadata(path)


def test_legacy_array_bytes_cannot_select_an_arbitrary_codec() -> None:
    with pytest.raises(
        pickle.UnpicklingError,
        match=r"^Legacy NumPy bytes require latin1 encoding\.$",
    ):
        _ = _latin1("value", "utf8")


def test_intrinsic_reader_keeps_named_rows_and_duplicate_selection(
    tmp_path: Path,
) -> None:
    parquet.write_table(
        Table.from_pydict({"smiles": ["A", "B", "C"]}),
        tmp_path / "molecules.parquet",
    )
    records = np.arange(3 * 34, dtype=np.float32).reshape(3, 34)
    capabilities = np.arange(3 * 22, dtype=np.uint8).reshape(3, 22)
    np.save(tmp_path / "records.npy", records)
    np.save(tmp_path / "capabilities.npy", capabilities)
    _ = (tmp_path / "record_names.txt").write_text("\n".join(record_fields()) + "\n")
    _ = (tmp_path / "capability_names.txt").write_text(
        "\n".join(str(i) for i in range(22)) + "\n",
    )
    table = IntrinsicTable.load(tmp_path)
    selected, caps = table.select(["C", "A", "C"])
    np.testing.assert_array_equal(selected, records[[2, 0, 2]])
    np.testing.assert_array_equal(caps, capabilities[[2, 0, 2]])
    np.testing.assert_array_equal(
        table.scaffold_keys(),
        np.array([(0 << 24) | 1, (34 << 24) | 35, (68 << 24) | 69]),
    )
    np.testing.assert_array_equal(
        table.scaffold_keys(generic=True),
        np.array([(2 << 24) | 3, (36 << 24) | 37, (70 << 24) | 71]),
    )
    with pytest.raises(KeyError):
        _ = table.select(["missing"])
    empty, empty_caps = table.select([])
    assert empty.shape == (0, 34)
    assert empty_caps.shape == (0, 22)
    np.save(tmp_path / "records.npy", records[:, :-1])
    with pytest.raises(ValueError, match="Intrinsic record layout"):
        _ = IntrinsicTable.load(tmp_path)
    np.save(tmp_path / "records.npy", records)
    _ = (tmp_path / "record_names.txt").write_text("wrong\n")
    with pytest.raises(ValueError, match="Intrinsic record layout"):
        _ = IntrinsicTable.load(tmp_path)
    _ = (tmp_path / "record_names.txt").write_text("\n".join(record_fields()) + "\n")
    np.save(tmp_path / "capabilities.npy", capabilities[:, :-1])
    with pytest.raises(ValueError, match="Capability layout"):
        _ = IntrinsicTable.load(tmp_path)
    np.save(tmp_path / "capabilities.npy", capabilities)
    _ = (tmp_path / "capability_names.txt").write_text("wrong\n")
    with pytest.raises(ValueError, match="Capability layout"):
        _ = IntrinsicTable.load(tmp_path)
    _ = (tmp_path / "capability_names.txt").write_text(
        "\n".join(str(i) for i in range(22)) + "\n",
    )
    np.save(tmp_path / "records.npy", records.astype(np.float64))
    with pytest.raises(ValueError, match="Intrinsic data types"):
        _ = IntrinsicTable.load(tmp_path)
    np.save(tmp_path / "records.npy", records)
    np.save(tmp_path / "capabilities.npy", capabilities.astype(np.int32))
    with pytest.raises(ValueError, match="Intrinsic data types"):
        _ = IntrinsicTable.load(tmp_path)


def test_raw_molecular_data_exposes_real_runtime_annotation_types() -> None:
    hints = get_type_hints(FragnetAtoms)
    assert hints["hydrogens"] == NDArray[np.int64]
    assert hints["bond_ring"] == NDArray[np.bool_]


def test_mobile_hydrogen_assignments_follow_the_ordered_graph_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    golden = _BASELINE / "testdata" / "fragnet_ordering.pt"
    expected = read_tensors(golden)
    actual: dict[str, Tensor] = {}
    for index in range(20):
        prefix = str(index)
        actual[f"{prefix}/input"] = expected[f"{prefix}/input"].clone()
        atoms = _atoms(expected, prefix)
        record = molecular_record(atoms, masses=ELEMENT_MASSES)
        actual[f"{prefix}/raw/symbols"] = torch.tensor(
            list("\0".join(atoms.symbols).encode()),
            dtype=torch.uint8,
        )
        for name, values in (
            ("hydrogens", atoms.hydrogens),
            ("degree", atoms.degree),
            ("charge", atoms.charge),
            ("aromatic", atoms.aromatic),
            ("bonds", atoms.bonds),
            ("bond_aromatic", atoms.bond_aromatic),
            ("bond_ring", atoms.bond_ring),
        ):
            actual[f"{prefix}/raw/{name}"] = torch.from_numpy(values)
        codes = from_plain(
            cast(object, expected[f"{prefix}/wl/codes"].tolist()),
            list[int],
        )
        actual[f"{prefix}/wl/codes"] = torch.tensor(codes, dtype=torch.int64)
        actual[f"{prefix}/wl/labels"] = torch.tensor(
            _labels(_properties(atoms), codes=codes),
            dtype=torch.int64,
        )
        # fragnet_records.pt pins every other molecular_record field.
        assert record is not None
        actual[f"{prefix}/nh"] = torch.from_numpy(record["nh"])
    assert_tensor_golden(golden, actual)


def _atoms(expected: dict[str, Tensor], prefix: str) -> FragnetAtoms:
    text = bytes(
        from_plain(cast(object, expected[f"{prefix}/raw/symbols"].tolist()), list[int]),
    ).decode()
    return FragnetAtoms(
        symbols=tuple(text.split("\0")) if text else (),
        hydrogens=cast(
            "NDArray[np.int64]",
            expected[f"{prefix}/raw/hydrogens"].numpy(),
        ).copy(),
        degree=cast(
            "NDArray[np.int64]",
            expected[f"{prefix}/raw/degree"].numpy(),
        ).copy(),
        charge=cast(
            "NDArray[np.int64]",
            expected[f"{prefix}/raw/charge"].numpy(),
        ).copy(),
        aromatic=cast(
            "NDArray[np.int64]",
            expected[f"{prefix}/raw/aromatic"].numpy(),
        ).copy(),
        bonds=cast("NDArray[np.int64]", expected[f"{prefix}/raw/bonds"].numpy()).copy(),
        bond_aromatic=cast(
            "NDArray[np.bool_]",
            expected[f"{prefix}/raw/bond_aromatic"].numpy(),
        ).copy(),
        bond_ring=cast(
            "NDArray[np.bool_]",
            expected[f"{prefix}/raw/bond_ring"].numpy(),
        ).copy(),
    )


def test_request_identity_materializes_every_default_without_mass_rounding() -> None:
    request: ReactionRequest = {"smiles": "parent", "delta": 1.0}
    header = normalize_request(request)
    assert header == {
        "smiles": "parent",
        "delta": 1.0,
        "operation": "derive",
        "rule": -1,
        "tolerance": 0.003,
        "max_steps": 2,
        "max_products": 60,
        "allow_isomerization": True,
        "application_budget": 800,
        "stage_parents": 24,
        "first_products": 32,
        "later_products": 16,
    }
    assert normalize_request(request, default_products=300)["max_products"] == 300
    assert (
        normalize_request({"smiles": "parent", "delta": 0.0, "operation": "apply"})[
            "max_products"
        ]
        == 64
    )
    explicit = header.copy()
    explicit["max_products"] = 7
    assert normalize_request(explicit, default_products=300) == explicit
    reordered = from_plain(dict(reversed(list(header.items()))), ReactionHeader)
    assert reaction_key(header) == reaction_key(reordered)
    changed = header.copy()
    changed["delta"] = math.nextafter(1.0, 2.0)
    assert reaction_key(header) != reaction_key(changed)
    changed = header.copy()
    changed["stage_parents"] = 25
    assert reaction_key(header) != reaction_key(changed)


def test_reader_rejects_malformed_layout_and_retains_first_duplicate(
    tmp_path: Path,
) -> None:
    header = normalize_request({"smiles": "parent", "delta": 0.0}, default_products=300)
    response: dict[str, object] = {
        "input_smiles": "parent",
        "products": ["first"],
        "plans": [[0]],
        "matches": [[1]],
        "applications": "every original application",
    }
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "default_products": 300,
    }
    _reaction_data_write(
        tmp_path,
        headers=[header, header],
        responses=[
            response,
            {
                **response,
                "products": ["second"],
            },
        ],
        manifest=manifest,
    )
    table = ReactionTable(tmp_path)
    assert table.products({"smiles": "parent", "delta": 0.0}) == (
        {"raw": "first", "combo": ("a",)},
    )
    assert table.default_products == 300
    assert table.responses[0]["applications"] == "every original application"
    for field, invalid in (("format_version", 2), ("rdkit", "different")):
        _reaction_data_write(
            tmp_path,
            headers=[header],
            responses=[response],
            manifest={
                **manifest,
                field: invalid,
            },
        )
        with pytest.raises(
            ValueError,
            match=r"^Reaction data requires format 1 and RDKit 2026\.03\.3\.$",
        ):
            ReactionTable(tmp_path)
    _reaction_data_write(
        tmp_path,
        headers=[header],
        responses=[response],
        manifest={
            **manifest,
            "default_products": "300",
        },
    )
    with pytest.raises(ReadError):
        ReactionTable(tmp_path)
    malformed: tuple[list[dict[str, object]], ...] = (
        [],
        [{**response, "input_smiles": "other"}],
        [{**response, "plans": []}],
        [{**response, "plans": [[-1]]}],
        [{**response, "plans": [[2]]}],
        [{**response, "matches": [[-1]]}],
        [{**response, "matches": [[2]]}],
    )
    for responses in malformed:
        _reaction_data_write(
            tmp_path,
            headers=[header],
            responses=responses,
            manifest=manifest,
        )
        with pytest.raises(
            ValueError,
            match=r"^Reaction results differ from their request headers\.$",
        ):
            ReactionTable(tmp_path)
    _reaction_data_write(
        tmp_path,
        headers=[header],
        responses=[
            {
                **response,
                "plans": [[]],
            },
        ],
        manifest=manifest,
    )
    assert ReactionTable(tmp_path).products({"smiles": "parent", "delta": 0.0}) == (
        {"raw": "first", "combo": None},
    )


def _reaction_data_write(
    directory: Path,
    *,
    headers: list[ReactionHeader],
    responses: list[dict[str, object]],
    manifest: dict[str, object],
) -> None:
    parquet.write_table(
        Table.from_pylist([dict(row) for row in headers]),
        directory / "requests.parquet",
    )
    if responses:
        parquet.write_table(Table.from_pylist(responses), directory / "results.parquet")
    else:
        parquet.write_table(
            Table.from_pydict(
                {
                    name: []
                    for name in (
                        "input_smiles",
                        "products",
                        "plans",
                        "matches",
                        "applications",
                    )
                },
            ),
            directory / "results.parquet",
        )
    parquet.write_table(
        Table.from_pydict({"name": ["a", "b"]}),
        directory / "rules.parquet",
    )
    (directory / "manifest.json").write_text(json.dumps(to_plain(manifest)))


def test_complete_proposals_have_explicit_empty_headers_and_no_runtime_sdk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    prepared = Prepared(read_tensors(_BASELINE / "testdata/generation_inputs.pt"))
    requests: list[dict[str, object]] = []
    responses: list[dict[str, object]] = []
    vocabulary = list(
        dict.fromkeys(
            name
            for row in prepared.inputs["requests"]
            for _, combo in row["products"]
            for name in combo or ()
        ),
    )
    for row in prepared.inputs["requests"]:
        requests.append(
            {
                "smiles": row["smiles"],
                "delta": row["delta"],
                "operation": "derive",
                "rule": -1,
                "tolerance": row["tol"],
                "max_steps": row["max_steps"],
                "max_products": row["max_per_analog"],
                "allow_isomerization": row["allow_isomerisation"],
                "application_budget": 800,
                "stage_parents": 24,
                "first_products": 32,
                "later_products": 16,
            },
        )
        responses.append(
            {
                "input_smiles": row["smiles"],
                "products": [value for value, _ in row["products"]],
                "plans": [
                    [vocabulary.index(name) for name in combo or ()]
                    for _, combo in row["products"]
                ],
                "matches": [],
                "applications": "",
            },
        )
    parquet.write_table(Table.from_pylist(requests), tmp_path / "requests.parquet")
    parquet.write_table(Table.from_pylist(responses), tmp_path / "results.parquet")
    parquet.write_table(
        Table.from_pydict({"name": vocabulary}),
        tmp_path / "rules.parquet",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            to_plain(
                {
                    "format_version": 1,
                    "rdkit": IDENTITY_RDKIT,
                    "default_products": 60,
                },
            ),
        ),
    )
    table = ReactionTable(tmp_path)
    for row in prepared.inputs["requests"]:
        query: ReactionRequest = {
            "smiles": row["smiles"],
            "delta": row["delta"],
            "tolerance": row["tol"],
            "max_steps": row["max_steps"],
            "max_products": row["max_per_analog"],
            "allow_isomerization": row["allow_isomerisation"],
        }
        assert table.products(query) == tuple(
            {"raw": raw, "combo": combo} for raw, combo in row["products"]
        )
    with pytest.raises(KeyError):
        table.products({"smiles": "unstaged", "delta": 0.0})
    first = prepared.inputs["requests"][0]
    with pytest.raises(KeyError):
        table.products(
            {
                "smiles": first["smiles"],
                "delta": first["delta"],
                "tolerance": first["tol"],
                "max_products": 61,
            },
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
