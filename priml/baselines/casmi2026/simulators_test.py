"""Checks for the simulators module."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override
from unittest import mock

import json
import math
import sys
import warnings

from configgle import traverse
from pyarrow import Table, parquet
from torch import Tensor, nn
from treekle import from_plain, to_plain

import numpy as np
import pyarrow as pa
import pytest
import torch

from priml.baselines.casmi2026 import simulators
from priml.baselines.casmi2026.isolated import (
    ELEMENT_MASSES,
    GLACIER_ROOT_DIR,
    ICE_ROOT_DIR,
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    SIMULATOR_SYMBOLS,
)
from priml.baselines.casmi2026.model import (
    GraphGenerator,
    GraphIntensity,
    ReleasedModel,
)
from priml.baselines.casmi2026.model_test import prepared_batch, tiny_glacier
from priml.baselines.casmi2026.simulators import (
    ADDUCT_ROWS,
    DENSE_NODE_LIMIT,
    HYDROGEN_ROW,
    AtomicGraph,
    AtomicGraphTable,
    FragmentGraph,
    GlacierBackend,
    GlacierGraphTable,
    GlacierScheduling,
    IceBackend,
    IceBatch,
    IceGraphTable,
    IceScheduling,
    MaxReduction,
    QueryScoring,
    RowGroups,
    SimulationJobState,
    SimulationMolecule,
    SimulationSpectrum,
    SimulatorAdmissionTable,
    SumReduction,
    _assemble,
    _components,
    _ids,
    _prefix,
    collision_energy,
    condition_ids,
    covered,
    energy_bucket,
    first_unique,
    formula_counts,
    fragment_masses,
    graphormer_input,
    grow_fragments,
    instrument_token,
    landing_probabilities,
    load_released_checkpoint,
    mask_nodes,
    merge_predictions,
    pack_ice_batch,
    post_prediction,
    predict_spectra,
    read_atomic_graphs,
    remove_atoms,
    required_jobs,
    run_simulator,
    spectrum_hash,
    take_graphs,
    walk_features,
)
from priml.baselines.casmi2026.testing import float32_comparands, set_heads
from priml.model.attention.attention import Attention
from priml.model.ggnn import GraphBatch
from priml.model.special import Dropout
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_tensor_golden, mismatches, read_tensors
from priml.train.budget import WallBudget


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import (
        IceRow,
        SimulatorAdmissionRow,
        SpectrumRequest,
    )
    from priml.baselines.casmi2026.model import GlacierBatch, IntensityBatch
    from priml.baselines.casmi2026.simulators import (
        PredictedPeaks,
        PreparedPeaks,
        SimulationJob,
        SimulatorResult,
    )


def without_dropout[ConfigT](config: ConfigT) -> ConfigT:
    """Set every dropout module and attention-probability dropout to zero.

    Args:
      config: Configuration tree, edited in place.

    Returns:
      config: The same tree.

    """
    for match in traverse(config, Dropout.Config, recurse=True):
        match.config.p = 0.0
    for match in traverse(config, Attention.Config):
        attention = match.config
        if isinstance(attention, Attention.Config):
            attention.dropout = 0.0
    return config


_BASELINE: Final = _CWD


def test_staged_decisions_and_canonical_cache_need_no_chemistry_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    rows = admission_rows()
    _write(tmp_path, rows)
    table = SimulatorAdmissionTable(tmp_path)
    assert table.rows == tuple(rows)
    for index, row in enumerate(rows):
        assert table.row(index) == row
        assert table.ice(row["smiles"]) == (row["ice_accepted"], row["ice_reason"])
        assert table.glacier(row["smiles"]) == (
            row["glacier_accepted"],
            row["glacier_reason"],
        )
        assert table.canonical[row["smiles"]] == (
            row["glacier_canonical"],
            row["glacier_heavy_atoms"],
        )
    for index in (-1, 16):
        with pytest.raises(IndexError) as error:
            _ = table.row(index)
        assert error.value.args == (index,)
    with pytest.raises(KeyError, match="unstaged"):
        _ = table.ice("unstaged")
    _write(tmp_path, [*rows, {**rows[0], "glacier_canonical": "later"}])
    assert SimulatorAdmissionTable(tmp_path).for_smiles("CCO") == rows[0]
    _write(tmp_path, [])
    empty = SimulatorAdmissionTable(tmp_path)
    assert empty.rows == ()
    assert empty.canonical == {}


def test_admission_reader_rejects_inconsistent_versions_and_decisions(
    tmp_path: Path,
) -> None:
    rows = admission_rows()
    invalid_rows: tuple[SimulatorAdmissionRow, ...] = (
        {**rows[0], "glacier_canonical": None},
        {**rows[0], "glacier_heavy_atoms": -1},
        {**rows[0], "ice_reason": "accepted with rejection"},
        {**rows[0], "glacier_reason": "accepted with rejection"},
    )
    for invalid in invalid_rows:
        _write(tmp_path, [invalid])
        with pytest.raises(ValueError, match="admission decision"):
            _ = SimulatorAdmissionTable(tmp_path)
    for header in (
        {"format_version": 2, "rdkit": SIMULATOR_RDKIT},
        {"format_version": 1, "rdkit": IDENTITY_RDKIT},
    ):
        _write(tmp_path, rows)
        _ = (tmp_path / "manifest.json").write_text(json.dumps(to_plain(header)))
        with pytest.raises(ValueError, match=r"format 1 and RDKit 2025\.03\.6"):
            _ = SimulatorAdmissionTable(tmp_path)


def admission_rows() -> list[SimulatorAdmissionRow]:
    """Return sixteen staged decisions covering every admission and rejection path.

    Returns:
      rows: A repeated input, an alias, both heavy-atom limits and each rejection.

    """
    outside = "element outside VALID_ATOM_NUM"
    return [
        {
            "smiles": smiles,
            "ice_accepted": ice,
            "ice_reason": ice_reason,
            "glacier_accepted": glacier,
            "glacier_reason": glacier_reason,
            "glacier_canonical": canonical,
            "glacier_heavy_atoms": heavy,
        }
        for smiles, ice, ice_reason, glacier, glacier_reason, canonical, heavy in (
            ("CCO", True, "", True, "", "CCO", 3),
            ("OCC", True, "", True, "", "CCO", 3),
            ("F/C=C/F", True, "", True, "", "F/C=C/F", 4),
            ("F/C=C\\F", True, "", True, "", "F/C=C\\F", 4),
            ("[13CH3][NH3+]", True, "", True, "", "[13CH3][NH3+]", 2),
            ("[Na+].[Cl-]", True, "", True, "", "[Cl-].[Na+]", 2),
            ("[Cu+2]", False, "elements ['Cu']", False, outside, None, 0),
            ("", False, "rdkit parse", False, "empty", None, 0),
            ("invalid", False, "rdkit parse", False, "rdkit parse", None, 0),
            ("C" * 100, True, "", True, "", "C" * 100, 100),
            ("C" * 101, True, "", False, "> 100 heavy atoms", None, 0),
            ("*", False, "elements ['*']", False, outside, None, 0),
            ("[H][H]", True, "", True, "", "[H][H]", 0),
            ("[He]", False, "elements ['He']", False, outside, None, 0),
            ("C.O", True, "", True, "", "C.O", 2),
            ("CCO", True, "", True, "", "CCO", 3),
        )
    ]


def _write(directory: Path, rows: list[SimulatorAdmissionRow]) -> None:
    _ = (directory / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": SIMULATOR_RDKIT})),
    )
    schema = pa.schema(
        [
            ("smiles", pa.string()),
            ("ice_accepted", pa.bool_()),
            ("ice_reason", pa.string()),
            ("glacier_accepted", pa.bool_()),
            ("glacier_reason", pa.string()),
            ("glacier_canonical", pa.string()),
            ("glacier_heavy_atoms", pa.int64()),
        ],
    )
    parquet.write_table(
        Table.from_pylist(rows, schema=schema),
        directory / "admission.parquet",
    )


def _text(value: str) -> Tensor:
    return torch.tensor(list(value.encode()), dtype=torch.uint8)


class _Affine(ReleasedModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3)

    @override
    def checkpoint_names(self) -> dict[str, str]:
        return {"fc.weight": "linear.weight", "fc.bias": "linear.bias"}


def test_checkpoint_state_dict_is_loaded_through_the_key_map(tmp_path: Path) -> None:
    source = _Affine()
    path = tmp_path / "model.pt"
    torch.save({"state_dict": source.released_state()}, path)
    target = _Affine()
    load_released_checkpoint(target, path)
    assert not mismatches(source.released_state(), target.released_state())


def test_checkpoints_load_on_cpu_without_unpickling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.pt"
    torch.save({"state_dict": _Affine().released_state()}, path)
    spy = mock.Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", spy)
    load_released_checkpoint(_Affine(), path)
    assert spy.call_args_list == [
        mock.call(path, map_location="cpu", weights_only=True),
    ]


def test_incomplete_checkpoints_are_rejected_before_any_weight_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pt"
    torch.save({"state_dict": {}}, path)
    model = _Affine()
    before = model.linear.weight.clone()
    with pytest.raises(ValueError, match="Checkpoint keys differ"):
        load_released_checkpoint(model, path)
    assert torch.equal(model.linear.weight, before)


def _groups(
    tmp_path: Path,
    sizes: tuple[int, ...],
    decoded: list[int],
) -> RowGroups[int]:
    path = tmp_path / "rows.parquet"
    total = sum(sizes)
    with parquet.ParquetWriter(
        path,
        Table.from_pylist([{"value": 0}]).schema,
    ) as writer:
        start = 0
        for size in sizes:
            writer.write_table(
                Table.from_pylist(
                    [{"value": 10 * value} for value in range(start, start + size)],
                ),
            )
            start += size
    assert total == start

    def decode(rows: object) -> list[int]:
        assert isinstance(rows, list)
        values = [row["value"] for row in cast("list[dict[str, int]]", rows)]
        decoded.append(values[0] // 10)
        return values

    return RowGroups(parquet.ParquetFile(path), decode)


def test_rows_are_addressed_across_group_boundaries(tmp_path: Path) -> None:
    decoded: list[int] = []
    groups = _groups(tmp_path, (2, 3, 1), decoded)
    assert [groups.row(index) for index in range(6)] == [0, 10, 20, 30, 40, 50]
    assert decoded == [0, 2, 5]
    assert groups.resident == 3


def test_resident_groups_are_reused_and_the_least_recent_is_evicted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulators, "RESIDENT_GROUPS", 2)
    decoded: list[int] = []
    groups = _groups(tmp_path, (1, 1, 1), decoded)
    for index in (0, 1, 0, 2):
        _ = groups.row(index)
    assert decoded == [0, 1, 2]
    _ = groups.row(0)
    assert decoded == [0, 1, 2]
    _ = groups.row(1)
    assert decoded == [0, 1, 2, 1]
    _ = groups.row(2)
    assert decoded == [0, 1, 2, 1, 2]
    assert groups.resident == 2


def test_release_forgets_every_decoded_group(tmp_path: Path) -> None:
    decoded: list[int] = []
    groups = _groups(tmp_path, (1, 1), decoded)
    _ = groups.row(0)
    _ = groups.row(1)
    groups.release()
    assert groups.resident == 0
    _ = groups.row(0)
    assert decoded == [0, 1, 0]


def test_dense_and_sparse_powers_retain_isolated_atoms() -> None:
    for count in (5, DENSE_NODE_LIMIT + 1):
        arguments = {
            "source": torch.tensor([0, 1, 1, 2]),
            "destination": torch.tensor([1, 0, 2, 1]),
            "weights": torch.tensor([1.0, 1.0, 3.0, 3.0]),
        }
        previous = torch.is_warn_always_enabled()
        torch.set_warn_always(True)
        try:
            with host_agnostic_numerics():
                if count > DENSE_NODE_LIMIT:
                    with pytest.warns(
                        UserWarning,
                        match="Sparse CSR tensor support is in beta state",
                    ):
                        actual = landing_probabilities(
                            **arguments,
                            count=count,
                            steps=3,
                        )
                else:
                    actual = landing_probabilities(**arguments, count=count, steps=3)
        finally:
            torch.set_warn_always(previous)
        expected = torch.zeros((count, 3))
        expected[:3, 1] = torch.tensor([0.25, 1.0, 0.75])
        assert torch.equal(actual, expected)


def test_a_single_step_is_the_normalized_self_return() -> None:
    source = torch.tensor([0, 1, 1])
    destination = torch.tensor([1, 0, 1])
    weights = torch.tensor([2.0, 1.0, 1.0])
    once = landing_probabilities(source, destination, weights, count=3, steps=1)
    assert once.shape == (3, 1)
    assert once[:, 0].tolist() == [0.0, 0.5, 0.0]


def test_weighted_walks_keep_isolated_atoms_and_smallest_parent_path() -> None:
    # The diamond's equal shortest paths must select parent 1 rather than 2.
    graph = AtomicGraph(
        features=torch.arange(15, dtype=torch.float32).reshape(5, 3),
        bonds=torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]]),
        bond_features=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        weights=torch.tensor([1.0, 3.0, 1.0, 3.0]),
        degrees=torch.tensor([2, 2, 2, 2, 0]),
    )
    actual = graphormer_input(graph, walk_steps=2, max_hops=2, spatial_limit=7)
    assert actual["x"].shape == (5, 5)
    assert torch.equal(actual["x"][:, :3], graph.features)
    assert torch.equal(actual["x"][:, 3], torch.zeros(5))
    assert torch.equal(
        actual["x"][:, 4],
        torch.tensor([0.5, 0.25, 0.75, 0.5, 0.0]),
    )
    assert actual["spatial_pos"][0].tolist() == [0, 1, 1, 2, 6]
    assert actual["spatial_pos"][4].tolist() == [6, 6, 6, 6, 0]
    assert actual["edge_input"][0, 3].tolist() == [[0, 1], [4, 5]]
    assert actual["edge_input"][3, 0].tolist() == [[4, 5], [0, 1]]
    assert torch.count_nonzero(actual["attn_bias"]) == 0
    assert actual["attn_bias"].shape == (6, 6)
    assert torch.equal(actual["degree"], graph.degrees)
    assert actual["num_atoms"].item() == 5


def test_no_bonds_and_disabled_walks_preserve_node_and_edge_shapes() -> None:
    graph = AtomicGraph(
        features=torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]),
        bonds=torch.empty((0, 2), dtype=torch.int64),
        bond_features=torch.empty((0, 4)),
        weights=torch.empty(0),
        degrees=torch.zeros(3, dtype=torch.int64),
    )
    actual = graphormer_input(graph, walk_steps=0, max_hops=3, spatial_limit=1024)
    assert torch.equal(actual["x"], graph.features)
    assert actual["attn_edge_type"].shape == (3, 3, 4)
    assert actual["edge_input"].shape == (3, 3, 3, 4)
    assert torch.equal(
        actual["spatial_pos"].diagonal(),
        torch.zeros(3, dtype=torch.int64),
    )
    for key in ("attn_bias", "attn_edge_type", "edge_input"):
        assert torch.count_nonzero(actual[key]) == 0


def test_one_walk_step_adds_one_channel_and_every_tensor_keeps_its_dtype() -> None:
    graph = AtomicGraph(
        features=torch.ones((3, 2)),
        bonds=torch.tensor([[0, 1], [1, 2]]),
        bond_features=torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        weights=torch.tensor([1.0, 1.0]),
        degrees=torch.tensor([1, 2, 1]),
    )
    actual = graphormer_input(graph, walk_steps=1, max_hops=2, spatial_limit=4)
    assert actual["x"].shape == (3, 3)
    assert actual["edge_input"][0, 2].tolist() == [[1, 0], [0, 2]]
    assert {key: value.dtype for key, value in actual.items()} == {
        "x": torch.float32,
        "attn_bias": torch.float32,
        "attn_edge_type": torch.float32,
        "spatial_pos": torch.int64,
        "degree": torch.int64,
        "edge_input": torch.int64,
        "num_atoms": torch.int64,
    }


def test_generated_graph_reader_preserves_valid_invalid_and_empty_bonds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, object]] = [
        {
            "input_smiles": "[He]",
            "smiles": "[He]",
            "valid": True,
            "features": [[float(i) for i in range(74)]],
            "bonds": [],
            "bond_features": [],
            "weights": [],
            "degrees": [0],
        },
        {
            "input_smiles": "invalid",
            "smiles": "",
            "valid": False,
            "features": [],
            "bonds": [],
            "bond_features": [],
            "weights": [],
            "degrees": [],
        },
    ]
    parquet.write_table(Table.from_pylist(rows), tmp_path / "graphs.parquet")
    monkeypatch.setitem(sys.modules, "rdkit", None)
    graphs = read_atomic_graphs(tmp_path)
    assert tuple(graphs) == ("[He]", "invalid")
    assert graphs["invalid"] is None
    graph = graphs["[He]"]
    assert graph is not None
    assert graph.features.shape == (1, 74)  # A real monoatomic graph has one atom.
    assert graph.features[0].tolist() == list(range(74))
    assert graph.bonds.shape == (0, 2)
    assert graph.bond_features.shape == (0, 28)
    assert graph.weights.shape == (0,)
    assert graph.degrees.tolist() == [0]


def test_atomic_table_retains_complete_rows_with_bounded_graph_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "input_smiles": "invalid",
        "smiles": "CO",
        "valid": False,
        "features": [[float(index) for index in range(74)], [0.0] * 74],
        "bonds": [[0, 1]],
        "bond_features": [[float(index) for index in range(28)]],
        "weights": [1.5],
        "degrees": [1, 1],
    }
    rows = [
        row,
        *[{**row, "input_smiles": str(index), "valid": True} for index in range(17)],
    ]
    parquet.write_table(
        Table.from_pylist(rows),
        tmp_path / "graphs.parquet",
        row_group_size=1,
    )
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(
            to_plain({"format_version": 1, "rdkit": SIMULATOR_RDKIT, "raw": False}),
        ),
    )
    monkeypatch.setitem(sys.modules, "rdkit", None)
    table = AtomicGraphTable(tmp_path)
    assert table.smiles == ("invalid", *map(str, range(17)))
    assert table.graph("invalid") is None
    assert table.identity("invalid") == (None, 0)
    for index in range(17):
        graph = table.graph(str(index))
        assert graph is not None
        assert graph.features.tolist() == row["features"]
        assert graph.bonds.tolist() == row["bonds"]
        assert graph.bond_features.tolist() == row["bond_features"]
        assert graph.weights.tolist() == [1.5]
        assert graph.degrees.tolist() == [1, 1]
        assert table.identity(str(index)) == ("CO", 2)
        assert table._rows.resident <= 16
    assert table.graph("0") is not None
    assert table.graph("0") is not None
    with pytest.raises(KeyError):
        _ = table.graph("unprepared")


@pytest.mark.parametrize(
    ("version", "format_version", "raw"),
    [
        (IDENTITY_RDKIT, 1, False),
        (SIMULATOR_RDKIT, 2, False),
        (SIMULATOR_RDKIT, 1, True),
    ],
)
def test_atomic_table_requires_canonical_exact_version_data(
    tmp_path: Path,
    version: str,
    format_version: int,
    raw: bool,
) -> None:
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(
            to_plain({"format_version": format_version, "rdkit": version, "raw": raw}),
        ),
    )
    with pytest.raises(ValueError, match="Canonical graph data requires"):
        _ = AtomicGraphTable(tmp_path)


def test_presence_sensitive_energy_coverage_and_instrument_conditions() -> None:
    spectrum: SpectrumRequest = {
        "mz": [],
        "it": [],
        "prec": 100.0,
        "mode": 1,
        "adduct": "[M+H]+",
        "ce": 17.5,
    }
    assert collision_energy(spectrum) == 17.5
    assert energy_bucket(collision_energy(spectrum)) == 20
    assert covered(spectrum)
    assert instrument_token(spectrum) == "QTOF"
    assert (
        collision_energy(
            {**spectrum, "ce_units": "NCE", "ce_orig": "10;30", "ce_ev": [5]},
        )
        == 20
    )
    assert collision_energy({**spectrum, "ce_ev": [10, 15]}) == 12.5
    assert math.isnan(collision_energy({**spectrum, "ce_units": None}))
    assert math.isnan(collision_energy({**spectrum, "ce": float("inf")}))
    assert energy_bucket(-5) == 5
    assert energy_bucket(12.5) == 10
    assert math.isnan(energy_bucket(float("nan")))
    assert not covered({**spectrum, "mode": "negative"})
    assert not covered({**spectrum, "adduct": "[M+K]+"})
    assert covered({**spectrum, "mode": "POS", "adduct": "[M+Na]+"})
    assert instrument_token({**spectrum, "instrument": "FT-ICR"}) == "Orbitrap"
    assert (
        instrument_token({**spectrum, "instrument": "FT-ICR", "instr": "QTOF"})
        == "QTOF"
    )


def test_condition_ids_follow_the_model_vocabularies_and_reject_uncovered_values() -> (
    None
):
    jobs = [("A", "[M+Na]+", "QTOF", 20.0), ("B", "[M+H]+", "Orbitrap", 40.0)]
    assert condition_ids(jobs) == ([1, 0], [1, 0])
    assert (HYDROGEN_ROW, ADDUCT_ROWS) == (7, (7, 16))
    with pytest.raises(ValueError, match="not in tuple"):
        _ = condition_ids([("A", "[M+K]+", "QTOF", 20.0)])
    with pytest.raises(ValueError, match="not in tuple"):
        _ = condition_ids([("A", "[M+H]+", "TOF", 20.0)])


def test_peak_reduction_retains_distinct_mass_and_tie_semantics() -> None:
    raw = np.array([[10, 1], [10, 2], [0, 1], [-2, 2], [20, 0]], dtype=np.float32)
    ice = post_prediction(raw, reduction=SumReduction.Config().make(), max_peaks=100)
    glacier = post_prediction(
        raw,
        reduction=MaxReduction.Config().make(),
        max_peaks=100,
    )
    np.testing.assert_array_equal(ice[0], [-2, 0, 10])
    np.testing.assert_array_equal(ice[1], np.array([2 / 3, 1 / 3, 1], dtype=np.float32))
    np.testing.assert_array_equal(glacier[0], [10])
    np.testing.assert_array_equal(glacier[1], [1])
    parts = [
        (np.array([10.0, 20]), np.array([1.0, 0.5])),
        (np.array([10.0]), np.array([1.0])),
    ]
    merged = merge_predictions(parts)
    np.testing.assert_array_equal(merged[0], [10, 20])
    np.testing.assert_array_equal(merged[1], [1, 0.25])


def test_jobs_and_scores_require_all_missing_energy_predictions() -> None:
    peaks = (np.array([10.0]), np.array([1.0]))
    plan = [
        SimulationSpectrum(
            query=peaks,
            adduct="[M+H]+",
            instrument="Orbitrap",
            energy=float("nan"),
        ),
    ]
    jobs = required_jobs(plan, ["A", "B", "A"])
    assert jobs == {
        (candidate, "[M+H]+", "Orbitrap", energy)
        for candidate in ("A", "B")
        for energy in (20.0, 40.0, 60.0)
    }
    cache = {key: peaks if key[0] == "A" or key[3] != 40 else None for key in jobs}
    assert QueryScoring.Config().make().score(plan, ["A", "B", "missing"], cache) == {
        "A": 1.0,
        "B": None,
        "missing": None,
    }


def test_complete_both_public_simulator_scoring_families() -> None:
    actual: dict[str, Tensor] = {}
    for family in ("iceberg", "glacier"):
        spectra = _spectra()
        scoring = QueryScoring.Config().make()
        plan = scoring.plan(spectra)
        for index, spectrum in enumerate(spectra):
            label = f"{family}/conditions/{index}"
            energy = collision_energy(spectrum)
            actual[f"{label}/energy"] = torch.tensor(energy, dtype=torch.float64)
            actual[f"{label}/bucket"] = torch.tensor(
                energy_bucket(energy),
                dtype=torch.float64,
            )
            actual[f"{label}/covered"] = torch.tensor(covered(spectrum))
            actual[f"{label}/instrument"] = _text(instrument_token(spectrum))
        for index, spectrum in enumerate(plan):
            label = f"{family}/plan/{index}"
            actual[f"{label}/mz"] = torch.from_numpy(spectrum.query[0])
            actual[f"{label}/weights"] = torch.from_numpy(spectrum.query[1])
            actual[f"{label}/adduct"] = _text(spectrum.adduct)
            actual[f"{label}/instrument"] = _text(spectrum.instrument)
            actual[f"{label}/energy"] = torch.tensor(
                spectrum.energy,
                dtype=torch.float64,
            )
        jobs = sorted(required_jobs(plan, ["A", "B", "A", "C"]))
        actual[f"{family}/jobs"] = _text(
            "\n".join("\t".join(map(str, job)) for job in jobs),
        )
        prepared: list[PreparedPeaks] = []
        for index, prediction in enumerate(_predictions()):
            mz, intensity = post_prediction(
                prediction,
                reduction=(
                    SumReduction.Config()
                    if family == "iceberg"
                    else MaxReduction.Config()
                ).make(),
                max_peaks=100,
            )
            actual[f"{family}/post/{index}/mz"] = torch.from_numpy(mz)
            actual[f"{family}/post/{index}/intensity"] = torch.from_numpy(intensity)
            if len(mz):
                prepared.append((mz.astype(np.float64), intensity.astype(np.float64)))
        merged = merge_predictions(prepared)
        actual[f"{family}/merge/mz"] = torch.from_numpy(merged[0])
        actual[f"{family}/merge/intensity"] = torch.from_numpy(merged[1])
        cache: dict[SimulationJob, PreparedPeaks | None] = {}
        for index, key in enumerate(jobs):
            candidate, _, _, energy = key
            cache[key] = (
                None
                if candidate == "C" or (candidate == "B" and energy == 40)
                else prepared[index % len(prepared)]
            )
        scores = scoring.score(plan, ["A", "B", "C", "missing"], cache)
        actual[f"{family}/scores/valid"] = torch.tensor(
            [value is not None for value in scores.values()],
        )
        actual[f"{family}/scores/values"] = torch.tensor(
            [value if value is not None else 0 for value in scores.values()],
            dtype=torch.float64,
        )
    golden = _BASELINE / "testdata" / "simulator_scoring.pt"
    assert_tensor_golden(golden, float32_comparands(actual))


def _spectra() -> list[SpectrumRequest]:
    base: SpectrumRequest = {
        "mz": [0, 10, 20, 30, 90, 102.1],
        "it": [1, 1, 0.1, 0.0004, 2, 99],
        "prec": 100,
        "mode": "positive",
        "adduct": "[M+H]+",
        "instrument": "Orbitrap",
    }
    return [
        {**base, "ce_units": "NCE", "ce_orig": "10;30", "ce_ev": [5]},
        {
            **base,
            "mode": 1,
            "adduct": "[M+Na]+",
            "instrument": "timsTOF",
            "ce_units": "eV",
            "ce_ev": [10, 15],
        },
        {**base, "mode": "POS", "ce": 17.5, "instr": "QTOF"},
        {**base, "mode": "+", "ce_units": "NCE", "ce_orig": "none", "ce": 25},
        {**base, "mode": "1", "ce": -5, "instrument": None},
        {**base, "mode": "negative", "ce": 20},
        {**base, "adduct": "[M+K]+", "ce": 20},
        {**base, "ce": float("inf"), "mz": [], "it": []},
        {**base, "ce_ev": [], "ce": 25, "mz": [200], "it": [1]},
        {**base, "ce_units": None, "ce": 25},
        {**base, "ce": None, "instr": "other", "instrument": "ft-icr"},
        {
            **base,
            "ce": 30,
            "mz": [float(index + 1) for index in range(514)],
            "it": [1] * 514,
            "prec": 600,
        },
    ]


def _predictions() -> list[NDArray[np.float32]]:
    return [
        np.array(
            [[10.00004, 1], [10.000049, 2], [20, 0], [30, -1], [0, 1], [-2, 2]],
            dtype=np.float32,
        ),
        np.column_stack(
            [
                np.arange(131, dtype=np.float32) + 10,
                np.arange(131, dtype=np.float32) % 5 + 1,
            ],
        ),
        np.array([[1, 0], [2, -1]], dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
    ]


def test_aliases_rejections_and_bounded_prepared_root_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepared(tmp_path)
    monkeypatch.setitem(sys.modules, "rdkit", None)
    table = GlacierGraphTable(tmp_path, walk_steps=0, max_hops=2, spatial_limit=7)
    assert table.smiles == ("raw", "other", "invalid")
    assert table.identity("raw") == table.identity("canonical") == ("canonical", 2)
    assert table.identity("other") == ("second", 3)
    assert table.identity("invalid") == (None, 0)
    with pytest.raises(KeyError):
        _ = table.identity("absent")
    with pytest.raises(ValueError, match="rejected prepared root"):
        _ = table.batch(
            [("invalid", "[M+H]+", "Orbitrap", 20.0)],
            device=torch.device("cpu"),
        )
    first = table.batch(
        [
            ("canonical", "[M+H]+", "Orbitrap", 20.0),
            ("second", "[M+Na]+", "QTOF", 40.0),
        ],
        device=torch.device("cpu"),
    )
    assert first.atom_counts.tolist() == [2, 3]
    assert first.graph.features.shape == (2, 3, 74)
    assert first.graph.paths.shape == (2, 3, 3, 2, 28)
    assert first.graph.paths.dtype == torch.float32
    assert first.graph.spatial.dtype == first.graph.degrees.dtype == torch.int64
    assert first.root_forms.dtype == first.atom_forms.dtype == torch.float64
    assert first.adduct.tolist() == first.instrument.tolist() == [0.0, 1.0]
    assert first.collision_energy.tolist() == [20.0, 40.0]
    assert first.masses.tolist() == [[10.0, 20.0, 0.0], [10.0, 20.0, 30.0]]
    assert first.atom_hs.tolist() == [[1, 2, 0], [1, 2, 3]]
    assert first.total_hs.tolist() == [3, 6]
    assert (
        first.adduct_shifts.tolist()
        == torch.tensor(
            [[7 - 0.00054858, -0.00054858], [16 - 0.00054858, -0.00054858]],
        ).tolist()
    )
    assert first.graph.bias[0, :3, :3].count_nonzero() == 0
    assert first.graph.bias[0, 3].tolist() == [-99_999.0] * 4
    assert first.graph.bias[0, :, 3].tolist() == [-99_999.0] * 4
    assert first.adjacency[0].tolist() == [[0, 1.5, 0], [1.5, 0, 0], [0, 0, 0]]
    assert len(table._prepared) == 2
    _ = first.atom_forms.fill_(100)
    _ = first.root_forms.fill_(100)
    again = table.batch(
        [("raw", "[M+H]+", "Orbitrap", 0.0)],
        device=torch.device("cpu"),
    )
    assert again.root_forms.tolist() == [list(range(18))]
    assert again.atom_forms.tolist() == [list(range(18)), list(range(18))]
    table.release_cache()
    assert len(table._prepared) == table._rows.resident == 0


def test_the_prepared_root_cache_evicts_the_least_recently_used_root(
    tmp_path: Path,
) -> None:
    _prepared(tmp_path)
    table = GlacierGraphTable(
        tmp_path,
        walk_steps=0,
        max_hops=2,
        spatial_limit=7,
        root_cache=1,
    )
    cpu = torch.device("cpu")
    _ = table.batch([("canonical", "[M+H]+", "Orbitrap", 1.0)], device=cpu)
    assert list(table._prepared) == [0]
    _ = table.batch([("second", "[M+H]+", "Orbitrap", 1.0)], device=cpu)
    assert list(table._prepared) == [1]


@pytest.mark.parametrize(("steps", "hops"), [(-1, 2), (2, 0)])
def test_reader_rejects_invalid_walk_and_hop_geometry(
    tmp_path: Path,
    steps: int,
    hops: int,
) -> None:
    _manifest(tmp_path)
    with pytest.raises(ValueError, match="Walk steps"):
        _ = GlacierGraphTable(
            tmp_path,
            walk_steps=steps,
            max_hops=hops,
            spatial_limit=7,
        )


@pytest.mark.parametrize(
    ("version", "format_version"),
    [(IDENTITY_RDKIT, 1), (SIMULATOR_RDKIT, 2)],
)
def test_reader_requires_the_pinned_physical_data_format(
    tmp_path: Path,
    version: str,
    format_version: int,
) -> None:
    _manifest(tmp_path, version=version, format_version=format_version)
    with pytest.raises(ValueError, match="GLACIER data requires"):
        _ = GlacierGraphTable(tmp_path, walk_steps=0, max_hops=2, spatial_limit=7)


def test_reader_rejects_misaligned_atomic_rows_and_element_masses(
    tmp_path: Path,
) -> None:
    _prepared(tmp_path)
    np.save(tmp_path / "element_masses.npy", np.arange(18, dtype=np.float32)[:, None])
    with pytest.raises(ValueError, match="float64 18-by-1"):
        _ = GlacierGraphTable(tmp_path, walk_steps=0, max_hops=2, spatial_limit=7)
    np.save(tmp_path / "element_masses.npy", np.arange(18, dtype=np.float64)[:, None])
    rows = parquet.read_table(tmp_path / "physical.parquet").to_pylist()
    parquet.write_table(Table.from_pylist(rows[::-1]), tmp_path / "physical.parquet")
    with pytest.raises(ValueError, match="same order"):
        _ = GlacierGraphTable(tmp_path, walk_steps=0, max_hops=2, spatial_limit=7)


def _manifest(
    directory: Path,
    *,
    version: str = SIMULATOR_RDKIT,
    format_version: int = 1,
) -> None:
    _ = (directory / "manifest.json").write_text(
        json.dumps(
            to_plain(
                {
                    "rdkit": version,
                    "format_version": format_version,
                    "raw": False,
                },
            ),
        ),
    )


def _prepared(directory: Path) -> None:
    graphs = directory / "graphs"
    graphs.mkdir()
    _manifest(directory)
    _manifest(graphs)
    np.save(directory / "element_masses.npy", np.arange(18, dtype=np.float64)[:, None])
    atomic: list[dict[str, object]] = []
    physical: list[dict[str, object]] = []
    for original, canonical, count, valid in (
        ("raw", "canonical", 2, True),
        ("other", "second", 3, True),
        ("invalid", "", 0, False),
    ):
        atomic.append(
            {
                "input_smiles": original,
                "smiles": canonical,
                "valid": valid,
                "features": [[float(i) for i in range(74)] for _ in range(count)],
                "bonds": [[0, 1]] if valid else [],
                "bond_features": [[float(i) for i in range(28)]] if valid else [],
                "weights": [1.5] if valid else [],
                "degrees": [1] * count,
            },
        )
        physical.append(
            {
                "input_smiles": original,
                "smiles": canonical,
                "valid": valid,
                "formula": list(range(18)),
                "atom_formulas": [list(range(18))] * count,
                "hydrogens": list(range(1, count + 1)),
                "atom_masses": [9.0] * count,
                "hydrogenated_masses": [10.0 * (i + 1) for i in range(count)],
                "total_hydrogens": count * (count + 1) // 2,
            },
        )
    parquet.write_table(
        Table.from_pylist(atomic),
        graphs / "graphs.parquet",
        row_group_size=1,
    )
    parquet.write_table(
        Table.from_pylist(physical),
        directory / "physical.parquet",
        row_group_size=1,
    )


def test_prepared_numeric_backend_replays_the_whole_trained_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = read_tensors(_BASELINE / "testdata/glacier.pt")
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: pytest.fail("CPU replay must not enumerate CUDA devices."),
    )
    with torch.random.fork_rng(devices=[]):
        trained = tiny_glacier().make()
    trained.load_state_dict(
        {
            name.removeprefix("final/"): value
            for name, value in expected.items()
            if name.startswith("final/") and name != "final/rng"
        },
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": trained.released_state()}, checkpoint)
    batch, _, _ = prepared_batch()
    batch.root_forms.copy_(expected["post/root_forms"])
    batch.atom_forms.copy_(expected["post/atom_forms"])
    batch.graph.features.copy_(expected["post/graph_features"])
    jobs = [(value, "[M+H]+", "Orbitrap", 20.0) for value in ("a", "b", "c")]
    roots = _NumericRoots(batch=batch, jobs=jobs)

    def table(
        directory: Path,
        *,
        walk_steps: int,
        max_hops: int,
        spatial_limit: int,
    ) -> _NumericRoots:
        assert directory == tmp_path
        assert walk_steps == 2
        assert max_hops == 3
        assert spatial_limit == 6
        return roots

    monkeypatch.setattr(simulators, "GlacierGraphTable", table)
    monkeypatch.setitem(sys.modules, "rdkit", None)
    config = GlacierBackend.Config()
    config.root_dir = tmp_path
    config.checkpoint = checkpoint
    config.model = tiny_glacier()
    config.walk_steps = 2
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        backend = config.make()
        with warnings.catch_warnings(record=True) as emitted:
            warnings.filterwarnings(
                "always",
                message="The PyTorch API of nested tensors is in prototype stage.*",
                category=UserWarning,
            )
            raw = backend.predict_fragments(jobs)
        assert all(
            "nested tensors is in prototype stage" in str(item.message)
            for item in emitted
        )
        counts = expected["inference/frags_pred/fragment_count"]
        masks = expected["inference/frags_pred/fragments"]
        cursor = 0
        shifts = 2 * (2 * backend.model.config.max_broken + 1)
        for index, count in enumerate(counts):
            length = int(count)
            assert torch.equal(
                raw["spec"][index],
                expected["inference/inten_pred_end_to_end/output"][
                    index,
                    : length * shifts,
                ],
            )
            assert torch.equal(
                raw["frag"][index],
                masks[cursor : cursor + length].repeat_interleave(shifts, dim=0),
            )
            cursor += length
        predictions = backend.predict([("invalid", *jobs[0][1:]), *jobs])
        assert predictions[0] is None
        assert all(value is not None for value in predictions[1:])
        assert backend.feature_failures == 1
        assert not backend.model.training
        assert all(
            not parameter.requires_grad for parameter in backend.model.parameters()
        )
        with pytest.raises(KeyError, match="unprepared"):
            _ = backend.predict([("unprepared", *jobs[0][1:])])
        released: list[bool] = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(True))
        backend.release_cache()
        assert not roots.released
        assert released == []
        backend.device = torch.device("cuda")
        backend.release_cache()
        assert roots.released
        assert released == [True]
        torch.save({"state_dict": {}}, checkpoint)
        with pytest.raises(ValueError, match="Checkpoint keys differ"):
            _ = config.make()


def test_assembly_excludes_padding_with_different_fragment_counts() -> None:
    masks = torch.tensor(
        [
            [True, False, False, False],
            [False, True, True, False],
            [True, True, False, True],
            [False, False, True, True],
            [True, False, True, False],
        ],
    )
    # _assemble consumes one m/z-intensity pair per spectrum peak.
    spectra = torch.arange(2 * 6 * 2).reshape(2, 6, 2).float()
    actual = _assemble(
        {
            "breakpoints_pred": {
                "frag_logits": torch.empty(0),
                "frag_card_logits": torch.empty(0),
            },
            "inten_pred_end_to_end": {"output": spectra},
            "frags_pred": {"fragments": masks, "fragment_count": torch.tensor([2, 3])},
        },
        shifts=2,
    )
    assert len(actual["spec"]) == len(actual["frag"]) == 2
    assert torch.equal(actual["spec"][0], spectra[0, :4])
    assert torch.equal(actual["spec"][1], spectra[1])
    assert torch.equal(actual["frag"][0], masks[:2].repeat_interleave(2, dim=0))
    assert torch.equal(actual["frag"][1], masks[2:].repeat_interleave(2, dim=0))


class _NumericRoots:
    def __init__(
        self,
        *,
        batch: GlacierBatch,
        jobs: list[tuple[str, str, str, float]],
    ) -> None:
        self.prepared = batch
        self.jobs = jobs
        self.released = False

    def identity(self, smiles: str) -> tuple[str | None, int]:
        return {
            "a": ("a", 4),
            "b": ("b", 5),
            "c": ("c", 5),
            "invalid": (None, 0),
        }[smiles]

    def batch(
        self,
        jobs: list[tuple[str, str, str, float]],
        *,
        device: torch.device,
    ) -> GlacierBatch:
        assert jobs == self.jobs
        assert device == torch.device("cpu")
        return self.prepared

    def release_cache(self) -> None:
        self.released = True


def test_glacier_backend_the_default_root_directory_is_where_the_preparer_writes() -> (
    None
):
    assert GlacierBackend.Config().root_dir == GLACIER_ROOT_DIR


_THIS: Final = Path(__file__).resolve()


def test_segment_replication_preserves_every_node_edge_and_root_identity() -> None:
    graph = _chains()
    result, sizes, offsets = take_graphs(graph, torch.tensor([1, 0, 1]))
    assert sizes.tolist() == [4, 5, 4]
    assert offsets.tolist() == [0, 4, 9]
    assert result.topology.node_counts.tolist() == [4, 5, 4]
    index = torch.tensor([5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert torch.equal(result.features, graph.features[index])
    assert result.node_ids.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3]
    assert result.topology.source.tolist() == [
        0,
        1,
        2,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        10,
        11,
        12,
    ]
    assert result.topology.destination.tolist() == [
        1,
        2,
        3,
        0,
        1,
        2,
        5,
        6,
        7,
        8,
        4,
        5,
        6,
        7,
        10,
        11,
        12,
        9,
        10,
        11,
    ]
    assert result.topology.edge_type.tolist() == [
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        1,
        2,
        3,
        1,
        1,
        2,
        3,
        1,
        2,
        3,
    ]


def test_atom_removal_splits_components_drops_singletons_and_maps_each_request() -> (
    None
):
    graph = _chains()
    result, broken, information = remove_atoms(
        graph,
        torch.tensor([0, 1]),
        torch.tensor([2, 1]),
        {"probability": torch.tensor([0.25, 0.75]), "molecule": torch.tensor([0, 1])},
    )
    assert result is not None
    assert broken is not None
    assert information is not None
    assert result.topology.node_counts.tolist() == [2, 2, 2]
    assert result.node_ids.tolist() == [0, 1, 3, 4, 2, 3]
    assert torch.equal(
        result.features,
        graph.features[torch.tensor([0, 1, 3, 4, 7, 8])],
    )
    assert result.topology.source.tolist() == [0, 1, 2, 3, 4, 5]
    assert result.topology.destination.tolist() == [1, 0, 3, 2, 5, 4]
    assert result.topology.edge_type.tolist() == [1, 1, 1, 1, 3, 3]
    assert broken.tolist() == [2.0, 3.0, 2.0]
    assert information["probability"].tolist() == [0.25, 0.25, 0.75]
    assert information["molecule"].tolist() == [0, 0, 1]
    pair = mask_nodes(
        graph,
        torch.tensor([True, True, False, False, False, False, False, False, False]),
        torch.tensor([2, 0]),
    )
    assert pair.features.shape == (2, 32)
    assert pair.topology.node_counts.tolist() == [2, 0]
    assert remove_atoms(pair, torch.tensor([0]), torch.tensor([0]), {}) == (
        None,
        None,
        None,
    )
    with pytest.raises(
        ValueError,
        match=r"^The retained nodes do not match their segment counts\.$",
    ):
        _ = mask_nodes(graph, torch.ones(9, dtype=torch.bool), torch.tensor([8]))


def test_unrelated_disconnected_component_is_not_a_fragment_of_the_removed_atom() -> (
    None
):
    original = _chains()
    graph = FragmentGraph(
        topology=GraphBatch(
            source=torch.tensor([0, 1, 1, 2, 3, 4]),
            destination=torch.tensor([1, 0, 2, 1, 4, 3]),
            edge_type=torch.ones(6, dtype=torch.int64),
            node_counts=torch.tensor([5]),
        ),
        features=original.features[:5],
        node_ids=torch.arange(5),
    )
    result, broken, information = remove_atoms(
        graph,
        torch.tensor([0]),
        torch.tensor([0]),
        {"value": torch.tensor([7])},
    )
    assert result is not None
    assert broken is not None
    assert information is not None
    assert result.node_ids.tolist() == [1, 2]
    assert result.topology.node_counts.tolist() == [2]
    assert broken.tolist() == [1.0]
    assert information["value"].tolist() == [7]


def test_hash_and_unique_preserve_weighted_messages_and_first_duplicate() -> None:
    graph = _chains()
    expected: list[Tensor] = []
    for start, stop in ((0, 5), (5, 9)):
        states = torch.zeros_like(graph.features)
        for source, destination, kind in zip(
            from_plain(graph.topology.source.tolist(), list[int]),
            from_plain(graph.topology.destination.tolist(), list[int]),
            from_plain(graph.topology.edge_type.tolist(), list[int]),
            strict=True,
        ):
            states[destination] += graph.features[source] * kind
        expected.append(states[start:stop].sum(0))
    assert torch.equal(spectrum_hash(graph), torch.stack(expected))
    values = torch.tensor([[3.0, 1.0], [2.0, 5.0], [3.0, 1.0], [0.0, 0.0]])
    unique, first = first_unique(values, dim=0)
    assert unique.tolist() == [[0.0, 0.0], [2.0, 5.0], [3.0, 1.0]]
    assert first.tolist() == [3, 1, 0]


def test_forms_replace_hydrogen_channel_and_mass_uses_the_formula_dtype() -> None:
    graph = _chains()
    graph.features.zero_()
    graph.features[:, 0] = 1
    graph.features[1, 0] = 0
    graph.features[1, 3] = 1
    graph.features[:, 7] = 99  # The attached-H channels replace the elemental H slot.
    graph.features[:, 26:32] = torch.eye(6)[torch.arange(9).remainder(6)]
    forms = formula_counts(graph)
    expected = torch.zeros((2, 18))
    expected[:, 0] = torch.tensor([4.0, 4.0])
    expected[0, 3] = 1
    expected[:, 7] = torch.tensor([10.0, 8.0])
    assert torch.equal(forms, expected)
    masses = torch.arange(18, dtype=torch.float64)[:, None] + 0.125
    assert torch.equal(
        fragment_masses(forms, masses),
        (expected @ masses.float()).squeeze(-1),
    )


def test_complete_component_labels_keep_isolated_nodes_and_contiguous_ids() -> None:
    graph = _chains()
    assert _components(graph.topology, 12).tolist() == [
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        0,
        0,
        0,
    ]


def test_weighted_dense_and_sparse_walks_retain_isolated_atoms() -> None:
    for count in (5, 2001):
        graph = FragmentGraph(
            topology=GraphBatch(
                source=torch.tensor([0, 1, 1, 2]),
                destination=torch.tensor([1, 0, 2, 1]),
                edge_type=torch.tensor([1, 1, 3, 3]),
                node_counts=torch.tensor([count]),
            ),
            features=torch.zeros((count, 32)),
            node_ids=torch.arange(count),
        )
        if count > 2000:
            previous = torch.is_warn_always_enabled()
            torch.set_warn_always(True)
            try:
                with (
                    pytest.warns(
                        UserWarning,
                        match="Sparse CSR tensor support is in beta state",
                    ),
                    host_agnostic_numerics(),
                ):
                    actual = walk_features(graph, steps=3)
            finally:
                torch.set_warn_always(previous)
        else:
            with host_agnostic_numerics():
                actual = walk_features(graph, steps=3)
        expected = torch.zeros((count, 3))
        expected[:3, 1] = torch.tensor([0.25, 1.0, 0.75])
        assert torch.equal(actual, expected)


def _chains() -> FragmentGraph:
    return FragmentGraph(
        topology=GraphBatch(
            source=torch.tensor([0, 1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 6, 7, 8]),
            destination=torch.tensor([1, 2, 3, 4, 0, 1, 2, 3, 6, 7, 8, 5, 6, 7]),
            edge_type=torch.tensor([1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3]),
            node_counts=torch.tensor([5, 4]),
        ),
        features=torch.arange(288, dtype=torch.float32).reshape(9, 32),
        node_ids=torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3]),
    )


def test_indexed_root_rows_preserve_full_data_and_bound_cached_graph_groups(
    tmp_path: Path,
) -> None:
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"rdkit": SIMULATOR_RDKIT, "format_version": 1})),
    )
    np.save(tmp_path / "element_masses.npy", np.arange(18, dtype=np.float64)[:, None])
    features = torch.zeros((2, 32))
    features[0, [0, 22, 29]] = 1
    features[1, [3, 23, 27]] = 1
    row: IceRow = {
        "input_smiles": "invalid",
        "smiles": "CO",
        "valid": False,
        "features": [[float(value) for value in atom] for atom in features],
        "bonds": [[0, 1]],
        "bond_orders": [1],
        "formula": [1, 0, 0, 1, 0, 0, 0, 4, *([0] * 10)],
        "hydrogens": [3, 1],
        "atom_masses": [12, 15.99491462],
        "hydrogenated_masses": [15.023475096, 17.002739652],
        "mass": 32.026214748,
    }
    rows = [
        row,
        *[{**row, "valid": True, "input_smiles": str(index)} for index in range(17)],
    ]
    parquet.write_table(
        Table.from_pylist(rows),
        tmp_path / "ice.parquet",
        row_group_size=1,
    )
    table = IceGraphTable(tmp_path)
    assert table.smiles == ("invalid", *map(str, range(17)))
    assert table.root("invalid") is None
    for index in range(17):
        root = table.root(str(index))
        assert root is not None
        assert root.smiles == "CO"
        assert torch.equal(root.graph.features, features)
        assert root.graph.topology.source.tolist() == [0, 1]
        assert root.graph.topology.destination.tolist() == [1, 0]
        assert root.graph.topology.edge_type.tolist() == [1, 1]
        assert root.graph.topology.node_counts.tolist() == [2]
        assert root.graph.node_ids.tolist() == [0, 1]
        assert root.formula.tolist() == row["formula"]
        assert root.hydrogens.tolist() == [3, 1]
        assert root.atom_masses.tolist() == row["atom_masses"]
        assert root.hydrogenated_masses.tolist() == row["hydrogenated_masses"]
        assert float(root.mass) == row["mass"]
        assert table._rows.resident <= 16
    assert table.root("0") is not None
    assert table.root("0") is not None
    with pytest.raises(KeyError):
        _ = table.root("unprepared")
    np.testing.assert_array_equal(table.element_masses, np.arange(18)[:, None])


@pytest.mark.parametrize(
    ("version", "format_version"),
    [(IDENTITY_RDKIT, 1), (SIMULATOR_RDKIT, 2)],
)
def test_root_reader_rejects_incompatible_chemistry_and_schema(
    tmp_path: Path,
    version: str,
    format_version: int,
) -> None:
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"rdkit": version, "format_version": format_version})),
    )
    with pytest.raises(ValueError, match="ICE root data requires"):
        _ = IceGraphTable(tmp_path)


@pytest.mark.parametrize("shape", [(18,), (18, 1)])
def test_root_reader_requires_complete_double_precision_element_masses(
    tmp_path: Path,
    shape: tuple[int, ...],
) -> None:
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"rdkit": SIMULATOR_RDKIT, "format_version": 1})),
    )
    np.save(
        tmp_path / "element_masses.npy",
        np.zeros(shape, dtype=np.float64 if len(shape) == 1 else np.float32),
    )
    with pytest.raises(ValueError, match="float64 18-by-1"):
        _ = IceGraphTable(tmp_path)


def test_complete_joint_autoregression_and_intensity_from_prepared_roots() -> None:
    golden = _THIS.parent / "testdata" / "joint.pt"
    inputs = _ice_roots()
    actual: dict[str, Tensor] = {}
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        generator, intensity = _models()
        batch = _batch(inputs)
        prediction = predict_spectra(
            generator,
            intensity,
            batch,
            max_nodes=5,
            max_depth=3,
            walk_steps=2,
            threshold=0,
        )
        assert not generator.training
        assert not intensity.training
        assert all(not parameter.requires_grad for parameter in generator.parameters())
        assert all(not parameter.requires_grad for parameter in intensity.parameters())
        actual.update(
            {
                f"generation/{key}": value
                for key, value in prediction.generation.items()
            },
        )
        for key, values in (
            ("spec", prediction.spectra),
            ("frag", prediction.fragments),
        ):
            actual.update(
                {f"joint/{key}/{index}": value for index, value in enumerate(values)},
            )
        _assert_roots(batch.roots, inputs)
    assert_tensor_golden(golden, float32_comparands(actual))


def test_stopping_threshold_singletons_and_full_capacity() -> None:
    golden = _THIS.parent / "testdata" / "generation_boundaries.pt"
    actual: dict[str, Tensor] = {}
    with torch.random.fork_rng(devices=[]), torch.no_grad(), host_agnostic_numerics():
        generator, _ = _models()
        generator.eval()
        for label, threshold, capacity in (
            ("threshold", 1.0, 5),
            ("singleton", 0.0, 5),
            ("capacity", 0.0, 1),
        ):
            inputs = _ice_roots(singleton=label == "singleton")
            batch = _batch(inputs)
            generation = grow_fragments(
                generator,
                batch,
                max_nodes=capacity,
                max_depth=3,
                walk_steps=2,
                threshold=threshold,
            )
            _assert_roots(batch.roots, inputs)
            actual.update(
                {
                    f"{label}/generation/{key}": value
                    for key, value in generation.items()
                },
            )
    assert_tensor_golden(golden, float32_comparands(actual))


def test_index_helpers_follow_the_device_of_their_reference_and_mark_prefixes() -> None:
    meta = torch.empty(2, device="meta")
    assert _ids(3, like=meta).device.type == "meta"
    assert _prefix(torch.tensor([2, 0, 3], device="meta"), 3).device.type == "meta"
    assert _ids(3, like=torch.zeros(1)).tolist() == [0, 1, 2]
    assert _prefix(torch.tensor([0, 2, 3]), 3).tolist() == [
        [False, False, False],
        [True, True, False],
        [True, True, True],
    ]


def test_the_intensity_model_receives_exactly_the_generated_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        generator, intensity = _models()
        batch = _batch(_ice_roots())
        spy = mock.Mock(wraps=intensity.forward)
        monkeypatch.setattr(intensity, "forward", spy)
        prediction = predict_spectra(
            generator,
            intensity,
            batch,
            max_nodes=9,
            max_depth=3,
            walk_steps=2,
            threshold=0,
        )
    received = cast("IntensityBatch", spy.call_args.args[0])
    counts = prediction.generation["nfrags"]
    width = int(counts.max())
    assert width < 9
    assert received.precursor is batch.precursor
    assert received.fragment_forms.shape[1] == width
    assert received.masses.shape[1] == width
    assert received.max_add_h.shape[1] == width
    assert received.max_remove_h.shape[1] == width
    assert [len(spectrum) for spectrum in prediction.spectra] == [
        int(count) * 2 * 13 for count in counts
    ]
    assert [len(mask) for mask in prediction.fragments] == [
        int(count) * 2 * 13 for count in counts
    ]


def _models() -> tuple[GraphGenerator, GraphIntensity]:
    generator_config = GraphGenerator.Config()
    generator_config.channels_in = 34
    generator_config.channels_hidden = 8
    generator_config.formula.max_count = 7
    generator_config.encoder.messages.num_layers = 2
    intensity_config = GraphIntensity.Config()
    intensity_config.channels_hidden = 8
    intensity_config.formula.max_count = 7
    intensity_config.encoder.messages.num_layers = 2
    intensity_config.num_layers = 2
    set_heads(intensity_config.block.attn, 2)
    _ = without_dropout(generator_config)
    _ = without_dropout(intensity_config)
    generator = generator_config.make()
    intensity = intensity_config.make()
    with torch.no_grad():
        for model in (generator, intensity):
            for index, parameter in enumerate(model.parameters()):
                if parameter.requires_grad:
                    values = torch.arange(parameter.numel(), dtype=torch.float32)
                    values = ((values + index * 3).remainder(29) - 14) / 101
                    _ = parameter.copy_(values.reshape(parameter.shape))
    return generator, intensity


def _batch(record: dict[str, Tensor]) -> IceBatch:
    graph = FragmentGraph(
        topology=GraphBatch(
            source=record["root/source"].clone(),
            destination=record["root/destination"].clone(),
            edge_type=record["root/edges/e_ind"].clone(),
            node_counts=record["root/node_counts"].clone(),
        ),
        features=record["root/nodes/h"].clone(),
        node_ids=record["root/nodes/n_id"].clone(),
    )
    roots = [take_graphs(graph, torch.tensor([index]))[0] for index in range(2)]
    return pack_ice_batch(
        roots,
        root_forms=record["input/root_forms"].clone(),
        element_masses=record["input/element_masses"].clone(),
        jobs=[
            ("first", "[M+H]+", "Orbitrap", 20.0),
            ("second", "[M+Na]+", "QTOF", float("nan")),
        ],
        precursor=[61.065, 107.083],
        device=torch.device("cpu"),
    )


type _IceMolecule = tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int], ...]]


def _ice_roots(*, singleton: bool = False) -> dict[str, Tensor]:
    """Return two prepared roots: methoxyethane and cyclohexane, or CH4 and NH3."""
    # Every atom sets its element, group and attached-hydrogen feature channels.
    chains: tuple[_IceMolecule, ...] = (
        (
            ((0, 22, 29), (0, 22, 28), (3, 23, 26), (0, 22, 29)),
            ((0, 1), (1, 2), (2, 3)),
        ),
        (((0, 22, 28),) * 6, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0))),
    )
    singletons: tuple[_IceMolecule, ...] = ((((0, 22, 30),), ()), (((1, 21, 29),), ()))
    molecules = singletons if singleton else chains
    atoms: list[tuple[int, int, int]] = []
    sources: list[int] = []
    destinations: list[int] = []
    forms = torch.zeros(len(molecules), 18)
    for index, (channels, bonds) in enumerate(molecules):
        left = [len(atoms) + first for first, _ in bonds]
        right = [len(atoms) + second for _, second in bonds]
        sources += left + right
        destinations += right + left
        atoms += channels
        for element, _, hydrogen in channels:
            forms[index, element] += 1
            forms[index, HYDROGEN_ROW] += hydrogen - 26
    features = torch.zeros(len(atoms), 32)
    for row, channels in enumerate(atoms):
        features[row, list(channels)] = 1
    counts = [len(channels) for channels, _ in molecules]
    return {
        "root/source": torch.tensor(sources, dtype=torch.int64),
        "root/destination": torch.tensor(destinations, dtype=torch.int64),
        "root/edges/e_ind": torch.ones(len(sources), dtype=torch.int64),
        "root/node_counts": torch.tensor(counts),
        "root/nodes/h": features,
        "root/nodes/n_id": torch.cat([torch.arange(count) for count in counts]),
        "input/root_forms": forms,
        "input/element_masses": torch.tensor(
            [[ELEMENT_MASSES[symbol]] for symbol in SIMULATOR_SYMBOLS],
            dtype=torch.float64,
        ),
    }


def _assert_roots(graph: FragmentGraph, inputs: dict[str, Tensor]) -> None:
    topology = graph.topology
    assert torch.equal(topology.source, inputs["root/source"])
    assert torch.equal(topology.destination, inputs["root/destination"])
    assert torch.equal(topology.edge_type, inputs["root/edges/e_ind"])
    assert torch.equal(topology.node_counts, inputs["root/node_counts"])
    assert torch.equal(graph.features, inputs["root/nodes/h"])
    assert torch.equal(graph.node_ids, inputs["root/nodes/n_id"])


def test_complete_prediction_backend_without_a_chemistry_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, name="rdkit", value=None)
    config = _ice_backend_config(tmp_path)
    actual: dict[str, Tensor] = {}
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        backend = config.make()
        for label, adduct, instrument, energy in (
            ("proton", "[M+H]+", "Orbitrap", 20.0),
            ("sodium", "[M+Na]+", "QTOF", 40.0),
        ):
            predictions = backend.predict(
                [
                    (smiles, adduct, instrument, energy)
                    for smiles in ("CCOC", "C1CCCCC1")
                ],
            )
            for index, prediction in enumerate(predictions):
                assert prediction is not None
                actual[f"{label}/{index}/mz"] = torch.from_numpy(prediction[0])
                actual[f"{label}/{index}/intensity"] = torch.from_numpy(prediction[1])
        assert backend.feature_failures == 0
        with pytest.raises(ValueError, match="Rejected prepared ICE root"):
            _ = backend.predict([("invalid", "[M+H]+", "Orbitrap", 20.0)])
        with pytest.raises(KeyError):
            _ = backend.predict([("unprepared", "[M+H]+", "Orbitrap", 20.0)])
    golden = _BASELINE / "testdata" / "ice_backend.pt"
    assert_tensor_golden(golden, actual)
    released: list[bool] = []
    monkeypatch.setattr(
        torch.cuda,
        name="empty_cache",
        value=lambda: released.append(True),
    )
    backend.release_cache()
    assert released == []
    backend.device = torch.device("cuda")
    backend.release_cache()
    assert released == [True]


def test_checkpoint_reader_rejects_incomplete_released_state(tmp_path: Path) -> None:
    config = _ice_backend_config(tmp_path)
    torch.save({"state_dict": {}}, config.generator_checkpoint)
    with pytest.raises(ValueError, match="Checkpoint keys differ"):
        _ = config.make()


def _ice_backend_config(directory: Path) -> IceBackend.Config:
    _prepared_roots(directory)
    config = IceBackend.Config()
    config.root_dir = directory
    config.generator.channels_in = 34
    config.generator.channels_hidden = 8
    config.generator.formula.max_count = 7
    config.generator.encoder.messages.num_layers = 2
    config.intensity.channels_hidden = 8
    config.intensity.formula.max_count = 7
    config.intensity.encoder.messages.num_layers = 2
    config.intensity.num_layers = 2
    set_heads(config.intensity.block.attn, 2)
    _ = without_dropout(config)
    config.generator_checkpoint = directory / "generator.pt"
    config.intensity_checkpoint = directory / "intensity.pt"
    with torch.random.fork_rng(devices=[]), torch.no_grad(), host_agnostic_numerics():
        for model, path in (
            (config.generator.make(), config.generator_checkpoint),
            (config.intensity.make(), config.intensity_checkpoint),
        ):
            assert isinstance(model, GraphGenerator | GraphIntensity)
            for index, parameter in enumerate(model.parameters()):
                if parameter.requires_grad:
                    values = torch.arange(parameter.numel(), dtype=torch.float32)
                    values = ((values + index * 3).remainder(29) - 14) / 101
                    _ = parameter.copy_(values.reshape(parameter.shape))
            _checkpoint(model, path)
    return config


def _checkpoint(model: ReleasedModel, path: Path) -> None:
    native = model.state_dict()
    state = {
        released: native[name] for released, name in model.checkpoint_names().items()
    }
    torch.save({"state_dict": state}, path)


def _prepared_roots(directory: Path) -> None:
    hydrogen, oxygen = ELEMENT_MASSES["H"], ELEMENT_MASSES["O"]
    rows: list[IceRow] = []
    # (smiles, element, degree and hydrogen channels per atom, bonds, hydrogens).
    for smiles, channels, bonds, hydrogens in (
        (
            "CCOC",
            ((0, 22, 29), (0, 22, 28), (3, 23, 26), (0, 22, 29)),
            [[0, 1], [1, 2], [2, 3]],
            [3, 2, 0, 3],
        ),
        (
            "C1CCCCC1",
            ((0, 22, 28),) * 6,
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
            [2] * 6,
        ),
    ):
        features = np.zeros((len(channels), 32))
        for atom, active in enumerate(channels):
            features[atom, list(active)] = 1
        masses = [oxygen if active[0] == 3 else 12.0 for active in channels]
        hydrogenated = [
            m + h * hydrogen for m, h in zip(masses, hydrogens, strict=True)
        ]
        formula = np.zeros(18)
        formula[[0, 3, 7]] = (
            sum(active[0] == 0 for active in channels),
            sum(active[0] == 3 for active in channels),
            sum(hydrogens),
        )
        rows.append(
            {
                "input_smiles": smiles,
                "smiles": smiles,
                "valid": True,
                "features": from_plain(features.tolist(), list[list[float]]),
                "bonds": bonds,
                "bond_orders": [1] * len(bonds),
                "formula": from_plain(formula.tolist(), list[float]),
                "hydrogens": hydrogens,
                "atom_masses": masses,
                "hydrogenated_masses": hydrogenated,
                "mass": sum(hydrogenated),
            },
        )
    rows.append({**rows[0], "input_smiles": "invalid", "valid": False})
    parquet.write_table(Table.from_pylist(rows), directory / "ice.parquet")
    np.save(
        directory / "element_masses.npy",
        np.array([[ELEMENT_MASSES[symbol]] for symbol in SIMULATOR_SYMBOLS]),
    )
    _ = (directory / "manifest.json").write_text(
        json.dumps(to_plain({"rdkit": SIMULATOR_RDKIT, "format_version": 1})),
    )


def test_ice_backend_the_default_root_directory_is_where_the_preparer_writes() -> None:
    assert IceBackend.Config().root_dir == ICE_ROOT_DIR


def test_complete_job_schedulers_and_failure_boundaries() -> None:
    actual: dict[str, Tensor] = {}
    for case, glacier, budget in (
        ("ice_complete", False, False),
        ("ice_budget", False, True),
        ("gl_complete", True, False),
        ("gl_budget", True, True),
        ("gl_failfast", True, False),
    ):
        jobs, canonical = _jobs_jobs(case)
        clock = _Clock()
        backend = _Backend(glacier=glacier)
        state = SimulationJobState(batch_size=32 if glacier else 8)
        if case == "gl_complete":
            state.canonical_cache[("warm", "[M+H]+", "Orbitrap", 20.0)] = (
                np.array([5.0]),
                np.array([1.0]),
            )
        finished, error = False, False
        try:
            wall = _wall(clock, 0.25 if budget else float("inf"))
            if glacier:
                finished = (
                    GlacierScheduling.Config()
                    .make()
                    .predict(
                        jobs,
                        canonical=canonical,
                        state=state,
                        backend=backend,
                        budget=wall,
                    )
                )
            else:
                finished = (
                    IceScheduling.Config()
                    .make()
                    .predict(
                        jobs,
                        canonical=canonical,
                        state=state,
                        backend=backend,
                        budget=wall,
                    )
                )
        except RuntimeError:
            if case != "gl_failfast":
                raise
            error = True
        actual[f"{case}/finished"] = torch.tensor(finished)
        actual[f"{case}/error"] = torch.tensor(error)
        actual[f"{case}/batch_size"] = torch.tensor(state.batch_size)
        actual[f"{case}/counts"] = torch.tensor(
            [state.predicted, state.failed, state.feature_failures, clock.calls],
        )
        actual[f"{case}/elapsed"] = torch.tensor(state.seconds, dtype=torch.float64)
        actual[f"{case}/last_error"] = _text(state.last_error)
        actual[f"{case}/calls"] = _text(
            "\n".join(
                ";".join("\t".join(map(str, job)) for job in call)
                for call in backend.calls
            ),
        )
        _cache_record(case + "/cache", state.cache, actual)
        if glacier:
            _cache_record(case + "/canonical_cache", state.canonical_cache, actual)
        if case == "gl_failfast":
            assert state.predicted >= 64
            assert state.predicted == state.failed
    golden = _BASELINE / "testdata" / "simulator_jobs.pt"
    assert_tensor_golden(golden, float32_comparands(actual))


def test_complete_length_boundaries_and_accumulated_time() -> None:
    actual: dict[str, Tensor] = {}
    for length, count in (
        (60, 9),
        (61, 9),
        (90, 5),
        (91, 5),
        (140, 5),
        (141, 3),
        (62, 9),
    ):
        names = [
            "C" * index + "N" + "C" * (length - index - 1) for index in range(count)
        ]
        jobs: set[SimulationJob] = {
            (name, "[M+H]+", "Orbitrap", 20.0) for name in names
        }
        for budget in (False, True) if length == 62 else (False,):
            label = f"{length}/{'budget' if budget else 'complete'}"
            clock = _Clock()
            clock.calls = 400
            state = SimulationJobState(batch_size=16, seconds=7.0)
            backend = _Backend(glacier=False, capacity=16)
            if length == 62 and not budget:
                state.cache[min(jobs)] = (np.array([5.0]), np.array([1.0]))
            finished = (
                IceScheduling.Config()
                .make()
                .predict(
                    jobs,
                    canonical={},
                    state=state,
                    backend=backend,
                    budget=_wall(clock, 100.25 if budget else float("inf")),
                )
            )
            actual[label + "/finished"] = torch.tensor(finished)
            actual[label + "/batch_size"] = torch.tensor(state.batch_size)
            actual[label + "/counts"] = torch.tensor(
                [state.predicted, state.failed, state.feature_failures, clock.calls],
            )
            actual[label + "/elapsed"] = torch.tensor(
                state.seconds,
                dtype=torch.float64,
            )
            actual[label + "/calls"] = _text(
                "\n".join(
                    ";".join("\t".join(map(str, job)) for job in call)
                    for call in backend.calls
                ),
            )
            _cache_record(label + "/cache", state.cache, actual)
    golden = _BASELINE / "testdata" / "simulator_boundaries.pt"
    assert_tensor_golden(golden, float32_comparands(actual))


def test_failure_diagnostic_retains_first_three_hundred_characters() -> None:
    state = SimulationJobState(batch_size=1)
    state.remember_failure(ValueError("x" * 400))
    assert state.last_error == "ValueError('" + "x" * 288
    state.remember_failure(RuntimeError("short"))
    assert state.last_error == "RuntimeError('short')"


def _jobs_jobs(
    case: str,
) -> tuple[set[SimulationJob], dict[str, tuple[str | None, int]]]:
    names = ["C" * length for length in (4, 5, 6, 7, 8, 9, 61, 91, 141)] + [
        "bad",
        "empty",
    ]
    canonical: dict[str, tuple[str | None, int]] = {}
    if case.startswith("gl_"):
        names = ["C" * length for length in (4, 5, 6, 7, 8, 9, 80, 90, 100)] + [
            "bad",
            "empty",
            "feature-error",
            "alias",
            "warm",
            "rejected",
        ]
        canonical = {name: (name, len(name)) for name in names}
        canonical["alias"] = ("CCCC", 4)
        canonical["rejected"] = (None, 0)
    if case == "gl_failfast":
        names = [f"bad{index:02}" for index in range(70)]
        canonical = {name: (name, 5) for name in names}
    jobs = {(name, "[M+H]+", "Orbitrap", 20.0) for name in names}
    if case != "gl_failfast":
        jobs.update((name, "[M+Na]+", "QTOF", 40.0) for name in names[:2])
    return jobs, canonical


class _Clock:
    def __init__(self) -> None:
        self.calls = 0

    def time(self) -> float:
        result = self.calls * 0.25
        self.calls += 1
        return result


def _wall(clock: _Clock, deadline: float) -> WallBudget:
    """Start a budget ending at an absolute clock reading without a net clock draw."""
    clock.calls -= 1
    started = clock.calls * 0.25
    return WallBudget.Config(seconds=deadline - started, clock=clock.time).make()


class _Backend:
    def __init__(self, *, glacier: bool, capacity: int = 4) -> None:
        self.glacier = glacier
        self.capacity = capacity
        self.calls: list[list[SimulationJob]] = []
        self.feature_failures = 0
        self.releases = 0

    def predict(self, jobs: list[SimulationJob]) -> list[PredictedPeaks | None]:
        selected: list[SimulationJob] = []
        indices: list[int] = []
        result: list[PredictedPeaks | None] = [None] * len(jobs)
        for index, job in enumerate(jobs):
            if self.glacier and job[0] == "feature-error":
                self.feature_failures += 1
            else:
                selected.append(job)
                indices.append(index)
        if not selected:
            return result
        self.calls.append(selected)
        if len(selected) > self.capacity:
            raise RuntimeError("out of memory")
        if any(job[0].startswith("bad") for job in selected):
            raise RuntimeError("prediction failure")
        for index, job in zip(indices, selected, strict=True):
            values = (
                np.empty((0, 2), dtype=np.float32)
                if job[0] == "empty"
                else np.array(
                    [
                        [len(job[0]), 1.0],
                        [len(job[0]) + 1, job[3] / 10 + 1],
                        [len(job[0]), 1.0],
                    ],
                    dtype=np.float32,
                )
            )
            result[index] = post_prediction(
                values,
                reduction=(
                    MaxReduction.Config() if self.glacier else SumReduction.Config()
                ).make(),
                max_peaks=100,
            )
        return result

    def release_cache(self) -> None:
        self.releases += 1


def _cache_record(
    prefix: str,
    cache: dict[SimulationJob, PreparedPeaks | None],
    record: dict[str, Tensor],
) -> None:
    keys = sorted(cache)
    record[prefix + "/keys"] = _text(
        "\n".join("\t".join(map(str, key)) for key in keys),
    )
    record[prefix + "/valid"] = torch.tensor([cache[key] is not None for key in keys])
    for index, key in enumerate(keys):
        peaks = cache[key]
        if peaks is not None:
            record[f"{prefix}/{index}/mz"] = torch.from_numpy(peaks[0])
            record[f"{prefix}/{index}/intensity"] = torch.from_numpy(peaks[1])


_SCORING: Final = QueryScoring.Config().make()


def test_both_complete_outer_publications_caches_and_budget_boundaries(
    tmp_path: Path,
) -> None:
    golden = _BASELINE / "testdata/simulator_outer.pt"
    _write(
        tmp_path,
        admission_rows(),
    )
    admission = SimulatorAdmissionTable(tmp_path)
    actual: dict[str, Tensor] = {}
    for family in ("iceberg", "glacier"):
        for case, budget in (("complete", 0.0), ("before", 0.5), ("partial", 2.75)):
            label = f"{family}/{case}"
            boundary = _Boundary(glacier=family == "glacier")
            result = run_simulator(
                _molecules(),
                scheduling=_scheduling(family == "glacier"),
                scoring=_SCORING,
                admission_factory=lambda: admission,
                backend_factory=boundary.load,
                publish=boundary.publish,
                budget=_budget(_Clock(), budget),
                batch_size=2,
                chunk_jobs=2,
            )
            actual.update(
                observe_result(label, result, boundary, glacier=family == "glacier"),
            )
            if case == "partial":
                assert result.jobs.predicted > 0
                assert result.scored_molecules == 1
                assert all(value is None for value in result.scores["d"].values())
            assert result.status == ("ok" if case == "complete" else "budget")
    assert_tensor_golden(golden, actual)


def test_empty_input_geometry_and_load_failures_preserve_initial_publication(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        admission_rows(),
    )
    admission = SimulatorAdmissionTable(tmp_path)
    for batch_size, chunk_jobs in ((0, 3), (3, 0)):
        boundary = _Boundary(glacier=False)
        with pytest.raises(
            ValueError,
            match=r"^Simulator batch and chunk sizes must be positive\.$",
        ):
            _ = run_simulator(
                (),
                scheduling=_scheduling(False),
                scoring=_SCORING,
                admission_factory=lambda: admission,
                backend_factory=boundary.load,
                publish=boundary.publish,
                budget=_budget(_Clock(), 0.0),
                batch_size=batch_size,
                chunk_jobs=chunk_jobs,
            )
        assert boundary.publications == []
    boundary = _Boundary(glacier=False)
    empty = run_simulator(
        (),
        scheduling=_scheduling(False),
        scoring=_SCORING,
        admission_factory=lambda: admission,
        backend_factory=boundary.load,
        publish=boundary.publish,
        budget=_budget(_Clock(), 0.0),
        batch_size=3,
        chunk_jobs=3000,
    )
    assert boundary.publications == [{}, {}]
    assert empty.status == "ok"
    assert empty.covered == empty.scored_molecules == empty.jobs.predicted == 0
    boundary = _Boundary(glacier=False)

    def stopped() -> _Backend:
        raise KeyboardInterrupt("load interrupted")

    failed = run_simulator(
        _molecules()[:1],
        scheduling=_scheduling(False),
        scoring=_SCORING,
        admission_factory=lambda: admission,
        backend_factory=stopped,
        publish=boundary.publish,
        budget=_budget(_Clock(), 0.0),
        batch_size=3,
        chunk_jobs=3000,
    )
    assert failed.status == "error"
    assert failed.errors == ["KeyboardInterrupt('load interrupted')"]
    assert len(boundary.publications) == 2
    assert boundary.publications[0] == boundary.publications[1]


def test_deadline_epoch_includes_loading_and_missing_admission_is_explicit(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        admission_rows(),
    )
    admission = SimulatorAdmissionTable(tmp_path)
    boundary = _Boundary(glacier=False)
    clock = _Clock()

    def load() -> _Backend:
        clock.calls += 100
        return boundary.load()

    result = run_simulator(
        _molecules()[:1],
        scheduling=_scheduling(False),
        scoring=_SCORING,
        admission_factory=lambda: admission,
        backend_factory=load,
        publish=boundary.publish,
        budget=_budget(clock, 1.0),
        batch_size=3,
        chunk_jobs=3000,
    )
    assert result.status == "budget"
    assert result.jobs.predicted == 0
    assert boundary.backend.calls == []
    assert boundary.publications[0] == boundary.publications[-1]
    boundary = _Boundary(glacier=False)
    absent = run_simulator(
        (replace(_molecules()[0], candidates=("unstaged",)),),
        scheduling=_scheduling(False),
        scoring=_SCORING,
        admission_factory=lambda: admission,
        backend_factory=boundary.load,
        publish=boundary.publish,
        budget=_budget(_Clock(), 0.0),
        batch_size=3,
        chunk_jobs=3000,
    )
    assert absent.status == "error"
    assert absent.errors == ["KeyError('unstaged')"]
    assert absent.scores == {"a": {"unstaged": None}}


def test_real_query_and_model_shape_failures_remain_per_molecule(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        admission_rows(),
    )
    admission = SimulatorAdmissionTable(tmp_path)
    malformed: SpectrumRequest = {
        **_molecules()[0].spectra[0],
        "mz": [1.0, 2.0],
        "it": [1.0],
    }
    good = replace(_molecules()[0], candidates=("CCO",))
    rejected = replace(good, mid="rejected", candidates=("*",))
    broken = replace(good, mid="bad", spectra=(malformed,))
    boundary = _Boundary(glacier=False)
    isolated = run_simulator(
        (broken, good, rejected),
        scheduling=_scheduling(False),
        scoring=_SCORING,
        admission_factory=lambda: admission,
        backend_factory=boundary.load,
        publish=boundary.publish,
        budget=_budget(_Clock(), 0.0),
        batch_size=3,
        chunk_jobs=3000,
    )
    assert isolated.status == "ok"
    assert isolated.covered == 2
    assert isolated.scored_molecules == isolated.scored_candidates == 1
    assert isolated.scores["bad"] == {"CCO": None}
    assert isolated.scores["rejected"] == {"*": None}
    assert isolated.errors[0].startswith("bad: plan ")
    assert isolated.bad_smiles == 1
    boundary = _Boundary(glacier=False)
    boundary.backend = _MalformedBackend(glacier=False)
    unscored = run_simulator(
        (good,),
        scheduling=_scheduling(False),
        scoring=_SCORING,
        admission_factory=lambda: admission,
        backend_factory=boundary.load,
        publish=boundary.publish,
        budget=_budget(_Clock(), 0.0),
        batch_size=3,
        chunk_jobs=3000,
    )
    assert unscored.status == "ok"
    assert unscored.jobs.predicted == 1
    assert unscored.scored_molecules == unscored.scored_candidates == 0
    assert unscored.scores == {"a": {"CCO": None}}
    assert len(unscored.errors) == 1
    assert unscored.errors[0].startswith("a: score ")


def test_first_fifty_distinct_rejections_abort_only_without_an_accepted_string(
    tmp_path: Path,
) -> None:
    rows = admission_rows()
    invalid: list[SimulatorAdmissionRow] = [
        {**rows[8], "smiles": f"bad{index}"} for index in range(50)
    ]
    _write(tmp_path, [*rows, *invalid])
    admission = SimulatorAdmissionTable(tmp_path)
    for accept in (False, True):
        candidates = (("CCO",) if accept else ()) + tuple(
            row["smiles"] for row in invalid
        )
        molecule = replace(_molecules()[0], candidates=candidates + candidates[:1])
        boundary = _Boundary(glacier=False)
        result = run_simulator(
            (molecule,),
            scheduling=_scheduling(False),
            scoring=_SCORING,
            admission_factory=lambda: admission,
            backend_factory=boundary.load,
            publish=boundary.publish,
            budget=_budget(_Clock(), 0.0),
            batch_size=3,
            chunk_jobs=3000,
        )
        assert result.bad_smiles == 50
        assert len(result.errors) == (5 if accept else 6)
        assert result.errors[:5] == [
            f"smiles rejected (rdkit parse): bad{index}" for index in range(5)
        ]
        if accept:
            assert result.status == "ok"
            assert result.scored_molecules == result.jobs.predicted == 1
        else:
            assert result.status == "error"
            assert (
                result.errors[-1]
                == "RuntimeError('first 50 SMILES all rejected, last: rdkit parse')"
            )
            assert result.scored_molecules == result.jobs.predicted == 0
        assert len(boundary.publications[0]["a"]) == len(candidates)


class _MalformedBackend(_Backend):
    @override
    def predict(self, jobs: list[SimulationJob]) -> list[PredictedPeaks | None]:
        """Expose an invalid model output shape to the real scoring implementation."""
        return [
            (np.zeros((2, 3), np.float32), np.ones((2, 3), np.float32)) for _ in jobs
        ]


class _Boundary:
    def __init__(self, *, glacier: bool) -> None:
        self.backend = _Backend(glacier=glacier)
        self.publications: list[dict[str, dict[str, float | None]]] = []

    def load(self) -> _Backend:
        """Require the all-null publication before backend construction begins."""
        assert len(self.publications) == 1
        assert all(
            value is None
            for row in self.publications[0].values()
            for value in row.values()
        )
        return self.backend

    def publish(self, scores: dict[str, dict[str, float | None]]) -> None:
        """Retain every complete immutable observation of the public score map."""
        self.publications.append({mid: dict(row) for mid, row in scores.items()})


def _scheduling(glacier: bool) -> IceScheduling | GlacierScheduling:
    return (GlacierScheduling if glacier else IceScheduling).Config().make()


def _budget(clock: _Clock, seconds: float) -> WallBudget.Config:
    """Bound a stage by `seconds` on a counting clock; zero leaves it unlimited."""
    return WallBudget.Config(seconds=seconds or float("inf"), clock=clock.time)


def _molecules() -> tuple[SimulationMolecule, ...]:
    rows: list[SimulationMolecule] = []
    for mid, candidates, mode, energy in (
        ("a", ("CCO", "OCC", "CCO", ""), "positive", 20.0),
        ("b", ("CCO", "*"), "positive", 20.0),
        ("c", ("OCC",), "negative", 20.0),
        ("d", ("OCC", "C" * 100), "positive", None),
    ):
        spectrum: SpectrumRequest = {
            "mz": [3.0, 4.0, 20.0],
            "it": [1.0, 0.3, 0.1],
            "prec": 100.0,
            "mode": mode,
            "adduct": "[M+H]+",
            "instrument": "Orbitrap",
            "ce": energy,
        }
        rows.append(
            SimulationMolecule(mid=mid, candidates=candidates, spectra=(spectrum,)),
        )
    return tuple(rows)


def _publication(
    prefix: str,
    values: dict[str, dict[str, float | None]],
    record: dict[str, Tensor],
) -> None:
    record[prefix + "/keys"] = _text(
        "\n".join(
            f"{mid}\t{candidate}" for mid, row in values.items() for candidate in row
        ),
    )
    ordered = [score for row in values.values() for score in row.values()]
    record[prefix + "/valid"] = torch.tensor([score is not None for score in ordered])
    record[prefix + "/values"] = torch.tensor(
        [score if score is not None else 0.0 for score in ordered],
        dtype=torch.float64,
    )


def observe_result(
    prefix: str,
    result: SimulatorResult,
    boundary: _Boundary,
    *,
    glacier: bool,
) -> dict[str, Tensor]:
    """Observe every outer record field through a shared native boundary."""
    actual = {
        prefix + "/status": _text(result.status),
        prefix + "/counts": torch.tensor(
            [
                len(result.scores),
                result.scored_molecules,
                result.covered,
                sum(len(row) for row in result.scores.values()),
                result.scored_candidates,
                result.jobs.predicted,
                result.jobs.failed,
                result.bad_smiles,
            ],
        ),
        prefix + "/elapsed": torch.tensor(result.jobs.seconds, dtype=torch.float64),
        prefix + "/errors": _text("\n".join(result.errors)),
        prefix + "/batch_size": torch.tensor(result.jobs.batch_size),
        prefix + "/feature_failures": torch.tensor(result.jobs.feature_failures),
        prefix + "/publication_count": torch.tensor(len(boundary.publications)),
        prefix + "/calls": _text(
            "\n".join(
                ";".join("\t".join(map(str, job)) for job in call)
                for call in boundary.backend.calls
            ),
        ),
    }
    for index, publication in enumerate(boundary.publications):
        _publication(f"{prefix}/publication/{index}", publication, actual)
    _cache_record(prefix + "/cache", result.jobs.cache, actual)
    if glacier:
        _cache_record(prefix + "/canonical_cache", result.jobs.canonical_cache, actual)
    return float32_comparands(actual)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
