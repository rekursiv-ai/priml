"""Checks for the main engine module."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

import hashlib
import json
import math
import pickle
import sys

from pyarrow import Table, parquet
from torch import Tensor
from treekle import ReadError, from_plain, json_loads, to_plain

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026.boosting import LightGBMModel
from priml.baselines.casmi2026.boosting_test import GOLDEN, tiny_text
from priml.baselines.casmi2026.chemistry import (
    CandidatePool,
    MainSearch,
    PeakStore,
    SpectralLibrary,
)
from priml.baselines.casmi2026.custom_types import (
    Analog,
    ReactionRequest,
    SpectrumRequest,
)
from priml.baselines.casmi2026.dreams import (
    MainBank,
    ModelViews,
    keep_count,
    modal_adduct,
)
from priml.baselines.casmi2026.dreams_test import (
    checkpoint_config,
    source_checkpoint,
)
from priml.baselines.casmi2026.fragnet import FragnetTable, StageA
from priml.baselines.casmi2026.isolated import (
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    AnalogRecord,
    AnalogTable,
    ElementGraph,
    IntrinsicTable,
    formula_vector as family_data_formula_vector,
    main_ion_deltas,
    normalize_request,
    record_fields,
)
from priml.baselines.casmi2026.main_engine import (
    CandidateRows,
    Combo,
    DerivationContext,
    DerivationLog,
    DerivationPriors,
    DerivationTable,
    GeneratedCandidate,
    GenerationResult,
    MainEngine,
    MainFamilyData,
    MainFamilyRows,
    MainFragmentTable,
    MainLibraryEvidence,
    MainPoolFamilies,
    MainRetrieval,
    PriorTables,
    ReactionScreen,
    TreeBank,
    _main_analog_features,
    _main_library,
    _main_model_features,
    _stable_ranks,
    aggregate,
    analog_structure_features,
    candidates,
    combo_mass,
    delta_codes,
    derivation_features,
    diagnostic_losses,
    family_features,
    formula_vector as engine_formula_vector,
    frag2_features,
    load_tree,
    main_feature_names,
    main_features,
    main_fragment_features,
    main_subformula_features,
    main_subformula_masses,
    select_features,
    supports,
    training_smiles,
)
from priml.baselines.casmi2026.testing import (
    float32_comparands,
    retrieval_inputs,
)
from priml.math import stats
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_tensor_golden, read_tensors


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import (
        FragnetRecord,
        ProposalPath,
        ReactionProduct,
    )
    from priml.baselines.casmi2026.main_engine import MainFamilyProvider
    from priml.math.custom_types import Tensorable


_BASELINE: Final = _CWD


ANALOG_STRUCTURE_GOLDEN: Final = _BASELINE / "testdata/analog_family.pt"


def test_complete_structural_family_with_every_prepared_graph() -> None:
    golden = ANALOG_STRUCTURE_GOLDEN
    expected = read_tensors(golden)
    actual = {
        name: value.clone()
        for name, value in expected.items()
        if name.startswith("input/")
    }
    structures: list[list[AnalogRecord]] = []
    for name, count in (("candidate", 6), ("analog", 16)):
        rows: list[AnalogRecord] = []
        for index in range(count):
            prefix = f"input/{name}/{index}"
            graphs: list[ElementGraph | None] = [
                ElementGraph(
                    elements=expected[f"{prefix}/{graph}/elements"].numpy(),
                    bonds=expected[f"{prefix}/{graph}/bonds"].numpy(),
                )
                if expected[f"{prefix}/{graph}/valid"].item()
                else None
                for graph in ("graph", "core")
            ]
            strings = {
                field: expected[prefix + "/" + field].numpy().tobytes().decode()
                for field in ("smiles", "formula", "element_smiles", "query_smiles")
            }
            rows.append(
                AnalogRecord(
                    smiles=strings["smiles"],
                    formula_input=strings["formula"],
                    hashes=expected[prefix + "/hashes"].numpy(),
                    numbers=expected[prefix + "/numbers"].numpy(),
                    formula=expected[prefix + "/formula_vector"].numpy(),
                    core_smiles=strings["query_smiles"],
                    element_smiles=strings["element_smiles"],
                    query_smiles=strings["query_smiles"],
                    graph=graphs[0],
                    core=graphs[1],
                    core_mask=np.zeros(
                        len(graphs[0].elements) if graphs[0] else 0,
                        np.bool_,
                    ),
                ),
            )
        structures.append(rows)
    candidates, analogs = structures
    differences = (
        np.stack([row.formula for row in candidates]).astype(np.int64)[:, None]
        - np.stack([row.formula for row in analogs]).astype(np.int64)[None]
    )
    actual["code"] = torch.from_numpy(delta_codes(differences))
    for case, (size, count) in enumerate(((6, 16), (6, 4), (6, 1), (6, 0), (0, 16))):
        output = analog_structure_features(
            candidates[:size],
            analogs=analogs[:count],
            similarities=expected["input/similarities"].numpy()[:count],
            analog_sid=np.arange(count, dtype=np.int64),
            parent_sid=expected["input/parents"].numpy()[:size],
            fingerprints=expected["input/fingerprints"].numpy()[:size],
            analog_fingerprints=expected["input/analog_fp"].numpy()[:count],
            bits=expected["input/bits"].numpy(),
        )
        actual[f"features/{case}"] = torch.from_numpy(
            np.column_stack(list(output.values())),
        )
    assert_tensor_golden(golden, actual)


def test_signed_codes_keep_vector_axes_and_exact_sentinel_boundaries() -> None:
    differences = np.zeros((2, 3, 10), np.int64)
    differences[0, 0, 0] = 40
    differences[0, 1, 1] = -40
    differences[0, 2, 0] = 41
    differences[1, 0, 1] = -41
    differences[1, 1, 9] = 40
    differences[1, 2, 9] = -40
    result = delta_codes(differences)
    assert np.array_equal(
        result,
        np.array(
            [
                [40, -40 * 81, np.iinfo(np.int64).max],
                [np.iinfo(np.int64).max, 40 * 81**9, -40 * 81**9],
            ],
            np.int64,
        ),
    )
    assert result.dtype == np.int64


def test_codes_keep_the_int64_absolute_value_wraparound() -> None:
    # np.abs maps the lowest int64 to itself, so it never exceeds the sentinel bound.
    differences = np.zeros((2, 10), np.int64)
    differences[0, 0] = differences[1, 9] = np.iinfo(np.int64).min
    np.testing.assert_array_equal(
        delta_codes(differences),
        np.array([-40, -40 * 81**9], np.int64),
        strict=True,
    )


_THIS: Final = Path(__file__).resolve()


def test_stable_ranks_reuses_shared_inversion_with_original_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = (np.arange(129) % 5).astype(np.float32)[::-1]
    order = np.argsort(-values, stable=True)
    expected = np.empty(len(values), np.float64)
    expected[order] = np.arange(len(values))
    observed: list[Tensor | None] = []
    original = stats.ordinal_ranks

    def capture(x: Tensorable, *, order: Tensor | None = None) -> Tensor:
        observed.append(order)
        return original(x, order=order)

    monkeypatch.setattr(stats, "ordinal_ranks", capture)
    actual = _stable_ranks(values)
    np.testing.assert_array_equal(
        actual.view(np.uint64),
        (expected / 128).view(np.uint64),
    )
    assert len(observed) == 1
    assert observed[0] is not None
    assert torch.equal(observed[0], torch.from_numpy(order))


def test_main_engine_empty_group_keeps_all94_columns() -> None:
    _, pool, _ = retrieval_inputs()
    rows = candidates(pool, target=900.0, ppm=10.0, fallback_ppm=30.0)
    empty = np.zeros(0, np.float64)
    result = main_features(
        rows,
        target=900.0,
        spectra=[],
        library=MainLibraryEvidence(
            maximum=empty,
            mean=empty,
            top3=empty,
            count=empty,
            cosine=empty,
            same_adduct=empty,
        ),
        analogs=(),
        # main_features reads the retrieval fixture's seven bits in one byte.
        train_fingerprints=np.zeros((9, 1), np.uint8),
        views=None,
        fragments=None,
        fragment_counts=np.zeros(0, np.float32),
        subformulas=None,
        similarity_power=4.0,
    )
    assert result.shape == (0, 94)


def test_entire_frag2_family_with_caps_and_ties() -> None:
    golden = _BASELINE / "testdata" / "frag2.pt"
    expected = read_tensors(golden)
    spectra: list[SpectrumRequest] = [
        {
            "mz": from_plain(
                expected[f"input/spectrum/{index}/mz"].tolist(),
                list[float],
            ),
            "it": from_plain(
                expected[f"input/spectrum/{index}/it"].tolist(),
                list[float],
            ),
            "prec": 151.0 if index == 2 else 600.0,
            "adduct": ("[M+H]+", "[M+Na]+", "[M-H]-")[index],
            "mode": -1 if index == 2 else 1,
        }
        for index in range(3)
    ]
    scores = expected["input/scores"].numpy()
    formulas = ("C2H6O", "C2H6O", "C3H8O", "C3H8O", "CH4")
    depth1 = [expected[f"input/depth1/{index}"].numpy() for index in range(5)]
    depth2 = [expected[f"input/depth2/{index}"].numpy() for index in range(5)]
    capabilities = expected["input/capabilities"].numpy()
    masses, specific, support = diagnostic_losses(capabilities)
    actual = {
        name: value for name, value in expected.items() if name.startswith("input/")
    }
    actual["loss/mass"] = torch.from_numpy(masses)
    actual["loss/specific"] = torch.from_numpy(specific)
    actual["loss/support"] = torch.from_numpy(support)
    for index, (query, limit, count) in enumerate(
        (
            (spectra, 600, 5),
            (spectra, 2, 5),
            (spectra, 3, 5),
            (list[SpectrumRequest](), 600, 5),
            (spectra, 600, 0),
        ),
    ):
        values = frag2_features(
            query,
            search=MainEngine.Config().fragment_search.make(),
            formulas=formulas[:count],
            model_score=scores[:count],
            depth1=depth1[:count],
            depth2=depth2[:count],
            capabilities=capabilities[:count],
            max_candidates=limit,
        )
        assert len(values) == 31
        actual[f"features/{index}"] = torch.from_numpy(
            np.column_stack(list(values.values())),
        )
    assert_tensor_golden(golden, float32_comparands(actual, inputs=("input/*",)))


def test_formula_vocabulary_retains_all_elements_and_complete_grammar() -> None:
    vector = engine_formula_vector("CH2N3O4P5S6F7Cl8Br9I10")
    assert vector is not None
    assert vector.dtype == np.int64
    assert np.array_equal(vector, np.arange(1, 11, dtype=np.int64))
    repeated = np.zeros(10, np.int64)
    repeated[:2] = [5, 1]
    vector = engine_formula_vector("C2C3H+-")
    assert vector is not None
    assert np.array_equal(vector, repeated)
    vector = engine_formula_vector("+")
    assert vector is not None
    assert np.array_equal(vector, np.zeros(10, np.int64))
    for formula in ("", "NaCl", "C.2H", "xCH", "C1xH", "[C]"):
        assert engine_formula_vector(formula) is None


def test_peaks_without_masses_or_intensity_are_skipped_by_the_explanation() -> None:
    masses = np.array([10.0, 20.0, 30.0])
    valid = (np.array([10.0, 20.0]), np.array([1.0, 0.5]), "[M+H]+")
    blank = [
        (np.zeros(0), np.zeros(0), "[M+H]+"),
        (np.array([10.0]), np.zeros(1), "[M+H]+"),
    ]
    assert np.array_equal(
        main_fragment_features(masses, peaks=[*blank, valid]),
        main_fragment_features(masses, peaks=[valid]),
    )
    assert not main_fragment_features(masses, peaks=blank).any()


def test_main_ranker_top_gap_keeps_float32_subtraction() -> None:
    _, pool, _ = retrieval_inputs()
    rows = candidates(pool, target=100.0, ppm=10.0, fallback_ppm=30.0, drop_row=0)
    logits = np.array([0.0, 0.0, -0.2, 0.3, 0.0, 0.0, 0.0], np.float32)
    # The pinned Full1 bank loads exactly one checkpoint for these base columns.
    views = ModelViews(
        single=np.tile(logits[None, None], (1, 3, 1)),
        merged=np.tile(logits[None, None], (1, 2, 1)),
        single_elements=np.zeros((1, 3, 10), np.float32),
        merged_elements=np.zeros((1, 2, 10), np.float32),
        modes=np.array([1, -1], np.float32),
    )
    values, scores = _main_model_features(rows, views=views)
    assert np.array_equal(scores, np.array([-0.2, 0.3], np.float32))
    gap = values["fz_grp_gap"]
    assert isinstance(gap, float)
    assert np.float32(gap) == (np.float32(0.3) - np.float32(-0.2)) / 100.0


def test_analog_sum_is_correctly_rounded_float32_log1p() -> None:
    similarity = 0.46771079301834106
    values = _main_analog_features(
        # _main_analog_features isolates one candidate's single similarity term.
        np.ones((1, 1), np.float32),
        (Analog(sid=0, similarity=similarity),),
        np.array([[128]], np.uint8),
        1.0,
    )
    expected = np.array([math.log1p(np.float32(similarity))], np.float32)
    assert np.array_equal(values["ap_sum"], expected)


def test_main_likelihood_is_evaluated_in_float64_and_rounded_once() -> None:
    _, pool, _ = retrieval_inputs()
    rows = replace(
        candidates(pool, target=100.0, ppm=10.0, fallback_ppm=30.0),
        fingerprints=np.array(
            [[1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]],
            np.float32,
        ),
    )
    logits = np.array(
        [
            -0.32868582010269165,
            -0.17201346158981323,
            0.22252917289733887,
            -0.4066565930843353,
            -0.29706329107284546,
            0.056098613888025284,
            -0.24937939643859863,
        ],
        np.float32,
    )
    views = ModelViews(
        single=logits[None, None],
        merged=logits[None, None],
        # ModelViews holds one Full1 checkpoint and one source query spectrum.
        single_elements=np.zeros((1, 1, 10), np.float32),
        merged_elements=np.zeros((1, 1, 10), np.float32),
        modes=np.ones(1, np.float32),
    )
    values, _ = _main_model_features(rows, views=views)
    one = np.float32(1)
    probability = [one / (one + np.float32(math.exp(-z))) for z in logits.tolist()]
    likelihood = np.zeros(3, np.float32)
    for row, bits in enumerate(rows.fingerprints.tolist()):
        present = absent = np.float32(0)
        for bit, p in zip(bits, probability, strict=True):
            present += np.float32(bit) * np.float32(math.log(p))
            absent += np.float32(1 - bit) * np.float32(math.log(one - p))
        likelihood[row] = present + absent
    assert np.array_equal(values["fp_ll"], likelihood / np.float32(7) * 100)


def test_complete_tree_asset_bank_binds_features_and_averages_its_models(
    tmp_path: Path,
) -> None:
    expected = read_tensors(GOLDEN)
    cases = (0, 3, 4, 5)
    models = [tiny_text(expected, case) for case in cases]
    payload = pickle.dumps(
        {
            "features": ["Column_0", "Column_1", "Column_2"],
            "boosters": models,
            "params": {},
            "rounds": 3,
        },
    )
    path = tmp_path / "ranker.pkl"
    _ = path.write_bytes(payload)
    bank = TreeBank.load(path, hashlib.sha256(payload).hexdigest())
    query = expected["case0/query"].numpy().astype(np.float32)
    actual = bank.score(query[:, ::-1], ["Column_2", "Column_1", "Column_0"])
    wanted = np.mean(
        [
            LightGBMModel.parse(text).predict(query.astype(np.float64))
            for text in models
        ],
        axis=0,
    )
    np.testing.assert_array_equal(actual, wanted, strict=True)
    assert len(bank.models) == 4
    assert [len(model.trees) for model in bank.models] == [3, 3, 3, 3]
    with pytest.raises(ValueError, match="missing"):
        _ = bank.score(query, ["Column_0", "other", "Column_2"])


def test_stage_b_text_is_read_as_a_complete_numeric_tree_asset(tmp_path: Path) -> None:
    text = tiny_text(read_tensors(GOLDEN), 0)
    path = tmp_path / "stageB.txt"
    _ = path.write_text(text)
    model = load_tree(path, hashlib.sha256(text.encode()).hexdigest())
    assert len(model.trees) == 3
    assert model.features == ("Column_0", "Column_1", "Column_2")


def test_tree_asset_digest_is_checked_before_decoding(tmp_path: Path) -> None:
    path = tmp_path / "ranker.pkl"
    _ = path.write_bytes(b"invalid model")
    with pytest.raises(RuntimeError, match="hashes to"):
        _ = TreeBank.load(path, "0" * 64)
    with pytest.raises(ValueError, match="requires its SHA256"):
        _ = TreeBank.load(path, "")


def test_tree_asset_rejects_pickled_runtime_objects(tmp_path: Path) -> None:
    payload = pickle.dumps(Path("runtime"))
    path = tmp_path / "ranker.pkl"
    _ = path.write_bytes(payload)
    with pytest.raises(pickle.UnpicklingError, match="plain data"):
        _ = TreeBank.load(path, hashlib.sha256(payload).hexdigest())


def test_each_booster_must_declare_the_complete_ranker_feature_order(
    tmp_path: Path,
) -> None:
    payload = pickle.dumps(
        {
            "features": ["wrong"],
            "boosters": [tiny_text(read_tensors(GOLDEN), 0)],
        },
    )
    path = tmp_path / "ranker.pkl"
    _ = path.write_bytes(payload)
    with pytest.raises(ValueError, match="feature"):
        _ = TreeBank.load(path, hashlib.sha256(payload).hexdigest())


class PreparedPaths:
    """Present every stored input through the real preparation boundary."""

    def __init__(self, record: dict[str, Tensor]) -> None:
        self.rows: dict[tuple[str, Combo], tuple[ProposalPath, ...]] = {}
        self.identities: dict[str, tuple[str | None, str | None]] = {}
        parents = _derivation_strings(record["input/parents"])
        combos = [
            tuple(combo.split("+"))
            for combo in _derivation_strings(record["input/combos"])
        ]
        strings = _derivation_strings(record["input/identity_smiles"])
        keys, skeletons = (
            _derivation_strings(record["input/identity_" + kind])
            for kind in ("key", "skeleton")
        )
        for i, string in enumerate(strings):
            self.identities[string] = (
                keys[i] if record["input/identity_key_valid"][i].item() else None,
                skeletons[i]
                if record["input/identity_skeleton_valid"][i].item()
                else None,
            )
        for parent in range(len(parents)):
            for position, combo in enumerate(combos):
                prefix = f"proposal/{parent * len(combos) + position}/"
                raw = _derivation_strings(record[prefix + "raw"])
                standardized = _derivation_strings(record[prefix + "standardized"])
                rows: list[ProposalPath] = []
                for index in range(int(record[prefix + "size"].item())):
                    key, skeleton = self.identity(raw[index])
                    rows.append(
                        {
                            "raw": raw[index],
                            "standardized": standardized[index]
                            if record[prefix + "valid"][index].item()
                            else None,
                            "key": key,
                            "skeleton": skeleton,
                            "environments": from_plain(
                                cast(
                                    object,
                                    record[prefix + "environments"][index].tolist(),
                                ),
                                tuple[tuple[int, int, int], ...],
                            ),
                        },
                    )
                self.rows[parents[parent], combo] = tuple(rows)

    def paths(
        self,
        parent_smiles: str,
        combo: Combo,
    ) -> tuple[ProposalPath, ...]:
        """Require a staged pair, including an explicitly empty tuple."""
        return self.rows[parent_smiles, combo]

    def identity(self, smiles: str) -> tuple[str | None, str | None]:
        """Require an explicitly staged nullable molecular identity."""
        return self.identities[smiles]


def test_entire_prior_path_family_supports_duplicates_and_all41_features() -> None:
    golden = _BASELINE / "testdata/derivation.pt"
    expected = read_tensors(golden)
    actual = {
        name: value.clone()
        for name, value in expected.items()
        if name.startswith(("input/", "proposal/"))
    }
    proposals = PreparedPaths(expected)
    combos = [
        tuple(combo.split("+"))
        for combo in _derivation_strings(expected["input/combos"])
    ]
    parents = _derivation_strings(expected["input/parents"])
    tables: PriorTables = {
        "glob": expected["input/glob"].item(),
        "tf": {
            name: (
                expected["input/tf_counts"][i, 0].item(),
                expected["input/tf_counts"][i, 1].item(),
            )
            for i, name in enumerate(_derivation_strings(expected["input/tf_names"]))
        },
        "env": {1: {}, 2: {}, 3: {}},
    }
    for i, name in enumerate(_derivation_strings(expected["input/env_names"])):
        radius, environment = (
            int(expected["input/env_ids"][i, j].item()) for j in range(2)
        )
        tables["env"][radius][name, environment] = (
            expected["input/env_counts"][i, 0].item(),
            expected["input/env_counts"][i, 1].item(),
        )
    priors = DerivationPriors(tables, proposals=proposals)
    for parent in range(len(parents)):
        for position, combo in enumerate(combos):
            group = priors.group(parents[parent], combo)
            prefix = f"group/{parent * len(combos) + position}/"
            actual[prefix + "smiles"] = torch.tensor(
                list("\0".join(group).encode()),
                dtype=torch.uint8,
            )
            actual[prefix + "values"] = torch.tensor(
                [
                    [row.tf_logfr, row.site_logfr, row.sibling_rank, row.siblings]
                    for row in group.values()
                ],
                dtype=torch.float64,
            ).reshape(-1, 4)
    actual["mass"] = torch.tensor(
        [
            combo_mass(combo)
            for combo in (*combos, ("dehexose2", "methyl_O", "demethyl_O"))
        ],
        dtype=torch.float64,
    )
    smiles = _derivation_strings(expected["input/smiles"])
    log_ident = _derivation_strings(expected["input/log_ident"])
    logs: list[DerivationLog] = []
    for i in range(len(expected["input/logs"])):
        sid, combo, kind, how = (
            int(expected["input/logs"][i, j].item()) for j in range(4)
        )
        logs.append(
            DerivationLog(
                parent=sid,
                similarity=expected["input/log_sim"][i].item(),
                combo=combos[combo] if combo >= 0 else None,
                kind="gen" if kind else "pool",
                how="key" if how else "smi",
                identity=log_ident[i],
            ),
        )
    for case, (size, own, log) in enumerate(
        ((10, True, True), (10, False, False), (0, True, True)),
    ):
        context = DerivationContext(
            smiles=tuple(smiles[:size]),
            keys=tuple(proposals.identity(smile)[0] for smile in smiles[:size]),
            formulas=tuple(_derivation_strings(expected["input/formulas"])[:size]),
            generated=expected["input/generated"].numpy()[:size].astype(np.bool_)
            if own
            else np.zeros(size, np.bool_),
            parent_sid=expected["input/parent_sid"].numpy()[:size],
            similarities=expected["input/similarities"].numpy()[:size],
            combos=tuple(
                combos[int(index)] if int(index) >= 0 else None
                for index in expected["input/gen_combo"][:size]
            ),
            analog_sid=np.array([1, 0, 1], np.int64),
            model_score=np.array(
                [0.3, 0.3, 0.2, 0.1, 0.9, 0, 0.8, 0.1, 0.3, -0.1],
                np.float32,
            )[:size],
            parent_smiles=tuple(parents),
            logs=tuple(logs) if log and size else (),
        )
        evidence = supports(context, proposals=proposals)
        actual[f"support/{case}/indices"] = torch.tensor(
            [
                [position, sid, combos.index(combo)]
                for position, items in enumerate(evidence)
                for sid, _, combo in items
            ],
            dtype=torch.int64,
        ).reshape(-1, 3)
        actual[f"support/{case}/similarities"] = torch.tensor(
            [sim for items in evidence for _, sim, _ in items],
            dtype=torch.float64,
        )
        actual[f"aggregate/{case}"] = torch.from_numpy(
            np.column_stack(
                list(
                    aggregate(
                        context,
                        priors=priors,
                        evidence=evidence,
                    ).values(),
                ),
            ),
        )
        actual[f"features/{case}"] = torch.from_numpy(
            np.column_stack(list(derivation_features(context, priors=priors).values())),
        )
    assert_tensor_golden(golden, float32_comparands(actual, inputs=("input/*",)))


def test_support_identity_routes_keep_generated_duplicates_and_known_invalid_keys() -> (
    None
):
    proposals = PreparedPaths(read_tensors(_BASELINE / "testdata/derivation.pt"))
    proposals.identities.update(
        fallback=("shared", "skeleton"),
        known_invalid=(None, None),
    )
    routes: tuple[
        tuple[
            Combo | None,
            Literal["pool", "gen"],
            Literal["smi", "key"],
            str | None,
        ],
        ...,
    ] = (
        (("a",), "pool", "smi", "pool"),
        (None, "pool", "smi", "pool"),
        (("a",), "gen", "smi", "duplicate"),
        (("b",), "pool", "key", "shared"),
        (("b",), "gen", "key", "shared"),
        (("a",), "gen", "smi", "fallback"),
        (("a",), "gen", "smi", "known_invalid"),
        (("a",), "gen", "key", None),
    )
    context = DerivationContext(
        smiles=("pool", "duplicate", "duplicate", "earlier", "invalid"),
        keys=("shared", "shared", "shared", "shared", None),
        formulas=("",) * 5,
        generated=np.array([False, True, True, True, True]),
        parent_sid=np.array([-1, -1, 2, 3, 4], np.int64),
        similarities=np.array([0, 0.2, 0.3, 0.4, 0.5]),
        combos=(None, ("a",), None, ("a",), ("b",)),
        analog_sid=np.zeros(0, np.int64),
        model_score=np.zeros(5),
        parent_smiles=(),
        logs=tuple(
            DerivationLog(
                parent=7,
                similarity=0.75,
                combo=combo,
                kind=kind,
                how=how,
                identity=identity,
            )
            for combo, kind, how, identity in routes
        ),
    )
    a, b = (7, 0.75, ("a",)), (7, 0.75, ("b",))
    assert supports(context, proposals=proposals) == [
        [a, b, a],
        [a, b, a],
        [a, b, a],
        [(3, 0.4, ("a",)), b, a],
        [(4, 0.5, ("b",)), a],
    ]
    with pytest.raises(ValueError, match="zip"):
        _ = supports(replace(context, keys=()), proposals=proposals)
    invalid = DerivationLog(
        parent=0,
        similarity=1.0,
        combo=("a",),
        kind="gen",
        how="smi",
        identity=None,
    )
    with pytest.raises(ValueError, match=r"^Expected log.identity is not None\.$"):
        _ = supports(replace(context, logs=(invalid,)), proposals=proposals)


def test_group_rejects_invalid_paths_preserves_empty_products_and_evicts_only_after5000() -> (
    None
):
    proposals = PreparedPaths(read_tensors(_BASELINE / "testdata/derivation.pt"))
    row: ProposalPath = {
        "raw": "",
        "standardized": "",
        "key": None,
        "skeleton": "",
        "environments": ((-1, -1, -1),),
    }
    serialization_failure: ProposalPath = {**row, "raw": None}
    standardization_failure: ProposalPath = {**row, "standardized": None}
    proposals.rows["", ("a",)] = (serialization_failure, standardization_failure, row)
    priors = DerivationPriors(
        {"glob": 0.2, "tf": {}, "env": {1: {}, 2: {}, 3: {}}},
        proposals=proposals,
    )
    assert tuple(priors.group("", ("a",))) == ("",)
    first = priors.group("", ("a",))
    for index in range(5_000):
        proposals.rows[str(index), ("a",)] = ()
        assert priors.group(str(index), ("a",)) == {}
    assert priors.group("", ("a",)) is first
    proposals.rows["last", ("a",)] = ()
    assert priors.group("last", ("a",)) == {}
    assert priors.group("", ("a",)) == first
    assert priors.group("", ("a",)) is not first


def test_prior_asset_pin_preserves_all_tables_and_metadata_without_runtime_objects(
    tmp_path: Path,
) -> None:
    proposals = PreparedPaths(read_tensors(_BASELINE / "testdata/derivation.pt"))
    tables: PriorTables = {
        "glob": 0.2,
        "tf": {"a": (3.0, 30.0)},
        "env": {1: {("a", 4): (2.0, 20.0)}, 2: {}, 3: {}},
        "meta": {"description": "complete tiny table", "rows": 2},
    }
    path = tmp_path / "prior.pkl"
    path.write_bytes(pickle.dumps(tables))
    seal = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = DerivationPriors.load(
        path,
        expected_sha256=seal,
        proposals=proposals,
    )
    assert loaded.tables == tables
    assert loaded.proposals is proposals
    assert loaded.p_site("a", (4, 5, 6)) == (
        2.0 + 2.0 * ((3.0 + 2.0 * 0.2) / (30.0 + 2.0))
    ) / (20.0 + 2.0)
    with pytest.raises(RuntimeError, match="hashes to"):
        _ = DerivationPriors.load(
            path,
            expected_sha256="invalid",
            proposals=proposals,
        )
    path.write_bytes(pickle.dumps(Path("forbidden-runtime-object")))
    with pytest.raises(pickle.UnpicklingError, match="plain data"):
        _ = DerivationPriors.load(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            proposals=proposals,
        )


def _derivation_strings(value: Tensor) -> list[str]:
    return value.numpy().tobytes().decode().split("\0") if value.numel() else []


def test_complete_paths_keep_invalids_multiplicity_zero_pairs_and_missing_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    header = {"format_version": 1, "rdkit": IDENTITY_RDKIT}
    _ = (tmp_path / "manifest.json").write_text(json.dumps(to_plain(header)))
    requests: list[dict[str, object]] = [
        {
            "parent_smiles": "parent",
            "combo": ["a", "b"],
            "max_first": 32,
            "max_second": 16,
            "max_mols": 24,
            "terminal_count": 2,
            "applications": "parent\ta\t32\t2",
        },
        {
            "parent_smiles": "empty",
            "combo": ["a"],
            "max_first": 32,
            "max_second": 16,
            "max_mols": 24,
            "terminal_count": 0,
            "applications": "empty\ta\t32\t0",
        },
    ]
    requests.append({**requests[0], "terminal_count": 0, "max_mols": 0})
    paths = [
        {
            "request": 0,
            "position": index,
            "raw": "raw" if index == 0 else None,
            "standardized": "product" if index == 0 else None,
            "key": "key" if index == 0 else None,
            "skeleton": "skeleton" if index == 0 else None,
            "environments": [[-1, 7, 7], [2, 3, 4]],
        }
        for index in range(2)
    ]
    identities = [
        {"smiles": "candidate", "key": "key", "skeleton": "skeleton"},
        {"smiles": "candidate", "key": "later", "skeleton": "later"},
        {"smiles": "invalid", "key": None, "skeleton": None},
    ]
    parquet.write_table(Table.from_pylist(requests), tmp_path / "requests.parquet")
    parquet.write_table(Table.from_pylist(paths), tmp_path / "paths.parquet")
    parquet.write_table(Table.from_pylist(identities), tmp_path / "identities.parquet")
    table = DerivationTable(tmp_path)
    assert table.paths("parent", ("a", "b")) == (
        {
            "request": 0,
            "position": 0,
            "raw": "raw",
            "standardized": "product",
            "key": "key",
            "skeleton": "skeleton",
            "environments": ((-1, 7, 7), (2, 3, 4)),
        },
        {
            "request": 0,
            "position": 1,
            "raw": None,
            "standardized": None,
            "key": None,
            "skeleton": None,
            "environments": ((-1, 7, 7), (2, 3, 4)),
        },
    )
    assert table.paths("empty", ("a",)) == ()
    assert table.identity("candidate") == ("key", "skeleton")
    assert table.identity("invalid") == (None, None)
    for call in (
        lambda: table.paths("unstaged", ("a",)),
        lambda: table.identity("unstaged"),
    ):
        with pytest.raises(KeyError, match="unstaged"):
            _ = call()
    for altered in (
        [{**requests[0], "terminal_count": 1}, requests[1]],
        [requests[1]],
    ):
        parquet.write_table(Table.from_pylist(altered), tmp_path / "requests.parquet")
        with pytest.raises(
            ValueError,
            match=r"^Derivation terminal path layout differs from its headers\.$",
        ):
            _ = DerivationTable(tmp_path)
    parquet.write_table(Table.from_pylist(requests), tmp_path / "requests.parquet")
    parquet.write_table(
        Table.from_pylist([*paths, {**paths[0], "request": 99}]),
        tmp_path / "paths.parquet",
    )
    with pytest.raises(
        ValueError,
        match=r"^Derivation terminal path layout differs from its headers\.$",
    ):
        _ = DerivationTable(tmp_path)
    parquet.write_table(
        Table.from_pylist([{**row, "position": 0} for row in paths]),
        tmp_path / "paths.parquet",
    )
    with pytest.raises(
        ValueError,
        match=r"^Derivation terminal path layout differs from its headers\.$",
    ):
        _ = DerivationTable(tmp_path)
    for altered_header in (
        {"format_version": 2, "rdkit": IDENTITY_RDKIT},
        {"format_version": 1, "rdkit": SIMULATOR_RDKIT},
    ):
        _ = (tmp_path / "manifest.json").write_text(
            json.dumps(to_plain(altered_header)),
        )
        with pytest.raises(
            ValueError,
            match=r"^Derivation data requires format 1 and RDKit 2026\.03\.3\.$",
        ):
            _ = DerivationTable(tmp_path)
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": True, "rdkit": IDENTITY_RDKIT})),
    )
    with pytest.raises(ValueError, match="cannot read True as int"):
        _ = DerivationTable(tmp_path)


def test_complete_main_public_pipeline_all_families_and_four_rankers(
    tmp_path: Path,
) -> None:
    assert_main_pipeline(tmp_path)


def test_selected_features_preserve_duplicates_order_dtype_and_empty_rows() -> None:
    values = {
        "base": np.array([3, 7, 3], np.float32),
        "family__value": np.array([11, 13, 11], np.float32),
    }
    result = select_features(values, names=("family__value", "base"), count=3)
    np.testing.assert_array_equal(result, [[11, 3], [13, 7], [11, 3]])
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    empty = select_features({"x": np.zeros(0, np.float32)}, names=("x",), count=0)
    assert empty.shape == (0, 1)
    no_columns = select_features({}, names=(), count=2)
    assert no_columns.shape == (2, 0)
    assert no_columns.dtype == np.float32
    with pytest.raises(ValueError, match=r"missing.*absent"):
        select_features(values, names=("absent",), count=3)
    with pytest.raises(
        ValueError,
        match=r"^Feature vectors differ from the candidate positions\.$",
    ):
        select_features(values, names=("base",), count=2)
    with pytest.raises(ValueError, match=r"^Every feature must be one-dimensional\.$"):
        # select_features must reject a column matrix in place of a flat vector.
        select_features({"x": np.zeros((3, 1), np.float32)}, names=("x",), count=3)


def analog_record(analogs: Sequence[Analog]) -> Tensor:
    """Retain every source tuple field with its original double conversion."""
    return torch.tensor(
        [[a.sid, a.similarity, a.shift, a.natural_product] for a in analogs],
        dtype=torch.float64,
    ).reshape(-1, 4)


def test_whole_main_engine_retrieval() -> None:
    library, pool, spectra = retrieval_inputs()
    actual: dict[str, Tensor] = {}
    for name, count in (("ties", 37), ("empty", 0)):
        metadata = replace(
            library,
            sid=np.arange(count, dtype=np.int64)[::-1],
            mode=np.array([-1, 0, 1], dtype=np.int8)[np.arange(count) % 3],
            library=(np.arange(count) % 11).astype(np.int8),
            peak_count=(np.arange(count) % 4 + 2).astype(np.int32),
            mass=np.full(count, 100.0, dtype=np.float64),
        )
        actual[f"main/{name}"] = torch.from_numpy(
            MainRetrieval.Config().make().representatives(metadata),
        )
    partition_library = replace(library, library=library.library.copy())
    partition_library.library[0] = 1
    config_partition = MainRetrieval.Config()
    config_partition.n_analog = 2
    partition = config_partition.make()
    representatives = partition.representatives(partition_library)
    assert (
        len(representatives[partition_library.mode[representatives] == 1])
        > 4 * config_partition.n_analog
    )
    actual["main/partition"] = analog_record(
        partition.analogs(
            spectra,
            library=partition_library,
            target=100.0,
        ),
    )
    for merged in (True, False):
        config = MainRetrieval.Config()
        config.n_analog = 2
        if not merged:
            config.merge = None
        search = config.make()
        actual[f"main/{merged}/representatives"] = torch.from_numpy(
            search.representatives(library),
        )
        excluded = np.isin(np.arange(13), [1, 5])
        for case, (query, target, mask, sid, lib) in enumerate(
            (
                (spectra, 100.0, None, -1, -1),
                (spectra, 100.0, excluded, 0, 2),
                (spectra, 100.0, None, 0, -1),
                (spectra, 900.0, None, -1, -1),
                (spectra[2:], 100.0, None, -1, -1),
            ),
        ):
            actual[f"main/{merged}/{case}"] = analog_record(
                search.analogs(
                    query,
                    library=library,
                    target=target,
                    excluded=mask,
                    excluded_sid=sid,
                    excluded_library=lib,
                ),
            )
    generated = [
        GeneratedCandidate(
            smiles="C",
            key="a",
            formula="CH4",
            mass=100.0,
            heavy_atoms=1,
            fingerprint=np.arange(7, dtype=np.float32) % 2,
            fragments=np.array([12, 13, 14], dtype=np.float64),
            parent_similarity=0.75,
            steps=1,
            parent_tanimoto=0.5,
        ),
        GeneratedCandidate(
            smiles="CC",
            key="new",
            formula="C2H6",
            mass=100.001,
            heavy_atoms=2,
            fingerprint=np.equal(np.arange(7) % 3, 0).astype(np.float32),
            fragments=np.array([22, 23], dtype=np.float64),
            parent_similarity=0.5,
            steps=2,
            parent_tanimoto=0.25,
        ),
    ]
    for case, (target, drop, products) in enumerate(
        (
            (100.0, -1, generated),
            (100.004, -1, list[GeneratedCandidate]()),
            (100.0, 0, list[GeneratedCandidate]()),
            (900.0, -1, list[GeneratedCandidate]()),
            (900.0, -1, generated),
        ),
    ):
        rows = candidates(
            pool,
            target=target,
            ppm=10.0,
            fallback_ppm=30.0,
            generated=products,
            drop_row=drop,
        )
        identities = rows.candidates
        if case == 0:
            assert identities.smiles == ("C", "CC", "C", "C", "CC")
            assert identities.keys == ("a", "b", "a", "a", "new")
            assert identities.formulas == ("CH4", "C2H6", "CH4", "CH4", "C2H6")
            assert identities.sources == (
                "pool",
                "pool",
                "pool",
                "generated",
                "generated",
            )
        for name, values in {
            "pid": identities.pool_rows,
            "mass": rows.mass,
            "n_heavy": rows.heavy_atoms,
            "train_sid": rows.train_sid,
            "fp": rows.fingerprints,
            "is_gen": identities.generated.astype(np.float32),
            "gen_sim": rows.parent_similarity,
            "gen_steps": rows.steps,
            "gen_tan": rows.parent_tanimoto,
        }.items():
            actual[f"candidates/{case}/{name}"] = torch.from_numpy(values)
        for position, fragments in enumerate(rows.fragments):
            if fragments is not None:
                actual[f"candidates/{case}/fragments/{position}"] = torch.from_numpy(
                    fragments,
                )
    assert_tensor_golden(
        _THIS.parent / "testdata" / "retrieval.pt",
        float32_comparands(actual),
    )


def test_candidate_metadata_retains_each_admitted_parent_and_path() -> None:
    _, pool, _ = retrieval_inputs()
    product = GeneratedCandidate(
        smiles="CCOC",
        key="generated",
        formula="C3H8O",
        mass=100.0,
        heavy_atoms=4,
        fingerprint=np.zeros(7, np.float32),
        fragments=np.array([15.0, 30.0]),
        parent_similarity=0.75,
        steps=1,
        parent_tanimoto=0.5,
        parent_sid=7,
        combo=("methyl_O",),
        raw="COCC",
    )
    rows = candidates(
        pool,
        target=100.0,
        ppm=10.0,
        fallback_ppm=30.0,
        generated=(product, product),
    )
    np.testing.assert_array_equal(rows.parent_sid, [-1, -1, -1, 7, 7])
    assert rows.combos == (None, None, None, ("methyl_O",), ("methyl_O",))
    assert rows.raw == (None, None, None, "COCC", "COCC")
    empty = candidates(pool, target=900.0, ppm=10.0, fallback_ppm=30.0)
    assert empty.parent_sid.shape == (0,)
    assert empty.parent_sid.dtype == np.int64
    assert empty.combos == empty.raw == ()


ENGINE_GOLDEN: Final = _BASELINE / "testdata" / "engine.pt"


def test_whole_main94_and_fragment_evidence(tmp_path: Path) -> None:
    golden = ENGINE_GOLDEN
    actual: dict[str, Tensor] = {}
    library, pool, spectra = retrieval_inputs()
    for item, energy in zip(spectra, (10.0, None, float("nan")), strict=True):
        item["ce"] = energy
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        path = tmp_path / "full1.pt"
        torch.save(
            source_checkpoint(
                read_tensors(ENGINE_GOLDEN.with_name("neural_bank.pt")),
                index=0,
            ),
            path,
        )
        config = MainBank.Config()
        config.checkpoints = [checkpoint_config(path, index=0)]
        config.merged_adduct = modal_adduct
        config.single_count = keep_count
        bank = config.make()
        for case, (query, target, neural, enabled, cap, excluded) in enumerate(
            (
                (spectra, 100.0, True, True, 2, None),
                (
                    spectra,
                    100.0,
                    False,
                    True,
                    600,
                    np.isin(np.arange(13), [0, 1, 2, 5]),
                ),
                (list[SpectrumRequest](), 100.0, False, False, 600, None),
                (spectra[:1], 130.0, True, True, 600, None),
                (spectra, 900.0, True, True, 600, None),
            ),
        ):
            engine = MainEngine.Config()
            assert engine.fragments is not None
            engine.fragments.max_candidates = cap
            if not enabled:
                engine.fragments = engine.subformulas = None
            result = engine.make().base(
                query,
                target=target,
                library=library,
                pool=pool,
                train_fingerprints=(np.arange(9)[:, None] * 17 % 251).astype(np.uint8),
                bank=bank if neural else None,
                excluded=excluded,
                merged_counts=[0, 12, 2][: len(query)],
            )
            rows = result.rows
            actual[f"base/{case}/features"] = torch.from_numpy(result.features)
            for name, value in (
                ("pid", rows.candidates.pool_rows),
                ("mass", rows.mass),
                ("n_heavy", rows.heavy_atoms),
                ("train_sid", rows.train_sid),
                ("fp", rows.fingerprints),
                ("is_gen", rows.candidates.generated.astype(np.float32)),
                ("gen_sim", rows.parent_similarity),
                ("gen_steps", rows.steps),
                ("gen_tan", rows.parent_tanimoto),
            ):
                actual[f"base/{case}/{name}"] = torch.from_numpy(value)
            diagnostic = result.diagnostics
            actual[f"base/{case}/info"] = torch.tensor(
                [
                    diagnostic.n_cand,
                    diagnostic.lib_max,
                    diagnostic.top_sim,
                    diagnostic.fzmax,
                    diagnostic.n_gen,
                ],
                dtype=torch.float64,
            )
            actual[f"base/{case}/analogs"] = analog_record(result.analogs)
        peaks = [
            (
                np.arange(12, dtype=np.float64) * 10 + 11.00727645217,
                np.arange(12, dtype=np.float64) / 12 + 0.1,
                "[M+H]+",
            ),
            (
                np.array([9001.00727645217, 11.00727645217]),
                np.array([1.0, 0.5]),
                "[M+Na]+",
            ),
        ]
        masses = np.array([10.0, 20.0, 30.0 - 2 * 1.00782503207, 9000.0], np.float64)
        for index, label in enumerate(
            (
                "[M+H]+",
                "[M+NH4]+",
                "[M+Na]+",
                "[M+K]+",
                "[M-H]-",
                "[M+Cl]-",
                "[M+CH2O2-H]-",
                "[M+HCOO]-",
                "unknown+",
                "unknown",
            ),
        ):
            actual[f"deltas/{index}"] = torch.tensor(
                main_ion_deltas(label),
                dtype=torch.float64,
            )
            actual[f"fragments/{index}"] = torch.from_numpy(
                main_fragment_features(
                    masses,
                    peaks=[(mass, value, label) for mass, value, _ in peaks],
                ),
            )
        for index, formula in enumerate(
            ("C2H6O", "CH2NOPS-FClBrI+", "C0H2C", "NaCl", "C.2H", ""),
        ):
            vector = engine_formula_vector(formula)
            actual[f"formula/{index}/valid"] = torch.tensor(vector is not None)
            actual[f"formula/{index}/counts"] = torch.from_numpy(
                vector if vector is not None else np.zeros(0, np.int64),
            )
            actual[f"formula/{index}/features"] = torch.from_numpy(
                main_subformula_features(formula, peaks=peaks, max_out=2_000_000),
            )
            if vector is not None:
                for maximum in (30, 2_000_000):
                    actual[f"formula/{index}/masses/{maximum}"] = torch.from_numpy(
                        main_subformula_masses(vector, max_out=maximum),
                    )
        actual["fragments/empty"] = torch.from_numpy(
            main_fragment_features(np.zeros(0), peaks=peaks),
        )
        actual["fragments/no_peaks"] = torch.from_numpy(
            main_fragment_features(masses, peaks=[]),
        )
    assert_tensor_golden(golden, float32_comparands(actual))


def test_main_engine_empty_candidates_preserve_all94_columns() -> None:
    library, pool, spectra = retrieval_inputs()
    evidence = (
        MainEngine.Config()
        .make()
        .base(
            spectra,
            target=900.0,
            library=library,
            pool=pool,
            # MainEngine.base consumes seven selected bits packed into one byte.
            train_fingerprints=np.zeros((9, 1), np.uint8),
        )
    )
    assert evidence.features.shape == (0, 94)
    assert len(evidence.rows.candidates.smiles) == 0


def test_library_aggregation_skips_empty_queries_but_retains_query_denominator() -> (
    None
):
    library, pool, _ = retrieval_inputs()
    library = replace(
        library,
        offsets=np.array([0, 2], np.int64),
        mz=np.array([20, 30], np.float32),
        intensity=np.array([1, 1], np.float32),
        sid=np.array([0], np.int64),
        library=np.array([2], np.int8),
        mode=np.array([1], np.int8),
        adduct=np.array([0], np.int16),
        precursor=np.array([101.0]),
        peak_count=np.array([2], np.int32),
        mass=np.array([100.0]),
        order=np.array([0], np.int64),
        mass_sorted=np.array([100.0]),
    )
    rows = candidates(pool, target=100.0, ppm=10.0, fallback_ppm=30.0)
    position = np.flatnonzero(np.equal(rows.train_sid, 0)).item()
    search = MainSearch.Config().make()
    store = PeakStore(
        offsets=library.offsets,
        mz=library.mz,
        intensity=library.intensity,
    )

    def request(
        mass: list[float],
        intensity: list[float],
        mode: int | str,
    ) -> SpectrumRequest:
        return {
            "mz": mass,
            "it": intensity,
            "prec": 101.0,
            "mode": mode,
            "adduct": "[M+H]+",
        }

    queries = [
        request([], [], 1),
        request([20, 30], [1, 1], -1),
        request([20, 30], [1, 1], "positive"),
        request([20], [1], 1),
        request([30], [1], 1),
        request([20, 30], [2, 1], 1),
        request([70, 80], [1, 1], 1),
    ]
    scores = np.array(
        [
            search.score(
                (np.array(query["mz"]), np.array(query["it"])),
                store=store,
                candidates=np.array([0], np.int64),
            ).item(0)
            for query in queries[2:6]
        ],
    )

    def evidence(query: list[SpectrumRequest]) -> MainLibraryEvidence:
        return _main_library(
            query,
            rows=rows,
            library=library,
            target=100.0,
            search=search,
            ppm=10.0,
            excluded=None,
        )

    combined = evidence(queries)
    assert combined.mean.item(position) == sum(
        scores.item(i) for i in range(len(scores))
    ) / len(queries)
    assert combined.top3.item(position) == np.mean(np.sort(scores)[-3:])
    positive = evidence(queries[2:3])
    assert positive.mean.item(position) == scores.item(0)
    empty_reference = replace(
        library,
        offsets=np.array([0, 2, 2], np.int64),
        sid=np.array([0, 0], np.int64),
        library=np.array([2, 2], np.int8),
        mode=np.array([1, -1], np.int8),
        adduct=np.array([0, 6], np.int16),
        mass=np.array([100.0, 100.0]),
        order=np.array([0, 1], np.int64),
        mass_sorted=np.array([100.0, 100.0]),
    )
    zero = _main_library(
        [queries[2], queries[1]],
        rows=rows,
        library=empty_reference,
        target=100.0,
        search=search,
        ppm=10.0,
        excluded=None,
    )
    assert zero.top3.item(position) == scores.item(0)
    assert zero.mean.item(position) == scores.item(0) / 2


def test_complete_depth_pair_reader_preserves_duplicates_and_invalid_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    _fragments_write(tmp_path)
    table = MainFragmentTable(tmp_path)
    rows: tuple[tuple[int, tuple[list[int], list[int]]], ...] = (
        (0, ([1, 3], [3])),
        (1, ([], [])),
        (2, ([4], [1, 4])),
    )
    for row, expected in rows:
        for actual, values in zip(table.row(row), expected, strict=True):
            np.testing.assert_array_equal(actual, values)
            assert actual.dtype == np.float32
            assert not actual.flags.writeable
    for actual, values in zip(table.for_smiles("same"), ([1, 3], [3]), strict=True):
        np.testing.assert_array_equal(actual, values)
    assert all(len(row) == 0 for row in table.for_smiles("invalid"))
    for index in (-1, 3):
        with pytest.raises(IndexError) as caught:
            _ = table.row(index)
        assert caught.value.args == (index,)
    with pytest.raises(KeyError, match="absent"):
        _ = table.for_smiles("absent")


def test_depth_pair_reader_rejects_wrong_types_offsets_and_provenance(
    tmp_path: Path,
) -> None:
    _fragments_write(tmp_path)
    for name, malformed, error in (
        ("frag_mass", np.array([1, 3, 4], dtype=np.float64), TypeError),
        # MainFragmentTable requires a flat mass vector, not a column matrix.
        ("f1_mass", np.zeros((3, 1), dtype=np.float32), ValueError),
        ("frag_off", np.array([0, 2, 2], dtype=np.int64), ValueError),
        ("frag_off", np.array([1, 2, 2, 3], dtype=np.int64), ValueError),
        ("frag_off", np.array([0, 2, 2, 4], dtype=np.int64), ValueError),
        ("f1_off", np.array([0, 2, 1, 3], dtype=np.int64), ValueError),
        ("f1_off", np.array([0, 1, 1, 3], dtype=np.int32), TypeError),
    ):
        path = tmp_path / f"{name}.npy"
        correct = cast("NDArray[np.generic]", np.load(path, allow_pickle=False))
        np.save(path, malformed, allow_pickle=False)
        with pytest.raises(error, match=r"Main fragment|expected"):
            _ = MainFragmentTable(tmp_path)
        np.save(path, correct, allow_pickle=False)
    for version, release in ((2, IDENTITY_RDKIT), (1, SIMULATOR_RDKIT)):
        _ = (tmp_path / "manifest.json").write_text(
            json.dumps(to_plain({"format_version": version, "rdkit": release})),
        )
        with pytest.raises(ValueError, match="Main fragment"):
            _ = MainFragmentTable(tmp_path)
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": True, "rdkit": IDENTITY_RDKIT})),
    )
    with pytest.raises(ReadError):
        _ = MainFragmentTable(tmp_path)


def test_main_base_rows_are_distinct_readonly_float64_and_missing_data_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    _fragments_write(tmp_path)
    values = np.array([1.000001, 3.000001, 4.000001], dtype=np.float64)
    np.save(tmp_path / "base_mass.npy", values, allow_pickle=False)
    table = MainFragmentTable(tmp_path)
    row = table.base_for_smiles("same")
    np.testing.assert_array_equal(row, values[:2])
    assert row.dtype == np.float64
    assert not row.flags.writeable
    assert row[0] != table.for_smiles("same")[0][0]
    assert len(table.base_for_smiles("invalid")) == 0
    with pytest.raises(KeyError, match="absent"):
        _ = table.base_for_smiles("absent")
    for wrong, error in (
        (values.astype(np.float32), TypeError),
        (values[:2], ValueError),
    ):
        np.save(tmp_path / "base_mass.npy", wrong, allow_pickle=False)
        with pytest.raises(error, match=r"Main fragment|expected"):
            _ = MainFragmentTable(tmp_path).base_for_smiles("same")
    (tmp_path / "base_mass.npy").unlink()
    with pytest.raises(FileNotFoundError):
        _ = MainFragmentTable(tmp_path).base_for_smiles("same")


def _fragments_write(directory: Path) -> None:
    parquet.write_table(
        Table.from_pydict({"smiles": ["same", "invalid", "same"]}),
        directory / "molecules.parquet",
    )
    for name, values in (
        ("frag_off", np.array([0, 2, 2, 3], dtype=np.int64)),
        ("frag_mass", np.array([1, 3, 4], dtype=np.float32)),
        ("f1_off", np.array([0, 1, 1, 3], dtype=np.int64)),
        ("f1_mass", np.array([3, 1, 4], dtype=np.float32)),
    ):
        np.save(directory / f"{name}.npy", values, allow_pickle=False)
    _ = (directory / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": IDENTITY_RDKIT})),
    )


def test_complete_family_positions_keep_pool_duplicates_generated_aliases_and_sids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    data, rows = _inputs(tmp_path)
    provider: MainFamilyProvider = data
    result = provider.select(
        rows,
        (Analog(sid=2, similarity=0.9), Analog(sid=0, similarity=0.8)),
    )
    assert [row.hashes.item(0) for row in result.candidate_structures] == [
        1,
        2,
        3,
        7,
        8,
    ]
    assert [row.hashes.item(0) for row in result.analog_structures] == [6, 4]
    assert result.train_smiles == ("parent0", "parent1", "parent0")
    for actual, expected in zip(
        result.depth1,
        ([10, 11], [12, 13], [14, 15], [70, 71], list[int]()),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float64
    for position, actual in enumerate(result.depth2):
        expected = (
            data.pool.fragments(position)
            if position < 3
            else data.generated_fragments.row(position - 3)[0].astype(np.float64)
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float64
    np.testing.assert_array_equal(
        result.capabilities,
        np.repeat(np.array([1, 2, 3, 7, 0], np.uint8)[:, None], 22, axis=1),
    )
    assert result.capabilities.dtype == np.uint8
    np.testing.assert_array_equal(
        result.candidate_structures[3].formula,
        family_data_formula_vector("C2H6O"),
    )
    assert result.fragnet[-1] is None
    assert [row["mass"] for row in result.fragnet[:-1] if row is not None] == [2] * 4
    assert result.priors is data.priors
    np.testing.assert_array_equal(rows.parent_sid, [-1, -1, -1, 2, 1])
    assert rows.combos == (None, None, None, ("step0",), ("step1",))
    assert rows.raw == (None, None, None, "raw0", "raw1")
    empty = provider.select(
        candidates(data.pool, target=900, ppm=10.0, fallback_ppm=30.0),
        (),
    )
    assert empty.candidate_structures == empty.analog_structures == ()
    assert empty.depth1 == empty.depth2 == empty.fragnet == ()
    assert empty.capabilities.shape == (0, 22)
    assert empty.train_smiles == result.train_smiles


def test_unstaged_or_misaligned_family_evidence_is_an_error(tmp_path: Path) -> None:
    data, rows = _inputs(tmp_path)
    for sid in (-1, 3):
        with pytest.raises(IndexError) as error:
            _ = data.select(rows, (Analog(sid=sid, similarity=0.5),))
        assert error.value.args == (sid,)
    with pytest.raises(ValueError, match="training SID"):
        _ = replace(data, train_smiles=("parent1", "parent0", "parent0"))
    with pytest.raises(ValueError, match="pool positions"):
        _ = replace(data, pool=replace(data.pool, smiles=("wrong", "C", "CCO")))
    with pytest.raises(ValueError, match="capability vocabulary"):
        _ = replace(data, pool_families=replace(data.pool_families, names=("wrong",)))
    with pytest.raises(ValueError, match="capability vocabulary"):
        _ = replace(
            data,
            pool_families=replace(
                data.pool_families,
                names=data.pool_families.names[::-1],
            ),
        )
    with pytest.raises(ValueError, match="pool positions"):
        _ = replace(
            data,
            pool_families=replace(
                data.pool_families,
                offsets=np.array([0, 2, 4], np.int64),
                mass=data.pool_families.mass[:4],
                capabilities=data.pool_families.capabilities[:2],
            ),
        )
    changed = replace(
        rows,
        candidates=replace(
            rows.candidates,
            smiles=("wrong", *rows.candidates.smiles[1:]),
        ),
    )
    with pytest.raises(ValueError, match="pool identity"):
        _ = data.select(changed, ())
    changed = replace(
        rows,
        candidates=replace(
            rows.candidates,
            smiles=(*rows.candidates.smiles[:-1], "missing"),
        ),
    )
    with pytest.raises(KeyError, match="missing"):
        _ = data.select(changed, ())
    with pytest.raises(ValueError, match="generated positions"):
        _ = data.select(
            replace(
                rows,
                candidates=replace(rows.candidates, generated=np.zeros(5, bool)),
            ),
            (),
        )
    with pytest.raises(IndexError) as error:
        _ = data.select(replace(rows, parent_sid=np.array([-1, -1, -1, 3, 1])), ())
    assert error.value.args == (3,)
    with pytest.raises(KeyError, match="unstaged"):
        _ = data.select(replace(rows, raw=(*rows.raw[:-1], "unstaged")), ())
    with pytest.raises(KeyError, match="unstaged"):
        _ = data.select(replace(rows, combos=(*rows.combos[:-1], ("unstaged",))), ())
    with pytest.raises(ValueError, match="candidate positions"):
        _ = data.select(replace(rows, raw=()), ())
    with pytest.raises(IndexError) as error:
        _ = data.select(
            replace(
                rows,
                candidates=replace(
                    rows.candidates,
                    pool_rows=np.array([9, 1, 2, -1, -1]),
                ),
            ),
            (),
        )
    assert error.value.args == (9,)


def test_training_sid_reader_reorders_rows_and_rejects_gaps_or_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "structures.parquet"
    parquet.write_table(
        Table.from_pydict({"sid": [2, 0, 1], "smiles": ["alias", "alias", "other"]}),
        path,
    )
    assert training_smiles(path) == ("alias", "other", "alias")
    for indices in ([0, 0, 2], [0, 1, 3], [-1, 0, 1]):
        parquet.write_table(
            Table.from_pydict({"sid": indices, "smiles": ["a", "b", "c"]}),
            path,
        )
        with pytest.raises(
            ValueError,
            match=r"^Expected one structure per contiguous training SID\.$",
        ) as error:
            _ = training_smiles(path)
        assert error.value.args == (
            "Expected one structure per contiguous training SID.",
        )
    parquet.write_table(Table.from_pydict({"sid": [], "smiles": []}), path)
    assert training_smiles(path) == ()


def test_provided_pool_family_reader_checks_every_shape_dtype_and_offset(
    tmp_path: Path,
) -> None:
    offsets = np.array([0, 2, 2, 5], np.int64)
    mass = np.array([1, 2, 3, 4, 5], np.float32)
    capabilities = np.arange(66, dtype=np.uint8).reshape(3, 22)
    names = tuple(f"cap{index}" for index in range(22))
    for filename, array in (
        ("pool_f1_off.npy", offsets),
        ("pool_f1_mass.npy", mass),
        ("pool_cap.npy", capabilities),
    ):
        np.save(tmp_path / filename, array)
    _ = (tmp_path / "caps.txt").write_text("\n".join(names))
    result = MainPoolFamilies.load(tmp_path, count=3)
    np.testing.assert_array_equal(result.offsets, offsets)
    np.testing.assert_array_equal(result.mass, mass)
    np.testing.assert_array_equal(result.capabilities, capabilities)
    assert result.names == names
    assert not result.mass.flags.writeable
    with pytest.raises(ValueError, match="pool family count"):
        _ = MainPoolFamilies.load(tmp_path, count=2)
    for filename, wrong, error, message in (
        ("pool_f1_off.npy", np.array([0, 2, 1, 5], np.int64), ValueError, "pool"),
        ("pool_f1_off.npy", offsets.astype(np.int32), TypeError, "expected int64"),
        ("pool_f1_off.npy", offsets[:-1], ValueError, "pool"),
        ("pool_f1_off.npy", offsets + 1, ValueError, "pool"),
        ("pool_f1_mass.npy", mass.astype(np.float64), TypeError, "expected float32"),
        ("pool_f1_mass.npy", mass[None], ValueError, "pool"),
        ("pool_cap.npy", capabilities.astype(np.int16), TypeError, "expected uint8"),
        ("pool_cap.npy", capabilities[:-1], ValueError, "pool"),
        ("pool_cap.npy", capabilities[:, :-1], ValueError, "pool"),
    ):
        original = (tmp_path / filename).read_bytes()
        np.save(tmp_path / filename, wrong)
        with pytest.raises(error, match=message):
            _ = MainPoolFamilies.load(tmp_path, count=3)
        _ = (tmp_path / filename).write_bytes(original)


def _inputs(tmp_path: Path) -> tuple[MainFamilyData, CandidateRows]:
    pool = CandidatePool(
        mass=np.full(3, 100.0),
        keys=("a", "b", "a"),
        smiles=("CCO", "C", "CCO"),
        formulas=("C2H6O", "CH4", "C2H6O"),
        heavy_atoms=np.array([3, 1, 3], np.int32),
        source=np.zeros(3, np.int8),
        train_sid=np.array([0, 1, 2], np.int64),
        packed=np.array([[32], [64], [32]], np.uint8),
        bits=np.arange(3, dtype=np.int64),
        fragment_offsets=np.arange(4, dtype=np.int64) * 2,
        fragment_mass=np.arange(6, dtype=np.float32) + np.float32(0.123),
        key_to_pid={"a": 2, "b": 1},
    )
    names = tuple(f"cap{index}" for index in range(22))
    fragment_dir = tmp_path / "fragments"
    fragment_dir.mkdir()
    _header(fragment_dir)
    parquet.write_table(
        Table.from_pydict({"smiles": ["CCO", "bad"]}),
        fragment_dir / "molecules.parquet",
    )
    for name, array in (
        ("frag_off", np.array([0, 2, 2], np.int64)),
        ("f1_off", np.array([0, 2, 2], np.int64)),
        ("frag_mass", np.array([70.123, 71.123], np.float32)),
        ("f1_mass", np.array([70, 71], np.float32)),
    ):
        np.save(fragment_dir / f"{name}.npy", array)
    generated_fragments = MainFragmentTable(fragment_dir)
    intrinsic_dir = tmp_path / "intrinsic"
    intrinsic_dir.mkdir()
    parquet.write_table(
        Table.from_pydict({"smiles": ["CCO", "bad"]}),
        intrinsic_dir / "molecules.parquet",
    )
    np.save(intrinsic_dir / "records.npy", np.zeros((2, 34), np.float32))
    np.save(
        intrinsic_dir / "capabilities.npy",
        np.repeat(np.array([7, 0], np.uint8)[:, None], 22, axis=1),
    )
    _ = (intrinsic_dir / "record_names.txt").write_text("\n".join(record_fields()))
    _ = (intrinsic_dir / "capability_names.txt").write_text("\n".join(names))
    data = MainFamilyData(
        pool=pool,
        train_smiles=("parent0", "parent1", "parent0"),
        pool_structures=_analog(
            tmp_path / "pool",
            pool.smiles,
            pool.formulas,
            (1, 2, 3),
        ),
        train_structures=_analog(
            tmp_path / "train",
            ("parent0", "parent1", "parent0"),
            (None,) * 3,
            (4, 5, 6),
        ),
        generated_structures=_analog(
            tmp_path / "generated",
            ("CCO", "bad"),
            (None, None),
            (7, 8),
        ),
        pool_families=MainPoolFamilies(
            offsets=np.arange(4, dtype=np.int64) * 2,
            mass=np.arange(6, dtype=np.float32) + 10,
            capabilities=np.repeat(np.array([1, 2, 3], np.uint8)[:, None], 22, axis=1),
            names=names,
        ),
        generated_fragments=generated_fragments,
        generated_intrinsic=IntrinsicTable.load(intrinsic_dir),
        fragnet=_fragnet(tmp_path / "fragnet"),
        priors=DerivationPriors(
            {"glob": 0.2, "tf": {}, "env": {1: {}, 2: {}, 3: {}}},
            proposals=_FamilyDataProposals(),
        ),
    )
    generated = tuple(
        GeneratedCandidate(
            smiles=smiles,
            key="a" if index == 0 else "invalid",
            formula="C2H6O" if index == 0 else "",
            mass=100,
            heavy_atoms=3,
            fingerprint=np.zeros(3, np.float32),
            fragments=generated_fragments.row(index)[0].astype(np.float64),
            parent_similarity=0.7,
            steps=1,
            parent_tanimoto=0.5,
            parent_sid=2 - index,
            combo=(f"step{index}",),
            raw=f"raw{index}",
        )
        for index, smiles in enumerate(("CCO", "bad"))
    )
    return data, candidates(
        pool,
        target=100,
        ppm=10.0,
        fallback_ppm=30.0,
        generated=generated,
    )


def _analog(
    directory: Path,
    smiles: tuple[str, ...],
    formulas: tuple[str | None, ...],
    markers: tuple[int, ...],
) -> AnalogTable:
    directory.mkdir()
    _header(directory)
    parquet.write_table(
        Table.from_pylist(
            [
                {
                    "smiles": value,
                    "formula_input": formula,
                    "core_smiles": "",
                    "element_smiles": "",
                    "query_smiles": "",
                }
                for value, formula in zip(smiles, formulas, strict=True)
            ],
        ),
        directory / "molecules.parquet",
    )
    for name, array in (
        ("hashes", np.repeat(np.array(markers, np.int64)[:, None], 8, axis=1)),
        ("numbers", np.zeros((len(smiles), 8), np.int16)),
        (
            "formula",
            np.stack([family_data_formula_vector(value) for value in formulas]),
        ),
        ("core_mask", np.zeros(0, bool)),
    ):
        np.save(directory / f"{name}.npy", array)
    for prefix in ("graph", "core"):
        for name, array in (
            ("elements", np.zeros(0, np.int16)),
            ("bonds", np.zeros((0, 2), np.int32)),
            ("atom_off", np.zeros(len(smiles) + 1, np.int64)),
            ("bond_off", np.zeros(len(smiles) + 1, np.int64)),
        ):
            np.save(directory / f"{prefix}_{name}.npy", array)
    return AnalogTable(directory)


def _fragnet(directory: Path) -> FragnetTable:
    directory.mkdir()
    _header(directory)
    parquet.write_table(
        Table.from_pydict(
            {
                "smiles": ["CCO", "C", "bad"],
                "parsed": [True, True, False],
                "valid": [True, True, False],
            },
        ),
        directory / "molecules.parquet",
    )
    for name, array in (
        ("amass", np.ones(4, np.float64)),
        ("mass", np.array([2, 2, 0], np.float64)),
        ("atom_off", np.array([0, 2, 4, 4], np.int64)),
        ("bond_off", np.array([0, 1, 2, 2], np.int64)),
        ("bonds", np.array([[0, 1], [0, 1]], np.int32)),
        ("bring", np.zeros(2, np.int8)),
        *((name, np.zeros(4, np.int8)) for name in ("kind", "el", "nh")),
        ("el10", np.zeros(4, np.uint8)),
    ):
        np.save(directory / f"{name}.npy", array)
    return FragnetTable(directory)


def _header(directory: Path) -> None:
    _ = (directory / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": IDENTITY_RDKIT})),
    )


class _FamilyDataProposals:
    def paths(
        self,
        parent_smiles: str,
        combo: tuple[str, ...],
    ) -> tuple[ProposalPath, ...]:
        paths: dict[tuple[str, tuple[str, ...]], tuple[ProposalPath, ...]] = {
            ("parent0", ("step0",)): (),
            ("parent1", ("step1",)): (),
        }
        return paths[parent_smiles, combo]

    def identity(self, smiles: str) -> tuple[str | None, str | None]:
        if smiles not in ("CCO", "C", "bad", "raw0", "raw1"):
            raise KeyError(smiles)
        return (None, None) if smiles == "bad" else ("key", "skeleton")


class _Request(TypedDict):
    smiles: str
    delta: float
    tol: float
    max_steps: int
    max_per_analog: int
    allow_isomerisation: bool
    products: list[tuple[str, tuple[str, ...] | None]]


class _Inputs(TypedDict):
    requests: list[_Request]
    standardized: dict[str, str | None]
    keys: dict[str, str | None]
    properties: dict[str, tuple[str, float, int] | None]
    fingerprints: list[str]
    fragments: list[str]


class _PlanCall(_Request):
    sid: int
    similarity: float


class _SourceConfig(TypedDict):
    parents: int
    minimum: float
    per_parent: int
    total: int
    tolerance: float
    ppm: float


class _SourceProducts:
    def __init__(self, rows: list[_PlanCall]) -> None:
        self.rows = rows
        self.calls: list[ReactionRequest] = []

    def products(self, request: ReactionRequest) -> tuple[ReactionProduct, ...]:
        self.calls.append(request)
        normalized = normalize_request(request)
        row = next(
            row
            for row in self.rows
            if row["smiles"] == request["smiles"]
            and float(row["delta"]).hex() == float(request["delta"]).hex()
            and float(row["tol"]).hex() == float(normalized["tolerance"]).hex()
        )
        return tuple({"raw": raw, "combo": combo} for raw, combo in row["products"])


class Prepared:
    """Supply every full source input through strict prepared-data boundaries."""

    def __init__(self, expected: dict[str, Tensor]) -> None:
        self.inputs = json_loads(decode(expected["preparation/metadata"]), _Inputs)
        self.fingerprints: dict[str, NDArray[np.uint8]] = {
            value: expected[f"preparation/fingerprints/{index}"].numpy()
            for index, value in enumerate(self.inputs["fingerprints"])
        }
        self.fragments: dict[str, NDArray[np.float64]] = {
            value: expected[f"preparation/fragments/{index}"].numpy()
            for index, value in enumerate(self.inputs["fragments"])
        }

    def products(self, request: ReactionRequest) -> tuple[ReactionProduct, ...]:
        assert request.get("application_budget", 800) == 800
        assert request.get("stage_parents", 24) == 24
        assert request.get("first_products", 32) == 32
        assert request.get("later_products", 16) == 16
        row = next(
            row
            for row in self.inputs["requests"]
            if row["smiles"] == request["smiles"]
            and float(row["delta"]).hex() == float(request["delta"]).hex()
            and float(row["tol"]).hex() == float(request.get("tolerance", 0.003)).hex()
            and row["max_steps"] == request.get("max_steps", 2)
            and row["max_per_analog"] == request.get("max_products", 60)
            and row["allow_isomerisation"] == request.get("allow_isomerization", True)
        )
        return tuple({"raw": raw, "combo": combo} for raw, combo in row["products"])

    def standardize(self, smiles: str) -> str | None:
        return self.inputs["standardized"][smiles]

    def key(self, smiles: str) -> str | None:
        return self.inputs["keys"][smiles]

    def properties(self, smiles: str) -> tuple[str, float, int] | None:
        return self.inputs["properties"][smiles]

    def fingerprint(self, smiles: str) -> NDArray[np.uint8]:
        return self.fingerprints[smiles]

    def base_for_smiles(self, smiles: str) -> NDArray[np.float64]:
        return self.fragments[smiles]


def test_complete_generation_all_eight_cases_and_every_admitted_field() -> None:
    golden = _BASELINE / "testdata/generation_inputs.pt"
    expected = read_tensors(golden)
    prepared = Prepared(expected)
    strings = tuple(decode(expected[f"parents/{index}"]) for index in range(9))
    actual = {
        name: value.clone()
        for name, value in expected.items()
        if name.startswith(("parents/", "preparation/")) or name == "bits"
    }
    for name in (
        "default",
        "excluded",
        "total_cap",
        "parent_cap",
        "no_parents",
        "charged",
        "salt",
        "wildcard",
    ):
        config = ReactionScreen.Config()
        config.parents, config.total = from_plain(
            expected[name + "/limits"].tolist(),
            tuple[int, int],
        )
        analogs = [
            Analog(
                sid=int(sid),
                similarity=similarity,
                shift=shift,
                natural_product=bool(row),
            )
            for sid, similarity, shift, row in from_plain(
                expected[name + "/analogs"].tolist(),
                list[tuple[float, float, float, float]],
            )
        ]
        excluded = decode(expected[name + "/exclude"])
        result = config.make().run(
            analogs,
            target=float(expected[name + "/target"]),
            excluded={excluded} if excluded else set(),
            train_smiles=strings,
            train_mass=expected["parents/mass"].numpy(),
            train_fingerprints=expected["parents/fingerprints"].numpy(),
            bits=expected["bits"].numpy(),
            proposals=prepared,
            chemistry=prepared,
            fragments=prepared,
        )
        for field in ("target", "limits", "exclude", "analogs"):
            actual[name + "/" + field] = expected[name + "/" + field].clone()
        actual[name + "/count"] = torch.tensor(len(result.candidates))
        actual[name + "/log"] = text(
            str(
                [
                    (
                        row.parent,
                        row.similarity,
                        row.combo,
                        row.kind,
                        row.how,
                        row.identity,
                    )
                    for row in result.logs
                ],
            ),
        )
        for index, product in enumerate(result.candidates):
            prefix = f"{name}/{index}"
            for field, value in (
                ("smiles", product.smiles),
                ("key", product.key),
                ("formula", product.formula),
                ("raw", product.raw),
                ("combo", str(product.combo)),
            ):
                assert value is not None
                actual[prefix + "/" + field] = text(value)
            actual[prefix + "/fp"] = torch.from_numpy(product.fingerprint.copy())
            actual[prefix + "/fragments"] = torch.from_numpy(product.fragments.copy())
            actual[prefix + "/values"] = torch.tensor(
                [
                    product.mass,
                    product.heavy_atoms,
                    product.parent_similarity,
                    product.steps,
                    product.parent_tanimoto,
                    product.parent_sid,
                ],
                dtype=torch.float64,
            )
    assert_tensor_golden(
        golden,
        float32_comparands(
            actual,
            inputs=("parents/*", "preparation/*", "*/target", "*/analogs"),
        ),
    )


@pytest.mark.parametrize("case", ["default", "parent_cap", "no_parents"])
def test_planned_requests_keep_parent_order_and_every_budget(case: str) -> None:
    expected = read_tensors(_BASELINE / "testdata/generation_inputs.pt")
    prepared = Prepared(expected)
    parents = tuple(decode(expected[f"parents/{index}"]) for index in range(9))
    config = ReactionScreen.Config()
    parent_limit, config.total = from_plain(
        expected[case + "/limits"].tolist(),
        tuple[int, int],
    )
    config.parents = parent_limit
    analogs = tuple(
        Analog(
            sid=int(sid),
            similarity=similarity,
            shift=shift,
            natural_product=bool(row),
        )
        for sid, similarity, shift, row in from_plain(
            expected[case + "/analogs"].tolist(),
            list[tuple[float, float, float, float]],
        )
    )
    planned = config.make().plan(
        analogs,
        target=float(expected[case + "/target"]),
        train_smiles=parents,
        train_mass=expected["parents/mass"].numpy(),
    )
    assert (
        tuple(analog for analog, _ in planned)
        == tuple(analog for analog in analogs if analog.similarity >= 0.3)[
            :parent_limit
        ]
    )
    for analog, request in planned:
        row = next(
            row
            for row in prepared.inputs["requests"]
            if row["smiles"] == parents[analog.sid]
            and float(row["delta"]).hex() == float(request["delta"]).hex()
        )
        assert request == {
            "smiles": row["smiles"],
            "delta": row["delta"],
            "tolerance": row["tol"],
            "max_steps": row["max_steps"],
            "max_products": row["max_per_analog"],
            "allow_isomerization": row["allow_isomerisation"],
            "application_budget": 800,
            "stage_parents": 24,
            "first_products": 32,
            "later_products": 16,
        }


def test_whole_relative_window_threshold_and_all_reached_requests() -> None:
    golden = _BASELINE / "testdata/generation_requests.pt"
    expected = read_tensors(golden)
    prepared = Prepared(read_tensors(_BASELINE / "testdata/generation_inputs.pt"))
    actual = {
        name: value.clone()
        for name, value in expected.items()
        if name.startswith(("parents/", "input/")) or name == "bits"
    }
    parents = tuple(decode(expected[f"parents/{index}"]) for index in range(9))
    analogs = tuple(
        Analog(sid=int(sid), similarity=sim, shift=shift, natural_product=bool(row))
        for sid, sim, shift, row in from_plain(
            expected["input/analogs"].tolist(),
            list[tuple[float, float, float, float]],
        )
    )
    for case in ("relative", "parent_cap", "empty"):
        source = json_loads(decode(expected[case + "/config"]), _SourceConfig)
        config = ReactionScreen.Config()
        config.parents = source["parents"]
        screen = config.make()
        actual[case + "/config"] = _json_text(
            {
                "parents": config.parents,
                "minimum": config.min_similarity,
                "per_parent": config.per_parent,
                "total": config.total,
                "tolerance": config.tolerance,
                "ppm": config.ppm,
            },
        )
        target = float(expected[case + "/target"])
        actual[case + "/target"] = torch.tensor(target, dtype=torch.float64)
        products = _SourceProducts(
            json_loads(decode(expected[case + "/requests"]), list[_PlanCall]),
        )
        plan = screen.plan(
            analogs,
            target=target,
            train_smiles=parents,
            train_mass=expected["parents/mass"].numpy(),
        )
        result = screen.run(
            analogs,
            target=target,
            excluded=set(),
            train_smiles=parents,
            train_mass=expected["parents/mass"].numpy(),
            train_fingerprints=expected["parents/fingerprints"].numpy(),
            bits=expected["bits"].numpy(),
            proposals=products,
            chemistry=prepared,
            fragments=prepared,
        )
        observed: list[_PlanCall] = []
        for (analog, request), called in zip(plan, products.calls, strict=True):
            assert called == request
            normalized = normalize_request(request)
            observed.append(
                {
                    "sid": analog.sid,
                    "similarity": analog.similarity,
                    "smiles": normalized["smiles"],
                    "delta": normalized["delta"],
                    "tol": normalized["tolerance"],
                    "max_steps": normalized["max_steps"],
                    "max_per_analog": normalized["max_products"],
                    "allow_isomerisation": normalized["allow_isomerization"],
                    "products": [],
                },
            )
        actual[case + "/requests"] = _json_text(observed)
        actual[case + "/count"] = torch.tensor(len(result.candidates))
        actual[case + "/log"] = _json_text(
            [
                (row.parent, row.similarity, row.combo, row.kind, row.how, row.identity)
                for row in result.logs
            ],
        )
    assert_tensor_golden(golden, actual)


def test_seen_keys_are_logged_before_missing_properties_and_mass_checks() -> None:
    expected = read_tensors(_BASELINE / "testdata/generation_inputs.pt")
    prepared = Prepared(expected)
    prepared.inputs["requests"] = [
        {
            "smiles": "parent",
            "delta": 100.0,
            "tol": 0.004,
            "max_steps": 2,
            "max_per_analog": 60,
            "allow_isomerisation": True,
            "products": [("raw", ("first", "second"))],
        },
    ]
    prepared.inputs["standardized"] = {"raw": "standardized"}
    prepared.inputs["keys"] = {"standardized": "seen"}
    prepared.inputs["properties"] = {}
    result = (
        ReactionScreen.Config()
        .make()
        .run(
            [Analog(sid=0, similarity=0.75)],
            target=100.0,
            excluded={"seen"},
            train_smiles=("parent",),
            train_mass=np.array([0.0]),
            train_fingerprints=np.array([[0]], np.uint8),
            bits=np.array([0], np.int64),
            proposals=prepared,
            chemistry=prepared,
            fragments=prepared,
        )
    )
    assert not result.candidates
    assert [
        (row.parent, row.similarity, row.combo, row.kind, row.how, row.identity)
        for row in result.logs
    ] == [
        (0, 0.75, ("first", "second"), "pool", "key", "seen"),
    ]
    prepared.inputs["keys"] = {}
    with pytest.raises(KeyError, match=r"^'standardized'$"):
        ReactionScreen.Config().make().run(
            [Analog(sid=0, similarity=0.75)],
            target=100.0,
            excluded=set(),
            train_smiles=("parent",),
            train_mass=np.array([0.0]),
            train_fingerprints=np.array([[0]], np.uint8),
            bits=np.array([0], np.int64),
            proposals=prepared,
            chemistry=prepared,
            fragments=prepared,
        )


class _Table:
    """Serve one parent's proposals and every prepared decision from plain tables."""

    def __init__(self, masses: dict[str, float]) -> None:
        self.masses = masses

    def products(self, request: ReactionRequest) -> tuple[ReactionProduct, ...]:
        assert request["smiles"] == "parent"
        return tuple({"raw": raw, "combo": ("rule",)} for raw in self.masses)

    def standardize(self, smiles: str) -> str | None:
        return None if smiles == "unstandardizable" else smiles

    def key(self, smiles: str) -> str | None:
        return {"keyless": None, "seen": "seen"}.get(smiles, "key-" + smiles)

    def properties(self, smiles: str) -> tuple[str, float, int] | None:
        return ("C2H6O", self.masses[smiles], 3)

    def fingerprint(self, smiles: str) -> NDArray[np.uint8] | None:
        if smiles == "unprinted":
            return None
        return np.packbits(np.array([1, 1, 0, 1, 1, 0, 0, 0], np.uint8))

    def base_for_smiles(self, smiles: str) -> NDArray[np.float64]:
        return np.array([self.masses[smiles]])


def test_each_rejected_product_skips_only_itself_inside_the_exact_mass_window() -> None:
    def admit(
        masses: dict[str, float],
        *,
        tolerance: float,
        ppm: float,
        target: float,
    ) -> GenerationResult:
        config = ReactionScreen.Config()
        config.tolerance, config.ppm = tolerance, ppm
        table = _Table(masses)
        return config.make().run(
            [Analog(sid=0, similarity=0.9)],
            target=target,
            excluded={"seen"},
            train_smiles=("parent",),
            train_mass=np.array([0.0]),
            train_fingerprints=np.packbits(
                np.array([[1, 1, 1, 0, 0, 0, 0, 0]], np.uint8),
                axis=1,
            ),
            bits=np.arange(8),
            proposals=table,
            chemistry=table,
            fragments=table,
        )

    result = admit(
        {
            "unstandardizable": 100.0,
            "keyless": 100.0,
            "seen": 100.0,
            "far": 102.0,
            "edge": 100.5,
            "unprinted": 100.0,
            "good": 100.0,
        },
        tolerance=0.5,
        ppm=0.0,
        target=100.0,
    )
    assert [candidate.raw for candidate in result.candidates] == ["edge", "good"]
    assert [(row.kind, row.identity) for row in result.logs] == [("pool", "seen")]
    for candidate in result.candidates:
        assert candidate.parent_tanimoto == float(np.float32(0.4))
        assert (candidate.parent_sid, candidate.steps) == (0, 1)
        assert candidate.combo == ("rule",)
    relative = admit(
        {"inside": 1000.05, "outside": 1000.5},
        tolerance=0.0,
        ppm=100.0,
        target=1000.0,
    )
    assert [candidate.raw for candidate in relative.candidates] == ["inside"]


def decode(value: Tensor) -> str:
    """Read a stored UTF8 input."""
    return bytes(from_plain(value.tolist(), list[int])).decode()


def text(value: str) -> Tensor:
    """Retain every byte of an observed source-compatible output."""
    return torch.tensor(list(value.encode()), dtype=torch.uint8)


def _json_text(value: object) -> Tensor:
    return text(json.dumps(to_plain(value), separators=(",", ":")))


def assert_main_pipeline(
    tmp_path: Path,
) -> None:
    """Replay every public main pipeline field against its golden."""
    golden = _BASELINE / "testdata/main_pipeline.pt"
    expected = read_tensors(golden)
    actual = {
        k: v.clone()
        for k, v in expected.items()
        if k.startswith(("input/", "prepared/", "train/", "proposal/", "identity/"))
    }
    bits = expected["input/bits"].numpy()
    train_mass = expected["input/train_mass"].numpy()
    mass = np.repeat(train_mass, 2)
    order = np.argsort(mass, kind="mergesort")
    counts = np.array([3, 4, 2, 5], np.int32)
    pool = CandidatePool(
        mass=expected["input/pool_mass"].numpy(),
        keys=_pipeline_strings(expected["input/pool_keys"]),
        smiles=_pipeline_strings(expected["input/pool_smiles"]),
        formulas=_pipeline_strings(expected["input/pool_formula"]),
        heavy_atoms=expected["input/pool_heavy"].numpy(),
        source=np.array([0], np.int8),
        train_sid=np.array([1], np.int64),
        packed=expected["input/pool_fp"].numpy(),
        bits=bits,
        fragment_offsets=np.array([0, len(expected["prepared/0/depth2"])], np.int64),
        fragment_mass=expected["prepared/0/depth2"].numpy().astype(np.float32),
        key_to_pid={
            key: i
            for i, key in enumerate(_pipeline_strings(expected["input/pool_keys"]))
        },
    )
    library = SpectralLibrary(
        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        mz=np.concatenate(
            [
                np.arange(counts.item(i), dtype=np.float32) * 10 + 20
                for i in range(len(counts))
            ],
        ),
        intensity=np.concatenate(
            [
                np.arange(counts.item(i), dtype=np.float32) % 3 + 1
                for i in range(len(counts))
            ],
        ),
        structure_mass=train_mass,
        keys=_pipeline_strings(expected["input/train_keys"]),
        smiles=_pipeline_strings(expected["input/train_smiles"]),
        sid=np.array([0, 0, 1, 1], np.int64),
        library=np.array([2, 3, 0, 2], np.int8),
        mode=np.array([1, -1, 1, -1], np.int8),
        instrument=np.array([0, 1, 2, 0], np.int8),
        adduct=np.array([0, 6, 0, 6], np.int16),
        precursor=mass + 1.00727645217,
        peak_count=counts,
        mass=mass,
        order=order,
        mass_sorted=mass[order],
    )
    smiles, keys, formulas = (
        _pipeline_strings(expected["candidates/" + k])
        for k in ("smiles", "key", "formula")
    )
    generated = tuple(
        GeneratedCandidate(
            smiles=smiles[i],
            key=keys[i],
            formula=formulas[i],
            mass=expected["candidates/mass"][i].item(),
            heavy_atoms=int(expected["candidates/n_heavy"][i].item()),
            fingerprint=expected["candidates/fp"][i].numpy(),
            fragments=expected[f"candidates/frags/{i}/mass"].numpy(),
            parent_similarity=expected["candidates/gen_sim"][i].item(),
            steps=int(expected["candidates/gen_steps"][i].item()),
            parent_tanimoto=expected["candidates/gen_tan"][i].item(),
            parent_sid=int(expected["context/parent_gen"][j].item()),
            combo=tuple(_pipeline_strings(expected["context/combo_gen"])[j].split("+")),
            raw=_pipeline_strings(expected["context/raw_gen"])[j],
        )
        for j, i in enumerate(np.flatnonzero(expected["candidates/is_gen"].numpy()))
    )
    target = pool.mass.item(0)
    spectra: list[SpectrumRequest] = [
        {
            "mz": [20, 30, 40, 105],
            "it": [1, 2, 3, 0.01],
            "prec": target + 1.00727645217,
            "mode": 1,
            "adduct": "[M+H]+",
            "ce": 10.0,
            "ce_n": 0,
        },
        {
            "mz": [20, 40, 50],
            "it": [3, 1, 2],
            "prec": target - 1.00727645217,
            "mode": -1,
            "adduct": "[M-H]-",
            "ce": float("nan"),
            "ce_n": 12,
        },
        {
            "mz": [],
            "it": [],
            "prec": target + 1.00727645217,
            "mode": 1,
            "adduct": "[M+H]+",
            "ce": float("nan"),
            "ce_n": 2,
        },
    ]
    proposals = _PipelineProposals(expected)
    provider = _Provider(
        record=expected,
        priors=DerivationPriors(
            {
                "glob": 0.2,
                "tf": {"methyl_O": (3.0, 40.0), "demethyl_O": (4.0, 20.0)},
                "env": {1: {}, 2: {}, 3: {}},
            },
            proposals=proposals,
        ),
    )
    numeric = expected["context/logs_numeric"]
    log_text = _pipeline_strings(expected["context/logs_strings"])
    logs = tuple(
        DerivationLog(
            parent=int(numeric[i, 0].item()),
            similarity=numeric[i, 1].item(),
            combo=tuple(parts[0].split("+")),
            kind=identity["kind"],
            how=identity["how"],
            identity=parts[3],
        )
        for i, line in enumerate(log_text)
        if line
        for parts in [line.split("\t")]
        for identity in [from_plain({"kind": parts[1], "how": parts[2]}, _LogIdentity)]
    )
    old = read_tensors(_BASELINE / "testdata/neural_bank.pt")
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        banks: list[MainBank] = []
        for name, indices in (("full", (0,)), ("views", (0, 1, 2))):
            paths: list[Path] = []
            for index in indices:
                checkpoint = source_checkpoint(old, index=index)
                if name == "full":
                    state = from_plain(checkpoint["model"], dict[str, Tensor])
                    state = {k: v.clone() for k, v in state.items()}
                    state["head.3.bias"] += 0.125
                    state["elem_head.2.bias"] += 0.25
                    checkpoint["model"] = state
                path = tmp_path / f"{name}{index}.pt"
                torch.save(checkpoint, path)
                paths.append(path)
            config = MainBank.Config()
            config.checkpoints = [
                checkpoint_config(path, index=index)
                for index, path in zip(indices, paths, strict=True)
            ]
            if name == "full":
                config.merged_adduct = modal_adduct
                config.single_count = keep_count
            banks.append(config.make())
        control = read_tensors(_BASELINE / "testdata/fragnet.pt")
        chains = tuple(
            tuple(
                tuple(_pipeline_strings(control[f"stageA/chain/{i}/{j}"]))
                for j in range(3)
            )
            for i in range(2)
        )
        tabs = tuple(
            tuple(
                (
                    control[f"stageA/keys/{i}/{j}"].numpy(),
                    control[f"stageA/values/{i}/{j}"].numpy(),
                )
                for j in range(3)
            )
            for i in range(2)
        )
        stage_a = StageA(chains=chains, tabs=tabs, alpha=control["stageA/alpha"].item())
        stage_b = _tree(control, "stageB/")
        engine = MainEngine.Config().make()
        train_fp = expected["input/train_fp"].numpy()
        base = engine.base(
            spectra,
            target=target,
            library=library,
            pool=pool,
            train_fingerprints=train_fp,
            bank=banks[0],
            generated=generated,
        )
        values, views = family_features(
            base,
            spectra,
            provider=provider,
            views_bank=banks[1],
            train_fingerprints=train_fp,
            bits=bits,
            stage_a=stage_a,
            stage_b=stage_b,
            logs=logs,
            search=engine.fragment_search,
            query_floor=engine.config.query_floor,
            query_max_peaks=engine.config.query_max_peaks,
            query_over=engine.config.query_over,
            max_candidates=engine.config.family_max_candidates,
            pairwise_rank_limit=engine.config.pairwise_rank_limit,
        )
        actual["base"] = torch.from_numpy(base.features)
        names = tuple(values)
        actual["families"] = torch.from_numpy(
            select_features(values, names=names, count=len(base.rows.mass)),
        )
        actual["family_names"] = _text("\0".join(names))
        for name, array in (
            ("z_s", views.single),
            ("el_s", views.single_elements),
            ("z_m", views.merged),
            ("el_m", views.merged_elements),
            ("m_modes", views.modes),
        ):
            actual["context/" + name] = torch.from_numpy(array)
        rows = base.rows
        for name, array in (
            ("pid", rows.candidates.pool_rows),
            ("mass", rows.mass),
            ("n_heavy", rows.heavy_atoms),
            ("train_sid", rows.train_sid),
            ("fp", rows.fingerprints),
            ("is_gen", rows.candidates.generated.astype(np.float32)),
            ("gen_sim", rows.parent_similarity),
            ("gen_steps", rows.steps),
            ("gen_tan", rows.parent_tanimoto),
        ):
            actual["candidates/" + name] = torch.from_numpy(array)
        for name, strings in (
            ("key", rows.candidates.keys),
            ("smiles", rows.candidates.smiles),
            ("formula", rows.candidates.formulas),
        ):
            actual["candidates/" + name] = _text("\0".join(strings))
        for i, row in enumerate(rows.fragments):
            actual[f"candidates/frags/{i}/valid"] = torch.tensor(row is not None)
            if row is not None:
                actual[f"candidates/frags/{i}/mass"] = torch.from_numpy(row)
        d = base.diagnostics
        actual["info"] = torch.tensor(
            [d.n_cand, d.lib_max, d.top_sim, d.fzmax, d.n_gen],
            dtype=torch.float64,
        )
        gi = np.flatnonzero(rows.candidates.generated)
        context = {
            "n_cand": torch.tensor(len(rows.mass)),
            "n_gen": torch.tensor(len(gi)),
            "pid": torch.from_numpy(rows.candidates.pool_rows.astype(np.int32)),
            "chan": torch.from_numpy(
                np.where(rows.candidates.generated, 2, 0).astype(np.int8),
            ),
            "key_gen": _text("\0".join(rows.candidates.keys[i] for i in gi)),
            "smiles_gen": _text("\0".join(rows.candidates.smiles[i] for i in gi)),
            "formula_gen": _text("\0".join(rows.candidates.formulas[i] for i in gi)),
            "mass_gen": torch.from_numpy(rows.mass[gi]),
            "n_heavy_gen": torch.from_numpy(rows.heavy_atoms[gi]),
            "fp_gen": torch.from_numpy(
                np.packbits(rows.fingerprints[gi].astype(np.uint8), axis=1),
            ),
            "parent_gen": torch.from_numpy(rows.parent_sid[gi]),
            "gen_sim": torch.from_numpy(rows.parent_similarity[gi]),
            "gen_steps": torch.from_numpy(rows.steps[gi]),
            "gen_tan": torch.from_numpy(rows.parent_tanimoto[gi]),
            "combo_gen": _text("\0".join("+".join(rows.combos[i] or ()) for i in gi)),
            "raw_gen": _text("\0".join(rows.raw[i] or "" for i in gi)),
            "raw_valid": torch.tensor([rows.raw[i] is not None for i in gi]),
            "an_sid": torch.tensor([a.sid for a in base.analogs], dtype=torch.int64),
            "an_sim": torch.tensor(
                [a.similarity for a in base.analogs],
                dtype=torch.float32,
            ),
            "an_shift": torch.tensor(
                [a.shift for a in base.analogs],
                dtype=torch.float64,
            ),
            "an_np": torch.tensor(
                [a.natural_product for a in base.analogs],
                dtype=torch.bool,
            ),
            "gen_order": _text("\0".join(g.key for g in generated)),
            "logs_numeric": torch.tensor(
                [[log.parent, log.similarity] for log in logs],
                dtype=torch.float64,
            ).reshape(-1, 2),
            "logs_strings": _text(
                "\0".join(
                    "\t".join(
                        (
                            "+".join(log.combo or ()),
                            log.kind,
                            log.how,
                            log.identity or "",
                        ),
                    )
                    for log in logs
                ),
            ),
        }
        actual.update(("context/" + name, value) for name, value in context.items())
        selected_names = _pipeline_strings(expected["selected_names"])
        all_values = {
            name: base.features[:, i] for i, name in enumerate(main_feature_names())
        } | values
        selected = select_features(
            all_values,
            names=selected_names,
            count=len(rows.mass),
        )
        actual["selected"] = torch.from_numpy(selected)
        actual["selected_names"] = _text("\0".join(selected_names))
        rankers = TreeBank(
            features=selected_names,
            models=tuple(_tree(expected, f"ranker/{i}/") for i in range(4)),
        )
        train = np.column_stack([np.arange(4) == j % 4 for j in range(160)]).astype(
            np.float64,
        )
        for i, model in enumerate(rankers.models):
            actual[f"ranker/{i}/header"] = expected[f"ranker/{i}/header"].clone()
            for step, tree in enumerate(model.trees):
                for name, array in tree.columns.items():
                    actual[f"ranker/{i}/tree/{step}/{name}"] = torch.from_numpy(array)
                actual[f"ranker/{i}/train/{step + 1}"] = torch.from_numpy(
                    model.predict(train, rounds=step + 1),
                )
            actual[f"ranker/{i}/scores"] = torch.from_numpy(
                model.predict(selected.astype(np.float64)),
            )
        actual["scores"] = torch.from_numpy(rankers.score(selected, selected_names))
        ranked = engine.run(
            spectra,
            target=target,
            library=library,
            pool=pool,
            train_fingerprints=train_fp,
            bank=banks[0],
            views_bank=banks[1],
            provider=provider,
            stage_a=stage_a,
            stage_b=stage_b,
            rankers=rankers,
            generated=generated,
            logs=logs,
        )
        np.testing.assert_array_equal(ranked.features, actual["selected"].numpy())
        np.testing.assert_array_equal(ranked.scores, actual["scores"].numpy())
        if ranked.feature_names != selected_names:
            raise ValueError("Expected ranked.feature_names == selected_names.")
        if ranked.features.dtype != np.float32:
            raise ValueError("Expected ranked.features.dtype == np.float32.")
        if ranked.scores.dtype != np.float64:
            raise ValueError("Expected ranked.scores.dtype == np.float64.")
        if ranked.candidates.smiles != rows.candidates.smiles:
            raise ValueError(
                "Expected ranked.candidates.smiles == rows.candidates.smiles.",
            )
        if ranked.candidates.keys != rows.candidates.keys:
            raise ValueError("Expected ranked.candidates.keys == rows.candidates.keys.")
        if ranked.candidates.formulas != rows.candidates.formulas:
            raise ValueError(
                "Expected ranked.candidates.formulas == rows.candidates.formulas.",
            )
        if ranked.candidates.sources != rows.candidates.sources:
            raise ValueError(
                "Expected ranked.candidates.sources == rows.candidates.sources.",
            )
        np.testing.assert_array_equal(
            ranked.candidates.pool_rows,
            rows.candidates.pool_rows,
        )
        np.testing.assert_array_equal(
            ranked.candidates.generated,
            rows.candidates.generated,
        )
        if ranked.diagnostics != base.diagnostics:
            raise ValueError("Expected ranked.diagnostics == base.diagnostics.")
        limited_config = MainEngine.Config()
        limited_config.max_query_spectra = 3
        limited = limited_config.make().run(
            [*spectra, spectra[0]],
            target=target,
            library=library,
            pool=pool,
            train_fingerprints=train_fp,
            bank=banks[0],
            views_bank=banks[1],
            provider=provider,
            stage_a=stage_a,
            stage_b=stage_b,
            rankers=rankers,
            generated=generated,
            logs=logs,
            merged_counts=(0, 12, 2, 99),
        )
        np.testing.assert_array_equal(limited.features, ranked.features)
        np.testing.assert_array_equal(limited.scores, ranked.scores)
        if [s.get("ce_n") for s in spectra] != [0, 12, 2]:
            raise ValueError("Source conditioning metadata was mutated.")
        empty = engine.run(
            spectra,
            target=900.0,
            library=library,
            pool=pool,
            train_fingerprints=train_fp,
            bank=banks[0],
            views_bank=banks[1],
            provider=provider,
            stage_a=stage_a,
            stage_b=stage_b,
            rankers=rankers,
        )
        if empty.features.shape != (0, 160):
            raise ValueError("Expected empty.features.shape == (0, 160).")
        if empty.features.dtype != np.float32:
            raise ValueError("Expected empty.features.dtype == np.float32.")
        if empty.scores.shape != (0,):
            raise ValueError("Expected empty.scores.shape == (0,).")
        if empty.scores.dtype != np.float64:
            raise ValueError("Expected empty.scores.dtype == np.float64.")
        if empty.feature_names != selected_names:
            raise ValueError("Expected empty.feature_names == selected_names.")
    assert_tensor_golden(
        golden,
        float32_comparands(
            actual,
            inputs=(
                "input/*",
                "prepared/*",
                "candidates/mass",
                "candidates/frags/*",
                "context/logs_numeric",
                "ranker/*/tree/*",
            ),
        ),
    )


class _LogIdentity(TypedDict):
    kind: Literal["pool", "gen"]
    how: Literal["smi", "key"]


def _pipeline_strings(value: Tensor) -> tuple[str, ...]:
    return tuple(value.numpy().tobytes().decode().split("\0"))


def _text(value: str) -> Tensor:
    return torch.tensor(list(value.encode()), dtype=torch.uint8)


def _tree(record: dict[str, Tensor], prefix: str) -> LightGBMModel:
    text = record[prefix + "header"].numpy().tobytes().decode()
    for step in range(3):
        text += f"Tree={step}\n"
        key = prefix + f"tree/{step}/"
        for name, tensor in record.items():
            if name.startswith(key):
                text += (
                    name.removeprefix(key)
                    + "="
                    + " ".join(str(v) for v in tensor.tolist())
                    + "\n"
                )
    return LightGBMModel.parse(text + "end of trees\n")


def _structure(record: dict[str, Tensor], prefix: str) -> AnalogRecord:
    smiles, formula, core, element, query = _pipeline_strings(
        record[prefix + "strings"],
    )
    graphs: dict[str, ElementGraph | None] = {}
    for name in ("graph", "core"):
        graphs[name] = (
            ElementGraph(
                elements=record[prefix + name + "/elements"].numpy(),
                bonds=record[prefix + name + "/bonds"].numpy(),
            )
            if record[prefix + name + "/valid"].item()
            else None
        )
    return AnalogRecord(
        smiles=smiles,
        formula_input=formula,
        hashes=record[prefix + "hashes"].numpy(),
        numbers=record[prefix + "numbers"].numpy(),
        formula=record[prefix + "formula"].numpy(),
        core_smiles=core,
        element_smiles=element,
        query_smiles=query,
        graph=graphs["graph"],
        core=graphs["core"],
        core_mask=record[prefix + "core_mask"].numpy(),
    )


class _PipelineProposals:
    def __init__(self, record: dict[str, Tensor]) -> None:
        self.identities: dict[str, tuple[str | None, str | None]] = {}
        smiles = _pipeline_strings(record["identity/smiles"])
        keys, skeletons = (
            _pipeline_strings(record["identity/key"]),
            _pipeline_strings(record["identity/skeleton"]),
        )
        for i, s in enumerate(smiles):
            self.identities[s] = (
                keys[i] if record["identity/key_valid"][i].item() else None,
                skeletons[i] if record["identity/skeleton_valid"][i].item() else None,
            )
        self.rows: dict[tuple[str, tuple[str, ...]], tuple[ProposalPath, ...]] = {}
        for i in range(int(record["proposal/count"].item())):
            prefix = f"proposal/{i}/"
            parent, combo = (
                record[prefix + "header"].numpy().tobytes().decode().split("\t")
            )
            rows: list[ProposalPath] = []
            raw, standard = (
                _pipeline_strings(record[prefix + "raw"]),
                _pipeline_strings(record[prefix + "standardized"]),
            )
            for j in range(int(record[prefix + "size"].item())):
                key, skeleton = self.identity(raw[j])
                rows.append(
                    {
                        "raw": raw[j],
                        "standardized": standard[j]
                        if record[prefix + "valid"][j].item()
                        else None,
                        "key": key,
                        "skeleton": skeleton,
                        "environments": from_plain(
                            cast(object, record[prefix + "env"][j].tolist()),
                            tuple[tuple[int, int, int], ...],
                        ),
                    },
                )
            self.rows[parent, tuple(combo.split("+"))] = tuple(rows)

    def paths(
        self,
        parent_smiles: str,
        combo: tuple[str, ...],
    ) -> tuple[ProposalPath, ...]:
        return self.rows[parent_smiles, combo]

    def identity(self, smiles: str) -> tuple[str | None, str | None]:
        return self.identities[smiles]


@dataclass(kw_only=True, slots=True)
class _Provider:
    record: dict[str, Tensor]
    priors: DerivationPriors

    def select(self, rows: CandidateRows, analogs: Sequence[Analog]) -> MainFamilyRows:
        structures: list[AnalogRecord] = []
        fragnet: list[FragnetRecord | None] = []
        for i in range(len(rows.mass)):
            prefix = f"prepared/{i}/"
            structures.append(_structure(self.record, prefix))
            molecular: FragnetRecord | None = None
            if self.record[prefix + "fragnet_valid"].item():
                molecular = {
                    "amass": self.record[prefix + "fragnet/amass"].numpy(),
                    "kind": self.record[prefix + "fragnet/kind"].numpy(),
                    "el": self.record[prefix + "fragnet/el"].numpy(),
                    "nh": self.record[prefix + "fragnet/nh"].numpy(),
                    "bonds": self.record[prefix + "fragnet/bonds"].numpy(),
                    "bring": self.record[prefix + "fragnet/bring"].numpy(),
                    "mass": self.record[prefix + "fragnet/mass"].item(),
                    "el10": self.record[prefix + "fragnet/el10"].numpy(),
                }
            fragnet.append(molecular)
        return MainFamilyRows(
            candidate_structures=tuple(structures),
            analog_structures=tuple(
                _structure(self.record, f"train/{a.sid}/") for a in analogs
            ),
            train_smiles=_pipeline_strings(self.record["input/train_smiles"]),
            depth1=tuple(
                self.record[f"prepared/{i}/depth1"].numpy()
                for i in range(len(rows.mass))
            ),
            depth2=tuple(
                self.record[f"prepared/{i}/depth2"].numpy()
                for i in range(len(rows.mass))
            ),
            capabilities=np.stack(
                [
                    self.record[f"prepared/{i}/caps"].numpy()
                    for i in range(len(rows.mass))
                ],
            ),
            fragnet=tuple(fragnet),
            priors=self.priors,
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
