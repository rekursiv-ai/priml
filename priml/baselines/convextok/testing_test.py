"""The golden readers must rebuild PSLP's programs and name the first mismatch."""

from pathlib import Path
from typing import Final

import json
import zipfile

from treekle import ReadError, json_loads

import numpy as np
import pytest
import torch

from priml.baselines.convextok import testing
from priml.baselines.convextok.presolver.presolve import Presolved, presolve
from priml.baselines.convextok.testing import (
    TESTDATA,
    assert_reduced_like,
    fixture_program,
    input_program,
    npz,
)


_CPU: Final = torch.device("cpu")
_GOLDEN: Final = TESTDATA / "presolve.npz"
_FUZZ: Final = TESTDATA / "presolve_fuzz" / "general_00801.npz"


def test_fixture_program_is_the_program_the_presolve_golden_reduced() -> None:
    program = fixture_program()
    with npz(_GOLDEN) as golden:
        assert_reduced_like(presolve(program, _CPU), golden)
    assert program.num_rows == len(program.row_lower)
    assert program.num_columns == len(program.objective)


@pytest.mark.parametrize(
    ("name", "edit", "error"),
    [
        ("pretokens.json", {"frequencies": [1]}, ValueError),
        ("pretokens.json", {"frequencies": "many"}, ReadError),
        ("candidates.json", {"counts": "many"}, ReadError),
        ("corpus.json", {"budget": "large"}, ReadError),
    ],
    ids=["unpaired-frequencies", "pretokens-schema", "candidates-schema", "corpus"],
)
def test_fixture_program_validates_each_golden_it_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    edit: dict[str, object],
    error: type[Exception],
) -> None:
    for golden in ("pretokens.json", "candidates.json", "corpus.json"):
        (tmp_path / golden).write_text((TESTDATA / golden).read_text())
    document = json_loads((tmp_path / name).read_text(), dict[str, object])
    (tmp_path / name).write_text(json.dumps({**document, **edit}))
    monkeypatch.setattr(testing, "TESTDATA", tmp_path)

    with pytest.raises(error):
        fixture_program()


def test_input_program_rebuilds_every_recorded_input_array() -> None:
    with npz(_FUZZ) as golden:
        program = input_program(golden)
        assert program.num_columns == int(golden["in_num_columns"])
        for key, got in (
            ("crow_indices", program.crow_indices),
            ("col_indices", program.col_indices),
            ("values", program.values),
            ("row_lower", program.row_lower),
            ("row_upper", program.row_upper),
            ("objective", program.objective),
            ("lower", program.lower),
            ("upper", program.upper),
        ):
            assert np.array_equal(got.numpy(), golden[f"in_{key}"]), key


@pytest.mark.parametrize("key", ["Ap", "Ai", "Ax", "lhs", "rhs", "c", "lbs", "ubs"])
def test_assert_reduced_like_names_a_differing_array(tmp_path: Path, key: str) -> None:
    presolved, arrays = _fixture_reduction()
    arrays[key] = arrays[key][:-1]

    with (
        npz(_saved(tmp_path, arrays)) as golden,
        pytest.raises(
            ValueError,
            match=rf"^{key}$",
        ),
    ):
        assert_reduced_like(presolved, golden)


def test_assert_reduced_like_rejects_another_column_count(tmp_path: Path) -> None:
    presolved, arrays = _fixture_reduction()
    reduced = presolved.program
    narrower = Presolved(
        program=type(reduced)(
            crow_indices=reduced.crow_indices,
            col_indices=reduced.col_indices,
            values=reduced.values,
            num_columns=reduced.num_columns + 1,
            row_lower=reduced.row_lower,
            row_upper=reduced.row_upper,
            objective=reduced.objective,
            lower=reduced.lower,
            upper=reduced.upper,
            objective_offset=reduced.objective_offset,
        ),
        state=presolved.state,
        num_columns=presolved.num_columns,
    )

    with (
        npz(_saved(tmp_path, arrays)) as golden,
        pytest.raises(
            ValueError,
            match=r'^Expected reduced\.num_columns == len\(golden\["c"\]\)\.$',
        ),
    ):
        assert_reduced_like(narrower, golden)


def test_assert_reduced_like_rejects_another_offset(tmp_path: Path) -> None:
    presolved, arrays = _fixture_reduction()
    arrays["offset"] = arrays["offset"] + 1.0

    with (
        npz(_saved(tmp_path, arrays)) as golden,
        pytest.raises(
            ValueError,
            match=r'^Expected reduced\.objective_offset == float\(golden\["offset"\]\)\.$',
        ),
    ):
        assert_reduced_like(presolved, golden)


def _fixture_reduction() -> tuple[Presolved, dict[str, np.ndarray]]:
    """Presolve recorded input and copy its golden arrays, ready to perturb."""
    with npz(_FUZZ) as golden:
        arrays = {key: np.array(golden[key]) for key in golden.files}
        program = input_program(golden)
    return presolve(program, _CPU), arrays


def _saved(tmp_path: Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write ``arrays`` as a golden, one ``<key>.npy`` member each, and return it."""
    path = tmp_path / "golden.npz"
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in arrays.items():
            with archive.open(f"{key}.npy", "w") as member:
                np.save(member, value)
    return path


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
