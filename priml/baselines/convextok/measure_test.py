"""A measured fit must save its tokenizer and measure the fit against its program.

On the fixture corpus, the default fit must save upstream's ``det`` vocabulary and
report the program's sizes, every stage's seconds, and PDLP's stop. A stand-in
solver with a known answer pins the objective, the infeasibility, and the
vocabulary diff.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import json
import resource
import time

from configgle import Fig
from tokenizers import Tokenizer, pre_tokenizers
from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.convextok.export import export_tokenizer
from priml.baselines.convextok.measure import MAXRSS_UNIT_BYTES, MeasuredFit
from priml.baselines.convextok.prepare_test import fixture_config
from priml.baselines.convextok.program import LinearProgram
from priml.lib.custom_json import convert, parse


_CWD: Final = Path(__file__).resolve().parent


# The whole fit with presolve and PDLP: 0.3 s with the kernels interpreted, past the
# unit tier's 100 ms.
@pytest.mark.compute_large_fixture
def test_fixture_fit_saves_upstream_vocabulary_and_measures_it(tmp_path: Path) -> None:
    config = _measured_config(tmp_path)
    config.make().run()

    run_dir = tmp_path / "runs/convextok/exp000"
    vocabulary = set(Tokenizer.from_file(str(run_dir / "tokenizer.json")).get_vocab())
    golden = set(convert(_read_json("vocab.json").get("det"), list[str]))
    assert vocabulary == golden | set(pre_tokenizers.ByteLevel.alphabet())
    metrics = _read_metrics(run_dir)
    seconds = convert(metrics.get("seconds"), dict[str, float])
    assert list(seconds) == [
        "read",
        "pretokens",
        "candidates",
        "program",
        "presolve",
        "solve",
        "postsolve",
        "rounding",
        "export",
        "total",
    ]
    assert min(seconds.values()) >= 0
    assert seconds["total"] == sum(
        value for stage, value in seconds.items() if stage != "total"
    )
    with _npz("program.npz") as program:
        vertices, columns = (
            int(size) for size in torch.from_numpy(program["A_eq_shape"])
        )
        token_edges, byte_edges, tokens = (
            int(size) for size in torch.from_numpy(program["sizes"])
        )
    built = convert(metrics.get("program"), dict[str, int])
    assert built == {
        "rows": vertices + token_edges + 1,
        "columns": columns,
        "nonzeros": 4 * token_edges + 2 * byte_edges + tokens,
    }
    solved = convert(metrics.get("solved"), dict[str, int])
    assert solved["columns"] < built["columns"]
    assert metrics.get("optimal") is True
    assert convert(metrics.get("iterations"), int) > 0
    assert metrics.get("peak_gpu_bytes") is None
    assert convert(metrics.get("peak_host_bytes"), int) > 0
    assert metrics.get("pieces") == len(vocabulary)
    assert "missing" not in metrics


def test_known_solution_pins_objective_infeasibility_and_diff(tmp_path: Path) -> None:
    """All zeros: no tokens counted, every pretoken's unit of flow unrouted."""
    config = _measured_config(tmp_path)
    reference = tmp_path / "reference.json"
    learned = sorted(
        set(convert(_read_json("vocab.json").get("det"), list[str]))
        - set(pre_tokenizers.ByteLevel.alphabet()),
    )
    export_tokenizer(
        learned,
        split_pattern=config.preparation.split_pattern,
    ).save(str(reference))
    config.preparation.presolve = None
    config.preparation.solver = _Zeros.Config()
    config.reference = reference
    config.make().run()

    metrics = _read_metrics(tmp_path / "runs/convextok/exp000")
    num_pretokens = len(
        convert(_read_json("pretokens.json").get("pretokens"), list[str]),
    )
    assert metrics.get("objective") == 0.0
    # A unit of flow leaves each pretoken's first vertex and enters its last, so the
    # zero point misses two equality rows by one each.
    assert metrics.get("infeasibility") == (2 * num_pretokens) ** 0.5
    assert metrics.get("program") == metrics.get("solved")
    assert "iterations" not in metrics
    assert "optimal" not in metrics
    assert metrics.get("pieces") == 256
    assert metrics.get("missing") == len(learned)
    assert metrics.get("extra") == 0


def test_run_writes_stage_seconds_memory_and_tokenizer(tmp_path: Path) -> None:
    config = _measured_config(tmp_path)
    config.preparation.presolve = None
    config.preparation.solver = _Zeros.Config()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * MAXRSS_UNIT_BYTES
    start = time.perf_counter()
    config.make().run()
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * MAXRSS_UNIT_BYTES

    run_dir = tmp_path / "runs/convextok/exp000"
    text = (run_dir / "metrics.json").read_text()
    metrics = parse(text, dict[str, object])
    assert text == json.dumps(metrics, indent=2) + "\n"
    seconds = convert(metrics["seconds"], dict[str, float])
    assert list(seconds) == [
        "read",
        "pretokens",
        "candidates",
        "program",
        "solve",
        "rounding",
        "export",
        "total",
    ]
    assert all(0 <= value <= wall for value in seconds.values())
    assert seconds["total"] == sum(
        value for stage, value in seconds.items() if stage != "total"
    )
    assert metrics["peak_gpu_bytes"] is None
    assert before <= convert(metrics["peak_host_bytes"], int) <= after
    tokenizer = Tokenizer.from_file(str(run_dir / "tokenizer.json"))
    assert tokenizer.get_vocab_size() == 256


def test_solution_is_measured_against_the_program_as_built(tmp_path: Path) -> None:
    """A ramp from -0.5 to 1.5 breaks both variable bounds and meets some rows."""
    config = _measured_config(tmp_path)
    config.preparation.presolve = None
    config.preparation.solver = _Ramp.Config()
    config.make().run()

    metrics = _read_metrics(tmp_path / "runs/convextok/exp000")
    with _npz("program.npz") as golden:
        rows_eq = int(torch.from_numpy(golden["A_eq_shape"])[0])
        c, lb, ub, b_eq, b_ub = (
            torch.from_numpy(golden[name]) for name in ("c", "lb", "ub", "b_eq", "b_ub")
        )
        sizes = {
            "rows": rows_eq + len(b_ub),
            "columns": len(c),
            "nonzeros": len(golden["A_eq_data"]) + len(golden["A_ub_data"]),
        }
        x = _ramp(len(c)).to(torch.float64)
        violation = torch.cat(
            [
                _golden_matrix(golden, name="A_eq", rows=rows_eq) @ x - b_eq,
                (_golden_matrix(golden, name="A_ub", rows=len(b_ub)) @ x - b_ub).clamp(
                    min=0,
                ),
                x - x.clamp(min=lb, max=ub),
            ],
        )
    assert metrics.get("objective") == pytest.approx(float(c @ x), rel=1e-12)
    assert metrics.get("infeasibility") == pytest.approx(
        float(torch.linalg.vector_norm(violation)),
        rel=1e-12,
    )
    assert metrics.get("iterations") == _Ramp.ITERATIONS
    assert metrics.get("optimal") is False
    assert metrics["program"] == metrics["solved"] == sizes
    tokenizer = Tokenizer.from_file(
        str(tmp_path / "runs/convextok/exp000/tokenizer.json"),
    )
    assert metrics["pieces"] == tokenizer.get_vocab_size()
    assert "missing" not in metrics


def test_run_directory_must_not_be_the_raw_shards(tmp_path: Path) -> None:
    """Without the guard, the run would fail later, on the directory existing."""
    config = _measured_config(tmp_path)
    config.base_dir = None
    config.working_dir = config.preparation.raw_dir
    with pytest.raises(ValueError, match="aliases protected input artifact"):
        config.make().run()


@pytest.mark.gpu_torch_cuda
def test_cuda_fit_reports_peak_device_memory(tmp_path: Path) -> None:
    config = _measured_config(tmp_path)
    config.preparation.device = "cuda"
    config.preparation.presolve = None
    config.preparation.solver = _Zeros.Config()
    config.make().run()

    metrics = _read_metrics(tmp_path / "runs/convextok/exp000")
    assert convert(metrics["peak_gpu_bytes"], int) > 0


def test_existing_run_directory_fails_before_fitting(tmp_path: Path) -> None:
    """Fitting first would raise FileNotFoundError on the absent shards instead."""
    config = _measured_config(tmp_path)
    config.preparation.raw_dir = tmp_path / "absent"
    run_dir = tmp_path / "runs/convextok/exp000"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        config.make().run()
    assert list(run_dir.iterdir()) == []


def test_run_directory_comes_from_study_and_experiment(tmp_path: Path) -> None:
    config = MeasuredFit.Config()
    config.study_name = "convextok"
    config.experiment_name = "exp007"
    config.base_dir = tmp_path
    finalized = config.copy_tree().finalize()
    assert finalized.working_dir == tmp_path / "runs/convextok/exp007"
    assert finalized.preparation.working_dir == finalized.working_dir


def _measured_config(tmp_path: Path) -> MeasuredFit.Config:
    """Point a CPU measured fit at the fixture corpus, written as one raw shard."""
    config = MeasuredFit.Config()
    config.study_name = "convextok"
    config.experiment_name = "exp000"
    config.base_dir = tmp_path
    config.preparation = fixture_config(tmp_path)
    return config


def _read_metrics(run_dir: Path) -> dict[str, object]:
    return parse((run_dir / "metrics.json").read_text(), dict[str, object])


def _npz(name: str) -> np.lib.npyio.NpzFile:
    loaded = cast(object, np.load(_CWD / "testdata" / name))
    assert isinstance(loaded, np.lib.npyio.NpzFile)
    return loaded


def _read_json(name: str) -> dict[str, object]:
    return parse((_CWD / "testdata" / name).read_text(), dict[str, object])


def _golden_matrix(
    golden: np.lib.npyio.NpzFile,
    *,
    name: str,
    rows: int,
) -> Tensor:
    """Upstream's SciPy CSR block, dense, independent of ``build_program``."""
    indptr, indices, data = (
        torch.from_numpy(golden[f"{name}_{part}"])
        for part in ("indptr", "indices", "data")
    )
    columns = int(torch.from_numpy(golden["A_eq_shape"])[1])
    dense = torch.zeros(rows, columns, dtype=data.dtype)
    owner = torch.repeat_interleave(torch.arange(rows), torch.diff(indptr))
    dense[owner, indices.to(torch.int64)] = data
    return dense


def _ramp(size: int) -> Tensor:
    # Float32, so a measurement that skips the cast to the objective's dtype fails.
    return torch.linspace(-0.5, 1.5, size, dtype=torch.float32)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Solution:
    primal: Tensor


class _Zeros:
    """A stand-in solver whose answer is every variable at zero."""

    class Config(Fig["_Zeros"]):
        """No settings."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, program: LinearProgram, /) -> _Solution:
        return _Solution(primal=torch.zeros(program.num_columns, dtype=torch.float64))


@dataclass(frozen=True, slots=True, kw_only=True)
class _ReportedSolution:
    primal: Tensor
    iterations: int
    optimal: bool


class _Ramp:
    """A stand-in solver that answers with a ramp and reports how it stopped."""

    ITERATIONS: Final = 3

    class Config(Fig["_Ramp"]):
        """No settings."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, program: LinearProgram, /) -> _ReportedSolution:
        return _ReportedSolution(
            primal=_ramp(program.num_columns),
            iterations=self.ITERATIONS,
            optimal=False,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
