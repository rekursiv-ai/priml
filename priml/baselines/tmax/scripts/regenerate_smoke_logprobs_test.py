"""Tests for regenerating the smoke fixture's behavior logprobs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import json
import shutil

import pytest
import torch

from priml.baselines.tmax.scripts import regenerate_smoke_logprobs


if TYPE_CHECKING:
    from priml.baselines.tmax.train_step import TMaxDPPOTrainStep


_CWD: Final = Path(__file__).resolve().parent
_FIXTURE: Final = _CWD.parent / "testdata" / "smoke_rollout.jsonl"


def _copy(tmp_path: Path) -> Path:
    fixture = tmp_path / _FIXTURE.name
    shutil.copyfile(_FIXTURE, fixture)
    return fixture


def test_regeneration_reproduces_the_committed_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The committed logprobs are exactly what the seeded smoke model scores."""
    fixture = _copy(tmp_path)
    monkeypatch.setattr("sys.argv", ["regen", "--fixture", str(fixture)])
    assert regenerate_smoke_logprobs.main() == 0
    assert fixture.read_text(encoding="utf-8") == _FIXTURE.read_text(encoding="utf-8")
    assert "verified through the training step" in capsys.readouterr().out


def test_regeneration_refuses_a_fixture_whose_ratio_drifts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _copy(tmp_path)
    monkeypatch.setattr("sys.argv", ["regen", "--fixture", str(fixture)])

    def drifted(*args: object) -> float:
        del args
        return 2.0

    monkeypatch.setattr(regenerate_smoke_logprobs, "_ratio_mean", drifted)
    with pytest.raises(SystemExit, match="differs from 1"):
        regenerate_smoke_logprobs.main()


def test_a_packed_row_must_carry_the_records_tokens() -> None:
    row: dict[str, object] = {
        "rollout_sample_ids": torch.tensor([0, 0]),
        "response_mask": torch.tensor([0, 1]),
        "query_responses": torch.tensor([5, 6]),
    }
    record: dict[str, object] = {"sample_idx": 0, "response_tokens": [7]}
    with pytest.raises(ValueError, match="disagree with the fixture record"):
        regenerate_smoke_logprobs._record_positions(row, record)


def test_a_packed_row_field_must_be_a_tensor() -> None:
    with pytest.raises(TypeError, match=r"must be a torch\.Tensor"):
        regenerate_smoke_logprobs._tensor({"x": [1]}, "x")


class _SilentStep:
    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        return batch

    def train_loss(self, **batch: object) -> dict[str, object]:
        return {"loss": batch}


def test_ratio_mean_requires_step_metrics() -> None:
    step = cast("TMaxDPPOTrainStep", _SilentStep())
    with pytest.raises(ValueError, match="reported no metrics"):
        regenerate_smoke_logprobs._ratio_mean(step, {})


def test_records_are_read_in_loader_order(tmp_path: Path) -> None:
    fixture = tmp_path / "f.jsonl"
    fixture.write_text(
        "\n".join(
            json.dumps({"step": 0, "prompt_idx": 0, "sample_idx": index})
            for index in (1, 0)
        )
        + "\n\n",
        encoding="utf-8",
    )
    records = regenerate_smoke_logprobs._read_records(fixture)
    assert [record["sample_idx"] for record in records] == [0, 1]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
