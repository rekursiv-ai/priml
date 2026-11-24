"""Tests for regenerating the smoke fixture's behavior logprobs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import argparse
import json
import shutil

from treekle import ReadError, from_plain, json_loads

import pytest
import torch

from priml.baselines.tmax.scripts import regenerate_smoke_logprobs


if TYPE_CHECKING:
    from priml.baselines.tmax.train_step import TMaxDPPOTrainStep


_CWD: Final[Path] = Path(__file__).resolve().parent
_FIXTURE: Final = _CWD.parent / "testdata" / "smoke_rollout.jsonl"


def _copy(tmp_path: Path) -> Path:
    fixture = tmp_path / _FIXTURE.name
    shutil.copyfile(_FIXTURE, fixture)
    return fixture


def _committed_records() -> list[dict[str, object]]:
    lines = _FIXTURE.read_text(encoding="utf-8").splitlines()
    return [json_loads(line, dict[str, object]) for line in lines if line.strip()]


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
    out = capsys.readouterr().out
    assert "verified through the training step" in out
    for record in _committed_records():
        logprobs = from_plain(record["logprobs"], list[float])
        assert (
            f"sample_idx={record['sample_idx']} "
            f"response_len={len(logprobs)} first_logprob={logprobs[0]:.6f}"
        ) in out


def test_regeneration_refuses_a_fixture_whose_ratio_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(regenerate_smoke_logprobs, "RATIO_TOLERANCE", 0.5)
    regenerate_smoke_logprobs._check_ratio(1.5)
    with pytest.raises(SystemExit, match="differs from 1"):
        regenerate_smoke_logprobs._check_ratio(1.75)


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
    with pytest.raises(TypeError, match=r"must be a torch\.Tensor, got list\."):
        regenerate_smoke_logprobs._tensor({"x": [1]}, "x")


def test_scores_pass_the_step_settings_to_the_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def scorer(model: object, **settings: object) -> torch.Tensor:
        del model
        calls.append(settings)
        return torch.zeros(2, 3)

    monkeypatch.setattr(regenerate_smoke_logprobs, "response_logprobs", scorer)
    step = cast(
        "TMaxDPPOTrainStep",
        SimpleNamespace(pad_token_id=7, config=SimpleNamespace(temperature=0.5)),
    )
    row: dict[str, object] = {
        name: torch.tensor([1, 2])
        for name in ("query_responses", "attention_mask", "position_ids")
    }
    regenerate_smoke_logprobs._scores(step, row)
    assert (calls[0]["pad_token_id"], calls[0]["temperature"]) == (7, 0.5)


def test_the_fixture_defaults_to_the_committed_rollout() -> None:
    parser = argparse.ArgumentParser()
    regenerate_smoke_logprobs._add_arguments(parser)
    assert vars(parser.parse_args([]))["fixture"] == _FIXTURE


def test_help_shows_the_regeneration_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["regen", "--help"])
    with pytest.raises(SystemExit) as exited:
        regenerate_smoke_logprobs.main()
    assert exited.value.code == 0
    text = capsys.readouterr().out
    assert (
        "The response tokens and rewards come from two published rollout slices."
        in text
    )
    assert "--project" not in text


def test_help_works_without_a_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(regenerate_smoke_logprobs, "__doc__", None)
    monkeypatch.setattr("sys.argv", ["regen", "--help"])
    with pytest.raises(SystemExit) as exited:
        regenerate_smoke_logprobs.main()
    assert exited.value.code == 0
    assert "usage: regen" in capsys.readouterr().out


class _SilentStep:
    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        return batch

    def train_loss(self, **batch: object) -> dict[str, object]:
        return {"loss": batch}


def test_ratio_mean_requires_step_metrics() -> None:
    step = cast("TMaxDPPOTrainStep", _SilentStep())
    with pytest.raises(ValueError, match=r"^The training step reported no metrics\.$"):
        regenerate_smoke_logprobs._ratio_mean(step, {})


def test_records_are_written_one_sorted_utf8_line_each(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "f.jsonl"
    real_write_text = Path.write_text
    encodings: list[str | None] = []

    def spy(self: Path, data: str, *, encoding: str | None = None) -> int:
        encodings.append(encoding)
        return real_write_text(self, data, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", spy)
    regenerate_smoke_logprobs._write_records(fixture, [{"b": "é", "a": 1}, {"c": 2}])
    assert encodings == ["utf-8"]
    assert fixture.read_bytes() == b'{"a": 1, "b": "\\u00e9"}\n{"c": 2}\n'


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


def test_a_record_must_be_a_json_object(tmp_path: Path) -> None:
    fixture = tmp_path / "f.jsonl"
    fixture.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ReadError):
        regenerate_smoke_logprobs._read_records(fixture)


def test_records_are_read_as_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "f.jsonl"
    fixture.write_text(
        '{"step": 0, "prompt_idx": 0, "sample_idx": 0}\n',
        encoding="utf-8",
    )
    real_read_text = Path.read_text
    encodings: list[str | None] = []

    def spy(self: Path, *, encoding: str | None = None) -> str:
        encodings.append(encoding)
        return real_read_text(self, encoding=encoding)

    monkeypatch.setattr(Path, "read_text", spy)
    regenerate_smoke_logprobs._read_records(fixture)
    assert encodings == ["utf-8"]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
