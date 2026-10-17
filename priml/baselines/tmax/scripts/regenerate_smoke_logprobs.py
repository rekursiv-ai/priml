#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Regenerate the smoke fixture's logprobs with the smoke model.

The response tokens and rewards come from two published rollout slices. This
script scores those fixed tokens with the seeded smoke model and writes the
scores to the fixture. The resulting update has a mean importance ratio near 1.

Run from the repository root:

    uv --quiet run --frozen -- python \
        priml/baselines/tmax/scripts/regenerate_smoke_logprobs.py

The script then reloads the fixture and checks the reported mean ratio.
'''
# fmt: on


from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import json

from torch import Tensor
from treekle import from_plain, json_loads

import torch

from priml.baselines.tmax.data import TMaxRolloutData
from priml.baselines.tmax.experiments import exp_smoke
from priml.baselines.tmax.scoring import LogitModel, response_logprobs
from priml.baselines.tmax.train_step import TMaxDPPOTrainStep
from priml.math.seed import set_seed_local


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from priml.baselines.tmax.experiments import TMaxTrainLoop

RATIO_TOLERANCE: Final = 1e-5


def main() -> int:
    """Rewrite the fixture logprobs and verify the mean ratio.

    Returns:
      exit_code: Zero when the rewritten fixture's mean ratio is 1.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    fixture = cast("_Flags", parser.parse_args()).fixture
    records = _read_records(fixture)
    # Write the order required by the loader.
    _write_records(fixture, records)
    config = exp_smoke()
    config.dataset.base_dir = fixture.parent
    config.dataset.working_dir = fixture.name
    step, _, rows = _packed_update(config)
    for row in rows:
        # The scorer omits the first token, so shift the mask by one.
        scores = _scores(step, row)
        for record in records:
            positions = _record_positions(row, record)
            logprobs = scores[positions[1:]].tolist()
            record["logprobs"] = logprobs
            print(
                f"sample_idx={record['sample_idx']} "
                f"response_len={len(logprobs)} first_logprob={logprobs[0]:.6f}",
            )
    _write_records(fixture, records)

    # Reload the fixture through the same path used by training.
    check_step, check_batch, _ = _packed_update(config)
    ratio = _ratio_mean(check_step, check_batch)
    if abs(ratio - 1.0) > RATIO_TOLERANCE:
        raise SystemExit(f"Fixture ratio_mean differs from 1: {ratio!r}")
    print(f"ratio_mean={ratio!r} (verified through the training step)")
    return 0


class _Flags(Protocol):
    """Parsed command-line flags."""

    fixture: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fixture",
        type=Path,
        default=_CWD.parent / "testdata" / "smoke_rollout.jsonl",
        help="Smoke rollout fixture to rewrite in place.",
    )


def _write_records(fixture: Path, records: list[dict[str, object]]) -> None:
    fixture.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )


def _tensor(row: dict[str, object], name: str) -> Tensor:
    """Read one packed row's tensor field."""
    value = row[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    return value


def _read_records(fixture: Path) -> list[dict[str, object]]:
    """Read fixture records in the order required by the loader."""
    lines = fixture.read_text(encoding="utf-8").splitlines()
    records = [json_loads(line, dict[str, object]) for line in lines if line.strip()]
    return sorted(
        records,
        key=lambda record: (
            from_plain(record["step"], int),
            from_plain(record["prompt_idx"], int),
            from_plain(record["sample_idx"], int),
        ),
    )


def _packed_update(
    config: TMaxTrainLoop.Config,
) -> tuple[TMaxDPPOTrainStep, dict[str, object], list[dict[str, object]]]:
    """Build the seeded smoke step and its only batch."""
    # Match the training loop's model initialization.
    set_seed_local(config.seed)
    step = config.step.make()
    assert isinstance(step, TMaxDPPOTrainStep)
    dataset = config.dataset.make()
    assert isinstance(dataset, TMaxRolloutData)
    batch = next(iter(dataset.train_dataloader()))
    rows = cast(list[dict[str, object]], step.preprocess_batch(batch)["rows"])
    return step, batch, rows


def _scores(step: TMaxDPPOTrainStep, row: dict[str, object]) -> Tensor:
    """Score one packed row with the smoke model."""
    with torch.no_grad():
        scored = response_logprobs(
            cast(LogitModel, step),
            tokens=_tensor(row, "query_responses"),
            segments=_tensor(row, "attention_mask"),
            positions=_tensor(row, "position_ids"),
            pad_token_id=step.pad_token_id,
            temperature=step.config.temperature,
        )
    return scored.squeeze(0)


def _record_positions(row: dict[str, object], record: dict[str, object]) -> Tensor:
    """Find this record's response tokens in a packed row."""
    sample = from_plain(record["sample_idx"], int)
    positions = (_tensor(row, "rollout_sample_ids") == sample) & (
        _tensor(row, "response_mask") != 0
    )
    tokens = _tensor(row, "query_responses")[positions].tolist()
    if tokens != record["response_tokens"]:
        raise ValueError(
            f"Packed response tokens {tokens} disagree with the fixture record "
            f"{record['response_tokens']}.",
        )
    return positions


def _ratio_mean(step: TMaxDPPOTrainStep, batch: dict[str, object]) -> float:
    """Return the update's mean importance ratio."""
    metrics = step.train_loss(**step.preprocess_batch(batch)).get("metrics")
    if metrics is None:
        raise ValueError("The training step reported no metrics.")
    return float(metrics["ratio_mean"])


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
