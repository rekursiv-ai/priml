"""Regenerate the smoke fixture's logprobs with the smoke model.

The response tokens and rewards come from two published rollout slices. This
script scores those fixed tokens with the seeded smoke model and writes the
scores to the fixture. The resulting update has a mean importance ratio near 1.

Run from the repository root:

    uv --quiet run --frozen -- python \
        priml/baselines/tmax/scripts/regenerate_smoke_logprobs.py

The script then reloads the fixture and checks the reported mean ratio.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import json

from torch import Tensor

import torch

from priml.baselines.tmax.experiments import exp_smoke
from priml.baselines.tmax.scoring import LogitModel, response_logprobs
from priml.baselines.tmax.train_step import TMaxDPPOTrainStep, _row_metrics
from priml.math.seed import set_seed_local


if TYPE_CHECKING:
    from priml.baselines.tmax.experiments import TMaxTrainLoop

FIXTURE = Path(__file__).resolve().parents[1] / "testdata" / "smoke_rollout.jsonl"
RATIO_TOLERANCE = 1e-5


def _tensor(row: dict[str, object], name: str) -> Tensor:
    """Read one packed row's tensor field."""
    value = row[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    return value


def _read_records() -> list[dict[str, object]]:
    """Read fixture records in the order required by the loader."""
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    records = [
        cast(dict[str, object], json.loads(line)) for line in lines if line.strip()
    ]
    return sorted(
        records,
        key=lambda record: (
            int(cast(int, record["step"])),
            int(cast(int, record["prompt_idx"])),
            int(cast(int, record["sample_idx"])),
        ),
    )


def _packed_update(
    config: TMaxTrainLoop,
) -> tuple[TMaxDPPOTrainStep, dict[str, object], list[dict[str, object]]]:
    """Build the seeded smoke step and its only batch."""
    # Match the training loop's model initialization.
    set_seed_local(config.seed)
    # configgle's generic return type is opaque to ty.
    step: TMaxDPPOTrainStep = config.step.make()  # ty: ignore[unsound-assignment]
    dataset = config.dataset.make()
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
    sample = int(cast(int, record["sample_idx"]))
    positions = (_tensor(row, "rollout_sample_ids") == sample) & (
        _tensor(row, "response_mask") != 0
    )
    if not bool(positions.any()):
        raise LookupError(f"sample {sample} is missing from the packed row.")
    tokens = _tensor(row, "query_responses")[positions].tolist()
    if tokens != record["response_tokens"]:
        raise ValueError(
            f"Packed response tokens {tokens} disagree with the fixture record "
            f"{record['response_tokens']}.",
        )
    return positions


def _ratio_mean(step: TMaxDPPOTrainStep, batch: dict[str, object]) -> float:
    """Return the update's mean importance ratio."""
    metrics = _row_metrics(step.train_loss(**step.preprocess_batch(batch)))
    return float(metrics["ratio_mean"])


def main() -> None:
    """Rewrite the fixture logprobs and verify the mean ratio."""
    records = _read_records()
    # Write the order required by the loader.
    FIXTURE.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )
    config = exp_smoke()
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
    FIXTURE.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )

    # Reload the fixture through the same path used by training.
    check_step, check_batch, _ = _packed_update(config)
    ratio = _ratio_mean(check_step, check_batch)
    if abs(ratio - 1.0) > RATIO_TOLERANCE:
        raise SystemExit(f"Fixture ratio_mean differs from 1: {ratio!r}")
    print(f"ratio_mean={ratio!r} (verified through the training step)")


if __name__ == "__main__":
    main()
