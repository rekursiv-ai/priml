"""Regenerate the smoke fixture's logprobs with the smoke model.

The fixture's response tokens and rewards are hand-written Lean-shaped episodes:
one accepted proof and one rejected one, so the prompt group's centered advantage
is nonzero. This script scores those fixed tokens with the seeded smoke model and
writes the scores into the fixture, so the first update's importance ratio starts
near 1 rather than at whatever placeholder numbers implied.

Run from the repository root:

    uv --quiet run --frozen -- python \
        priml/baselines/openaimath/scripts/regenerate_smoke_logprobs.py

The script then reloads the fixture and checks the reported mean ratio.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import json

from torch import Tensor

import torch

from priml.baselines.openaimath.experiments import exp_smoke
from priml.baselines.tmax.scoring import LogitModel, response_logprobs
from priml.baselines.tmax.train_step import _row_metrics
from priml.math.seed import set_seed_local


if TYPE_CHECKING:
    from priml.baselines.openaimath.experiments import OpenMathTrainLoop
    from priml.baselines.tmax.train_step import TMaxDPPOTrainStep

FIXTURE: Final = (
    Path(__file__).resolve().parents[1] / "testdata" / "smoke_rollout.jsonl"
)
RATIO_TOLERANCE: Final = 1e-5


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
            cast(int, record["step"]),
            cast(int, record["prompt_idx"]),
            cast(int, record["sample_idx"]),
        ),
    )


def _packed_update(
    config: OpenMathTrainLoop,
) -> tuple[TMaxDPPOTrainStep, dict[str, object]]:
    """Build the seeded smoke step and its only batch."""
    # Match the training loop's model initialization.
    set_seed_local(config.seed)
    # configgle's generic return type is opaque to ty.
    step: TMaxDPPOTrainStep = config.step.make()  # ty: ignore[unsound-assignment]
    dataset = config.dataset.make()
    batch: dict[str, object] = next(  # ty: ignore[unsound-assignment]
        iter(dataset.train_dataloader()),
    )
    return step, batch


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
    sample = cast(int, record["sample_idx"])
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
    output = step.train_loss(**step.preprocess_batch(batch))
    return float(cast(float, _row_metrics(output)["ratio_mean"]))


def _gradient_norm(step: TMaxDPPOTrainStep, batch: dict[str, object]) -> float:
    """Return the update's total gradient norm.

    The loss itself is zero at ratio 1 with two centered advantages that cancel,
    so the loss value cannot show that the fixture trains anything. The gradient
    can: the two rollouts differ, so their gradients do not cancel.
    """
    output = step.train_loss(**step.preprocess_batch(batch))
    output["loss"].backward()
    total = 0.0
    for parameter in step.model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return total**0.5


def main() -> None:
    """Rewrite the fixture logprobs and verify the mean ratio."""
    records = _read_records()
    config = exp_smoke()
    step, batch = _packed_update(config)
    rows = cast(list[dict[str, object]], step.preprocess_batch(batch)["rows"])
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
    check_step, check_batch = _packed_update(config)
    ratio = _ratio_mean(check_step, check_batch)
    if abs(ratio - 1.0) > RATIO_TOLERANCE:
        raise SystemExit(f"Fixture ratio_mean differs from 1: {ratio!r}")
    print(f"ratio_mean={ratio!r} (verified through the training step)")

    gradient = _gradient_norm(check_step, check_batch)
    if not gradient > 0.0:
        raise SystemExit(f"Fixture produces no gradient: norm={gradient!r}")
    print(f"gradient_norm={gradient!r} (the update trains something)")


if __name__ == "__main__":
    main()
