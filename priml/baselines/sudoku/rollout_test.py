from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor

import pytest
import torch

from priml.baselines.sudoku.rollout import (
    LearnedCheckpointRollout,
    family_rollout,
    learned_checkpoint_rollout_rows,
    learned_persistence_scores,
    learned_persistent_rollout_rows,
    segmented_rollout_rows,
)
from priml.baselines.sudoku.testing import tiny_recursion


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.sudoku import latent_recursion


def test_rollouts_return_row_logits_and_checkpoint_scores() -> None:
    typed_model = tiny_recursion()
    boards = torch.full((2, 81), 1, dtype=torch.long)
    boards[:, 0] = 2
    kwargs: dict[str, Tensor] = {}
    logits, scores = segmented_rollout_rows(
        typed_model,
        kwargs,
        10,
        boards,
        torch.tensor([0, 1]),
        continue_threshold=1.0,
        early_exit_at_q8=True,
    )
    assert logits.shape == (2, 81, 11)
    assert scores.shape == (2,)
    checkpoint = learned_checkpoint_rollout_rows(
        typed_model,
        kwargs,
        3,
        boards,
        torch.tensor([0, 1]),
        checkpoints=(2, 3),
    )
    assert checkpoint.q_scores.shape == (2, 2)
    assert learned_persistence_scores(
        checkpoint,
        require_prediction_stability=False,
    ).shape == (2,)


def test_checkpoint_rollout_needs_steps_and_persistence_rejects_drift() -> None:
    media = torch.full((2, 81), 1, dtype=torch.long)
    media[:, 0] = 2
    logits = torch.zeros(2, 81, 11)
    logits[..., 2] = 4
    preds = torch.full((2, 81), 2, dtype=torch.long)
    with pytest.raises(ValueError, match="at least one"):
        learned_checkpoint_rollout_rows(
            tiny_recursion(),
            {},
            2,
            media,
            torch.arange(2),
            checkpoints=(),
        )
    rollout = LearnedCheckpointRollout(
        logits=logits,
        q_scores=torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]),
        predictions=preds.unsqueeze(1).expand(-1, 3, -1),
    )
    assert torch.equal(
        learned_persistence_scores(rollout, require_prediction_stability=True),
        torch.tensor([2.0, 2.0]),
    )
    unstable = LearnedCheckpointRollout(
        logits=rollout.logits,
        q_scores=rollout.q_scores,
        predictions=rollout.predictions.clone(),
    )
    unstable.predictions[0, 0, 0] = 3
    assert torch.isneginf(
        learned_persistence_scores(unstable, require_prediction_stability=True)[0],
    )


def test_segmented_rollout_continues_confident_rows_past_q8() -> None:
    model = tiny_recursion()
    boards = torch.full((3, 81), 1, dtype=torch.long)
    boards[:, 0] = 2
    rows = torch.arange(3)
    with pytest.raises(ValueError, match="max_steps"):
        segmented_rollout_rows(
            model,
            {},
            0,
            boards,
            rows,
            continue_threshold=0.0,
            early_exit_at_q8=False,
        )
    one_step_logits, one_step_scores = segmented_rollout_rows(
        model,
        {},
        1,
        boards,
        rows,
        continue_threshold=0.0,
        early_exit_at_q8=False,
    )
    assert one_step_logits.shape == (3, 81, 11)
    assert one_step_scores.shape == (3,)
    logits, scores = segmented_rollout_rows(
        model,
        {},
        10,
        boards,
        rows,
        continue_threshold=0.0,
        early_exit_at_q8=False,
    )
    assert logits.shape == (3, 81, 11)
    assert scores.shape == (3,)


def test_segmented_rollout_without_q8_gate_continues_all_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boards = torch.tensor([[2, 1], [1, 3]])
    arange = cast("Callable[..., Tensor]", torch.arange)
    devices: list[object] = []

    def record_arange(*args: object, **kwargs: object) -> Tensor:
        devices.append(kwargs.get("device"))
        return arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", record_arange)

    class RecordingModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
            return torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)

        def act_step(
            self,
            input_ids: Tensor,
            z_slow: Tensor,
            z_fast: Tensor,
            *,
            puzzle_identifiers: Tensor | None = None,
            feedback_ids: Tensor,
        ) -> dict[str, Tensor]:
            del puzzle_identifiers, feedback_ids
            self.batch_sizes.append(input_ids.shape[0])
            logits = torch.zeros(len(input_ids), 2, 4)
            logits[..., 2] = 1
            return {
                "logits": logits,
                "q_halt": torch.tensor([-1.0, -2.0])[: len(input_ids)],
                "z_slow": z_slow + 1,
                "z_fast": z_fast + 1,
            }

    model = RecordingModel()
    logits, scores = segmented_rollout_rows(
        cast("latent_recursion.LatentRecursion", model),
        {},
        10,
        boards,
        torch.tensor([1, 0]),
        continue_threshold=0.0,
        early_exit_at_q8=False,
    )

    assert model.batch_sizes == [2] * 10
    assert devices == [boards.device]
    assert logits.device == scores.device == boards.device
    assert scores.tolist() == [-1.0, -2.0]


def test_checkpoint_rollout_rejects_disordered_or_out_of_range_steps() -> None:
    model = tiny_recursion()
    boards = torch.full((2, 81), 1, dtype=torch.long)
    for checkpoints, message in (((3, 2), "strictly increasing"), ((2, 5), "within")):
        with pytest.raises(ValueError, match=message):
            learned_checkpoint_rollout_rows(
                model,
                {},
                3,
                boards,
                torch.arange(2),
                checkpoints=checkpoints,
            )
    rollout = LearnedCheckpointRollout(
        logits=torch.zeros(2, 81, 11),
        q_scores=torch.zeros(2, 3),
        predictions=torch.zeros(2, 4, 81),
    )
    with pytest.raises(ValueError, match="align by row and step"):
        learned_persistence_scores(rollout, require_prediction_stability=False)
    flat = LearnedCheckpointRollout(
        logits=torch.zeros(2, 81, 11),
        q_scores=torch.zeros(2),
        predictions=torch.zeros(2, 4, 81),
    )
    with pytest.raises(ValueError, match="ranks 2 and 3"):
        learned_persistence_scores(flat, require_prediction_stability=False)


def test_segmented_rollout_returns_q8_for_early_rows_and_continues_selected_rows() -> (
    None
):
    boards = torch.tensor([[2, 1, 1], [1, 1, 3]])
    rows = torch.tensor([1, 0])
    puzzle_ids = torch.tensor([17, 29, 41])

    class RecordingModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.steps = 0
            self.calls: list[tuple[Tensor, Tensor, Tensor]] = []

        def init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
            return torch.zeros(batch_size, 3), torch.zeros(batch_size, 4)

        def act_step(
            self,
            input_ids: Tensor,
            z_slow: Tensor,
            z_fast: Tensor,
            *,
            puzzle_identifiers: Tensor | None = None,
            feedback_ids: Tensor,
        ) -> dict[str, Tensor]:
            assert puzzle_identifiers is not None
            self.steps += 1
            self.calls.append(
                (input_ids.clone(), feedback_ids.clone(), puzzle_identifiers.clone()),
            )
            logits = torch.zeros(len(input_ids), 3, 5)
            logits[..., self.steps % 5] = 1
            return {
                "logits": logits,
                "q_halt": torch.tensor([0.0, -1.0])[: len(input_ids)],
                "z_slow": z_slow + 1,
                "z_fast": z_fast + 1,
            }

    model = RecordingModel()
    logits, scores = segmented_rollout_rows(
        cast("latent_recursion.LatentRecursion", model),
        {"puzzle_identifiers": puzzle_ids},
        10,
        boards,
        rows,
        continue_threshold=0.0,
        early_exit_at_q8=True,
    )
    assert model.steps == 10
    assert [call[0].shape[0] for call in model.calls] == [2] * 8 + [1] * 2
    assert model.calls[0][2].tolist() == [29, 17]
    assert model.calls[8][2].tolist() == [29]
    assert torch.equal(model.calls[0][1], boards)
    assert torch.equal(model.calls[8][1], torch.tensor([[2, 3, 3]]))
    assert logits.argmax(dim=-1).tolist() == [[0, 0, 0], [3, 3, 3]]
    assert scores.tolist() == [0.0, -1.0]


def test_segmented_rollout_keeps_q8_boundary_and_gathers_live_rows() -> None:
    boards = torch.tensor([[10, 1, 1], [11, 1, 3], [1, 4, 1]])
    puzzle_ids = torch.tensor([10, 20, 30, 40])

    class RecordingModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.calls: list[tuple[Tensor, Tensor, Tensor]] = []

        def init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
            return torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)

        def act_step(
            self,
            input_ids: Tensor,
            z_slow: Tensor,
            z_fast: Tensor,
            *,
            puzzle_identifiers: Tensor | None = None,
            feedback_ids: Tensor,
        ) -> dict[str, Tensor]:
            assert puzzle_identifiers is not None
            self.calls.append(
                (input_ids.clone(), feedback_ids.clone(), puzzle_identifiers.clone()),
            )
            step = int(z_slow[0, 0]) + 1
            logits = torch.zeros(len(input_ids), 3, 5)
            logits[..., step % 5] = 1
            q_by_id = {10: -1.0, 20: -1.0, 30: 0.0, 40: 1.0}
            q_halt = torch.tensor(
                [q_by_id[int(identifier)] for identifier in puzzle_identifiers],
            )
            return {
                "logits": logits,
                "q_halt": q_halt,
                "z_slow": z_slow + 1,
                "z_fast": z_fast + 1,
            }

    model = RecordingModel()
    logits, scores = segmented_rollout_rows(
        cast("latent_recursion.LatentRecursion", model),
        {"puzzle_identifiers": puzzle_ids},
        10,
        boards,
        torch.tensor([2, 3, 0]),
        continue_threshold=0.0,
        early_exit_at_q8=True,
    )

    assert len(model.calls) == 10
    assert [call[0].shape[0] for call in model.calls] == [3] * 8 + [2] * 2
    assert model.calls[0][2].tolist() == [30, 40, 10]
    assert model.calls[8][2].tolist() == [30, 40]
    assert torch.equal(model.calls[0][1], boards)
    assert torch.equal(model.calls[8][1], torch.tensor([[10, 3, 3], [3, 3, 3]]))
    assert logits.argmax(dim=-1).tolist() == [[0, 0, 0], [0, 0, 0], [3, 3, 3]]
    assert scores.tolist() == [0.0, 1.0, -1.0]


def test_segmented_rollout_feedback_uses_per_cell_predictions() -> None:
    boards = torch.ones((2, 3), dtype=torch.int64)

    class RecordingModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.feedback: list[Tensor] = []

        def init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
            return torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)

        def act_step(
            self,
            input_ids: Tensor,
            z_slow: Tensor,
            z_fast: Tensor,
            *,
            puzzle_identifiers: Tensor | None = None,
            feedback_ids: Tensor,
        ) -> dict[str, Tensor]:
            del puzzle_identifiers
            self.feedback.append(feedback_ids.clone())
            logits = torch.zeros(len(input_ids), 3, 4)
            logits[:, 0, 1] = 1
            logits[:, 1, 2] = 2
            logits[:, 2, 3] = 3
            return {
                "logits": logits,
                "q_halt": torch.ones(len(input_ids)),
                "z_slow": z_slow + 1,
                "z_fast": z_fast + 1,
            }

    model = RecordingModel()
    segmented_rollout_rows(
        cast("latent_recursion.LatentRecursion", model),
        {},
        10,
        boards,
        torch.tensor([0, 1]),
        continue_threshold=0.0,
        early_exit_at_q8=True,
    )

    assert len(model.feedback) == 10
    assert torch.equal(model.feedback[9], torch.tensor([[1, 2, 3], [1, 2, 3]]))


def test_family_rollout_reads_q8_at_the_cap_and_needs_a_step() -> None:
    """Below eight ACT steps the UNSAT probe reads the last step's halt logit."""
    torch.manual_seed(0)
    model = tiny_recursion()
    boards = torch.full((2, 81), 1, dtype=torch.long)
    boards[:, 0] = 2
    out, q_halt8 = family_rollout(model, boards, step_kwargs={}, max_steps=2)
    assert out["logits"].shape == (2, 81, 11)
    assert torch.equal(q_halt8, out["q_halt"])
    with pytest.raises(ValueError, match="q_halt8"):
        family_rollout(model, boards, step_kwargs={}, max_steps=0)


def test_persistent_rollout_scores_the_weakest_checkpoint() -> None:
    torch.manual_seed(0)
    model = tiny_recursion()
    boards = torch.full((2, 81), 1, dtype=torch.long)
    boards[:, 0] = 2
    rows = torch.arange(2)
    checkpoint = learned_checkpoint_rollout_rows(
        model,
        {},
        3,
        boards,
        rows,
        checkpoints=(2, 3),
    )
    logits, scores = learned_persistent_rollout_rows(
        model,
        {},
        3,
        boards,
        rows,
        checkpoints=(2, 3),
        require_prediction_stability=False,
    )
    assert torch.equal(logits, checkpoint.logits)
    assert torch.equal(scores, checkpoint.q_scores.min(dim=1).values)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
