from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import logging

from configgle import Fig
from torch import Tensor
from treekle import from_plain

import numpy as np
import pytest
import torch

from priml.baselines.sudoku import puzzle_data
from priml.baselines.sudoku.hypotheses import HypothesisRecursion
from priml.baselines.sudoku.puzzle_data import PuzzleDataset
from priml.baselines.sudoku.rollout import family_rollout
from priml.baselines.sudoku.runner import open_npz
from priml.baselines.sudoku.search_dump import learned_hps_output_width
from priml.baselines.sudoku.sieve import (
    FittedSieve,
    SieveEval,
    StaircaseSieveEval,
    _modal_tail,
    det_pass,
    qsel_grids,
)
from priml.baselines.sudoku.testing import (
    Harness,
    load_golden,
    metric_tensors,
    replay,
    sudoku_solution,
    tiny_recursion_config,
    write_dataset,
)
from priml.baselines.sudoku.verifier import VerifierAcceptor
from priml.baselines.sudoku.verifier_fit import VerifierFit
from priml.baselines.sudoku.views import NINE_VIEWS, View
from priml.data.arrays import load_array
from priml.model.swiglu import SwiGLU
from priml.testing.golden import mismatches
from priml.train.train_loop import EvalTimeLimitError


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.sudoku.routed_act_step import RoutedActTrainStep
    from priml.baselines.sudoku.search import HpsSearch


def test_sieve_configs_require_their_checkpoints() -> None:
    with pytest.raises(ValueError, match="checkpoint_path is required"):
        SieveEval.Config(experiment_name="x").make()
    with pytest.raises(ValueError, match="hypothesis_checkpoint_path is required"):
        StaircaseSieveEval.Config(experiment_name="x", checkpoint_path="g.pt").make()


def test_sieve_requires_a_name() -> None:
    with pytest.raises(ValueError, match="experiment_name is required"):
        SieveEval.Config(checkpoint_path="g.pt").make()


class _PaddingStep:
    """A det step whose every batch is padding, so it is never evaluated."""

    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        return {**batch, "valid_count": 0}

    def eval_loss(self, **_: object) -> dict[str, Tensor]:
        raise AssertionError("an all-padding batch must not be evaluated")


def test_det_pass_skips_padding_and_rejects_an_empty_split(tmp_path: Path) -> None:
    dataset = PuzzleDataset.Config(
        working_dir=write_dataset(tmp_path / "data"),
        batch_size=2,
        device="cpu",
    ).make()
    with pytest.raises(ValueError, match="det_pass found no valid eval rows"):
        det_pass(cast("RoutedActTrainStep", _PaddingStep()), dataset)


class _FirstCellNine:
    """A committee lock accepting exactly the grids whose first cell is 9."""

    class Config(Fig["_FirstCellNine"]):
        """No settings."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, preds: Tensor, media: Tensor) -> Tensor:
        del media
        return preds[:, 0] == 9


_LABELS: Final = torch.tensor([[9, 2, 3, 4], [5, 6, 7, 8], [6, 6, 6, 6]])
"""Solutions of the 3-puzzle toy population (4 cells, no givens)."""


class _StubRounds:
    """Round outputs per survivor: the det round solves puzzle 0 only."""

    def __init__(self, *, skew_media: bool = False) -> None:
        self.skew_media = skew_media
        self.calls: list[tuple[str, list[int]]] = []

    def det_round(self, model: object, checkpoint: object) -> tuple[Tensor, ...]:
        del model, checkpoint
        self.calls.append(("det_cond_halt", [0, 1, 2]))
        grids = torch.tensor([[9, 2, 3, 4], [5, 6, 7, 2], [6, 6, 6, 2]])
        return grids, torch.ones(3, 4, dtype=torch.int64), _LABELS

    def hps_round(
        self,
        view: View,
        search: HpsSearch.Config,
        survivors: Tensor,
        **_: object,
    ) -> tuple[Tensor, ...]:
        self.calls.append((view.name, from_plain(survivors.tolist(), list[int])))
        if view.name == "tail_escalated":
            assert search.search_candidates == 1
        # Puzzle 1 gets its solution in both later rounds, puzzle 2 never does.
        grids = torch.stack(
            [_LABELS[1] if p == 1 else _LABELS[0].flip(0) for p in survivors],
        )
        media = torch.ones(len(survivors), 4, dtype=torch.int64)
        media += int(self.skew_media)
        return grids, media, _LABELS[survivors]


def _stubbed_sieve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rounds: _StubRounds,
) -> SieveEval:
    cfg = SieveEval.Config()
    cfg.experiment_name = "sieve"
    cfg.checkpoint_path = "/runs/generator.pt"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.views = NINE_VIEWS[:1]
    cfg.tail_search = (1, 1, 1, 1)
    cfg.evaluation_count = 3
    cfg.acceptor = _FirstCellNine.Config()
    sieve = cfg.make()
    monkeypatch.setattr(sieve, "_det_round", rounds.det_round)
    monkeypatch.setattr(sieve, "_hps_round", rounds.hps_round)
    return sieve


def test_sieve_locks_per_round_and_takes_the_modal_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Puzzle 0 locks at det; 1 and 2 ride the ladder and the tail to the modal."""
    rounds = _StubRounds()
    sieve = _stubbed_sieve(tmp_path, monkeypatch, rounds)
    with caplog.at_level(logging.INFO):
        metrics = sieve.run("--extra")
    assert "--extra" in caplog.text
    assert rounds.calls == [
        ("det_cond_halt", [0, 1, 2]),
        (NINE_VIEWS[0].name, [1, 2]),
        ("tail_escalated", [1, 2]),
    ]
    assert metrics["eval/locked_total"] == 1
    assert metrics["eval/tail_modal_count"] == 2
    # Puzzle 1's ladder and tail grids outvote its det grid; puzzle 2 never solves.
    assert metrics["eval/n_correct"] == 2
    with open_npz(tmp_path / "runs" / "sieve" / "dumps" / "sieve_dump.npz") as dump:
        assert torch.equal(torch.from_numpy(dump["tail"]), torch.tensor([1, 2]))
        assert torch.equal(torch.from_numpy(dump["final_grids"][1]).long(), _LABELS[1])


def test_sieve_refuses_a_round_misaligned_with_round_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sieve = _stubbed_sieve(tmp_path, monkeypatch, _StubRounds(skew_media=True))
    with pytest.raises(RuntimeError, match="misaligned with round 0"):
        sieve.run()


def _qsel_sieve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: HypothesisRecursion,
) -> tuple[StaircaseSieveEval, list[bool]]:
    cfg = StaircaseSieveEval.Config()
    cfg.experiment_name = "staircase"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.checkpoint_path = "/runs/ladder.pt"
    cfg.hypothesis_checkpoint_path = "/runs/hypotheses.pt"
    cfg.dataset.working_dir = write_dataset(tmp_path / "data")
    cfg.dataset.batch_size = 2
    cfg.search.max_act_steps = 2
    cfg.search.dtype_autocast = None
    cfg.qsel_hypotheses = 1
    released: list[bool] = []

    def eval_model(*_: object) -> HypothesisRecursion:
        return model

    monkeypatch.setattr("priml.baselines.sudoku.sieve.eval_model", eval_model)
    monkeypatch.setattr(
        "priml.baselines.sudoku.sieve.release",
        lambda: released.append(True),
    )
    return cfg.make(), released


def test_qsel_round_selects_over_the_survivors_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    model = _tiny_hypotheses()
    sieve, released = _qsel_sieve(tmp_path, monkeypatch, model)
    grids, media, labels = sieve._qsel_round(torch.tensor([1]))
    test = tmp_path / "data" / "test"
    test_inputs = torch.from_numpy(load_array(test / "all__inputs.npy", np.generic))
    test_labels = torch.from_numpy(load_array(test / "all__labels.npy", np.generic))
    assert torch.equal(media, test_inputs[1:].long())
    assert torch.equal(labels, test_labels[1:].long())
    expected, _ = qsel_grids(
        model,
        test_inputs[1:].long(),
        puzzle_identifiers=None,
        hypotheses=1,
        seed=sieve.config.qsel_seed,
        max_steps=2,
    )
    assert torch.equal(grids, expected)
    assert released == [True]


def test_qsel_round_stops_at_its_deadline_and_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sieve, released = _qsel_sieve(tmp_path, monkeypatch, _tiny_hypotheses())
    sieve.config.max_round_eval_seconds = -1.0
    with pytest.raises(EvalTimeLimitError, match="hypothesis_qsel exceeded"):
        sieve._qsel_round(torch.tensor([0, 1]))
    assert released == [True]


def test_qsel_round_rejects_a_split_of_only_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sieve, _ = _qsel_sieve(tmp_path, monkeypatch, _tiny_hypotheses())
    loader = PuzzleDataset.eval_dataloader

    def padding_only(dataset: PuzzleDataset) -> list[dict[str, object]]:
        return [{**batch, "valid_count": 0} for batch in loader(dataset)]

    monkeypatch.setattr(PuzzleDataset, "eval_dataloader", padding_only)
    with pytest.raises(ValueError, match="hypothesis_qsel found no valid eval rows"):
        sieve._qsel_round(torch.tensor([0, 1]))


def test_sieve_hps_round_passes_survivors_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SieveEval.Config()
    config.checkpoint_path = "/runs/generator.pt"
    config.experiment_name = "sieve"
    config.base_dir = tmp_path
    config.runtime.device = "cpu"
    config.tail_search = (3, 2, 4, 30)
    sieve = config.make()
    observed: dict[str, object] = {}

    def eval_model(*_: object) -> object:
        return object()

    def search_pass(**kwargs: object) -> tuple[Tensor, Tensor, Tensor]:
        dataset = cast("PuzzleDataset", kwargs["dataset"])
        observed["indices"] = dataset.config.eval_instance_indices
        observed["search"] = kwargs["search"]
        return (
            torch.zeros(2, learned_hps_output_width(4)),
            torch.ones(2, 4),
            torch.ones(2, 4),
        )

    monkeypatch.setattr("priml.baselines.sudoku.sieve.eval_model", eval_model)
    monkeypatch.setattr("priml.baselines.sudoku.sieve.search_pass", search_pass)
    grids, media, labels = sieve._hps_round(
        NINE_VIEWS[0],
        config.search,
        torch.tensor([1, 3]),
        model=config.model,
        checkpoint=config.checkpoint_path,
    )
    assert observed == {"indices": (1, 3), "search": config.search}
    assert grids.shape == media.shape == labels.shape == (2, 4)
    assert grids.dtype == torch.int64


def _committee_median(
    h: Harness,
    acceptor: VerifierAcceptor.Config,
    members: tuple[Path, ...],
) -> float:
    """Median unanimity score of the committee over the test puzzles' labels."""
    probe = acceptor.copy_tree()
    probe.checkpoint_paths = members
    probe.device = "cpu"
    labels = torch.from_numpy(np.load(h.data / "test" / "all__labels.npy")).long()
    media = torch.from_numpy(np.load(h.data / "test" / "all__inputs.npy")).long()
    scores = probe.make().scores(labels, media)
    return float(scores.double().quantile(0.5))


def _record_staircase(h: Harness) -> dict[str, Tensor]:
    ladder = h.generator("gen_a", seed=0)
    staircase = h.hypothesis_generator("gen_hypotheses", seed=0)
    h.harvest(ladder)
    members = tuple(h.verifier(seed) for seed in range(3))
    cfg = StaircaseSieveEval.Config()
    cfg.experiment_name = "staircase"
    cfg.base_dir = h.scratch
    cfg.runtime.device = "cpu"
    cfg.checkpoint_path = ladder
    cfg.model = h.model()
    cfg.hypothesis_checkpoint_path = staircase
    cfg.hypothesis_model = h.hypothesis_model()
    cfg.qsel_hypotheses = 2
    cfg.dataset.working_dir = h.data
    cfg.dataset.batch_size = 2
    cfg.search = h.search()
    cfg.views = NINE_VIEWS[:1]
    cfg.tail_search = (1, 1, 1, 1)
    cfg.evaluation_count = 2
    cfg.verifier_checkpoints = members
    acceptor = VerifierAcceptor.Config()
    acceptor.model = h.verifier_model()
    # Three-step verifiers score every grid near zero, so the recipe's 0 locks
    # nothing. The threshold is the median committee score on the test
    # puzzles' givens-only boards, so each round both locks and defers.
    acceptor.threshold = _committee_median(h, acceptor, members)
    cfg.acceptor = acceptor
    metrics = cfg.make().run()
    rounds = cast("list[dict[str, object]]", metrics["eval/rounds"])
    per_round = {
        f"rounds/{i}/{key}": torch.tensor(cast("int", stat[key]), dtype=torch.int64)
        for i, stat in enumerate(rounds)
        for key in ("entering", "locked", "deferred")
    }
    return {**metric_tensors(metrics), **per_round, **h.dumps("staircase")}


def test_modal_tail_votes_over_every_round_a_survivor_reached() -> None:
    first = torch.full((3, 81), 2, dtype=torch.float32)
    second = torch.full((2, 81), 3, dtype=torch.float32)
    third = torch.full((2, 81), 3, dtype=torch.float32)
    collected = [
        (torch.tensor([0, 1, 2]), first),
        (torch.tensor([1, 2]), second),
        (torch.tensor([2, 1]), third),
    ]
    tail = _modal_tail(collected, torch.tensor([2, 1, 4]))
    assert tail.dtype == torch.float32
    assert torch.equal(tail[:2], torch.full((2, 81), 3, dtype=torch.float32))
    assert torch.equal(tail[2], torch.full((81,), 255, dtype=torch.float32))


def test_sieve_hps_round_scopes_search_to_the_survivors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SieveEval.Config()
    cfg.checkpoint_path = "/runs/generator.pt"
    cfg.experiment_name = "sieve"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.dataset.working_dir = write_dataset(tmp_path / "data")
    cfg.model = tiny_recursion_config()
    sieve = cfg.make()
    seen: dict[str, object] = {}
    model = object()
    released: list[bool] = []

    def eval_model(
        model_config: object,
        checkpoint_path: Path,
        device: torch.device,
    ) -> object:
        seen["eval_model"] = (model_config, checkpoint_path, device)
        return model

    monkeypatch.setattr("priml.baselines.sudoku.sieve.eval_model", eval_model)
    monkeypatch.setattr(
        "priml.baselines.sudoku.sieve.release",
        lambda: released.append(True),
    )

    def search_pass(**kwargs: object) -> tuple[Tensor, Tensor, Tensor]:
        dataset = cast("PuzzleDataset", kwargs["dataset"])
        seen["indices"] = dataset.config.eval_instance_indices
        seen["eval_num_instances"] = dataset.config.eval_num_instances
        # search_pass returns batch x grid rows x grid columns for this stub.
        media = torch.ones(2, 2, 3, dtype=torch.int64)
        labels = torch.full((2, 2, 3), 2, dtype=torch.int64)
        rows = torch.zeros(len(media), learned_hps_output_width(3))
        rows[:, -3:] = 7
        seen["view"] = kwargs["view"]
        seen["search"] = kwargs["search"]
        seen["deadline"] = kwargs["deadline_seconds"]
        seen["label"] = kwargs["label"]
        seen["model"] = kwargs["model"]
        seen["device"] = kwargs["device"]
        return rows, media, labels

    monkeypatch.setattr("priml.baselines.sudoku.sieve.search_pass", search_pass)
    view = NINE_VIEWS[0]
    search = cfg.search.copy_tree()
    grids, media, labels = sieve._hps_round(
        view,
        search,
        torch.tensor([0, 1]),
        model=sieve.config.model,
        checkpoint=sieve.config.checkpoint_path,
    )
    model_config, checkpoint_path, device = cast(
        "tuple[object, Path, torch.device]",
        seen.pop("eval_model"),
    )
    assert seen == {
        "indices": (0, 1),
        "eval_num_instances": None,
        "view": view,
        "search": search,
        "deadline": cfg.max_round_eval_seconds,
        "label": f"round {view.name}",
        "model": model,
        "device": sieve.device,
    }
    assert model_config is sieve.config.model
    assert checkpoint_path == sieve._path(cfg.checkpoint_path)
    assert device == sieve.device
    assert grids.shape == (2, 3)
    assert media.shape == labels.shape == (2, 2, 3)
    assert grids.dtype == torch.int64
    assert torch.equal(grids, torch.full((2, 3), 7, dtype=torch.int64))
    assert released == [True]
    with pytest.raises(ValueError, match="strictly ascending"):
        sieve._hps_round(
            view,
            search,
            torch.tensor([7, 2]),
            model=cfg.model,
            checkpoint=cfg.checkpoint_path,
        )


def test_sieve_hps_round_releases_model_after_search_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SieveEval.Config()
    cfg.checkpoint_path = "/runs/generator.pt"
    cfg.experiment_name = "sieve"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.dataset.working_dir = write_dataset(tmp_path / "data")
    cfg.model = tiny_recursion_config()
    sieve = cfg.make()
    released: list[bool] = []

    def eval_model(
        config: object,
        checkpoint_path: Path,
        device: torch.device,
    ) -> object:
        del config, checkpoint_path, device
        return object()

    monkeypatch.setattr("priml.baselines.sudoku.sieve.eval_model", eval_model)

    def fail_search(**_kwargs: object) -> tuple[Tensor, Tensor, Tensor]:
        raise RuntimeError("search failed")

    monkeypatch.setattr("priml.baselines.sudoku.sieve.search_pass", fail_search)
    monkeypatch.setattr(
        "priml.baselines.sudoku.sieve.release",
        lambda: released.append(True),
    )

    with pytest.raises(RuntimeError, match="search failed"):
        sieve._hps_round(
            NINE_VIEWS[0],
            cfg.search,
            torch.tensor([4]),
            model=cfg.model,
            checkpoint=cfg.checkpoint_path,
        )

    assert released == [True]


def test_sieve_tail_search_escalates_the_view_policy(tmp_path: Path) -> None:
    cfg = SieveEval.Config()
    cfg.checkpoint_path = "/runs/generator.pt"
    cfg.experiment_name = "sieve"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.tail_search = (5, 4, 3, 7)
    sieve = cfg.make()
    tail = sieve._tail_search_config()
    assert (
        tail.search_candidates,
        tail.search_depth,
        tail.search_cell_attempts,
        tail.search_budget,
    ) == (5, 4, 3, 7)
    assert tail.max_act_steps == cfg.search.max_act_steps
    assert sieve._round_names()[0] == "det_cond_halt"
    assert sieve._round_names()[-1] == "tail_escalated"
    cfg.evaluation_count = 0
    with pytest.raises(ValueError, match="evaluation_count"):
        cfg.make()


def test_sieve_tail_search_preserves_non_escalated_policy() -> None:
    config = SieveEval.Config()
    config.search.max_act_steps = 3
    config.search.acceptance_threshold = 2.5
    config.tail_search = (5, 4, 3, 7)
    sieve = object.__new__(SieveEval)
    sieve.config = config
    tail = sieve._tail_search_config()
    assert (
        tail.search_candidates,
        tail.search_depth,
        tail.search_cell_attempts,
        tail.search_budget,
    ) == (5, 4, 3, 7)
    assert tail.max_act_steps == 3
    assert tail.acceptance_threshold == 2.5
    assert tail.search_max_rows == config.search.search_max_rows


def test_modal_tail_preserves_dtype_sentinel_and_survivor_order() -> None:
    first = torch.full((3, 5), 2, dtype=torch.float32)
    second = torch.full((2, 5), 3, dtype=torch.float32)
    collected = [
        (torch.tensor([0, 1, 2]), first),
        (torch.tensor([1, 2]), second),
    ]
    result = _modal_tail(collected, torch.tensor([2, 1, 3]))

    assert result.dtype == torch.float32
    assert torch.equal(result[0], torch.full((5,), 2.0))
    assert torch.equal(result[1], torch.full((5,), 2.0))
    assert torch.equal(result[2], torch.full((5,), 255.0))


def _fixed_start(slow: Tensor, fast: Tensor, rows: int) -> tuple[Tensor, Tensor]:
    """Return fixed start latents, ignoring the requested row count."""
    del rows
    return slow, fast


def _tiny_hypotheses() -> HypothesisRecursion:
    """Build the smallest staircase generator whose alignment is well defined."""
    cfg = HypothesisRecursion.Config().update(tiny_recursion_config())
    cfg.channels_in = 12
    assert cfg.block is not None
    assert isinstance(cfg.block.ffn, SwiGLU.Config)
    cfg.block.ffn.round_to = 4
    cfg.block.ffn.channels_hidden = 4
    return cfg.make().eval()


def test_qsel_grids_keep_each_puzzles_highest_q_hypothesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection is the q-argmax over the deployed start and the fresh draws."""
    torch.manual_seed(0)
    model = _tiny_hypotheses()
    media = torch.stack([sudoku_solution(), sudoku_solution()])
    media[:, ::2] = 1
    slow, deployed = model.init_z(2)
    starts = [
        deployed,
        model.fresh_fast_states(2, seed=7)[:, 0],
        model.fresh_fast_states(2, seed=8)[:, 0],
    ]
    expected_grids: list[Tensor] = []
    expected_q: list[Tensor] = []
    with monkeypatch.context() as patch:
        for start in starts:
            patch.setattr(model, "init_z", partial(_fixed_start, slow, start))
            out, _ = family_rollout(
                model,
                media,
                step_kwargs={},
                max_steps=2,
            )
            expected_grids.append(out["logits"].argmax(dim=-1))
            expected_q.append(out["q_halt"].float())
    pick = torch.stack(expected_q).argmax(dim=0)
    grids, q_halt = qsel_grids(
        model,
        media,
        puzzle_identifiers=None,
        hypotheses=2,
        seed=7,
        max_steps=2,
    )
    assert torch.equal(q_halt, torch.stack(expected_q).max(dim=0).values)
    assert torch.equal(grids, torch.stack(expected_grids)[pick, torch.arange(2)])


def test_qsel_grids_without_fresh_draws_is_the_plain_rollout() -> None:
    torch.manual_seed(0)
    model = _tiny_hypotheses()
    media = torch.stack([sudoku_solution(), sudoku_solution()])
    media[:, 1::3] = 1
    out, _ = family_rollout(model, media, step_kwargs={}, max_steps=3)
    grids, q_halt = qsel_grids(
        model,
        media,
        puzzle_identifiers=None,
        hypotheses=0,
        seed=0,
        max_steps=3,
    )
    assert torch.equal(grids, out["logits"].argmax(dim=-1))
    assert torch.equal(q_halt, out["q_halt"].float())


def test_staircase_rounds_route_to_the_staircase_then_the_ladder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = StaircaseSieveEval.Config()
    cfg.experiment_name = "staircase"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.model = tiny_recursion_config()
    cfg.hypothesis_model = HypothesisRecursion.Config().update(tiny_recursion_config())
    cfg.checkpoint_path = "/runs/ladder.pt"
    cfg.hypothesis_checkpoint_path = "/runs/hypotheses.pt"
    sieve = cfg.make()
    calls: list[tuple[object, ...]] = []
    result = (torch.zeros(1), torch.zeros(1), torch.zeros(1))

    def det_round(model: object, checkpoint: object) -> tuple[Tensor, Tensor, Tensor]:
        calls.append(("det", model, checkpoint))
        return result

    def hps_round(
        view: View,
        search: object,
        survivors: Tensor,
        *,
        model: object,
        checkpoint: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del search, survivors
        calls.append(("hps", view.name, model, checkpoint))
        return result

    def qsel_round(survivors: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        calls.append(("qsel", survivors.tolist()))
        return result

    monkeypatch.setattr(sieve, "_det_round", det_round)
    monkeypatch.setattr(sieve, "_hps_round", hps_round)
    monkeypatch.setattr(sieve, "_qsel_round", qsel_round)
    survivors = torch.tensor([2, 5])
    for index in range(5):
        sieve._run_round(index, survivors)
    config = sieve.config
    assert calls == [
        ("det", config.hypothesis_model, config.hypothesis_checkpoint_path),
        ("qsel", [2, 5]),
        (
            "hps",
            "hypothesis_canonical_hps",
            config.hypothesis_model,
            config.hypothesis_checkpoint_path,
        ),
        ("hps", NINE_VIEWS[0].name, config.model, config.checkpoint_path),
        ("hps", NINE_VIEWS[1].name, config.model, config.checkpoint_path),
    ]
    cfg.qsel_hypotheses = 0
    with pytest.raises(ValueError, match="qsel_hypotheses"):
        cfg.make()


def test_fitted_sieve_roots_an_unrooted_member_at_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A member with no base_dir resolves its checkpoint beneath /opt/scratch."""
    cfg = FittedSieve.Config()
    cfg.experiment_name = "fitted"
    cfg.sieve.runtime.device = "cpu"
    cfg.sieve.checkpoint_path = "/runs/generator.pt"
    member = VerifierFit.Config()
    member.experiment_name = "member_s0"
    cfg.members.append(member)
    resolved: list[tuple[object, str]] = []

    def resolve(base: object, path: str) -> Path:
        resolved.append((base, path))
        return Path("/nonexistent") / path.lstrip("/")

    def fit_init(fit: VerifierFit, config: VerifierFit.Config) -> None:
        del fit, config
        raise _FitAttemptedError

    monkeypatch.setattr(
        "priml.baselines.sudoku.sieve.resolve_working_dir",
        resolve,
    )
    monkeypatch.setattr(VerifierFit, "__init__", fit_init)
    with pytest.raises(_FitAttemptedError):
        cfg.make().run()
    assert [base for base, _ in resolved] == ["/opt/scratch"]


class _FitAttemptedError(Exception):
    """Raised by a stub fit, so a test stops once the fit is reached."""


def test_fitted_sieve_fits_only_missing_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = FittedSieve.Config()
    cfg.experiment_name = "fitted"
    cfg.base_dir = tmp_path
    cfg.sieve.runtime.device = "cpu"
    cfg.sieve.checkpoint_path = "/runs/generator.pt"
    for seed in range(2):
        member = VerifierFit.Config()
        member.experiment_name = f"member_s{seed}"
        member.max_steps = 3
        cfg.members.append(member)
    existing = tmp_path / "runs" / "member_s0" / "checkpoints" / "step_00000003.pt"
    existing.parent.mkdir(parents=True)
    existing.touch()
    fitted: list[str] = []

    def fit_init(fit: VerifierFit, config: VerifierFit.Config) -> None:
        fit.config = config

    def fit_run(fit: VerifierFit) -> Path:
        fitted.append(fit.config.experiment_name)
        return existing

    monkeypatch.setattr(VerifierFit, "__init__", fit_init)
    monkeypatch.setattr(VerifierFit, "run", fit_run)

    def sieve_run(sieve: SieveEval) -> dict[str, object]:
        del sieve
        return {"ran": 1}

    monkeypatch.setattr(SieveEval, "run", sieve_run)
    with caplog.at_level(logging.INFO):
        assert cfg.make().run("--extra") == {"ran": 1}
    assert caplog.messages[:2] == [
        "Ignoring launcher passthrough args: ('--extra',).",
        "member member_s0 already fit; reusing.",
    ]
    assert fitted == ["member_s1"]
    # A committee rebuilt after finalize must not silently drop a member.
    fitted_sieve = cfg.make()
    fitted_sieve.config.sieve.verifier_checkpoints = (str(existing),)
    with pytest.raises(ValueError, match="zip"):
        fitted_sieve.run()
    cfg.members = []
    with pytest.raises(ValueError, match="at least one member"):
        cfg.make()
    cfg.experiment_name = ""
    with pytest.raises(ValueError, match="experiment_name"):
        cfg.make()


_RECORDERS: Final[dict[str, Callable[[Harness], dict[str, Tensor]]]] = {
    "staircase": _record_staircase,
}


@pytest.mark.parametrize("case", _RECORDERS)
def test_golden_replays_bit_for_bit(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner reproduces the frozen evaluation with zero mismatches."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "torchinductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    puzzle_data.dihedral_indices.cache_clear()
    actual = replay(_RECORDERS[case], tmp_path, train=False)
    report = mismatches(load_golden(case), actual)
    assert not report, f"{len(report)} mismatches:\n" + "\n".join(report)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
