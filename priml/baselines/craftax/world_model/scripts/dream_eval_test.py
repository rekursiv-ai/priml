"""Check dream evaluation: graphed-order rollouts, features, comparisons, and outputs."""

from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import cast

import dataclasses
import sys

import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    Receipt,
    write_shard,
)
from priml.baselines.craftax.world_model.batch import Segment
from priml.baselines.craftax.world_model.dream import Rollout, dream
from priml.baselines.craftax.world_model.engine import (
    Control,
    Decision,
    Engine,
    Outcome,
    Prefix,
    StartResult,
)
from priml.baselines.craftax.world_model.experiments import (
    WorldModelLoop,
    exp_smoke,
)
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.scripts import dream_eval
from priml.baselines.craftax.world_model.scripts.engine_checks import (
    prefix_of,
)
from priml.baselines.craftax.world_model.testing import (
    random_segment,
    small_schema,
    tiny_model,
)
from priml.lib.codec import PlainTree, from_plain, loads


HORIZONS = (2, 4)


def aux_frame(**values: int) -> torch.Tensor:
    """Return one frame's aux values: attributes 1, facing down, the rest 0 unless given."""
    merged = {
        "dexterity": 1,
        "strength": 1,
        "intelligence": 1,
        "facing_down": 1,
        "health": 180,
        "food": 9,
        "drink": 9,
        "energy": 9,
    } | values
    names = craftax_schema().scalar_names
    return torch.tensor([merged.get(name, 0) for name in names], dtype=torch.int16)


def board(seed: int) -> torch.Tensor:
    """Return a visible 99-cell board with random blocks and no mobs."""
    generator = torch.Generator().manual_seed(seed)
    cells = torch.zeros(99, 8, dtype=torch.uint8)
    cells[:, 0] = torch.randint(2, 30, (99,), generator=generator).to(torch.uint8)
    cells[:, 1] = 1
    cells[:, 2] = 1
    return cells


def shifted_left(cells: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Return the view after a move left: every column moves one to the right."""
    grid = cells.view(9, 11, 8).clone()
    grid[:, 1:] = cells.view(9, 11, 8)[:, :-1]
    grid[:, 0] = board(seed).view(9, 11, 8)[:, 0]
    return grid.view(99, 8)


def played_segment(*, terminal: bool) -> Segment:
    """Return four decisions: collect wood, descend, craft a pickaxe, move left.

    The move is consistent with the board; with ``terminal`` the last decision
    dies, otherwise the frame after it exists.
    """
    first = board(0)
    frames = [first, first, board(1), board(1)]
    frames.append(shifted_left(frames[-1], seed=2))
    aux = [
        aux_frame(),
        aux_frame(wood=1),
        aux_frame(wood=1, floor=1, xp=1),
        aux_frame(wood=0, pickaxe=1, floor=1, xp=1),
        aux_frame(pickaxe=1, floor=1, xp=1, facing_down=0, facing_left=1),
    ]
    decisions = 4
    kept = decisions if terminal else decisions + 1
    return Segment(
        cells=torch.stack(frames[:kept]),
        aux=torch.stack(aux[:kept]),
        actions=torch.tensor([5, 18, 11, 1], dtype=torch.uint8),
        reward=torch.tensor([1, 2, 1, -1 if terminal else 0], dtype=torch.int16),
        done=torch.tensor([False, False, False, terminal]),
        starts_episode=True,
    )


def features_of(segment: Segment) -> dream_eval.Features:
    return dream_eval.episode_features(segment, horizons=HORIZONS)


def check(features: dream_eval.Features, name: str) -> tuple[int, int]:
    """Return a validity check's violations and opportunities."""
    row = features.checks[dream_eval.CHECKS.index(name)]
    return int(row[0]), int(row[1])


# Two three-decision rollouts through a two-layer global stack, one row prefixed.
def test_rollout_in_graphed_order_equals_dream() -> None:
    schema = small_schema()
    model = tiny_model(schema, global_layers=2)
    prefixes = [prefix_of(random_segment(schema, 3, seed=1)), None]
    actions = torch.tensor([[1, 2, 3], [3, 2, 1]])
    runs: list[Rollout] = []
    for use_dream in (True, False):
        engine = Engine(
            model,
            rows=2,
            t_max=8,
            generator=torch.Generator().manual_seed(4),
        )
        if use_dream:
            runs.append(dream(engine, decisions=3, prefixes=prefixes, actions=actions))
        else:
            runs.append(
                dream_eval.rollout(
                    engine,
                    engine.step,
                    decisions=3,
                    prefixes=prefixes,
                    actions=actions,
                ),
            )
    for field in dataclasses.fields(Rollout):
        expected, actual = (
            cast("torch.Tensor", getattr(run, field.name)) for run in runs
        )
        assert torch.equal(expected, actual), field.name


def test_rollout_from_new_worlds_equals_dream() -> None:
    model = tiny_model(small_schema())
    engines = [
        Engine(model, rows=2, t_max=16, generator=torch.Generator().manual_seed(2))
        for _ in range(2)
    ]
    runs = [
        dream(engines[0], decisions=4),
        dream_eval.rollout(engines[1], engines[1].step, decisions=4),
    ]
    assert bool(runs[0].done.any())
    for field in dataclasses.fields(Rollout):
        expected, actual = (
            cast("torch.Tensor", getattr(run, field.name)) for run in runs
        )
        assert torch.equal(expected, actual), field.name


def test_rollout_samples_actions_given_as_negative() -> None:
    model = tiny_model(small_schema())
    runs: list[Rollout] = []
    for actions in (None, torch.tensor([[-1] * 4, [5] * 4])):
        engine = Engine(
            model,
            rows=2,
            t_max=16,
            generator=torch.Generator().manual_seed(9),
        )
        runs.append(
            dream_eval.rollout(engine, engine.step, decisions=4, actions=actions),
        )
    free, mixed = runs
    assert mixed.action[1].tolist() == [5] * 4
    assert torch.equal(mixed.action[0], free.action[0])
    assert torch.equal(mixed.cells[0], free.cells[0])


def test_teacher_forces_every_act_job_of_its_rows() -> None:
    schema = small_schema()
    model = tiny_model(schema)
    real = random_segment(schema, 4, seed=3)
    engine = Engine(model, rows=2, t_max=16, generator=torch.Generator().manual_seed(1))
    teacher = dream_eval.Teacher(
        rows=torch.tensor([True, False]),
        outcome=Outcome(
            reward=real.reward[None].expand(2, -1),
            done=real.done[None].expand(2, -1),
            cells=real.cells[None, 1:].expand(2, -1, -1, -1),
            aux=real.aux[None, 1:].expand(2, -1, -1),
        ),
    )
    actions = torch.stack([real.actions.long(), torch.full((4,), -1)])
    # A prefix of no decisions after ``start`` shows the real first frame.
    prefix = Prefix(
        cells=real.cells[None, :1],
        aux=real.aux[None, :1],
        actions=real.actions[None, :0],
        starts_episode=True,
    )
    result = dream_eval.rollout(
        engine,
        engine.step,
        decisions=4,
        prefixes=[prefix, None],
        actions=actions,
        teacher=teacher,
    )
    assert torch.equal(result.cells[0], real.cells)
    assert torch.equal(result.aux[0], real.aux)
    assert torch.equal(result.reward[0], real.reward)
    assert torch.equal(result.action[0], real.actions)
    logp = torch.cat(
        [result.action_logp[0], result.reward_logp[0], result.frame_logp[0].flatten()],
    )
    assert bool(torch.isfinite(logp).all())
    assert bool((result.frame_logp[0, 1:] < 0).all())
    assert not torch.equal(result.cells[1, 1:], real.cells[1:])


def test_episode_features_of_a_played_death() -> None:
    features = features_of(played_segment(terminal=True))
    assert features.length == 4
    assert features.died
    assert features.floor_first[:3].tolist() == [0, 1, -1]
    assert features.returns.tolist() == [3.0, 3.0]
    assert int(features.actions.sum()) == 4
    assert int(features.actions[1, 11]) == 1
    events = dict(zip(dream_eval.EVENTS, features.events.tolist(), strict=True))
    assert events["collect_wood"] == 0
    assert events["wood_pickaxe"] == 2
    assert events["gain_xp"] == 1
    assert events["take_damage"] == -1
    # The terminal decision has no next frame, so the move is not checked.
    assert check(features, "move_inconsistent") == (0, 0)
    for name in (
        "floor_change_without_ladder",
        "xp_without_descend",
        "pickaxe_without_craft",
        "material_without_do",
    ):
        assert check(features, name) == (0, 1), name
    assert check(features, "illegal_action")[1] == 4
    assert int(features.aux[dream_eval.FIELDS.index("wood")].sum()) == 4


def test_episode_features_check_moves_and_hud_rules() -> None:
    censored = played_segment(terminal=False)
    features = features_of(censored)
    assert not features.died
    assert features.length == 4
    assert check(features, "move_inconsistent") == (0, 1)
    assert check(features, "facing_wrong") == (0, 1)
    broken = dataclasses.replace(
        censored,
        cells=torch.cat([censored.cells[:-1], board(7)[None]]),
        actions=torch.tensor([0, 0, 11, 1], dtype=torch.uint8),
    )
    features = features_of(broken)
    assert check(features, "move_inconsistent") == (1, 1)
    assert check(features, "floor_change_without_ladder") == (1, 1)
    assert check(features, "xp_without_descend") == (1, 1)
    assert check(features, "inventory_change_on_move")[0] == 1


def test_episode_features_flag_invalid_cells_and_values_over_their_maximum() -> None:
    segment = played_segment(terminal=False)
    cells = segment.cells.clone()
    cells[3, 10] = torch.tensor([0, 0, 0, 0, 3, 0, 0, 0])
    aux = segment.aux.clone()
    aux[2, dream_eval.FIELDS.index("health")] = 200
    features = features_of(dataclasses.replace(segment, cells=cells, aux=aux))
    assert check(features, "invalid_frame") == (1, 5)
    assert check(features, "over_maximum") == (1, 5)


def test_compare_finds_nothing_between_equal_samples() -> None:
    real = [features_of(played_segment(terminal=True))] * 6
    result = dream_eval.compare(
        real,
        real,
        horizons=HORIZONS,
        resamples=64,
        permutations=64,
        seed=0,
    )
    died = from_plain(
        from_plain(result["died_by"], dict[str, object])["4"],
        dict[str, object],
    )
    assert not died["distinguishable"]
    actions = from_plain(
        from_plain(result["actions"], dict[str, object])["all"],
        dict[str, object],
    )
    assert from_plain(actions["tv"], float) == 0.0
    assert from_plain(result["distinguishable"], list[str]) == []


def test_compare_separates_deaths_from_survivals() -> None:
    real = [features_of(played_segment(terminal=False))] * 8
    dream_side = [features_of(played_segment(terminal=True))] * 8
    result = dream_eval.compare(
        real,
        dream_side,
        horizons=HORIZONS,
        resamples=64,
        permutations=64,
        seed=0,
    )
    died = from_plain(
        from_plain(result["died_by"], dict[str, object])["4"],
        dict[str, object],
    )
    assert (
        from_plain(from_plain(died["real"], dict[str, object])["value"], float) == 0.0
    )
    assert (
        from_plain(from_plain(died["dream"], dict[str, object])["value"], float) == 1.0
    )
    assert died["distinguishable"]
    assert "died_by/4" in from_plain(result["distinguishable"], list[str])


def test_permutation_tv_detects_a_shifted_distribution() -> None:
    real = torch.tensor([[9.0, 1.0]] * 10)
    same = dream_eval.permutation_tv(
        real,
        real,
        permutations=200,
        generator=torch.Generator().manual_seed(0),
    )
    other = dream_eval.permutation_tv(
        real,
        torch.tensor([[1.0, 9.0]] * 10),
        permutations=200,
        generator=torch.Generator().manual_seed(0),
    )
    assert same["tv"] == 0.0
    assert from_plain(same["p_value"], float) == 1.0
    assert from_plain(other["tv"], float) == pytest.approx(0.8)
    assert from_plain(other["p_value"], float) < 0.05


def test_ratio_interval_pools_units_and_brackets_the_ratio() -> None:
    entry, draws = dream_eval.ratio_interval(
        torch.tensor([1.0, 3.0, 2.0, 6.0]),
        torch.tensor([10.0, 10.0, 10.0, 10.0]),
        resamples=256,
        generator=torch.Generator().manual_seed(0),
    )
    assert entry["value"] == pytest.approx(0.3)
    low, high = from_plain(entry["low"], float), from_plain(entry["high"], float)
    assert 0.1 <= low < 0.3 < high <= 0.6
    assert draws.shape == (256,)
    empty, _ = dream_eval.ratio_interval(
        torch.zeros(2),
        torch.zeros(2),
        resamples=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert empty["value"] is None


def test_compare_ratio_takes_the_second_sample_minus_the_first() -> None:
    generator = torch.Generator().manual_seed(0)
    entry = dream_eval.compare_ratio(
        torch.tensor([1.0, 2.0, 1.0]),
        torch.full((3,), 10.0),
        torch.tensor([3.0, 3.0]),
        torch.full((2,), 10.0),
        resamples=64,
        generator=generator,
    )
    assert from_plain(entry["real"], dict[str, object])["value"] == pytest.approx(
        4 / 30,
    )
    assert from_plain(entry["dream"], dict[str, object])["value"] == pytest.approx(0.3)
    diff = from_plain(entry["diff"], dict[str, object])
    assert diff["value"] == pytest.approx(0.3 - 4 / 30)
    # The second sample never varies, so the interval spans the first's spread.
    assert diff["low"] == pytest.approx(0.3 - 0.2)
    assert diff["high"] == pytest.approx(0.2)
    empty = dream_eval.compare_ratio(
        torch.ones(1),
        torch.zeros(1),
        torch.ones(1),
        torch.ones(1),
        resamples=4,
        generator=generator,
    )
    assert empty["diff"] is None
    assert empty["p_value"] is None


def test_sampling_engine_steps_eagerly_on_the_cpu() -> None:
    model = tiny_model(small_schema())
    engine, step = dream_eval.sampling_engine(model, rows=2, t_max=16, seed=3)
    assert engine.rows == 2
    assert engine.t_max == 16
    assert step == engine.step


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_sampling_engine_seeds_its_models_device_not_the_current_one() -> None:
    assert torch.cuda.current_device() == 0
    model = tiny_model(small_schema()).to("cuda:1")
    generator = torch.cuda.default_generators[model.start.device.index or 0]
    generator.manual_seed(99)
    engine, _ = dream_eval.sampling_engine(model, rows=2, t_max=16, seed=3)
    assert engine.generator is generator
    assert generator.initial_seed() == 3


@pytest.mark.parametrize(("rows", "prefix"), [(5, 2), (4, 20)])
def test_evaluate_refuses_too_few_windows_before_generating(
    tmp_path: Path,
    rows: int,
    prefix: int,
) -> None:
    _write_val_archive(tmp_path / "archive")
    output = tmp_path / "dreams"
    with pytest.raises(ValueError, match="Continuations need rows / 2"):
        dream_eval.evaluate(
            tiny_model(craftax_schema()),
            archive=tmp_path / "archive",
            output=output,
            rows=rows,
            t_max=16,
            decisions=4,
            prefix=prefix,
            continuation=3,
            horizons=HORIZONS,
            seed=0,
            resamples=16,
            permutations=16,
        )
    assert not output.exists()


def test_first_finds_each_columns_first_true_row_or_none() -> None:
    mask = torch.tensor([[False, True, False], [True, True, False]])
    assert dream_eval._first(mask).tolist() == [1, 0, -1]
    assert dream_eval._first(torch.zeros(0, 3, dtype=torch.bool)).tolist() == [-1] * 3


def test_free_summary_counts_first_episodes_that_ended_without_a_death() -> None:
    done = torch.tensor([[False, True, False], [False, False, False]])
    free = Rollout(
        cells=torch.zeros(2, 4, 99, 8, dtype=torch.uint8),
        aux=torch.zeros(2, 4, 51, dtype=torch.int16),
        # Frame 0 starts each row's episode; a done starts the next frame's.
        starts=torch.cat([torch.ones(2, 1, dtype=torch.bool), done], dim=1),
        frame_logp=torch.zeros(2, 4, 150),
        invalid=torch.zeros(2, 4, 99, dtype=torch.bool),
        action=torch.zeros(2, 3, dtype=torch.uint8),
        reward=torch.zeros(2, 3, dtype=torch.int16),
        done=done,
        action_logp=torch.zeros(2, 3),
        reward_logp=torch.zeros(2, 3),
        done_logp=torch.zeros(2, 3),
    )
    summary = dream_eval._free_summary(free)
    assert summary["deaths"] == 0
    assert summary["first_episode_ended"] == 1


def test_main_samples_in_float32_off_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"")
    dtypes: list[torch.dtype] = []

    def evaluate(model: WorldModel, **kwargs: object) -> dict[str, PlainTree]:
        del kwargs
        dtypes.append(model.start.dtype)
        return {"comparison": {"distinguishable": []}}

    monkeypatch.setattr(dream_eval, "load_world_model", _tiny_trained)
    monkeypatch.setattr(dream_eval, "evaluate", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *("dream_eval.py", str(checkpoint), "--archive", str(tmp_path)),
            *("--output", str(tmp_path / "out"), "--device", "cpu"),
        ],
    )
    assert dream_eval.main() == 0
    assert dtypes == [torch.float32]


def test_divergence_finds_the_first_differing_frame() -> None:
    real = played_segment(terminal=False)
    cells = real.cells.clone()
    cells[3, 5, 0] += 1
    dreamed = dataclasses.replace(real, cells=cells)
    result = dream_eval.divergence(
        [dreamed, real],
        [real, real],
        checkpoints=(1, 3),
        resamples=32,
        seed=0,
    )
    first = from_plain(result["first_divergence"], dict[str, object])
    assert from_plain(first["frame"], list[object]) == [3, None]
    exact = from_plain(result["frame_exact"], dict[str, object])
    assert from_plain(from_plain(exact["1"], dict[str, object])["value"], float) == 1.0
    assert from_plain(from_plain(exact["3"], dict[str, object])["value"], float) == 0.5
    reward = from_plain(result["reward"], dict[str, object])
    assert from_plain(reward["compared"], int) == 8
    agreement = from_plain(reward["agreement"], dict[str, object])
    assert from_plain(agreement["value"], float) == 1.0
    assert from_plain(first["board"], list[object]) == [3, None]
    assert from_plain(first["hud"], list[object]) == [None, None]
    fields = from_plain(result["board_field_mismatch"], dict[str, object])
    # One block of one cell differs in one of the two pairs.
    at_three = from_plain(fields["3"], list[float])
    assert at_three[0] == pytest.approx(0.5 / 99)
    assert at_three[1:] == [0.0] * 7


def test_divergence_counts_every_frame_after_an_early_end_as_differing() -> None:
    real = played_segment(terminal=False)
    # This continuation dies in decision 1, so it holds frames 0-1 of real's 0-4.
    ended = Segment(
        cells=real.cells[:2],
        aux=real.aux[:2],
        actions=real.actions[:2],
        reward=torch.tensor([1, -1], dtype=torch.int16),
        done=torch.tensor([False, True]),
        starts_episode=True,
    )
    result = dream_eval.divergence(
        [ended, real],
        [real, real],
        checkpoints=(1, 3),
        resamples=32,
        seed=0,
    )
    # Both pairs count at every frame the real continuation holds.
    for name in ("board_mismatch", "hud_mismatch"):
        table = from_plain(result[name], dict[str, object])
        at_one, at_three = (from_plain(table[k], dict[str, object]) for k in ("1", "3"))
        assert at_three["n"] == 2, name
        assert from_plain(at_one["value"], float) == 0.0, name
        assert from_plain(at_three["value"], float) == 0.5, name
    assert from_plain(result["ended"], dict[str, object]) == {"1": 0, "3": 1}
    # Each pair's own value, so two reports can be compared window by window.
    per_pair = from_plain(result["per_pair"], dict[str, object])
    board = from_plain(per_pair["board_mismatch"], dict[str, object])
    assert from_plain(board["1"], list[float]) == [0.0, 0.0]
    assert from_plain(board["3"], list[float]) == [1.0, 0.0]
    exact = from_plain(per_pair["frame_exact"], dict[str, object])
    assert from_plain(exact["3"], list[float]) == [0.0, 1.0]
    first = from_plain(result["first_divergence"], dict[str, object])
    assert from_plain(first["frame"], list[object]) == [2, None]
    fields = from_plain(result["board_field_mismatch"], dict[str, object])
    assert from_plain(fields["3"], list[float]) == [0.5] * 8


def test_divergence_counts_an_end_only_at_steps_the_real_continuation_holds() -> None:
    real = played_segment(terminal=False)
    # The real continuation dies in decision 1, holding frames 0-1; the
    # generated one dies in decision 0 and holds frame 0 alone.
    short = Segment(
        cells=real.cells[:2],
        aux=real.aux[:2],
        actions=real.actions[:2],
        reward=torch.tensor([1, -1], dtype=torch.int16),
        done=torch.tensor([False, True]),
        starts_episode=True,
    )
    shorter = Segment(
        cells=real.cells[:1],
        aux=real.aux[:1],
        actions=real.actions[:1],
        reward=torch.tensor([-1], dtype=torch.int16),
        done=torch.tensor([True]),
        starts_episode=True,
    )
    result = dream_eval.divergence(
        [shorter, real],
        [short, real],
        checkpoints=(1, 3),
        resamples=32,
        seed=0,
    )
    assert from_plain(result["ended"], dict[str, object]) == {"1": 1, "3": 0}
    board = from_plain(result["board_mismatch"], dict[str, object])
    assert from_plain(board["1"], dict[str, object])["n"] == 2
    assert from_plain(board["3"], dict[str, object])["n"] == 1
    # A pair the step does not count holds no value there.
    per_pair = from_plain(
        from_plain(result["per_pair"], dict[str, object])["board_mismatch"],
        dict[str, object],
    )
    assert from_plain(per_pair["3"], list[object]) == [None, 0.0]


def test_teacher_nll_counts_each_real_decision_once() -> None:
    real = played_segment(terminal=True)
    frames = len(real.cells) + 1
    # One rollout row, for the one real segment ``teacher_nll`` scores it against.
    rollout = Rollout(
        cells=torch.zeros(1, frames, 99, 8, dtype=torch.uint8),
        aux=torch.zeros(1, frames, 51, dtype=torch.int16),
        starts=torch.tensor([[False, False, False, False, True]]),
        frame_logp=torch.full((1, frames, 150), -0.01),
        invalid=torch.zeros(1, frames, 99, dtype=torch.bool),
        action=real.actions[None],
        reward=real.reward[None],
        done=real.done[None],
        action_logp=torch.full((1, 4), -1.0),
        reward_logp=torch.full((1, 4), -0.5),
        done_logp=torch.full((1, 4), -0.25),
    )
    result = dream_eval.teacher_nll(rollout, [real], resamples=16, seed=0)
    nats = from_plain(result["nats_per_decision"], dict[str, object])
    # Three next frames are scored; the terminal decision's is not.
    expected = (4 * 1.75 + 3 * 1.5) / 4
    assert from_plain(nats["value"], float) == pytest.approx(expected)
    assert from_plain(result["decisions"], int) == 4


def test_truncate_keeps_the_frame_after_a_decision_that_did_not_end() -> None:
    episode = _episode(played_segment(terminal=True))
    cut = dream_eval.truncate(episode, decisions=2)
    assert len(cut.actions) == 2
    assert torch.equal(cut.cells, episode.cells[:3])
    whole = dream_eval.truncate(episode, decisions=10)
    assert len(whole.actions) == len(whole.cells) == 4
    assert bool(whole.done[-1])


@pytest.mark.parametrize("origin", [b"", b"branch state"])
def test_a_cut_at_decision_0_starts_an_episode_only_from_its_worlds_reset(
    origin: bytes,
) -> None:
    """A branch begins mid-episode: its decision 0 continues, as training reads it."""
    segment = played_segment(terminal=True)
    episode = dataclasses.replace(_episode(segment), origin=origin)
    leading = dream_eval.truncate(episode, decisions=2)
    assert leading.starts_episode == (not origin)
    assert torch.equal(leading.actions, segment.actions[:2])
    cut = dream_eval.window(
        episode,
        anchor=0,
        prefix=2,
        decisions=5,
        kind="k",
        name="e",
    )
    assert cut.prefix.starts_episode == (not origin)
    assert torch.equal(cut.prefix.actions[0], segment.actions[:2])
    assert torch.equal(cut.real.actions, segment.actions[2:])


def test_window_cuts_a_prefix_and_its_real_continuation() -> None:
    episode = _episode(played_segment(terminal=True))
    cut = dream_eval.window(
        episode,
        anchor=1,
        prefix=1,
        decisions=5,
        kind="k",
        name="e",
    )
    assert cut.name == "e@1"
    assert not cut.prefix.starts_episode
    assert torch.equal(cut.prefix.cells[0], episode.cells[1:3])
    assert torch.equal(cut.prefix.actions[0], episode.actions[1:2])
    # The continuation starts at the prefix's last frame and ends in the death.
    assert torch.equal(cut.real.cells, episode.cells[2:])
    assert cut.real.actions.tolist() == [11, 1]
    assert cut.real.done.tolist() == [False, True]


def test_choose_windows_places_deaths_inside_the_continuation(tmp_path: Path) -> None:
    _write_val_archive(tmp_path)
    sources = dream_eval.val_sources(tmp_path, arm=3)
    windows = dream_eval.choose_windows(
        sources,
        count=4,
        prefix=2,
        decisions=4,
        generator=torch.Generator().manual_seed(0),
    )
    kinds = [w.kind for w in windows]
    assert kinds == ["pre_death", "uniform", "uniform", "uniform"]
    death = windows[0]
    assert bool(death.real.done[-1])
    assert int(death.real.reward[-1]) == -1
    assert len(death.real.actions) <= 4
    for w in windows[1:]:
        assert len(w.real.actions) == 4
        assert len(w.real.cells) == 5
        assert not bool(w.real.done.any())
    assert len({w.name.split("@")[0] for w in windows}) == 4
    # The death lands uniformly on every continuation decision.
    lengths = {
        len(dream_eval.choose_windows(
            sources, count=4, prefix=2, decisions=4,
            generator=torch.Generator().manual_seed(seed),
        )[0].real.actions)
        for seed in range(24)
    }  # fmt: skip
    assert lengths == {1, 2, 3, 4}


def test_episode_record_holds_every_frame_it_keeps() -> None:
    segment = played_segment(terminal=False)
    record = dream_eval.episode_record(
        segment,
        name="demo",
        source="model",
        max_frames=4,
    )
    assert record["name"] == "demo"
    frames = from_plain(record["frame_index"], list[int])
    assert frames == [0, 1, 3, 4]
    assert len(from_plain(record["aux"], list[object])) == 4
    assert from_plain(record["action"], list[int]) == [5, 18, 11, 1]


def test_evaluate_writes_statistics_episodes_and_bundles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    without_decoding(monkeypatch, dream_eval)
    archive = tmp_path / "archive"
    _write_val_archive(archive)
    model = tiny_model(craftax_schema())
    output = tmp_path / "dreams"
    stats = dream_eval.evaluate(
        model,
        archive=archive,
        output=output,
        rows=2,
        t_max=16,
        decisions=4,
        prefix=2,
        continuation=3,
        horizons=HORIZONS,
        seed=0,
        resamples=4,
        permutations=4,
    )
    written = from_plain(
        loads((output / "stats.json").read_text()),
        dict[str, object],
    )
    assert written["schema"] == stats["schema"] == "craftax-dream-eval/v1"
    for key in ("free_running", "comparison", "continuation", "bundles", "provenance"):
        assert key in written, key
    episodes = from_plain(
        loads((output / "episodes.json").read_text()),
        dict[str, object],
    )
    assert len(from_plain(episodes["episodes"], list[object])) >= 2
    bundles = sorted(p.name for p in (output / "bundles").iterdir())
    assert len(bundles) >= 2
    for name in bundles:
        assert (output / "bundles" / name / "manifest.json").is_file()


def test_main_rejects_an_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dream_eval.py",
            str(tmp_path / "ckpt.pt"),
            "--archive",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(FileExistsError):
        dream_eval.main()


def test_vocabularies_name_the_games_actions_blocks_and_items() -> None:
    actions = dream_eval.ACTION_NAMES
    assert len(actions) == 43
    assert actions[:6] == ("noop", "left", "right", "up", "down", "do")
    assert (actions[18], actions[19], actions[42]) == (
        "descend",
        "ascend",
        "enchant_bow",
    )
    blocks = dream_eval.BLOCK_NAMES
    assert len(blocks) == 37
    assert (blocks[2], blocks[36]) == ("grass", "necromancer_vulnerable")
    assert dream_eval.ITEM_NAMES == (
        "unseen",
        "none",
        "torch",
        "ladder_down",
        "ladder_up",
        "ladder_down_blocked",
    )


def without_decoding(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> None:
    """Stand in for a script's sampling engine and its rollouts: no model decodes.

    The rollouts have the tests above; a test of what a script makes of them
    reads :func:`dreamed`'s frames.
    """
    monkeypatch.setattr(module, "sampling_engine", stand_in_engine)
    monkeypatch.setattr(module, "rollout", dreamed)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Engine:
    """Stand in for an ``Engine``: :func:`dreamed` reads its rows alone."""

    rows: int


def stand_in_engine(
    model: WorldModel,
    *,
    rows: int,
    t_max: int,
    seed: int,
) -> tuple[Engine, dream_eval.Step]:
    """Stand in for ``sampling_engine``: an engine of ``rows`` that decodes nothing.

    Args:
      model: Not decoded.
      rows: The engine's rows.
      t_max: Not used.
      seed: Not used.

    Returns:
      engine: Its rows, all :func:`dreamed` reads.
      step: A step never taken.

    """
    del model, t_max, seed
    return cast("Engine", _Engine(rows=rows)), _undecoded


def _undecoded(control: Control) -> tuple[StartResult, Decision]:
    """Stand in for an engine's step, which :func:`dreamed` never takes."""
    raise AssertionError(f"The stand-in engine decodes nothing: {control}.")


def dreamed(
    engine: Engine,
    step: dream_eval.Step,
    *,
    decisions: int,
    prefixes: Sequence[Prefix | None] = (),
    actions: torch.Tensor | None = None,
    teacher: dream_eval.Teacher | None = None,
) -> Rollout:
    """Stand in for ``rollout``: each row random valid frames, its forced actions taken.

    A prefixed row's frame 0 is its prefix's last frame, and a taught row's
    next frames and outcomes are its teacher's, as a rollout's are; a sampled
    action is a noop, and no other row ends.

    Args:
      engine: Its rows.
      step: Not taken.
      decisions: Decisions per row.
      prefixes: One entry per row, or empty: every row a new world.
      actions: Actions to force, negative where sampled; None samples all.
      teacher: Rows whose outcomes are forced.

    Returns:
      rollout: Random valid frames of the Craftax schema, logp 0.

    """
    del step
    rows = engine.rows
    played = [random_segment(craftax_schema(), decisions, seed=r) for r in range(rows)]
    cells = torch.stack([segment.cells for segment in played])
    aux = torch.stack([segment.aux for segment in played])
    reward = torch.zeros(rows, decisions, dtype=torch.int16)
    done = torch.zeros(rows, decisions, dtype=torch.bool)
    starts = torch.zeros(rows, decisions + 1, dtype=torch.bool)
    for row in range(rows):
        prefix = prefixes[row] if prefixes else None
        starts[row, 0] = prefix is None
        if prefix is not None:
            cells[row, 0] = prefix.cells[0, -1]
            aux[row, 0] = prefix.aux[0, -1]
        if teacher is not None and bool(teacher.rows[row]):
            outcome = teacher.outcome
            cells[row, 1:] = outcome.cells[row]
            aux[row, 1:] = outcome.aux[row]
            reward[row] = outcome.reward[row]
            done[row] = outcome.done[row]
    taken = torch.zeros(rows, decisions) if actions is None else actions.clamp(min=0)
    zeros = torch.zeros(rows, decisions)
    return Rollout(
        cells=cells,
        aux=aux,
        starts=starts,
        frame_logp=torch.zeros(rows, decisions + 1, craftax_schema().frame_slots),
        invalid=torch.zeros(rows, decisions + 1, cells.shape[2], dtype=torch.bool),
        action=taken.to(torch.uint8),
        reward=reward,
        done=done,
        action_logp=zeros,
        reward_logp=zeros,
        done_logp=zeros,
    )


def _episode(segment: Segment) -> Episode:
    """Return an archived episode of a complete segment."""
    return Episode(
        receipt=Receipt(
            world_seed=0,
            sampling_seed=1,
            initial_state_hash=2,
            arm=3,
            split=1,
        ),
        actions=segment.actions,
        hashes=torch.zeros(1, dtype=torch.int64),
        cells=segment.cells,
        aux=segment.aux,
        reward=segment.reward,
        done=segment.done,
        summary={"death": 1, "achievements": []},
    )


def _tiny_trained(
    experiment: str,
    checkpoint: Path,
    *,
    overrides: Sequence[str] = (),
) -> tuple[WorldModel, WorldModelLoop.Config]:
    """Stand in for ``load_world_model``: a tiny model and the smoke experiment."""
    del experiment, checkpoint, overrides
    return tiny_model(small_schema()), exp_smoke()


def _write_val_archive(root: Path) -> None:
    """Publish one validation shard of four arm-3 episodes, two of them deaths."""
    directory = root / "val" / "arm3" / "w0"
    directory.mkdir(parents=True)
    episodes: list[Episode] = []
    for index, decisions in enumerate((9, 12, 10, 14)):
        segment = random_segment(craftax_schema(), decisions, seed=index, terminal=True)
        aux = segment.aux.clone()
        aux[:, dream_eval.FIELDS.index("floor")] = (
            torch.arange(decisions) * 8 // decisions
        )
        died = index % 2 == 0
        reward = segment.reward.clone().clamp(min=0)
        reward[-1] = -1 if died else 0
        episodes.append(
            Episode(
                receipt=Receipt(
                    world_seed=index,
                    sampling_seed=1,
                    initial_state_hash=2,
                    arm=3,
                    split=1,
                ),
                actions=segment.actions,
                hashes=torch.zeros(1, dtype=torch.int64),
                cells=segment.cells,
                aux=aux,
                reward=reward,
                done=segment.done,
                summary={"death": int(died), "achievements": []},
            ),
        )
    write_shard(directory, index=0, episodes=episodes, provenance={})


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
