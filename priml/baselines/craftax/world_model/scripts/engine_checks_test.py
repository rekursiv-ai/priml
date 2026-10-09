"""Check the engine checks, each on a tiny model of the cut schema, and the script.

The checks pass on any correct engine whatever its schema; the cut schema
decodes 7 slots a frame where Craftax's decodes 150.
"""

from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import copy
import sys

from scipy import stats

import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    ManifestLine,
    Receipt,
    write_corpus,
    write_shard,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.scripts import engine_checks
from priml.baselines.craftax.world_model.testing import (
    random_segment,
    small_schema,
    tiny_model,
)
from priml.lib.codec import PlainTree, from_plain, loads


@pytest.fixture(scope="module")
def checked() -> engine_checks.Checked:
    """Return a tiny model of the cut schema in its three forms, and two decisions."""
    model = tiny_model(small_schema())
    return engine_checks.Checked(
        sdpa=model,
        trained=copy.deepcopy(model),
        sampler=copy.deepcopy(model).to(dtype=torch.bfloat16),
        autocast=nullcontext(),
        segment=random_segment(small_schema(), 2, seed=3),
        t_max=32,
    )


def test_the_engine_scores_a_segment_as_the_training_forward_does(
    checked: engine_checks.Checked,
) -> None:
    results = engine_checks.check_teacher_forced(checked, tolerance=2e-3)
    fp32 = from_plain(results["1_teacher_forced_fp32"], dict[str, object])
    # A start job and one act job per decision.
    assert fp32["jobs"] == 3
    assert fp32["passed"] is True
    bf16 = from_plain(
        results["1_teacher_forced_bf16_vs_training_numerics"],
        dict[str, object],
    )
    assert from_plain(bf16["max_abs_slot_logp_diff"], float) >= 0


def test_a_prefilled_row_decides_as_a_row_that_stepped_there(
    checked: engine_checks.Checked,
) -> None:
    results = engine_checks.check_prefill(checked, tolerance=2e-3)
    entry = from_plain(
        results["2_prefill_then_step_equals_step_only_fp32"],
        dict[str, object],
    )
    assert entry["free_decisions_identical_tokens"] == "8/8"
    assert entry["passed"] is True


def test_a_reset_row_samples_anew_and_leaves_the_other_row_exact(
    checked: engine_checks.Checked,
) -> None:
    results = engine_checks.check_reset(checked.sampler, t_max=checked.t_max)
    entry = from_plain(results["3_reset_one_row_leaves_others_bf16"], dict[str, object])
    assert entry["passed"] is True


def test_gumbel_max_draws_follow_the_action_heads_softmax(
    checked: engine_checks.Checked,
) -> None:
    results = engine_checks.check_gumbel(
        checked.sampler,
        segment=checked.segment,
        t_max=checked.t_max,
        draws=2_000,
    )
    entry = from_plain(
        results["4_gumbel_max_matches_softmax_action_head"],
        dict[str, object],
    )
    chi, degrees = from_plain(entry["chi_square"], float), entry["degrees_of_freedom"]
    assert entry["p_value"] == stats.chi2.sf(chi, from_plain(degrees, int))
    assert entry["passed"] is True


def test_a_seed_repeats_its_tokens_and_a_session_reloads_as_saved(
    checked: engine_checks.Checked,
) -> None:
    # The float32 model: the check is the same in any dtype, and on the CPU it takes
    # 1.8x as long in bfloat16 (measured). The reset and Gumbel checks run bfloat16.
    results = engine_checks.check_seed_and_session(checked.sdpa, t_max=checked.t_max)
    entry = from_plain(results["5_fixed_seed_and_session_reload"], dict[str, object])
    assert entry["passed"] is True


@pytest.mark.parametrize("failing", [None, "3_reset", "5_seed"])
def test_run_checks_runs_the_five_in_order_and_passes_if_each_does(
    checked: engine_checks.Checked,
    monkeypatch: pytest.MonkeyPatch,
    failing: str | None,
) -> None:
    ran: list[tuple[str, object]] = []

    def check(name: str, model: object, **settings: object) -> dict[str, PlainTree]:
        ran.append((name, model))
        assert settings in (
            {"tolerance": 2e-3},
            {"t_max": 32},
            {"segment": checked.segment, "t_max": 32, "draws": 9},
        )
        return {name: {"passed": name != failing}, f"{name}_numerics": {"diff": 0.5}}

    for name, attribute in (
        ("1_teacher", "check_teacher_forced"),
        ("2_prefill", "check_prefill"),
        ("3_reset", "check_reset"),
        ("4_gumbel", "check_gumbel"),
        ("5_seed", "check_seed_and_session"),
    ):
        monkeypatch.setattr(engine_checks, attribute, partial(check, name))
    results = engine_checks.run_checks(checked, tolerance=2e-3, draws=9)
    # The judged checks take the float32 forms, the sampling ones the sampler.
    assert ran == [
        ("1_teacher", checked),
        ("2_prefill", checked),
        ("3_reset", checked.sampler),
        ("4_gumbel", checked.sampler),
        ("5_seed", checked.sampler),
    ]
    assert list(results)[-1] == "all_passed"
    assert len(results) == 2 * 5 + 1
    assert results["all_passed"] is (failing is None)


def test_main_runs_the_checks_on_the_runs_models_and_the_corpus_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model(small_schema())
    config = SimpleNamespace(
        dataset=SimpleNamespace(t_g=32),
        step=SimpleNamespace(dtype_autocast=None),
    )
    loads_of: list[tuple[str, Path, list[str]]] = []

    def load_world_model(
        experiment: str,
        checkpoint: Path,
        *,
        overrides: list[str],
    ) -> tuple[WorldModel, object]:
        loads_of.append((experiment, checkpoint, overrides))
        return model, config

    def load_trained(
        experiment: str,
        checkpoint: Path,
        *,
        overrides: list[str],
        device: torch.device,
    ) -> tuple[WorldModel, object]:
        assert device == torch.device("cpu")
        loads_of.append((experiment, checkpoint, overrides))
        return model, config

    ran: list[tuple[engine_checks.Checked, float, int]] = []

    def run_checks(
        given: engine_checks.Checked,
        *,
        tolerance: float,
        draws: int,
    ) -> dict[str, PlainTree]:
        ran.append((given, tolerance, draws))
        return {"1_teacher_forced_fp32": {"passed": True}, "all_passed": True}

    monkeypatch.setattr(engine_checks, "load_world_model", load_world_model)
    monkeypatch.setattr(engine_checks, "load_trained", load_trained)
    monkeypatch.setattr(engine_checks, "run_checks", run_checks)
    output = tmp_path / "checks.json"
    checkpoint = tmp_path / "step.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "engine_checks.py",
            *(str(checkpoint), str(_corpus(tmp_path)), str(output)),
            *("--experiment", "experiments.exp_smoke", "--override", "a=1"),
            *("--decisions", "4", "--draws", "200", "--device", "cpu"),
        ],
    )
    assert engine_checks.main() == 0
    assert loads_of == [("experiments.exp_smoke", checkpoint, ["a=1"])] * 2
    ((given, tolerance, draws),) = ran
    assert (tolerance, draws, given.t_max) == (2e-3, 200, 32)
    assert given.sdpa is given.trained is model
    # The sampler is its own bfloat16 copy: the float32 forms stay as loaded.
    assert given.sampler is not model
    assert next(given.sampler.parameters()).dtype == torch.bfloat16
    assert next(model.parameters()).dtype == torch.float32
    # The first validation episode long enough: episode 8, its first 4 decisions.
    assert given.segment.actions.tolist() == [8] * 4
    results = from_plain(loads(output.read_text()), dict[str, object])
    assert list(results) == [
        "checkpoint",
        "corpus",
        "decisions",
        "1_teacher_forced_fp32",
        "all_passed",
    ]
    assert results["decisions"] == 4


@pytest.mark.parametrize("origin", [b"", b"branch state"])
def test_the_segment_is_the_first_long_enough_validation_episode(
    tmp_path: Path,
    origin: bytes,
) -> None:
    corpus = _corpus(tmp_path, origin=origin)
    segment = engine_checks.real_segment(corpus, decisions=7)
    # Episode 8 has 10 decisions, episode 9 has 11, and the training episode of
    # 12 is never read.
    assert len(segment.actions) == 7
    assert len(segment.cells) == 8
    assert int(segment.actions[0]) == 8
    # A branch's first decision continues its parent's episode.
    assert segment.starts_episode == (not origin)
    with pytest.raises(ValueError, match="long enough"):
        engine_checks.real_segment(corpus, decisions=10)


def _corpus(root: Path, *, origin: bytes = b"") -> Path:
    """Publish one long training episode, then validation episodes from ``origin``."""
    entries: list[tuple[Path, ManifestLine]] = []
    for split, seeds in ((TRAIN, [10]), (VALIDATION, [3, 8, 9])):
        directory = root / ("train" if split == TRAIN else "val") / "arm3" / "w0"
        directory.mkdir(parents=True)
        episodes = [
            _episode(
                2 + seed,
                split=split,
                world_seed=seed,
                origin=origin if split == VALIDATION else b"",
            )
            for seed in seeds
        ]
        line = write_shard(directory, index=0, episodes=episodes, provenance={})
        entries.append((directory, line))
    path = root / "corpora" / "test.json"
    write_corpus(path, entries=entries)
    return path


def _episode(decisions: int, *, split: int, world_seed: int, origin: bytes) -> Episode:
    """Return a terminal episode of constant frames, its actions the seed."""
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = True
    return Episode(
        receipt=Receipt(
            world_seed=world_seed,
            sampling_seed=0,
            initial_state_hash=0,
            arm=3,
            split=split,
        ),
        actions=torch.full((decisions,), world_seed, dtype=torch.uint8),
        hashes=torch.zeros(decisions // 256 + 1, dtype=torch.int64),
        cells=torch.zeros(decisions, 99, 8, dtype=torch.uint8),
        aux=torch.ones(decisions, 51, dtype=torch.int16),
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=done,
        summary={},
        origin=origin,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
