"""Tests for the nanochat train step."""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

from configgle import PartialConfig
from torch import nn

import pytest
import torch

from priml.baselines.nanochat import experiments
from priml.baselines.nanochat.train_step import (
    NanoChatTrainStep,
    matrix_parameters,
    nanochat_optimizer,
)
from priml.model.value_gated_attention import ValueGatedAttention
from priml.testing.bfb import assert_bfb_against_golden
from priml.train.parallelism import NoParallel


_GOLDEN_DIR = Path(__file__).parent / "goldens"

VOCAB = 32
SEQ = 8


def _step(**overrides: Any) -> NanoChatTrainStep:
    config = NanoChatTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    config.compile = None
    # The recipe's own optimizer, stepping eagerly. Dynamo traces each member's
    # kernel on first use and never caches it -- 9 of these tests' 9.5 seconds
    # -- and what the compiled graph changes is the update's last bits, which
    # the goldens and the parity script measure and none of these assert.
    config.optimizer = nanochat_optimizer(compile=False)
    config.model.vocab_size = VOCAB
    config.model.max_seq_len = SEQ
    config.model.channels_in = 16
    config.model.num_layers = 1
    attention = config.model.template.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    attention.window_pattern = "L"
    attention.channels_head = 8
    attention.gate_channels = 4
    config.rows_per_pass = 2
    config.tokens_per_optimizer_step = 2 * SEQ
    config.budget_warmup_steps = 0
    for name, value in overrides.items():
        setattr(config, name, value)
    torch.manual_seed(0)
    return config.make()


def _batch() -> dict[str, Any]:
    torch.manual_seed(1)
    rows = torch.randint(0, VOCAB, (2, SEQ + 1))
    return {"media": rows[:, :-1], "label": rows[:, 1:]}


def test_loss_decreases_over_a_few_steps() -> None:
    """The recipe actually learns on a repeated batch.

    Two steps rather than four: the orthogonalization is five batched matmuls
    per step and dominates the runtime here, and a second step is enough to
    show the loss moving the right way.
    """
    step = _step()
    batch = _batch()
    losses = [float(step.train_step(**batch)["loss"]) for _ in range(2)]
    assert losses[-1] < losses[0]


def test_the_optimizer_partitions_the_model() -> None:
    """NorMuon takes the matrices; AdamW takes the tables and the head.

    Each parameter belongs to exactly one member, which the composite
    verifies; this pins WHICH, since the split is the recipe.
    """
    step = _step()
    named = {id(p): n for n, p in step.model.named_parameters()}
    assigned = [
        {named[id(p)] for p in group["params"]} for group in step.optimizer.param_groups
    ]
    everything: set[str] = set()
    for names in assigned:
        everything |= names
    assert everything == set(named.values())
    assert sum(len(names) for names in assigned) == len(everything)
    for names in assigned:
        if any("blocks" in name for name in names):
            assert not any("embed" in name or "lm_head" in name for name in names)


def test_an_optimizer_step_waits_for_the_whole_token_batch() -> None:
    """Gradient accumulation is what holds the token batch fixed.

    A step that updated per pass would train at a different batch size than
    the recipe was tuned for, and the budget comparison would be against a
    different experiment.
    """
    step = _step(tokens_per_optimizer_step=4 * SEQ)  # two passes per update
    assert step.accumulate_passes == 2
    batch = _batch()
    step.train_step(**batch)
    assert step.global_step == 0  # accumulated, not yet applied
    step.train_step(**batch)
    assert step.global_step == 1


def test_a_token_batch_no_whole_number_of_passes_reaches_is_rejected() -> None:
    """Otherwise the run silently trains at a batch size nobody configured."""
    config = NanoChatTrainStep.Config()
    config.model.max_seq_len = 100
    config.rows_per_pass = 3
    config.tokens_per_optimizer_step = 1_000
    with pytest.raises(ValueError, match="not divisible"):
        config.copy_tree().finalize()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows_per_pass", 0),
        ("rows_per_pass", -1),
        ("tokens_per_optimizer_step", 0),
        ("tokens_per_optimizer_step", -1),
        ("gradient_clip_norm", -1.0),
        ("divergence_threshold", 0.0),
        # NaN does not fail a `<= 0` test -- every comparison against it is
        # False -- so it slips through and DISABLES the guard it configures.
        ("gradient_clip_norm", float("nan")),
        ("divergence_threshold", float("nan")),
        ("momentum_start", 1.0),
        ("momentum_end", 1.5),
        ("momentum_start", -0.1),
    ],
)
def test_an_invalid_geometry_is_rejected_by_name(field: str, value: float) -> None:
    """Every bound states its own field, at config time.

    ``tokens_per_optimizer_step=0`` is the sharp one: it passes a divisibility
    check, then makes ``accumulate_passes`` zero, so the run divides the loss
    by zero rather than ever stepping. ``rows_per_pass=0`` reaches a modulo by
    zero inside ``finalize``, which also runs from ``pprint`` -- so a bare
    ZeroDivisionError there hides the whole config a reader was inspecting.
    """
    config = NanoChatTrainStep.Config()
    config.model.max_seq_len = 8
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        config.copy_tree().finalize()


def test_the_budget_clock_excludes_warmup_steps() -> None:
    """Compilation must not consume the budget it is supposed to precede."""
    step = _step(budget_warmup_steps=2)
    batch = _batch()
    step.train_step(**batch)
    step.train_step(**batch)
    assert step.elapsed_sec == 0.0
    step.train_step(**batch)
    assert step.elapsed_sec > 0.0


def test_resuming_does_not_rerun_the_budget_warmup() -> None:
    """A resumed run must not get free, uncharged training.

    The warmup exclusion is gated on ``local_step``, so a resume that reset it
    would grant ``budget_warmup_steps`` more steps that cost no budget -- and a
    run resumed often enough would train unboundedly on a fixed budget, which
    is exactly the comparison this baseline exists to make.
    """
    step = _step(budget_warmup_steps=2)
    batch = _batch()
    for _ in range(4):  # two warmup, two charged
        step.train_step(**batch)
    charged = step.elapsed_sec
    assert charged > 0.0

    resumed = _step(budget_warmup_steps=2)
    resumed.load_state_dict(step.state_dict())
    resumed.train_step(**batch)
    assert resumed.elapsed_sec > charged


def test_progress_drives_the_learning_rate() -> None:
    """Every schedule reads budget progress, not step index.

    A budgeted run does not know its step count in advance, so a step-indexed
    schedule could not be written -- and one that crept in would anneal
    against a horizon that does not exist.
    """
    step = _step(train_budget_sec=100.0)
    initial = step.optimizer.param_groups[0]["initial_lr"]
    step.train_step(**_batch())
    assert step.optimizer.param_groups[0]["lr"] == pytest.approx(initial)

    # Most of the budget spent: the trapezoid is into its decay.
    step.elapsed_sec = 75.0
    step.train_step(**_batch())
    assert step.optimizer.param_groups[0]["lr"] < initial


def test_every_optimizer_members_rate_is_reported() -> None:
    """One ``lr`` would name whichever member the composite happens to list first.

    The recipe runs two algorithms at rates an order of magnitude apart -- the
    orthogonalizing member's is the one it is tuned on -- so a single number
    reports one and hides the other.
    """
    step = _step()
    metrics = step.train_step(**_batch()).get("metrics", {})
    rates = {name: value for name, value in metrics.items() if name.startswith("lr_")}
    assert set(rates) == {"lr_fusedadamw", "lr_normuon"}
    assert rates["lr_normuon"] != rates["lr_fusedadamw"]


def test_weight_decay_anneals_with_the_budget() -> None:
    """Decay outliving the learning rate shrinks the final weights for nothing.

    Only the orthogonalizing member carries decay -- AdamW's is 0.0 in the
    recipe, since the tables it owns are mostly untouched by any one batch --
    so the annealing is asserted where there is something to anneal.
    """
    step = _step(train_budget_sec=100.0)
    decayed = [
        group
        for group in step.optimizer.param_groups
        if group.get("initial_weight_decay", 0.0) > 0
    ]
    assert decayed
    step.elapsed_sec = 99.0
    step.train_step(**_batch())
    for group in decayed:
        assert group["weight_decay"] < group["initial_weight_decay"]


def test_divergence_raises_rather_than_burning_the_budget() -> None:
    """A diverged language-model run does not recover.

    Left alone it would spend the whole budget proving that, and report a
    number as though it were a result.
    """
    step = _step(divergence_threshold=1e-6)
    with pytest.raises(RuntimeError, match="diverged"):
        step.train_step(**_batch())


def test_divergence_clears_the_pending_accumulation() -> None:
    """A caught divergence must not leave half a token batch behind.

    The guard zeroes the gradients, so the passes already accumulated are gone;
    leaving their COUNT would make the next update fire early, on a token batch
    smaller than the one the recipe is tuned against -- the invariant the
    divisibility check in ``finalize`` exists to hold.
    """
    step = _step(tokens_per_optimizer_step=4 * SEQ)  # two passes per update
    step.train_step(**_batch())
    assert step._pending_passes == 1

    step.config.divergence_threshold = 1e-6
    with pytest.raises(RuntimeError, match="diverged"):
        step.train_step(**_batch())
    assert step._pending_passes == 0


def test_eval_returns_per_token_loss_for_the_metric() -> None:
    """The metric weights each token by its byte length, so it needs them
    unreduced.
    """
    step = _step()
    out = step.eval_loss(**_batch())
    assert out["model"].shape == (2, SEQ)


def test_state_round_trips_including_the_clock() -> None:
    """A resumed run must not re-anneal from the top.

    The clock drives every schedule, so restarting it would undo the decay
    already applied and train the tail at the full learning rate.
    """
    step = _step()
    step.train_step(**_batch())
    step.elapsed_sec = 42.0
    state = step.state_dict()

    restored = _step()
    restored.load_state_dict(state)
    assert restored.global_step == step.global_step
    assert restored.elapsed_sec == 42.0
    for (name, a), (_, b) in zip(
        step.model.named_parameters(),
        restored.model.named_parameters(),
        strict=True,
    ):
        assert torch.equal(a, b), name


def test_a_compiled_run_checkpoints_in_the_form_it_reloads() -> None:
    """Compiling must not change what a checkpoint's keys are named.

    ``torch.compile`` returns a wrapper whose ``state_dict`` prefixes every key
    with ``_orig_mod``. Saving through one form and loading through the other
    fails on every parameter, which is a whole run lost at its first resume --
    and every other test here pins ``compile=None``, so nothing else sees it.
    """
    compiling = PartialConfig(torch.compile, backend="eager")
    step = _step(compile=compiling)
    step.train_step(**_batch())
    state = step.state_dict()
    assert not any(name.startswith("_orig_mod") for name in state["model"])

    restored = _step(compile=compiling)
    restored.load_state_dict(state)
    for (name, a), (_, b) in zip(
        step.model.named_parameters(),
        restored.model.named_parameters(),
        strict=True,
    ):
        assert torch.equal(a, b), name


def test_a_checkpoint_without_the_warmup_gate_is_refused() -> None:
    """A pre-fix checkpoint cannot say how much warmup it already spent.

    Resuming it would restart the exclusion and grant uncharged training, so
    the incompatibility is named rather than surfacing as a bare KeyError.
    """
    step = _step()
    state = step.state_dict()
    del state["local_step"]
    with pytest.raises(ValueError, match="local_step"):
        step.load_state_dict(state)


def test_a_partial_accumulation_is_dropped_at_a_boundary() -> None:
    """Gradients must not mix across a pass over the data."""
    step = _step(tokens_per_optimizer_step=4 * SEQ)
    step.train_step(**_batch())
    step.on_epoch_end()
    assert all(p.grad is None for p in step.model.parameters())


def test_the_recipes_schedule_holds_then_decays_to_zero() -> None:
    """Flat while there is budget left, zero exactly at the end.

    Pinned against the CURVE the config injects, so a change of default shape
    fails here rather than silently retuning the recipe.
    """
    schedule = NanoChatTrainStep.Config().schedule.make()
    assert schedule(0.0) == 1.0
    assert schedule(0.25) == 1.0
    assert schedule(0.75) == pytest.approx(0.5)
    assert schedule(1.0) == 0.0


def test_the_selector_is_comparable_not_a_closure() -> None:
    """A closure's repr carries an address, so a config holding one never
    equals its parent and every experiment diff shows a change.
    """
    assert matrix_parameters() == matrix_parameters()


def _smoke_step() -> NanoChatTrainStep:
    """``exp_smoke``'s step, built for a golden.

    The EXPERIMENT's config, not a hand-built one: a golden over a config
    assembled here would freeze whatever this file happens to say, and the
    ladder could then change underneath it without the golden noticing. Only
    the device is pinned, because the harness is CPU-only.
    """
    config = experiments.exp_smoke().step
    config.parallelism = NoParallel.Config(device="cpu")
    torch.manual_seed(0)
    built = config.make()
    assert isinstance(built, NanoChatTrainStep)
    return built


def _smoke_batch(step: NanoChatTrainStep) -> dict[str, Any]:
    """One batch at the step's own geometry."""
    model = step.config.model
    torch.manual_seed(1)
    rows = torch.randint(
        0,
        model.vocab_size,
        (step.config.rows_per_pass, model.max_seq_len + 1),
    )
    return {"media": rows[:, :-1], "label": rows[:, 1:]}


class _SmokeSteps(nn.Module):
    """A module wrapper so the bfb harness can drive five training steps.

    The harness randomizes ``parameters()`` and snapshots ``state_dict()``, so
    the thing it is handed has to BE the model. Wrapping rather than passing
    the model directly is what lets the optimizer -- whose moments are half of
    what this golden exists to freeze -- be constructed after that
    randomization and against those same tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.step = _smoke_step()
        self.model = self.step.model

    @override
    def forward(self, batch: dict[str, Any], clock: list[float]) -> torch.Tensor:
        """Run one step per clock reading; return the losses."""
        losses: list[torch.Tensor] = []
        for elapsed in clock:
            self.step.elapsed_sec = elapsed
            losses.append(self.step.train_step(**batch)["loss"].reshape(1))
        return torch.cat(losses)


def test_five_steps_bfb() -> None:
    """Freeze five optimizer steps of the recipe, end to end.

    The forward goldens in ``model_test`` freeze one pass. This freezes what
    that pass FEEDS: the backward, both optimizer members, the accumulated
    moment buffers, and the schedules. Minted over ``exp_smoke``, which differs
    from ``exp001`` only in size, so a change to any shared mechanism lands
    here.

    The budget clock is written before each step rather than left to run: every
    schedule reads ``elapsed_sec / train_budget_sec``, and that clock is a
    ``perf_counter`` reading (train_step.py:450, 483), so letting it run would
    freeze how fast this machine is. The readings span the whole budget because
    the trapezoid holds flat over the first half -- five closely-spaced ones all
    land at multiplier 1.0 and never exercise the decay.
    """
    budget = experiments.exp_smoke().step.train_budget_sec
    clock = [fraction * budget for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)]

    def run(module: nn.Module, batch: Any) -> torch.Tensor:
        assert isinstance(module, _SmokeSteps)
        return module(batch, clock)

    assert_bfb_against_golden(
        golden_dir=_GOLDEN_DIR,
        golden_name="five_steps_min_cpu",
        build_module=_SmokeSteps,
        build_input=lambda: _smoke_batch(_smoke_step()),
        seed=0,
        run=run,
    )


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
