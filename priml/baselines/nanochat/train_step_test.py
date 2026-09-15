"""Tests for the nanochat train step."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast, override

import math

from configgle import PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.nanochat import experiments, train_step
from priml.baselines.nanochat.attention import CausalAttention
from priml.baselines.nanochat.model import MemoryNanoChatLM
from priml.baselines.nanochat.optimizers import (
    BiasCorrectedRMSProp,
    FFNScaledNorMuon,
)
from priml.baselines.nanochat.train_step import (
    NanoChatTrainStep,
    matrix_parameters,
    nanochat_optimizer,
)
from priml.model.attention.value_gated_attention import (
    ValueGatedAttention,
    sdpa_attention,
)
from priml.model.linear import Linear
from priml.model.narrow_embedding import NarrowEmbedding
from priml.model.softcap import SoftCap
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.fused_adamw import FusedAdamW
from priml.testing.bfb import assert_bfb_against_golden
from priml.train.parallelism import NoParallel


_CWD: Final = Path(__file__).resolve().parent

VOCAB = 32
SEQ = 8


def test_reference_metric_preserves_native_reduction_and_two_denominators() -> None:
    """Reference accounting must not silently replace FP32 sums with FP64."""
    assert "ReferenceBitsPerByte" in vars(train_step)
    metric = train_step.ReferenceBitsPerByte.Config().make()
    losses = torch.tensor([[2**24, 1.0, 1.0]])
    mask = torch.ones_like(losses, dtype=torch.bool)
    metric.update(
        losses,
        score_mask=mask,
        evaluation_batch=0,
        evaluation_batches=1,
        reference_bytes=17,
        literal_bytes=13,
    )
    nats = float((losses.view(-1) * mask.view(-1)).sum())
    assert nats != float(losses.double().sum())
    assert metric.compute() == {
        "bpb": nats / (math.log(2) * 17),
        "literal_bpb": nats / (math.log(2) * 13),
    }


def test_reference_metric_requires_complete_ordered_batches() -> None:
    """Missing or duplicated reference rows cannot produce a valid score."""
    assert "ReferenceBitsPerByte" in vars(train_step)
    metric = train_step.ReferenceBitsPerByte.Config().make()
    losses = torch.ones(1, 2)
    batch = {
        "score_mask": torch.ones(1, 2, dtype=torch.bool),
        "evaluation_batch": 0,
        "evaluation_batches": 2,
        "reference_bytes": 2,
        "literal_bytes": 2,
    }
    metric.update(losses, **batch)
    with pytest.raises(ValueError, match="incomplete"):
        metric.compute()
    with pytest.raises(ValueError, match="order or extent"):
        metric.update(losses, **batch)


def test_bounded_loss_preserves_saved_precision_and_ignored_gradient() -> None:
    """The bounded loss uses narrow logits and subtracts targets before casting."""
    assert "BoundedTokenCrossEntropy" in vars(train_step)
    loss = train_step.BoundedTokenCrossEntropy.Config(dtype=torch.bfloat16).make()
    logits = torch.tensor([[[0.0, 1.0, -1.0], [1.0, 0.0, -1.0]]], requires_grad=True)
    labels = torch.tensor([[1, -1]])
    output = loss(logits, label=labels)["loss"]
    narrow = logits.detach().to(torch.bfloat16).float()
    lse = torch.log(torch.exp(narrow - 15.0).sum(-1)) + 15.0
    assert torch.equal(output, torch.stack((lse[0, 0] - 1, lse[0, 1] * 0))[None])
    output.sum().backward()
    expected = torch.exp(narrow - lse.unsqueeze(-1))
    expected[0, 0, 1] -= 1
    expected[0, 1] = 0
    assert logits.grad is not None
    assert torch.equal(logits.grad, expected.to(torch.bfloat16).float())


@pytest.mark.parametrize("bound", [-1.0, 0.0, 50.0, float("inf")])
def test_bounded_loss_rejects_an_unsafe_fixed_shift(bound: float) -> None:
    """The symmetric logit domain must stay inside normal FP32 exponentials."""
    with pytest.raises(ValueError, match="logit_upper_bound"):
        train_step.BoundedTokenCrossEntropy.Config(logit_upper_bound=bound).make()


@pytest.mark.parametrize("capped", [False, True])
def test_bounded_loss_requires_a_matching_symmetric_readout(capped: bool) -> None:
    """An upper bound alone cannot prevent very negative rows from underflowing."""
    config = train_step.NgramTrainStep.Config()
    config.loss = train_step.BoundedTokenCrossEntropy.Config()
    config.model.lm_head = SoftCap.Config(cap=1000) if capped else Linear.Config()
    with pytest.raises(ValueError, match=r"symmetric.*bound"):
        config.finalize()


def test_ngram_step_charges_the_receiving_update_after_warmup() -> None:
    """Loading update two is charged even before update two completes."""
    assert "NgramTrainStep" in vars(train_step)
    step = train_step.NgramTrainStep.__new__(train_step.NgramTrainStep)
    step.config = train_step.NgramTrainStep.Config(budget_warmup_steps=1)
    step.elapsed_sec = 0.0
    step._steps_this_process = 0
    step.charge_budget(5.0)
    assert step.elapsed_sec == 0.0
    step._steps_this_process = 1
    step.charge_budget(5.0)
    assert step.elapsed_sec == 5.0
    assert step.completed_updates == 1


class _EndpointTrajectory(nn.Module):
    """Two exp022 updates, retaining all eight layers at CPU-test dimensions."""

    def __init__(self) -> None:
        super().__init__()
        config = experiments.exp022().step
        assert isinstance(config, train_step.NgramTrainStep.Config)
        model = config.model
        assert isinstance(model, MemoryNanoChatLM.Config)
        model.vocab_size = 16
        model.channels_in = 16
        model.max_seq_len = 4
        model.dtype = torch.float32
        model.rope.dtype = torch.float32
        # Portable goldens widen Torch arithmetic; opaque fused operators require
        # fixed FP32 sinks. Native kernel coverage is separate from this CPU replay.
        model.fused_ngram = False
        model.ngram_dirty_clear = False
        embedding = model.embedding
        assert isinstance(embedding, NarrowEmbedding.Config)
        embedding.dtype = None
        assert isinstance(model.block, list)
        for block in model.block:
            attention = block.attn
            assert isinstance(attention, CausalAttention.Config)
            attention.channels_head = 8
            attention.gate_channels = 4
            attention.fused_qk_rope = False
            attention.window = model.max_seq_len if attention.window == 2048 else 2
            attention.kernel = PartialConfig(sdpa_attention)
        for table in (*model.bigrams.values(), *model.trigrams.values()):
            table.num_embeddings = 8
        config.parallelism = NoParallel.Config(device="cpu")
        config.compile = None
        config.dtype_autocast = None
        config.rows_per_pass = 2
        config.tokens_per_optimizer_step = config.rows_per_pass * model.max_seq_len
        loss = config.loss
        assert isinstance(loss, train_step.BoundedTokenCrossEntropy.Config)
        loss.dtype = torch.float32
        optimizer = config.optimizer
        assert isinstance(optimizer, CompositeOptimizer.Config)
        for member in optimizer.optimizers:
            if isinstance(
                member, (PartialConfig, BiasCorrectedRMSProp.Config, FusedAdamW.Config)
            ):
                member.compile = False
                if isinstance(member, BiasCorrectedRMSProp.Config):
                    member.sparse_rows = False
            elif isinstance(member, FFNScaledNorMuon.Config):
                member.channels_in = model.channels_in
                member.optimizer.compile = False
        step = config.make()
        assert isinstance(step, train_step.NgramTrainStep)
        self._step = step
        self.model = step.model

    @override
    def forward(self, rows: Tensor) -> Tensor:
        losses: list[Tensor] = []
        for progress in (0.0, 0.8):
            self._step.elapsed_sec = progress * self._step.config.train_budget_sec
            result = self._step.train_step(media=rows[:, :-1], label=rows[:, 1:])
            losses.append(result["loss"])
        return torch.cat(losses)


@pytest.mark.compute_training
def test_exp022_two_updates_match_portable_golden() -> None:
    """Pin the CPU reference model and optimizer interaction across two updates."""
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="exp022",
        build_module=_EndpointTrajectory,
        build_input=lambda: torch.arange(10).reshape(2, 5),
        seed=42,
    )


def _step(
    *, config: NanoChatTrainStep.Config | None = None, **overrides: object
) -> NanoChatTrainStep:
    if config is None:
        config = NanoChatTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    config.compile = None
    # The recipe's own optimizer, stepping eagerly. Dynamo traces each member's
    # kernel on first use and never caches it -- 9 of these tests' 9.5 seconds
    # -- and what the compiled graph changes is the update's last bits, which
    # the test artifacts and parity script measure and none of these assert.
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
    built = config.make()
    assert isinstance(built, NanoChatTrainStep)
    return built


@pytest.mark.parametrize("injected", [False, True])
@pytest.mark.parametrize("limit", [1.0, float("inf")])
def test_ngram_updates_clip_both_parameter_gradients_and_persistent_sinks(
    injected: bool, limit: float
) -> None:
    """The update policy must not bypass the configured global gradient norm."""
    step = _step(config=train_step.NgramTrainStep.Config(), gradient_clip_norm=limit)
    assert isinstance(step, train_step.NgramTrainStep)
    parameters = list(step.model.parameters())
    optimizer = BiasCorrectedRMSProp.Config().make()(parameters)
    step.optimizer = optimizer
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        group["initial_weight_decay"] = group["weight_decay"]
    parameters[0].grad = torch.ones_like(parameters[0])
    # Only the sink is consumed for this parameter; a stale .grad must not count.
    parameters[1].grad = torch.full_like(parameters[1], 1000)
    sink = torch.full_like(parameters[1], 2, dtype=torch.float32)
    optimizer.gradient_sinks[parameters[1]] = sink
    regular = parameters[0].grad
    initial_regular = regular.clone()
    initial_sink = sink.clone()
    before = nn.utils.get_total_norm([regular, sink])

    def update(step: train_step.NgramTrainStep) -> dict[str, float | Tensor]:
        del step
        return {"injected": 1.0}

    step._optimizer_update = update if injected else None
    metrics = step._apply_update()
    coefficient = torch.clamp(limit / (before + 1e-6), max=1.0)
    assert torch.equal(regular, initial_regular * coefficient)
    assert torch.equal(sink, initial_sink * coefficient)
    if math.isfinite(limit):
        measured = metrics["grad_norm"]
        assert isinstance(measured, Tensor)
        assert torch.equal(measured, before)
    else:
        assert "grad_norm" not in metrics


def _batch() -> dict[str, Tensor]:
    torch.manual_seed(1)
    rows = torch.randint(0, VOCAB, (2, SEQ + 1))
    return {"media": rows[:, :-1], "label": rows[:, 1:]}


@pytest.mark.compute_training
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


@pytest.mark.compute_training
def test_an_optimizer_step_waits_for_the_whole_token_batch() -> None:
    """Gradient accumulation is what holds the token batch fixed.

    A step that updated per pass would train at a different batch size than
    the recipe was tuned for, and the budget comparison would be against a
    different experiment.
    """
    step = _step(tokens_per_optimizer_step=4 * SEQ)  # Two passes per update.
    assert step.accumulate_passes == 2
    batch = _batch()
    step.train_step(**batch)
    assert step.global_step == 0  # Accumulated, not yet applied.
    step.train_step(**batch)
    assert step.global_step == 1


def test_checkpoint_refuses_a_partial_token_batch() -> None:
    """The incomplete token-batch gradients cannot be reconstructed on resume."""
    step = _step(tokens_per_optimizer_step=4 * SEQ)
    step.train_step(**_batch())

    with pytest.raises(RuntimeError, match="incomplete gradient accumulation"):
        step.state_dict()


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


@pytest.mark.compute_training
def test_the_budget_clock_excludes_warmup_steps() -> None:
    """Compilation must not consume the budget it is supposed to precede."""
    step = _step(budget_warmup_steps=2)
    batch = _batch()
    step.train_step(**batch)
    step.train_step(**batch)
    assert step.elapsed_sec == 0.0
    step.train_step(**batch)
    assert step.elapsed_sec > 0.0


def test_the_default_warmup_matches_the_reference() -> None:
    """The reference leaves ELEVEN of its own steps unbilled.

    Its ``step`` starts at 0 (``train.py:539``) and increments at the bottom
    of the loop (``:598``), so the ``if step > 10`` at ``:576`` is False for
    updates one through eleven. Ours tests a counter already incremented, so
    the same eleven needs the field to say eleven -- at ten we would charge
    one update the comparison gives away, and run it a step short.
    """
    assert NanoChatTrainStep.Config().budget_warmup_steps == 11


@pytest.mark.compute_training
def test_the_warmup_is_counted_in_steps_not_passes() -> None:
    """Otherwise accumulation silently shortens it by its own factor.

    The reference excludes eleven of its own steps (``train.py:576``, whose
    ``step`` increments below it). Counting passes here would exclude
    ``budget_warmup_steps / accumulate_passes`` -- 1.375 steps at the shipped
    geometry -- so the budget would pay for the compilation it skips.
    """
    step = _step(tokens_per_optimizer_step=4 * SEQ, budget_warmup_steps=1)
    assert step.accumulate_passes == 2
    batch = _batch()
    for _ in range(2):  # One whole optimizer step: the warmup.
        step.train_step(**batch)
    assert step.global_step == 1
    assert step.elapsed_sec == 0.0
    for _ in range(2):
        step.train_step(**batch)
    assert step.elapsed_sec > 0.0


@pytest.mark.compute_training
def test_loop_side_work_is_charged_to_the_budget_past_warmup() -> None:
    """Loading is training time: the reference's clock brackets it too.

    Its ``next(train_loader)`` sits at ``train.py:550``, between the ``t0`` at
    543 and the ``t1`` at 573. Ours happens in the loop, outside the step, and
    measured at 0.160 of 1.683 s/step -- so a budget that skipped it would buy
    a tenth more steps than the comparison granted.
    """
    step = _step(budget_warmup_steps=1)
    batch = _batch()
    step.charge_budget(5.0)
    assert step.elapsed_sec == 0.0  # Warmup: not yet charged.
    step.train_step(**batch)  # The warmup step.
    step.train_step(**batch)  # Past it, and itself charged.
    charged = step.elapsed_sec
    step.charge_budget(5.0)
    assert step.elapsed_sec >= charged + 5.0


@pytest.mark.compute_training
def test_resuming_does_not_rerun_the_budget_warmup() -> None:
    """A resumed run must not get free, uncharged training.

    The warmup exclusion is gated on ``local_step``, so a resume that reset it
    would grant ``budget_warmup_steps`` more steps that cost no budget -- and a
    run resumed often enough would train unboundedly on a fixed budget, which
    is exactly the comparison this baseline exists to make.
    """
    step = _step(budget_warmup_steps=2)
    batch = _batch()
    for _ in range(4):  # Two warmup, two charged.
        step.train_step(**batch)
    charged = step.elapsed_sec
    assert charged > 0.0

    resumed = _step(budget_warmup_steps=2)
    resumed.load_state_dict(step.state_dict())
    resumed.train_step(**batch)
    assert resumed.elapsed_sec > charged


@pytest.mark.compute_training
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


@pytest.mark.compute_training
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


@pytest.mark.compute_training
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


@pytest.mark.compute_training
def test_divergence_raises_rather_than_burning_the_budget() -> None:
    """A diverged language-model run does not recover.

    Left alone it would spend the whole budget proving that, and report a
    number as though it were a result.
    """
    step = _step(divergence_threshold=1e-6)
    with pytest.raises(RuntimeError, match="diverged"):
        step.train_step(**_batch())


@pytest.mark.compute_training
def test_a_diverged_pass_is_caught_at_the_batch_it_belongs_to() -> None:
    """The guard reads once per UPDATE, so a mid-batch pass must still abort.

    Its loss is reduced on device and read at the boundary, so a pass that
    diverged and then recovered is still caught -- the last pass alone says
    nothing about the ones behind it.

    Read AFTER the update, where the reference reads it (train.py:565): the
    read stalls the CPU on the device, so doing it first leaves the optimizer
    un-enqueued while waiting. The run therefore stops one update later, which
    is the same guarantee -- that update ran on gradients whose loss was
    finite, and the diverged batch never reaches a second one.
    """
    step = _step(tokens_per_optimizer_step=4 * SEQ)  # Two passes per update.
    step.config.divergence_threshold = 1e-6
    step.train_step(**_batch())  # The diverged pass, mid-batch.
    assert step.global_step == 0
    with pytest.raises(RuntimeError, match="diverged"):
        step.train_step(**_batch())
    assert step.global_step == 1


@pytest.mark.compute_training
def test_divergence_clears_the_pending_accumulation() -> None:
    """A caught divergence must not leave half a token batch behind.

    The guard zeroes the gradients, so the passes already accumulated are gone;
    leaving their COUNT would make the next update fire early, on a token batch
    smaller than the one the recipe is tuned against -- the invariant the
    divisibility check in ``finalize`` exists to hold.
    """
    step = _step(tokens_per_optimizer_step=4 * SEQ)  # Two passes per update.
    step.train_step(**_batch())
    assert step._pending_passes == 1

    step.config.divergence_threshold = 1e-6
    with pytest.raises(RuntimeError, match="diverged"):
        step.train_step(**_batch())
    assert step._pending_passes == 0


def test_eval_returns_per_token_loss_for_the_metric() -> None:
    """The metric weights each token by its byte length, so it needs them unreduced."""
    step = _step()
    out = step.eval_loss(**_batch())
    assert out["model"].shape == (2, SEQ)


@pytest.mark.compute_training
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


@pytest.mark.compute_torch_compile
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
    model_state = cast("dict[str, Tensor]", state["model"])
    assert not any(name.startswith("_orig_mod") for name in model_state)

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


@pytest.mark.compute_training
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
    """A closure's repr carries an address, so a config holding one never equals its.

    Parent and every experiment diff shows a change.
    """
    assert matrix_parameters() == matrix_parameters()


# The EXPERIMENT's config, not a hand-built one: a golden over a config assembled here
# would freeze whatever this file happens to say, and the ladder could then change
# underneath it without the golden noticing. Only the device is pinned, because the
# harness is CPU-only.
def _smoke_step() -> NanoChatTrainStep:
    """``exp_smoke``'s step, built for a golden."""
    config = experiments.exp_smoke().step
    config.parallelism = NoParallel.Config(device="cpu")
    torch.manual_seed(0)
    built = config.make()
    assert isinstance(built, NanoChatTrainStep)
    return built


def _smoke_batch(step: NanoChatTrainStep) -> dict[str, Tensor]:
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
    def forward(self, batch: dict[str, Tensor], clock: list[float]) -> Tensor:
        """Run one step per clock reading; return the losses."""
        losses: list[Tensor] = []
        for elapsed in clock:
            self.step.elapsed_sec = elapsed
            losses.append(self.step.train_step(**batch)["loss"].reshape(1))
        return torch.cat(losses)


@pytest.mark.compute_training
def test_five_steps_bfb() -> None:
    """Freeze five optimizer steps of the recipe, end to end.

    The forward test artifacts in ``model_test`` freeze one pass. This freezes
    what that pass FEEDS: the backward, both optimizer members, the accumulated
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

    def run(module: nn.Module, batch: dict[str, Tensor]) -> Tensor:
        assert isinstance(module, _SmokeSteps)
        return module(batch, clock)

    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="five_steps",
        build_module=_SmokeSteps,
        build_input=lambda: _smoke_batch(_smoke_step()),
        seed=0,
        run=run,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
