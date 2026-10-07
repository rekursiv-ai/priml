"""Tests for the nanochat train step."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast, override

import math

from configgle import PartialConfig
from torch import Tensor, nn

import pytest
import torch
import torch.distributed as dist

from priml.baselines.nanochat import experiments, train_step
from priml.baselines.nanochat.attention import CausalAttention
from priml.baselines.nanochat.model import (
    MemoryNanoChatLM,
    NanoChatLM,
    OutputNormFeedForward,
)
from priml.baselines.nanochat.optimizers import (
    BiasCorrectedRMSProp,
    FFNScaledNorMuon,
)
from priml.baselines.nanochat.train_step import (
    NanoChatTrainStep,
    matrix_parameters,
    nanochat_optimizer,
)
from priml.lib.codec import from_plain
from priml.math.schedules import trapezoidal
from priml.model.attention.value_gated_attention import (
    ValueGatedAttention,
    sdpa_attention,
)
from priml.model.linear import Linear
from priml.model.narrow_embedding import NarrowEmbedding
from priml.model.softcap import SoftCap
from priml.model.transformer.block import TransformerBlock
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.fused_adamw import FusedAdamW
from priml.testing.bfb import assert_bfb_against_golden
from priml.train.parallelism import NoParallel


_CWD: Final = Path(__file__).resolve().parent

VOCAB: Final = 32
SEQ: Final = 8


def test_reference_metric_accepts_single_byte_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The metric accepts a positive one-byte count."""
    metric = train_step.ReferenceBitsPerByte.Config().make()
    metric.update(
        torch.ones(2, 3),
        score_mask=torch.ones(2, 3, dtype=torch.bool),
        evaluation_batch=0,
        evaluation_batches=1,
        reference_bytes=1,
        literal_bytes=1,
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    assert metric.compute() == {
        "bpb": 6 / math.log(2),
        "literal_bpb": 6 / math.log(2),
    }


def test_reference_metric_refuses_a_sharded_evaluation_at_its_first_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second process is refused before the evaluation does any work."""
    metric = train_step.ReferenceBitsPerByte.Config().make()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    with pytest.raises(
        ValueError,
        match=r"^Exact reference evaluation requires one process\.$",
    ):
        metric.update(
            torch.ones(2, 3),
            score_mask=torch.ones(2, 3, dtype=torch.bool),
            evaluation_batch=0,
            evaluation_batches=1,
            reference_bytes=1,
            literal_bytes=1,
        )


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
    losses = torch.ones(3, 2)
    batch = {
        "score_mask": torch.ones(3, 2, dtype=torch.bool),
        "evaluation_batch": 0,
        "evaluation_batches": 2,
        "reference_bytes": 2,
        "literal_bytes": 2,
    }
    with pytest.raises(
        ValueError,
        match=r"^Reference evaluation batch order or extent changed\.$",
    ):
        metric.update(losses, **(batch | {"evaluation_batches": 0}))
    metric.update(losses, **batch)
    with pytest.raises(ValueError, match=r"^Reference evaluation is incomplete\.$"):
        metric.compute()
    with pytest.raises(
        ValueError,
        match=r"^Reference evaluation batch order or extent changed\.$",
    ):
        metric.update(losses, **batch)


def test_reference_metric_accumulates_batches_and_round_trips_state() -> None:
    """Every batch contributes, and reset discards the complete snapshot."""
    metric = train_step.ReferenceBitsPerByte.Config().make()
    losses = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [True, True]])
    for index, (reference_bytes, literal_bytes) in enumerate(((2, 1), (3, 2))):
        metric.update(
            losses + index,
            score_mask=mask,
            evaluation_batch=index,
            evaluation_batches=2,
            reference_bytes=reference_bytes,
            literal_bytes=literal_bytes,
        )
    expected = {
        "nats": 19.0,
        "bytes": 5,
        "literal_bytes": 3,
        "batches": 2,
        "expected_batches": 2,
    }
    assert metric.state_dict() == expected
    restored = train_step.ReferenceBitsPerByte.Config().make()
    restored.load_state_dict(expected)
    assert restored.state_dict() == expected
    assert restored.compute() == {
        "bpb": 19.0 / (math.log(2) * 5),
        "literal_bpb": 19.0 / (math.log(2) * 3),
    }
    restored.reset()
    assert restored.state_dict() == {
        "nats": 0.0,
        "bytes": 0,
        "literal_bytes": 0,
        "batches": 0,
        "expected_batches": 0,
    }
    with pytest.raises(ValueError, match=r"^Reference evaluation is incomplete\.$"):
        restored.compute()


def test_reference_metric_rejects_zero_denominators_and_bad_mask_metadata() -> None:
    """The metric rejects empty byte counts and either mask-contract violation."""
    metric = train_step.ReferenceBitsPerByte.Config().make()
    with pytest.raises(ValueError, match=r"^Reference evaluation is incomplete\.$"):
        metric.compute()
    metric.update(
        torch.ones(2, 3),
        score_mask=torch.ones(2, 3, dtype=torch.bool),
        evaluation_batch=0,
        evaluation_batches=1,
        reference_bytes=0,
        literal_bytes=1,
    )
    with pytest.raises(ValueError, match=r"^Reference evaluation is incomplete\.$"):
        metric.compute()
    for bad_mask in (torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 3)):
        metric.reset()
        with pytest.raises(
            ValueError,
            match=r"^Reference scoring mask differs from loss geometry\.$",
        ):
            metric.update(
                torch.ones(2, 3),
                score_mask=bad_mask,
                evaluation_batch=0,
                evaluation_batches=1,
                reference_bytes=2,
                literal_bytes=1,
            )


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
    config.finalize()  # Validation is make()'s, so the tree still prints.
    with pytest.raises(ValueError, match=r"symmetric.*bound"):
        config.make()


@pytest.mark.parametrize(
    "step_type",
    [NanoChatTrainStep, train_step.NgramTrainStep],
)
def test_loading_is_charged_to_the_update_it_feeds(
    step_type: type[NanoChatTrainStep],
) -> None:
    """Loading update two is charged even before update two completes."""
    step = step_type.__new__(step_type)
    step.config = step_type.Config(budget_warmup_steps=1)
    step.elapsed_sec = 0.0
    step._steps_this_process = 0
    step.charge_budget(5.0)
    assert step.elapsed_sec == 0.0
    step._steps_this_process = 1
    step.charge_budget(5.0)
    assert step.elapsed_sec == 5.0


@pytest.mark.parametrize(
    "config",
    [NanoChatTrainStep.Config(), train_step.NgramTrainStep.Config()],
)
def test_every_pass_of_the_first_billed_update_is_charged(
    config: NanoChatTrainStep.Config,
) -> None:
    """Warmup zero bills update one -- its first pass and its loading included."""
    step = _step(config=config, tokens_per_optimizer_step=4 * SEQ)
    assert step.accumulate_passes == 2
    step.charge_budget(5.0)
    assert step.elapsed_sec == 5.0
    step.train_step(**_batch())  # Update one's FIRST pass; it has not completed.
    assert step.global_step == 0
    assert step.elapsed_sec > 5.0


def test_an_injected_update_rejects_a_schedule_it_would_ignore() -> None:
    config = train_step.NgramTrainStep.Config()
    config.optimizer_update = PartialConfig(_no_update)
    config.schedule = PartialConfig(trapezoidal, flat=0.0)
    with pytest.raises(ValueError, match="schedule is ignored"):
        _step(config=config)


def _no_update(step: train_step.NgramTrainStep) -> dict[str, float | Tensor]:
    del step
    return {}


class _EndpointTrajectory(nn.Module):
    """Three exp022 updates over all eight layers."""

    def __init__(self) -> None:
        super().__init__()
        config = experiments.exp022().step
        assert isinstance(config, train_step.NgramTrainStep.Config)
        model = config.model
        assert isinstance(model, MemoryNanoChatLM.Config)
        model.vocab_size = 2
        # Trigram attention reads three gate slices, so
        # ``3 * gate_channels <= channels_in`` with gate width two.
        model.channels_in = 6
        model.max_seq_len = 3
        model.dtype = torch.float32
        model.rope.dtype = torch.float32
        # Portable replay widens Torch arithmetic; opaque fused operators require
        # fixed FP32 sinks. Native kernel coverage is separate from this CPU replay.
        model.fused_ngram = False
        model.ngram_dirty_clear = False
        embedding = model.embedding
        assert isinstance(embedding, NarrowEmbedding.Config)
        embedding.dtype = None
        assert isinstance(model.block, list)
        for block in model.block:
            assert isinstance(block, TransformerBlock.Config)
            attention = block.attn
            assert isinstance(attention, CausalAttention.Config)
            attention.channels_head = 2
            attention.gate_channels = 2
            attention.fused_qk_rope = False
            attention.window = model.max_seq_len if attention.window == 2048 else 2
            attention.kernel = PartialConfig(sdpa_attention)
            # A four-wide FFN retains the relu-square and output-norm paths.
            ffn = block.ffn
            assert isinstance(ffn, OutputNormFeedForward.Config)
            ffn.channels_hidden = 4
        for table in (*model.bigrams.values(), *model.trigrams.values()):
            table.num_embeddings = 4
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
                member,
                (PartialConfig, BiasCorrectedRMSProp.Config, FusedAdamW.Config),
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
        for progress in (0.0, 0.5, 1.0):
            self._step.elapsed_sec = progress * self._step.config.train_budget_sec
            result = self._step.train_step(media=rows[:, :-1], label=rows[:, 1:])
            losses.append(result["loss"])
        return torch.cat(losses)


@pytest.mark.compute_training
def test_exp022_three_updates_match_portable_golden() -> None:
    """Pin the CPU reference model and optimizer interaction across three updates."""
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="exp022",
        build_module=_EndpointTrajectory,
        build_input=lambda: torch.arange(8).remainder(2).reshape(2, 4),
        seed=42,
    )


def _step(
    *,
    config: NanoChatTrainStep.Config | None = None,
    **overrides: object,
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
    injected: bool,
    limit: float,
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
        {named[id(parameter)] for parameter in cast("list[Tensor]", group["params"])}
        for group in step.optimizer.param_groups
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
        config.make()


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
    by zero rather than ever stepping. ``rows_per_pass=0`` would reach a modulo
    by zero. Both are refused at ``make()``, never in ``finalize``, which also
    runs from ``pprint`` -- so the invalid tree still prints.
    """
    config = NanoChatTrainStep.Config()
    config.model.max_seq_len = SEQ
    setattr(config, field, value)
    config.copy_tree().pprint()
    with pytest.raises(ValueError, match=field):
        config.make()


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
    step = _step(tokens_per_optimizer_step=4 * 8, budget_warmup_steps=1)
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
    initial = from_plain(
        cast(object, step.optimizer.param_groups[0]["initial_lr"]),
        float,
    )
    step.train_step(**_batch())
    assert from_plain(
        cast(object, step.optimizer.param_groups[0]["lr"]),
        float,
    ) == pytest.approx(initial)

    # Most of the budget spent: the trapezoid is into its decay.
    step.elapsed_sec = 75.0
    step.train_step(**_batch())
    assert (
        from_plain(cast(object, step.optimizer.param_groups[0]["lr"]), float) < initial
    )


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
    # _step's model has value embeddings off, so drop_empty removes that member.
    assert set(rates) == {
        *(f"lr_{index}_fusedadamw" for index in range(4)),
        "lr_4_normuon",
    }
    assert rates["lr_4_normuon"] != rates["lr_0_fusedadamw"]


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
    assert out["model"].shape == (2, 8)


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
    state = dict(step.state_dict())
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
    """A closure's repr carries an address, so no two configs holding one are equal.

    Then every experiment would diff against its parent as changed.
    """
    assert matrix_parameters() == matrix_parameters()


# The experiment config carries the optimizer and schedule under test. Only the
# device is pinned because this replay runs on CPU.
def _smoke_step() -> NanoChatTrainStep:
    """Build the smoke experiment step on CPU."""
    config = experiments.exp_smoke().step
    config.parallelism = NoParallel.Config(device="cpu")
    model = config.model
    assert isinstance(model, NanoChatLM.Config)
    model.channels_in = 4
    model.max_seq_len = 2
    model.vocab_size = 2
    attention = model.template.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    attention.channels_head = 2
    attention.gate_channels = 4
    # Keep exp_smoke's two accumulation passes while shrinking the sequence.
    config.tokens_per_optimizer_step = 2 * config.rows_per_pass * model.max_seq_len
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
    """A module wrapper so the bfb harness can drive three training steps.

    The harness randomizes ``parameters()`` and snapshots ``state_dict()``, so
    the thing it is handed has to BE the model. Wrapping rather than passing
    the model directly lets the optimizer moments be constructed after
    randomization against those same tensors.
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
def test_three_steps_bfb() -> None:
    """Freeze three optimizer steps of the recipe, end to end.

    The forward test artifacts in ``model_test`` freeze one pass. This freezes
    what that pass FEEDS: the backward, both optimizer members, the accumulated
    moment buffers, and the schedules. Minted over ``exp_smoke``, which differs
    from ``exp001`` only in size, so a change to any shared mechanism lands
    here.

    The budget clock is written before each step rather than left to run: every
    schedule reads ``elapsed_sec / train_budget_sec``, and that clock is a
    ``perf_counter`` reading (train_step.py:450, 483), so letting it run would
    freeze how fast this machine is. The readings span the whole budget because
    the trapezoid holds flat over the first half and then decays.
    """
    budget = experiments.exp_smoke().step.train_budget_sec
    clock = [fraction * budget for fraction in (0.0, 0.5, 1.0)]

    def run(module: nn.Module, batch: dict[str, Tensor]) -> Tensor:
        assert isinstance(module, _SmokeSteps)
        return module(batch, clock)

    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="three_steps",
        build_module=_SmokeSteps,
        build_input=lambda: _smoke_batch(_smoke_step()),
        seed=0,
        run=run,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("budget_warmup_steps", -1, "budget_warmup_steps"),
        ("rows_per_pass", 0, "rows_per_pass"),
        ("tokens_per_optimizer_step", 0, "tokens_per_optimizer_step"),
        ("momentum_warmup_steps", 0, "momentum_warmup_steps"),
    ],
)
def test_train_config_rejects_invalid_accumulation_settings(
    field: str,
    value: int,
    message: str,
) -> None:
    config = NanoChatTrainStep.Config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=message):
        config.make()


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cuda_synchronize_path() -> None:
    step = _smoke_step()
    step.parallelism.device = torch.device("cuda")
    step._synchronize()


def test_no_pending_divergence_and_gradient_clipping_paths() -> None:
    step = _smoke_step()
    step._assert_not_diverged()
    step.config.gradient_clip_norm = float("inf")
    assert step._clip_gradients() == {}
    step.config.gradient_clip_norm = 1.0
    parameter = next(step.model.parameters())
    parameter.grad = torch.ones_like(parameter)
    result = step._clip_gradients()
    assert isinstance(result["grad_norm"], Tensor)


def test_fused_ngram_binding_routes_every_table_to_rmsprop() -> None:
    trajectory = _EndpointTrajectory()
    step = trajectory._step
    assert isinstance(step.model, MemoryNanoChatLM)
    assert isinstance(step.optimizer, CompositeOptimizer)
    tables = step.model._tables()
    for table in tables:
        table.prepare_gradient_sinks(dirty_bitmaps=True)
    step._bind_ngram_gradients(step.model)
    members = step.optimizer.optimizers
    rmsprop = [member for member in members if isinstance(member, BiasCorrectedRMSProp)]
    assert rmsprop
    expected: set[Tensor] = set()
    expected_sinks: dict[Tensor, Tensor] = {}
    expected_bitmaps: dict[Tensor, Tensor] = {}
    for table in tables:
        for part, sink, bitmap in zip(
            table.tables,
            table.gradient_sinks,
            table.gradient_bitmaps,
            strict=True,
        ):
            assert isinstance(part, nn.Embedding)
            expected.add(part.weight)
            expected_sinks[part.weight] = sink
            expected_bitmaps[part.weight] = bitmap
    actual_sinks = {
        parameter: sink
        for member in rmsprop
        for parameter, sink in member.gradient_sinks.items()
    }
    actual_bitmaps = {
        parameter: bitmap
        for member in rmsprop
        for parameter, bitmap in member.gradient_bitmaps.items()
    }
    assert set(actual_sinks) == expected
    assert all(
        actual_sinks[parameter] is sink for parameter, sink in expected_sinks.items()
    )
    assert actual_bitmaps.keys() == expected_bitmaps.keys()
    assert all(
        actual_bitmaps[parameter] is bitmap
        for parameter, bitmap in expected_bitmaps.items()
    )

    for table in tables:
        table.gradient_bitmaps = []
    step._bind_ngram_gradients(step.model)
    assert all(not member.gradient_bitmaps for member in rmsprop)

    tables[0].gradient_sinks.pop()
    with pytest.raises(ValueError, match="zip\\(\\) argument"):
        step._bind_ngram_gradients(step.model)

    tables[0].prepare_gradient_sinks(dirty_bitmaps=True)
    step._bind_ngram_gradients(step.model)
    first = rmsprop[0]
    first.param_groups[0]["params"] = []
    with pytest.raises(
        ValueError,
        match=r"^Every fused n-gram table must route to RMSProp\.$",
    ):
        step._bind_ngram_gradients(step.model)


def test_ngram_train_step_runs_one_cpu_update() -> None:
    step = _step(config=train_step.NgramTrainStep.Config())
    assert isinstance(step, train_step.NgramTrainStep)
    result = step.train_step(**_batch())
    assert result["loss"].shape == (1,)


def test_a_two_pass_update_charges_the_budget_and_guards_the_worst_pass() -> None:
    step = _step(tokens_per_optimizer_step=4 * SEQ)
    step.config.budget_warmup_steps = 0
    batch = _batch()
    # The update-boundary assertions need a short, valid sequence.
    batch["media"] = batch["media"][:, :2]
    batch["label"] = batch["label"][:, :2]
    first = step.train_step(**batch)
    assert step.global_step == 0
    assert not step.accumulation_complete
    assert first.get("metrics") == {}
    second = step.train_step(**batch)
    assert step.global_step == 1
    assert step.accumulation_complete
    assert second.get("metrics")
    assert step.elapsed_sec > 0
    step.config.divergence_threshold = 1e-6
    step.train_step(**batch)
    with pytest.raises(RuntimeError, match="diverged"):
        step.train_step(**batch)


def test_call_eval_and_partial_epoch_cleanup() -> None:
    step = _smoke_step()
    batch = _smoke_batch(step)
    logits = step.call_eval(**batch)
    assert logits.shape[:2] == batch["media"].shape
    step._pending_passes = 1
    step._pending_worst = torch.tensor(2.0)
    step.on_epoch_end()
    assert step.accumulation_complete
    assert step._pending_worst is None


def test_load_state_requires_budget_clock() -> None:
    step = _smoke_step()
    with pytest.raises(ValueError, match="local_step"):
        step.load_state_dict({})


def test_bounded_cross_entropy_masks_ignored_targets_and_has_gradients() -> None:
    loss = train_step.BoundedTokenCrossEntropy.Config().make()
    logits = torch.tensor([[[-2.0, 1.0, 3.0], [1.0, -1.0, 2.0]]], requires_grad=True)
    labels = torch.tensor([[2, -1]])
    result = loss(logits, label=labels)["loss"]
    expected = torch.logsumexp(logits[0, 0].detach(), 0) - logits[0, 0, 2].detach()
    # The logits literal is one sequence, so the per-token loss is [1, tokens].
    torch.testing.assert_close(
        result,
        torch.stack((expected, torch.tensor(0.0))).reshape(1, 2),
    )
    result.sum().backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad[0, 1], torch.zeros(3))


def test_train_step_rejects_invalid_budget_and_gradient_settings() -> None:
    config = NanoChatTrainStep.Config()
    config.train_budget_sec = 0
    assert "train_budget_sec=0" in config.copy_tree().finalize().pformat()
    with pytest.raises(ValueError, match="train_budget_sec"):
        config.make()
    config = NanoChatTrainStep.Config()
    config.gradient_clip_norm = 0
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        config.make()


def test_an_integer_rate_is_rescaled_with_width() -> None:
    config = NanoChatTrainStep.Config()
    config.model.channels_in = 192
    member = PartialConfig(FusedAdamW, lr=1, width_scaled=True)
    config.optimizer = CompositeOptimizer.Config(optimizers=[member])
    config.adam_lr_tuned_at_channels = 768
    finalized = config.finalize().optimizer
    assert isinstance(finalized, CompositeOptimizer.Config)
    (scaled,) = finalized.optimizers
    assert isinstance(scaled, PartialConfig)
    assert from_plain(cast(object, scaled.lr), float) == 1 / (192 / 768) ** 0.5


def test_a_scaled_member_without_a_numeric_rate_is_refused() -> None:
    config = NanoChatTrainStep.Config()
    member = PartialConfig(FusedAdamW, width_scaled=True)
    config.optimizer = CompositeOptimizer.Config(optimizers=[member])
    with pytest.raises(TypeError, match="numeric lr"):
        config.finalize()


def test_learning_rates_reports_a_plain_optimizer() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = torch.optim.SGD([parameter], lr=0.25)
    assert train_step._learning_rates(optimizer) == {"all": 0.25}


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
