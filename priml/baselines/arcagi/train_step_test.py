"""Replay frozen trajectories through the reference and released recipes."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast
from unittest.mock import Mock

import traceback

from configgle import PartialConfig
from torch import Tensor, nn
from torch._inductor import config as inductor_config

import pytest
import torch
import torch.distributed as dist

from priml.baselines.arcagi import experiments
from priml.baselines.arcagi.data import (
    ArcInstanceMetadata,
)
from priml.baselines.arcagi.model import ConvSwiGLU, RotaryBlock
from priml.baselines.arcagi.train_step import (
    AfterOptimizerPoolTransition,
    ArcDataParallel,
    BeforeBackwardPoolTransition,
    EvalSignals,
    PooledActTrainStep,
    _accuracy_metrics,
    _Correctness,
    _empty_eval,
    apply_direct_schedule_to_body_and_sparse_optimizer_lrs,
)
from priml.baselines.sudoku.act import (
    AtomicPool,
    GlobalSampledMinimum,
    HaltTraining,
    StreamingPool,
)
from priml.loss.act import ACTLoss
from priml.math.schedules import cosine
from priml.model.attention.attention import Attention
from priml.model.conditioning import CategorySlotFiLM
from priml.model.conv import SequenceConv1d
from priml.model.grid_embedding import GridEmbedding
from priml.model.grid_solver import SingleLatentGridSolver
from priml.model.norm import RMSNorm
from priml.model.prefix import ComposedSparsePrefix, SparsePuzzleEmbedding
from priml.model.recurrence import DeepRecurrence, SingleLatentRecurrence
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon, adjust_lr_match_rms_adamw
from priml.runtime import MultiProcess
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
    assert_tensor_golden,
    joined,
    mismatches,
    put_steps,
    read_tensors,
    rng_fingerprint,
    stored,
)
from priml.train.ema import EMA
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter
    from priml.model.grid_solver import GridSolver
    from priml.optimizers.sign_sgd import SignSGD
    from priml.train.custom_types import TrainStepOutput


_CWD: Final = Path(__file__).resolve().parent

ORIGINAL: Final = _CWD / "testdata" / "train_step.pt"
"""``exp000``'s trajectory, minted from the original implementation at port time."""

REFERENCE_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "embed_tokens.": "embedding.embed_tokens.",
        "embed_feedback": "embedding.channels.0.embed_feedback",
        "q_head.": "halt_head.",
        "register_tokens": "prefix.register_tokens",
        "puzzle_emb.weights": "prefix.weights",
    },
)
"""The original implementation's parameter-name prefixes, to ours.

Every recorded name follows the original's; every other name is shared."""

RECIPES: Final = ("exp000", "exp001", "exp003", "exp005", "exp006")
"""The rungs with a step of their own: the reference, each new code path, the best.

exp002 changes only an optimizer rate, exp004's step is exp005's, and the
ARC-AGI-2 rungs run exp000's and exp005's steps; ``experiments_test`` pins those."""


def test_pool_transition_can_advance_before_backward() -> None:
    """Make source carry and halt state visible before backward begins."""
    transition = BeforeBackwardPoolTransition.Config().make()
    events: list[str] = []
    advance = partial(events.append, "advance")

    transition.before_backward(advance)
    events.append("backward")
    transition.after_optimizer(advance)

    assert events == ["advance", "backward"]


def test_pool_transition_default_advances_after_optimizer() -> None:
    """Keep the existing pooled ACT side-effect boundary as the default."""
    transition = AfterOptimizerPoolTransition.Config().make()
    events: list[str] = []
    advance = partial(events.append, "advance")

    transition.before_backward(advance)
    events.append("backward")
    events.append("optimizer")
    transition.after_optimizer(advance)

    assert events == ["backward", "optimizer", "advance"]


def test_pooled_step_maps_seated_ids_to_semantic_prefix_metadata() -> None:
    """Resolve task and view metadata by the pool's seated instance ids."""
    step = PortSubject("exp003").step
    metadata = (
        None,
        ArcInstanceMetadata(
            puzzle_name="task-a",
            task_id=19,
            dih_idx=5,
            colorperm_slots=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        ),
        ArcInstanceMetadata(
            puzzle_name="task-b",
            task_id=7,
            dih_idx=3,
            colorperm_slots=(8, 7, 6, 5, 4, 3, 2, 1, 0),
        ),
    )
    step.bind_instance_metadata(metadata)

    prefix = step._prefix_kwargs(torch.tensor([2, 1], dtype=torch.int64))

    assert set(prefix) == {
        "category_indices",
        "augmentation_indices",
        "slot_permutation",
        "puzzle_identifiers",
    }
    assert torch.equal(prefix["category_indices"], torch.tensor([7, 19]))
    assert torch.equal(prefix["augmentation_indices"], torch.tensor([3, 5]))
    assert torch.equal(
        prefix["slot_permutation"],
        torch.tensor([[8, 7, 6, 5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8]]),
    )
    assert torch.equal(prefix["puzzle_identifiers"], torch.tensor([2, 1]))


class Subject(Protocol):
    """One implementation of a recipe, seen through the golden's parameter names."""

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Return the trainable parameters under canonical names."""
        ...

    def state(self) -> dict[str, Tensor]:
        """Return the parameters and persistent buffers under canonical names."""
        ...

    def ema(self) -> dict[str, Tensor]:
        """Return the EMA shadow under canonical names; empty before it is seeded."""
        ...

    def pool(self) -> dict[str, Tensor]:
        """Return the ACT slot state."""
        ...

    def init_z(self, rows: int) -> tuple[Tensor, Tensor]:
        """Return the initial latents."""
        ...

    def forward(
        self,
        tokens: Tensor,
        z: tuple[Tensor, Tensor],
        ids: Tensor,
        *,
        single: bool,
    ) -> tuple[Tensor, ...]:
        """``(logits, halt, z_slow, z_fast)`` of a forward, or of one core pass."""
        ...

    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one training call."""
        ...

    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Run one evaluation call."""
        ...

    def call_eval(self, **batch: object) -> Tensor:
        """Return the evaluation logits."""
        ...


def port_config(recipe: str) -> PooledActTrainStep.Config:
    """Return the compact ``recipe`` step, its own precision kept.

    Args:
      recipe: One of :data:`RECIPES`.

    Returns:
      config: A CPU-sized step config.

    """
    step = cast(
        "Callable[[], experiments.Arc1TrainLoop]",
        getattr(experiments, recipe),
    )().step
    step.parallelism = NoParallel.Config(device="cpu")
    step.model.compile_core = None
    step.total_train_steps = 3
    step.warmup_steps = 0
    if isinstance(step.ema, EMA.Config):
        step.ema.update_after_step = 2
    pool = step.pool
    assert isinstance(pool, AtomicPool.Config)
    pool.batch_size = 2
    pool.max_steps = 2
    shrink_model(step.model)
    return step


class PortSubject:
    """The priml recipe under canonical names."""

    def __init__(self, recipe: str) -> None:
        self.step = PooledActTrainStep(port_config(recipe).finalize())

    @classmethod
    def from_config(cls, config: PooledActTrainStep.Config) -> PortSubject:
        """Wrap a step built from an already-edited port config."""
        subject = cls.__new__(cls)
        subject.step = PooledActTrainStep(config.finalize())
        return subject

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Return the trainable parameters under canonical names."""
        for name, parameter in self.step.model.named_parameters():
            yield canonical_name(name), parameter

    def state(self) -> dict[str, Tensor]:
        """Return the parameters and persistent buffers under canonical names."""
        return {
            canonical_name(k): v
            for k, v in self.step.model.state_dict().items()
            if k != "_dummy"
        }

    def ema(self) -> dict[str, Tensor]:
        """Return the EMA shadow under canonical names."""
        shadow = self.step.ema_shadow or {}
        return {canonical_name(k): v for k, v in shadow.items()}

    def pool(self) -> dict[str, Tensor]:
        """Return the ACT slot state."""
        pool = self.step.pool
        state = {
            "inputs": pool.inputs,
            "labels": pool.labels,
            "z_slow": pool.z_slow,
            "z_fast": pool.z_fast,
            "steps": pool.steps,
            "puzzle_ids": pool.puzzle_ids,
            "halted": pool.halted,
        }
        if pool.carry is not None:
            state["feedback"] = pool.feedback
        return state

    def init_z(self, rows: int) -> tuple[Tensor, Tensor]:
        """Return the initial latents."""
        return self.step.net.init_latents(rows)

    def forward(
        self,
        tokens: Tensor,
        z: tuple[Tensor, Tensor],
        ids: Tensor,
        *,
        single: bool,
    ) -> tuple[Tensor, ...]:
        """Forward (or one core pass) with the task ids the model reads."""
        net = self.step.net
        if self.step.puzzle_table is None:
            out = net.step(tokens, *z) if single else net(tokens, *z)
        elif single:
            out = net.step(tokens, *z, puzzle_identifiers=ids)
        else:
            out = net(tokens, *z, puzzle_identifiers=ids)
        return out.logits, out.halt, out.z_slow, out.z_fast

    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one training call."""
        return self.step.train_step(**batch)

    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Run one evaluation call."""
        return self.step.eval_loss(**batch)

    def call_eval(self, **batch: object) -> Tensor:
        """Return the evaluation logits."""
        return self.step.call_eval(**batch)


def canonical_name(name: str) -> str:
    """Map a priml parameter or buffer name to the one the golden records."""
    for recorded, port in REFERENCE_NAMES.items():
        if name.startswith(port):
            return recorded + name.removeprefix(port)
    return name


def batches() -> list[dict[str, object]]:
    """Three changing batches; step 2 is short, exercising the pad mask."""
    generator = torch.Generator().manual_seed(0)
    out: list[dict[str, object]] = []
    for index in range(3):
        media = torch.randint(
            2,
            4,
            (2, 3),
            generator=generator,
        )
        label = torch.randint(
            2,
            4,
            (2, 3),
            generator=generator,
        )
        media[:, -1:] = 0
        label[:, -1:] = -100
        out.append(
            {
                "media": media,
                "label": label,
                "puzzle_identifiers": torch.tensor(
                    [index % 2, (index + 1) % 2],
                    dtype=torch.int32,
                ),
                "valid_count": 1 if index == 1 else 2,
            },
        )
    return out


def record(subject: Subject) -> dict[str, Tensor]:
    """Run the unit protocol; call inside ``host_agnostic_numerics``.

    Args:
      subject: The implementation to record, built by :func:`unit_config`.

    Returns:
      trajectory: Flat name-to-tensor record.

    """
    out: dict[str, Tensor] = {"rng": rng_fingerprint()}
    states = [{"state": _joined(subject.state())}]
    data = batches()
    gradients: dict[str, Tensor] = {}
    handles = [
        parameter.register_post_accumulate_grad_hook(_capture(gradients, name))
        for name, parameter in subject.named_parameters()
        if parameter.requires_grad
    ]
    train: list[dict[str, Tensor]] = []
    pools: list[dict[str, Tensor]] = []
    grads: list[dict[str, Tensor]] = []
    emas: list[dict[str, Tensor]] = []
    try:
        # The second batch is short, so the pad mask reaches the loss.
        for batch in data[:2]:
            gradients.clear()
            train.append(flatten(subject.train_step(**batch)))
            pools.append(
                {k: v for k, v in subject.pool().items() if not k.startswith("z_")},
            )
            grads.append({"state": _joined(gradients)})
            states.append({"state": _joined(subject.state())})
            emas.append({"state": _joined(subject.ema())})
    finally:
        for handle in handles:
            handle.remove()
    put_steps(out, "train", train)
    put_steps(out, "pool", pools)
    put_steps(out, "grad", grads)
    put_steps(out, "state", states)
    put_steps(out, "ema", emas)
    # The latents are carried state, so their final value already depends on every
    # step before it.
    final = subject.pool()
    _put(out, "pool/final", {k: v for k, v in final.items() if k.startswith("z_")})
    evaluation = subject.eval_loss(**{**data[2], "valid_count": 1})
    out.update({f"eval/{key}": value for key, value in flatten(evaluation).items()})
    return out


def record_original(subject: Subject) -> dict[str, Tensor]:
    """Run the protocol the original implementation was recorded under.

    Args:
      subject: The implementation to record, built by :func:`port_config`.

    Returns:
      trajectory: Flat name-to-tensor record.

    """
    out: dict[str, Tensor] = {"rng": rng_fingerprint()}
    states = [{"state": _joined(subject.state())}]
    data = batches()
    tokens = cast(Tensor, data[0]["media"])
    ids = cast(Tensor, data[0]["puzzle_identifiers"])
    generator = torch.Generator().manual_seed(1)
    z_init = subject.init_z(2)
    z_random = tuple(torch.randn(z.shape, generator=generator) for z in z_init)
    with torch.no_grad():
        for label, single in (("forward", False), ("core", True)):
            outputs = [
                subject.forward(tokens, (z[0], z[1]), ids, single=single)
                for z in (z_init, z_random)
            ]
            put_steps(out, label, [_forward_record(o) for o in outputs])
    evaluation = [data[0], {**data[1], "valid_count": 1}]
    evaluations = [_evaluate(subject, evaluation)]
    gradients: dict[str, Tensor] = {}
    handles = [
        parameter.register_post_accumulate_grad_hook(_capture(gradients, name))
        for name, parameter in subject.named_parameters()
        if parameter.requires_grad
    ]
    train: list[dict[str, Tensor]] = []
    pools: list[dict[str, Tensor]] = []
    grads: list[dict[str, Tensor]] = []
    emas: list[dict[str, Tensor]] = []
    try:
        for index, batch in enumerate(data, start=1):
            gradients.clear()
            train.append(flatten(subject.train_step(**batch)))
            pools.append(
                {k: v for k, v in subject.pool().items() if not k.startswith("z_")},
            )
            if index in (1, 3):
                grads.append({"state": _joined(gradients)})
                states.append({"state": _joined(subject.state())})
                emas.append({"state": _joined(subject.ema())})
    finally:
        for handle in handles:
            handle.remove()
    put_steps(out, "train", train)
    put_steps(out, "pool", pools)
    put_steps(out, "grad", grads)
    put_steps(out, "state", states)
    put_steps(out, "ema", emas)
    # The latents are carried state, so their final value already depends on every
    # step before it.
    final = subject.pool()
    _put(out, "pool/final", {k: v for k, v in final.items() if k.startswith("z_")})
    evaluations.append(_evaluate(subject, evaluation))
    put_steps(out, "eval", evaluations)
    return out


def unit_config(recipe: str) -> PooledActTrainStep.Config:
    """Return :func:`port_config` cut to the unit protocol's two steps.

    The EMA seeds on the first step and averages on the second; one fast pass
    per cycle still runs the cycle's loop, and two slow cycles keep the
    gradient-free cycle ahead of the differentiated one.
    """
    step = port_config(recipe)
    assert isinstance(step.ema, EMA.Config)
    step.ema.update_after_step = 0
    assert isinstance(step.model.recurrence, DeepRecurrence.Config)
    step.model.recurrence.fast_cycles = 1
    return step


def run_port(recipe: str) -> dict[str, Tensor]:
    """Build the unit-sized recipe from seed 0 and record it."""
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        return record(PortSubject.from_config(unit_config(recipe)))


def golden_path(recipe: str) -> Path:
    """Where the golden for ``recipe`` lives."""
    return _CWD / "testdata" / f"{recipe}.pt"


@pytest.mark.parametrize("recipe", RECIPES)
def test_golden_replays_bit_for_bit(recipe: str) -> None:
    """The recipe reproduces the frozen trajectory with zero mismatches."""
    assert_tensor_golden(golden_path(recipe), run_port(recipe))


# Each bite records only up to the first quantity its change reaches: the
# replay above already shows an unchanged recipe reproduces every key.
@pytest.mark.parametrize("recipe", RECIPES)
def test_golden_bites_on_a_one_ulp_weight(recipe: str) -> None:
    """A one-ULP nudge to one initial weight is reported, not absorbed."""
    expected = read_tensors(golden_path(recipe))
    actual: dict[str, Tensor] = {}
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        subject = PortSubject.from_config(unit_config(recipe))
        # On the integer view: a float nudge is done in float64 here and rounds
        # straight back to the weight's own width.
        with torch.no_grad():
            weight = next(subject.step.model.parameters())
            bits = torch.int32 if weight.dtype == torch.float32 else torch.int16
            weight.view(bits).view(-1)[0] += 1
        put_steps(actual, "state", [{"state": _joined(subject.state())}])
    assert mismatches({"state/state": expected["state/state"][:1]}, actual)


@pytest.mark.parametrize("recipe", RECIPES)
def test_golden_bites_on_a_changed_halt_weight(recipe: str) -> None:
    """A changed halt-loss weight is reported in the first step's loss."""
    expected = read_tensors(golden_path(recipe))
    actual: dict[str, Tensor] = {}
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        subject = PortSubject.from_config(unit_config(recipe))
        assert subject.step.pool.halting is not None
        subject.step.pool.halting.weight *= 1.5
        put_steps(actual, "train", [flatten(subject.train_step(**batches()[0]))])
    assert mismatches(
        {"train/loss": expected["train/loss"][:1]},
        {"train/loss": actual["train/loss"]},
    )


@pytest.mark.compute_training
def test_the_original_implementations_trajectory_replays_bit_for_bit() -> None:
    """``exp000`` reproduces the trajectory minted from the original implementation.

    The record was minted from the original implementation at port time, so it
    is the one comparison against that implementation rather than against this
    port's own past output; no unit golden can stand in for it. It is never
    re-minted, so it is compared without the regeneration hook, and its bite is
    checked in the same run.
    """
    expected = read_tensors(ORIGINAL)
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        actual = record_original(PortSubject("exp000"))
        torch.manual_seed(0)
        subject = PortSubject("exp000")
        with torch.no_grad():
            next(subject.step.model.parameters()).view(torch.int32).view(-1)[0] += 1
        nudged = record_original(subject)
    assert not mismatches(expected, actual)
    assert mismatches(expected, nudged)


@pytest.mark.parametrize("recipe", RECIPES)
def test_an_all_padding_eval_batch_keeps_the_packed_width(recipe: str) -> None:
    """A rank's all-padding eval tail returns the same packed columns as any batch.

    The metric reads the header width from the column count and rejects any
    other, so a narrower zero-filled output would fail the whole evaluation on
    the rank that drew the padding.
    """
    batch = batches()[0]
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        step = PortSubject(recipe).step
        full = step.eval_loss(**batch)
        empty = step.eval_loss(**{**batch, "valid_count": 0})
    assert empty["model"].shape == full["model"].shape
    assert empty.get("metrics", {}).keys() == full.get("metrics", {}).keys()


def test_an_untrained_halt_head_leaves_the_eval_loss_to_the_tokens() -> None:
    """Without halt training the eval loss is the token loss alone."""
    config = port_config("exp000")
    config.pool.halting = None
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        out = PortSubject.from_config(config).step.eval_loss(**batches()[0])
        metrics = out.get("metrics", {})
        lm_loss = torch.as_tensor(metrics["lm_loss"])
        halt_loss = torch.as_tensor(metrics["q_halt_loss"])
    assert float(halt_loss) > 0
    loss = out["loss"].reshape(())
    assert torch.equal(loss, lm_loss.to(loss.dtype))


def test_eval_signal_header_widths() -> None:
    assert EvalSignals(EvalSignals.Config()).header_width(4) == 3
    assert EvalSignals(EvalSignals.Config(per_step=True)).header_width(4) == 13


def test_empty_eval_returns_exact_zero_contract() -> None:
    result = _empty_eval(2, 13, torch.device("cpu"), signals=True)

    assert result["loss"].shape == (1,)
    assert result["loss"].device == torch.device("cpu")
    assert torch.equal(result["loss"], torch.zeros(1))
    assert result["model"].shape == (2, 13)
    assert torch.equal(result["model"], torch.zeros(2, 13))
    assert result.get("metrics") == {
        "lm_loss": 0.0,
        "q_halt_loss": 0.0,
        "cell_accuracy": 0.0,
        "cell_weighted_accuracy": 0.0,
        "exact_accuracy": 0.0,
        "q_halt_accuracy": 0.0,
        "act_steps": 0.0,
        "q_continue_loss": 0.0,
        "grid_stable_frac": 0.0,
    }


def test_empty_eval_without_signals_has_exact_zero_contract() -> None:
    result = _empty_eval(3, 7, torch.device("cpu"), signals=False)

    assert result["loss"].shape == (1,)
    assert result["loss"].dtype == torch.float32
    assert result["loss"].device == torch.device("cpu")
    assert torch.equal(result["loss"], torch.zeros(1))
    assert result["model"].shape == (3, 7)
    assert result["model"].dtype == torch.float32
    assert result["model"].device == torch.device("cpu")
    assert torch.equal(result["model"], torch.zeros(3, 7))
    assert result.get("metrics") == {
        "lm_loss": 0.0,
        "q_halt_loss": 0.0,
        "cell_accuracy": 0.0,
        "cell_weighted_accuracy": 0.0,
        "exact_accuracy": 0.0,
        "q_halt_accuracy": 0.0,
        "act_steps": 0.0,
        "q_continue_loss": 0.0,
    }


def test_empty_eval_allocates_every_output_on_the_requested_device() -> None:
    result = _empty_eval(2, 13, torch.device("meta"), signals=True)

    assert result["loss"].device == torch.device("meta")
    assert result["model"].device == torch.device("meta")


def test_a_whole_model_compile_is_rejected() -> None:
    """The step calls the model directly, so a whole-model compile would be ignored."""
    config = port_config("exp000")
    config.compile = PartialConfig(torch.compile)
    with pytest.raises(ValueError, match="compile_core"):
        PooledActTrainStep(config.finalize())


def test_pooled_step_rejects_gradient_accumulation_before_construction() -> None:
    """Reject gradient accumulation before pooled step construction."""
    config = port_config("exp000")
    config.accumulate_grad_batches = 2

    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        PooledActTrainStep(config.finalize())


def test_pooled_step_rejects_nonfinite_gradient_skipping_before_construction() -> None:
    """Reject non-finite gradient skipping before pooled step construction."""
    config = port_config("exp000")
    config.skip_step_on_nonfinite_grad = True

    with pytest.raises(ValueError, match="skip_step_on_nonfinite_grad"):
        PooledActTrainStep(config.finalize())


def test_pooled_step_rejects_generic_lr_schedule_before_construction() -> None:
    """Reject a generic learning-rate schedule before pooled step construction."""
    config = port_config("exp000")
    config.lr_schedule = PartialConfig(cosine)

    with pytest.raises(ValueError, match="lr_schedule"):
        PooledActTrainStep(config.finalize())


def test_pooled_step_accepts_generic_train_step_defaults() -> None:
    """Keep generic step defaults for released pooled recipes."""
    step = PooledActTrainStep(port_config("exp000").finalize())

    assert step.accumulate_grad_batches == 1
    assert not step.skip_step_on_nonfinite_grad


def test_pooled_step_propagates_the_precision_cast_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set Inductor's process flag from the recipe before model construction."""
    monkeypatch.setattr(inductor_config, "emulate_precision_casts", False)
    config = port_config("exp005")
    assert config.emulate_precision_casts is True

    PooledActTrainStep(config.finalize())

    assert inductor_config.emulate_precision_casts is True


def test_pooled_step_places_the_act_pool_on_the_model_device() -> None:
    """Move every ACT slot tensor with the model under a non-CPU strategy."""
    config = port_config("exp000")
    config.parallelism = NoParallel.Config(device="meta")
    config.pool.halting = None
    config.pool.feedback = None
    step = PortSubject.from_config(config).step

    pool = step.pool
    tensors = (
        pool.inputs,
        pool.labels,
        pool.z_slow,
        pool.z_fast,
        pool.steps,
        pool.active,
        pool.halted,
        pool.puzzle_ids,
        pool.feedback,
    )
    assert step.device == torch.device("meta")
    assert all(tensor.device == step.device for tensor in tensors)


def test_charm_train_step_runs_injected_reducer_and_reporter_in_order() -> None:
    """Reduce gradients before stepping and report the returned metrics."""
    events: list[str] = []

    def reduce_gradients(model: nn.Module) -> None:
        events.append("reduce")
        assert any(parameter.grad is not None for parameter in model.parameters())

    def report_metrics(metrics: Mapping[str, Tensor | float]) -> dict[str, float]:
        del metrics
        events.append("report")
        return {"injected": 1.0}

    def mark_optimizer_step(optimizer: object, args: object, kwargs: object) -> None:
        del optimizer, args, kwargs
        events.append("step")

    config = port_config("exp009")
    config.model.prefix = None
    block = config.model.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    assert isinstance(block.ffn.middle, SequenceConv1d.Config)
    block.ffn.channels_hidden = 2
    block.ffn.middle.conv.groups = 2
    config.gradient_reduction = PartialConfig(reduce_gradients)
    config.metric_reporter = PartialConfig(report_metrics)
    # A bf16 convolution has no oneDNN backward on some CPUs; float64 does.
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        step = PortSubject.from_config(config).step
        assert isinstance(step.optimizer, CompositeOptimizer)
        for optimizer in step.optimizer.optimizers:
            optimizer.register_step_pre_hook(mark_optimizer_step)
        result = step.train_step(**batches()[0])

    assert events[0] == "reduce"
    assert events[-1] == "report"
    assert set(events[1:-1]) == {"step"}
    assert result.get("metrics") == {"injected": 1.0}


def test_pooled_step_initializes_ema_only_when_the_recipe_requests_it() -> None:
    """Keep legacy EMA lazy and CHARM's initial shadow explicit."""
    legacy = PortSubject("exp000").step
    assert isinstance(legacy.ema, EMA)
    assert legacy.ema.state_dict() == {"global_step": 0, "local_step": 0}

    for recipe in ("exp009", "exp010"):
        config = port_config(recipe)
        assert config.ema_initialization is EMA.initialize
        charm = PortSubject.from_config(config).step
        assert isinstance(charm.ema, EMA)
        shadow = charm.ema_shadow
        assert shadow is not None
        assert all(
            torch.equal(shadow[name], parameter)
            for name, parameter in charm.model.named_parameters()
        )


def shrink_model(model: GridSolver.Config) -> None:
    """Configure the compact model without changing recipe behavior."""
    model.channels_in = 4
    model.vocab_size = 4
    model.num_layers = 1
    assert isinstance(model.embedding, GridEmbedding.Config)
    model.embedding.grid_shape = (3,)
    assert isinstance(model.recurrence, DeepRecurrence.Config)
    model.recurrence.slow_cycles = 2
    model.recurrence.fast_cycles = 2
    block = model.block
    if isinstance(block, RotaryBlock.Config):
        assert isinstance(block.attn, Attention.Config)
        block.attn.num_heads = 2
        block.attn.channels_head = 4 // 2
        if isinstance(block.attn.norm_qk, RMSNorm.Config):
            block.attn.norm_qk.channels_in = 4 // 2
        assert block.rope is not None
        block.rope.channels_head = 4 // 2
        assert isinstance(block.ffn, SwiGLU.Config | ConvSwiGLU.Config)
        block.ffn.round_to = 4
    if isinstance(model.prefix, SparsePuzzleEmbedding.Config):
        model.prefix.num_puzzles = 2
        model.prefix.num_tokens = 2
        model.prefix.batch_size = 2
    if isinstance(model.prefix, ComposedSparsePrefix.Config):
        # One channel per slot and the base, as CategorySlotFiLM requires.
        model.prefix.channels_out = 10
        model.prefix.num_tokens = 1
        assert isinstance(model.prefix.composition, CategorySlotFiLM.Config)
        model.prefix.composition.num_categories = 2
        model.prefix.composition.channels_out = 10
        model.prefix.residual.num_puzzles = 3
        model.prefix.residual.channels_out = 3
        model.prefix.residual.batch_size = 2


def _evaluate(
    subject: Subject,
    evaluation: list[dict[str, object]],
) -> dict[str, Tensor]:
    """Record each evaluation batch's outputs and the rollout logits."""
    out: dict[str, Tensor] = {}
    for index, batch in enumerate(evaluation):
        for key, value in flatten(subject.eval_loss(**batch)).items():
            out[f"{index}/{key}"] = value
    out["call_eval"] = subject.call_eval(**evaluation[0])
    return out


def flatten(result: TrainStepOutput) -> dict[str, Tensor]:
    """Loss, probe, and every metric as tensors, float64 narrowed to float32.

    The stablemax loss runs in float64, which ``host_agnostic_numerics`` does
    not normalize: its last bit follows the host's libm. Rounding to float32
    is what absorbs that, as it does for every upcast float32 op.
    """
    out = {"loss": result["loss"], "model": result["model"]}
    for key, value in result.get("metrics", {}).items():
        out[f"metrics/{key}"] = torch.as_tensor(value)
    return {
        key: value.float() if value.dtype == torch.float64 else value
        for key, value in out.items()
    }


def _put(out: dict[str, Tensor], prefix: str, values: Mapping[str, Tensor]) -> None:
    """Store :func:`stored` copies of ``values`` under ``prefix``."""
    for key, value in values.items():
        out[f"{prefix}/{key}"] = stored(value)


def _forward_record(outputs: tuple[Tensor, ...]) -> dict[str, Tensor]:
    """Logits, halt, and the returned latents, all whole."""
    logits, halt, *latents = outputs
    return {"logits": logits, "halt": halt, "latents": joined(latents)}


def _joined(state: Mapping[str, Tensor]) -> Tensor:
    """Return every tensor, in name order, joined into one."""
    return joined(state[name] for name in sorted(state))


def _capture(store: dict[str, Tensor], name: str) -> Callable[[Tensor], None]:
    """Record a parameter's accumulated gradient, before clipping."""

    def capture(parameter: Tensor) -> None:
        assert parameter.grad is not None
        store[name] = parameter.grad.detach().clone()

    return capture


def test_optional_prefix_and_checkpoint_branches() -> None:
    config = port_config("exp000")
    subject = PortSubject.from_config(config)
    with pytest.raises(TypeError, match="needs puzzle_identifiers"):
        subject.step._prefix_kwargs(None)
    state = subject.step.state_dict()
    restored = PortSubject.from_config(config)
    restored.step.load_state_dict(state)
    assert restored.step.state_dict().keys() == state.keys()


def test_train_step_rejects_non_tensor_identifiers() -> None:
    step = PortSubject("exp000").step
    with pytest.raises(TypeError, match="must be a Tensor; got <class 'list'>"):
        step.train_step(**{**batches()[0], "puzzle_identifiers": [0, 1]})


def test_a_model_without_a_task_table_trains_without_identifiers() -> None:
    config = port_config("exp000")
    config.model.prefix = None
    torch.manual_seed(0)
    step = PortSubject.from_config(config).step
    assert step.puzzle_table is None
    assert step.sparse_optimizer is None
    assert step._prefix_kwargs(None) == {}
    batch = {k: v for k, v in batches()[0].items() if k != "puzzle_identifiers"}
    before = [p.detach().clone() for p in step.model.parameters()]
    step.train_step(**batch)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, step.model.parameters(), strict=True)
    )
    assert "sparse_optimizer" not in step.state_dict()


def test_train_loss_scores_a_batch_as_evaluation_does() -> None:
    batch = batches()[0]
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        step = PortSubject("exp000").step
        train = step.train_loss(**batch)
        evaluation = step.eval_loss(**batch)
    assert torch.equal(train["loss"], evaluation["loss"])
    assert torch.equal(train["model"], evaluation["model"])


def test_call_eval_returns_the_rollout_logits_eval_decodes() -> None:
    batch = batches()[0]
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        step = PortSubject("exp000").step
        logits = step.call_eval(
            media=batch["media"],
            puzzle_identifiers=batch["puzzle_identifiers"],
        )
        packed = step.eval_loss(**batch)["model"]
    assert logits.shape == (2, 3, 4)
    assert torch.equal(logits.argmax(dim=-1).float(), packed[:, 1:])
    with pytest.raises(ValueError, match="no positional arguments"):
        step.call_eval(batch["media"])


def test_eval_loss_accepts_label_free_arc_batches() -> None:
    """Evaluate source parquet rows without consulting their hidden targets."""
    step = PortSubject("exp003").step
    batch = {
        "media": torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int32),
        "valid_count": 2,
        "puzzle_identifiers": torch.tensor([0, 1], dtype=torch.int32),
        "canonical_outputs": [None, None],
    }

    result = step.eval_loss(**batch)

    assert "label" not in batch
    assert result["model"].shape == (2, 4)
    assert torch.equal(result["loss"], torch.zeros(1))
    assert result.get("metrics", {}).get("q_continue_loss") == 0.0


def test_sparse_optimizer_can_follow_the_step_learning_rate_schedule() -> None:
    """Scale only opted-in sparse groups while retaining legacy defaults."""
    config = port_config("exp000")
    config.warmup_steps = 10
    config.lr_step_offset = 1
    config.lr_min_ratio = 1.0
    config.optimizer_lr_policy = apply_direct_schedule_to_body_and_sparse_optimizer_lrs
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        step = PooledActTrainStep(config.finalize())
        assert step.sparse_optimizer is not None
        step._update({})
        assert step.sparse_optimizer.param_groups[0]["lr"] == 1e-3

        legacy_config = port_config("exp000")
        legacy_config.warmup_steps = 10
        legacy_config.lr_step_offset = 1
        legacy_config.lr_min_ratio = 1.0
        legacy_step = PooledActTrainStep(legacy_config.finalize())
        assert legacy_step.sparse_optimizer is not None
        legacy_step._update({})
        assert legacy_step.sparse_optimizer.param_groups[0]["lr"] == 1e-2


def test_three_step_source_golden_replays_through_the_shared_pooled_step() -> None:
    """Replay the immutable released trace through the canonical pooled step."""
    _replay_source_golden()


def test_three_step_source_golden_rejects_a_perturbed_sparse_rate() -> None:
    """Prove the source trajectory detects a changed live optimizer rate."""
    with pytest.raises(AssertionError, match=r"loss step|final model|optimizer|EMA"):
        _replay_source_golden(sparse_lr_multiplier=1.25)


def test_three_step_source_golden_rejects_extra_persistent_model_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject source model entries omitted by native persistent state."""
    original_read_tensors = read_tensors

    def read_with_extra_model_key(path: Path) -> dict[str, Tensor]:
        records = dict(original_read_tensors(path))
        records["init/model/inner.layers.99.mlp.down_proj.weight"] = torch.zeros(
            (2, 3),
        )
        return records

    monkeypatch.setattr(
        "priml.baselines.arcagi.train_step_test.read_tensors",
        read_with_extra_model_key,
    )
    with pytest.raises(AssertionError, match="initial model"):
        _replay_source_golden()


def _replay_source_golden(*, sparse_lr_multiplier: float = 1.0) -> None:
    """Replay the immutable three-update record, optionally perturbing sparse LR."""
    expected = read_tensors(_CWD / "testdata" / "exp009.pt")
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(13)
        step = PooledActTrainStep(_source_replay_config().finalize())
        model = step.net
        if not isinstance(model, SingleLatentGridSolver):
            raise TypeError("The source replay requires SingleLatentGridSolver.")
        table = step.puzzle_table
        sparse_optimizer = step.sparse_optimizer
        if table is None or sparse_optimizer is None:
            raise TypeError("The source replay requires sparse prefix state.")
        assert torch.equal(torch.random.get_rng_state(), expected["rng/init_state"])
        sparse_group = sparse_optimizer.param_groups[0]
        sparse_group["initial_lr"] = cast(float, sparse_group["lr"]) * (
            sparse_lr_multiplier
        )

        initial = _record_namespace(expected, "init/model")
        source_initial = {f"model.{name}": value for name, value in initial.items()}
        mapped_initial = {
            _source_target_key(name): value for name, value in source_initial.items()
        }
        _assert_tensor_records_equal(
            "initial model",
            mapped_initial,
            model.state_dict(),
        )

        step.bind_instance_metadata(
            (
                None,
                ArcInstanceMetadata(
                    puzzle_name="one",
                    task_id=0,
                    dih_idx=1,
                    colorperm_slots=(1, 0, 2, 3, 4, 5, 6, 7, 8),
                ),
                ArcInstanceMetadata(
                    puzzle_name="two",
                    task_id=1,
                    dih_idx=2,
                    colorperm_slots=tuple(reversed(range(9))),
                ),
            ),
        )
        batch = {
            "media": torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int32),
            "label": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.int32),
            "valid_count": 2,
            "puzzle_identifiers": torch.tensor([1, 2]),
        }
        for index, expected_loss in enumerate(expected["trace/losses"]):
            result = step.train_step(**batch)
            actual_loss = result["loss"].squeeze()
            assert torch.equal(actual_loss, expected_loss / 2), f"loss step {index + 1}"

        final = _record_namespace(expected, "final/model")
        source_final = {f"model.{name}": value for name, value in final.items()}
        mapped_final = {
            _source_target_key(name): value for name, value in source_final.items()
        }
        _assert_tensor_records_equal("final model", mapped_final, model.state_dict())

        parameter_names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        parameter_names.update(
            {id(buffer): name for name, buffer in model.named_buffers()},
        )
        dense = _normalize_source_optimizer(
            _record_namespace(expected, "final/optimizer/dense"),
        )
        if not isinstance(step.optimizer, CompositeOptimizer):
            raise TypeError("The source replay requires CompositeOptimizer.")
        optimizer = step.optimizer
        dense_actual = _normalize_native_optimizer(
            _optimizer_snapshot(optimizer, parameter_names),
            optimizer=optimizer,
        )
        _assert_source_optimizer_policy(optimizer=optimizer)
        _assert_tensor_records_equal("dense optimizer", dense, dense_actual)
        sparse = _normalize_source_optimizer(
            _record_namespace(expected, "final/optimizer/sparse"),
        )
        _assert_source_sparse_optimizer_policy(
            source=_record_namespace(expected, "final/optimizer/sparse"),
            optimizer=sparse_optimizer,
        )
        _assert_tensor_records_equal(
            "sparse optimizer",
            sparse,
            _optimizer_snapshot(sparse_optimizer, parameter_names),
        )
        for role, member in zip(("muon", "adam"), optimizer.optimizers, strict=True):
            _assert_tensor_records_equal(
                f"optimizer role {role}",
                _source_parameter_role(expected, role),
                _parameter_role_snapshot(member, parameter_names),
            )

        source_ema = _record_namespace(expected, "final/ema")
        mapped_ema = {
            _source_target_key(name): value for name, value in source_ema.items()
        }
        if not isinstance(step.ema, EMA):
            raise TypeError("The source replay config needs dense EMA.")
        _assert_tensor_records_equal("EMA", mapped_ema, step.ema.shadow_params)
        carry = _record_namespace(expected, "final/carry")
        assert torch.equal(step.pool.z_slow, carry["current_hidden"])
        assert torch.equal(step.pool.steps, carry["steps"])
        assert torch.equal(step.pool.halted, carry["halted"])
        assert torch.equal(step.pool.inputs, carry["inputs"].long())
        assert torch.equal(step.pool.labels, carry["labels"].long())
        assert torch.equal(step.pool.puzzle_ids, carry["puzzle_embed_idxs"].int())
        assert torch.equal(torch.rand(8), expected["rng/final_next8"])


def _source_replay_config() -> PooledActTrainStep.Config:
    """Shrink exp009 to the immutable source oracle's nondegenerate geometry."""
    step = experiments.exp009().step
    step.total_train_steps = 1_600_000
    step.warmup_steps = 2_000
    step.lr_step_offset = 1
    step.backward_loss_scale = 0.5
    step.metric_reporter = PartialConfig(
        experiments._report_source_act_metrics,
        global_batch_size=2,
    )
    step.dtype_autocast = None
    step.parallelism = NoParallel.Config(device="cpu")
    model = cast(SingleLatentGridSolver.Config, step.model)
    model.channels_in = 10
    model.num_layers = 1
    model.vocab_size = 5
    model.halt_outputs = 1
    model.compile_core = None
    embedding = cast(GridEmbedding.Config, model.embedding)
    embedding.grid_shape = (1, 4)
    block = cast(TransformerBlock.Config, model.block)
    attention = cast(Attention.Config, block.attn)
    attention.num_heads = 1
    attention.num_heads_kv = 1
    attention.channels_head = 10
    ffn = cast(SwiGLU.Config, block.ffn)
    ffn.channels_hidden = 2
    middle = cast(SequenceConv1d.Config, ffn.middle)
    middle.conv.groups = 2
    recurrence = cast(SingleLatentRecurrence.Config, model.recurrence)
    recurrence.slow_cycles = 1
    recurrence.fast_cycles = 2
    recurrence.inner_grad_loops = 1
    prefix = cast(ComposedSparsePrefix.Config, model.prefix)
    prefix.channels_out = 10
    prefix.num_tokens = 1
    composer = cast(CategorySlotFiLM.Config, prefix.composition)
    composer.num_categories = 2
    composer.channels_out = 10
    prefix.residual.num_puzzles = 3
    prefix.residual.channels_out = 3
    prefix.residual.num_tokens = 1
    prefix.residual.batch_size = 2
    pool = cast(AtomicPool.Config, step.pool)
    pool.batch_size = 2
    pool.max_steps = 3
    pool.steps_dtype = torch.int32
    pool.halting = HaltTraining.Config(
        weight=0.5,
        exploration=GlobalSampledMinimum.Config(prob=0.1),
    )
    return step


def _record_namespace(
    record: Mapping[str, Tensor],
    prefix: str,
) -> dict[str, Tensor]:
    """Return one slash-delimited source record namespace."""
    marker = f"{prefix}/"
    return {
        key.removeprefix(marker): value
        for key, value in record.items()
        if key.startswith(marker)
    }


def _source_parameter_name(source_name: str) -> str:
    """Map a source optimizer identity through the private release-name overlay."""
    source_name = source_name.removeprefix("model.inner.").removeprefix("inner.")
    if source_name == "puzzle_emb.instance_residual_embed.local_weights":
        return "prefix.residual.local_weights"
    if source_name == "puzzle_emb.instance_residual_embed.local_ids":
        return "prefix.residual.local_ids"
    return _source_target_key(f"model.inner.{source_name}")


def _normalize_source_optimizer(source: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename source identities while preserving all comparable optimizer tensors."""
    result: dict[str, Tensor] = {}
    for name, value in source.items():
        normalized_name = name
        if normalized_name.startswith("group/1/adamw_"):
            normalized_name = normalized_name.replace("group/1/adamw_", "group/1/", 1)
        if normalized_name.endswith("/muon_buffer"):
            normalized_name = (
                normalized_name.removesuffix("muon_buffer") + "momentum_buffer"
            )
        if normalized_name == "group/0/world_size":
            # Checked against native runtime size and aggregate_distributed
            # by _assert_source_sparse_optimizer_policy.
            continue
        if normalized_name.endswith("/parameter_order"):
            names = bytes(_integer_list(value.tolist())).decode().split("\0")
            mapped = [_source_parameter_name(item) for item in names]
            result[normalized_name] = torch.tensor(
                list("\0".join(mapped).encode()),
                dtype=torch.uint8,
            )
        elif normalized_name.startswith("state/"):
            body = normalized_name.removeprefix("state/")
            parameter, slot = body.rsplit("/", 1)
            result[f"state/{_source_parameter_name(parameter)}/{slot}"] = value
        else:
            result[normalized_name] = value
    return result


def _normalize_native_optimizer(
    native: Mapping[str, Tensor],
    *,
    optimizer: CompositeOptimizer,
) -> dict[str, Tensor]:
    """Map canonical optimizer state into the source replay's field names."""
    result: dict[str, Tensor] = {}
    for name, value in native.items():
        if name.startswith("group/0/") and name.rsplit("/", maxsplit=1)[-1] in {
            "ensemble_dims",
            "eps",
            "ns_coefficients",
            "reference_numerics",
        }:
            continue
        result[name] = value
    muon_index = next(
        (
            index
            for index, member in enumerate(optimizer.optimizers)
            if isinstance(member, Muon)
        ),
        None,
    )
    if muon_index is None:
        raise TypeError("The source replay requires a Muon member.")
    muon = optimizer.optimizers[muon_index]
    if not isinstance(muon, Muon):
        raise TypeError("The source replay requires a Muon member.")
    result["group/0/matched_adamw_rms"] = torch.tensor(
        0.2 if muon.adjust_lr_fn is adjust_lr_match_rms_adamw else 0.0,
        dtype=torch.float64,
    )
    for index, member in enumerate(optimizer.optimizers):
        result[f"group/{index}/use_muon"] = torch.tensor(
            isinstance(member, Muon),
            dtype=torch.bool,
        )
    return result


def _assert_source_optimizer_policy(*, optimizer: CompositeOptimizer) -> None:
    """Check native-only Muon metadata that implements the source contract."""
    muon = next(
        (member for member in optimizer.optimizers if isinstance(member, Muon)),
        None,
    )
    if muon is None:
        raise TypeError("The source replay requires a Muon member.")
    group = muon.param_groups[0]
    assert group["momentum"] == 0.95
    assert group["nesterov"] is True
    assert group["weight_decay"] == 0.1
    assert group["ensemble_dims"] == 0
    assert group["eps"] == 1e-7
    assert group["ns_coefficients"] == (3.4445, -4.775, 2.0315)
    assert group["ns_steps"] == 5
    assert group["reference_numerics"] is False
    auxiliary = next(
        member for member in optimizer.optimizers if not isinstance(member, Muon)
    )
    auxiliary_group = auxiliary.param_groups[0]
    assert auxiliary_group["betas"] == (0.9, 0.95)
    assert auxiliary_group["eps"] == 1e-8
    assert auxiliary_group["weight_decay"] == 0.1


def _assert_source_sparse_optimizer_policy(
    *,
    source: Mapping[str, Tensor],
    optimizer: SignSGD,
) -> None:
    """Compare the stored gather world with native runtime gather behavior."""
    source_world_size = source["group/0/world_size"]
    native_world_size = (
        dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    )
    assert torch.equal(
        source_world_size,
        torch.tensor(native_world_size, dtype=source_world_size.dtype),
    )
    assert optimizer.aggregate_distributed is True


def _optimizer_snapshot(
    optimizer: torch.optim.Optimizer,
    names_by_parameter: Mapping[int, str],
) -> dict[str, Tensor]:
    """Flatten each optimizer group, ordered identity, and complete state."""
    if isinstance(optimizer, CompositeOptimizer):
        composite_records: dict[str, Tensor] = {}
        group_offset = 0
        for member in optimizer.optimizers:
            member_records = _optimizer_snapshot(member, names_by_parameter)
            for member_name, value in member_records.items():
                normalized_name = member_name
                if member_name.startswith("group/"):
                    _, index, suffix = member_name.split("/", maxsplit=2)
                    normalized_name = f"group/{group_offset + int(index)}/{suffix}"
                composite_records[normalized_name] = value
            group_offset += len(member.param_groups)
        return composite_records
    state = cast(dict[str, object], optimizer.state_dict())
    groups = cast(list[dict[str, object]], state["param_groups"])
    all_state = cast(dict[int, dict[str, object]], state["state"])
    actual_groups = [cast(dict[str, object], group) for group in optimizer.param_groups]
    result: dict[str, Tensor] = {}
    for index, (group, actual) in enumerate(zip(groups, actual_groups, strict=True)):
        prefix = f"group/{index}"
        ids = cast(list[int], group["params"])
        parameters = cast(list[Tensor], actual["params"])
        names = [names_by_parameter[id(parameter)] for parameter in parameters]
        result[f"{prefix}/parameter_order"] = torch.tensor(
            list("\0".join(names).encode()),
            dtype=torch.uint8,
        )
        for name, value in group.items():
            if name in {"params", "initial_lr", "sparse_embedding", "world_size"}:
                continue
            if isinstance(value, bool):
                result[f"{prefix}/{name}"] = torch.tensor(value, dtype=torch.bool)
            elif isinstance(value, (int, float)) or (
                isinstance(value, tuple)
                and all(
                    isinstance(item, (int, float))
                    for item in cast(tuple[object, ...], value)
                )
            ):
                result[f"{prefix}/{name}"] = torch.tensor(value, dtype=torch.float64)
        for parameter_id, parameter_name in zip(ids, names, strict=True):
            for slot, value in all_state.get(parameter_id, {}).items():
                key = f"state/{parameter_name}/{slot}"
                if isinstance(value, Tensor):
                    result[key] = value.detach().clone()
                elif isinstance(value, (int, float)):
                    result[key] = torch.tensor(value, dtype=torch.float64)
    return result


def _source_parameter_role(
    source: Mapping[str, Tensor],
    role: str,
) -> dict[str, Tensor]:
    """Normalize a source role's ordered identities and tensor layouts."""
    prefix = f"optimizer/roles/{role}/"
    names = bytes(_integer_list(source[f"{prefix}names"].tolist())).decode().split("\0")
    mapped = [_source_parameter_name(name) for name in names]
    return {
        "names": torch.tensor(list("\0".join(mapped).encode()), dtype=torch.uint8),
        "shapes": source[f"{prefix}shapes"],
        "rank": source[f"{prefix}rank"],
    }


def _parameter_role_snapshot(
    optimizer: torch.optim.Optimizer,
    names_by_parameter: Mapping[int, str],
) -> dict[str, Tensor]:
    """Describe an optimizer member's parameter order, shape, and rank."""
    parameters = [
        parameter
        for raw_group in optimizer.param_groups
        for parameter in cast(
            list[Tensor],
            cast(dict[str, object], raw_group)["params"],
        )
    ]
    names = [names_by_parameter[id(parameter)] for parameter in parameters]
    return {
        "names": torch.tensor(list("\0".join(names).encode()), dtype=torch.uint8),
        "shapes": torch.tensor(
            [axis for parameter in parameters for axis in parameter.shape],
            dtype=torch.int64,
        ),
        "rank": torch.tensor([parameter.ndim for parameter in parameters]),
    }


def _assert_tensor_records_equal(
    label: str,
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
) -> None:
    """Require exact key, shape, dtype, and value equality for a record group."""
    differences: list[str] = []
    for name in sorted(expected.keys() | actual.keys()):
        wanted, found = expected.get(name), actual.get(name)
        if wanted is None or found is None:
            differences.append(f"{name}: missing={wanted is None}/{found is None}")
        elif wanted.shape != found.shape or wanted.dtype != found.dtype:
            differences.append(
                f"{name}: expected={wanted.shape}/{wanted.dtype} "
                f"actual={found.shape}/{found.dtype}",
            )
        elif not torch.equal(wanted, found):
            detail = (
                f" expected={wanted.tolist()} actual={found.tolist()}"
                if wanted.numel() <= 4
                else ""
            )
            differences.append(f"{name}: values differ{detail}")
    assert not differences, f"{label}:\n" + "\n".join(differences)


def _integer_list(values: list[object]) -> list[int]:
    """Narrow values returned by tensor-list conversion to integers."""
    result = [value for value in values if isinstance(value, int)]
    if len(result) != len(values):
        raise TypeError("Expected integer tensor records.")
    return result


def test_the_corruption_generator_round_trips_through_the_state_dict() -> None:
    torch.manual_seed(0)
    source = PortSubject("exp006").step
    carry = source.pool.carry
    assert carry is not None
    torch.rand(3, generator=carry.generator)
    state = source.state_dict()
    assert "corruption_rng" in state
    saved = state["corruption_rng"]
    assert torch.equal(saved, carry.generator.get_state())
    restored = PortSubject("exp006").step
    restored_carry = restored.pool.carry
    assert restored_carry is not None
    assert not torch.equal(restored_carry.generator.get_state(), saved)
    restored.load_state_dict(state)
    assert torch.equal(restored_carry.generator.get_state(), saved)


def test_accuracy_metrics_average_over_the_scored_slots_only() -> None:
    # Slot 0 alone is scored: one counted cell, correct, a zero halt logit.
    correctness = _Correctness(
        is_correct=torch.tensor([[True, False], [True, True], [False, False]]),
        loss_counts=torch.tensor([1, 2, 0]),
        correct=torch.tensor([1.0, 1.0, 0.0]),
    )
    metrics = _accuracy_metrics(
        correctness,
        halt=torch.tensor([0.0, -1.0, 2.0]),
        scored=torch.tensor([True, False, False]),
        steps=torch.tensor([3, 5, 7]),
    )
    assert {key: float(value) for key, value in metrics.items()} == {
        "cell_accuracy": 1.0,
        "exact_accuracy": 1.0,
        "q_halt_accuracy": 1.0,
        "act_steps": 3.0,
    }


def test_accuracy_metrics_treat_every_leading_axis_as_slots() -> None:
    generator = torch.Generator().manual_seed(0)
    is_correct = torch.rand(2, 3, 4, generator=generator) > 0.5
    # Power-of-two counts keep every per-slot fraction exact in any sum order.
    loss_counts = torch.tensor([[1, 2, 4], [4, 2, 1]])
    correct = (torch.rand(2, 3, generator=generator) > 0.5).float()
    halt = torch.randn(2, 3, generator=generator)
    scored = torch.tensor([[True, False, True], [True, True, False]])
    steps = torch.tensor([[1, 2, 3], [4, 5, 6]])
    batched = _accuracy_metrics(
        _Correctness(is_correct, loss_counts, correct),
        halt=halt,
        scored=scored,
        steps=steps,
    )
    flat = _accuracy_metrics(
        _Correctness(
            is_correct.flatten(0, 1),
            loss_counts.flatten(),
            correct.flatten(),
        ),
        halt=halt.flatten(),
        scored=scored.flatten(),
        steps=steps.flatten(),
    )
    assert batched.keys() == flat.keys()
    for key, value in batched.items():
        assert torch.equal(torch.as_tensor(value), torch.as_tensor(flat[key]))


def test_without_halt_training_the_loss_is_the_token_loss() -> None:
    config = port_config("exp000")
    config.pool.halting = None
    torch.manual_seed(0)
    step = PortSubject.from_config(config).step
    result = step.train_step(**batches()[0])
    metrics = result.get("metrics", {})
    assert "q_halt_loss" not in metrics
    assert "halted_frac" not in metrics
    lm_loss = torch.as_tensor(metrics["lm_loss"])
    assert torch.equal(result["loss"], lm_loss.reshape(1))


def test_pooled_step_accepts_the_composable_act_objective() -> None:
    """Use ACTLoss's summed objective through the shared pooled step."""
    config = port_config("exp000")
    config.objective = ACTLoss.Config(halt_weight=0.5)
    step = PortSubject.from_config(config).step
    logits = torch.tensor(
        [
            [[0.25, 1.0, -0.5], [1.5, -0.25, 0.75]],
            [[0.5, -1.0, 1.25], [-0.75, 0.5, 0.25]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[1, -100], [2, 0]])
    halt = torch.tensor([0.25, -0.5], requires_grad=True)
    active = torch.ones(2, dtype=torch.bool)
    expected = ACTLoss.Config(halt_weight=0.5).make()(
        logits=logits,
        labels=labels,
        halt_logits=halt,
    )
    step.pool.labels = labels

    actual, metrics, correctness = step._train_loss(
        logits,
        halt=halt,
        labels=labels,
        active=active,
    )

    assert torch.equal(actual, expected.loss)
    lm_loss = cast(Tensor, metrics["lm_loss"])
    assert torch.equal(lm_loss, expected.lm_loss.detach())
    assert correctness is not None
    assert torch.equal(correctness.correct, expected.correct.float())


def test_before_backward_streaming_transition_preserves_forward_values() -> None:
    """A pre-backward transition can reseat slots without mutating saved inputs."""
    config = port_config("exp000")
    config.model.prefix = None
    config.objective = ACTLoss.Config()
    config.pool = StreamingPool.Config(batch_size=2, max_steps=1)
    config.pool_transition = BeforeBackwardPoolTransition.Config()
    step = PortSubject.from_config(config).step
    media = torch.tensor(
        [[2, 3, 0], [3, 2, 0], [2, 2, 0], [3, 3, 0]],
    )
    labels = torch.tensor(
        [[3, 2, -100], [2, 3, -100], [3, 3, -100], [2, 2, -100]],
    )

    step.train_step(media=media, label=labels, valid_count=4)

    assert torch.equal(step.pool.inputs, media[2:])
    assert torch.equal(step.pool.labels, labels[2:])
    assert step.pool.active.all()
    next_media = torch.tensor([[3, 2, 0], [2, 3, 0]])
    next_labels = torch.tensor([[2, 3, -100], [3, 2, -100]])
    step.train_step(media=next_media, label=next_labels, valid_count=2)
    assert step.global_step == 2


def test_pooled_step_rejects_two_ignored_label_owners() -> None:
    """Pool refill and ACT objective must share one ignored-label ID."""
    config = port_config("exp000")
    config.ignore_label_id = 0
    config.objective = ACTLoss.Config()

    with pytest.raises(ValueError, match=r"ignore_label_id.*ignore_index"):
        PooledActTrainStep(config.finalize())


def test_a_finite_clip_norm_bounds_the_body_update() -> None:
    config = port_config("exp000")
    config.gradient_clip_norm = 1e-3
    torch.manual_seed(0)
    step = PortSubject.from_config(config).step
    clipped: list[Tensor] = []

    def capture(optimizer: object, args: object, kwargs: object) -> None:
        del optimizer, args, kwargs
        clipped.extend(
            p.grad.detach().clone()
            for p in step.model.parameters()
            if p.grad is not None
        )

    assert isinstance(step.optimizer, torch.optim.Optimizer)
    step.optimizer.register_step_pre_hook(capture)
    metrics = step.train_step(**batches()[0]).get("metrics", {})
    norm = metrics["grad_norm"]
    assert isinstance(norm, Tensor)
    assert float(norm) > 1e-3
    total = torch.linalg.vector_norm(torch.stack([g.norm() for g in clipped]))
    torch.testing.assert_close(total, torch.tensor(1e-3), rtol=1e-4, atol=0.0)


def test_arc_data_parallel_broadcasts_every_tensor_from_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=(tmp_path / "rendezvous").resolve().as_uri(),
        rank=0,
        world_size=1,
    )
    runtime = MultiProcess.Config(device="cpu", mesh_topology={"dp": 1}).make()
    try:
        runtime.initialize()
        broadcast = Mock(wraps=dist.broadcast)
        monkeypatch.setattr(dist, "broadcast", broadcast)
        model = nn.Linear(2, 3)
        model.register_buffer("scale", torch.ones(3))
        parallel = ArcDataParallel.Config().make()
        replicated = parallel(model)
        tensors = [*model.parameters(), *model.buffers()]
        assert [call.args[0] for call in broadcast.call_args_list] == tensors
        assert all(
            call.kwargs == {"src": 0, "group": parallel.process_group}
            for call in broadcast.call_args_list
        )
        assert replicated is model
    finally:
        runtime.destroy()
        dist.destroy_process_group()


@pytest.mark.compute_distributed
def test_a_warm_eval_before_training_leaves_replication_trainable(
    tmp_path: Path,
    warm_pools: WarmPoolGetter,
) -> None:
    """The loop's warm eval is the first forward, and it runs in inference mode.

    Replication builds its gradient buckets on the first forward, so buckets
    built there are inference tensors and the first backward cannot write them.
    """
    warm_pools({"dp": 2})(partial(_warm_eval_then_train, tmp_path))
    failures = [path.read_text() for path in sorted(tmp_path.glob("rank*.txt"))]
    assert not failures, "\n".join(failures)
    assert len(list(tmp_path.glob("rank*.ok"))) == 2


def _warm_eval_then_train(root: Path, mesh: DeviceMesh) -> None:
    """Run one eval batch, then one train step, on this rank."""
    runtime = MultiProcess.Config(
        device="cpu",
        mesh_topology={"dp": 2, "pp": 1, "tp": 1},
    ).make()
    rank = mesh.get_rank()
    try:
        runtime.initialize()
        config = port_config("exp000")
        config.parallelism = ArcDataParallel.Config(gradient_as_bucket_view=True)
        step = PooledActTrainStep(config.finalize())
        batch = batches()[0]
        step.eval_loss(**batch)
        step.train_step(**batch)
        (root / f"rank{rank}.ok").touch()
    except RuntimeError:
        (root / f"rank{rank}.txt").write_text(traceback.format_exc())
    finally:
        runtime.destroy()


def _source_target_key(source_name: str) -> str:
    """Map immutable source-oracle names into the native recipe namespace."""
    key = source_name
    for prefix in ("_orig_mod.model.inner.", "model.inner.", "inner."):
        if key.startswith(prefix):
            key = key.removeprefix(prefix)
            break
    direct = {
        "embed_tokens.embedding_weight": "embedding.embed_tokens.weight",
        "lm_head.weight": "head.weight",
        "q_head.weight": "halt_head.weight",
        "q_head.bias": "halt_head.bias",
        "init_hidden": "slow_init",
        "puzzle_emb.instance_residual_embed.local_weights": "prefix.residual.local_weights",
        "puzzle_emb.instance_residual_embed.local_ids": "prefix.residual.local_ids",
        "puzzle_emb.instance_residual_embed.weights": "prefix.residual.weights",
        "puzzle_emb.instance_residual_up_proj.weight": "prefix.residual_projection.weight",
        "puzzle_emb.interaction_gate.weight": "prefix.residual_gate.linear.weight",
        "puzzle_emb.interaction_gate.bias": "prefix.residual_gate.linear.bias",
    }
    mapped = direct.get(key)
    if mapped is not None:
        return mapped
    if key.startswith("embed_feedback"):
        return key.replace("embed_feedback", "embedding.channels.0.embed_feedback", 1)
    if key == "register_tokens":
        return "prefix.register_tokens"
    if key.startswith("puzzle_emb."):
        composition = {
            "puzzle_embed.weight": "category_embed.weight",
            "category_embed.weight": "category_embed.weight",
            "dih_embed.weight": "augmentation_embed.weight",
            "augmentation_embed.weight": "augmentation_embed.weight",
            "colorperm_base": "slot_base",
            "colorperm_slots": "slot_embeddings",
            "slot_base": "slot_base",
            "slot_embeddings": "slot_embeddings",
            "film_gamma.weight": "film_gamma.weight",
            "film_gamma.bias": "film_gamma.bias",
            "film_beta.weight": "film_beta.weight",
            "film_beta.bias": "film_beta.bias",
        }
        suffix = key.removeprefix("puzzle_emb.")
        target = composition.get(suffix)
        if target is not None:
            return f"prefix.composition.{target}"
    if key.startswith("layers."):
        parts = key.split(".", maxsplit=3)
        if len(parts) == 4 and parts[2] in {"self_attn", "mlp"}:
            layer, child, suffix = parts[1], parts[2], parts[3]
            target_child = "attn" if child == "self_attn" else "ffn"
            projection = (
                {
                    "qkv_proj.weight": "proj_qkv.weight",
                    "o_proj.weight": "proj_out.weight",
                }
                if child == "self_attn"
                else {
                    "gate_up_proj.weight": "up_proj.weight",
                    "dwconv.weight": "middle.conv.weight",
                    "dwconv.bias": "middle.conv.bias",
                    "down_proj.weight": "down_proj.weight",
                }
            ).get(suffix)
            if projection is not None:
                return f"reasoning.{layer}.{target_child}.{projection}"
    raise ValueError(f"Unexpected source-oracle model key {source_name!r}.")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
