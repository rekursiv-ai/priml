"""Replay frozen trajectories through the reference TRM recipes.

``testdata/<expNNN>.pt`` records each recipe as it runs through :func:`record`,
on one compact CPU configuration that keeps its architectural, optimizer,
precision, and recurrence branches. Compared whole with ``torch.equal``: every
parameter and persistent buffer after init and a fingerprint of the global RNG;
two train steps (loss, probe, every metric, and the ACT pool), their gradients,
post-update state, and EMA shadow, the first seeding it and the second
averaging; the carried latents; one padded evaluation under the EMA weights.

``testdata/train_step.pt`` is the trajectory minted from the original
implementation at port time, through :func:`record_original`; replaying it is
the evidence the port matches that implementation.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast
from unittest.mock import Mock

import traceback

from configgle import PartialConfig
from torch import Tensor, nn

import pytest
import torch
import torch.distributed as dist

from priml.baselines.arcagi import experiments
from priml.baselines.arcagi.model import ConvSwiGLU, RotaryBlock
from priml.baselines.arcagi.train_step import (
    ArcDataParallel,
    EvalSignals,
    PooledActTrainStep,
    _accuracy_metrics,
    _Correctness,
    _empty_eval,
)
from priml.baselines.sudoku.act import AtomicPool
from priml.model.attention.attention import Attention
from priml.model.grid_embedding import GridEmbedding
from priml.model.norm import RMSNorm
from priml.model.prefix import SparsePuzzleEmbedding
from priml.model.recurrence import DeepRecurrence
from priml.model.swiglu import SwiGLU
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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
