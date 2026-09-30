"""Replay frozen trajectories through the reference TRM recipes.

The goldens in ``testdata/<expNNN>[_<precision>].pt`` record each recipe
through :func:`record`; this module imports none of the implementation it
reproduces. Every recipe uses one compact CPU configuration while retaining
its architectural, optimizer, precision, and recurrence branches.

Recorded per recipe, all compared whole with ``torch.equal``: every parameter
and persistent buffer after init and a fingerprint of the global RNG; one
forward and one single-core step; an evaluation rollout before and after
training (loss, packed output, every metric, and the rollout logits); three
train steps (loss, probe, every metric, the ACT pool, and its final latents);
gradients and post-update state, EMA shadow included, after the first and
third.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

from configgle import PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.arcagi1 import experiments
from priml.baselines.arcagi1.act import AtomicPool
from priml.baselines.arcagi1.model import ConvSwiGLU
from priml.baselines.arcagi1.train_step import TrmTrainStep
from priml.baselines.arcagi2.model import RotaryBlock
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.model import DeepRecurrence, SudokuNet
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.model.attention.self_attention import SelfAttention
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
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

    from priml.train.custom_types import TrainStepOutput


_CWD: Final = Path(__file__).resolve().parent
WIDTH: Final = 4
HEADS: Final = 2
VOCAB: Final = 4
PUZZLES: Final = 2
GRID: Final = 3
BATCH: Final = 2
MAX_STEPS: Final = 2
TRAIN_STEPS: Final = 3
SNAPSHOT_STEPS: Final = (1, TRAIN_STEPS)

RECIPES: Final = ("exp004", "exp005", "exp007", "exp008")
"""The reference recipes: each reproduces a published TRM run."""

PRECISIONS: Final = ("fp32", "bf16_autocast")
"""``fp32`` runs without autocast; ``bf16_autocast`` keeps float32 masters and
autocasts forwards to bfloat16, as every recipe here sets it."""

CASES: Final = tuple(
    (recipe, precision) for precision in PRECISIONS for recipe in RECIPES
)


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


def port_config(recipe: str, precision: str = "fp32") -> TrmTrainStep.Config:
    """Return the compact ``recipe`` step in one precision arm.

    Args:
      recipe: One of :data:`RECIPES`.
      precision: One of :data:`PRECISIONS`.

    Returns:
      config: A CPU-sized step config.

    """
    step = cast(
        "Callable[[], experiments.TrmTrainLoop]",
        getattr(experiments, recipe),
    )().step
    step.parallelism = NoParallel.Config(device="cpu")
    step.model.compile_core = None
    step.dtype_autocast = None if precision == "fp32" else torch.bfloat16
    step.total_train_steps = 3
    step.warmup_steps = 0
    if isinstance(step.ema, EMA.Config):
        step.ema.update_after_step = 2
    pool = step.pool
    assert isinstance(pool, AtomicPool.Config)
    pool.batch_size = 2
    pool.max_steps = 2
    _configure_model(step.model)
    return step


class PortSubject:
    """The priml recipe under canonical names."""

    def __init__(self, recipe: str, precision: str = "fp32") -> None:
        self.step = TrmTrainStep(port_config(recipe, precision).finalize())

    @classmethod
    def from_config(cls, config: TrmTrainStep.Config) -> PortSubject:
        """Wrap a step built from an already-edited port config."""
        subject = cls.__new__(cls)
        subject.step = TrmTrainStep(config.finalize())
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
    for port, recorded in (
        ("embedding.embed_tokens.", "embed_tokens."),
        ("embedding.channels.0.embed_feedback", "embed_feedback"),
        ("halt_head.", "q_head."),
        ("prefix.register_tokens", "register_tokens"),
        ("prefix.weights", "puzzle_emb.weights"),
    ):
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
    """Run the recorded protocol; call inside ``host_agnostic_numerics``.

    Args:
      subject: The implementation to record.

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


def run_port(recipe: str, precision: str = "fp32") -> dict[str, Tensor]:
    """Build and record the priml recipe from seed 0."""
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        return record(PortSubject(recipe, precision))


def golden_path(recipe: str, precision: str = "fp32") -> Path:
    """Where the golden for ``recipe`` in ``precision`` lives."""
    suffix = "" if precision == "fp32" else f"_{precision}"
    return _CWD / "testdata" / f"{recipe}{suffix}.pt"


def load_golden(recipe: str, precision: str = "fp32") -> dict[str, Tensor]:
    """Load a frozen golden."""
    return read_tensors(golden_path(recipe, precision))


@pytest.mark.parametrize(("recipe", "precision"), CASES)
def test_golden_replays_bit_for_bit(recipe: str, precision: str) -> None:
    """The recipe reproduces the frozen trajectory with zero mismatches."""
    report = mismatches(load_golden(recipe, precision), run_port(recipe, precision))
    assert not report, "\n".join(report)


@pytest.mark.parametrize("perturb", ["halt_weight", "parameter"])
@pytest.mark.parametrize(("recipe", "precision"), CASES)
def test_golden_bites(recipe: str, precision: str, perturb: str) -> None:
    """A changed constant or a one-ULP weight nudge is reported, not absorbed."""
    expected = load_golden(recipe, precision)
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        subject = PortSubject(recipe, precision)
        if perturb == "halt_weight":
            assert subject.step.halting is not None
            subject.step.halting.weight *= 1.5
        else:
            # On the integer view: a float nudge is done in float64 here and
            # rounds straight back to the weight's own width.
            with torch.no_grad():
                weight = next(subject.step.model.parameters())
                bits = torch.int32 if weight.dtype == torch.float32 else torch.int16
                weight.view(bits).view(-1)[0] += 1
        actual = record(subject)
    assert mismatches(expected, actual)


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


def test_a_whole_model_compile_is_rejected() -> None:
    """The step calls the model directly, so a whole-model compile would be ignored."""
    config = port_config("exp004")
    config.compile = PartialConfig(torch.compile)
    with pytest.raises(ValueError, match="compile_core"):
        TrmTrainStep(config.finalize())


def _configure_model(model: SudokuNet.Config) -> None:
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
        assert isinstance(block.attn, SelfAttention.Config)
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
    config = port_config("exp004")
    subject = PortSubject.from_config(config)
    with pytest.raises(TypeError, match="needs puzzle_identifiers"):
        subject.step._prefix_kwargs(None)
    state = subject.step.state_dict()
    restored = PortSubject.from_config(config)
    restored.step.load_state_dict(state)
    assert restored.step.state_dict().keys() == state.keys()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
