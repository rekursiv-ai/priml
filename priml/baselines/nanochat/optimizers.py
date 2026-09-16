"""Sparse table updates, matrix learning rates, and NanoChat optimizer schedules.

Triton parses source annotations as device code. Keep tuple annotations quoted
and omit future annotations, which makes the formatter remove those quotes.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache, partial
from importlib import import_module
from types import FunctionType
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    TypedDict,
    cast,
    overload,
    override,
)

import math

from configgle import Fig
from torch import Tensor, nn
from torch._dynamo import config as torch_dynamo_config
from torch.optim import Optimizer, optimizer

import torch

from priml.lib.custom_json import FloatCodec
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.normuon import NorMuon
from priml.train.custom_types import OptimizerProtocol


if TYPE_CHECKING:
    from triton import language
    from triton.language.extra.cuda import libdevice

    import triton
else:
    from wrapt import lazy_import

    triton = lazy_import("triton")
    language = lazy_import("triton.language")
    libdevice = lazy_import("triton.language.extra.cuda.libdevice")


class _ScheduleConfig(Protocol):
    """The momentum-warmup knobs the update policy reads off a step's config."""

    @property
    def momentum_warmup_steps(self) -> int: ...
    @property
    def momentum_start(self) -> float: ...
    @property
    def momentum_end(self) -> float: ...


class _ScheduledTrainStep(Protocol):
    """What :class:`ScheduledOptimizerUpdate` needs of the step it drives.

    Read-only properties so a step whose ``optimizer`` is typed by the wider
    ``OptimizerProtocol`` still conforms; the policy narrows to
    :class:`CompositeOptimizer` at the call site.
    """

    @property
    def config(self) -> _ScheduleConfig: ...
    @property
    def progress_learning_schedule(self) -> float: ...
    @property
    def completed_updates(self) -> int: ...
    @property
    def optimizer(self) -> OptimizerProtocol: ...
    @property
    def model(self) -> nn.Module: ...


class BiasCorrectedRMSProp(Optimizer):
    """Apply bias-corrected RMSProp with elementwise or rowwise second moments."""

    class Config(Fig["Callable[..., BiasCorrectedRMSProp]"]):
        lr: float = 0.01
        """Table learning rate."""

        beta2: float = 0.99
        """Second-moment decay, optionally changed by the optimizer schedule."""

        eps: float = 1e-8
        """Denominator floor, after the bias-corrected square root."""

        weight_decay: float = 0.0
        """Decoupled parameter decay."""

        compile: bool = False
        """Fuse the update while keeping scheduled scalars as CPU tensors."""

        rowwise: bool = False
        """Average squared gradients across each row into an FP32 second moment."""

        sparse_rows: bool = False
        """Update marked rows and advance all row moments with eager-moment kernels."""

        @override
        def make(self) -> "Callable[..., BiasCorrectedRMSProp]":
            return partial(BiasCorrectedRMSProp, config=self.copy_tree().finalize())

    def __init__(self, params: Iterable[Tensor], *, config: Config) -> None:
        self.rowwise = config.rowwise
        self.sparse_rows = config.sparse_rows
        self.gradient_sinks: dict[Tensor, Tensor] = {}
        self.gradient_bitmaps: dict[Tensor, Tensor] = {}
        super().__init__(
            params,
            {
                "lr": config.lr,
                "beta2": config.beta2,
                "eps": config.eps,
                "weight_decay": config.weight_decay,
            },
        )
        self.scalars = {
            name: torch.zeros((), dtype=torch.float32, device="cpu")
            for name in ("step", "lr", "beta2", "eps", "weight_decay")
        }
        self.update = (
            torch.compile(_rmsprop_update, fullgraph=True, dynamic=False)
            if config.compile
            else _rmsprop_update
        )

    class StateDict(TypedDict):
        """Torch optimizer checkpoint payload."""

        state: dict[int, optimizer.StateDict]
        param_groups: list[optimizer.StateDict]

    @override
    def state_dict(self) -> optimizer.StateDict:
        raw = super().state_dict()
        state: BiasCorrectedRMSProp.StateDict = {
            "state": raw["state"],
            "param_groups": raw["param_groups"],
        }
        return {**state}

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore moments and indices without casting them to the weight dtype.

        Args:
          state_dict: Checkpoint produced by this optimizer's state_dict method.

        """
        incoming: dict[str, object] = {}

        def capture(optimizer: Optimizer, state: dict[str, object]) -> None:
            del optimizer
            incoming.update(state)

        # Capture AFTER user pre-hooks and restore BEFORE user post-hooks. The base
        # still validates groups and handles their metadata, but its blanket cast to
        # parameter dtype must not quantize FP32 moments or integer sparse buffers.
        with (
            self.register_load_state_dict_pre_hook(capture),
            self.register_load_state_dict_post_hook(
                partial(self._restore_state_precision, incoming), prepend=True
            ),
        ):
            super().load_state_dict(cast(optimizer.StateDict, state_dict))

    def _restore_state_precision(
        self, incoming: dict[str, object], optimizer: Optimizer
    ) -> None:
        del optimizer
        groups = cast("list[dict[str, object]]", incoming["param_groups"])
        states = cast("dict[int, dict[str, object]]", incoming["state"])
        for saved, current in zip(groups, self.param_groups, strict=True):
            indices = cast("list[int]", saved["params"])
            parameters = cast("list[Tensor]", current["params"])
            for index, parameter in zip(indices, parameters, strict=True):
                if index in states:
                    self.state[parameter] = cast(
                        "dict[str, object]",
                        _copy_optimizer_state(states[index], parameter.device),
                    )

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], Tensor | float]) -> Tensor | float: ...

    @override
    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], Tensor | float] | None = None,
    ) -> Tensor | float | None:
        loss: Tensor | float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group_values = cast("dict[str, object]", group)
            for parameter in cast("list[Tensor]", group_values["params"]):
                gradient = self.gradient_sinks.get(parameter, parameter.grad)
                if gradient is None:
                    continue
                state = cast("dict[str, object]", self.state[parameter])
                if not state:
                    state["step"] = 0
                    state["second_moment"] = (
                        torch.zeros(
                            (*parameter.shape[:-1], 1),
                            dtype=torch.float32,
                            device=parameter.device,
                        )
                        if self.rowwise
                        else torch.zeros_like(parameter)
                    )
                step_count = cast(int, state["step"]) + 1
                state["step"] = step_count
                if self.sparse_rows:
                    self._sparse_step(parameter, gradient, state, group)
                    continue
                self.scalars["step"].fill_(step_count)
                self.scalars["lr"].fill_(FloatCodec.coerce(group_values["lr"], None))
                self.scalars["beta2"].fill_(
                    FloatCodec.coerce(group_values["beta2"], None)
                )
                self.scalars["eps"].fill_(FloatCodec.coerce(group_values["eps"], None))
                self.scalars["weight_decay"].fill_(
                    FloatCodec.coerce(group_values["weight_decay"], None)
                )
                self.update(
                    parameter,
                    gradient,
                    cast(Tensor, state["second_moment"]),
                    rowwise=self.rowwise,
                    **self.scalars,
                )
        return loss

    def _sparse_step(
        self,
        parameter: Tensor,
        gradient: Tensor,
        state: dict[str, object],
        group: dict[str, object],
    ) -> None:
        """Update flagged rows and decay idle moments with rowwise state."""
        bitmap = self.gradient_bitmaps.get(parameter)
        if bitmap is None:
            raise ValueError(
                "sparse_rows is set but this table has no dirty bitmap; refusing to "
                "fall back to the dense path and report it as a sparse step",
            )
        weight_decay = FloatCodec.coerce(group["weight_decay"], None)
        if weight_decay != 0.0:
            raise ValueError(
                f"sparse_rows does not implement decoupled decay; group carries "
                f"weight_decay={weight_decay}",
            )
        # The sparse kernel requires a contiguous table and one FP32 moment per row.
        if parameter.dim() != 2:
            raise ValueError(f"expected a 2-D table, got {tuple(parameter.shape)}")
        if gradient.shape != parameter.shape:
            raise ValueError(
                f"grad {tuple(gradient.shape)} != table {tuple(parameter.shape)}",
            )
        if not self.rowwise:
            raise ValueError(
                "sparse_rows requires rowwise: the kernel reads one "
                "second-moment entry per row",
            )
        moment = cast(Tensor, state["second_moment"])
        rows = parameter.shape[0]
        if tuple(moment.shape) != (rows, 1) or moment.dtype != torch.float32:
            raise ValueError(
                f"second moment must be per-row fp32 [rows, 1], got "
                f"{tuple(moment.shape)} {moment.dtype}",
            )
        for name, tensor in (("table", parameter), ("grad", gradient)):
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous for the row-subset kernel")
        if tuple(bitmap.shape) != (rows,):
            raise ValueError(
                f"bitmap must be one flag per row [{rows}], got {tuple(bitmap.shape)}",
            )
        if "sparse_index" not in state:
            device = parameter.device
            state["sparse_index"] = torch.zeros(rows, dtype=torch.int32, device=device)
            state["sparse_count"] = torch.zeros((), dtype=torch.int32, device=device)
            state["sparse_scratch"] = torch.zeros(
                rows + 1,
                dtype=torch.int32,
                device=device,
            )
            state["last_step"] = torch.zeros(rows, dtype=torch.int32, device=device)
            state["last_cum"] = torch.zeros(rows, dtype=torch.float32, device=device)
            state["cum_log"] = 0.0
            state["sparse_scalars"] = {
                name: torch.zeros((), dtype=torch.float32, device=device)
                for name in (
                    "step",
                    "lr",
                    "beta2",
                    "eps",
                    "one_minus_lr_wd",
                    "cum_before",
                    "cum_after",
                )
            }
        # Round before the bias correction; using the Python float changes updates.
        beta2 = float(
            torch.tensor(FloatCodec.coerce(group["beta2"], None), dtype=torch.float32)
        )
        cum_before = cast(float, state["cum_log"])
        state["cum_log"] = cum_before + (math.log(beta2) if beta2 != 0.0 else -math.inf)
        lr = FloatCodec.coerce(group["lr"], None)
        scalars = cast("dict[str, Tensor]", state["sparse_scalars"])
        for name, value in (
            ("step", float(cast(int, state["step"]))),
            ("lr", lr),
            ("beta2", beta2),
            ("eps", FloatCodec.coerce(group["eps"], None)),
            ("one_minus_lr_wd", 1.0 - lr * weight_decay),
            ("cum_before", cum_before),
            ("cum_after", state["cum_log"]),
        ):
            scalars[name].fill_(value)
        sparse_rmsprop_rows(
            parameter,
            gradient,
            moment,
            cast(Tensor, state["last_step"]),
            cast(Tensor, state["last_cum"]),
            bitmap,
            cast(Tensor, state["sparse_index"]),
            cast(Tensor, state["sparse_count"]),
            cast(Tensor, state["sparse_scratch"]),
            scalars,
        )


class NorMuonFactory(Protocol):
    """Configure the rates and execution of a deferred NorMuon constructor."""

    lr: float
    weight_decay: float
    compile: bool

    def make(self) -> Callable[..., NorMuon]:
        """Return a constructor accepting parameters or parameter groups."""
        ...


class FFNScaledNorMuon:
    """Build NorMuon with FFN learning rates scaled by matrix shape."""

    class Config(Fig["Callable[..., NorMuon]"]):
        channels_in: int = -1
        """Residual-stream width; -1 disables shape-specific rate adjustments."""

        reference_expansion: float = 4.0
        """Expansion at which the input-projection rate was tuned."""

        ffn_lr_multiplier: float = 1.0
        """Rate multiplier for rectangular FFN input and output matrices."""

        optimizer: NorMuonFactory = field(
            default_factory=lambda: NorMuon.Config(lr=0.04)
        )
        """Deferred NorMuon constructor with configurable rates and compilation."""

        @override
        def make(self) -> Callable[..., NorMuon]:
            return partial(_ffn_scaled_normuon, config=self.copy_tree().finalize())


@dataclass(slots=True, kw_only=True, frozen=True)
class WeightDecayPulse:
    """Multiply weight decay within an open interval of training progress."""

    center: float = 0.5
    """Training progress at the center of the interval."""

    half_width: float = 0.5
    """Progress interval from the center to either boundary."""

    multiplier: float = 1.0
    """Peak multiplier relative to the underlying weight-decay schedule."""

    triangular: bool = False
    """Ramp linearly to the peak; False keeps a rectangular multiplier."""


class ScheduledOptimizerUpdate:
    """Apply separate dense, matrix and sparse-table optimizer schedules."""

    class Config(Fig["ScheduledOptimizerUpdate"]):
        """Configure schedule endpoints, group routing and optional decay pulses."""

        muon_warmdown: float = 0.5
        """Budget fraction spent annealing matrix and original-input skip LRs."""

        adam_warmdown: float = 0.5
        """Budget fraction spent annealing other dense parameter LRs."""

        final_lr_fraction: float = 0.0
        """Final LR relative to each group's initial rate."""

        momentum_final: float = 0.95
        """Matrix momentum after its quadratic warmdown."""

        muon_beta2_final: float = 0.95
        """Final NorMuon variance decay."""

        adam_beta1_final: float = 0.8
        """Final first-moment decay for embedding and readout groups."""

        ngram_beta2_final: float = 0.999
        """Final sparse-table variance decay."""

        ngram_ramp_fraction: float = 1.0
        """Final fraction of matrix warmdown containing the sparse-table ramp."""

        optimizer_recompile_limit: int | None = None
        """Override the compiler variant limit; None retains its current value."""

        skip_member: int = -1
        """Composite member whose LR follows matrices but retains constant decay."""

        adam_beta1_members: tuple[int, ...] = ()
        """Composite members whose Adam first moment follows the dense warmdown."""

        weight_decay_pulses: tuple[WeightDecayPulse, ...] = ()
        """Ordered weight-decay pulses; the first matching interval takes precedence."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        for name, value in (
            ("muon_warmdown", config.muon_warmdown),
            ("adam_warmdown", config.adam_warmdown),
            ("ngram_ramp_fraction", config.ngram_ramp_fraction),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive; got {value}.")
        self.config = config

    def __call__(self, step: _ScheduledTrainStep) -> dict[str, float | Tensor]:
        """Apply schedules, update parameters, and clear gradients.

        Args:
          step: Training state with progress and a composite optimizer.

        Returns:
          metrics: Current matrix and dense learning-rate multipliers.

        """
        cfg = self.config
        base = step.config
        progress = step.progress_learning_schedule
        muon_fraction = max(
            0.0,
            (progress - (1 - cfg.muon_warmdown)) / cfg.muon_warmdown,
        )
        adam_fraction = max(
            0.0,
            (progress - (1 - cfg.adam_warmdown)) / cfg.adam_warmdown,
        )
        muon_lr = 1 - muon_fraction * (1 - cfg.final_lr_fraction)
        adam_lr = 1 - adam_fraction * (1 - cfg.final_lr_fraction)
        late_fraction = max(
            0.0,
            (muon_fraction - (1 - cfg.ngram_ramp_fraction)) / cfg.ngram_ramp_fraction,
        )
        early = min(step.completed_updates / base.momentum_warmup_steps, 1.0)
        momentum = (1 - early) * base.momentum_start + early * base.momentum_end
        if progress > 1 - cfg.muon_warmdown:
            momentum = base.momentum_end + muon_fraction**2 * (
                cfg.momentum_final - base.momentum_end
            )
        decay = self.weight_decay_multiplier(progress)
        assert isinstance(step.optimizer, CompositeOptimizer)
        for index, member in enumerate(step.optimizer.optimizers):
            for group in member.param_groups:
                if isinstance(member, NorMuon):
                    group["lr"] = group["initial_lr"] * muon_lr
                    group["momentum"] = momentum
                    group["beta2"] = group["initial_beta2"] + muon_fraction * (
                        cfg.muon_beta2_final - group["initial_beta2"]
                    )
                    group["weight_decay"] = group["initial_weight_decay"] * decay
                elif isinstance(member, BiasCorrectedRMSProp):
                    group["beta2"] = group["initial_beta2"] + late_fraction * (
                        cfg.ngram_beta2_final - group["initial_beta2"]
                    )
                else:
                    group["lr"] = group["initial_lr"] * (
                        muon_lr if index == cfg.skip_member else adam_lr
                    )
                    if index in cfg.adam_beta1_members:
                        beta1, beta2 = cast(
                            "tuple[float, float]", group["initial_betas"]
                        )
                        group["betas"] = (
                            beta1 + adam_fraction * (cfg.adam_beta1_final - beta1),
                            beta2,
                        )
        previous_limit = torch_dynamo_config.recompile_limit
        if cfg.optimizer_recompile_limit is not None:
            torch_dynamo_config.recompile_limit = cfg.optimizer_recompile_limit
        try:
            step.optimizer.step()
        finally:
            torch_dynamo_config.recompile_limit = previous_limit
        step.model.zero_grad(set_to_none=True)
        return {"lr/muon_multiplier": muon_lr, "lr/adam_multiplier": adam_lr}

    def weight_decay_multiplier(self, progress: float) -> float:
        """Apply the first matching pulse to linear weight-decay warmdown.

        Args:
          progress: Fraction of the training budget consumed.

        Returns:
          multiplier: Weight-decay scale at this progress.

        """
        for pulse in self.config.weight_decay_pulses:
            distance = abs(progress - pulse.center)
            if distance < pulse.half_width:
                bump = 1 - distance / pulse.half_width if pulse.triangular else 1.0
                return (1 - progress) * (1 + bump * (pulse.multiplier - 1))
        return 1 - progress


def compact_bitmap(
    bitmap: Tensor, out_index: Tensor, out_count: Tensor, scratch: Tensor
) -> None:
    """Compact row flags into fixed-size index and count buffers.

    Inactive rows scatter to a spare slot. Boolean indexing would introduce a
    data-dependent output shape and prevent CUDA-graph capture.

    Args:
      bitmap: One zero-or-one flag per row.
      out_index: Output row-index buffer with capacity for every row.
      out_count: Scalar output count of flagged rows.
      scratch: Scatter buffer with one extra slot for inactive rows.

    """
    flags = bitmap.to(torch.int32)
    count = flags.numel()
    positions = torch.cumsum(flags, dim=0, dtype=torch.int64) - 1
    rows = torch.arange(count, device=flags.device, dtype=torch.int32)
    slot = torch.where(flags != 0, positions, torch.full_like(positions, count))
    scratch.scatter_(0, slot, rows)
    out_index.copy_(scratch[:count])
    out_count.copy_(flags.sum(dtype=torch.int32).reshape(()))


def sparse_rmsprop_rows(  # noqa: PLR0917 -- Each sparse kernel operand requires a positional argument.
    parameter: Tensor,
    gradient: Tensor,
    second_moment: Tensor,
    last_step: Tensor,
    last_cum: Tensor,
    bitmap: Tensor,
    index: Tensor,
    count: Tensor,
    scratch: Tensor,
    scalars: dict[str, Tensor],
) -> None:
    """Decay idle moments and update flagged parameter rows with RMSProp.

    Args:
      parameter: Contiguous two-dimensional embedding table.
      gradient: Table gradients matching ``parameter``.
      second_moment: FP32 rowwise state shaped ``[rows, 1]``.
      last_step: Last update index for each active row; written in place.
      last_cum: Unused cumulative-decay state; left unchanged.
      bitmap: Zero-or-one active flag per row.
      index: Preallocated row-index buffer filled by compaction.
      count: Scalar active-row count filled by compaction.
      scratch: Compaction buffer with one extra slot.
      scalars: Device tensors for step, lr, beta2, eps, and one_minus_lr_wd.
        Cumulative-decay fields are accepted but unused.

    """
    compact_bitmap(bitmap, index, count, scratch)
    if parameter.is_cuda:
        _sparse_rmsprop_rows_cuda(
            parameter,
            gradient,
            second_moment,
            last_step,
            last_cum,
            bitmap,
            index,
            count,
            scalars["step"],
            scalars["lr"],
            scalars["beta2"],
            scalars["eps"],
            scalars["one_minus_lr_wd"],
            scalars["cum_before"],
            scalars["cum_after"],
        )
        return
    _inactive_moment_reference(second_moment, bitmap, scalars["beta2"])
    _sparse_rmsprop_reference(
        parameter,
        gradient,
        second_moment,
        last_step,
        last_cum,
        index[: int(count)].to(torch.int64),
        scalars,
    )


def _inactive_moment_reference(
    second_moment: Tensor, bitmap: Tensor, beta2: Tensor
) -> None:
    """Decay idle CPU moments with lerp; multiplication alone rounds differently."""
    idle = bitmap.to(torch.bool).logical_not()
    if not bool(idle.any()):
        return
    rows = second_moment[idle]
    second_moment[idle] = torch.lerp(rows, torch.zeros_like(rows), 1 - float(beta2))


def _sparse_rmsprop_reference(  # noqa: PLR0917 -- The reference mirrors the sparse kernel signature.
    parameter: Tensor,
    gradient: Tensor,
    second_moment: Tensor,
    last_step: Tensor,
    last_cum: Tensor,
    rows: Tensor,
    scalars: dict[str, Tensor],
) -> None:
    """Update active CPU rows with single-rounding weight arithmetic."""
    if rows.numel() == 0:
        return
    step = int(scalars["step"])
    beta2 = float(scalars["beta2"])
    eps = float(scalars["eps"])
    lr = float(scalars["lr"])
    moment_rows = second_moment[rows]
    grad_rows = gradient[rows]
    parameter_rows = parameter[rows].clone()
    parameter_rows.mul_(float(scalars["one_minus_lr_wd"]))
    moment_rows = torch.lerp(
        moment_rows,
        grad_rows.float().square().mean(dim=-1, keepdim=True),
        1 - beta2,
    )
    denominator = (moment_rows / (1 - beta2**step)).sqrt() + eps
    parameter_rows = (
        (parameter_rows.double() - lr * (grad_rows / denominator).double())
        .float()
        .to(parameter_rows.dtype)
    )
    parameter[rows] = parameter_rows
    second_moment[rows] = moment_rows
    # `last_cum` is deliberately not written; see the docstring and USES_LAZY_ANCHOR.
    del last_cum
    last_step[rows] = step


USES_LAZY_ANCHOR: Final = False
"""Whether idle moments use deferred decay; all moments advance every step."""

MAINTAINS_LAST_CUM: Final = False
"""Whether cumulative-decay state is updated; it is unused and unchanged."""

MAINTAINS_LAST_STEP: Final = True
"""Whether active rows record their latest optimizer step."""

INACTIVE_BLOCK: Final = 1024
"""Rows per program in the inactive-moment kernel."""

SPARSE_WARPS: Final = 1
"""Warps per program for inactive-moment and active-row kernels."""

BITMAP_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int32)
"""Bitmap element types supported by the inactive-moment kernel."""


def _jit_kernel(function: Callable[..., None]) -> "triton.JITFunction[..., None]":
    """Bind concrete language modules before Triton hashes and compiles the function."""
    assert isinstance(function, FunctionType)
    bound = FunctionType(
        function.__code__,
        function.__globals__
        | {
            "language": import_module("triton.language"),
            "libdevice": import_module("triton.language.extra.cuda.libdevice"),
        },
        function.__name__,
        function.__defaults__,
    )
    bound.__annotations__ = function.__annotations__
    return triton.jit(bound)


@lru_cache(maxsize=1)
def _compiled_inactive_moment() -> "triton.JITFunction[..., None]":
    return _jit_kernel(_inactive_moment_rows_kernel)


def _inactive_moment_rows_kernel(
    buffers: "tuple[language.tensor[language.pointer_type], ...]",
    n_rows: int,
    block: "language.constexpr",
) -> None:
    """Decay idle row moments using stable lerp; preserve active moments."""
    (v_ptr, bitmap_ptr, beta2_ptr) = buffers
    pid = language.program_id(0)
    offs = pid * block + language.arange(0, block)
    mask = offs < n_rows

    beta2 = language.load(beta2_ptr)
    dirty = language.load(bitmap_ptr + offs, mask=mask, other=0)
    v_old = language.load(v_ptr + offs, mask=mask, other=0.0).to(language.float32)

    one = 1.0
    weight = one - beta2
    mean = 0.0
    take_end = language.abs(weight) >= 0.5
    start = language.where(take_end, mean, v_old)
    coeff = language.where(take_end, -(one - weight), weight)
    v_new = language.fma(coeff, mean - v_old, start)

    keep = dirty != 0
    language.store(v_ptr + offs, language.where(keep, v_old, v_new), mask=mask)


@lru_cache(maxsize=1)
def _compiled_sparse_rmsprop() -> "triton.JITFunction[..., None]":
    return _jit_kernel(_sparse_rmsprop_rows_kernel)


def _sparse_rmsprop_rows_kernel(
    buffers: "tuple[language.tensor[language.pointer_type], ...]",
    n_cols: "language.constexpr",
    programs: "language.constexpr",
    block_w: "language.constexpr",
) -> None:
    """Update active moments and weights with bias correction and fused arithmetic."""
    (
        p_ptr,
        grad_ptr,
        v_ptr,
        last_step_ptr,
        _last_cum_ptr,
        index_ptr,
        count_ptr,
        step_ptr,
        lr_ptr,
        beta2_ptr,
        eps_ptr,
        one_minus_lr_wd_ptr,
        _cum_before_ptr,
        _cum_after_ptr,
    ) = buffers
    pid = language.program_id(0)
    count = language.load(count_ptr)
    step = language.load(step_ptr)
    lr = language.load(lr_ptr)
    beta2 = language.load(beta2_ptr)
    eps = language.load(eps_ptr)
    one_minus_lr_wd = language.load(one_minus_lr_wd_ptr)
    # Cumulative-decay pointers are unused because idle moments advance every step.
    offs = language.arange(0, block_w)
    mask = offs < n_cols

    slot = pid
    while slot < count:
        row = language.load(index_ptr + slot).to(language.int64)
        base = row * n_cols

        grad = language.load(grad_ptr + base + offs, mask=mask, other=0.0).to(
            language.float32
        )
        v_old = language.load(v_ptr + row).to(language.float32)

        mean = language.sum(language.where(mask, grad * grad, 0.0), axis=0) / n_cols
        one = 1.0
        weight = one - beta2
        take_end = language.abs(weight) >= 0.5
        start = language.where(take_end, mean, v_old)
        coeff = language.where(take_end, -(one - weight), weight)
        v_new = language.fma(coeff, mean - v_old, start)

        bias2 = one - libdevice.pow(beta2, step.to(language.float32))
        denom = language.sqrt_rn(v_new / bias2) + eps
        term = grad / denom
        pv = language.load(p_ptr + base + offs, mask=mask, other=0.0).to(
            language.float32
        )
        pv = language.fma(term, -lr, pv * one_minus_lr_wd)

        language.store(p_ptr + base + offs, pv.to(p_ptr.dtype.element_ty), mask=mask)
        language.store(v_ptr + row, v_new)
        # Leave unused cumulative-decay state untouched.
        language.store(last_step_ptr + row, step.to(language.int32))
        slot += programs


def _sparse_rmsprop_rows_cuda(  # noqa: PLR0917 -- Each sparse kernel operand requires a positional argument.
    parameter: Tensor,
    gradient: Tensor,
    second_moment: Tensor,
    last_step: Tensor,
    last_cum: Tensor,
    bitmap: Tensor,
    index: Tensor,
    count: Tensor,
    step: Tensor,
    lr: Tensor,
    beta2: Tensor,
    eps: Tensor,
    one_minus_lr_wd: Tensor,
    cum_before: Tensor,
    cum_after: Tensor,
    *,
    programs: int = 8192,
    block_w: int = 512,
) -> None:
    """Decay idle moments, then update compacted rows with a fixed reduction width."""
    (rows, cols) = parameter.shape
    if bitmap.dtype not in BITMAP_DTYPES:
        raise ValueError(
            f"the inactive sweep LOADS the bitmap from a kernel, so its dtype must be "
            f"one of {BITMAP_DTYPES}, got {bitmap.dtype}"
        )
    if cols > block_w:
        raise ValueError(
            f"row width {cols} exceeds the {block_w}-lane block; one row must fit in "
            "one block because the kernel reduces a row per iteration"
        )
    # Use a shape-derived grid to avoid device synchronization and preserve graph
    # capture.
    _compiled_inactive_moment()[(rows + INACTIVE_BLOCK - 1) // INACTIVE_BLOCK,](
        buffers=(second_moment, bitmap, beta2),
        n_rows=rows,
        block=INACTIVE_BLOCK,
        num_warps=SPARSE_WARPS,
    )
    _compiled_sparse_rmsprop()[programs,](
        buffers=(
            parameter,
            gradient,
            second_moment,
            last_step,
            last_cum,
            index,
            count,
            step,
            lr,
            beta2,
            eps,
            one_minus_lr_wd,
            cum_before,
            cum_after,
        ),
        n_cols=cols,
        programs=programs,
        block_w=block_w,
        num_warps=SPARSE_WARPS,
    )


def _ffn_scaled_normuon(
    params: Iterable[Tensor], *, config: FFNScaledNorMuon.Config
) -> NorMuon:
    parameters = list(params)
    groups: list[dict[str, object]] = []
    for rows, columns in sorted({tuple(p.shape) for p in parameters}):
        rate = config.optimizer.lr
        if columns == config.channels_in and rows > config.channels_in:
            rate *= (config.reference_expansion * config.channels_in / rows) ** 0.5
        if (columns == config.channels_in and rows > config.channels_in) or (
            rows == config.channels_in and columns > config.channels_in
        ):
            rate *= config.ffn_lr_multiplier
        groups.append(
            {
                "params": [p for p in parameters if tuple(p.shape) == (rows, columns)],
                "lr": rate,
            },
        )
    optimizer = config.optimizer.make()(groups)
    assert isinstance(optimizer, NorMuon)
    return optimizer


def _rmsprop_update(
    parameter: Tensor,
    gradient: Tensor,
    second_moment: Tensor,
    *,
    step: Tensor,
    lr: Tensor,
    beta2: Tensor,
    eps: Tensor,
    weight_decay: Tensor,
    rowwise: bool,
) -> None:
    parameter.mul_(1 - lr * weight_decay)
    squared = (
        gradient.float().square().mean(dim=-1, keepdim=True)
        if rowwise
        else gradient.square()
    )
    second_moment.lerp_(squared, 1 - beta2)
    denominator = (second_moment / (1 - beta2**step)).sqrt() + eps
    parameter.sub_(gradient / denominator * lr)


def _copy_optimizer_state(value: object, device: torch.device) -> object:
    """Copy tensor state onto its owner's device while retaining its precision."""
    if isinstance(value, Tensor):
        return value.to(device=device, copy=True)
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {
            key: _copy_optimizer_state(item, device) for key, item in mapping.items()
        }
    return value
