"""Check table update arithmetic, matrix routing, and schedule boundaries."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import cast
from unittest.mock import MagicMock, PropertyMock, call, patch

import math
import re

from torch import Tensor, nn
from torch.optim import Optimizer

import pytest
import torch

from priml.baselines.nanochat import optimizers
from priml.lib.codec import ReadError, from_plain
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.normuon import NorMuon


def _state(optimizer: Optimizer, parameter: Tensor) -> dict[str, object]:
    return cast("dict[str, object]", optimizer.state[parameter])


def _tensor(state: dict[str, object], name: str) -> Tensor:
    value = state[name]
    assert isinstance(value, Tensor)
    return value


@dataclass(slots=True, kw_only=True)
class _IndependentNorMuonConfig:
    lr: float = 0.04
    weight_decay: float = 0.2
    compile: bool = False

    def make(self) -> Callable[..., NorMuon]:
        return partial(
            NorMuon,
            lr=self.lr,
            weight_decay=self.weight_decay,
            compile=self.compile,
        )


def test_ffn_factory_accepts_an_independent_config() -> None:
    """An injected factory needs the protocol, not NorMuon.Config inheritance."""
    config = optimizers.FFNScaledNorMuon.Config(channels_in=4, ffn_lr_multiplier=1.25)
    config.optimizer = _IndependentNorMuonConfig(lr=0.08)
    parameter = torch.nn.Parameter(torch.ones(8, 4))
    optimizer = config.make()([parameter])
    assert optimizer.param_groups[0]["lr"] == 0.08 * 2**0.5 * 1.25


def test_weight_decay_pulses_survive_copy_and_serialization() -> None:
    """Value records retain their order and schedule through Configgle roundtrips."""
    config = optimizers.ScheduledOptimizerUpdate.Config()
    config.weight_decay_pulses = (
        optimizers.WeightDecayPulse(center=0.5, half_width=0.25, multiplier=3),
        optimizers.WeightDecayPulse(center=0.75, multiplier=5, triangular=True),
    )
    copied = config.copy_tree()
    restored = optimizers.ScheduledOptimizerUpdate.Config.deserialize(
        config.serialize(),
    )
    for candidate in (copied, restored):
        assert candidate.weight_decay_pulses == config.weight_decay_pulses
        assert candidate.make().weight_decay_multiplier(0.5) == 1.5
        assert candidate.make().weight_decay_multiplier(0.875) == 0.5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sparse", [False, True])
def test_zero_beta_rmsprop_survives_checkpoint_and_positive_beta_update(
    dtype: torch.dtype,
    sparse: bool,
) -> None:
    """Zero decay replaces moments; a checkpoint must reproduce the next update."""
    weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype))
    config = optimizers.BiasCorrectedRMSProp.Config(
        lr=0.25,
        beta2=0.0,
        eps=2.0,
        rowwise=True,
        sparse_rows=sparse,
    )
    optimizer = config.make()([weight])
    sink = torch.tensor([[2.0, -2.0], [0.0, 0.0]])
    bitmap = torch.tensor([1, 0], dtype=torch.uint8)
    optimizer.gradient_sinks[weight] = sink
    optimizer.gradient_bitmaps[weight] = bitmap
    optimizer.step()
    assert torch.equal(weight, torch.tensor([[0.875, 2.125], [3.0, 4.0]], dtype=dtype))
    assert torch.equal(
        _tensor(_state(optimizer, weight), "second_moment"),
        torch.tensor([[4.0], [0.0]]),
    )

    restored_weight = torch.nn.Parameter(weight.detach().clone())
    restored = config.make()([restored_weight])
    restored.gradient_sinks[restored_weight] = sink
    restored.gradient_bitmaps[restored_weight] = bitmap
    restored.load_state_dict(optimizer.state_dict())
    sink.mul_(0.5)
    for member in (optimizer, restored):
        member.param_groups[0]["beta2"] = 0.5
        member.step()
    assert torch.isfinite(weight).all()
    assert torch.equal(restored_weight, weight)
    assert torch.equal(
        _tensor(_state(restored, restored_weight), "second_moment"),
        _tensor(_state(optimizer, weight), "second_moment"),
    )


@pytest.mark.parametrize(
    "field",
    ["muon_warmdown", "adam_warmdown", "ngram_ramp_fraction"],
)
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_schedule_fractions_require_finite_positive_values(
    field: str,
    value: float,
) -> None:
    """Reject invalid denominators while constructing the update policy."""
    config = optimizers.ScheduledOptimizerUpdate.Config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        config.make()


def test_schedule_fractions_above_one_remain_valid() -> None:
    """Positive denominator validation must not add an upper bound."""
    config = optimizers.ScheduledOptimizerUpdate.Config()
    config.muon_warmdown = 2.0
    config.adam_warmdown = 2.0
    config.ngram_ramp_fraction = 2.0
    config.make()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sparse", [False, True])
def test_rowwise_checkpoint_preserves_state_precision_and_next_update(
    dtype: torch.dtype,
    sparse: bool,
) -> None:
    """Reloading narrow weights must retain FP32 moments and integer row indices."""
    weight = torch.nn.Parameter(torch.ones(4, 3, dtype=dtype))
    config = optimizers.BiasCorrectedRMSProp.Config(
        lr=0.13,
        beta2=0.91,
        rowwise=True,
        sparse_rows=sparse,
    )
    optimizer = config.make()([weight])
    sink = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7
    bitmap = torch.tensor([1, 0, 1, 1], dtype=torch.uint8)
    optimizer.gradient_sinks[weight] = sink
    optimizer.gradient_bitmaps[weight] = bitmap
    optimizer.step()
    restored_weight = torch.nn.Parameter(weight.detach().clone())
    restored = config.make()([restored_weight])
    restored.gradient_sinks[restored_weight] = sink
    restored.gradient_bitmaps[restored_weight] = bitmap
    restored.load_state_dict(optimizer.state_dict())
    before = _state(optimizer, weight)
    after = _state(restored, restored_weight)
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            after_value = _tensor(after, name)
            assert after_value.dtype == value.dtype, name
            assert torch.equal(after_value, value), name
            assert after_value.data_ptr() != value.data_ptr(), name
        elif isinstance(value, dict):
            for key, scalar in cast(dict[str, object], value).items():
                assert isinstance(scalar, torch.Tensor)
                after_scalars = from_plain(after[name], dict[str, Tensor])
                assert after_scalars[key].dtype == scalar.dtype, key
                assert torch.equal(after_scalars[key], scalar), key
    sink.mul_(0.37)
    bitmap.fill_(1)
    optimizer.step()
    restored.step()
    assert torch.equal(restored_weight, weight)
    assert torch.equal(
        _tensor(after, "second_moment"),
        _tensor(before, "second_moment"),
    )


def test_rmsprop_bias_correction_follows_scheduled_beta() -> None:
    """Keep FP32 scalar rounding and the source's current-beta correction."""
    assert "BiasCorrectedRMSProp" in vars(optimizers)
    weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    config = optimizers.BiasCorrectedRMSProp.Config(lr=0.1, beta2=0.9)
    optimizer = config.make()([weight])
    expected = weight.detach().clone()
    moment = torch.zeros_like(weight)
    for index, beta in enumerate((0.9, 0.95, 0.99), start=1):
        gradient = torch.tensor([2.0, -0.5]) * index
        weight.grad = gradient
        optimizer.param_groups[0]["beta2"] = beta
        beta2 = torch.tensor(beta)
        lr = torch.tensor(config.lr)
        moment.lerp_(gradient.square(), 1 - beta2)
        denominator = (moment / (1 - beta2 ** torch.tensor(float(index)))).sqrt()
        expected.sub_(gradient / (denominator + torch.tensor(config.eps)) * lr)
        optimizer.step()
        assert torch.equal(weight, expected)
        assert torch.equal(_tensor(_state(optimizer, weight), "second_moment"), moment)


def test_sparse_rmsprop_keeps_idle_weights_and_advances_idle_moments() -> None:
    """Skipping an idle parameter update must not skip its moment evolution."""
    assert "BiasCorrectedRMSProp" in vars(optimizers)
    weight = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.bfloat16))
    config = optimizers.BiasCorrectedRMSProp.Config(
        lr=0.1,
        beta2=0.9,
        rowwise=True,
        sparse_rows=True,
    )
    optimizer = config.make()([weight])
    sink = torch.ones(4, 3)
    bitmap = torch.tensor([0, 1, 0, 0], dtype=torch.uint8)
    optimizer.gradient_sinks[weight] = sink
    optimizer.gradient_bitmaps[weight] = bitmap
    optimizer.step()
    before = weight.detach().clone()
    moment = _tensor(_state(optimizer, weight), "second_moment").clone()
    bitmap.zero_()
    optimizer.param_groups[0]["beta2"] = 0.95
    optimizer.step()
    expected = torch.lerp(
        moment,
        torch.zeros_like(moment),
        1 - float(torch.tensor(0.95)),
    )
    assert torch.equal(weight, before)
    second_moment = _tensor(_state(optimizer, weight), "second_moment")
    assert torch.equal(second_moment, expected)
    assert second_moment.dtype == torch.float32


def test_sparse_rmsprop_refuses_missing_bitmap() -> None:
    """A miswired sparse run cannot silently fall back to dense updates."""
    assert "BiasCorrectedRMSProp" in vars(optimizers)
    weight = torch.nn.Parameter(torch.ones(2, 3))
    config = optimizers.BiasCorrectedRMSProp.Config(rowwise=True, sparse_rows=True)
    optimizer = config.make()([weight])
    weight.grad = torch.ones_like(weight)
    with pytest.raises(ValueError, match="sparse_rows is set") as error:
        optimizer.step()
    assert str(error.value) == (
        "sparse_rows is set but this table has no dirty bitmap; refusing to fall back "
        "to the dense path and report it as a sparse step"
    )


def test_ffn_multiplier_changes_both_rectangular_projections() -> None:
    """The attention square and unrelated matrices keep their original rates."""
    assert "FFNScaledNorMuon" in vars(optimizers)
    parameters = [
        torch.nn.Parameter(torch.ones(shape))
        for shape in ((4, 4), (8, 4), (4, 8), (2, 3))
    ]
    config = optimizers.FFNScaledNorMuon.Config(channels_in=4, ffn_lr_multiplier=1.25)
    config.optimizer.compile = False
    optimizer = config.make()(parameters)
    rates = {
        tuple(cast("list[Tensor]", group["params"])[0].shape): from_plain(
            cast(object, group["lr"]),
            float,
        )
        for group in optimizer.param_groups
    }
    assert rates == pytest.approx(
        {(4, 4): 0.04, (8, 4): 0.05 * 2**0.5, (4, 8): 0.05, (2, 3): 0.04},
    )


@pytest.mark.parametrize("progress", [0.0, 0.25, 0.5, 1.0])
def test_weight_decay_pulse_has_open_endpoints(progress: float) -> None:
    """A rectangular pulse modifies only its open interval."""
    assert "ScheduledOptimizerUpdate" in vars(optimizers)
    config = optimizers.ScheduledOptimizerUpdate.Config()
    config.weight_decay_pulses = (
        optimizers.WeightDecayPulse(center=0.5, half_width=0.25, multiplier=3),
    )
    expected = (1 - progress) * (3 if 0.25 < progress < 0.75 else 1)
    assert config.make().weight_decay_multiplier(progress) == expected


def test_first_weight_decay_pulse_wins_and_triangle_reaches_peak() -> None:
    """Overlapping pulses are ordered policies, not multiplied together."""
    assert "ScheduledOptimizerUpdate" in vars(optimizers)
    config = optimizers.ScheduledOptimizerUpdate.Config()
    config.weight_decay_pulses = (
        optimizers.WeightDecayPulse(
            center=0.5,
            half_width=0.25,
            multiplier=3,
            triangular=True,
        ),
        optimizers.WeightDecayPulse(center=0.5, half_width=0.5, multiplier=9),
    )
    update = config.make()
    assert update.weight_decay_multiplier(0.5) == 1.5
    assert update.weight_decay_multiplier(0.375) == 1.25


def test_sparse_rmsprop_rejects_incompatible_tables() -> None:
    weight = torch.nn.Parameter(torch.ones(2, 3))
    config = optimizers.BiasCorrectedRMSProp.Config(rowwise=False, sparse_rows=True)
    optimizer = config.make()([weight])
    optimizer.gradient_bitmaps[weight] = torch.zeros(2, dtype=torch.uint8)
    weight.grad = torch.ones_like(weight)
    with pytest.raises(ValueError, match="sparse_rows requires rowwise") as error:
        optimizer.step()
    assert str(error.value) == (
        "sparse_rows requires rowwise: the kernel reads one second-moment entry per row"
    )

    config = optimizers.BiasCorrectedRMSProp.Config(
        rowwise=True,
        sparse_rows=True,
        weight_decay=0.1,
    )
    parameter = nn.Parameter(torch.ones(2, 3))
    optimizer = config.make()([parameter])
    optimizer.gradient_bitmaps[parameter] = torch.zeros(2, dtype=torch.uint8)
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(ValueError, match="decoupled decay"):
        optimizer.step()


def test_scheduled_update_rejects_nonpositive_warmdown() -> None:
    with pytest.raises(ValueError, match="muon_warmdown"):
        optimizers.ScheduledOptimizerUpdate.Config(muon_warmdown=0).make()


def test_scheduled_update_drives_each_optimizer_family() -> None:
    matrix = nn.Parameter(torch.ones(2, 3))
    table = nn.Parameter(torch.ones(3, 4))
    dense = nn.Parameter(torch.ones(4, 5))
    muon = NorMuon.Config(lr=0.1, weight_decay=0.01).make()([matrix])
    rmsprop = optimizers.BiasCorrectedRMSProp.Config(lr=0.1).make()([table])
    adam = torch.optim.Adam([dense], lr=0.2, betas=(0.8, 0.99))
    members = CompositeOptimizer([muon, rmsprop, adam])
    for member in members.optimizers:
        for group in member.param_groups:
            group["initial_lr"] = group["lr"]
            group["initial_weight_decay"] = group.get("weight_decay", 0.0)
            if "beta2" in group:
                group["initial_beta2"] = group["beta2"]
            if "betas" in group:
                group["initial_betas"] = group["betas"]

    class Config:
        momentum_warmup_steps = 2
        momentum_start = 0.8
        momentum_end = 0.9

    class ScheduleStep:
        config = Config()
        progress_learning_schedule = 0.75
        completed_updates = 1
        optimizer = members
        model = nn.Module()

    result = optimizers.ScheduledOptimizerUpdate.Config(
        skip_member=2,
        adam_beta1_members=(2,),
    ).make()(ScheduleStep())
    assert result["lr/muon_multiplier"] < 1.0
    assert result["lr/adam_multiplier"] < 1.0


@pytest.mark.parametrize("dtype", optimizers.BITMAP_DTYPES)
def test_compact_bitmap_packs_active_row_indices(dtype: torch.dtype) -> None:
    bitmap = torch.tensor([0, 1, 0, 1], dtype=dtype)
    indices = torch.full((4,), -1, dtype=torch.int32)
    count = torch.zeros((), dtype=torch.int32)
    scratch = torch.full((5,), -1, dtype=torch.int32)

    with (
        patch.object(torch, "cumsum", wraps=torch.cumsum) as cumsum,
        patch.object(torch, "arange", wraps=torch.arange) as arange,
    ):
        optimizers.compact_bitmap(bitmap, indices, count, scratch)

    assert torch.equal(indices, torch.tensor([1, 3, -1, -1], dtype=torch.int32))
    assert cumsum.call_count == 1
    cumsum_args = cumsum.call_args
    assert cumsum_args is not None
    flags = cumsum_args.args[0]
    assert isinstance(flags, Tensor)
    assert torch.equal(flags, bitmap != 0)
    assert cumsum_args.kwargs == {"dim": 0}
    arange.assert_called_once_with(4, device=bitmap.device, dtype=torch.int32)
    assert count.dtype == torch.int32
    assert count.shape == ()
    assert count.item() == 2


def test_sparse_cpu_reference_receives_int64_indices() -> None:
    parameter = torch.ones(2, 3)
    scalars = {
        name: torch.tensor(value)
        for name, value in (
            ("step", 1),
            ("lr", 0.1),
            ("beta2", 0.9),
            ("eps", 1e-8),
        )
    }
    with patch.object(optimizers, "_sparse_rmsprop_reference") as reference:
        optimizers.sparse_rmsprop_rows(
            parameter,
            torch.ones_like(parameter),
            torch.zeros(2, 1),  # sparse_rmsprop_rows requires [rows, 1] moments.
            torch.tensor([0, 1], dtype=torch.uint8),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros((), dtype=torch.int32),
            torch.zeros(3, dtype=torch.int32),
            scalars,
        )

    assert reference.call_count == 1
    call_args = reference.call_args
    assert call_args is not None
    rows = call_args.args[3]
    assert isinstance(rows, Tensor)
    assert rows.dtype == torch.int64
    assert rows.tolist() == [1]


def test_sparse_rmsprop_cuda_route_passes_each_buffer() -> None:
    parameter = nn.Parameter(torch.ones(2, 3))
    gradient = torch.ones_like(parameter)
    # sparse_rmsprop_rows keeps one FP32 second moment per row: [rows, 1].
    moment = torch.zeros(2, 1)
    bitmap = torch.tensor([0, 1], dtype=torch.uint8)
    index = torch.full((2,), -1, dtype=torch.int32)
    count = torch.zeros((), dtype=torch.int32)
    scratch = torch.full((3,), -1, dtype=torch.int32)
    scalars = {name: torch.tensor(0.0) for name in ("step", "lr", "beta2", "eps")}

    with (
        patch.object(Tensor, "is_cuda", new_callable=PropertyMock, return_value=True),
        patch.object(optimizers, "_sparse_rmsprop_rows_cuda") as launch,
    ):
        optimizers.sparse_rmsprop_rows(
            parameter,
            gradient,
            moment,
            bitmap,
            index,
            count,
            scratch,
            scalars,
        )

    launch.assert_called_once_with(
        parameter,
        gradient,
        moment,
        bitmap,
        index,
        count,
        scalars["step"],
        scalars["lr"],
        scalars["beta2"],
        scalars["eps"],
    )


def test_sparse_rmsprop_initializes_exact_row_state_and_errors() -> None:
    parameter = torch.nn.Parameter(torch.ones(3, 2))
    config = optimizers.BiasCorrectedRMSProp.Config(rowwise=True, sparse_rows=True)
    optimizer = config.make()([parameter])
    bitmap = torch.tensor([0, 1, 0], dtype=torch.uint8)
    optimizer.gradient_bitmaps[parameter] = bitmap
    optimizer.gradient_sinks[parameter] = torch.ones(3, 2)
    optimizer.step()

    state = _state(optimizer, parameter)
    expected = {
        "sparse_index": ((3,), torch.int32),
        "sparse_count": ((), torch.int32),
        "sparse_scratch": ((4,), torch.int32),
    }
    for name, (shape, dtype) in expected.items():
        value = _tensor(state, name)
        assert value.shape == shape, name
        assert value.dtype == dtype, name
        assert value.device.type == "cpu", name
    assert set(state) == {
        "step",
        "second_moment",
        "sparse_index",
        "sparse_count",
        "sparse_scratch",
        "sparse_scalars",
    }
    sparse_scalars = from_plain(state["sparse_scalars"], dict[str, Tensor])
    assert set(sparse_scalars) == {"step", "lr", "beta2", "eps"}
    assert all(
        value.shape == () and value.dtype == torch.float32
        for value in sparse_scalars.values()
    )
    assert torch.equal(
        _tensor(state, "sparse_index"),
        torch.tensor([1, 0, 0], dtype=torch.int32),
    )
    assert _tensor(state, "sparse_count").item() == 1


def test_sparse_rmsprop_requests_state_factory_dtype_and_device() -> None:
    parameter = torch.nn.Parameter(torch.ones(3, 2))
    optimizer = optimizers.BiasCorrectedRMSProp.Config(
        rowwise=True,
        sparse_rows=True,
    ).make()([parameter])
    optimizer.gradient_bitmaps[parameter] = torch.tensor([0, 1, 0], dtype=torch.uint8)
    optimizer.gradient_sinks[parameter] = torch.ones(3, 2)

    with patch.object(torch, "zeros", wraps=torch.zeros) as zeros:
        optimizer.step()

    device = parameter.device
    assert zeros.call_args_list == [
        call((3, 1), dtype=torch.float32, device=device),
        call(3, dtype=torch.int32, device=device),
        call((), dtype=torch.int32, device=device),
        call(4, dtype=torch.int32, device=device),
        *[call((), dtype=torch.float32, device=device) for _ in range(4)],
    ]


def test_sparse_cuda_launcher_passes_all_buffers_and_fixed_grid() -> None:
    parameter = torch.ones(3, 512)
    gradient = torch.ones_like(parameter)
    moment = torch.zeros(3, 1)  # _sparse_rmsprop_rows_cuda uses [rows, 1] moments.
    bitmap = torch.tensor([0, 1, 0], dtype=torch.uint8)
    index = torch.zeros(3, dtype=torch.int32)
    count = torch.zeros((), dtype=torch.int32)
    scalars = {name: torch.zeros(()) for name in ("step", "lr", "beta2", "eps")}
    inactive_kernel = MagicMock()
    sparse_kernel = MagicMock()
    inactive_launch: MagicMock = MagicMock()
    sparse_launch: MagicMock = MagicMock()
    inactive_kernel.__getitem__.return_value = inactive_launch
    sparse_kernel.__getitem__.return_value = sparse_launch
    with (
        patch.object(
            optimizers,
            "_compiled_inactive_moment",
            return_value=inactive_kernel,
        ) as compiled_inactive,
        patch.object(
            optimizers,
            "_compiled_sparse_rmsprop",
            return_value=sparse_kernel,
        ) as compiled_sparse,
    ):
        optimizers._sparse_rmsprop_rows_cuda(
            parameter,
            gradient,
            moment,
            bitmap,
            index,
            count,
            scalars["step"],
            scalars["lr"],
            scalars["beta2"],
            scalars["eps"],
        )
    assert compiled_inactive.call_count == 1
    assert compiled_sparse.call_count == 1
    inactive_kernel.__getitem__.assert_called_once_with((1,))
    inactive_launch.assert_called_once_with(
        buffers=(moment, bitmap, scalars["beta2"]),
        n_rows=3,
        block=optimizers.INACTIVE_BLOCK,
        num_warps=optimizers.SPARSE_WARPS,
    )
    sparse_kernel.__getitem__.assert_called_once_with((8192,))
    sparse_launch.assert_called_once_with(
        buffers=(
            parameter,
            gradient,
            moment,
            index,
            count,
            scalars["step"],
            scalars["lr"],
            scalars["beta2"],
            scalars["eps"],
        ),
        n_cols=512,
        programs=8192,
        block_w=512,
        num_warps=optimizers.SPARSE_WARPS,
    )


@pytest.mark.parametrize(
    ("moment", "message"),
    [
        (
            torch.zeros(3, 2),
            "second moment must be per-row fp32 [rows, 1], got (3, 2) torch.float32",
        ),
        (
            torch.zeros(
                3,
                1,
                dtype=torch.float64,
            ),  # sparse_rmsprop_rows requires [rows, 1] moments.
            "second moment must be per-row fp32 [rows, 1], got (3, 1) torch.float64",
        ),
    ],
)
def test_sparse_rmsprop_requires_fp32_moment_per_row(
    moment: Tensor,
    message: str,
) -> None:
    parameter = nn.Parameter(torch.ones(3, 2))
    optimizer = optimizers.BiasCorrectedRMSProp.Config(
        rowwise=True,
        sparse_rows=True,
    ).make()([parameter])
    optimizer.gradient_bitmaps[parameter] = torch.zeros(3, dtype=torch.uint8)
    optimizer.gradient_sinks[parameter] = torch.ones_like(parameter)
    state = cast(dict[str, object], optimizer.state[parameter])
    state.update(step=0, second_moment=moment)

    with pytest.raises(ValueError, match="second moment must be per-row fp32") as error:
        optimizer.step()
    assert str(error.value) == message


@pytest.mark.parametrize(("rows", "grid"), [(1024, 1), (1025, 2)])
def test_sparse_cuda_inactive_grid_covers_row_boundaries(rows: int, grid: int) -> None:
    parameter = torch.ones(rows, 2)
    inactive_kernel = MagicMock()
    sparse_kernel = MagicMock()
    with (
        patch.object(
            optimizers,
            "_compiled_inactive_moment",
            return_value=inactive_kernel,
        ),
        patch.object(
            optimizers,
            "_compiled_sparse_rmsprop",
            return_value=sparse_kernel,
        ),
    ):
        optimizers._sparse_rmsprop_rows_cuda(
            parameter,
            torch.ones_like(parameter),
            torch.zeros(rows, 1),
            torch.zeros(rows, dtype=torch.uint8),
            torch.zeros(rows, dtype=torch.int32),
            torch.zeros((), dtype=torch.int32),
            *(torch.zeros(()) for _ in range(4)),
        )
    inactive_kernel.__getitem__.assert_called_once_with((grid,))


@pytest.mark.parametrize(
    ("bitmap", "block_w", "message"),
    [
        (
            torch.zeros(3, dtype=torch.float32),
            512,
            (
                f"the inactive sweep LOADS the bitmap from a kernel, so its dtype must be "
                f"one of {optimizers.BITMAP_DTYPES}, got torch.float32"
            ),
        ),
        (
            torch.zeros(3, dtype=torch.uint8),
            2,
            (
                "row width 4 exceeds the 2-lane block; one row must fit in one block "
                "because the kernel reduces a row per iteration"
            ),
        ),
    ],
)
def test_sparse_cuda_rejects_invalid_kernel_geometry(
    bitmap: Tensor,
    block_w: int,
    message: str,
) -> None:
    parameter = torch.ones(3, 4)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        optimizers._sparse_rmsprop_rows_cuda(
            parameter,
            torch.ones_like(parameter),
            # _sparse_rmsprop_rows_cuda takes one FP32 second moment per row.
            torch.zeros(3, 1),
            bitmap,
            torch.zeros(3, dtype=torch.int32),
            torch.zeros((), dtype=torch.int32),
            *(torch.zeros(()) for _ in range(4)),
            block_w=block_w,
        )


def test_sparse_rmsprop_requires_contiguous_table_and_gradient() -> None:
    parameter = nn.Parameter(torch.ones(3, 4).t())
    optimizer = optimizers.BiasCorrectedRMSProp.Config(
        rowwise=True,
        sparse_rows=True,
    ).make()([parameter])
    optimizer.gradient_bitmaps[parameter] = torch.ones(4, dtype=torch.uint8)
    optimizer.gradient_sinks[parameter] = torch.ones(4, 3)
    with pytest.raises(ValueError, match="table must be contiguous"):
        optimizer.step()

    parameter = nn.Parameter(torch.ones(3, 4))
    optimizer = optimizers.BiasCorrectedRMSProp.Config(
        rowwise=True,
        sparse_rows=True,
    ).make()([parameter])
    optimizer.gradient_bitmaps[parameter] = torch.ones(3, dtype=torch.uint8)
    optimizer.gradient_sinks[parameter] = torch.ones(4, 3).t()
    with pytest.raises(ValueError, match="grad must be contiguous"):
        optimizer.step()

    parameter = nn.Parameter(torch.ones(2, 3))
    optimizer = optimizers.BiasCorrectedRMSProp.Config(
        rowwise=True,
        sparse_rows=True,
    ).make()([parameter])
    optimizer.gradient_bitmaps[parameter] = torch.ones(2, dtype=torch.uint8)
    optimizer.gradient_sinks[parameter] = torch.ones(3, 2)
    message = "grad (3, 2) != table (2, 3)"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        optimizer.step()


def test_sparse_rmsprop_rejects_bad_bitmap_and_non_2d_table() -> None:
    parameter = nn.Parameter(torch.ones(3, 2, 4))
    optimizer = optimizers.BiasCorrectedRMSProp.Config(
        rowwise=True,
        sparse_rows=True,
    ).make()([parameter])
    optimizer.gradient_bitmaps[parameter] = torch.ones(3, dtype=torch.uint8)
    optimizer.gradient_sinks[parameter] = torch.ones_like(parameter)
    with pytest.raises(ValueError, match=r"expected a 2-D table, got \(3, 2, 4\)"):
        optimizer.step()

    parameter = nn.Parameter(torch.ones(3, 2))
    optimizer = optimizers.BiasCorrectedRMSProp.Config(
        rowwise=True,
        sparse_rows=True,
    ).make()([parameter])
    optimizer.gradient_bitmaps[parameter] = torch.ones(2, dtype=torch.uint8)
    optimizer.gradient_sinks[parameter] = torch.ones_like(parameter)
    with pytest.raises(
        ValueError,
        match=r"bitmap must be one flag per row \[3\], got \(2,\)",
    ):
        optimizer.step()


@pytest.mark.parametrize("field", ["weight_decay", "beta2", "lr", "eps"])
def test_sparse_rmsprop_rejects_uncoercible_scalars(field: str) -> None:
    parameter = nn.Parameter(torch.ones(2, 3))
    optimizer = optimizers.BiasCorrectedRMSProp.Config(
        rowwise=True,
        sparse_rows=True,
    ).make()([parameter])
    optimizer.gradient_bitmaps[parameter] = torch.zeros(2, dtype=torch.uint8)
    optimizer.gradient_sinks[parameter] = torch.ones_like(parameter)
    optimizer.param_groups[0][field] = "invalid"

    with pytest.raises(ReadError):
        optimizer.step()


def test_sparse_rmsprop_uses_per_row_moments_and_bias_correction() -> None:
    parameter = nn.Parameter(torch.ones(3, 2))
    optimizer = optimizers.BiasCorrectedRMSProp.Config(
        lr=0.1,
        beta2=0.5,
        eps=1.0,
        rowwise=True,
        sparse_rows=True,
    ).make()([parameter])
    optimizer.gradient_bitmaps[parameter] = torch.tensor([1, 1, 0], dtype=torch.uint8)
    gradient = torch.tensor([[2.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
    optimizer.gradient_sinks[parameter] = gradient

    optimizer.step()

    state = _state(optimizer, parameter)
    assert torch.equal(
        _tensor(state, "second_moment"),
        torch.tensor([[1.0], [4.0], [0.0]]),
    )
    denominator = (torch.tensor([[1.0], [4.0], [0.0]]) / 0.5).sqrt() + 1.0
    expected = (
        torch.ones(3, 2).double() - 0.1 * (gradient / denominator).double()
    ).float()
    assert torch.equal(parameter, expected)


def test_sparse_state_allocations_follow_parameter_device_and_dtype() -> None:
    parameter = nn.Parameter(torch.empty((3, 2), device="meta"))
    optimizer = optimizers.BiasCorrectedRMSProp.Config(
        rowwise=True,
        sparse_rows=True,
    ).make()([parameter])
    bitmap = torch.empty((3,), dtype=torch.uint8, device="meta")
    optimizer.gradient_bitmaps[parameter] = bitmap
    state = cast(dict[str, object], optimizer.state[parameter])
    state["step"] = 1
    state["second_moment"] = torch.empty(
        (3, 1),
        dtype=torch.float32,
        device="meta",
    )  # _sparse_step requires [rows, 1] moments.

    with (
        patch.object(optimizers, "sparse_rmsprop_rows"),
        patch.object(torch, "tensor", wraps=torch.tensor) as tensor_factory,
    ):
        optimizer._sparse_step(
            parameter,
            torch.empty_like(parameter),
            state,
            cast("dict[str, object]", optimizer.param_groups[0]),
        )

    group = cast(dict[str, object], optimizer.param_groups[0])
    assert tensor_factory.call_args == call(
        from_plain(group["beta2"], float),
        dtype=torch.float32,
    )
    expected = {
        "sparse_index": ((3,), torch.int32),
        "sparse_count": ((), torch.int32),
        "sparse_scratch": ((4,), torch.int32),
    }
    for name, (shape, dtype) in expected.items():
        value = _tensor(state, name)
        assert value.shape == shape, name
        assert value.dtype == dtype, name
        assert value.device.type == "meta", name
    scalars = from_plain(state["sparse_scalars"], dict[str, Tensor])
    assert all(value.device.type == "meta" for value in scalars.values())


def test_rowwise_rmsprop_reduces_only_the_last_dimension() -> None:
    parameter = nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 1)
    gradient = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 3
    config = optimizers.BiasCorrectedRMSProp.Config(
        lr=0.07,
        beta2=0.5,
        eps=0.2,
        rowwise=True,
    )
    optimizer = config.make()([parameter])
    beta2 = torch.tensor(config.beta2)
    expected_moment = torch.lerp(
        torch.zeros_like(gradient[..., :1]),
        gradient.square().mean(dim=-1, keepdim=True),
        1 - beta2,
    )
    denominator = (
        expected_moment / (1 - beta2 ** torch.tensor(1.0))
    ).sqrt() + torch.tensor(
        config.eps,
    )
    expected = parameter.detach().clone()
    expected.sub_(gradient / denominator * torch.tensor(config.lr))
    parameter.grad = gradient

    optimizer.step()

    assert torch.equal(parameter, expected)
    assert torch.equal(
        _tensor(_state(optimizer, parameter), "second_moment"),
        expected_moment,
    )


@pytest.mark.parametrize(
    "saved_groups",
    [[], [{"params": []}], [{"params": [0]}, {"params": [1]}]],
)
def test_restore_state_precision_rejects_misaligned_checkpoint_lists(
    saved_groups: list[dict[str, object]],
) -> None:
    parameter = nn.Parameter(torch.ones(2, 3))
    optimizer = optimizers.BiasCorrectedRMSProp.Config().make()([parameter])
    incoming: dict[str, object] = {"param_groups": saved_groups, "state": {}}

    with pytest.raises(ValueError, match=r"zip\(\) argument"):
        optimizer._restore_state_precision(incoming, optimizer)


def test_restore_state_precision_copies_to_parameter_device() -> None:
    parameter = nn.Parameter(torch.empty(2, 3, device="meta"))
    optimizer = optimizers.BiasCorrectedRMSProp.Config().make()([parameter])
    saved = torch.ones(2, 3)

    optimizer._restore_state_precision(
        {"param_groups": [{"params": [0]}], "state": {0: {"moment": saved}}},
        optimizer,
    )

    restored = _tensor(_state(optimizer, parameter), "moment")
    assert restored.device == parameter.device
    assert restored.dtype == saved.dtype
    assert restored.shape == saved.shape


def test_copy_optimizer_state_moves_nested_tensors_to_requested_device() -> None:
    original = torch.ones(2, 3)
    copied = optimizers._copy_optimizer_state(
        {"nested": {"weight": original}},
        torch.device("meta"),
    )
    nested = cast("dict[str, object]", cast("dict[str, object]", copied)["nested"])
    tensor = _tensor(nested, "weight")
    assert tensor.device.type == "meta"
    assert tensor.shape == original.shape
    assert tensor.dtype == original.dtype


def test_rmsprop_dense_decay_and_update_match_hand_calculation() -> None:
    parameter = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    optimizer = optimizers.BiasCorrectedRMSProp.Config(
        lr=0.25,
        beta2=0.0,
        eps=2.0,
        weight_decay=0.1,
    ).make()([parameter])
    gradient = torch.tensor([[2.0, -2.0], [0.0, 0.0]])
    parameter.grad = gradient
    expected = parameter.detach().clone()
    moment = gradient.square()
    expected.mul_(
        1 - torch.tensor(optimizer.param_groups[0]["lr"]) * torch.tensor(0.1),
    )
    expected.sub_(
        gradient / (moment.sqrt() + torch.tensor(2.0)) * torch.tensor(0.25),
    )
    optimizer.step()
    assert torch.equal(parameter, expected)
    assert torch.equal(
        _tensor(_state(optimizer, parameter), "second_moment"),
        torch.tensor([[4.0, 4.0], [0.0, 0.0]]),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
