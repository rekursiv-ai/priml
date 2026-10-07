"""Tests for the bfb golden harness itself."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import ast
import os
import tempfile
import traceback

from torch import Tensor, nn
from torch._decomp import decomposition_table
from torch.utils._python_dispatch import TorchDispatchMode

import pytest
import torch
import torch.distributed as dist


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._ops import OpOverload
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter
    from priml.math.custom_types import TensorFn

from priml.lib.codec import from_plain
from priml.model.attention.attention import Attention
from priml.model.transformer.block import TransformerBlock
from priml.testing import bfb, regenerate
from priml.testing.bfb import (
    _EXACT_F32_OPS,
    _FLOAT_FACTORIES,
    _assert_equal,
    _assert_portable_output_dtype,
    _assert_portable_state_changes,
    _assert_same_input,
    _byte_span,
    _compact_copies,
    _copy_back,
    _cpu_state_dict,
    _downcast_f64,
    _downcast_result,
    _floating_tensors,
    _max_ulp_diff,
    _module_device,
    _op_name,
    _ordered,
    _replay_golden,
    _resolve_output,
    _run_unfused,
    _scales_operand,
    _scales_or_fuses,
    _seed_bfb,
    _to_cpu,
    _unfused_convolution,
    _upcast,
    _write_back,
    _write_golden,
    assert_bfb_against_golden,
    bfb_devices,
    changed_state,
    first_tensor,
    host_agnostic_numerics,
    load_golden,
    move_to_device,
    portable_half_precision,
    randomize_parameters,
    regenerate_golden,
    save_golden,
    stale_post_states,
)
from priml.testing.golden import expect_golden_mismatch, pack, tensor_bits_equal


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent


class _Load(Protocol):
    def __call__(
        self,
        f: str | Path,
        *,
        map_location: str | torch.device | None = None,
        weights_only: bool,
    ) -> object: ...


_torch_load: _Load = torch.load


def _scale_tensor(value: Tensor, factor: float) -> Tensor:
    return value * factor


@dataclass(frozen=True, kw_only=True, slots=True)
class _FixedResult:
    result: Tensor

    def __call__(self, /, *args: object, **kwargs: object) -> Tensor:
        del args, kwargs
        return self.result


@pytest.fixture(autouse=True)
def isolate_regenerate_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ``--regenerate-b4b`` so a global regen run cannot disable these tests.

    The drift-detection and round-trip tests mint into ``tmp_path`` and must
    compare, not regenerate. A suite-wide ``--regenerate-b4b`` (set to remint
    committed goldens) would otherwise force every ``assert_bfb_against_golden``
    to regenerate, so the drift tests see no mismatch and fail "DID NOT RAISE".
    Tests that need regeneration call ``regenerate_golden``, which sets the flag
    locally.
    """
    regenerate.override(monkeypatch, b4b=False)


def _build_min_linear() -> nn.Linear:
    return nn.Linear(4, 3, bias=True)


def _build_min_input() -> Tensor:
    return torch.linspace(-1.0, 1.0, 8).reshape(2, 4)


def _raise_runner_failure(module: nn.Module, inp: object) -> Tensor:
    del module, inp
    raise RuntimeError("runner failed")


def _fake_cuda_module_device(module: nn.Module) -> str:
    del module
    return "cuda"


def _seed_cpu(seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.set_rng_state(generator.get_state())


@dataclass(frozen=True, kw_only=True, slots=True)
class _TorchProcessState:
    """Independent snapshot used to verify BFB process-state restoration."""

    algorithms_enabled: bool
    warn_only_enabled: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    flash_sdp_enabled: bool
    memory_efficient_sdp_enabled: bool
    rng_state: Tensor
    cublas_workspace_config: str | None


def _capture_torch_process_state() -> _TorchProcessState:
    return _TorchProcessState(
        algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        warn_only_enabled=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),
        memory_efficient_sdp_enabled=(torch.backends.cuda.mem_efficient_sdp_enabled()),
        rng_state=torch.get_rng_state(),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def _restore_torch_process_state(state: _TorchProcessState) -> None:
    torch.use_deterministic_algorithms(
        state.algorithms_enabled,
        warn_only=state.warn_only_enabled,
    )
    torch.backends.cudnn.benchmark = state.cudnn_benchmark
    torch.backends.cudnn.deterministic = state.cudnn_deterministic
    torch.backends.cuda.enable_flash_sdp(state.flash_sdp_enabled)
    torch.backends.cuda.enable_mem_efficient_sdp(state.memory_efficient_sdp_enabled)
    torch.set_rng_state(state.rng_state)
    if state.cublas_workspace_config is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = state.cublas_workspace_config


def _assert_torch_process_state_equal(
    actual: _TorchProcessState,
    expected: _TorchProcessState,
) -> None:
    assert actual.algorithms_enabled == expected.algorithms_enabled
    assert actual.warn_only_enabled == expected.warn_only_enabled
    assert actual.cudnn_benchmark == expected.cudnn_benchmark
    assert actual.cudnn_deterministic == expected.cudnn_deterministic
    assert actual.flash_sdp_enabled == expected.flash_sdp_enabled
    assert actual.memory_efficient_sdp_enabled == expected.memory_efficient_sdp_enabled
    assert torch.equal(actual.rng_state, expected.rng_state)
    assert actual.cublas_workspace_config == expected.cublas_workspace_config


def test_randomize_parameters_replaces_all() -> None:
    m = nn.Linear(4, 3)
    assert m.bias is not None
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
    randomize_parameters(m, seed=42)
    assert not torch.equal(m.weight, torch.zeros_like(m.weight))
    assert not torch.equal(m.bias, torch.zeros_like(m.bias))


def test_randomize_parameters_is_deterministic_per_seed() -> None:
    m1 = nn.Linear(4, 3)
    m2 = nn.Linear(4, 3)
    randomize_parameters(m1, seed=42)
    randomize_parameters(m2, seed=42)
    assert m1.bias is not None
    assert m2.bias is not None
    assert torch.equal(m1.weight, m2.weight)
    assert torch.equal(m1.bias, m2.bias)


def test_bfb_round_trip(tmp_path: Path) -> None:
    testdata = tmp_path / "nested" / "testdata"
    # A missing committed golden is regenerated but remains red until reviewed.
    with pytest.raises(AssertionError, match="Missing golden regenerated") as error:
        assert_bfb_against_golden(
            golden_dir=testdata,
            golden_name="linear_min",
            build_module=_build_min_linear,
            build_input=_build_min_input,
            seed=0,
        )
    assert str(error.value) == (
        f"Missing golden regenerated at {testdata / 'linear_min.pt'}; inspect it, "
        "then rerun the test."
    )
    assert (testdata / "linear_min.pt").exists()

    # Second call: golden exists -> compares; must pass.
    assert_bfb_against_golden(
        golden_dir=testdata,
        golden_name="linear_min",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )


class _ManySmallTensors(nn.Module):
    """Ninety-six small parameters: per-tensor pickle framing dominates their bytes."""

    def __init__(self) -> None:
        super().__init__()
        self.cells = nn.ParameterList(nn.Parameter(torch.zeros(4)) for _ in range(96))

    @override
    def forward(self, input: Tensor) -> Tensor:
        return input * torch.stack(list(self.cells)).sum()


def test_golden_state_is_stored_packed(tmp_path: Path) -> None:
    """State dicts cost their bytes, not ~300 bytes of framing per tensor."""
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="many",
        build_module=_ManySmallTensors,
        build_input=_build_min_input,
    )
    raw = 96 * 4 * 4
    assert (tmp_path / "many.pt").stat().st_size < raw + 4096
    assert_bfb_against_golden(
        golden_dir=tmp_path,
        golden_name="many",
        build_module=_ManySmallTensors,
        build_input=_build_min_input,
    )


class _WideMutated(nn.Module):
    """One parameter of ``width`` elements, mutated by the run."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))

    @override
    def forward(self, input: Tensor) -> Tensor:
        with torch.no_grad():
            self.weight.add_(1.0)
        return input * self.weight.sum()


def test_post_run_state_is_stored_whole(tmp_path: Path) -> None:
    """A mutated tensor is stored twice: its pre-run copy and its post-run value."""
    sizes: list[int] = []
    for width in (4, 4_004):
        name = f"wide{width}"
        regenerate_golden(
            golden_dir=tmp_path,
            golden_name=name,
            build_module=partial(_WideMutated, width),
            build_input=_build_min_input,
        )
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name=name,
            build_module=partial(_WideMutated, width),
            build_input=_build_min_input,
        )
        sizes.append((tmp_path / f"{name}.pt").stat().st_size)
    growth = sizes[1] - sizes[0]
    one_copy = 4_000 * 4
    assert 2 * one_copy <= growth < 2 * one_copy + 512


class _WideOutput(nn.Module):
    """Returns ``width`` values, all depending on the one parameter."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.width = width

    @override
    def forward(self, input: Tensor) -> Tensor:
        return torch.arange(self.width, dtype=torch.float32) * self.scale * input.sum()


def test_output_is_stored_whole(tmp_path: Path) -> None:
    """The stored output is the run's output, readable without rerunning."""
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="out",
        build_module=partial(_WideOutput, 64),
        build_input=_build_min_input,
    )
    output = load_golden(tmp_path / "out.pt")["output"]
    assert output.shape == (64,)
    assert torch.equal(output[:2], torch.tensor([0.0, 1.0]) * output[1])


def test_output_drift_reports_ulps(tmp_path: Path) -> None:
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="out",
        build_module=partial(_WideOutput, 64),
        build_input=_build_min_input,
    )

    def late_drift(module: nn.Module, inp: Tensor) -> Tensor:
        output = cast(object, module(inp))
        assert isinstance(output, Tensor)
        return output.index_fill(0, torch.tensor([63]), 0.5)

    with pytest.raises(AssertionError, match=r"output: .*max_ulp_diff=\d+"):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="out",
            build_module=partial(_WideOutput, 64),
            build_input=_build_min_input,
            run=late_drift,
        )


def test_changed_state_keeps_new_and_bitwise_changed_entries() -> None:
    value = torch.zeros(2)
    assert set(changed_state({"a": value}, {"a": value, "b": value})) == {"b"}
    assert set(changed_state({"a": value}, {"a": torch.zeros(3)})) == {"a"}
    assert set(changed_state({"a": value}, {"a": -value})) == {"a"}
    assert not changed_state({"a": value}, {"a": value.clone()})


def test_changed_state_detaches_and_copies_live_values() -> None:
    live = torch.tensor([1.0, 2.0], requires_grad=True)
    captured = changed_state({}, {"weight": live})["weight"]
    assert not captured.requires_grad
    with torch.no_grad():
        live.add_(1)
    assert torch.equal(captured, torch.tensor([1.0, 2.0]))


@pytest.mark.gpu_torch_cuda
def test_changed_state_copies_cuda_values_to_cpu() -> None:
    live = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    captured = changed_state({}, {"weight": live})["weight"]
    assert captured.device.type == "cpu"
    assert not captured.requires_grad
    with torch.no_grad():
        live.add_(1)
    assert torch.equal(captured, torch.tensor([1.0, 2.0]))


def test_missing_golden_always_fails_after_minting(tmp_path: Path) -> None:
    golden_dir = tmp_path / "goldens"

    with pytest.raises(AssertionError, match="Missing golden regenerated"):
        assert_bfb_against_golden(
            golden_dir=golden_dir,
            golden_name="linear_min",
            build_module=_build_min_linear,
            build_input=_build_min_input,
            seed=0,
        )

    assert (golden_dir / "linear_min.pt").exists()


def test_bfb_devices_is_cpu_only() -> None:
    assert bfb_devices() == ["cpu"]


@pytest.mark.parametrize("enabled_before", [True, False])
def test_portable_half_precision_disables_onednn_then_restores_it(
    enabled_before: bool,
) -> None:
    (original,) = torch.backends.mkldnn.set_flags(enabled_before, _fp32_precision=None)[
        :1
    ]
    try:
        with portable_half_precision():
            assert not torch.backends.mkldnn.is_available() or not _onednn_enabled()
        assert _onednn_enabled() == enabled_before
    finally:
        torch.backends.mkldnn.set_flags(original, _fp32_precision=None)


def _onednn_enabled() -> bool:
    """Read the oneDNN enable bit through the same flags API the harness uses."""
    return torch.backends.mkldnn.set_flags(_fp32_precision=None)[0]


def test_move_to_device_recurses_into_containers_and_passes_scalars() -> None:
    tensor = torch.zeros(1)
    moved = move_to_device({"a": (tensor, 3), "b": [tensor, "x"]}, "cpu")
    inner = moved["a"]
    assert isinstance(inner, tuple)
    assert inner[1] == 3
    first = inner[0]
    assert torch.equal(first, tensor)
    items = moved["b"]
    assert isinstance(items, list)
    assert items[1] == "x"
    assert move_to_device(7, "cpu") == 7


def test_to_cpu_clones_tensors_and_recurses_into_containers() -> None:
    tensor = torch.zeros(1)
    snapshot = cast(
        dict[str, object],
        _to_cpu({"a": (tensor, 3), "b": [tensor, "x"]}),
    )
    inner = cast(tuple[object, ...], snapshot["a"])
    copy = inner[0]
    assert isinstance(copy, Tensor)
    tensor.fill_(1)
    assert torch.equal(copy, torch.zeros(1))
    assert inner[1] == 3
    items = cast(list[object], snapshot["b"])
    assert items[1] == "x"
    assert _to_cpu(None) is None


class _OtherDevice(Tensor):
    """A CPU tensor reporting a different device, to reach the cross-device path."""

    device = torch.device("meta")


def test_equality_checks_compare_across_devices_on_cpu() -> None:
    elsewhere = _OtherDevice(torch.tensor([1.0, 2.0]))
    local = torch.tensor([1.0, 2.0])
    assert elsewhere.device != local.device
    _assert_equal(elsewhere, local, label="output")
    assert not changed_state({"a": elsewhere}, {"a": local})


def test_assert_equal_reports_non_tensor_shape_and_dtype_mismatches() -> None:
    _assert_equal(3, 3, label="seed")
    with pytest.raises(AssertionError, match="seed: non-tensor mismatch 3 vs 4"):
        _assert_equal(3, 4, label="seed")
    with pytest.raises(AssertionError, match=r"shape mismatch \(2,\) vs \(3,\)"):
        _assert_equal(torch.zeros(2), torch.zeros(3), label="output")
    with pytest.raises(
        AssertionError,
        match=r"dtype mismatch torch\.float32 vs torch\.float64",
    ):
        _assert_equal(
            torch.zeros(2),
            torch.zeros(2, dtype=torch.float64),
            label="output",
        )


def test_module_device_falls_back_to_buffers_then_cpu() -> None:
    buffered = nn.Module()
    buffered.register_buffer("scale", torch.ones(1))
    assert _module_device(buffered) == "cpu"
    assert _module_device(nn.Module()) == "cpu"


def test_upcast_and_downcast_recurse_into_tuples_and_pass_scalars() -> None:
    narrow = torch.zeros(1, dtype=torch.bfloat16)
    wide = torch.zeros(1, dtype=torch.float64)
    up = cast(tuple[object, ...], _upcast((narrow, 2)))
    assert cast(Tensor, up[0]).dtype == torch.float64
    assert up[1] == 2
    down = cast(list[object], _downcast_f64([wide, (wide, "x")]))
    assert cast(Tensor, down[0]).dtype == torch.float32
    pair = cast(tuple[object, ...], down[1])
    assert cast(Tensor, pair[0]).dtype == torch.float32
    assert pair[1] == "x"


def test_downcast_result_reads_dtypes_from_tuple_arguments() -> None:
    wide = torch.zeros(1, dtype=torch.float64)
    half = torch.zeros(1, dtype=torch.float16)
    result = cast(
        list[object],
        _downcast_result([wide, wide], ((half, half),), {}, torch.float32),
    )
    assert [cast(Tensor, value).dtype for value in result] == [torch.float16] * 2
    assert (
        cast(Tensor, _downcast_result(wide, (), {}, torch.float32)).dtype
        == torch.float32
    )


def test_write_back_maps_a_listed_result_onto_the_original() -> None:
    original = torch.zeros(2, dtype=torch.float32)
    other = torch.ones(2, dtype=torch.float32)
    computed = torch.full((2,), 0.5, dtype=torch.float64)
    add_ = cast(_AtenInPlaceAdd, torch.ops.aten.add_).Tensor
    returned = _write_back(
        add_,
        (original, other),
        {},
        (computed, other.double()),
        {},
        result=[computed],
        target=torch.float32,
    )
    assert isinstance(returned, list)
    assert returned[0] is original
    assert torch.equal(original, torch.full((2,), 0.5))


def test_bfb_files_do_not_use_typing_any() -> None:
    paths = [
        _THIS,
        _CWD / "bfb.py",
        _CWD.parents[0] / "model" / "transformer" / "block_test.py",
    ]
    offenders = list[str]()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            (isinstance(node, ast.Name) and node.id == "Any")
            or (isinstance(node, ast.Attribute) and node.attr == "Any")
            for node in ast.walk(tree)
        ):
            offenders.append(str(path))

    assert not offenders, f"typing.Any used in {offenders}"


def test_first_tensor_extracts_and_validates_primary_output() -> None:
    tensor = torch.tensor([1.0])

    assert first_tensor(tensor) is tensor
    assert first_tensor((tensor, object())) is tensor
    assert first_tensor([tensor, object()]) is tensor
    with pytest.raises(TypeError):
        first_tensor(object())
    with pytest.raises(TypeError):
        first_tensor((object(),))
    with pytest.raises(TypeError):
        first_tensor([object()])
    with pytest.raises(TypeError):
        first_tensor(())
    with pytest.raises(TypeError):
        first_tensor([])


def test_bfb_preserves_a_falsey_runner(tmp_path: Path) -> None:
    class FalseyRunner:
        calls = 0

        def __bool__(self) -> bool:
            return False

        def __call__(self, module: nn.Module, inp: Tensor) -> Tensor:
            self.calls += 1
            output = cast(object, module(inp))
            assert isinstance(output, Tensor)
            return output

    runner = FalseyRunner()

    with pytest.raises(AssertionError, match="Missing golden regenerated"):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="falsey_runner",
            build_module=_build_min_linear,
            build_input=_build_min_input,
            run=runner,
        )

    assert runner.calls == 2


def test_bfb_rejects_non_cpu_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "priml.testing.bfb._module_device",
        _fake_cuda_module_device,
    )

    with pytest.raises(ValueError, match="CPU-only") as error:
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="cuda_is_not_hermetic",
            build_module=_build_min_linear,
            build_input=_build_min_input,
        )
    assert str(error.value) == "The BFB harness is CPU-only."


def test_bfb_rejects_non_cpu_module_on_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A golden minted on CPU is not replayed against a module elsewhere."""
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="linear_min",
        build_module=_build_min_linear,
        build_input=_build_min_input,
    )
    monkeypatch.setattr(
        "priml.testing.bfb._module_device",
        _fake_cuda_module_device,
    )

    with pytest.raises(ValueError, match="CPU-only") as error:
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="linear_min",
            build_module=_build_min_linear,
            build_input=_build_min_input,
        )
    assert str(error.value) == "The BFB harness is CPU-only."


def test_explicit_regeneration_of_an_existing_golden_returns_normally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a MISSING golden stays red after minting; a remint is reviewed by diff."""
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="linear_min",
        build_module=_build_min_linear,
        build_input=_build_min_input,
    )
    path = tmp_path / "linear_min.pt"
    before = _loaded_golden(path)["state_dict"]
    regenerate.override(monkeypatch, b4b=True)

    assert_bfb_against_golden(
        golden_dir=tmp_path,
        golden_name="linear_min",
        build_module=_build_min_linear,
        build_input=_build_min_input,
    )

    # The archive's internal name changes with the temp file; the payload does not.
    assert not changed_state(before, _loaded_golden(path)["state_dict"])


def test_default_runner_splats_a_tuple_input_and_requires_a_tensor(
    tmp_path: Path,
) -> None:
    class TwoArgModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 4)

        @override
        def forward(self, a: Tensor, b: Tensor) -> Tensor:
            return self.lin(a) + b

    def build_input() -> tuple[Tensor, Tensor]:
        return (
            torch.linspace(-1.0, 1.0, 6).reshape(2, 3),
            torch.linspace(0.0, 1.0, 8).reshape(2, 4),
        )

    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="two_arg_tuple",
        build_module=TwoArgModule,
        build_input=build_input,
    )
    assert_bfb_against_golden(
        golden_dir=tmp_path,
        golden_name="two_arg_tuple",
        build_module=TwoArgModule,
        build_input=build_input,
    )

    class ListModule(nn.Module):
        @override
        def forward(self, a: Tensor) -> list[Tensor]:
            return [a]

    with pytest.raises(TypeError, match="returns a Tensor"):
        regenerate_golden(
            golden_dir=tmp_path,
            golden_name="list_output",
            build_module=ListModule,
            build_input=_build_min_input,
        )


def test_bfb_reports_state_keys_added_by_the_run(tmp_path: Path) -> None:
    """A runner that grows the state dict fails on the KEY set, not a value."""
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="linear_min",
        build_module=_build_min_linear,
        build_input=_build_min_input,
    )

    def growing_runner(module: nn.Module, inp: Tensor) -> Tensor:
        output = cast(object, module(inp))
        assert isinstance(output, Tensor)
        module.register_buffer("extra", torch.zeros(1))
        return output

    with pytest.raises(AssertionError, match=r"added=\['extra'\] removed=\[\]"):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="linear_min",
            build_module=_build_min_linear,
            build_input=_build_min_input,
            run=growing_runner,
        )


def test_bfb_restores_process_state_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _capture_torch_process_state()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        _seed_cpu(981)
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        expected = _capture_torch_process_state()

        regenerate_golden(
            golden_dir=tmp_path,
            golden_name="restores_success",
            build_module=_build_min_linear,
            build_input=_build_min_input,
        )

        _assert_torch_process_state_equal(
            _capture_torch_process_state(),
            expected,
        )
    finally:
        _restore_torch_process_state(original)


def test_bfb_restores_process_state_after_runner_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _capture_torch_process_state()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        _seed_cpu(982)
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        expected = _capture_torch_process_state()

        with pytest.raises(RuntimeError, match="runner failed"):
            assert_bfb_against_golden(
                golden_dir=tmp_path,
                golden_name="restores_failure",
                build_module=_build_min_linear,
                build_input=_build_min_input,
                run=_raise_runner_failure,
            )

        _assert_torch_process_state_equal(
            _capture_torch_process_state(),
            expected,
        )
    finally:
        _restore_torch_process_state(original)


def test_bfb_detects_forward_drift(tmp_path: Path) -> None:
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="linear_min",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )

    class _DriftLinear(nn.Linear):
        @override
        def forward(self, input: Tensor) -> Tensor:
            return super().forward(input) + 1e-3

    with pytest.raises(AssertionError, match="output"):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="linear_min",
            build_module=lambda: _DriftLinear(4, 3, bias=True),
            build_input=_build_min_input,
            seed=0,
        )


def test_bfb_detects_a_changed_input(tmp_path: Path) -> None:
    """A test whose input moved on no longer matches the golden it replays."""
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="linear_input",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )

    with pytest.raises(AssertionError, match="input") as error:
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="linear_input",
            build_module=_build_min_linear,
            build_input=lambda: torch.randn(3, 4),
            seed=0,
        )
    assert str(error.value) == "input: shape mismatch (3, 4) vs (2, 4)"


def test_expect_golden_mismatch_blocks_bfb_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="linear_min",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )

    class _DriftLinear(nn.Linear):
        @override
        def forward(self, input: Tensor) -> Tensor:
            return super().forward(input) + 1e-3

    regenerate.override(monkeypatch, b4b=True)
    before = (tmp_path / "linear_min.pt").read_bytes()
    assert_bfb_against_golden(
        golden_dir=tmp_path,
        golden_name="linear_min",
        build_module=lambda: _DriftLinear(4, 3, bias=True),
        build_input=_build_min_input,
        seed=0,
    )
    assert (tmp_path / "linear_min.pt").read_bytes() != before

    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="linear_min",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )
    before = (tmp_path / "linear_min.pt").read_bytes()
    with expect_golden_mismatch(match=r"output"):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="linear_min",
            build_module=lambda: _DriftLinear(4, 3, bias=True),
            build_input=_build_min_input,
            seed=0,
        )
    assert (tmp_path / "linear_min.pt").read_bytes() == before


def test_bfb_detects_state_dict_key_drift(tmp_path: Path) -> None:
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="linear_min",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )

    class _ExtraParamLinear(nn.Linear):
        def __init__(self) -> None:
            super().__init__(4, 3, bias=True)
            self.scale = nn.Parameter(torch.ones(1))

    with pytest.raises((AssertionError, RuntimeError)):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="linear_min",
            build_module=_ExtraParamLinear,
            build_input=_build_min_input,
            seed=0,
        )


def test_regenerate_golden_helper_overwrites(tmp_path: Path) -> None:
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="linear_min",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )
    assert not regenerate.b4b()
    assert (tmp_path / "linear_min.pt").exists()


def test_regenerate_golden_does_not_swallow_runner_assertion(tmp_path: Path) -> None:
    def runner(module: nn.Module, inp: Tensor) -> Tensor:
        del module, inp
        raise AssertionError("Missing golden regenerated: model failure")

    with pytest.raises(AssertionError, match="model failure"):
        regenerate_golden(
            golden_dir=tmp_path,
            golden_name="runner_failure",
            build_module=_build_min_linear,
            build_input=_build_min_input,
            run=runner,
        )


def test_dict_input_dispatches_via_kwargs(tmp_path: Path) -> None:
    class TwoArgModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 4)

        @override
        def forward(self, a: Tensor, b: Tensor) -> Tensor:
            return self.lin(a) + b

    def build_module() -> nn.Module:
        return TwoArgModule()

    def build_input() -> dict[str, Tensor]:
        return {
            "a": torch.linspace(-1.0, 1.0, 6).reshape(2, 3),
            "b": torch.linspace(0.0, 1.0, 8).reshape(2, 4),
        }

    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="two_arg",
        build_module=build_module,
        build_input=build_input,
        seed=0,
    )
    assert_bfb_against_golden(
        golden_dir=tmp_path,
        golden_name="two_arg",
        build_module=build_module,
        build_input=build_input,
        seed=0,
    )


def test_param_mutating_runner_captures_post_state(tmp_path: Path) -> None:
    """A runner that overwrites params should produce a stable post-state."""

    def build_module() -> nn.Module:
        return nn.Linear(4, 3, bias=True)

    def build_input() -> Tensor:
        return torch.linspace(-1.0, 1.0, 8).reshape(2, 4)

    def mutating_runner(module: nn.Module, inp: Tensor) -> Tensor:
        assert isinstance(module, nn.Linear)
        assert module.bias is not None
        out = module(inp)
        with torch.no_grad():
            module.bias.add_(out.mean())
        return out

    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="mutating",
        build_module=build_module,
        build_input=build_input,
        seed=0,
        run=mutating_runner,
    )
    # Second call must pass (verifies post-state captured).
    assert_bfb_against_golden(
        golden_dir=tmp_path,
        golden_name="mutating",
        build_module=build_module,
        build_input=build_input,
        seed=0,
        run=mutating_runner,
    )


def test_no_checked_in_golden_stores_an_unchanged_post_state() -> None:
    """Every priml golden's post-run state holds only tensors the run changed.

    Gated rather than trusted because the omission arrived as a WRITER change
    with no migration: nineteen goldens minted before it kept the copy, the
    tolerant reader never went red, and 1.1 MiB sat unnoticed until someone
    read a size diff. A guard that only stops new violations leaves the
    existing ones invisible forever.
    """
    goldens = sorted(_CWD.parent.rglob("*.pt"))
    assert goldens, "no goldens found; the glob no longer matches the layout"
    stale = stale_post_states(goldens)
    assert not stale, f"goldens storing an unchanged post-state: {stale}"


def test_stale_post_states_reports_then_clears_a_synthetic_golden(
    tmp_path: Path,
) -> None:
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="buffer_mutating",
        build_module=_BufferMutatingModule,
        build_input=_build_min_input,
    )
    path = tmp_path / "buffer_mutating.pt"
    assert stale_post_states([path]) == []
    payload = load_golden(path)
    post = payload.get("post_state")
    assert post is not None
    payload["post_state"] = {**post, "lin.bias": payload["state_dict"]["lin.bias"]}
    save_golden(path, payload)
    assert stale_post_states([path]) == [path]


# Only the state entries are returned: a golden also holds an input, an output digest,
# and a seed, and typing those as state dicts to satisfy one reader would be a false
# annotation.
def _loaded_golden(path: Path) -> dict[str, dict[str, Tensor]]:
    """Read a golden's two state dicts; a file with none (not a bfb golden) is empty."""
    raw = from_plain(
        _torch_load(path, map_location="cpu", weights_only=False),
        dict[str, object],
    )
    if "state_dict" not in raw:
        return {}
    payload = load_golden(path)
    states = {"state_dict": payload["state_dict"]}
    if "post_state" in payload:
        states["post_state"] = payload["post_state"]
    return states


def test_detects_post_state_drift(tmp_path: Path) -> None:
    """A second runner that produces a different post-state should fail."""

    def build_module() -> nn.Module:
        return nn.Linear(4, 3, bias=True)

    def build_input() -> Tensor:
        return torch.linspace(-1.0, 1.0, 8).reshape(2, 4)

    def runner_v1(module: nn.Module, inp: Tensor) -> Tensor:
        assert isinstance(module, nn.Linear)
        assert module.bias is not None
        out = module(inp)
        with torch.no_grad():
            module.bias.add_(0.5)
        return out

    def runner_v2(module: nn.Module, inp: Tensor) -> Tensor:
        # Same output (because we mutate AFTER computing out), but a
        # different post-state mutation.
        assert isinstance(module, nn.Linear)
        assert module.bias is not None
        out = module(inp)
        with torch.no_grad():
            module.bias.add_(0.7)
        return out

    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="mutating_drift",
        build_module=build_module,
        build_input=build_input,
        seed=0,
        run=runner_v1,
    )
    with pytest.raises(AssertionError, match="state\\["):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="mutating_drift",
            build_module=build_module,
            build_input=build_input,
            seed=0,
            run=runner_v2,
        )


class _BufferMutatingModule(nn.Module):
    """Module that mutates a registered buffer during a non-mutating forward.

    Mirrors BatchNorm-style ``running_mean`` updates: ``forward`` does not
    touch parameters, but the live ``state_dict`` changes after the call.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3, bias=True)
        self._running_sum: Tensor = torch.zeros(3)
        self.register_buffer("running_sum", self._running_sum)
        self.running_sum = self._running_sum

    @override
    def forward(self, input: Tensor) -> Tensor:
        out = self.lin(input)
        self.running_sum.add_(out.detach().sum(dim=0))
        return out


def test_bfb_captures_forward_buffer_mutation(tmp_path: Path) -> None:
    """A forward that mutates a buffer must round-trip bit-exactly.

    Comparing the post-forward live state to the PRE-run golden would falsely
    fail. The buffer mutation must be captured and compared bit-for-bit on
    its own.
    """
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="buffer_mutating",
        build_module=_BufferMutatingModule,
        build_input=_build_min_input,
        seed=0,
    )
    # Second call loads the pre-run golden, reruns, and must match the
    # captured post-forward buffer exactly -- not the pre-run zeros.
    assert_bfb_against_golden(
        golden_dir=tmp_path,
        golden_name="buffer_mutating",
        build_module=_BufferMutatingModule,
        build_input=_build_min_input,
        seed=0,
    )
    payload = _loaded_golden(tmp_path / "buffer_mutating.pt")
    pre = payload["state_dict"]["running_sum"]
    post = payload["post_state"]["running_sum"]
    assert torch.equal(pre, torch.zeros(3))
    assert not torch.equal(post, torch.zeros(3))


def test_bfb_detects_forward_buffer_drift(tmp_path: Path) -> None:
    """A divergent buffer mutation must be caught."""
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="buffer_mutating",
        build_module=_BufferMutatingModule,
        build_input=_build_min_input,
        seed=0,
    )

    class _DriftBuffer(_BufferMutatingModule):
        @override
        def forward(self, input: Tensor) -> Tensor:
            out = super().forward(input)
            self.running_sum.add_(1.0)
            return out

    with pytest.raises(AssertionError, match="running_sum"):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="buffer_mutating",
            build_module=_DriftBuffer,
            build_input=_build_min_input,
            seed=0,
        )


def test_regenerate_round_trips_immediately(tmp_path: Path) -> None:
    """Regeneration must verify the freshly written golden round-trips.

    A regenerator that writes a golden whose output is not reproducible on
    reload must fail loudly during regeneration, not silently pass.
    """
    drift = {"n": 0}

    def flaky_runner(module: nn.Module, inp: Tensor) -> Tensor:
        # First call (capture) returns the clean output; the verification
        # rerun returns a perturbed output, so the golden cannot round-trip.
        out = cast(object, module(inp))
        assert isinstance(out, Tensor)
        drift["n"] += 1
        if drift["n"] >= 2:
            return out + 1e-3
        return out

    with pytest.raises(AssertionError):
        regenerate_golden(
            golden_dir=tmp_path,
            golden_name="flaky",
            build_module=_build_min_linear,
            build_input=_build_min_input,
            seed=0,
            run=flaky_runner,
        )


def test_failed_regeneration_preserves_the_last_valid_golden(tmp_path: Path) -> None:
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="linear_min",
        build_module=_build_min_linear,
        build_input=_build_min_input,
    )
    path = tmp_path / "linear_min.pt"
    original = path.read_bytes()
    calls = 0

    def flaky_runner(module: nn.Module, inp: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        output = cast(object, module(inp))
        assert isinstance(output, Tensor)
        return output + calls * 1e-3

    with pytest.raises(AssertionError):
        regenerate_golden(
            golden_dir=tmp_path,
            golden_name="linear_min",
            build_module=_build_min_linear,
            build_input=_build_min_input,
            run=flaky_runner,
        )

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_mint_keeps_its_candidate_in_the_golden_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden_dir = tmp_path / "nested" / "goldens"
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(outside))
    calls = 0

    def runner(module: nn.Module, value: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            # Verification sees the candidate before publication. Keeping it
            # here gives rename the destination's filesystem and permissions.
            assert not (golden_dir / "identity.pt").exists()
            assert any(path.is_file() for path in golden_dir.rglob("*"))
            # An interrupted mint can be traced to its owning golden.
            assert all(
                path.name.startswith("identity.pt") for path in golden_dir.iterdir()
            )
        output = cast(object, module(value))
        assert isinstance(output, Tensor)
        return output

    regenerate_golden(
        golden_dir=golden_dir,
        golden_name="identity",
        build_module=nn.Identity,
        build_input=lambda: torch.zeros(2, 3),
        run=runner,
    )
    assert calls == 2
    assert list(golden_dir.iterdir()) == [golden_dir / "identity.pt"]
    assert not list(outside.iterdir())


def test_regenerate_round_trip_passes_for_clean_module(tmp_path: Path) -> None:
    """A deterministic module regenerates and self-verifies cleanly."""
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="clean",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )
    assert (tmp_path / "clean.pt").exists()
    assert_bfb_against_golden(
        golden_dir=tmp_path,
        golden_name="clean",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )


# An exact-allowlist op must be host-independent: computing it in float64 and narrowing
# back to float32 must reproduce the native float32 result bit-for-bit. An op that
# fails this (e.g. a fused multiply-add rounding differently, or a vector-width-
# dependent reduction) must NOT be allowlisted -- it has to be upcast like every other
# arithmetic op.
#
# Each probe applies exactly ONE allowlisted op to its float32 input; any auxiliary
# operand must be exactly float32-representable (an integer, a power of two, or a
# flip/copy of the input) so the probe isolates the op under test rather than folding in
# a second op's rounding.
def _f32_equals_f64_downcast(op: TensorFn) -> bool:
    """Report whether ``op``'s float32 result equals its float64-then-downcast."""
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(4096, dtype=torch.float32, generator=gen)
    return torch.equal(op(a), op(a.double()).float())


# Arithmetic exact-allowlist ops, each paired with a single-op probe whose
# auxiliary operand is exactly float32-representable (so only the named op's
# rounding is under test). Pure data-movement ops (views, reshapes, gathers)
# carry no arithmetic and so are trivially host-independent; only ops that
# compute a value need proving here.
_EXACT_ARITHMETIC_PROBES: Final[dict[str, TensorFn]] = {
    "add": lambda a: a + a,
    "sub": lambda a: a - a.flip(0),
    "mul": lambda a: a * a,
    "div": lambda a: a / 3.0,
    "neg": lambda a: -a,
    "abs": lambda a: a.abs(),
    "clamp": lambda a: a.clamp(-0.5, 0.5),
    "clamp_min": lambda a: a.clamp_min(0.1),
    "clamp_max": lambda a: a.clamp_max(0.1),
    "sign": lambda a: a.sign(),
    "maximum": lambda a: torch.maximum(a, a.flip(0)),
    "minimum": lambda a: torch.minimum(a, a.flip(0)),
    "where": lambda a: torch.where(a > 0, a, a.flip(0)),
}


@pytest.mark.parametrize("name", sorted(_EXACT_ARITHMETIC_PROBES))
def test_exact_f32_ops_are_host_independent(name: str) -> None:
    """Every arithmetic op on the exact-allowlist is float64-recompute-stable.

    This is the completeness guard for the upcast-by-default policy. The
    allowlist is the only place a host-dependent op can hide: anything NOT
    listed is upcast and therefore safe. So each listed arithmetic op must
    prove its float32 result equals the float64-then-downcast result; a wrongly
    added entry (e.g. ``addcmul_``, whose fused rounding differs) fails here at
    commit time rather than silently minting a non-portable golden.
    """
    assert name in _EXACT_F32_OPS
    assert _f32_equals_f64_downcast(_EXACT_ARITHMETIC_PROBES[name])


# Declared non-arithmetic ops that touch float32 data, each with a single-op
# probe using a FIXED index/operand so the f32 and f64 calls are identical. Used
# to prove the "host-independent by construction" claim rather than trust it.
_NONARITHMETIC_PROBES: Final[dict[str, TensorFn]] = {
    "gather": lambda a: a.gather(0, torch.arange(0, a.numel(), 7) % a.numel()),
    "index_select": lambda a: a.index_select(0, torch.arange(0, 50)),
    "masked_fill": lambda a: a.masked_fill(a > 0, 0.123),
    "masked_fill_": lambda a: a.clone().masked_fill_(a > 0, 0.123),
    "masked_select": lambda a: a.masked_select(a > 0),
    "cat": lambda a: torch.cat([a, a.flip(0)]),
    "stack": lambda a: torch.stack([a, a.flip(0)]),
    "clone": lambda a: a.clone(),
    "copy_": lambda a: torch.empty_like(a).copy_(a),
    "_to_copy": lambda a: cast(
        Tensor,
        torch.ops.aten._to_copy(a, dtype=a.dtype),
    ),
    "fill_": lambda a: a.clone().fill_(0.123),
    "to": lambda a: a.to(torch.float32),
    "where": lambda a: torch.where(a > 0, a, a.flip(0)),
}


# Allowlisted ops tagged ``movement``/``compare`` that synthesize or select
# float32 VALUES (dtype conversion, scalar fill, value selection) rather than
# only moving bytes or returning bool/index results. Their category does not
# require an ``arith`` probe, so each must appear in ``_NONARITHMETIC_PROBES``
# explicitly -- ``test_value_producing_nonarith_ops_are_probed`` enforces this so
# a future value op mistagged ``movement`` cannot dodge proof.
_VALUE_PRODUCING_NONARITH: Final = frozenset(
    {"where", "copy_", "_to_copy", "to", "fill_", "masked_fill", "masked_fill_"},
)


def test_value_producing_nonarith_ops_are_probed() -> None:
    """Value-producing movement/compare ops carry an explicit recompute probe.

    The category guard only forces a probe for ``arith``. Ops tagged
    ``movement``/``compare`` that still synthesize/select float values (``to``,
    ``copy_``, ``fill_``, ``masked_fill*``, ``where``) could otherwise be exact
    "by construction" on the author's word alone. Require each to be probed so
    its host-independence is proven, not asserted -- and so a future value op
    mistagged ``movement`` to skip the arith requirement fails here.
    """
    assert _EXACT_F32_OPS.keys() >= _VALUE_PRODUCING_NONARITH
    probed = set(_EXACT_ARITHMETIC_PROBES) | set(_NONARITHMETIC_PROBES)
    unprobed = _VALUE_PRODUCING_NONARITH - probed
    assert not unprobed, (
        f"value-producing non-arith ops with no probe: {sorted(unprobed)}"
    )


@pytest.mark.parametrize("name", sorted(_NONARITHMETIC_PROBES))
def test_declared_nonarithmetic_ops_are_actually_exact(name: str) -> None:
    """Each declared 'pure-movement/comparison' op really is f64-recompute-stable.

    The allowlist tags these ops ``movement``/``compare`` (exact by
    construction); this proves it for every member that touches float32 data,
    so a future entry that secretly does arithmetic (rounding-dependent) fails
    here rather than minting
    a non-portable golden. Pure metadata ops with no float32 result (views,
    allocation) carry no value to compare and are exempt by inspection.
    """
    assert _EXACT_F32_OPS.get(name) in ("movement", "compare")
    assert _f32_equals_f64_downcast(_NONARITHMETIC_PROBES[name])


def test_every_allowlist_entry_is_categorized_and_arith_is_probed() -> None:
    """No allowlist entry escapes vetting; every ``arith`` op has a probe.

    ``_EXACT_F32_OPS`` tags each op ``arith`` / ``compare`` / ``movement`` at its
    definition site -- the single source of truth. This guard enforces two
    invariants on it:

    1. Every ``arith`` op has a float64-recompute probe in
       ``_EXACT_ARITHMETIC_PROBES`` (arithmetic CAN round-diverge, so it must be
       proven; a future ``addcmul_`` mis-tagged ``arith`` without a probe fails
       here).
    2. ``compare`` / ``movement`` ops carry no float rounding and are exact by
       construction; the divergence scan
       (``test_no_allowlisted_op_is_width_divergent``) independently checks them
       against torch, so they need no per-op probe -- only a valid category.

    An op in no category at all (a bare addition to the allowlist) fails the
    category check below.
    """
    uncategorized = {
        n
        for n, c in _EXACT_F32_OPS.items()
        if c not in {"arith", "compare", "movement"}
    }
    assert not uncategorized, (
        f"allowlist entries with no valid category: {sorted(uncategorized)}"
    )
    # An in-place arith op (``add_``) shares the functional op's kernel, so its
    # probe is the functional form (``add``) with the trailing underscore dropped.
    arith = {n for n, c in _EXACT_F32_OPS.items() if c == "arith"}
    unprobed = {n for n in arith if n.rstrip("_") not in _EXACT_ARITHMETIC_PROBES}
    assert not unprobed, (
        f"arith allowlist entries with no recompute probe: {sorted(unprobed)}"
    )


# Allocation ops return uninitialized memory, so any equality probe is
# meaningless (two calls differ). Excluded from the enumeration scan -- they are
# pure allocation, carry no value to diverge, and are host-independent by nature.
_ALLOCATION_OPS: Final = frozenset(
    {"empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided"},
)


# Only allowlisted names can invalidate this guard. Calling unrelated Aten packets is
# both unnecessary and unsafe: some nominally single-tensor defaults initialize platform
# backends before argument validation, and PyTorch's MPS ``pin_memory`` packet can
# segfault the interpreter. Probe the exact policy surface with a fixed seeded input
# instead.
def _divergent_allowlisted_single_tensor_ops() -> set[str]:
    """Allowlisted Aten ops that differ from their float64 recomputation."""
    divergent: set[str] = set()
    for name in sorted(_EXACT_F32_OPS):
        if name in _ALLOCATION_OPS:
            continue
        packet = cast(_AtenPacket, getattr(torch.ops.aten, name))
        # Fresh seeded input per op: an in-place or alias op must not corrupt the
        # input a later op sees, and ``op(f32)`` and ``op(f64)`` must run on the
        # same values. Clone so an in-place op mutates a private copy.
        gen = torch.Generator().manual_seed(0)
        base = torch.randn(64, dtype=torch.float32, generator=gen)
        try:
            op = packet.default
            r32 = op(base.clone())
            # Only dense (strided) float32 results carry a value comparable by
            # ``torch.equal``. A sparse / nested / non-tensor result is not a
            # plain-value op the allowlist would ever cover -- skip rather than
            # crash on ``equal`` (which raises NotImplementedError on sparse).
            if (
                not isinstance(r32, Tensor)
                or r32.dtype != torch.float32
                or r32.layout != torch.strided
            ):
                continue
            r64 = op(base.clone().double())
            if not isinstance(r64, Tensor) or r64.layout != torch.strided:
                continue
            equal = torch.equal(r32, r64.float())
        except (
            RuntimeError,
            TypeError,
            AttributeError,
            IndexError,
            ValueError,
            NotImplementedError,
        ):
            continue
        if not equal:
            divergent.add(name)
    return divergent


def test_no_allowlisted_op_is_width_divergent() -> None:
    """No allowlisted single-tensor op diverges across precision.

    An allowlisted op that fails its float64 recomputation would run native
    float32 and mint a non-portable golden. Probe every allowlisted packet
    directly instead of invoking unrelated platform-sensitive Aten defaults.
    """
    bad = _divergent_allowlisted_single_tensor_ops()
    assert not bad, (
        f"allowlisted ops whose float32 result is width-divergent (must be "
        f"upcast, not allowlisted): {sorted(bad)}"
    )


# The trace mode is entered OUTSIDE ``host_agnostic_numerics`` so it observes each op's
# arguments AFTER the upcast dispatch has run -- i.e. the dtype the real kernel actually
# computes on. Invariant the harness guarantees: every op is either allowlisted (runs
# native float32, proven host-independent) or upcast to float64. So a non-allowlisted op
# still seeing a float32 argument here ran native float32 without being upcast -- a
# cross-host-divergence leak (the failure mode the flash-attention kernel exhibited
# before the SDPA-math pin).
# A tensorless sampler (``randn``) has no float32 argument to see, so it is caught by
# its float32 result instead. ``wrap=False`` traces a call that is responsible for
# entering ``host_agnostic_numerics`` itself.
def _f32_leaking_ops(
    run: Callable[[], object],
    *,
    wrap: bool = True,
) -> set[str]:
    """Names of non-allowlisted ops that still compute in float32 under the harness."""
    leaks: set[str] = set()

    def has_f32(value: object) -> bool:
        if isinstance(value, Tensor):
            return value.dtype == torch.float32
        if isinstance(value, (list, tuple)):
            return any(
                has_f32(v) for v in cast(list[object] | tuple[object, ...], value)
            )
        return False

    class _Trace(TorchDispatchMode):
        @override
        def __torch_dispatch__(
            self,
            func: OpOverload[..., object],
            types: tuple[type, ...],
            args: tuple[object, ...] = (),
            kwargs: dict[str, object] | None = None,
        ) -> object:
            name = _op_name(func)
            values = (*args, *(kwargs or {}).values())
            result = func(*args, **(kwargs or {}))
            if (func.namespace != "aten" or name not in _EXACT_F32_OPS) and (
                any(has_f32(value) for value in values)
                or (name in _FLOAT_FACTORIES and has_f32(result))
            ):
                leaks.add(name)
            return result

    with _Trace():
        if wrap:
            with host_agnostic_numerics():
                run()
        else:
            run()
    return leaks


def test_no_unvetted_f32_op_in_transformer_forward_backward() -> None:
    """No non-allowlisted op runs natively on float32 in a real fwd+bwd.

    Traces a transformer block forward and backward under the harness and
    asserts every non-allowlisted op was upcast to float64 -- none ran on a
    native float32 kernel. Catches the class where an arithmetic/reduction op
    escapes the upcast (e.g. a future ``addcmul_``/``scatter_add_`` mistakenly
    treated as exact), whose float32 last bit is vector-width dependent.

    Scope limit (read before trusting this): it does NOT catch an op that IS
    upcast but whose *float64* result is still arch-divergent -- the
    flash-attention kernel reduces with width-dependent tiling even in float64.
    That failure is invisible on one host; only the cross-arch golden replay in
    CI catches it. This test guards the native-f32 leak; the SDPA-math pin plus
    cross-arch replay guard the f64-divergence leak.
    """
    torch.manual_seed(0)
    block = TransformerBlock.Config(
        channels_in=16,
        attn=Attention.Config(num_heads=2, channels_head=8),
    ).make()
    randomize_parameters(block, seed=0)
    inp = torch.randn(2, 4, 16, requires_grad=True)

    leaks = _f32_leaking_ops(lambda: block(inp).sum().backward())
    assert not leaks, (
        "non-allowlisted ops ran on float32 (not upcast; cross-host-divergent): "
        f"{sorted(leaks)}"
    )


class _DrawsAtConstruction(nn.Module):
    """Initializes, fills an unsaved buffer, and draws, all while being built.

    Replay rebuilds the unsaved buffer instead of loading it, and the module's
    initializer consumes the generator ``build_input`` draws from next, so both
    reach the golden. A float32 ``randn`` of 16+ elements and a ``sin`` each
    land on vector-ISA-specific bits unless computed host-agnostically.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.register_buffer(
            "phase",
            torch.randn(18)[:3].sin(),
            persistent=False,
        )

    @override
    def forward(self, input: Tensor) -> Tensor:
        return self.lin(input) + cast(Tensor, self.phase)


def test_the_harness_computes_nothing_natively_in_float32(tmp_path: Path) -> None:
    """Building, input, randomization, and the run are all host-agnostic.

    A golden is portable only if EVERY value it stores or replays was computed
    under ``host_agnostic_numerics``. Wrapping the run alone left the module's
    construction, its unsaved buffers, the parameter randomization, and the
    input outside it, so a golden minted natively failed replay under
    ``ATEN_CPU_CAPABILITY=default``.
    """

    def check() -> None:
        with pytest.raises(AssertionError, match="Missing golden regenerated"):
            assert_bfb_against_golden(
                golden_dir=tmp_path,
                golden_name="draws",
                build_module=_DrawsAtConstruction,
                build_input=lambda: torch.randn(2, 4),
            )
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="draws",
            build_module=_DrawsAtConstruction,
            build_input=lambda: torch.randn(2, 4),
        )

    leaks = _f32_leaking_ops(check, wrap=False)
    assert not leaks, f"ran natively in float32 outside the harness: {sorted(leaks)}"


def _fused_operands() -> tuple[Tensor, Tensor, Tensor]:
    """Operands built without torch's ISA-dependent sampler."""
    generator = torch.Generator().manual_seed(0)
    m = torch.randn(4096, generator=generator, dtype=torch.float64).float()
    x = torch.randn(4096, generator=generator, dtype=torch.float64).float()
    return m, x, x.abs() + 0.5


def lerp_fused(m: Tensor, x: Tensor, d: Tensor) -> Tensor:
    del d
    return m.lerp(x, 0.9)


def lerp_unfused(m: Tensor, x: Tensor, d: Tensor) -> Tensor:
    del d
    return (0.9 - 1) * (x - m) + x


def addcdiv_fused(m: Tensor, x: Tensor, d: Tensor) -> Tensor:
    return m.addcdiv(x, d, value=-0.01)


def addcdiv_unfused(m: Tensor, x: Tensor, d: Tensor) -> Tensor:
    return m + -0.01 * x / d


def addcmul_fused(m: Tensor, x: Tensor, d: Tensor) -> Tensor:
    del d
    return m.addcmul(x, x, value=0.01)


def addcmul_unfused(m: Tensor, x: Tensor, d: Tensor) -> Tensor:
    del d
    return m + 0.01 * x * x


@pytest.mark.parametrize(
    ("fused", "unfused"),
    [
        (lerp_fused, lerp_unfused),
        (addcdiv_fused, addcdiv_unfused),
        (addcmul_fused, addcmul_unfused),
    ],
)
def test_fused_multiply_add_rounds_like_separate_ops(
    fused: Callable[[Tensor, Tensor, Tensor], Tensor],
    unfused: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> None:
    """A fused kernel runs as separate correctly-rounded ops under the harness.

    The float64 kernel of ``lerp``/``addcmul``/``addcdiv`` fuses an FMA on a
    vector ISA and rounds twice on the scalar one; under cancellation the gap
    survives the round to float32. Separate float64 ops round the same on
    every host.
    """
    m, x, d = _fused_operands()
    with host_agnostic_numerics():
        got = fused(m, x, d)
    assert torch.equal(got, unfused(m.double(), x.double(), d.double()).float())


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("keyword_bias", [False, True])
def test_convolution_adds_bias_after_accumulating_products(
    dimensions: int,
    transposed: bool,
    keyword_bias: bool,
) -> None:
    """Bias survives cancellation in the dot product on ARM and x86."""
    channels = torch.tensor([2.0**40, 1, -(2.0**40), 1, 2])
    # Unit axes repeat the same _unfused_convolution cancellation at every position.
    x = (
        channels.reshape(1, 5, *((1,) * dimensions))
        .expand(
            2,
            5,
            *((4,) * dimensions),
        )
        .clone()
    )
    w = torch.ones((5, 3) if transposed else (3, 5))
    if transposed:
        w[[0, 2]] *= 2.0**40
    else:
        w[:, [0, 2]] *= 2.0**40
    w = (
        w.reshape(*w.shape, *((1,) * dimensions))
        .expand(
            *w.shape,
            *((2,) * dimensions),
        )
        .clone()
    )
    bias = torch.arange(3, dtype=torch.float32)
    options: dict[str, object] = {
        "stride": [1] * dimensions,
        "padding": [0] * dimensions,
        "dilation": [1] * dimensions,
        "transposed": transposed,
        "output_padding": [0] * dimensions,
        "groups": 1,
    }
    op = cast("OpOverload[..., object]", torch.ops.aten.convolution.default)
    wide = cast(Tensor, op(x.double(), w.double(), None, **options))
    # Unit axes keep _unfused_convolution's bias independent of batch and location.
    expected = (wide + bias.double().reshape(1, 3, *((1,) * dimensions))).float()
    with host_agnostic_numerics():
        actual = cast(
            Tensor,
            op(x, w, bias=bias, **options)
            if keyword_bias
            else op(x, w, bias, **options),
        )
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("name", ["addmm", "addbmm", "baddbmm"])
@pytest.mark.parametrize("form", ["functional", "inplace", "out"])
@pytest.mark.parametrize(("alpha", "beta"), [(1, 1), (0.3, 0.7), (1, 0)])
def test_affine_matmul_adds_bias_after_accumulating_products(
    name: str,
    form: str,
    alpha: float,
    beta: float,
) -> None:
    """Affine GEMMs separate product, scaling, and bias without losing aliases."""
    a = torch.tensor([2.0**40, 1, -(2.0**40), 1, 2]).repeat(2, 1)
    b = torch.ones(5, 3)
    b[[0, 2]] *= 2.0**40
    bias = torch.arange(3, dtype=torch.float32).expand(2, 3).clone()
    if name != "addmm":
        a, b = a.repeat(4, 1, 1), b.repeat(4, 1, 1)
    if name == "baddbmm":
        bias = bias.repeat(4, 1, 1)
    product = a.double() @ b.double()
    if name == "addbmm":
        product = product.sum(0)
    if beta == 0:
        bias.fill_(float("nan"))
    expected = (
        alpha * product if beta == 0 else alpha * product + beta * bias.double()
    ).float()
    destination = torch.empty_like(bias)
    with host_agnostic_numerics():
        if form == "inplace":
            actual = cast(
                Tensor,
                getattr(bias, name + "_")(a, b, alpha=alpha, beta=beta),
            )
            assert actual is bias
        elif form == "out":
            actual = cast(
                Tensor,
                getattr(torch, name)(
                    bias,
                    a,
                    b,
                    alpha=alpha,
                    beta=beta,
                    out=destination,
                ),
            )
            assert actual is destination
        else:
            actual = cast(
                Tensor,
                getattr(torch, name)(bias, a, b, alpha=alpha, beta=beta),
            )
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "op",
    [
        torch.ops.aten._log_softmax_backward_data.default,
        torch.ops.aten._softmax_backward_data.default,
    ],
    ids=["log_softmax", "softmax"],
)
def test_softmax_backward_rounds_like_separate_ops(
    op: OpOverload[..., object],
) -> None:
    """Softmax backwards run as separate correctly-rounded ops under the harness.

    Their float64 kernels fuse ``grad - out * sum(grad)`` differently per vector
    ISA. A gradient that nearly cancels the softmax term exposes the gap: the
    native kernel and the decomposition then disagree in float32 on every ISA.
    """
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(64, 4096, generator=generator, dtype=torch.float64)
    noise = torch.randn(64, 4096, generator=generator, dtype=torch.float64)
    probabilities = logits.softmax(-1)
    if "log" in str(op):
        out, grad = logits.log_softmax(-1), probabilities + 1e-12 * noise
    else:
        out, grad = probabilities, 1 + 1e-12 * noise
    with host_agnostic_numerics():
        got = op(grad.float(), out.float(), -1, torch.float32)
    want = decomposition_table[op](
        grad.float().double(),
        out.float().double(),
        -1,
        torch.float64,
    )
    assert isinstance(got, Tensor)
    assert isinstance(want, Tensor)
    assert torch.equal(got, want.float())


def add_in_place(m: Tensor, x: Tensor) -> Tensor:
    return m.clone().add_(x, alpha=0.1)


def add_scaled(m: Tensor, x: Tensor) -> Tensor:
    return torch.add(m, x, alpha=0.1)


def sub_scaled(m: Tensor, x: Tensor) -> Tensor:
    return torch.sub(m, x, alpha=0.1)


@pytest.mark.parametrize(
    ("op", "sign"),
    [(add_in_place, 1), (add_scaled, 1), (sub_scaled, -1)],
)
def test_scaled_add_is_upcast(
    op: Callable[[Tensor, Tensor], Tensor],
    sign: int,
) -> None:
    """``add``/``sub`` with ``alpha`` round once in float64, not in a fused kernel.

    Vectorized CPU kernels fuse ``a + alpha * b`` into one FMA rounding and the
    scalar kernel rounds twice, so leaving it native mints a golden that only
    replays on hosts with the same vector ISA.
    """
    m, x, _ = _fused_operands()
    with host_agnostic_numerics():
        got = op(m, x)
    assert torch.equal(got, (m.double() + sign * (0.1 * x.double())).float())


def test_known_host_dependent_ops_are_not_allowlisted() -> None:
    """Multi-arg host/order-sensitive ops stay off the allowlist (so are upcast).

    The single-tensor divergence scan cannot reach these, so they are guarded by
    name here. Three classes, each with a numeric witness where one applies:

    - Fused multiply-add (``addcmul_``/``addcdiv_``): ``a + b*c`` rounds
      differently than the float64 path -> vector-width dependent.
    - Matmul (``mm``/``bmm``/``addmm``/...): kernel and reduction order are
      microarchitecture/thread dependent (the reason matmul is upcast, not
      CBWR-pinned).
    - Accumulating index ops (``scatter_add_``/``index_add_``/
      ``embedding_dense_backward``, and ``index_put_`` whose ``accumulate=True``
      overload is additive): sum float32 in a host-dependent order.
    """
    must_upcast = (
        "addcmul_",
        "addcdiv_",
        "mm",
        "bmm",
        "addmm",
        "baddbmm",
        "addbmm",
        "matmul",
        "scatter_add_",
        "index_add_",
        "embedding_dense_backward",
        "index_put_",
    )
    for name in must_upcast:
        assert name not in _EXACT_F32_OPS

    # Constructed inputs, not random ones: a witness on a random draw asserts a
    # probability, and the scatter one below coincided for 91/2000 seeds. These
    # values make the divergence a property of IEEE-754, true on every host.
    #
    # The float32 kernel scales by ``float32(0.7)`` and the float64 one by
    # ``0.7``; with these operands that gap flips the float32 round under both
    # the scalar kernel (two roundings) and the vectorized one (one fused FMA).
    # Measured on ATEN_CPU_CAPABILITY=default, avx2, and avx512.
    a = torch.full((64,), 1.0 + 1731 * 2.0**-14, dtype=torch.float32)
    b = torch.full((64,), 1.0 + 2395 * 2.0**-12, dtype=torch.float32)
    c = torch.full((64,), 1.0 + 3316 * 2.0**-14, dtype=torch.float32)
    fma32 = a.clone().addcmul_(b, c, value=0.7)
    fma64 = a.double().addcmul_(b.double(), c.double(), value=0.7).float()
    assert not torch.equal(fma32, fma64)

    # Mixed magnitudes: adding 1e-8 to 1.0 in float32 is absorbed on every
    # host, while the float64 sum keeps it, so the two paths cannot coincide.
    idx = torch.zeros(1024, dtype=torch.long)
    src = torch.full((1024,), 1e-8, dtype=torch.float32)
    src[0] = 1.0
    acc32 = torch.zeros(1, dtype=torch.float32).scatter_add_(0, idx, src)
    acc64 = (
        torch.zeros(1, dtype=torch.float64).scatter_add_(0, idx, src.double()).float()
    )
    assert not torch.equal(acc32, acc64)


def draw_randn(dtype: torch.dtype | None = None) -> Tensor:
    return torch.randn(18, dtype=dtype)


def draw_normal(dtype: torch.dtype | None = None) -> Tensor:
    return torch.normal(0.5, std=2.0, size=(18,), dtype=dtype)


@pytest.mark.parametrize(
    "sample",
    [pytest.param(draw_randn, id="randn"), pytest.param(draw_normal, id="normal")],
)
def test_host_agnostic_draws_normal_factories_in_float64(
    sample: Callable[[torch.dtype | None], Tensor],
) -> None:
    """A tensorless normal sampler draws in float64 and rounds to its dtype.

    ``randn`` has no float32 argument, so the input-driven upcast never fires
    and the vectorized float32 fill (16+ elements) runs natively -- SLEEF under
    AVX2, libm elsewhere -- and lands on host-specific bits. The witness at the
    end proves the two paths differ, so the assertion is not vacuous.
    """
    torch.manual_seed(0)
    with host_agnostic_numerics():
        drawn = sample(None)
    torch.manual_seed(0)
    expected = sample(torch.float64).float()
    assert drawn.dtype == torch.float32
    assert torch.equal(drawn, expected)

    torch.manual_seed(0)
    with host_agnostic_numerics():
        drawn_half = sample(torch.bfloat16)
    torch.manual_seed(0)
    assert torch.equal(drawn_half, sample(torch.float64).bfloat16())

    torch.manual_seed(0)
    with host_agnostic_numerics():
        drawn_wide = sample(torch.float64)
    torch.manual_seed(0)
    assert torch.equal(drawn_wide, sample(torch.float64))

    torch.manual_seed(0)
    assert not torch.equal(sample(None), expected)


def test_host_agnostic_keeps_uniform_factories_native() -> None:
    """``rand`` stays float32: a uniform draw is an exact integer scaling.

    Routing it through float64 would only change generator consumption and
    invalidate every golden whose runner draws uniform inputs, for no gain in
    portability.
    """
    torch.manual_seed(0)
    with host_agnostic_numerics():
        drawn = torch.rand(18)
    torch.manual_seed(0)
    assert torch.equal(drawn, torch.rand(18))


def test_host_agnostic_numerics_loads_float32_checkpoints(tmp_path: Path) -> None:
    """``torch.load`` rebinds float32 storage via ``set_``; it must pass through.

    A runner that restores a checkpoint mid-golden (an evaluation reading a
    trained model) otherwise dies: an upcast ``set_`` binds a float64 view to
    the file's float32 bytes and the storage refuses to resize.
    """
    saved = torch.linspace(-1.0, 1.0, 9)
    path = tmp_path / "state.pt"
    torch.save({"weight": saved}, path)
    with host_agnostic_numerics():
        loaded = from_plain(
            _torch_load(path, weights_only=True),
            dict[str, Tensor],
        )
    assert torch.equal(loaded["weight"], saved)


def test_host_agnostic_upcasts_inplace_op_and_mutates_original() -> None:
    """An upcast in-place op writes the float64 result back into the original.

    A non-allowlisted in-place op (e.g. ``addcmul_``) runs on a float64 copy;
    without write-back the caller's float32 tensor would stay stale and the
    returned tensor would be a float64 temporary. The harness must narrow the
    result back into the original and return it, preserving in-place semantics.
    """
    v = torch.full((4,), 0.5, dtype=torch.float32)
    g = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    with host_agnostic_numerics():
        ret = v.addcmul_(g, g, value=0.05)
    expected = (
        torch.full((4,), 0.5, dtype=torch.float64)
        .addcmul_(g.double(), g.double(), value=0.05)
        .float()
    )
    assert ret is v
    assert v.dtype == torch.float32
    assert torch.equal(v, expected)


def test_host_agnostic_upcasts_out_kwarg_and_writes_destination() -> None:
    """An upcast ``out=`` op writes the float64 result into the destination.

    ``torch.sin(x, out=y)`` runs on a float64 copy; the harness must narrow the
    result into the caller's float32 ``y`` and return it, not leave ``y`` stale.
    """
    x = torch.tensor([0.0, 1.0], dtype=torch.float32)
    y = torch.full_like(x, -9.0)
    with host_agnostic_numerics():
        ret = torch.sin(x, out=y)
    expected = torch.sin(x.double()).float()
    assert ret is y
    assert torch.equal(y, expected)


def test_host_agnostic_out_kwarg_preserves_native_resize_semantics() -> None:
    x = torch.tensor([0.0, 1.0], dtype=torch.float32)
    output = torch.empty(1, dtype=torch.float32)
    expected = torch.sin(x.double()).float()

    with pytest.warns(UserWarning, match="resized"), host_agnostic_numerics():
        returned = torch.sin(x, out=output)

    assert returned is output
    assert output.shape == x.shape
    assert torch.equal(output, expected)


def test_host_agnostic_numerics_preserves_mixed_float64_promotion() -> None:
    wide = torch.tensor([1.25], dtype=torch.float64)
    narrow = torch.tensor([0.5], dtype=torch.float32)
    expected = torch.atan2(wide, narrow)

    with host_agnostic_numerics():
        actual = torch.atan2(wide, narrow)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_host_agnostic_numerics_preserves_explicit_output_dtype() -> None:
    value = torch.tensor([1.25, 0.5], dtype=torch.float32)
    expected = torch.sum(value, dtype=torch.float64)

    with host_agnostic_numerics():
        actual = torch.sum(value, dtype=torch.float64)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


# A [2, 1] by [1, 2] product is the smallest one-term contraction x86's matrix
# kernel stores as the product itself; aarch64 adds it to a zero accumulator.
def test_host_agnostic_one_term_matmul_keeps_the_sign_of_a_zero_product() -> None:
    left = torch.tensor([[-1.0], [1.0]])
    right = torch.tensor([[0.0, -0.0]])

    with host_agnostic_numerics():
        product = torch.mm(left, right)

    assert torch.signbit(product).tolist() == [[True, False], [False, True]]


# A single output column takes x86's vector kernel, which adds to +0.0 too.
def test_host_agnostic_matmul_by_a_single_column_adds_to_positive_zero() -> None:
    left = torch.tensor([[-1.0], [1.0]])
    right = torch.tensor([[0.0]])

    with host_agnostic_numerics():
        product = torch.mm(left, right)

    assert torch.signbit(product).tolist() == [[False], [False]]


def test_host_agnostic_numerics_preserves_mixed_foreach_dtypes() -> None:
    inputs = [
        torch.tensor([0.5], dtype=torch.float16),
        torch.tensor([0.5], dtype=torch.float32),
    ]
    expected = torch._foreach_sin(inputs)

    with host_agnostic_numerics():
        actual = torch._foreach_sin(inputs)

    assert [value.dtype for value in actual] == [value.dtype for value in expected]
    assert all(
        torch.equal(value, reference)
        for value, reference in zip(actual, expected, strict=True)
    )


def test_host_agnostic_numerics_upcasts_foreach_norm() -> None:
    """``_foreach_*`` ops (list[Tensor] args) are upcast, not silently skipped.

    ``clip_grad_norm_`` routes through ``_foreach_norm``, whose argument is a
    list of tensors. A direct ``isinstance(arg, Tensor)`` guard misses it, so
    the float64 upcast must recurse into list/tuple args or the train-step CPU
    golden can still diverge across AVX widths.
    """
    x = torch.tensor([1e20, 1.0, -1e20, 3.0], dtype=torch.float32)
    expected = torch.linalg.vector_norm(x.double(), ord=2).float()
    with host_agnostic_numerics():
        actual = torch._foreach_norm([x], 2.0)[0]
    assert torch.equal(actual, expected)


def _collective_worker(root: Path, mesh: DeviceMesh) -> None:
    rank = mesh.get_rank()
    try:
        with host_agnostic_numerics():
            broadcast = torch.tensor([float(rank + 1)])
            dist.broadcast(broadcast, src=0)
            summed = torch.tensor([float(rank + 1)])
            dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        torch.save(
            {"broadcast": broadcast, "allreduce": summed},
            root / f"record_{rank}.pt",
        )
    except (AssertionError, RuntimeError, ValueError, TypeError, KeyError):
        (root / f"record_{rank}.txt").write_text(traceback.format_exc())


@pytest.mark.compute_distributed
def test_host_agnostic_numerics_preserves_collectives(
    tmp_path: Path,
    warm_pools: WarmPoolGetter,
) -> None:
    """``host_agnostic_numerics`` must not turn a collective into a no-op."""
    warm_pools({"dp": 2})(partial(_collective_worker, tmp_path))
    for rank in range(2):
        failure = tmp_path / f"record_{rank}.txt"
        assert not failure.is_file(), failure.read_text()
        record = cast(
            "dict[str, Tensor]",
            _torch_load(tmp_path / f"record_{rank}.pt", weights_only=True),
        )
        assert record["broadcast"].item() == 1.0, f"rank{rank} broadcast"
        assert record["allreduce"].item() == 3.0, f"rank{rank} allreduce"


def test_host_agnostic_foreach_inplace_writes_back_list_targets() -> None:
    """A void ``_foreach_*_`` in-place op mutates every original list element.

    ``_foreach_mul_`` writes its ``Tensor[]`` ``self`` in place and returns
    ``()`` (void at dispatch). The harness ran it on float64 copies, so each
    original float32 tensor must receive the narrowed result, and the dispatch
    must still see a ``None`` return. Regression guard for the optimizer/clip
    foreach path that train-step goldens exercise.
    """
    xs = [torch.full((3,), 0.5, dtype=torch.float32) for _ in range(2)]
    ys = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32) for _ in range(2)]
    expected = [
        (x.double() * y.double()).float()
        for x, y in zip([torch.full((3,), 0.5) for _ in range(2)], ys, strict=True)
    ]
    with host_agnostic_numerics():
        torch._foreach_mul_(xs, ys)
    for got, exp in zip(xs, expected, strict=True):
        assert torch.equal(got, exp)


def test_host_agnostic_multi_output_write_op_keeps_all_returns() -> None:
    """A write op returning fresh non-write tensors keeps them, not just writes.

    ``_native_batch_norm_legit`` (training) writes ``running_mean``/
    ``running_var`` in place but returns a 3-tuple ``(output, save_mean,
    save_invstd)`` -- none of which is a write target. The write-back must return
    the full 3-tuple of computed outputs, not a tuple of the 2 write
    originals. Asserts the returned ``output`` has the input's
    shape and is the normalized result, proving non-write returns survive.
    """
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    running_mean = torch.zeros(3, dtype=torch.float32)
    running_var = torch.ones(3, dtype=torch.float32)
    weight = torch.ones(3, dtype=torch.float32)
    bias = torch.zeros(3, dtype=torch.float32)
    with host_agnostic_numerics():
        out, save_mean, save_invstd = cast(
            tuple[Tensor, Tensor, Tensor],
            torch.ops.aten._native_batch_norm_legit(
                x,
                weight,
                bias,
                running_mean,
                running_var,
                True,
                0.1,
                1e-5,
            ),
        )
    expected = cast(
        tuple[Tensor, Tensor, Tensor],
        torch.ops.aten._native_batch_norm_legit(
            x.double(),
            weight.double(),
            bias.double(),
            running_mean.double(),
            running_var.double(),
            True,
            0.1,
            1e-5,
        ),
    )[0].float()
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert torch.equal(out, expected)
    assert save_mean.shape == (3,)
    assert save_invstd.shape == (3,)


@pytest.mark.parametrize("name", ["add_", "sub_", "mul_", "div_"])
def test_inplace_arith_matches_functional_recompute(name: str) -> None:
    """In-place arithmetic is itself float64-recompute-stable, not just by proxy.

    The category guard vets ``add_`` via the functional ``add`` probe (shared
    kernel assumption). This runs the in-place op directly and proves its float32
    result equals the float64-then-downcast result, so a future torch change that
    gave the in-place kernel a different rounding path is caught rather than
    trusted.
    """
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(4096, dtype=torch.float32, generator=gen)
    b = torch.randn(4096, dtype=torch.float32, generator=gen).abs() + 1.0
    f32 = a.clone()
    getattr(f32, name)(b)
    f64 = a.double()
    getattr(f64, name)(b.double())
    assert torch.equal(f32, f64.float())


class _AtenPacket(Protocol):
    """The typed slice of an Aten packet used by the divergence scan."""

    default: Callable[[Tensor], object]


class _AtenInPlaceAdd(Protocol):
    """The ``add_`` packet's tensor overload, whose schema marks ``self`` written."""

    Tensor: OpOverload[..., object]


def _f64_output_runner(module: nn.Module, inp: Tensor) -> Tensor:
    """Return the float64 scratch, skipping the round back to float32."""
    output = cast(object, module(inp))
    assert isinstance(output, Tensor)
    return output.double()


def test_bfb_rejects_a_float64_golden_output(tmp_path: Path) -> None:
    """A runner returning float64 is refused, at mint AND at replay.

    ``host_agnostic_numerics`` computes in float64 and the round back to
    float32 is what makes a value host-independent: a float64 kernel is itself
    approximate (torch 2.11 ``sigmoid`` misses the correctly rounded float64
    answer on 1316 of 4096 inputs), and only discarding ~29 bits swamps that.
    A golden that stores the float64 scratch therefore pins itself to the host
    that minted it -- one did, off by 1 ULP between Intel and AMD -- so the
    harness refuses it rather than letting the mismatch surface on someone
    else's machine as an opaque last-bit failure.
    """
    with pytest.raises(TypeError, match="float64"):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="f64_output",
            build_module=_build_min_linear,
            build_input=_build_min_input,
            seed=0,
            run=_f64_output_runner,
        )


def test_bfb_rejects_a_float64_golden_on_replay(tmp_path: Path) -> None:
    """A runner changed to return float64 after minting is refused by cause.

    Minting is not the only entry point, so the replay path must name the
    cause too rather than failing on a digest mismatch the reader cannot
    attribute.
    """
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="f64_replay",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )
    with pytest.raises(TypeError, match="float64"):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="f64_replay",
            build_module=_build_min_linear,
            build_input=_build_min_input,
            seed=0,
            run=_f64_output_runner,
        )


def test_ulp_diff_counts_steps_across_zero() -> None:
    """Two neighbours straddling zero are 2 steps apart, not 2 billion.

    Floats are ordered by bit pattern only WITHIN a sign; the negative half is
    stored sign-magnitude, so subtracting raw patterns across zero yields
    ~2**31. That is the magnitude a catastrophic regression produces, so the
    one number meant to separate host drift from a real break reports the
    opposite of the truth exactly where the values are closest.
    """
    negative = torch.tensor([-1.4012984643248171e-45], dtype=torch.float32)
    positive = torch.tensor([1.4012984643248171e-45], dtype=torch.float32)
    assert _max_ulp_diff(negative, positive) == 2
    ordered = _ordered(torch.tensor([-1.0, 0.0, 1.0]), torch.int32)
    assert ordered[0] < ordered[1] == 0
    assert ordered[1] < ordered[2]


def test_ulp_diff_counts_steps_within_the_negative_half() -> None:
    """The fix must not break the ordinary same-sign case it already handled."""
    value = torch.tensor([-1.0], dtype=torch.float32)
    neighbour = torch.nextafter(value, torch.tensor([-2.0]))
    assert _max_ulp_diff(value, neighbour) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ulp_diff_counts_half_precision_steps(dtype: torch.dtype) -> None:
    value = torch.tensor([1.0], dtype=dtype)
    neighbour = torch.nextafter(value, torch.tensor([2.0], dtype=dtype))
    assert _max_ulp_diff(value, neighbour) == 1


def test_ulp_diff_names_nan_rather_than_counting_it() -> None:
    """A NaN mismatch reports NaN, not a bit-pattern distance.

    NaN is the most common real regression and has no meaningful ULP distance
    from a number; printing one (measured: 1077936128) reads as a huge but
    genuine drift and hides what actually happened.
    """
    nan = torch.tensor([float("nan")], dtype=torch.float32)
    one = torch.tensor([1.0], dtype=torch.float32)
    assert _max_ulp_diff(nan, one) == "nan"


def test_assert_equal_reports_an_integer_difference() -> None:
    """An integer mismatch reports its real magnitude.

    Integer arrays are goldens too -- RNG state, sampled actions, token ids --
    and a report of ``0.000e+00`` on a failing comparison reads as a passing
    one that somehow raised.
    """
    with pytest.raises(AssertionError, match=r"max_abs_diff=7"):
        _assert_equal(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([1, 9], dtype=torch.int64),
            label="output",
        )


def test_assert_equal_reports_an_unsigned_integer_difference() -> None:
    with pytest.raises(AssertionError, match=r"max_abs_diff=255"):
        _assert_equal(
            torch.tensor([0], dtype=torch.uint8),
            torch.tensor([255], dtype=torch.uint8),
            label="output",
        )


def test_assert_equal_reports_exact_int64_extreme_difference() -> None:
    with pytest.raises(
        AssertionError,
        match=r"max_abs_diff=18446744073709551615",
    ):
        _assert_equal(
            torch.tensor([torch.iinfo(torch.int64).min]),
            torch.tensor([torch.iinfo(torch.int64).max]),
            label="output",
        )


def test_assert_equal_reports_adjacent_large_uint64_difference() -> None:
    with pytest.raises(AssertionError, match=r"max_abs_diff=1"):
        _assert_equal(
            torch.tensor([2**63], dtype=torch.uint64),
            torch.tensor([2**63 + 1], dtype=torch.uint64),
            label="output",
        )


def test_bfb_compares_float_bits_not_value_equality() -> None:
    positive_zero = torch.tensor([0.0])
    negative_zero = torch.tensor([-0.0])
    nan = torch.tensor([float("nan")])

    with pytest.raises(AssertionError):
        _assert_equal(positive_zero, negative_zero, label="output")
    _assert_equal(nan, nan.clone(), label="output")
    assert not changed_state({"value": nan}, {"value": nan.clone()})


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_bfb_rejects_every_non_float32_golden_output(dtype: torch.dtype) -> None:
    """Every narrow-float comparand is refused, not float64 alone.

    ``_is_narrow_float`` treats bfloat16 and float16 as compute dtypes the
    harness upcasts, so a runner returning either skipped the same round back
    to float32 and stores this host's libm error just as a float64 one does.
    """
    with pytest.raises(TypeError, match="float32"):
        _assert_portable_output_dtype(torch.zeros(2, dtype=dtype))


def test_bfb_accepts_a_float32_golden_output() -> None:
    """The dtype the whole mechanism produces is the one that passes."""
    _assert_portable_output_dtype(torch.zeros(2, dtype=torch.float32))


def test_bfb_rejects_complex_golden_output() -> None:
    with pytest.raises(TypeError, match="complex64"):
        _assert_portable_output_dtype(torch.zeros(2, dtype=torch.complex64))


def test_bfb_accepts_an_integer_golden_output() -> None:
    """An integer comparand carries no rounding, so it needs no narrowing."""
    _assert_portable_output_dtype(torch.zeros(2, dtype=torch.int64))


@pytest.mark.parametrize("name", ["addmm", "addbmm", "baddbmm", "addmv"])
def test_host_agnostic_keeps_integer_affine_matmul_native(name: str) -> None:
    bias = torch.ones(2, 3, dtype=torch.int64)
    left = torch.ones(2, 5, dtype=torch.int64)
    right = torch.ones(5, 3, dtype=torch.int64)
    if name in {"addbmm", "baddbmm"}:
        left, right = left.expand(4, -1, -1), right.expand(4, -1, -1)
    if name == "baddbmm":
        bias = bias.expand(4, -1, -1)
    if name == "addmv":
        right, bias = torch.ones(5, dtype=torch.int64), torch.ones(2, dtype=torch.int64)
    op = cast("Callable[..., Tensor]", getattr(torch, name))
    expected = op(bias, left, right, alpha=0.5)
    with host_agnostic_numerics():
        actual = op(bias, left, right, alpha=0.5)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("name", ["addmm", "addbmm", "baddbmm", "addmv"])
@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("form", ["functional", "inplace", "out"])
def test_host_agnostic_affine_rejects_bias_that_expands_the_output(
    name: str,
    beta: int,
    form: str,
) -> None:
    # A unit row exposes _run_unfused accepting bias that enlarges its product.
    left, right, bias = torch.ones(1, 4), torch.ones(4, 3), torch.zeros(2, 3)
    if name in {"addbmm", "baddbmm"}:
        left, right = left.unsqueeze(0), right.unsqueeze(0)
    if name == "baddbmm":
        bias = bias.unsqueeze(0)
    if name == "addmv":
        right, bias = torch.ones(4), torch.zeros(2)
    if form == "inplace":
        op = partial(
            cast("Callable[..., Tensor]", getattr(bias, name + "_")),
            left,
            right,
            beta=beta,
        )
    else:
        functional = cast("Callable[..., Tensor]", getattr(torch, name))
        op = (
            partial(functional, bias, left, right, beta=beta, out=torch.empty(0))
            if form == "out"
            else partial(functional, bias, left, right, beta=beta)
        )
    with pytest.raises(RuntimeError):
        op()
    with pytest.raises(RuntimeError), host_agnostic_numerics():
        op()


@pytest.mark.parametrize("name", ["addmm", "addbmm", "baddbmm"])
@pytest.mark.parametrize("integer", ["bias", "matrices"])
def test_host_agnostic_affine_rejects_mixed_operand_dtypes(
    name: str,
    integer: str,
) -> None:
    # The unfused path never reaches the native dtype check, so widening the
    # float operands would silently absorb integer ones torch rejects.
    bias, left, right = torch.zeros(2, 3), torch.ones(2, 4), torch.ones(4, 3)
    if integer == "bias":
        bias = bias.long()
    else:
        left, right = left.long(), right.long()
    if name in {"addbmm", "baddbmm"}:
        left, right = left.unsqueeze(0), right.unsqueeze(0)
    if name == "baddbmm":
        bias = bias.unsqueeze(0)
    op = partial(cast("Callable[..., Tensor]", getattr(torch, name)), bias, left, right)
    with pytest.raises(RuntimeError):
        op()
    with pytest.raises(RuntimeError), host_agnostic_numerics():
        op()


def _stochastic_runner(module: nn.Module, value: Tensor) -> Tensor:
    del module
    return value + torch.rand_like(value)


def _input_mutating_runner(module: nn.Module, value: Tensor) -> Tensor:
    del module
    return value.add_(1)


@pytest.mark.parametrize("runner", [_stochastic_runner, _input_mutating_runner])
def test_bfb_replays_random_and_mutated_inputs(
    tmp_path: Path,
    runner: Callable[[nn.Module, Tensor], Tensor],
) -> None:
    for check in (regenerate_golden, assert_bfb_against_golden):
        check(
            golden_dir=tmp_path,
            golden_name="input_lifecycle",
            build_module=nn.Identity,
            build_input=lambda: torch.rand(2, 3),
            run=runner,
        )
    torch.manual_seed(0)
    assert torch.equal(
        cast(Tensor, load_golden(tmp_path / "input_lifecycle.pt")["input"]),
        torch.rand(2, 3),
    )


@pytest.mark.parametrize("name", ["linspace", "logspace", "arange"])
def test_host_agnostic_widens_arithmetic_factories(name: str) -> None:
    op = cast("Callable[..., Tensor]", getattr(torch, name))
    args = (0.13, 10.17, 0.27) if name == "arange" else (-0.7, 0.9, 49)
    expected = op(*args, dtype=torch.float64).float()
    with host_agnostic_numerics():
        actual = op(*args)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_host_agnostic_preserves_scalar_tensor_promotion() -> None:
    vector = torch.tensor([0.3, 0.7], dtype=torch.float32)
    scalar = torch.tensor(0.2, dtype=torch.float64)
    expected = torch.atan2(vector, scalar)
    with host_agnostic_numerics():
        actual = torch.atan2(vector, scalar)
    assert actual.dtype == expected.dtype


def _custom_affine_kernel(value: Tensor) -> Tensor:
    return value + 1


@pytest.mark.parametrize("name", ["addbmm", "add"])
def test_host_agnostic_keeps_custom_namespace_implementations(name: str) -> None:
    with torch.library._scoped_library("bfb_contract_probe", "FRAGMENT") as library:
        library.define(f"{name}(Tensor value) -> Tensor")
        library.impl(name, _custom_affine_kernel, "CompositeExplicitAutograd")
        op = cast(
            "Callable[[Tensor], Tensor]",
            getattr(torch.ops.bfb_contract_probe, name),
        )
        value = torch.tensor([0.5, 1.5])
        assert torch.equal(op(value), value + 1)
        assert not _f32_leaking_ops(lambda: op(value))
    # Only Library._destroy drops the cached torch.ops entry; a leaked entry
    # points at a freed operator, so mutmut's in-process rerun crashed.
    assert not hasattr(torch.ops.bfb_contract_probe, name)


def test_assert_equal_reports_a_boolean_difference() -> None:
    with pytest.raises(AssertionError, match="max_abs_diff=1"):
        _assert_equal(torch.tensor([True]), torch.tensor([False]), label="output")


def test_ulp_diff_does_not_overflow_float64_distance() -> None:
    assert (
        _max_ulp_diff(
            torch.tensor([-2.0], dtype=torch.float64),
            torch.tensor([2.0], dtype=torch.float64),
        )
        == 2**63
    )


def test_portable_half_precision_preserves_precision_policy() -> None:
    original = torch.backends.mkldnn.set_flags(_fp32_precision="ieee")
    try:
        before = torch.backends.mkldnn.set_flags(_fp32_precision=None)
        with portable_half_precision():
            assert torch.backends.mkldnn.set_flags(_fp32_precision=None)[3] == before[3]
        assert torch.backends.mkldnn.set_flags(_fp32_precision=None) == before
    finally:
        torch.backends.mkldnn.set_flags(_fp32_precision=original[3])


def test_ordered_preserves_signed_zero_and_neighbors() -> None:
    values = torch.tensor([-torch.finfo(torch.float32).tiny, -0.0, 0.0, 1.0])
    ordered = _ordered(values, torch.int32)
    assert ordered[0] < ordered[1]
    assert ordered[1] == ordered[2]
    assert ordered[2] < ordered[3]


@pytest.mark.parametrize(
    ("dtype", "kind"),
    [
        (torch.float16, torch.int16),
        (torch.bfloat16, torch.int16),
        (torch.float32, torch.int32),
    ],
)
def test_ordered_widens_bit_patterns_before_reflecting(
    dtype: torch.dtype,
    kind: torch.dtype,
) -> None:
    values = torch.tensor([-1.0, -0.0, 0.0, 1.0], dtype=dtype)
    ordered = _ordered(values, kind)
    assert ordered.dtype == torch.int64
    assert ordered[0] < ordered[1] == ordered[2] < ordered[3]


def test_move_to_device_recurses_through_distinct_container_shapes() -> None:
    source = torch.arange(6).reshape(2, 3)
    moved = move_to_device(
        {"tuple": (source, 7), "list": [source + 1, "metadata"]},
        "meta",
    )
    nested = moved["tuple"]
    assert isinstance(nested, tuple)
    assert isinstance(nested[0], Tensor)
    assert nested[0].device.type == "meta"
    assert nested[1] == 7
    sequence = moved["list"]
    assert isinstance(sequence, list)
    assert isinstance(sequence[0], Tensor)
    assert sequence[0].device.type == "meta"
    assert sequence[1] == "metadata"


class _CustomGoldenInput:
    """A pickled input value that the safe tensor-only loader cannot decode."""

    def __init__(self, value: int) -> None:
        self.value = value


def test_load_golden_unpacks_state_and_post_state(tmp_path: Path) -> None:
    path = tmp_path / "packed.pt"
    state = {"weight": torch.arange(6.0).reshape(2, 3)}
    post = {"weight": torch.full((2, 3), 4.0)}
    torch.save(
        {
            "state_dict": pack(state),
            "input": torch.zeros(2, 3),
            "output": torch.ones(2, 3),
            "seed": 17,
            "post_state": pack(post),
        },
        path,
    )
    payload = load_golden(path)
    assert "post_state" in payload
    assert torch.equal(payload["state_dict"]["weight"], state["weight"])
    assert torch.equal(payload["post_state"]["weight"], post["weight"])
    assert payload["seed"] == 17


def test_load_golden_accepts_custom_input_objects(tmp_path: Path) -> None:
    path = tmp_path / "custom.pt"
    value = _CustomGoldenInput(23)
    torch.save(
        {
            "state_dict": pack({}),
            "input": value,
            "output": torch.ones(2),
            "seed": 0,
        },
        path,
    )
    payload = load_golden(path)
    stored = payload["input"]
    assert isinstance(stored, _CustomGoldenInput)
    assert stored.value == 23


def test_regenerate_failure_keeps_existing_golden(tmp_path: Path) -> None:
    class SmallModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 3))

        @override
        def forward(self, value: Tensor) -> Tensor:
            return value * self.weight

    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="stable",
        build_module=SmallModule,
        build_input=lambda: torch.ones(2, 3),
    )
    path = tmp_path / "stable.pt"
    original = path.read_bytes()
    calls = 0

    def unstable(module: nn.Module, value: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        output = cast(object, module(value))
        assert isinstance(output, Tensor)
        return output if calls == 1 else output + 1

    with pytest.raises(AssertionError):
        regenerate_golden(
            golden_dir=tmp_path,
            golden_name="stable",
            build_module=SmallModule,
            build_input=lambda: torch.ones(2, 3),
            run=unstable,
        )
    assert path.read_bytes() == original
    assert sorted(item.name for item in tmp_path.iterdir()) == ["stable.pt"]


def test_seed_bfb_sets_reproducible_cpu_generator_state() -> None:
    _seed_bfb(29)
    first = torch.rand(2, 3)
    _seed_bfb(29)
    second = torch.rand(2, 3)
    assert torch.equal(first, second)


def test_write_golden_copies_output_to_cpu_with_explicit_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = torch.tensor([1.0, 2.0])
    original_to = Tensor.to
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    prior = torch.are_deterministic_algorithms_enabled()

    def to_spy(
        tensor: Tensor,
        device: str | torch.device | None = None,
        *,
        copy: bool = False,
    ) -> Tensor:
        if tensor.untyped_storage().data_ptr() == output.untyped_storage().data_ptr():
            calls.append(((device,), {"copy": copy}))
        return original_to(tensor, device=device, copy=copy)

    def run(module: nn.Module, inp: Tensor) -> Tensor:
        del module, inp
        return output

    monkeypatch.setattr(Tensor, "to", to_spy)
    try:
        _write_golden(
            golden_path=tmp_path / "write-policy.pt",
            build_module=nn.Identity,
            build_input=lambda: torch.zeros(2),
            seed=7,
            run=run,
        )
    finally:
        torch.use_deterministic_algorithms(prior)

    assert calls == [(("cpu",), {"copy": True})]


def test_load_golden_passes_explicit_weights_only_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "load-policy.pt"
    save_golden(
        path,
        {
            "state_dict": {},
            "input": torch.zeros(2),
            "output": torch.ones(2),
            "seed": 0,
        },
    )
    original_load = _torch_load
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_spy(
        file: Path,
        *,
        map_location: str | torch.device | None = None,
        weights_only: bool,
    ) -> object:
        calls.append(((), {"weights_only": weights_only}))
        return original_load(
            file,
            map_location=map_location,
            weights_only=weights_only,
        )

    monkeypatch.setattr(torch, "load", load_spy)

    load_golden(path)

    assert calls == [((), {"weights_only": False})]


def test_stale_post_states_uses_explicit_weights_only_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "custom-input.pt"
    save_golden(
        path,
        {
            "state_dict": {},
            "input": _CustomGoldenInput(23),
            "output": torch.ones(2),
            "seed": 0,
            "post_state": {},
        },
    )
    original_load = _torch_load
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_spy(
        file: Path,
        *,
        map_location: str | torch.device | None = None,
        weights_only: bool,
    ) -> object:
        calls.append(((), {"weights_only": weights_only}))
        return original_load(
            file,
            map_location=map_location,
            weights_only=weights_only,
        )

    monkeypatch.setattr(torch, "load", load_spy)

    assert stale_post_states([path]) == []
    assert calls == [((), {"weights_only": False})] * 2


def test_seed_bfb_constructs_default_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator_type = torch.Generator
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def generator_spy(device: str | torch.device | None = None) -> torch.Generator:
        calls.append(((device,), {}))
        return generator_type(device=device)

    monkeypatch.setattr(torch, "Generator", generator_spy)
    _seed_bfb(41)
    assert calls == [((None,), {})]


def test_ordered_converts_integer_patterns_to_int64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    original_to = Tensor.to
    calls: list[tuple[object, ...]] = []

    def to_spy(
        tensor: Tensor,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if tensor.dtype is torch.int32:
            calls.append((dtype,))
        return original_to(tensor, dtype=dtype)

    monkeypatch.setattr(Tensor, "to", to_spy)
    ordered = _ordered(values, torch.int32)

    assert calls == [(torch.int64,)]
    assert ordered.dtype is torch.int64


def test_regenerate_golden_persists_requested_seed(tmp_path: Path) -> None:
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="seeded",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=29,
    )
    assert load_golden(tmp_path / "seeded.pt")["seed"] == 29


@pytest.mark.parametrize("view", [False, True])
def test_to_cpu_preserves_input_identity_and_shared_storage(view: bool) -> None:
    base = torch.arange(6.0)
    first, second = (base[:4], base[2:]) if view else (base, base)
    snapshot = cast(tuple[Tensor, Tensor], _to_cpu((first, second)))
    if not view:
        assert snapshot[0] is snapshot[1]
    snapshot[0].add_(1)
    assert torch.equal(
        snapshot[1],
        second + (torch.tensor([1, 1, 0, 0]) if view else 1),
    )
    assert torch.equal(base, torch.arange(6.0))


def test_to_cpu_stores_only_the_span_its_views_share() -> None:
    # A golden serializes whole storages, so snapshotting a slice of a large
    # tensor with its base would push the golden past the size gate.
    base = torch.arange(1000.0)
    first, second = base[10:14], base[12:16].view(torch.int32)
    snapshot = cast(tuple[Tensor, Tensor], _to_cpu((first, second)))
    storage = snapshot[0].untyped_storage()
    assert storage.nbytes() == 6 * base.element_size()
    assert snapshot[1].untyped_storage().data_ptr() == storage.data_ptr()
    assert torch.equal(snapshot[0], first)
    assert torch.equal(snapshot[1], second)


def _deterministic_identity() -> nn.Module:
    assert torch.are_deterministic_algorithms_enabled()
    return nn.Identity()


def test_bfb_builds_under_deterministic_algorithms(tmp_path: Path) -> None:
    before = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        for check in (regenerate_golden, assert_bfb_against_golden):
            check(
                golden_dir=tmp_path,
                golden_name="deterministic_build",
                build_module=_deterministic_identity,
                build_input=lambda: torch.zeros(2, 3),
            )
            assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(before)


def _build_wide_buffer() -> nn.Module:
    module = nn.Module()
    module.register_buffer("wide", torch.tensor([0.3], dtype=torch.float64))
    return module


def _mutate_wide_buffer(module: nn.Module, value: Tensor) -> Tensor:
    module.get_buffer("wide").sigmoid_()
    return value.clone()


def test_bfb_rejects_changed_float64_state(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="float64"):
        regenerate_golden(
            golden_dir=tmp_path,
            golden_name="wide_state",
            build_module=_build_wide_buffer,
            build_input=lambda: torch.zeros(2),
            run=_mutate_wide_buffer,
        )


def test_bfb_accepts_float32_precision_in_float64_state() -> None:
    _assert_portable_state_changes(
        {"scores": torch.tensor([0.0, -0.0, 0.5, 1.5], dtype=torch.float64)},
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex64, torch.complex128])
def test_bfb_rejects_unrounded_or_complex_state(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match=rf"state\[scores\] is {dtype}") as error:
        _assert_portable_state_changes({"scores": torch.tensor([0.3], dtype=dtype)})
    assert str(error.value) == (
        f"bfb golden state[scores] is {dtype}, which is not portable; "
        "narrow computed state before recording it."
    )


@pytest.mark.parametrize(
    ("live", "stored"),
    [((torch.zeros(2),), [torch.zeros(2)]), (1, True), (1, 1.0)],
)
def test_input_validation_distinguishes_container_and_scalar_types(
    live: object,
    stored: object,
) -> None:
    with pytest.raises(AssertionError, match="input"):
        _assert_same_input(live, stored, label="input")


def test_regenerate_golden_preserves_existing_regenerate_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``regenerate_golden`` restores a caller-set ``--regenerate-b4b``.

    Unconditionally clearing it would silently disable regeneration for later
    tests in a run launched with ``--regenerate-b4b``.
    """
    regenerate.override(monkeypatch, b4b=True)
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="clean",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )
    assert regenerate.b4b()


def test_input_comparison_reports_nested_paths_and_container_lengths() -> None:
    with pytest.raises(AssertionError, match=r"batch\['media'\]\[1\]:"):
        _assert_same_input(
            {"media": (torch.zeros(2), torch.ones(2))},
            {"media": (torch.zeros(2), torch.zeros(2))},
            label="batch",
        )
    with pytest.raises(AssertionError, match=r"batch: length 2 vs 1"):
        _assert_same_input([1, 2], [1], label="batch")
    with pytest.raises(AssertionError, match=r"batch: keys differ"):
        _assert_same_input({"left": 1}, {"right": 1}, label="batch")


def test_assert_equal_handles_tensor_and_non_tensor_mismatches() -> None:
    with pytest.raises(AssertionError, match="non-tensor mismatch"):
        _assert_equal(torch.tensor(1.0), 0, label="output")
    with pytest.raises(AssertionError, match=r"output: .*max_abs_diff=2.000e\+00"):
        _assert_equal(
            torch.tensor([0.0, 1.0]),
            torch.tensor([2.0, 1.0]),
            label="output",
        )
    with pytest.raises(AssertionError, match=r"max_abs_diff=3.000e\+00"):
        _assert_equal(torch.tensor([-1.0]), torch.tensor([2.0]), label="output")
    with pytest.raises(AssertionError, match=r"output: .*max_abs_diff=2"):
        _assert_equal(torch.tensor([0, 1]), torch.tensor([2, 1]), label="output")


def test_first_tensor_uses_exact_type_errors() -> None:
    with pytest.raises(TypeError, match=r"^module result must be a Tensor$"):
        first_tensor(object())
    with pytest.raises(TypeError, match=r"^module result must start with a Tensor$"):
        first_tensor([None])


def test_byte_span_tracks_strides_offsets_and_empty_tensors() -> None:
    base = torch.arange(20.0).reshape(4, 5)
    assert _byte_span(base[1:3, 1:4]) == (24, 56)
    assert _byte_span(torch.empty((0, 3), dtype=torch.float32)) == (0, 0)
    assert _byte_span(torch.empty_strided((2, 3), (5, 1))) == (0, 32)


def test_compact_copies_preserves_shared_views_and_offsets() -> None:
    base = torch.arange(12.0)
    left, right = base[2:8:2], base[4:10:2]
    copied = _to_cpu((left, right))
    assert isinstance(copied, tuple)
    copied_left, copied_right = cast(tuple[Tensor, Tensor], copied)
    assert (
        copied_left.untyped_storage().data_ptr()
        == copied_right.untyped_storage().data_ptr()
    )
    assert copied_right.storage_offset() - copied_left.storage_offset() == 2
    assert copied_left.stride() == left.stride()
    assert copied_right.stride() == right.stride()
    assert torch.equal(copied_left, left)
    assert torch.equal(copied_right, right)
    copied_left[1] = -1
    assert copied_right[0] == -1
    assert base[5] == 5


def test_cpu_state_dict_is_an_independent_detached_snapshot() -> None:
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    snapshot = _cpu_state_dict({"weight": value})["weight"]
    assert snapshot.device.type == "cpu"
    assert not snapshot.requires_grad
    assert snapshot.data_ptr() != value.data_ptr()
    with torch.no_grad():
        value.add_(1)
    assert torch.equal(snapshot, torch.tensor([1.0, 2.0]))


def test_copy_back_recurses_and_ignores_non_narrow_targets() -> None:
    original = [torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.int64)]
    computed = [torch.tensor([1.0, 2.0], dtype=torch.float64), torch.ones(2)]
    _copy_back(original, computed)
    assert torch.equal(original[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(original[1], torch.zeros(2, dtype=torch.int64))


def test_stale_post_states_skips_non_golden_and_scans_later_entries(
    tmp_path: Path,
) -> None:
    not_golden = tmp_path / "not-golden.pt"
    torch.save({"unrelated": torch.ones(2)}, not_golden)
    incomplete = tmp_path / "incomplete.pt"
    torch.save({"state_dict": {}}, incomplete)
    non_mapping = tmp_path / "non-mapping.pt"
    torch.save([torch.ones(2)], non_mapping)
    custom_input = tmp_path / "custom-input.pt"
    save_golden(
        custom_input,
        {
            "state_dict": {},
            "input": _CustomGoldenInput(5),
            "output": torch.zeros(2),
            "seed": 0,
            "post_state": {},
        },
    )
    stale = tmp_path / "stale.pt"
    save_golden(
        stale,
        {
            "state_dict": {"weight": torch.zeros(2)},
            "input": torch.zeros(2),
            "output": torch.zeros(2),
            "seed": 0,
            "post_state": {"weight": torch.zeros(2)},
        },
    )
    assert stale_post_states(
        [not_golden, incomplete, non_mapping, custom_input, stale],
    ) == [stale]


def test_assert_equal_reports_exact_mismatch_messages() -> None:
    with pytest.raises(AssertionError, match=r"^output: dtype mismatch") as error:
        _assert_equal(
            torch.zeros(2),
            torch.zeros(2, dtype=torch.float64),
            label="output",
        )
    assert str(error.value) == "output: dtype mismatch torch.float32 vs torch.float64"
    with pytest.raises(AssertionError, match=r"^output: shape mismatch"):
        _assert_equal(torch.zeros(2), torch.zeros(3), label="output")


def test_max_ulp_diff_handles_empty_int64_and_unsupported_dtype() -> None:
    empty = torch.empty(0, dtype=torch.float64)
    assert _max_ulp_diff(empty, empty.clone()) == 0
    assert (
        _max_ulp_diff(
            torch.ones(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )
        == "n/a"
    )
    with pytest.raises(ValueError, match=r"shape mismatch \(2,\) vs \(1,\)"):
        _max_ulp_diff(
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )


def test_module_device_prefers_parameter_then_buffer() -> None:
    module = nn.Module()
    module.register_buffer("buffer", torch.empty(1, device="meta"))
    assert _module_device(module) == "meta"
    module.register_parameter("parameter", nn.Parameter(torch.empty(1)))
    assert _module_device(module) == "cpu"


def test_move_to_device_applies_destination_at_every_nesting_level() -> None:
    moved = move_to_device(
        {"tuple": (torch.ones(2),), "list": [torch.zeros(2)]},
        "meta",
    )
    assert moved["tuple"][0].device.type == "meta"
    assert moved["list"][0].device.type == "meta"


def test_default_runner_reports_exact_non_tensor_error(tmp_path: Path) -> None:
    class NoTensor(nn.Module):
        @override
        def forward(self, value: Tensor) -> object:
            return [value]

    with pytest.raises(
        TypeError,
        match=r"^The default runner requires a module that returns a Tensor\.$",
    ):
        regenerate_golden(
            golden_dir=tmp_path,
            golden_name="non_tensor",
            build_module=NoTensor,
            build_input=_build_min_input,
        )


@pytest.mark.parametrize(
    ("dtype", "message"),
    [
        (
            torch.complex64,
            "".join(
                (
                    "bfb golden output is torch.complex64, which is not supported; ",
                    "return a float32 or integer tensor.",
                ),
            ),
        ),
        (
            torch.float16,
            "".join(
                (
                    "bfb golden output is torch.float16, which is not portable across ",
                    "hosts; it must be float32. host_agnostic_numerics computes in ",
                    "float64 and the ROUND BACK to float32 is what makes the result ",
                    "host-independent; returning the unrounded value stores this ",
                    "host's libm error. Narrow in the runner: `return value.float()`.",
                ),
            ),
        ),
        (
            torch.bfloat16,
            "".join(
                (
                    "bfb golden output is torch.bfloat16, which is not portable across ",
                    "hosts; it must be float32. host_agnostic_numerics computes in ",
                    "float64 and the ROUND BACK to float32 is what makes the result ",
                    "host-independent; returning the unrounded value stores this ",
                    "host's libm error. Narrow in the runner: `return value.float()`.",
                ),
            ),
        ),
        (
            torch.float64,
            "".join(
                (
                    "bfb golden output is torch.float64, which is not portable across ",
                    "hosts; it must be float32. host_agnostic_numerics computes in ",
                    "float64 and the ROUND BACK to float32 is what makes the result ",
                    "host-independent; returning the unrounded value stores this ",
                    "host's libm error. Narrow in the runner: `return value.float()`.",
                ),
            ),
        ),
    ],
)
def test_output_dtype_diagnostic(
    dtype: torch.dtype,
    message: str,
) -> None:
    with pytest.raises(TypeError) as error:
        _assert_portable_output_dtype(torch.zeros(2, dtype=dtype))
    assert str(error.value) == message


def test_compact_copies_preserves_shared_mixed_dtype_views_and_contents() -> None:
    base = torch.arange(12, dtype=torch.int32)
    first = base[2:8]
    second = base[4:10].view(torch.int16)
    copied = _compact_copies([first, second])
    got_first, got_second = copied[id(first)], copied[id(second)]

    assert got_first.device.type == got_second.device.type == "cpu"
    assert got_first.dtype == first.dtype
    assert got_second.dtype == second.dtype
    assert got_first.shape == first.shape
    assert got_second.shape == second.shape
    assert got_first.stride() == first.stride()
    assert got_second.stride() == second.stride()
    assert torch.equal(got_first, first)
    assert torch.equal(got_second, second)
    assert (
        got_first.untyped_storage().data_ptr()
        == got_second.untyped_storage().data_ptr()
    )
    assert got_second.storage_offset() - 2 * got_first.storage_offset() == 4


def test_compact_copies_resolves_lazy_conjugation() -> None:
    base = torch.tensor([1 + 2j, 3 + 4j, 5 + 6j])
    conjugate = base.conj()
    copied = _compact_copies([conjugate])[id(conjugate)]

    assert torch.equal(copied, conjugate)
    assert not copied.is_conj()
    assert copied.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()


def test_randomize_parameters_pins_seeded_draw_order_and_std() -> None:
    module = nn.Module()
    module.register_parameter("first", nn.Parameter(torch.zeros(2, 3)))
    module.register_parameter("second", nn.Parameter(torch.zeros(3)))
    module.register_buffer("cache", torch.tensor([7.0, 8.0]))
    generator = torch.Generator(device="cpu").manual_seed(41)
    expected_first = torch.randn((2, 3), generator=generator) * 0.25
    expected_second = torch.randn((3,), generator=generator) * 0.25

    randomize_parameters(module, seed=41, std=0.25)

    first = cast(Tensor, module.first)
    second = cast(Tensor, module.second)
    cache = cast(Tensor, module.cache)
    assert torch.equal(first, expected_first)
    assert torch.equal(second, expected_second)
    assert torch.equal(cache, torch.tensor([7.0, 8.0]))


def test_randomize_parameters_uses_explicit_float32_sampling_dtype() -> None:
    module = nn.Linear(3, 2)
    generator = torch.Generator(device="cpu").manual_seed(53)
    expected_weight = torch.randn((2, 3), generator=generator, dtype=torch.float32)
    expected_bias = torch.randn((2,), generator=generator, dtype=torch.float32)
    prior = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        randomize_parameters(module, seed=53)
    finally:
        torch.set_default_dtype(prior)

    assert torch.equal(module.weight, expected_weight)
    assert module.bias is not None
    assert torch.equal(module.bias, expected_bias)


def test_resolve_output_keeps_alias_and_downcasts_fresh_results() -> None:
    original = torch.tensor([1.0, 2.0], dtype=torch.float32)
    computed = original.double()
    fresh = torch.tensor([3.0, 4.0], dtype=torch.float64)
    writes: list[tuple[object, object]] = [(computed, original)]

    restored = _resolve_output(computed, writes, torch.bfloat16)
    narrowed = _resolve_output(fresh, writes, torch.bfloat16)
    unchanged = _resolve_output("metadata", writes, torch.float16)

    assert restored is original
    assert isinstance(narrowed, Tensor)
    assert narrowed.dtype == torch.bfloat16
    assert torch.equal(narrowed, fresh.to(torch.bfloat16))
    assert unchanged == "metadata"


def test_scales_operand_reads_default_and_alpha_forms() -> None:
    add = torch.ops.aten.add.Tensor
    left, right = torch.ones(2), torch.ones(2)
    assert not _scales_operand(add, (left, right), {})
    assert not _scales_operand(add, (left, right, 1), {})
    assert _scales_operand(add, (left, right, 0.25), {})
    assert _scales_operand(add, (left, right), {"alpha": 0.25})


def test_scales_or_fuses_distinguishes_add_sub_and_decomposed_ops() -> None:
    left, right = torch.ones(2), torch.ones(2)
    assert not _scales_or_fuses(torch.ops.aten.add.Tensor, (left, right), {})
    assert _scales_or_fuses(torch.ops.aten.add_.Tensor, (left, right, 0.25), {})
    assert not _scales_or_fuses(torch.ops.aten.sub.Tensor, (left, right), {"alpha": 1})
    assert _scales_or_fuses(torch.ops.aten.sub.Tensor, (left, right), {"alpha": 0.25})
    assert _scales_or_fuses(torch.ops.aten.lerp.Tensor, (left, right, 0.25), {})


def test_scales_or_fuses_only_strips_trailing_underscores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_name(func: OpOverload[..., object]) -> str:
        del func
        return "addX"

    op = torch.ops.aten.add.Tensor
    monkeypatch.setattr("priml.testing.bfb._op_name", fake_name)
    monkeypatch.delitem(decomposition_table, op)
    operands = (torch.ones(2), torch.ones(2), 0.25)
    assert not _scales_or_fuses(op, operands, {})


def test_scales_or_fuses_does_not_strip_non_underscore_suffixes() -> None:
    class NamedOp:
        _schema = SimpleNamespace(arguments=[SimpleNamespace(name="alpha")])

        def name(self) -> str:
            return "custom::addX"

    op = cast("OpOverload[..., object]", NamedOp())
    assert not _scales_or_fuses(op, (), {"alpha": 0.25})


def test_assert_equal_moves_cross_device_values_before_comparing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = _OtherDevice(torch.tensor([1.0, 2.0]))
    local = torch.tensor([1.0, 3.0])
    original_cpu = Tensor.cpu
    original_equal = tensor_bits_equal
    compared_devices: list[tuple[torch.device, torch.device]] = []

    def cpu_spy(tensor: Tensor) -> Tensor:
        if tensor is other:
            tensor = tensor.as_subclass(Tensor)
        return original_cpu(tensor)

    def compare_spy(left: Tensor, right: Tensor) -> bool:
        compared_devices.append((left.device, right.device))
        return original_equal(left, right)

    monkeypatch.setattr(Tensor, "cpu", cpu_spy)
    monkeypatch.setattr(bfb, "tensor_bits_equal", compare_spy)
    with pytest.raises(AssertionError, match=r"max_abs_diff=1.000e\+00"):
        _assert_equal(other, local, label="output")
    assert compared_devices == [(torch.device("cpu"), torch.device("cpu"))]


def test_downcast_result_scans_sequences_after_non_tensor_input() -> None:
    wide = torch.ones(2, dtype=torch.float64)
    half = torch.ones(2, dtype=torch.float16)
    integer = torch.ones(2, dtype=torch.int64)
    result = _downcast_result([wide], (integer, [half]), {}, torch.float32)
    assert isinstance(result, list)
    assert isinstance(result[0], Tensor)
    assert result[0].dtype == torch.float16


def test_scales_or_fuses_strips_whitespace_from_names() -> None:
    class NamedOp:
        _schema = SimpleNamespace(arguments=[SimpleNamespace(name="alpha")])

        def name(self) -> str:
            return "custom::add "

    op = cast("OpOverload[..., object]", NamedOp())
    assert not _scales_or_fuses(op, (), {"alpha": 0.25})


def test_scales_or_fuses_does_not_strip_prefix_underscores() -> None:
    class NamedOp:
        _schema = SimpleNamespace(arguments=[SimpleNamespace(name="alpha")])

        def name(self) -> str:
            return "custom::_add"

    op = cast("OpOverload[..., object]", NamedOp())
    assert not _scales_or_fuses(op, (), {"alpha": 0.25})


def test_unfused_convolution_broadcasts_bias_without_batch_axis() -> None:
    conv = torch.ops.aten.convolution.default
    # convolution's 1-D image/weight contract fixes these channel axes.
    image = torch.arange(18.0).reshape(3, 2, 3)
    weight = torch.ones(3, 2, 2)
    bias = torch.arange(3.0)
    options: dict[str, object] = {
        "stride": (1,),
        "padding": (0,),
        "dilation": (1,),
        "transposed": False,
        "output_padding": (0,),
        "groups": 1,
    }
    result = _unfused_convolution(
        conv,
        (image, weight),
        {**options, "bias": bias},
        bias=bias,
    )
    # conv1d bias broadcasts over the production batch and sequence axes.
    expected = torch.nn.functional.conv1d(image, weight, None) + bias.reshape(1, 3, 1)
    assert torch.equal(result, expected)


def test_write_back_returns_scalar_result_after_copying_write_target() -> None:
    original = torch.zeros(2)
    computed = torch.ones(2, dtype=torch.float64)
    result = _write_back(
        torch.ops.aten.add_.Tensor,
        (original, torch.ones(2)),
        {},
        (computed, torch.ones(2, dtype=torch.float64)),
        {},
        result=computed,
        target=torch.float32,
    )
    assert result is original
    assert torch.equal(original, torch.ones(2))


def test_write_back_uses_target_for_fresh_scalar_result() -> None:
    original = torch.zeros(2)
    computed = torch.ones(2, dtype=torch.float64)
    fresh = torch.tensor([3.0, 4.0], dtype=torch.float64)
    result = _write_back(
        torch.ops.aten.add_.Tensor,
        (original, torch.ones(2)),
        {},
        (computed, torch.ones(2, dtype=torch.float64)),
        {},
        result=fresh,
        target=torch.bfloat16,
    )
    assert isinstance(result, Tensor)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, torch.tensor([3.0, 4.0], dtype=torch.bfloat16))


def test_copy_back_checks_lengths_and_ignores_mismatched_types() -> None:
    original = torch.zeros(2)
    computed = torch.ones(2, dtype=torch.float64)
    with pytest.raises(
        ValueError,
        match=r"^zip\(\) argument 2 is longer than argument 1$",
    ) as error:
        _copy_back([original], [computed, computed])
    assert str(error.value) == "zip() argument 2 is longer than argument 1"

    unchanged = torch.zeros(2)
    _copy_back([unchanged], computed)
    assert torch.equal(unchanged, torch.zeros(2))


def test_downcast_f64_propagates_target_through_nested_sequences() -> None:
    wide = torch.ones(2, dtype=torch.float64)
    got = _downcast_f64([wide, (wide,)], torch.bfloat16)
    assert isinstance(got, list)
    assert isinstance(got[0], Tensor)
    assert isinstance(got[1], tuple)
    assert isinstance(got[1][0], Tensor)
    assert got[0].dtype == torch.bfloat16
    assert got[1][0].dtype == torch.bfloat16


def test_downcast_result_uses_matching_and_shared_dtype_sources() -> None:
    wide = torch.ones(2, dtype=torch.float64)
    half = torch.ones(2, dtype=torch.float16)
    bfloat = torch.ones(2, dtype=torch.bfloat16)
    got = _downcast_result([wide], ([half, half], [bfloat]), {}, torch.float32)
    assert isinstance(got, list)
    assert isinstance(got[0], Tensor)
    assert got[0].dtype == torch.bfloat16

    shared = _downcast_result([wide], (half,), {}, torch.float32)
    assert isinstance(shared, list)
    assert isinstance(shared[0], Tensor)
    assert shared[0].dtype == torch.float16
    fallback = _downcast_result([wide], (), {}, torch.float16)
    assert isinstance(fallback, list)
    assert isinstance(fallback[0], Tensor)
    assert fallback[0].dtype == torch.float16


def test_run_unfused_addmm_reads_keyword_and_partial_operands() -> None:
    bias = torch.arange(6.0).reshape(2, 3)
    left = torch.arange(8.0).reshape(2, 4)
    right = torch.arange(12.0).reshape(4, 3)
    addmm = torch.ops.aten.addmm.default
    expected = (bias.double() + left.double() @ right.double()).float()

    keyword_result = _run_unfused(
        addmm,
        (),
        {"self": bias, "mat1": left, "mat2": right},
    )
    partial_result = _run_unfused(
        addmm,
        (bias,),
        {"mat1": left, "mat2": right},
    )
    assert isinstance(keyword_result, Tensor)
    assert isinstance(partial_result, Tensor)
    assert torch.equal(keyword_result, expected)
    assert torch.equal(partial_result, expected)


def test_run_unfused_addbmm_sums_completed_products_before_bias() -> None:
    generator = torch.Generator().manual_seed(814)
    # addbmm's square bias and batch matrices are the production stress shape.
    bias = torch.randn(64, 64, generator=generator, dtype=torch.float64)
    left = torch.randn(3, 64, 64, generator=generator, dtype=torch.float64)
    right = torch.randn(3, 64, 64, generator=generator, dtype=torch.float64)
    alpha, beta = 0.7, 0.3
    expected = alpha * torch.bmm(left, right).sum(dim=0) + beta * bias
    got = _run_unfused(
        torch.ops.aten.addbmm.default,
        (bias, left, right),
        {"alpha": alpha, "beta": beta},
    )

    assert isinstance(got, Tensor)
    assert torch.equal(got, expected)
    assert not torch.equal(
        torch.addbmm(bias, left, right, alpha=alpha, beta=beta),
        expected,
    )


def test_unfused_convolution_validates_and_adds_bias_after_convolution() -> None:
    conv = torch.ops.aten.convolution.default
    # convolution's 1-D image/weight contract fixes these channel axes.
    image = torch.arange(16.0).reshape(2, 2, 4)
    weight = torch.ones(3, 2, 2)
    bias = torch.arange(3.0)
    options: dict[str, object] = {
        "stride": (1,),
        "padding": (0,),
        "dilation": (1,),
        "transposed": False,
        "output_padding": (0,),
        "groups": 1,
    }
    # conv1d bias broadcasts over the production batch and sequence axes.
    expected = torch.nn.functional.conv1d(image, weight, None) + bias.reshape(
        1,
        3,
        1,
    )
    keyword = _unfused_convolution(
        conv,
        (image, weight),
        {**options, "bias": bias},
        bias=bias,
    )
    positional = _unfused_convolution(
        conv,
        (image, weight, bias),
        options,
        bias=bias,
    )
    assert torch.equal(keyword, expected)
    assert torch.equal(positional, expected)

    with pytest.raises(
        RuntimeError,
        match=r"^convolution bias must be one-dimensional",
    ):
        _unfused_convolution(
            conv,
            (image, weight),
            {**options, "bias": bias},
            # `convolution` requires a one-dimensional output-channel bias.
            bias=torch.ones(3, 1),
        )


def test_write_back_restores_write_identity_and_downcasts_other_outputs() -> None:
    original = torch.zeros(2)
    other = torch.ones(2)
    up_original, up_other = original.double(), other.double()
    computed = up_original.add_(up_other)
    fresh = torch.tensor([3.0, 4.0], dtype=torch.float64)
    result = _write_back(
        torch.ops.aten.add_.Tensor,
        (original, other),
        {},
        (up_original, up_other),
        {},
        result=[computed, fresh],
        target=torch.bfloat16,
    )

    assert isinstance(result, list)
    assert result[0] is original
    assert torch.equal(original, torch.ones(2))
    assert isinstance(result[1], Tensor)
    assert result[1].dtype == torch.bfloat16
    assert torch.equal(result[1], torch.tensor([3.0, 4.0], dtype=torch.bfloat16))


def test_ints_filters_non_integer_values() -> None:
    assert hasattr(bfb, "_ints")
    assert bfb._ints([1, "a", 3, True]) == [1, 3]
    assert bfb._ints("not a list") == []


def test_floating_tensors_ignores_integer_leaves_in_nested_inputs() -> None:
    floating = torch.ones(2)
    integer = torch.ones(2, dtype=torch.int64)
    found = _floating_tensors([integer, (floating, [integer])])

    assert found == [floating]
    assert found[0] is floating


def test_op_name_uses_final_namespace_component_before_overload() -> None:
    class NamedOp:
        def name(self) -> str:
            return "aten::nested::add.Tensor"

    assert _op_name(cast("OpOverload[..., object]", NamedOp())) == "add"


def test_replay_golden_moves_saved_input_to_module_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SmallModule(nn.Module):
        @override
        def forward(self, value: Tensor) -> Tensor:
            return value + 1

    path = tmp_path / "golden.pt"
    save_golden(
        path,
        {
            "state_dict": {},
            "input": torch.arange(6.0).reshape(2, 3),
            "output": torch.arange(6.0).reshape(2, 3) + 1,
            "seed": 17,
        },
    )
    original_move = move_to_device
    devices: list[str] = []

    def move_spy(value: object, device: str) -> object:
        devices.append(device)
        return original_move(value, device)

    monkeypatch.setattr("priml.testing.bfb.move_to_device", move_spy)

    def run(module: nn.Module, value: Tensor) -> Tensor:
        return cast(Tensor, module(value))

    _replay_golden(
        golden_path=path,
        build_module=SmallModule,
        build_input=lambda: torch.arange(6.0).reshape(2, 3),
        seed=17,
        run=run,
    )
    assert devices == ["cpu"]


def test_compact_copies_pins_empty_allocations_and_storage_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = torch.arange(12, dtype=torch.int32)
    first, second = base[2:8:2], base[4:10].view(torch.int16)
    original_empty = torch.empty
    original_set = Tensor.set_
    original_to = Tensor.to
    allocations: list[tuple[tuple[object, ...], dict[str, object]]] = []
    storage_sets: list[tuple[tuple[object, ...], dict[str, object]]] = []
    transfers: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def empty_spy(
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> Tensor:
        if dtype in {torch.uint8, first.dtype, second.dtype}:
            call_kwargs: dict[str, object] = {"dtype": dtype}
            if device is not None:
                call_kwargs["device"] = device
            allocations.append(((size,), call_kwargs))
        return original_empty(size, dtype=dtype, device=device)

    def set_spy(
        tensor: Tensor,
        storage: torch.UntypedStorage,
        storage_offset: int,
        size: tuple[int, ...],
        stride: tuple[int, ...],
    ) -> Tensor:
        if tensor.dtype in {torch.uint8, first.dtype, second.dtype}:
            storage_sets.append(((storage, storage_offset, size, stride), {}))
        return original_set(tensor, storage, storage_offset, size, stride)

    def to_spy(
        tensor: Tensor,
        device: str | torch.device | None = None,
        *,
        copy: bool = False,
    ) -> Tensor:
        if tensor.dtype is torch.uint8:
            transfers.append(((device,), {"copy": copy}))
        return original_to(tensor, device=device, copy=copy)

    monkeypatch.setattr(torch, "empty", empty_spy)
    monkeypatch.setattr(Tensor, "set_", set_spy)
    monkeypatch.setattr(Tensor, "to", to_spy)

    copied = _compact_copies([first, second])

    assert allocations[0] == (
        (0,),
        {"dtype": torch.uint8, "device": torch.device("cpu")},
    )
    assert allocations[1:] == [
        ((0,), {"dtype": first.dtype}),
        ((0,), {"dtype": second.dtype}),
    ]
    assert [call[0][1:] for call in storage_sets] == [
        (8, (32,), (1,)),
        (0, first.shape, first.stride()),
        (4, second.shape, second.stride()),
    ]
    assert transfers == [(("cpu",), {"copy": True})]
    first_copy = copied[id(first)]
    second_copy = copied[id(second)]
    assert (
        first_copy.untyped_storage().data_ptr()
        == second_copy.untyped_storage().data_ptr()
    )
    assert torch.equal(first_copy, first)
    assert torch.equal(second_copy, second)


def test_compact_copies_preserves_independent_tensor_contents() -> None:
    first = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    second = torch.arange(8, dtype=torch.float64).reshape(2, 4)

    copied = _compact_copies([first, second])

    assert torch.equal(copied[id(first)], first)
    assert torch.equal(copied[id(second)], second)
    assert (
        copied[id(first)].untyped_storage().data_ptr()
        != copied[id(second)].untyped_storage().data_ptr()
    )


def test_randomize_parameters_requests_cpu_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor = torch.Generator
    observed: list[object] = []

    def generator_factory(*, device: str) -> torch.Generator:
        observed.append(device)
        return constructor(device=device)

    monkeypatch.setattr(torch, "Generator", generator_factory)
    randomize_parameters(nn.Linear(3, 2), seed=53)
    assert observed == ["cpu"]


# iter11: _run_unfused second-half survivor probes.
def test_run_unfused_addmv_rejects_wrong_matrix_rank() -> None:
    bias = torch.zeros(2, dtype=torch.float64)
    matrix = torch.zeros(2, 3, dtype=torch.float64)
    vector = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        _run_unfused(
            torch.ops.aten.addmv.default,
            (bias, matrix.unsqueeze(0), vector),
            {},
        )


def test_run_unfused_addmv_rejects_wrong_vector_rank() -> None:
    bias = torch.zeros(2, dtype=torch.float64)
    matrix = torch.zeros(2, 3, dtype=torch.float64)
    # `addmv` requires rank-1 vectors; this singleton-axis case is intentional.
    vector = torch.zeros(3, 1, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        _run_unfused(torch.ops.aten.addmv.default, (bias, matrix, vector), {})


@pytest.mark.parametrize("name", ["addmm", "baddbmm", "addbmm", "addmv"])
def test_run_unfused_inplace_affine_op_returns_and_updates_bias(name: str) -> None:
    bias = torch.zeros(
        (1,)
        if name == "addmv"
        else (1, 1)
        if name in {"addmm", "addbmm"}
        else (1, 1, 1),
        dtype=torch.float64,
    )
    left = torch.tensor([[2.0**54, 1, -(2.0**54), 1, 2]], dtype=torch.float64)
    # Affine operators intentionally exercise their singleton output width.
    right = torch.ones((5, 1), dtype=torch.float64)
    if name in {"baddbmm", "addbmm"}:
        left, right = left.unsqueeze(0), right.unsqueeze(0)
    if name == "addmv":
        right = torch.ones(5, dtype=torch.float64)
    original = bias.clone()
    alpha, beta = 0.7, 0.3
    product = left @ right if name in {"addmm", "addmv"} else torch.bmm(left, right)
    if name == "addbmm":
        product = product.sum(dim=0)
    expected = alpha * product + beta * original
    func = cast("OpOverload[..., object]", getattr(torch.ops.aten, name + "_").default)
    result = _run_unfused(
        func,
        (bias, left, right),
        {"alpha": alpha, "beta": beta},
    )
    assert result is bias
    assert torch.equal(bias, expected)


def test_run_unfused_inplace_addmv_reads_keyword_self(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bias = torch.zeros(2, dtype=torch.float64)
    matrix = torch.ones((2, 4), dtype=torch.float64)
    vector = torch.arange(4.0, dtype=torch.float64)
    functional = torch.ops.aten.addmv.default

    def decomposed(*args: object, **kwargs: object) -> torch.Tensor:
        del args
        original, left, right = kwargs["self"], kwargs["mat"], kwargs["vec"]
        assert isinstance(original, torch.Tensor)
        assert isinstance(left, torch.Tensor)
        assert isinstance(right, torch.Tensor)
        return original + left @ right

    monkeypatch.setitem(decomposition_table, functional, decomposed)
    result = _run_unfused(
        torch.ops.aten.addmv_.default,
        (),
        {"self": bias, "mat": matrix, "vec": vector},
    )
    assert result is bias
    assert torch.equal(bias, torch.tensor([6.0, 6.0], dtype=torch.float64))


def test_run_unfused_baddbmm_unfuses_scaled_batch_products() -> None:
    # baddbmm's cancellation witness intentionally uses a repeated width.
    values = torch.tensor([2.0**40, 1, -(2.0**40), 1, 2], dtype=torch.float64)
    left = values.repeat(2, 1).unsqueeze(0)
    # baddbmm's production contract uses a singleton batch of matrices here.
    right = torch.ones((1, 5, 3), dtype=torch.float64)
    right[:, [0, 2]] *= 2.0**40
    # `baddbmm` broadcasts a singleton batch bias by its production contract.
    bias = torch.arange(6.0, dtype=torch.float64).reshape(1, 2, 3)
    alpha, beta = 0.3, 0.7
    expected = alpha * torch.bmm(left, right) + beta * bias
    with host_agnostic_numerics():
        result = torch.baddbmm(
            bias.float(),
            left.float(),
            right.float(),
            alpha=alpha,
            beta=beta,
        )
    assert result.dtype == torch.float32
    assert torch.equal(result, expected.float())


@pytest.mark.parametrize(
    "func",
    [torch.ops.aten.convolution.default, torch.ops.aten._convolution.default],
)
@pytest.mark.parametrize("positional_bias", [False, True])
def test_run_unfused_convolution_uses_positional_or_keyword_bias(
    func: OpOverload[..., object],
    positional_bias: bool,
) -> None:
    # convolution's 1-D image/weight contract fixes these channel axes.
    image = torch.arange(16.0, dtype=torch.float64).reshape(2, 2, 4)
    weight = torch.ones(3, 2, 2, dtype=torch.float64)
    bias = torch.arange(3.0, dtype=torch.float64)
    options: dict[str, object] = {
        "stride": (1,),
        "padding": (0,),
        "dilation": (1,),
        "transposed": False,
        "output_padding": (0,),
        "groups": 1,
    }
    args: tuple[object, ...] = (
        (image, weight, bias) if positional_bias else (image, weight)
    )
    kwargs: dict[str, object] = {} if positional_bias else {"bias": bias}
    if func is torch.ops.aten._convolution.default:
        kwargs.update(
            benchmark=False,
            deterministic=False,
            cudnn_enabled=False,
            allow_tf32=False,
        )
    # conv1d bias broadcasts over the production batch and sequence axes.
    expected = torch.nn.functional.conv1d(image, weight, None) + bias.reshape(1, 3, 1)
    result = _run_unfused(func, args, {**options, **kwargs})
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, expected)


def test_run_unfused_addmv_reads_positional_matrix_vector_operands() -> None:
    bias = torch.arange(2.0, dtype=torch.float64)
    matrix = torch.arange(8.0, dtype=torch.float64).reshape(2, 4)
    vector = torch.arange(4.0, dtype=torch.float64)
    alpha, beta = 0.25, 0.5
    result = _run_unfused(
        torch.ops.aten.addmv.default,
        (bias, matrix, vector),
        {"alpha": alpha, "beta": beta},
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2,)
    assert torch.equal(result, alpha * (matrix @ vector) + beta * bias)


def test_run_unfused_addmv_keyword_operands_reach_decomposition() -> None:
    bias = torch.arange(2.0, dtype=torch.float64)
    matrix = torch.arange(8.0, dtype=torch.float64).reshape(2, 4)
    vector = torch.arange(4.0, dtype=torch.float64)
    with pytest.raises(TypeError, match="unexpected keyword argument 'mat'"):
        _run_unfused(
            torch.ops.aten.addmv.default,
            (),
            {"self": bias, "mat": matrix, "vec": vector, "alpha": 0.25, "beta": 0.5},
        )


def test_run_unfused_addmv_rejects_wrong_vector_rank_via_native_op() -> None:
    bias = torch.zeros(2, dtype=torch.float64)
    matrix = torch.zeros(2, 4, dtype=torch.float64)
    vector = torch.zeros(2, 4, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        _run_unfused(torch.ops.aten.addmv.default, (bias, matrix, vector), {})


def test_run_unfused_baddbmm_reads_positional_batch_operands() -> None:
    # `baddbmm` bias follows its fixed batch-matrix output contract.
    bias = torch.arange(16.0, dtype=torch.float64).reshape(2, 2, 4)
    # baddbmm's batch and inner dimensions are fixed by the operator contract.
    left = torch.arange(12.0, dtype=torch.float64).reshape(2, 2, 3)
    # baddbmm's batch and inner dimensions are fixed by the operator contract.
    right = torch.arange(24.0, dtype=torch.float64).reshape(2, 3, 4)
    result = _run_unfused(
        torch.ops.aten.baddbmm.default,
        (bias, left, right),
        {"alpha": 0.25, "beta": 0.5},
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 2, 4)
    assert torch.equal(result, 0.25 * torch.bmm(left, right) + 0.5 * bias)


def test_run_unfused_baddbmm_reads_keyword_batch_operands() -> None:
    # `baddbmm` bias follows its fixed batch-matrix output contract.
    bias = torch.arange(16.0, dtype=torch.float64).reshape(2, 2, 4)
    # baddbmm's batch and inner dimensions are fixed by the operator contract.
    left = torch.arange(12.0, dtype=torch.float64).reshape(2, 2, 3)
    # baddbmm's batch and inner dimensions are fixed by the operator contract.
    right = torch.arange(24.0, dtype=torch.float64).reshape(2, 3, 4)
    alpha, beta = 0.25, 0.5
    result = _run_unfused(
        torch.ops.aten.baddbmm.default,
        (),
        {"self": bias, "batch1": left, "batch2": right, "alpha": alpha, "beta": beta},
    )
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, alpha * torch.bmm(left, right) + beta * bias)


def test_run_unfused_baddbmm_rejects_wrong_batch_matrix_rank() -> None:
    # `baddbmm` requires a batch matrix, while this case intentionally supplies rank 2.
    bias = torch.zeros(2, 2, 4, dtype=torch.float64)
    left = torch.zeros(2, 3, dtype=torch.float64)
    right = torch.zeros(2, 3, 4, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        _run_unfused(
            torch.ops.aten.baddbmm.default,
            (),
            {"self": bias, "batch1": left, "batch2": right},
        )


def test_run_unfused_addmm_activation_uses_named_operands() -> None:
    bias = torch.arange(6.0, dtype=torch.float64).reshape(2, 3)
    left = torch.arange(8.0, dtype=torch.float64).reshape(2, 4)
    right = torch.arange(12.0, dtype=torch.float64).reshape(4, 3)
    op = torch.ops.aten._addmm_activation.default
    result = _run_unfused(
        op,
        (bias, left, right),
        {"beta": 0.5, "alpha": 0.25, "use_gelu": True},
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 3)
    expected = cast(
        Tensor,
        op(bias, left, right, beta=0.5, alpha=0.25, use_gelu=True),
    )
    assert torch.equal(result, expected)


def test_run_unfused_addbmm_reads_keyword_batch_operands() -> None:
    # `addbmm` reduces its batch matrices into this fixed bias geometry.
    bias = torch.arange(8.0, dtype=torch.float64).reshape(2, 4)
    # baddbmm's batch and inner dimensions are fixed by the operator contract.
    left = torch.arange(12.0, dtype=torch.float64).reshape(2, 2, 3)
    # baddbmm's batch and inner dimensions are fixed by the operator contract.
    right = torch.arange(24.0, dtype=torch.float64).reshape(2, 3, 4)
    alpha, beta = 0.25, 0.5
    result = _run_unfused(
        torch.ops.aten.addbmm.default,
        (),
        {"self": bias, "batch1": left, "batch2": right, "alpha": alpha, "beta": beta},
    )
    expected = alpha * torch.bmm(left, right).sum(dim=0) + beta * bias
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 4)
    assert torch.equal(result, expected)


def test_run_unfused_addbmm_keeps_batched_product_sum_unfused() -> None:
    generator = torch.Generator().manual_seed(814)
    # addbmm's square bias and batch matrices are the production stress shape.
    bias = torch.randn(64, 64, generator=generator, dtype=torch.float64)
    left = torch.randn(3, 64, 64, generator=generator, dtype=torch.float64)
    right = torch.randn(3, 64, 64, generator=generator, dtype=torch.float64)
    alpha, beta = 0.7, 0.3
    expected = alpha * torch.bmm(left, right).sum(dim=0) + beta * bias
    result = _run_unfused(
        torch.ops.aten.addbmm.default,
        (bias, left, right),
        {"alpha": alpha, "beta": beta},
    )
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, expected)
    assert not torch.equal(
        torch.addbmm(bias, left, right, alpha=alpha, beta=beta),
        expected,
    )


def _run_unfused_op(name: str) -> OpOverload[..., object]:
    return cast("OpOverload[..., object]", getattr(torch.ops.aten, name).default)


def test_non_aten_run_forwards_keyword_arguments() -> None:
    with torch.library._scoped_library("bfb_run_unfused_probe", "FRAGMENT") as library:
        library.define("scale(Tensor value, float factor) -> Tensor")
        library.impl("scale", _scale_tensor, "CompositeExplicitAutograd")
        op = cast(
            "OpOverload[..., object]",
            torch.ops.bfb_run_unfused_probe.scale.default,
        )
        value = torch.tensor([2.0, 3.0])
        result = _run_unfused(op, (value,), {"factor": 4.0})
    assert isinstance(result, Tensor)
    assert torch.equal(result, torch.tensor([8.0, 12.0]))


def test_run_unfused_validates_addmm_bias_before_decomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bias = torch.zeros((3, 5), dtype=torch.float64)
    left = torch.ones((2, 3, 4), dtype=torch.float64)
    right = torch.ones((4, 5), dtype=torch.float64)
    monkeypatch.setitem(
        decomposition_table,
        torch.ops.aten.addmm.default,
        _FixedResult(result=torch.full((2, 3, 5), 17.0)),
    )
    with pytest.raises(RuntimeError):
        _run_unfused(torch.ops.aten.addmm.default, (bias, left, right), {})


@pytest.mark.parametrize("name", ["addmm", "addbmm", "baddbmm", "_addmm_activation"])
def test_run_unfused_affine_dispatch_accepts_keyword_operands(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    bias = torch.zeros((2, 3), dtype=torch.float64)
    left = torch.arange(8.0, dtype=torch.float64).reshape(2, 4)
    right = torch.arange(12.0, dtype=torch.float64).reshape(4, 3)
    if name in {"addbmm", "baddbmm"}:
        left = left.unsqueeze(0).expand(2, -1, -1).contiguous()
        right = right.unsqueeze(0).expand(2, -1, -1).contiguous()
    if name == "baddbmm":
        bias = bias.unsqueeze(0).expand(2, -1, -1).contiguous()
    operands: dict[str, object] = {"self": bias, "alpha": 0.25, "beta": 0.5}
    if name in {"addbmm", "baddbmm"}:
        operands["batch1"] = left
        operands["batch2"] = right
    else:
        operands["mat1"] = left
        operands["mat2"] = right
    if name == "_addmm_activation":
        operands["use_gelu"] = True
    op = _run_unfused_op(name)
    if name == "addbmm":
        result = _run_unfused(op, (), operands)
        assert isinstance(result, Tensor)
        assert torch.equal(
            result,
            0.25 * torch.bmm(left, right).sum(dim=0) + 0.5 * bias,
        )
    else:
        sentinel = torch.full_like(bias, 37.0)
        monkeypatch.setitem(decomposition_table, op, _FixedResult(result=sentinel))
        assert _run_unfused(op, (), operands) is sentinel


def test_inplace_addmv_preserves_alias_and_scaled_result() -> None:
    bias = torch.zeros(2, dtype=torch.float64)
    matrix = torch.arange(8.0, dtype=torch.float64).reshape(2, 4)
    vector = torch.arange(4.0, dtype=torch.float64)
    result = _run_unfused(
        torch.ops.aten.addmv_.default,
        (bias, matrix, vector),
        {"alpha": 0.5, "beta": 1.0},
    )
    assert result is bias
    assert torch.equal(bias, 0.5 * (matrix @ vector))


@pytest.mark.parametrize("name", ["addmm", "baddbmm", "addmv"])
def test_inplace_affine_alias_uses_functional_decomposition(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    bias = torch.zeros((2, 3), dtype=torch.float64)
    left = torch.ones((2, 4), dtype=torch.float64)
    right = torch.ones((4, 3), dtype=torch.float64)
    if name == "baddbmm":
        bias = bias.unsqueeze(0).expand(2, -1, -1).clone()
        left = left.unsqueeze(0).expand(2, -1, -1).contiguous()
        right = right.unsqueeze(0).expand(2, -1, -1).contiguous()
    elif name == "addmv":
        bias = torch.zeros(2, dtype=torch.float64)
        right = torch.ones(4, dtype=torch.float64)
    sentinel = torch.full_like(bias, 23.0)
    functional = _run_unfused_op(name)
    monkeypatch.setitem(decomposition_table, functional, _FixedResult(result=sentinel))
    result = _run_unfused(
        _run_unfused_op(name + "_"),
        (bias, left, right),
        {"alpha": 0.5, "beta": 1.0},
    )
    assert result is bias
    assert torch.equal(bias, sentinel)


def test_addmm_activation_rejects_bias_that_expands_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = torch.ops.aten._addmm_activation.default
    # _addmm_activation intentionally tests a bias that expands output.
    bias = torch.zeros((2, 2, 3), dtype=torch.float64)
    left = torch.ones((2, 4), dtype=torch.float64)
    right = torch.ones((4, 3), dtype=torch.float64)
    # _addmm_activation intentionally tests a bias that expands output.
    sentinel = torch.full((2, 2, 3), 29.0)
    monkeypatch.setitem(decomposition_table, op, _FixedResult(result=sentinel))
    with pytest.raises(RuntimeError):
        _run_unfused(
            op,
            (bias, left, right),
            {"alpha": 0.5, "beta": 1.0, "use_gelu": True},
        )


def test_inplace_baddbmm_keyword_dispatch_validates_and_writes_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # baddbmm's batch matrix contract fixes the two leading dimensions.
    # _addmm_activation intentionally tests a bias that expands output.
    bias = torch.zeros((2, 2, 3), dtype=torch.float64)
    left = torch.ones((2, 2, 4), dtype=torch.float64)
    right = torch.ones((2, 4, 3), dtype=torch.float64)
    sentinel = torch.full_like(bias, 31.0)
    monkeypatch.setitem(
        decomposition_table,
        torch.ops.aten.baddbmm.default,
        _FixedResult(result=sentinel),
    )
    result = _run_unfused(
        torch.ops.aten.baddbmm_.default,
        (),
        {
            "self": bias,
            "batch1": left,
            "batch2": right,
            "alpha": 0.5,
            "beta": 1.0,
        },
    )
    assert result is bias
    assert torch.equal(bias, sentinel)


@pytest.mark.parametrize("name", ["convolution", "_convolution"])
@pytest.mark.parametrize("positional", [False, True])
def test_run_unfused_routes_convolution_bias_forms(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    positional: bool,
) -> None:
    # convolution's 1-D image/weight contract fixes these channel axes.
    image = torch.ones((2, 2, 4), dtype=torch.float64)
    weight = torch.ones((3, 2, 2), dtype=torch.float64)
    bias = torch.arange(3.0, dtype=torch.float64)
    # `convolution` output preserves the batch/channel/sequence contract.
    result = torch.full((2, 3, 3), 19.0)
    seen: list[Tensor] = []

    def unfused(
        func: OpOverload[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        *,
        bias: Tensor,
    ) -> Tensor:
        del func, args, kwargs
        seen.append(bias)
        return result

    monkeypatch.setattr("priml.testing.bfb._unfused_convolution", unfused)
    arguments: tuple[object, ...] = (
        (image, weight, bias) if positional else (image, weight)
    )
    options: dict[str, object] = {} if positional else {"bias": bias}
    if name == "_convolution":
        options.update(
            benchmark=False,
            deterministic=False,
            cudnn_enabled=False,
            allow_tf32=False,
        )
    actual = _run_unfused(_run_unfused_op(name), arguments, options)
    assert actual is result
    assert seen == [bias]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
