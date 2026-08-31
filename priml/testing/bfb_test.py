"""Tests for the bfb golden harness itself."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import ast
import os

from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode

import pytest
import torch


if TYPE_CHECKING:
    from torch._ops import OpOverload

from priml.math.custom_types import TensorFn
from priml.model.attention.self_attention import SelfAttention
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import (
    _ENV_REGENERATE,
    _EXACT_F32_OPS,
    _assert_equal,
    _assert_portable_output_dtype,
    _max_ulp_diff,
    _op_name,
    assert_bfb_against_golden,
    bfb_devices,
    first_tensor,
    host_agnostic_numerics,
    randomize_parameters,
    regenerate_golden,
    state_differs,
)


@pytest.fixture(autouse=True)
def _isolate_regenerate_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction] -- pytest invokes autouse fixtures by injection, not by name
    """Clear ``BFB_REGENERATE`` so a global regen run cannot disable these tests.

    The drift-detection and round-trip tests mint into ``tmp_path`` and must
    compare, not regenerate. A suite-wide ``BFB_REGENERATE=1`` (set to remint
    committed goldens) would otherwise force every ``assert_bfb_against_golden``
    to regenerate, so the drift tests see no mismatch and fail "DID NOT RAISE".
    Tests that need regeneration call ``regenerate_golden``, which sets the flag
    locally.
    """
    monkeypatch.delenv(_ENV_REGENERATE, raising=False)


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
    testdata = tmp_path / "testdata"
    # A missing committed golden is regenerated but remains red until reviewed.
    with pytest.raises(AssertionError, match="Missing golden regenerated"):
        assert_bfb_against_golden(
            golden_dir=testdata,
            golden_name="linear_min",
            build_module=_build_min_linear,
            build_input=_build_min_input,
            seed=0,
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


def test_bfb_files_do_not_use_typing_any() -> None:
    paths = [
        Path(__file__),
        Path(__file__).with_name("bfb.py"),
        Path(__file__).parents[1] / "model" / "transformer" / "block_test.py",
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
            output = module(inp)
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

    with pytest.raises(ValueError, match="CPU-only"):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="cuda_is_not_hermetic",
            build_module=_build_min_linear,
            build_input=_build_min_input,
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
    assert _ENV_REGENERATE not in os.environ
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
            self.lin = nn.Linear(3, 2)

        @override
        def forward(self, a: Tensor, b: Tensor) -> Tensor:
            return self.lin(a) + b

    def build_module() -> nn.Module:
        return TwoArgModule()

    def build_input() -> dict[str, Tensor]:
        return {
            "a": torch.linspace(-1.0, 1.0, 6).reshape(2, 3),
            "b": torch.linspace(0.0, 1.0, 4).reshape(2, 2),
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
    """A golden whose run mutated nothing must not carry ``post_state_dict``.

    ``_write_golden`` omits the key when the state is unchanged and
    ``_replay_golden`` reads an absent one as "equal to ``state_dict``", so a
    stored copy of the pre-state asserts exactly what no key asserts -- at
    twice the weights on disk.

    Gated rather than trusted because the omission arrived as a WRITER change
    with no migration: nineteen goldens minted before it kept the copy, the
    tolerant reader never went red, and 1.1 MiB sat unnoticed until someone
    read a size diff. A guard that only stops new violations leaves the
    existing ones invisible forever.
    """
    goldens = sorted(Path(__file__).resolve().parent.parent.rglob("*.pt"))
    assert goldens, "no goldens found; the glob no longer matches the layout"
    stale: list[Path] = []
    for path in goldens:
        payload = _loaded_golden(path)
        post = payload.get("post_state_dict")
        if post is None:
            continue
        if not state_differs(payload["state_dict"], post):
            stale.append(path)
    assert not stale, (
        f"{len(stale)} golden(s) store a post-state equal to their pre-state: "
        f"{[str(p) for p in stale]}"
    )


def _loaded_golden(path: Path) -> dict[str, dict[str, Tensor]]:
    """Read a golden's two state dicts, which are all this check reads.

    ``torch.load`` is untyped, so the shape is narrowed once here rather than
    cast at each use. Only the state entries are typed: a golden also holds an
    input, an output, and a seed, and claiming those are state dicts to
    satisfy one reader would be a false annotation.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    return {
        key: value
        for key, value in cast(dict[str, object], payload).items()
        if key in ("state_dict", "post_state_dict") and isinstance(value, dict)
    }


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
        self.register_buffer("running_sum", torch.zeros(3))

    @override
    def forward(self, input: Tensor) -> Tensor:
        out = self.lin(input)
        assert isinstance(self.running_sum, Tensor)
        self.running_sum.add_(out.detach().sum(dim=0))
        return out


def test_bfb_captures_forward_buffer_mutation(tmp_path: Path) -> None:
    """INF-017: a forward that mutates a buffer must round-trip bit-exactly.

    Without ``mutates_state`` the harness used to compare the post-forward
    live state to the PRE-run golden and falsely fail. The buffer mutation
    must be captured and compared bit-for-bit on its own.
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
    payload = torch.load(
        tmp_path / "buffer_mutating.pt", weights_only=False, map_location="cpu"
    )
    pre = payload["state_dict"]["running_sum"]
    post = payload["post_state_dict"]["running_sum"]
    assert torch.equal(pre, torch.zeros(3))
    assert not torch.equal(post, torch.zeros(3))


def test_bfb_detects_forward_buffer_drift(tmp_path: Path) -> None:
    """INF-017: a divergent buffer mutation must be caught."""
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
            assert isinstance(self.running_sum, Tensor)
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
    """INF-018: regeneration must verify the freshly written golden round-trips.

    A regenerator that writes a golden whose output is not reproducible on
    reload must fail loudly during regeneration, not silently pass.
    """
    drift = {"n": 0}

    def flaky_runner(module: nn.Module, inp: Tensor) -> Tensor:
        # First call (capture) returns the clean output; the verification
        # rerun returns a perturbed output, so the golden cannot round-trip.
        out = module(inp)
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
        output = module(inp)
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


def test_regenerate_round_trip_passes_for_clean_module(tmp_path: Path) -> None:
    """INF-018: a deterministic module regenerates and self-verifies cleanly."""
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


def _f32_equals_f64_downcast(op: TensorFn) -> bool:
    """True if ``op``'s float32 result equals its float64-then-downcast result.

    An exact-allowlist op must be host-independent: computing it in float64 and
    narrowing back to float32 must reproduce the native float32 result bit-for-
    bit. An op that fails this (e.g. a fused multiply-add rounding differently,
    or a vector-width-dependent reduction) must NOT be allowlisted -- it has to
    be upcast like every other arithmetic op.

    Each probe applies exactly ONE allowlisted op to its float32 input; any
    auxiliary operand must be exactly float32-representable (an integer, a power
    of two, or a flip/copy of the input) so the probe isolates the op under test
    rather than folding in a second op's rounding.
    """
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(4096, dtype=torch.float32, generator=gen)
    return torch.equal(op(a), op(a.double()).float())


# Arithmetic exact-allowlist ops, each paired with a single-op probe whose
# auxiliary operand is exactly float32-representable (so only the named op's
# rounding is under test). Pure data-movement ops (views, reshapes, gathers)
# carry no arithmetic and so are trivially host-independent; only ops that
# compute a value need proving here.
_EXACT_ARITHMETIC_PROBES: dict[str, TensorFn] = {
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
_NONARITHMETIC_PROBES: dict[str, TensorFn] = {
    "gather": lambda a: a.gather(0, torch.arange(0, a.numel(), 7) % a.numel()),
    "index_select": lambda a: a.index_select(0, torch.arange(0, 50)),
    "masked_fill": lambda a: a.masked_fill(a > 0, 0.123),
    "masked_fill_": lambda a: a.clone().masked_fill_(a > 0, 0.123),
    "masked_select": lambda a: a.masked_select(a > 0),
    "cat": lambda a: torch.cat([a, a.flip(0)]),
    "stack": lambda a: torch.stack([a, a.flip(0)]),
    "clone": lambda a: a.clone(),
    "copy_": lambda a: torch.empty_like(a).copy_(a),
    "_to_copy": lambda a: torch.ops.aten._to_copy(a, dtype=a.dtype),
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
_VALUE_PRODUCING_NONARITH = frozenset(
    {"where", "copy_", "_to_copy", "to", "fill_", "masked_fill", "masked_fill_"}
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
    construction); this proves it for every member that touches float32 data, so a future entry that
    secretly does arithmetic (rounding-dependent) fails here rather than minting
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
_ALLOCATION_OPS = frozenset(
    {"empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided"}
)


def _divergent_allowlisted_single_tensor_ops() -> set[str]:
    """Allowlisted Aten ops that differ from their float64 recomputation.

    Only allowlisted names can invalidate this guard. Calling unrelated Aten
    packets is both unnecessary and unsafe: some nominally single-tensor
    defaults initialize platform backends before argument validation, and
    PyTorch's MPS ``pin_memory`` packet can segfault the interpreter. Probe the
    exact policy surface with a fixed seeded input instead.
    """
    divergent: set[str] = set()
    for name in sorted(_EXACT_F32_OPS):
        if name in _ALLOCATION_OPS:
            continue
        packet = getattr(torch.ops.aten, name)
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


def _f32_leaking_ops(run: Callable[[], object]) -> set[str]:
    """Names of non-allowlisted ops that still receive float32 under the harness.

    The trace mode is entered OUTSIDE ``host_agnostic_numerics`` so it observes
    each op's arguments AFTER the upcast dispatch has run -- i.e. the dtype the
    real kernel actually computes on. Invariant the harness guarantees: every
    op is either allowlisted (runs native float32, proven host-independent) or
    upcast to float64. So a non-allowlisted op still seeing a float32 argument
    here ran native float32 without being upcast -- a cross-host-divergence leak
    (the failure mode the flash-attention kernel exhibited before the SDPA-math
    pin).
    """
    leaks: set[str] = set()

    def has_f32(value: object) -> bool:
        if isinstance(value, Tensor):
            return value.dtype == torch.float32
        if isinstance(value, (list, tuple)):
            return any(
                has_f32(v) for v in cast("list[object] | tuple[object, ...]", value)
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
            if name not in _EXACT_F32_OPS and any(has_f32(value) for value in values):
                leaks.add(name)
            return func(*args, **(kwargs or {}))

    with _Trace(), host_agnostic_numerics():
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
        channels_in=16, attn=SelfAttention.Config(num_heads=2, channels_head=8)
    ).make()
    randomize_parameters(block, seed=0)
    inp = torch.randn(2, 4, 16, requires_grad=True)

    leaks = _f32_leaking_ops(lambda: block(inp).sum().backward())
    assert not leaks, (
        "non-allowlisted ops ran on float32 (not upcast; cross-host-divergent): "
        f"{sorted(leaks)}"
    )


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

    a = torch.randn(4096, dtype=torch.float32)
    b = torch.randn(4096, dtype=torch.float32)
    fma32 = a.clone().addcmul_(b, b, value=0.7)
    fma64 = a.double().addcmul_(b.double(), b.double(), value=0.7).float()
    assert not torch.equal(fma32, fma64)

    idx = torch.zeros(1024, dtype=torch.long)
    src = torch.randn(1024, dtype=torch.float32)
    acc32 = torch.zeros(1, dtype=torch.float32).scatter_add_(0, idx, src)
    acc64 = (
        torch.zeros(1, dtype=torch.float64).scatter_add_(0, idx, src.double()).float()
    )
    assert not torch.equal(acc32, acc64)


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


def test_host_agnostic_numerics_preserves_mixed_foreach_dtypes() -> None:
    inputs = [
        torch.tensor([0.5], dtype=torch.float16),
        torch.tensor([0.5], dtype=torch.float32),
    ]
    foreach_sin = cast(
        Callable[[list[Tensor]], list[Tensor]],
        getattr(torch, "_foreach_sin"),  # noqa: B009 -- stub-less torch member
    )
    expected = foreach_sin(inputs)

    with host_agnostic_numerics():
        actual = foreach_sin(inputs)

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
    # torch stubs omit _foreach_norm; resolve it through a typed Callable so
    # both type checkers see a known signature for this public foreach op.
    foreach_norm = cast(
        Callable[[list[Tensor], float], list[Tensor]],
        getattr(torch, "_foreach_norm"),  # noqa: B009 -- stub-less torch member
    )
    with host_agnostic_numerics():
        actual = foreach_norm([x], 2.0)[0]
    assert torch.equal(actual, expected)


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
    # torch stubs omit the in-place foreach ops; resolve through a typed Callable.
    foreach_mul_ = cast(
        Callable[[list[Tensor], list[Tensor]], None],
        getattr(torch, "_foreach_mul_"),  # noqa: B009 -- stub-less torch member
    )
    with host_agnostic_numerics():
        foreach_mul_(xs, ys)
    for got, exp in zip(xs, expected, strict=True):
        assert torch.equal(got, exp)


def test_host_agnostic_multi_output_write_op_keeps_all_returns() -> None:
    """A write op returning fresh non-write tensors keeps them, not just writes.

    ``_native_batch_norm_legit`` (training) writes ``running_mean``/
    ``running_var`` in place but returns a 3-tuple ``(output, save_mean,
    save_invstd)`` -- none of which is a write target. The write-back must return
    the full 3-tuple of computed outputs, not a tuple of the 2 write originals
    (the REFAC-001 collapse bug). Asserts the returned ``output`` has the input's
    shape and is the normalized result, proving non-write returns survive.
    """
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    running_mean = torch.zeros(3, dtype=torch.float32)
    running_var = torch.ones(3, dtype=torch.float32)
    weight = torch.ones(3, dtype=torch.float32)
    bias = torch.zeros(3, dtype=torch.float32)
    with host_agnostic_numerics():
        out, save_mean, save_invstd = torch.ops.aten._native_batch_norm_legit(
            x, weight, bias, running_mean, running_var, True, 0.1, 1e-5
        )
    expected = torch.ops.aten._native_batch_norm_legit(
        x.double(),
        weight.double(),
        bias.double(),
        running_mean.double(),
        running_var.double(),
        True,
        0.1,
        1e-5,
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


def _f64_output_runner(module: nn.Module, inp: Tensor) -> Tensor:
    """Return the float64 scratch, skipping the round back to float32."""
    return cast(Tensor, module(inp)).double()


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
    """A golden written BEFORE the gate still reports why it is unportable.

    Minting is not the only entry point: the committed goldens predate this
    check, so the replay path must name the cause too rather than failing on a
    one-ULP comparison the reader cannot attribute.
    """
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="legacy_f64",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )
    path = tmp_path / "legacy_f64.pt"
    payload = torch.load(path, weights_only=False, map_location="cpu")
    payload["output"] = payload["output"].double()
    torch.save(payload, path)
    with pytest.raises(TypeError, match="float64"):
        assert_bfb_against_golden(
            golden_dir=tmp_path,
            golden_name="legacy_f64",
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
    assert not state_differs({"value": nan}, {"value": nan.clone()})


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


def test_regenerate_golden_preserves_existing_regenerate_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``regenerate_golden`` restores a caller-set ``BFB_REGENERATE``.

    Unconditionally clearing it would silently disable regeneration for later
    tests in a process launched with ``BFB_REGENERATE=1``.
    """
    monkeypatch.setenv(_ENV_REGENERATE, "1")
    regenerate_golden(
        golden_dir=tmp_path,
        golden_name="clean",
        build_module=_build_min_linear,
        build_input=_build_min_input,
        seed=0,
    )
    assert os.environ[_ENV_REGENERATE] == "1"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
