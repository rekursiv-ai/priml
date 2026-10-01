"""Tests for the minimal gated recurrent units: the fp32 stack and the scan variants.

The scan tests pin the recurrence's contract: shapes and dtypes, the
step/sequence equivalence, the reset semantics, and derivatives that agree with
autograd on the same arithmetic. On a GPU the Triton kernels agree with the
torch reference at torch's bf16 tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, override

from configgle.testing import assert_pprint_golden
from torch import Tensor

import pytest
import torch

from priml.model import min_gru
from priml.model.min_gru import (
    MinGRU,
    MinGRUBlock,
    ScanBackward,
    ScanForward,
    TorchScan,
    TritonScan,
    _check_layout,
    _runs_triton,
    _scan_affine,
    _terminals_view,
)
from priml.testing.bfb import assert_bfb_against_golden


_CWD: Final = Path(__file__).resolve().parent


def _allow_cpu_scan(combined: Tensor) -> None:
    del combined


def _force_triton(*tensors: Tensor) -> bool:
    del tensors
    return True


class _FakeKernel:
    def __getitem__(self, grid: tuple[int, ...]) -> _FakeKernel:
        del grid
        return self

    def __call__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _FakeScanKernels:
    forward = _FakeKernel()
    backward = _FakeKernel()
    step = _FakeKernel()


def _fake_kernels(**helpers: object) -> _FakeScanKernels:
    del helpers
    return _FakeScanKernels()


def test_min_gru_matches_hand_forward_and_returns_layer_state() -> None:
    model = MinGRU(input_channels=5, channels=6, layers=2)
    with torch.no_grad():
        model.input_projection.weight.zero_()
        for layer in model.layers:
            layer.weight.zero_()
    inputs = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
    state = torch.zeros(2, 3, 6)
    reset = torch.tensor(
        [
            [True, False, False, False],
            [True, False, False, False],
            [True, False, False, False],
        ],
    )
    outputs, final = model(inputs, state=state, reset=reset)
    update = 1 / 2
    candidate = 1 / 2
    strength = 1 / 2
    carry0 = update * candidate
    carry1 = (1 - update) * carry0 + update * candidate
    carry2 = (1 - update) * carry1 + update * candidate
    carry3 = (1 - update) * carry2 + update * candidate
    layer1_values = [strength * carry for carry in (carry0, carry1, carry2, carry3)]
    layer2_values = [
        (1 - strength) * value + strength * carry
        for value, carry in zip(
            layer1_values,
            (carry0, carry1, carry2, carry3),
            strict=True,
        )
    ]
    expected = torch.tensor(
        [[[value] * 6 for value in layer2_values]],
    ).expand(3, -1, -1)
    expected_final = torch.full((2, 3, 6), carry3)
    assert outputs.shape == (3, 4, 6)
    assert final.shape == expected_final.shape
    torch.testing.assert_close(outputs, expected)
    torch.testing.assert_close(final, expected_final)


def test_min_gru_repeated_step_equals_sequence() -> None:
    torch.manual_seed(0)
    model = MinGRU(input_channels=3, channels=4, layers=2)
    inputs = torch.randn(2, 5, 3)
    reset = torch.zeros(2, 5, dtype=torch.bool)
    sequence, state = model(inputs, reset=reset)
    carry = model.initial_state(2)
    pieces: list[torch.Tensor] = []
    for index in range(inputs.shape[1]):
        output, carry = model(
            inputs[:, index : index + 1],
            state=carry,
            reset=reset[:, index : index + 1],
        )
        pieces.append(output)
    torch.testing.assert_close(sequence, torch.cat(pieces, dim=1))
    torch.testing.assert_close(state, carry)


def test_min_gru_reset_discards_previous_state() -> None:
    model = MinGRU(input_channels=4, channels=5, layers=1)
    inputs = torch.randn(3, 2, 4)
    reset = torch.tensor([[False, True], [False, True], [False, True]])
    outputs, _ = model(inputs, reset=reset)
    fresh_inputs = inputs[:, 1:].expand(-1, 2, -1)
    fresh, _ = model(
        fresh_inputs,
        state=model.initial_state(3),
        reset=torch.tensor([[True, False], [True, False], [True, False]]),
    )
    torch.testing.assert_close(outputs[:, 1:], fresh[:, :1])


def test_min_gru_rejects_an_empty_time_dimension() -> None:
    """Reject T=0 at the public boundary before stacked outputs fail incidentally."""
    model = MinGRU(input_channels=2, channels=2)

    with pytest.raises(ValueError, match="time"):
        # Degenerate zero-time input is intentional for this pytest.raises case.
        model(torch.zeros(2, 0, 3))


def test_min_gru_validates_constructor_and_call_shapes() -> None:
    for values in ((0, 2, 1), (2, 0, 1), (2, 2, 0)):
        with pytest.raises(ValueError, match="positive"):
            MinGRU(values[0], values[1], values[2])
    model = MinGRU(2, 4, layers=2)
    inputs = torch.randn(3, 5, 2)
    with pytest.raises(ValueError, match="shape"):
        model(inputs[..., 0])
    with pytest.raises(ValueError, match="width"):
        model(torch.randn(4, 5, 3))
    with pytest.raises(ValueError, match="reset"):
        model(inputs, reset=torch.zeros(3, 4))
    with pytest.raises(ValueError, match="state"):
        model(inputs, state=torch.zeros(2, 3, 5))
    with pytest.raises(ValueError, match="Batch"):
        model.initial_state(0)


def test_scan_config_cost_and_triton_cpu_dispatch() -> None:
    config = MinGRUBlock.Config()
    config.channels_hidden = 4
    config.finalize()
    estimate = config.cost(seq_len=2, batch_size=3, dtype=torch.float32)
    assert estimate.params > 0
    block = config.make()
    inputs = torch.randn(2, 3, 4)
    state = torch.zeros(2, 4)
    terminals = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="CUDA"):
        TritonScan.Config().make()(torch.randn(2, 3, 12), inputs, state, terminals)
    with pytest.raises(ValueError, match="CUDA"):
        TritonScan.Config().make().step(torch.randn(2, 12), inputs[:, 0], state)
    output, final = block(inputs, state, terminals)
    assert output.shape == inputs.shape
    assert final.shape == state.shape


def test_scan_private_metadata_paths() -> None:
    decay = torch.full((2, 3, 4), 0.5)
    result = _scan_affine(decay, innovation=decay, initial=torch.zeros(2, 4))
    assert result.shape == decay.shape
    assert not _runs_triton(decay)
    assert _terminals_view(torch.ones(2, 3, dtype=torch.bool)).dtype == torch.uint8
    with pytest.raises(ValueError, match="contiguous"):
        _check_layout(torch.zeros(2, 3), torch.zeros(3, 2).t())


def test_min_gru_has_gradient_through_inputs_and_parameters() -> None:
    model = MinGRU(input_channels=3, channels=4, layers=2)
    inputs = torch.randn(2, 4, 3, requires_grad=True)
    outputs, _ = model(inputs)
    outputs.square().mean().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    ("batch", "time", "channels", "layers"),
    [
        (1, 1, 3, 1),
        (2, 5, 4, 2),
        (3, 17, 6, 3),
        (4, 64, 8, 2),
    ],
)
def test_min_gru_vectorized_scan_matches_sequential_reference(
    batch: int,
    time: int,
    channels: int,
    layers: int,
) -> None:
    """The doubling scan must reproduce the loop it replaced, resets included."""
    torch.manual_seed(0)
    model = MinGRU(input_channels=channels, channels=channels, layers=layers)
    inputs = torch.randn(batch, time, channels)
    reset = torch.rand(batch, time) < 0.3
    state = torch.randn(layers, batch, channels)

    outputs, final = model(inputs, state=state, reset=reset)
    expected_outputs, expected_final = _sequential_min_gru(
        model,
        inputs=inputs,
        state=state,
        reset=reset,
    )

    torch.testing.assert_close(outputs, expected_outputs, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(final, expected_final, atol=1e-5, rtol=1e-5)


def test_min_gru_forced_fp32_ignores_an_ambient_bf16_autocast() -> None:
    """MinGRU's carry must not degrade under a caller's bf16 autocast region."""
    torch.manual_seed(1)
    model = MinGRU(input_channels=4, channels=4, layers=2)
    inputs = torch.randn(2, 6, 4)
    reset = torch.rand(2, 6) < 0.3

    plain_outputs, plain_final = model(inputs, reset=reset)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_outputs, autocast_final = model(inputs, reset=reset)

    assert autocast_outputs.dtype == torch.float32
    assert autocast_final.dtype == torch.float32
    torch.testing.assert_close(autocast_outputs, plain_outputs, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(autocast_final, plain_final, atol=1e-6, rtol=1e-6)


def test_forward_shapes_and_dtypes_follow_the_inputs() -> None:
    combined, inputs, initial, terminals = _inputs()
    result = TorchScan.Config().make()(combined, inputs, initial, terminals)
    assert result.outputs.shape == inputs.shape
    assert result.outputs.dtype == torch.bfloat16
    assert result.final.shape == initial.shape
    assert result.final.dtype == torch.bfloat16
    assert result.states.shape == inputs.shape
    assert result.states.dtype == torch.bfloat16


def test_a_sequence_is_the_step_applied_repeatedly() -> None:
    combined, inputs, initial, terminals = _inputs()
    scan = TorchScan.Config().make()
    whole = scan(combined, inputs, initial, terminals)
    state = initial
    for time in range(inputs.shape[1]):
        step = scan(
            combined[:, time : time + 1],
            inputs[:, time : time + 1],
            state,
            terminals[:, time : time + 1],
        )
        assert torch.equal(step.outputs[:, 0], whole.outputs[:, time])
        assert torch.equal(step.states[:, 0], whole.states[:, time])
        state = step.final
    assert torch.equal(state, whole.final)


def test_a_reset_discards_the_carry_before_the_step() -> None:
    combined, inputs, initial, terminals = _inputs(batch=3, time=5, width=4)
    scan = TorchScan.Config().make()
    with_reset = scan(combined, inputs, initial, terminals)
    # Row 0 resets at t=0, so it must match a zero initial state with no reset.
    fresh = scan(
        combined[0:1],
        inputs[0:1],
        torch.zeros_like(initial[0:1]),
        torch.zeros_like(terminals[0:1]),
    )
    assert torch.equal(with_reset.outputs[0], fresh.outputs[0])
    assert torch.equal(with_reset.states[0, 0], torch.zeros_like(initial[0]))
    # Row 2 never resets, so a nonzero carry must matter.
    other = scan(combined[2:3], inputs[2:3], initial[2:3] + 1, terminals[2:3])
    assert not torch.equal(with_reset.outputs[2], other.outputs[0])


def test_the_carry_is_rounded_to_the_state_dtype_between_steps() -> None:
    combined, inputs, initial, terminals = _inputs(time=2)
    scan = TorchScan.Config().make()
    rounded = scan(combined, inputs, initial, terminals)
    # The same fp32 arithmetic from the same bf16 state, with an fp32 carry:
    # the bf16 run's stored state after step 0 must be that carry rounded once.
    exact = scan(
        combined.float(),
        inputs.float(),
        initial.float(),
        terminals.float(),
    )
    assert torch.equal(rounded.states[:, 1], exact.states[:, 1].to(torch.bfloat16))
    assert not torch.equal(
        exact.states[:, 1],
        exact.states[:, 1].to(torch.bfloat16).float(),
    )


def test_an_fp32_carry_keeps_the_updates_a_bf16_carry_rounds_away() -> None:
    """A slow unit decays in an fp32 carry over bf16 gates and freezes in a bf16 one."""
    combined, inputs, terminals, expected = _slow_unit(time=64)
    scan = TorchScan.Config().make()
    exact = scan(combined, inputs, torch.ones(2, 3), terminals)
    frozen = scan(combined, inputs, torch.ones(2, 3, dtype=torch.bfloat16), terminals)
    assert exact.states.dtype == exact.final.dtype == torch.float32
    # A step moves the carry by 2^-13 at most, 12 times this tolerance.
    torch.testing.assert_close(
        exact.states[0, :, 0].double(),
        expected[:-1],
        rtol=0,
        atol=1e-5,
    )
    torch.testing.assert_close(
        exact.final[0, 0].double(),
        expected[-1],
        rtol=0,
        atol=1e-5,
    )
    assert torch.equal(frozen.states, torch.ones_like(frozen.states))
    assert torch.equal(frozen.final, torch.ones_like(frozen.final))


def test_backward_agrees_with_autograd_in_fp32() -> None:
    combined, inputs, initial, terminals = _inputs(dtype=torch.float32)
    scan = TorchScan.Config().make()
    forward = scan(combined, inputs, initial, terminals)
    grad_outputs = torch.randn_like(forward.outputs)
    backward = scan.backward(combined, inputs, forward.states, terminals, grad_outputs)

    leaves = (
        combined.double().requires_grad_(),
        inputs.double().requires_grad_(),
        initial.double().requires_grad_(),
    )
    outputs, _ = _reference(*leaves, terminals.double())
    outputs.backward(grad_outputs.double())
    for ours, reference in zip(
        (backward.grad_combined, backward.grad_inputs, backward.grad_initial),
        leaves,
        strict=True,
    ):
        assert reference.grad is not None
        torch.testing.assert_close(
            ours.double(),
            reference.grad,
            rtol=1e-4,
            atol=1e-5,
        )


def test_the_gradient_is_cut_at_a_reset() -> None:
    combined, inputs, initial, terminals = _inputs(
        batch=3,
        time=5,
        width=4,
        dtype=torch.float32,
    )
    scan = TorchScan.Config().make()
    forward = scan(combined, inputs, initial, terminals)
    backward = scan.backward(
        combined,
        inputs,
        forward.states,
        terminals,
        torch.ones_like(forward.outputs),
    )
    # Row 0 resets at t=0, so nothing reaches its initial carry. Row 1 resets
    # at the last step only, so the earlier steps still do; row 2 never resets.
    assert torch.equal(backward.grad_initial[0], torch.zeros_like(initial[0]))
    assert not torch.equal(backward.grad_initial[1], torch.zeros_like(initial[1]))
    assert not torch.equal(backward.grad_initial[2], torch.zeros_like(initial[2]))


def test_backward_output_dtypes_follow_the_inputs() -> None:
    combined, inputs, initial, terminals = _inputs()
    scan = TorchScan.Config().make()
    forward = scan(combined, inputs, initial, terminals)
    backward = scan.backward(
        combined,
        inputs,
        forward.states,
        terminals,
        torch.ones_like(forward.outputs),
    )
    assert backward.grad_combined.dtype == torch.bfloat16
    assert backward.grad_combined.shape == combined.shape
    assert backward.grad_inputs.dtype == torch.bfloat16
    assert backward.grad_inputs.shape == inputs.shape
    assert backward.grad_initial.dtype == torch.bfloat16
    assert backward.grad_initial.shape == initial.shape


def test_the_step_is_the_scan_at_one_time_step() -> None:
    combined, inputs, initial, terminals = _inputs(time=1)
    scan = TorchScan.Config().make()
    whole = scan(combined, inputs, initial, torch.zeros_like(terminals))
    outputs, state = scan.step(combined[:, 0], inputs[:, 0], initial)
    assert torch.equal(state, whole.final)
    assert torch.equal(outputs, whole.outputs[:, 0])
    assert outputs.dtype == inputs.dtype
    assert state.dtype == initial.dtype


def test_the_step_advances_a_carry_in_place() -> None:
    combined, inputs, initial, _ = _inputs(time=1)
    scan = TorchScan.Config().make()
    expected = scan.step(combined[:, 0], inputs[:, 0], initial)
    carry = initial.clone()
    outputs, state = scan.step(combined[:, 0], inputs[:, 0], carry, carry=carry)
    assert state is carry
    assert torch.equal(carry, expected[1])
    assert torch.equal(outputs, expected[0])


@pytest.mark.parametrize("time", [1, 2])
def test_short_sequences_run(time: int) -> None:
    combined, inputs, initial, terminals = _inputs(time=time)
    result = TorchScan.Config().make()(combined, inputs, initial, terminals)
    assert result.outputs.shape[1] == time


@pytest.mark.gpu_triton
@pytest.mark.parametrize(
    ("batch", "time", "width"),
    [(128, 256, 1024), (512, 1, 1024), (3, 5, 24)],
    ids=("learner", "step", "ragged"),
)
def test_the_triton_scan_matches_the_torch_reference(
    batch: int,
    time: int,
    width: int,
) -> None:
    combined, inputs, initial, terminals = _cuda_inputs(
        batch=batch,
        time=time,
        width=width,
    )
    # Triton's sigmoid and its contracted lerp and highway round a few fp32 ulp
    # from torch's. Where that flips a bf16 rounding of the carry and the result
    # then cancels, an element moves by one bf16 ulp of the values' scale (inputs
    # up to 16: 2^-4). Measured at the learner shape on an H200: at most 2^-5,
    # and 5 of 33.5M outputs and 30 of 101M gate gradients beyond torch's
    # default bf16 tolerance.
    torch_scan = TorchScan.Config().make()
    triton_scan = TritonScan.Config().make()
    expected = torch_scan(combined, inputs, initial, terminals)
    actual = triton_scan(combined, inputs, initial, terminals)
    grad_outputs = torch.randn_like(inputs, dtype=torch.float32).bfloat16()
    expected_backward = torch_scan.backward(
        combined,
        inputs,
        expected.states,
        terminals,
        grad_outputs,
    )
    actual_backward = triton_scan.backward(
        combined,
        inputs,
        actual.states,
        terminals,
        grad_outputs,
    )
    for ours, theirs in (
        (actual.states, expected.states),
        (actual.outputs, expected.outputs),
        (actual.final, expected.final),
        (actual_backward.grad_inputs, expected_backward.grad_inputs),
        (actual_backward.grad_combined, expected_backward.grad_combined),
        (actual_backward.grad_initial, expected_backward.grad_initial),
    ):
        torch.testing.assert_close(ours, theirs, rtol=1.6e-2, atol=2**-4)


@pytest.mark.gpu_triton
def test_the_triton_step_matches_the_torch_step() -> None:
    combined, inputs, initial, _ = _cuda_inputs(batch=512, time=1, width=1024)
    expected = TorchScan.Config().make().step(combined[:, 0], inputs[:, 0], initial)
    actual = TritonScan.Config().make().step(combined[:, 0], inputs[:, 0], initial)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    carry = initial.clone()
    in_place = (
        TritonScan.Config()
        .make()
        .step(
            combined[:, 0],
            inputs[:, 0],
            carry,
            carry=carry,
        )
    )
    assert in_place[1] is carry
    assert torch.equal(carry, actual[1])
    assert torch.equal(in_place[0], actual[0])


@pytest.mark.gpu_triton
def test_the_triton_scan_accepts_bool_terminals() -> None:
    combined, inputs, initial, terminals = _cuda_inputs(batch=4, time=3, width=16)
    scan = TritonScan.Config().make()
    assert torch.equal(
        scan(combined, inputs, initial, terminals != 0).outputs,
        scan(combined, inputs, initial, terminals).outputs,
    )


@pytest.mark.gpu_triton
def test_the_triton_scan_runs_the_reference_for_any_value_neither_bf16_nor_fp32() -> (
    None
):
    """The dtype rule covers every value, not the first three arguments."""
    combined, inputs, initial, terminals = _cuda_inputs(batch=2, time=2, width=8)
    reference = TorchScan.Config().make()
    scan = TritonScan.Config().make()
    wide = scan(combined.double(), inputs, initial, terminals)
    assert torch.equal(
        wide.outputs,
        reference(combined.double(), inputs, initial, terminals).outputs,
    )
    forward = scan(combined, inputs, initial, terminals)
    grad_outputs = torch.randn_like(inputs, dtype=torch.float64)
    actual = scan.backward(combined, inputs, forward.states, terminals, grad_outputs)
    expected = reference.backward(
        combined,
        inputs,
        forward.states,
        terminals,
        grad_outputs,
    )
    assert torch.equal(actual.grad_combined, expected.grad_combined)
    carry = torch.empty_like(initial, dtype=torch.float64)
    stepped = scan.step(combined[:, 0], inputs[:, 0], initial, carry=carry)
    assert stepped[1] is carry
    expected_carry = torch.empty_like(carry)
    reference.step(combined[:, 0], inputs[:, 0], initial, carry=expected_carry)
    assert torch.equal(carry, expected_carry)


@pytest.mark.gpu_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=("bf16", "fp32"))
def test_the_triton_scan_keeps_an_fp32_carry_as_the_torch_reference_does(
    dtype: torch.dtype,
) -> None:
    """An fp32 carry over bf16 or fp32 gates, forward, backward and step, on the GPU.

    The kernels keep the carry unrounded, as the reference does. Tolerances: the
    carry, and every fp32 value, to 1e-5 -- a few fp32 ulp per step, which the
    lerp's contraction keeps from growing; a bf16 value to torch's bf16
    tolerance, one rounding flip of the fp32 value it stores.
    """
    combined, inputs, initial, terminals = _cuda_inputs(batch=16, time=64, width=256)
    combined, inputs, initial = combined.to(dtype), inputs.to(dtype), initial.float()
    torch_scan = TorchScan.Config().make()
    triton_scan = _kernels_only()
    expected = torch_scan(combined, inputs, initial, terminals)
    actual = triton_scan(combined, inputs, initial, terminals)
    assert actual.states.dtype == actual.final.dtype == torch.float32
    assert actual.outputs.dtype == dtype
    assert not torch.equal(actual.states, actual.states.bfloat16().float())
    grad_outputs = torch.randn_like(inputs, dtype=torch.float32).to(dtype)
    expected_backward = torch_scan.backward(
        combined,
        inputs,
        expected.states,
        terminals,
        grad_outputs,
    )
    actual_backward = triton_scan.backward(
        combined,
        inputs,
        actual.states,
        terminals,
        grad_outputs,
    )
    assert actual_backward.grad_initial.dtype == torch.float32
    # The kernels address contiguous memory, which one step of a window is not.
    step_combined, step_inputs = combined[:, 0].contiguous(), inputs[:, 0].contiguous()
    carry = initial.clone()
    stepped = triton_scan.step(step_combined, step_inputs, carry, carry=carry)
    reference_step = torch_scan.step(step_combined, step_inputs, initial)
    for ours, theirs in (
        (actual.states, expected.states),
        (actual.outputs, expected.outputs),
        (actual.final, expected.final),
        (actual_backward.grad_inputs, expected_backward.grad_inputs),
        (actual_backward.grad_combined, expected_backward.grad_combined),
        (actual_backward.grad_initial, expected_backward.grad_initial),
        (stepped[0], reference_step[0]),
        (carry, reference_step[1]),
    ):
        if ours.dtype == torch.bfloat16:
            torch.testing.assert_close(ours, theirs)
        else:
            torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu_triton
def test_the_triton_kernels_keep_a_slow_units_updates_in_an_fp32_carry() -> None:
    """The scan and the rollout's step decay a slow unit that a bf16 carry freezes."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    combined, inputs, terminals, expected = (
        value.cuda() for value in _slow_unit(time=64)
    )
    scan = _kernels_only()
    ones = torch.ones(2, 3, device="cuda")
    exact = scan(combined, inputs, ones, terminals)
    frozen = scan(combined, inputs, ones.bfloat16(), terminals)
    torch.testing.assert_close(
        exact.states[0, :, 0].double(),
        expected[:-1],
        rtol=0,
        atol=1e-5,
    )
    torch.testing.assert_close(
        exact.final[0, 0].double(),
        expected[-1],
        rtol=0,
        atol=1e-5,
    )
    assert torch.equal(frozen.final, ones.bfloat16())
    carry = ones.clone()
    for time in range(combined.shape[1]):
        scan.step(
            combined[:, time].contiguous(),
            inputs[:, time].contiguous(),
            carry,
            carry=carry,
        )
    torch.testing.assert_close(carry[0, 0].double(), expected[-1], rtol=0, atol=1e-5)


def test_the_triton_scan_refuses_cpu_tensors_rather_than_changing_their_bits() -> None:
    combined, inputs, initial, terminals = _inputs()
    scan = TritonScan.Config().make()
    forward = TorchScan.Config().make()(combined, inputs, initial, terminals)
    with pytest.raises(ValueError, match="TorchScan"):
        scan(combined, inputs, initial, terminals)
    with pytest.raises(ValueError, match="TorchScan"):
        scan.step(combined[:, 0], inputs[:, 0], initial)
    with pytest.raises(ValueError, match="TorchScan"):
        scan.backward(combined, inputs, forward.states, terminals, forward.outputs)


@pytest.mark.parametrize("field", ["block", "num_warps"])
def test_the_triton_scan_refuses_a_geometry_triton_cannot_tile(field: str) -> None:
    config = TritonScan.Config()
    setattr(config, field, 96)
    with pytest.raises(ValueError, match=field):
        config.make()


def test_a_blocks_gradients_agree_with_float64_autograd_in_fp32() -> None:
    config = MinGRUBlock.Config()
    config.channels_hidden = 4
    block = config.make()
    _, inputs, initial, terminals = _inputs(width=4, dtype=torch.float32)
    inputs.requires_grad_()
    outputs, final = block(inputs, initial, terminals)
    assert not final.requires_grad
    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)

    weight = block.proj_gates.weight.detach().double().requires_grad_()
    leaves = (inputs.detach().double().requires_grad_(), weight)
    combined = leaves[0] @ weight.t()
    expected, _ = _reference(
        combined,
        leaves[0],
        initial.double(),
        terminals.double(),
    )
    expected.backward(grad_outputs.double())
    for ours, reference in zip(
        (inputs.grad, block.proj_gates.weight.grad),
        leaves,
        strict=True,
    ):
        assert ours is not None
        assert reference.grad is not None
        torch.testing.assert_close(ours.double(), reference.grad, rtol=1e-4, atol=1e-5)


def test_a_blocks_scan_gradient_is_the_scans_own_backward() -> None:
    """Autograd reaches the gates and the highway input through ``Scan.backward``."""
    config = MinGRUBlock.Config()
    config.channels_hidden = 4
    config.dtype = torch.bfloat16
    block = config.make()
    _, inputs, initial, terminals = _inputs(width=4)
    inputs.requires_grad_()
    initial.requires_grad_()
    outputs, _ = block(inputs, initial, terminals)
    generator = torch.Generator().manual_seed(1)
    grad_outputs = torch.randn(outputs.shape, generator=generator).bfloat16()
    outputs.backward(grad_outputs)

    with torch.no_grad():
        combined = block.proj_gates(inputs)
        forward = block.scan(combined, inputs, initial, terminals)
        expected = block.scan.backward(
            combined,
            inputs,
            forward.states,
            terminals,
            grad_outputs,
        )
        weight = block.proj_gates.weight
        grad_combined = expected.grad_combined.flatten(0, 1)
        grad_weight = torch.mm(grad_combined.t(), inputs.flatten(0, 1))
        grad_inputs = torch.mm(grad_combined, weight) + expected.grad_inputs.flatten(
            0,
            1,
        )
    assert torch.equal(outputs.detach(), forward.outputs)
    assert block.proj_gates.weight.grad is not None
    assert torch.equal(block.proj_gates.weight.grad, grad_weight)
    assert inputs.grad is not None
    assert torch.equal(inputs.grad, grad_inputs.reshape(inputs.shape))
    assert initial.grad is not None
    assert torch.equal(initial.grad, expected.grad_initial)


def test_a_blocks_step_is_its_forward_at_one_time_step() -> None:
    config = MinGRUBlock.Config()
    config.channels_hidden = 4
    config.dtype = torch.bfloat16
    block = config.make()
    _, inputs, initial, _ = _inputs(width=4, time=1)
    with torch.no_grad():
        whole, final = block(inputs, initial, torch.zeros(inputs.shape[:2]))
        outputs, state = block.step(inputs[:, 0], initial)
    assert torch.equal(outputs, whole[:, 0])
    assert torch.equal(state, final)


def test_min_gru_block_config_pprint() -> None:
    config = MinGRUBlock.Config()
    config.channels_hidden = 4
    assert_pprint_golden(test_file=__file__, name="min_gru_block", config=config)


def test_min_gru_block_bfb() -> None:
    config = MinGRUBlock.Config()
    config.channels_hidden = 2
    _, inputs, initial, terminals = _inputs(dtype=torch.float32)
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="min_gru_block",
        build_module=config.make,
        build_input=lambda: (inputs, initial, terminals),
        seed=0,
        run=lambda module, values: (
            module(*values)[0]
            if isinstance(module, MinGRUBlock)
            else (_ for _ in ()).throw(TypeError(type(module).__name__))
        ),
    )


def _sequential_min_gru(
    model: MinGRU,
    *,
    inputs: torch.Tensor,
    state: torch.Tensor,
    reset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reimplement the pre-scan per-step loop as an independent oracle."""
    time = inputs.shape[1]
    value = model.input_projection(inputs.float())
    carries: list[torch.Tensor] = []
    for layer_index, layer in enumerate(model.layers):
        combined = layer(value).float()
        hidden, gate, highway = combined.chunk(3, dim=-1)
        carry = state[layer_index].float()
        outputs: list[torch.Tensor] = []
        for step in range(time):
            carry = torch.where(reset[:, step, None], 0.0, carry)
            candidate = torch.where(
                hidden[:, step] >= 0,
                hidden[:, step] + 0.5,
                torch.sigmoid(hidden[:, step]),
            )
            carry = torch.lerp(carry, candidate, torch.sigmoid(gate[:, step]))
            strength = torch.sigmoid(highway[:, step])
            outputs.append((1 - strength) * value[:, step] + strength * carry)
        value = torch.stack(outputs, dim=1)
        carries.append(carry)
    return value, torch.stack(carries)


def _inputs(
    *,
    batch: int = 2,
    time: int = 2,
    width: int = 2,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    combined = torch.randn(batch, time, 3 * width, generator=generator) * 3
    inputs = torch.randn(batch, time, width, generator=generator)
    initial = torch.randn(batch, width, generator=generator)
    terminals = torch.zeros(batch, time)
    terminals[0, 0] = 1.0
    terminals[1, -1] = 1.0
    return combined.to(dtype), inputs.to(dtype), initial.to(dtype), terminals.to(dtype)


# Gate -8.3125 is exact in bf16 and gives z = 2.45e-4, a time constant of 4,000 steps; a
# zero hidden gives the candidate 0.5. From a carry of 1 each step moves it by
# ``z * (0.5 - h)``, at most 2^-13, under half a bf16 ulp just below 1 (2^-9). The exact
# carries are float64, ``[time + 1]``: each step's, then the final one.
def _slow_unit(*, time: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return a slow unit's bf16 gates, inputs and terminals, and its exact carries."""
    combined = torch.zeros(2, time, 9, dtype=torch.bfloat16)
    combined[..., 3:6] = -8.3125
    z = torch.sigmoid(torch.tensor(-8.3125, dtype=torch.float64))
    expected = 0.5 + 0.5 * (1 - z) ** torch.arange(time + 1, dtype=torch.float64)
    return (
        combined,
        torch.zeros(2, time, 3, dtype=torch.bfloat16),
        torch.zeros(2, time),
        expected,
    )


def _reference(
    combined: Tensor,
    inputs: Tensor,
    initial: Tensor,
    terminals: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run the same recurrence in float64 through autograd."""
    hidden, gate, highway = combined.chunk(3, dim=-1)
    state = initial
    outputs: list[Tensor] = []
    for time in range(inputs.shape[1]):
        state = torch.where(terminals[:, time, None] != 0, 0.0, state)
        candidate = torch.where(
            hidden[:, time] >= 0,
            hidden[:, time] + 0.5,
            torch.sigmoid(hidden[:, time]),
        )
        state = torch.lerp(state, candidate, torch.sigmoid(gate[:, time]))
        strength = torch.sigmoid(highway[:, time])
        outputs.append(strength * state + (1 - strength) * inputs[:, time])
    return torch.stack(outputs, dim=1), state


def _cuda_inputs(
    *,
    batch: int,
    time: int,
    width: int,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Production-shaped bf16 inputs with sparse resets, on the GPU."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    combined = torch.randn(batch, time, 3 * width, device="cuda", generator=generator)
    inputs = torch.randn(batch, time, width, device="cuda", generator=generator)
    initial = torch.randn(batch, width, device="cuda", generator=generator)
    terminals = torch.rand(batch, time, device="cuda", generator=generator) < 0.01
    return (
        (combined * 3).bfloat16(),
        inputs.bfloat16(),
        initial.bfloat16(),
        terminals.bfloat16(),
    )


class _NoFallback(TorchScan):
    """A reference that refuses every call, so a test can tell the kernels ran."""

    @override
    def __call__(
        self,
        combined: Tensor,
        inputs: Tensor,
        initial: Tensor,
        terminals: Tensor,
    ) -> ScanForward:
        raise AssertionError("the scan fell back to its torch reference")

    @override
    def step(
        self,
        combined: Tensor,
        inputs: Tensor,
        state: Tensor,
        *,
        carry: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise AssertionError("the step fell back to its torch reference")

    @override
    def backward(
        self,
        combined: Tensor,
        inputs: Tensor,
        states: Tensor,
        terminals: Tensor,
        grad_outputs: Tensor,
    ) -> ScanBackward:
        raise AssertionError("the backward fell back to its torch reference")


def _kernels_only() -> TritonScan:
    """Return a TritonScan whose every call must run its kernels."""
    scan = TritonScan.Config().make()
    scan.reference = _NoFallback(TorchScan.Config())
    return scan


def test_triton_scan_uses_fake_host_launches_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The kernels are faked, but the host launch still sizes its grid with
    # ``triton.cdiv``; Triton ships Linux wheels only.
    pytest.importorskip("triton")
    monkeypatch.setattr(min_gru, "_require_cuda", _allow_cpu_scan)
    monkeypatch.setattr(min_gru, "_runs_triton", _force_triton)
    monkeypatch.setattr(min_gru, "_kernels", _fake_kernels)
    scan = TritonScan.Config(block=4, num_warps=2).make()
    combined, inputs, initial, terminals = _inputs(batch=2, time=3, width=4)
    result = scan(combined, inputs, initial, terminals)
    assert result.outputs.shape == inputs.shape
    output, state = scan.step(
        combined[:, 0].contiguous(),
        inputs[:, 0].contiguous(),
        initial,
    )
    assert output.shape == inputs[:, 0].shape
    assert state.shape == initial.shape
    backward = scan.backward(
        combined,
        inputs,
        result.states,
        terminals,
        torch.ones_like(result.outputs),
    )
    assert backward.grad_combined.shape == combined.shape


def test_triton_scan_uses_torch_reference_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(min_gru, "_require_cuda", _allow_cpu_scan)
    scan = TritonScan.Config(block=4, num_warps=2).make()
    combined, inputs, initial, terminals = _inputs(batch=2, time=3, width=4)
    result = scan(combined, inputs, initial, terminals)
    assert result.outputs.shape == inputs.shape
    output, state = scan.step(
        combined[:, 0].contiguous(),
        inputs[:, 0].contiguous(),
        initial,
    )
    assert output.shape == inputs[:, 0].shape
    assert state.shape == initial.shape
    backward = scan.backward(
        combined,
        inputs,
        result.states,
        terminals,
        torch.ones_like(result.outputs),
    )
    assert backward.grad_combined.shape == combined.shape


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
