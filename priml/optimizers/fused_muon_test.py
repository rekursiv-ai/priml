"""Unit tests of FusedMuon's stages, its step, and its fused kernels.

Each stage is checked on inputs small enough to work out by hand, on the CPU.
On CUDA the Triton kernels are held to the torch stages bit for bit. A golden
freezes three steps of a five-weight stack from fp32 masters, run as the torch
reference inside ``host_agnostic_numerics`` so every CPU computes its bits.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, cast

import hashlib
import struct

import pytest
import torch

from priml.optimizers.fused_muon import (
    FusedMuon,
    apply_update,
    aspect_scale,
    clip_coefficient,
    nesterov,
    newton_schulz,
    normalize,
)
from priml.optimizers.lr import lr_scale
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_text_golden


if TYPE_CHECKING:
    from torch import Tensor


def _parameter(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.nn.Parameter:
    generator = torch.Generator().manual_seed(sum(shape))
    return torch.nn.Parameter(torch.randn(shape, generator=generator).to(dtype))


def test_the_config_makes_a_constructor_awaiting_parameters() -> None:
    weight = _parameter(4, 2)
    optimizer = FusedMuon.Config().make()([weight])
    assert isinstance(optimizer, FusedMuon)
    assert optimizer.param_groups[0]["lr"] == 0.015
    assert optimizer.param_groups[0]["momentum"] == 0.95
    assert optimizer.max_grad_norm == 1.5
    assert len(optimizer.ns_coefficients) == 5


def test_the_norm_spans_every_tensor_and_returns_fp32() -> None:
    optimizer = FusedMuon.Config().make()([_parameter(4, 2)])
    norm = optimizer.norm([torch.tensor([3.0], dtype=torch.bfloat16), torch.ones(4, 4)])
    assert norm.dtype == torch.float32
    assert norm.shape == ()
    assert float(norm) == 5.0
    # bf16 would round sqrt(3) to 1.734375; one matrix's norm accumulates in fp32.
    ones = torch.ones(3, dtype=torch.bfloat16)
    assert float(optimizer.norm([ones])) == float(torch.tensor(3.0).sqrt())
    # Several bf16 gradients reduce in bf16, as ``clip_grad_norm_`` reduces them.
    assert float(optimizer.norm([ones, ones])) == float(
        torch.tensor(6.0).sqrt().bfloat16(),
    )


def test_clipping_scales_a_large_gradient_and_leaves_a_small_one() -> None:
    expected = torch.tensor(1.0) / (torch.tensor(5.0) + torch.tensor(1e-6))
    assert float(clip_coefficient(torch.tensor(5.0), 1.0)) == float(expected)
    assert float(clip_coefficient(torch.tensor(0.5), 1.0)) == 1.0


def test_nesterov_keeps_fp32_momentum_and_rounds_the_update() -> None:
    gradient = torch.ones(2, dtype=torch.bfloat16)
    buffer = torch.zeros(2)
    coefficient = torch.tensor(1.0)
    update = nesterov(gradient, buffer, coefficient=coefficient, momentum=0.5)
    assert update.dtype == torch.bfloat16
    assert torch.equal(buffer, torch.ones(2))
    assert torch.equal(update, torch.full((2,), 1.5, dtype=torch.bfloat16))
    update = nesterov(gradient, buffer, coefficient=coefficient, momentum=0.5)
    assert torch.equal(buffer, torch.full((2,), 1.5))
    assert torch.equal(update, torch.full((2,), 1.75, dtype=torch.bfloat16))


def test_normalize_divides_by_the_norm_floored_at_eps() -> None:
    matrix = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
    expected = torch.tensor([[0.6015625, 0.80078125]], dtype=torch.bfloat16)
    assert torch.equal(normalize(matrix, norm=torch.tensor(5.0), eps=1e-7), expected)
    zeros = torch.zeros(2, 3, dtype=torch.bfloat16)
    assert torch.equal(normalize(zeros, norm=torch.tensor(0.0), eps=1e-7), zeros)


@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])
def test_newton_schulz_moves_a_matrix_toward_its_polar_factor(
    shape: tuple[int, int],
) -> None:
    generator = torch.Generator().manual_seed(3)
    draws = torch.randn(shape, generator=generator)
    matrix = normalize(draws, norm=torch.linalg.vector_norm(draws), eps=1e-7)
    result = newton_schulz(matrix, FusedMuon.Config().ns_coefficients)
    singular = torch.linalg.svdvals(result)
    assert float(singular.min()) > 0.5
    assert float(singular.max()) < 1.5


def test_the_aspect_scale_scales_tall_matrices_only() -> None:
    assert aspect_scale((8, 2)) == 2.0
    assert aspect_scale((2, 8)) == 1.0
    assert aspect_scale((3072, 1024)) == 3**0.5


def test_apply_update_scales_rounds_steps_and_casts() -> None:
    update = torch.ones(8, 2, dtype=torch.bfloat16)
    master = torch.zeros(8, 2)
    parameter = torch.empty(8, 2, dtype=torch.bfloat16)
    apply_update(parameter, master, update, lr=-0.5, scale=aspect_scale((8, 2)))
    assert torch.equal(master, torch.ones(8, 2))
    assert torch.equal(parameter, master.bfloat16())
    # The scaled update is rounded to bf16 before the step: sqrt(3) in bf16.
    apply_update(parameter, master, update, lr=-1.0, scale=aspect_scale((3072, 1024)))
    scaled = torch.tensor(3.0).sqrt().bfloat16().float()
    assert torch.equal(master, 1.0 + scaled.expand(8, 2))


def test_a_step_updates_the_masters_and_rounds_the_parameters() -> None:
    matrix, vector = _parameter(4, 2), _parameter(3)
    before = [matrix.detach().float().clone(), vector.detach().float().clone()]
    config = FusedMuon.Config()
    config.lr = 0.5
    config.momentum = 0.0
    config.max_grad_norm = 1e6
    optimizer = config.make()([matrix, vector])
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    optimizer.step()
    masters = optimizer.master_weights
    assert [master.dtype for master in masters] == [torch.float32, torch.float32]
    assert [buffer.dtype for buffer in optimizer.momentum_buffers] == [
        torch.float32,
        torch.float32,
    ]
    # A vector's update is its clipped Nesterov gradient; a matrix's is
    # orthogonalized, so its magnitude changed.
    assert torch.equal(masters[1], before[1] - 0.5)
    assert not torch.equal(masters[0], before[0] - 0.5)
    for parameter, master in zip((matrix, vector), masters, strict=True):
        assert torch.equal(parameter, master.to(torch.bfloat16))
    assert torch.equal(vector.grad, torch.ones_like(vector))


def test_a_tensor_rate_steps_as_its_float_does() -> None:
    rate = _rate(7)
    by_float = _two_steps(rate)
    by_tensor = _two_steps(torch.tensor(rate))
    for ours, theirs in zip(
        by_tensor.master_weights,
        by_float.master_weights,
        strict=True,
    ):
        assert torch.equal(ours, theirs)


@pytest.mark.gpu_triton
def test_a_captured_step_reads_its_tensor_rate_at_every_replay() -> None:
    if not torch.cuda.is_available():
        pytest.skip("capture needs a CUDA device")
    rates = [_rate(epoch) for epoch in (0, 9, 20)]
    eager = _two_steps(rates[0], device="cuda")
    rate_tensor = torch.tensor(rates[0], device="cuda")
    captured = _two_steps(rate_tensor, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured.step()
    for rate in rates[1:]:
        eager.param_groups[0]["lr"] = rate
        eager.step()
        rate_tensor.fill_(rate)
        graph.replay()
        for ours, theirs in zip(
            captured.master_weights,
            eager.master_weights,
            strict=True,
        ):
            assert torch.equal(ours, theirs)


@pytest.mark.gpu_triton
def test_the_fused_cuda_step_matches_the_torch_stages() -> None:
    """Tall, wide, square and vector updates, float then tensor rate, two steps.

    The gradients span eight decades, so every rounding the kernels make shows
    in the low bits. The Nesterov kernel issues the torch stage's operations,
    so the momentum is equal; the master update fuses its product into the
    subtraction and the normalization divides with Triton's fp32 division, so
    the masters agree to a few fp32 ulp, amplified through Newton-Schulz.
    """
    if not torch.cuda.is_available():
        pytest.skip("the Triton kernels need a CUDA device")
    config = FusedMuon.Config()
    shapes = ((300, 70), (70, 300), (64, 64), (1000,))
    generator = torch.Generator().manual_seed(5)
    parameters = [
        torch.nn.Parameter(torch.randn(shape, generator=generator).bfloat16().cuda())
        for shape in shapes
    ]
    optimizer = config.make()(parameters)
    masters = [parameter.detach().float().clone() for parameter in parameters]
    momenta = [
        torch.randn(shape, generator=generator).cuda() * 0.01 for shape in shapes
    ]
    for target, source in zip(optimizer.momentum_buffers, momenta, strict=True):
        target.copy_(source)
    momentum = float(torch.tensor(config.momentum, dtype=torch.float32))
    rates = (0.003, torch.tensor(0.002, device="cuda"))
    for rate in rates:
        gradients = [
            (
                torch.randn(shape, generator=generator)
                * 10.0 ** torch.randint(-6, 2, shape, generator=generator)
            )
            .bfloat16()
            .cuda()
            for shape in shapes
        ]
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = gradient
        optimizer.param_groups[0]["lr"] = rate
        optimizer.step()
        coefficient = clip_coefficient(optimizer.norm(gradients), config.max_grad_norm)
        for gradient, master, buffer in zip(gradients, masters, momenta, strict=True):
            update = nesterov(
                gradient,
                buffer,
                coefficient=coefficient,
                momentum=momentum,
            )
            scale = 1.0
            if update.ndim >= 2:
                update = newton_schulz(
                    normalize(update, norm=optimizer.norm([update]), eps=config.eps),
                    config.ns_coefficients,
                )
                scale = aspect_scale(update.shape)
            apply_update(
                torch.empty_like(update),
                master,
                update,
                lr=rate,
                scale=scale,
            )
        for ours, theirs in zip(optimizer.master_weights, masters, strict=True):
            torch.testing.assert_close(ours, theirs, rtol=1e-6, atol=1e-5)
        for ours, theirs in zip(optimizer.momentum_buffers, momenta, strict=True):
            assert torch.equal(ours, theirs)
        for parameter, master in zip(parameters, optimizer.master_weights, strict=True):
            assert torch.equal(parameter, master.bfloat16())


def test_a_missing_gradient_is_an_error() -> None:
    matrix, vector = _parameter(4, 2), _parameter(3)
    optimizer = FusedMuon.Config().make()([matrix, vector])
    matrix.grad = torch.ones_like(matrix)
    with pytest.raises(ValueError, match="gradient"):
        optimizer.step()


def test_the_state_round_trips_fp32_masters_for_bf16_parameters() -> None:
    weight = _parameter(4, 2)
    optimizer = FusedMuon.Config().make()([weight])
    weight.grad = torch.ones_like(weight)
    optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    twin = torch.nn.Parameter(weight.detach().clone())
    restored = FusedMuon.Config().make()([twin])
    restored.load_state_dict(saved)
    assert torch.equal(restored.master_weights[0], optimizer.master_weights[0])
    assert torch.equal(restored.momentum_buffers[0], optimizer.momentum_buffers[0])
    assert restored.master_weights[0].dtype == torch.float32


def test_a_tiny_stack_matches_its_golden(request: pytest.FixtureRequest) -> None:
    """A two-layer recurrent policy's five weights: three steps from fp32 masters.

    The masters are portable draws that the bf16 parameters only round, as a
    checkpoint's are, and reach the optimizer through ``load_state_dict``. The
    optimizer takes the weights in another order than they were drawn in, as a
    model whose global norm runs in its own order hands them over. The first
    and last steps' gradients are clipped and the second's are not.
    """
    shapes = {
        "embedding.weight": (154, 2),
        "proj_in.weight": (8, 249),
        "blocks.0.proj_gates.weight": (24, 8),
        "blocks.1.proj_gates.weight": (24, 8),
        "proj_out.weight": (44, 8),
    }
    generator = torch.Generator().manual_seed(0)
    masters = {
        name: _portable_uniform(shape, bound=shape[-1] ** -0.5, generator=generator)
        for name, shape in shapes.items()
    }
    parameters = {
        name: torch.nn.Parameter(master.bfloat16()) for name, master in masters.items()
    }
    order = [
        "embedding.weight",
        "proj_in.weight",
        "proj_out.weight",
        "blocks.0.proj_gates.weight",
        "blocks.1.proj_gates.weight",
    ]
    config = FusedMuon.Config()
    config.lr = 0.00472887093
    config.momentum = 0.930526733
    config.max_grad_norm = 0.818616688
    optimizer = config.make()([parameters[name] for name in order])
    state = {
        index: {
            "master_weight": masters[name].clone(),
            "momentum_buffer": torch.zeros_like(masters[name]),
        }
        for index, name in enumerate(order)
    }
    optimizer.load_state_dict({**optimizer.state_dict(), "state": state})
    generator = torch.Generator().manual_seed(1)
    steps = [
        {
            name: _portable_uniform(shape, bound=scale, generator=generator).bfloat16()
            for name, shape in shapes.items()
        }
        for scale in (2**-4, 2**-7, 2**-4)
    ]
    with host_agnostic_numerics():
        lines = _step_entries(optimizer, parameters, order=order, steps=steps)
    assert_text_golden(
        request,
        test_file=__file__,
        name="fused_muon_tiny",
        rendered="\n".join(lines),
    )


def _two_steps(rate: float | torch.Tensor, *, device: str = "cpu") -> FusedMuon:
    """Two steps of random gradients at ``rate``, a float or a tensor."""
    parameters = [
        torch.nn.Parameter(_parameter(*shape).detach().to(device))
        for shape in ((6, 4), (5,))
    ]
    optimizer = FusedMuon.Config().make()(parameters)
    optimizer.param_groups[0]["lr"] = rate
    generator = torch.Generator().manual_seed(7)
    for _ in range(2):
        for parameter in parameters:
            parameter.grad = torch.randn(parameter.shape, generator=generator).to(
                device=device,
                dtype=parameter.dtype,
            )
        optimizer.step()
    return optimizer


def _rate(epoch: int) -> float:
    """Return epoch ``epoch`` of a 30-epoch cosine, rounded to fp32 as a group holds it."""
    rate = torch.tensor(0.00472887093 * lr_scale(epoch, 30), dtype=torch.float32)
    return float(rate)


def _step_entries(
    optimizer: FusedMuon,
    parameters: dict[str, torch.nn.Parameter],
    *,
    order: list[str],
    steps: list[dict[str, Tensor]],
) -> list[str]:
    """Step once per gradient set; digest the clip, masters, momentum and weights."""
    lines: list[str] = []
    for index, gradients in enumerate(steps):
        for name, parameter in parameters.items():
            parameter.grad = gradients[name]
        clip = clip_coefficient(
            optimizer.norm([gradients[name] for name in order]),
            optimizer.max_grad_norm,
        )
        lines += [f"step {index} clip {_fp32_entry(float(clip))}"]
        optimizer.step()
        state = cast("dict[int, dict[str, Tensor]]", optimizer.state_dict()["state"])
        for key in ("master_weight", "momentum_buffer"):
            lines += [
                f"step {index} {key} {name} {_digest(state[position][key])}"
                for position, name in enumerate(order)
            ]
        lines += [
            f"step {index} parameter {name} {_digest(parameter)}"
            for name, parameter in parameters.items()
        ]
    return lines


def _portable_uniform(
    shape: tuple[int, ...],
    *,
    bound: float,
    generator: torch.Generator,
) -> Tensor:
    """Draw fp32 ``U(-bound, bound)`` whose bits are the same on every host."""
    # ``torch.rand`` fills multiples of 2^-24 and ``2u - 1`` is exact, so only the
    # product with ``bound`` rounds. ``randn`` would not port: its transform runs a
    # vectorized ``log``/``cos`` on some hosts and libm's on others.
    return (torch.rand(shape, generator=generator) * 2 - 1) * bound


def _digest(value: Tensor) -> str:
    """Return a tensor's dtype, shape and the sha256 of its bytes."""
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
    return f"{value.dtype} {tuple(value.shape)} {digest}"


def _fp32_entry(value: float) -> str:
    """Return an fp32 value's bits, then its decimal."""
    single = struct.unpack("<f", struct.pack("<f", value))[0]
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return f"0x{bits:08x} {single!r}"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
