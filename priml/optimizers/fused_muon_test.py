"""Unit tests of FusedMuon's stages, its step, and its fused kernels.

Each stage is checked on inputs small enough to work out by hand, on the CPU.
On CUDA the Triton kernels are held to the torch stages bit for bit. A golden
freezes three steps of a five-weight stack from fp32 masters, run as the torch
reference inside ``host_agnostic_numerics`` so every CPU computes its bits.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

from torch import Tensor

import pytest
import torch

from priml.optimizers import fused_muon
from priml.optimizers.fused_muon import (
    FusedMuon,
    _group_scalars,
    apply_update,
    aspect_scale,
    clip_coefficient,
    nesterov,
    newton_schulz,
    normalize,
)
from priml.optimizers.lr import lr_scale
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_tensor_golden


if TYPE_CHECKING:
    from collections.abc import Callable


_CWD: Final = Path(__file__).resolve().parent


def _parameter(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.nn.Parameter:
    generator = torch.Generator().manual_seed(sum(shape))
    return torch.nn.Parameter(torch.randn(*shape, generator=generator).to(dtype))


class _KernelLaunch:
    def __init__(self, run: Callable[..., object]) -> None:
        self.run = run

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., object]:
        return partial(self.run, grid=grid)


def test_the_config_makes_a_constructor_awaiting_parameters() -> None:
    weight = _parameter(4, 2)
    optimizer = FusedMuon.Config().make()([weight])
    assert optimizer.param_groups[0]["lr"] == 0.015
    assert optimizer.param_groups[0]["momentum"] == 0.95
    assert optimizer.max_grad_norm == 1.5
    assert len(optimizer.ns_coefficients) == 5


def test_the_norm_spans_every_tensor_and_returns_fp32() -> None:
    optimizer = FusedMuon.Config().make()([_parameter(4, 2)])
    norm = optimizer.norm([torch.tensor([3.0], dtype=torch.bfloat16), torch.ones(4, 5)])
    assert norm.dtype == torch.float32
    assert norm.shape == ()
    assert float(norm) == pytest.approx(29.0**0.5)
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
    draws = torch.randn(*shape, generator=generator)
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


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.gpu_triton)],
)
@pytest.mark.parametrize("shape", [(5, 2, 3, 4), (25, 2, 3, 4)])
def test_a_conv_weight_steps_as_the_matrix_of_its_rows(
    shape: tuple[int, ...],
    device: str,
) -> None:
    """A 4-D weight is orthogonalized as ``[out, -1]``, as priml's ``Muon`` reshapes it.

    ``[5, 24]`` is wide and ``[25, 24]`` tall, so the aspect scale is the
    matrix's ``sqrt(25 / 24)``, not one from the weight's first two axes. On
    CUDA the fused kernels read both through the same flat memory.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("the fused kernels need a CUDA device")
    conv = torch.nn.Parameter(_parameter(*shape).detach().to(device))
    matrix = torch.nn.Parameter(conv.detach().reshape(shape[0], -1).clone())
    optimizers = [FusedMuon.Config().make()([weight]) for weight in (conv, matrix)]
    generator = torch.Generator().manual_seed(2)
    for _ in range(2):
        gradient = torch.randn(*shape, generator=generator).bfloat16().to(device)
        conv.grad = gradient
        matrix.grad = gradient.reshape(matrix.shape)
        for optimizer in optimizers:
            optimizer.step()
    ours, theirs = optimizers
    assert torch.equal(
        ours.master_weights[0].reshape(matrix.shape),
        theirs.master_weights[0],
    )
    assert torch.equal(
        ours.momentum_buffers[0].reshape(matrix.shape),
        theirs.momentum_buffers[0],
    )
    assert torch.equal(conv.detach().reshape(matrix.shape), matrix.detach())
    assert not torch.equal(conv.detach().cpu(), _parameter(*shape).detach())


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
        torch.nn.Parameter(torch.randn(*shape, generator=generator).bfloat16().cuda())
        for shape in shapes
    ]
    optimizer = config.make()(parameters)
    masters = [parameter.detach().float().clone() for parameter in parameters]
    momenta = [
        torch.randn(*shape, generator=generator).cuda() * 0.01 for shape in shapes
    ]
    for target, source in zip(optimizer.momentum_buffers, momenta, strict=True):
        target.copy_(source)
    momentum = float(torch.tensor(config.momentum, dtype=torch.float32))
    rates = (0.003, torch.tensor(0.002, device="cuda"))
    for rate in rates:
        gradients = [
            (
                torch.randn(*shape, generator=generator)
                * 10.0 ** torch.randint(-6, 2, size=shape, generator=generator)
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


def test_invalid_hyperparameters_are_rejected() -> None:
    for field_name in ("lr", "momentum", "max_grad_norm", "eps"):
        config = FusedMuon.Config()
        setattr(config, field_name, -1.0)
        with pytest.raises(ValueError, match=f"FusedMuon {field_name}"):
            config.make()([_parameter(2, 3)])


def test_step_returns_closure_result_and_accepts_tensor_rate() -> None:
    parameter = _parameter(2, 3)
    optimizer = FusedMuon.Config().make()([parameter])
    optimizer.param_groups[0]["lr"] = torch.tensor(0.01)
    parameter.grad = torch.ones_like(parameter)
    result = optimizer.step(lambda: torch.tensor(3.0))
    assert isinstance(result, torch.Tensor)
    assert result == 3
    assert optimizer.param_groups[0]["lr"].dtype == torch.float32


def test_group_scalars_rounds_python_values_and_preserves_tensor_rate() -> None:
    tensor_rate = torch.tensor(0.123456789)
    rate, momentum, eps = _group_scalars(
        {"lr": tensor_rate, "momentum": 0.123456789, "eps": 0.987654321},
    )

    assert rate is tensor_rate
    assert momentum == float(torch.tensor(0.123456789, dtype=torch.float32))
    assert eps == 0.987654321


def test_group_scalars_rounds_float_rate_and_requires_scalar_values() -> None:
    rate = 0.123456789
    result = _group_scalars({"lr": rate, "momentum": 0.5, "eps": 1e-7})
    assert result == (float(torch.tensor(rate, dtype=torch.float32)), 0.5, 1e-7)

    for name in ("lr", "momentum", "eps"):
        group: dict[str, object] = {
            "lr": rate,
            "momentum": 0.5,
            "eps": 1e-7,
        }
        group[name] = None
        with pytest.raises(TypeError):
            _group_scalars(group)


def test_fused_apply_requires_a_device_rate() -> None:
    parameter = torch.empty(2, 3, dtype=torch.bfloat16)
    master = torch.zeros(2, 3)
    update = torch.ones(2, 3, dtype=torch.bfloat16)
    optimizer = FusedMuon.Config().make()([_parameter(2, 3)])
    with pytest.raises(TypeError) as error:
        optimizer._apply_cuda(parameter, master, update, lr=0.01, scale=1.0)

    assert str(error.value) == "the fused update reads its rate from a device tensor"


def test_cuda_wrappers_execute_their_torch_reference_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_nesterov(
        *args: object,
        grid: tuple[int, ...],
        **kwargs: object,
    ) -> None:
        gradient, buffer, update, coefficient, momentum, count = cast(
            tuple[Tensor, Tensor, Tensor, Tensor, float, int],
            args,
        )
        block = cast(int, kwargs["block"])
        num_warps = cast(int, kwargs["num_warps"])
        assert kwargs == {"block": 1024, "num_warps": 4, "test_flag": True}
        assert grid == ((count + block - 1) // block,)
        assert (count, block, num_warps) == (gradient.numel(), 1024, 4)
        update.copy_(
            nesterov(gradient, buffer, coefficient=coefficient, momentum=momentum),
        )

    def run_normalize(
        *args: object,
        grid: tuple[int, ...],
        **kwargs: object,
    ) -> None:
        matrix, output, norm, eps, count = cast(
            tuple[Tensor, Tensor, Tensor, float, int],
            args,
        )
        block = cast(int, kwargs["block"])
        num_warps = cast(int, kwargs["num_warps"])
        assert kwargs == {"block": 1024, "num_warps": 4, "test_flag": True}
        assert grid == ((count + block - 1) // block,)
        assert (count, block, num_warps) == (matrix.numel(), 1024, 4)
        output.copy_(normalize(matrix, norm=norm, eps=eps))

    def run_apply(
        *args: object,
        grid: tuple[int, ...],
        **kwargs: object,
    ) -> None:
        update, master, parameter, lr, scale, count = cast(
            tuple[Tensor, Tensor, Tensor, Tensor, float, int],
            args,
        )
        block = cast(int, kwargs["block"])
        num_warps = cast(int, kwargs["num_warps"])
        scaled = cast(bool, kwargs["scaled"])
        assert kwargs == {
            "scaled": scale != 1.0,
            "block": 1024,
            "num_warps": 4,
            "test_flag": True,
        }
        assert grid == ((count + block - 1) // block,)
        assert (count, block, num_warps) == (update.numel(), 1024, 4)
        assert scaled is (scale != 1.0)
        apply_update(parameter, master, update, lr=lr, scale=scale)

    def fake_kernels(**helpers: Callable[..., object]) -> SimpleNamespace:
        del helpers
        return SimpleNamespace(
            nesterov=_KernelLaunch(run_nesterov),
            normalize=_KernelLaunch(run_normalize),
            apply=_KernelLaunch(run_apply),
        )

    monkeypatch.setattr(FusedMuon, "launch_options", {"test_flag": True})
    monkeypatch.setattr(fused_muon, "_kernels", fake_kernels)
    optimizer = FusedMuon.Config().make()([_parameter(2, 3)])

    gradient = torch.ones(2, 3, dtype=torch.bfloat16)
    buffer = torch.zeros(2, 3)
    coefficient = torch.tensor(0.5)
    expected_buffer = buffer.clone()
    expected_update = nesterov(
        gradient,
        expected_buffer,
        coefficient=coefficient,
        momentum=0.5,
    )
    actual_update = optimizer._nesterov_cuda(
        gradient,
        buffer,
        coefficient=coefficient,
        momentum=0.5,
    )
    assert torch.equal(actual_update, expected_update)
    assert torch.equal(buffer, expected_buffer)

    norm = optimizer.norm([actual_update])
    expected_normalized = normalize(actual_update, norm=norm, eps=1e-7)
    assert torch.equal(
        optimizer._normalize_cuda(actual_update, norm=norm, eps=1e-7),
        expected_normalized,
    )

    master = torch.zeros(2, 3)
    parameter = torch.empty(2, 3, dtype=torch.bfloat16)
    rate = torch.tensor(0.25)
    optimizer._apply_cuda(parameter, master, actual_update, lr=rate, scale=1.0)
    assert torch.equal(master, -actual_update.float() * rate)
    assert torch.equal(parameter, master.bfloat16())


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


def test_a_tiny_stack_matches_its_golden() -> None:
    """A five-weight stack: three steps from fp32 masters.

    The masters are portable draws that the bf16 parameters only round, as a
    checkpoint's are, and reach the optimizer through ``load_state_dict``. The
    optimizer takes the weights in another order than they were drawn in, as a
    model whose global norm runs in its own order hands them over. The first
    and last steps' gradients are clipped and the second's are not.
    """
    shapes = {
        "embedding.weight": (2, 2),
        "proj_in.weight": (3, 2),
        "blocks.0.proj_gates.weight": (2, 3),
        "blocks.1.proj_gates.weight": (2, 2),
        "proj_out.weight": (2, 2),
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
        record = _step_record(optimizer, parameters, order=order, steps=steps)
    assert_tensor_golden(_CWD / "testdata" / "fused_muon_tiny.pt", record)


def _two_steps(rate: float | Tensor, *, device: str = "cpu") -> FusedMuon:
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
            parameter.grad = torch.randn(*parameter.shape, generator=generator).to(
                device=device,
                dtype=parameter.dtype,
            )
        optimizer.step()
    return optimizer


def _rate(epoch: int) -> float:
    """Return epoch ``epoch`` of a 30-epoch cosine, rounded to fp32 as a group holds it."""
    rate = torch.tensor(0.00472887093 * lr_scale(epoch, 30), dtype=torch.float32)
    return float(rate)


# Only the last step's state is stored: every earlier step's masters and momentum feed
# it, and the per-step clips say which step broke. The bf16 parameters are the masters
# rounded, which is asserted rather than stored.
def _step_record(
    optimizer: FusedMuon,
    parameters: dict[str, torch.nn.Parameter],
    *,
    order: list[str],
    steps: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Step once per gradient set; return each clip and the final masters and momentum."""
    clips: list[Tensor] = []
    for gradients in steps:
        for name, parameter in parameters.items():
            parameter.grad = gradients[name]
        clip = clip_coefficient(
            optimizer.norm([gradients[name] for name in order]),
            optimizer.max_grad_norm,
        )
        clips.append(torch.as_tensor(clip, dtype=torch.float32).reshape(()))
        optimizer.step()
    state = cast("dict[int, dict[str, Tensor]]", optimizer.state_dict()["state"])
    record = {"clip": torch.stack(clips)}
    for position, name in enumerate(order):
        master = state[position]["master_weight"]
        assert torch.equal(parameters[name].detach(), master.bfloat16())
        record[f"master_weight/{name}"] = master.detach().clone()
        record[f"momentum_buffer/{name}"] = state[position]["momentum_buffer"].clone()
    return record


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
    return (torch.rand(*shape, generator=generator) * 2 - 1) * bound


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
