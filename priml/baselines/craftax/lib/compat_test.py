"""Tests of PufferLib's arithmetic as the compat classes swap it in.

The CPU tests pin the reduction tree, class wiring, and ExactMuon's
three-step tensor trace. On the GPU, the PPO kernels' PTX is read for
PufferLib's intrinsics. Every class's kernel bits are pinned by exp000's GPU
goldens, which select them, on the RTX 5090 they are keyed to (the exact
sampler's on any GPU, by the torch sampler's golden), and against PufferLib's
own values by the parity suite (README, "Parity").
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import inspect

from torch import Tensor
from triton.runtime import interpreter, jit

import pytest
import torch
import triton

from priml.baselines.craftax.experiments import exp000
from priml.baselines.craftax.lib import compat
from priml.baselines.craftax.lib.compat import (
    LAUNCH_OPTIONS,
    ExactMuon,
    ExactPhiloxSampler,
    ExactPPO,
    ExactScan,
    sum_squares,
)
from priml.baselines.craftax.testing import portable_uniform
from priml.baselines.craftax.train_step import AgentWindows
from priml.loss import policy_gradient_kernel
from priml.loss.policy_gradient import TorchPPO
from priml.optimizers.fused_muon import FusedMuon, clip_coefficient
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_tensor_golden
from priml.testing.policy_gradient import random_minibatch


_CWD: Final = Path(__file__).resolve().parent


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from typing_extensions import TypeIs

    from numpy.typing import NDArray

    import numpy as np

    from priml.baselines.craftax.model import MinGRUPolicy


def test_the_sum_of_squares_uses_the_halving_tree() -> None:
    # Sequential fp32 addition loses every 1 after 2^24; the tree adds the
    # ones to each other first, so the result is 2^24 + 2.
    values = torch.tensor([4096.0, 1.0, 1.0, 1.0])
    assert float(sum_squares(values)) == 16_777_218.0


@pytest.mark.parametrize("count", [1, 300, 256 * 256 * 2 + 5])
def test_the_sum_of_squares_covers_every_element_across_stripes(count: int) -> None:
    # Ones sum exactly whatever the order, so this checks the striping alone:
    # one block, two blocks, and three grid-stride passes with padding.
    assert float(sum_squares(torch.ones(count, dtype=torch.bfloat16))) == count


@pytest.mark.gpu_triton
@pytest.mark.parametrize("count", [1, 300, 256 * 256 * 2 + 5, 14_304_672])
def test_the_cuda_kernels_sum_the_squares_in_the_torch_trees_order(count: int) -> None:
    # Random values make every reassociation visible in the low bits, so the
    # Triton kernels must reproduce the CPU tree's result bit for bit at one
    # block, two blocks, three passes with a tail, and a 14.3M-weight model.
    if not torch.cuda.is_available():
        pytest.skip("the Triton kernels need a CUDA device")
    generator = torch.Generator().manual_seed(count)
    values = torch.randn(count, generator=generator).to(torch.bfloat16)
    assert torch.equal(sum_squares(values.cuda()).cpu(), sum_squares(values))


class _ReportsCuda(Tensor):
    """A CPU tensor that reports itself on CUDA, to reach the kernels' dispatch."""

    is_cuda = True


def test_the_sum_of_squares_hands_a_cuda_tensor_to_the_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[Tensor] = []
    total = torch.tensor(5.0)

    def kernels(values: Tensor) -> Tensor:
        launched.append(values)
        return total

    monkeypatch.setattr(compat, "_sum_squares_cuda", kernels)
    on_cuda = torch.ones(3).as_subclass(_ReportsCuda)
    assert sum_squares(on_cuda) is total
    assert len(launched) == 1
    assert launched[0] is on_cuda
    assert float(sum_squares(torch.ones(3))) == 3.0
    assert len(launched) == 1


def test_the_exact_muon_takes_fused_muons_hyperparameters() -> None:
    config = ExactMuon.Config()
    config.lr = 0.25
    parameter = torch.nn.Parameter(torch.ones(4, 2, dtype=torch.bfloat16))
    optimizer = config.make()([parameter])
    assert type(optimizer) is ExactMuon
    assert optimizer.param_groups[0]["lr"] == 0.25
    assert optimizer.max_grad_norm == FusedMuon.Config().max_grad_norm


def test_the_exact_norm_is_the_tree_over_the_concatenation() -> None:
    optimizer = ExactMuon.Config().make()([torch.nn.Parameter(torch.ones(2))])
    tensors = [torch.tensor([4096.0]), torch.ones(3, dtype=torch.bfloat16)]
    assert float(optimizer.norm(tensors)) == float(torch.tensor(16_777_218.0).sqrt())


def test_the_exact_muon_reproduces_its_golden() -> None:
    """Three ExactMuon steps over square, tall, and wide matrices."""
    shapes = {
        "embedding.weight": (2, 2),
        "proj_in.weight": (3, 2),
        "blocks.0.proj_gates.weight": (2, 3),
        "blocks.1.proj_gates.weight": (2, 2),
        "proj_out.weight": (2, 2),
    }
    generator = torch.Generator().manual_seed(0)
    masters = {
        name: portable_uniform(*shape, bound=shape[-1] ** -0.5, generator=generator)
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
    config = ExactMuon.Config()
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
            name: portable_uniform(*shape, bound=scale, generator=generator).bfloat16()
            for name, shape in shapes.items()
        }
        for scale in (2**-4, 2**-7, 2**-4)
    ]
    with host_agnostic_numerics():
        record = _muon_record(optimizer, parameters, order=order, steps=steps)
    assert_tensor_golden(
        _CWD / "testdata" / "exact_muon_tiny.pt",
        record,
    )


def test_the_sum_kernels_jit_their_bodies_once_with_the_shared_tree() -> None:
    kernels = compat._sum_squares_kernels()
    assert compat._sum_squares_kernels() is kernels
    for kernel, body in (
        (kernels.partials, compat._sum_squares_partials_triton),
        (kernels.total, compat._sum_squares_total_triton),
    ):
        assert kernel.fn.__code__ is body.__code__
        tree = _jit_global(kernel.fn.__globals__, "_halving_tree_triton")
        assert tree.fn.__code__ is compat._halving_tree_triton.__code__
    ordered = compat._ordered_sum_kernel()
    assert compat._ordered_sum_kernel() is ordered
    assert ordered.fn.__code__ is compat._ordered_sum_triton.__code__


@pytest.mark.usefixtures("interpreted_kernels")
def test_the_interpreted_sum_of_squares_kernels_follow_the_torch_tree() -> None:
    # Two blocks, the second part padding. A second grid-stride pass needs over
    # 65,536 values, 1.2 s interpreted; the GPU test covers it.
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(300, generator=generator).to(torch.bfloat16)
    assert torch.equal(compat._sum_squares_cuda(values), sum_squares(values))


@pytest.mark.usefixtures("interpreted_kernels")
def test_the_interpreted_block_sum_adds_the_blocks_in_order() -> None:
    # 2^24 + 1 + 1 in block order loses both ones; any other order keeps one.
    partials = torch.zeros(3, len(TorchPPO.Config.LOSS_NAMES))
    partials[:, 0] = torch.tensor([16_777_216.0, 1.0, 1.0])
    partials[:, 1:] = torch.randn(3, 7, generator=torch.Generator().manual_seed(2))
    losses = ExactPPO.Config().make().sum_blocks(partials)
    assert torch.equal(losses, partials[0] + partials[1] + partials[2])
    assert float(losses[0]) == 16_777_216.0


def test_exp000_selects_every_compat_class() -> None:
    step = exp000().step.copy_tree().finalize()
    model = cast("MinGRUPolicy.Config", step.model)
    assert type(model.block.scan) is ExactScan.Config
    assert isinstance(step.learner, AgentWindows.Config)
    assert type(step.learner.objective) is ExactPPO.Config
    assert type(step.optimizer) is ExactMuon.Config
    assert type(step.sampler) is ExactPhiloxSampler.Config
    assert type(step.evaluation.sampler) is ExactPhiloxSampler.Config


class _Compiled(Protocol):
    asm: dict[str, str]


@pytest.mark.gpu_triton
def test_the_exact_rule_issues_pufferlibs_intrinsics() -> None:
    """Read the scoring kernel's PTX: ``__expf``, ``__logf``, one fma, no fast division.

    ``fast_expf`` was ``ex2.approx.f32`` of ``x * log2 e``, without
    flush-to-zero; Triton's own ``exp`` must lower to that instruction.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    batch = random_minibatch(rows=2, horizon=256, device="cuda", seed=3)
    rows, horizon = 2, 256
    values = torch.empty(rows, horizon, dtype=torch.bfloat16, device="cuda")
    logps = torch.empty(rows, horizon, 43, device="cuda")
    new_lp = torch.empty(rows, horizon, device="cuda")
    kernels = policy_gradient_kernel._kernels(**ExactPPO.helpers)
    cache = cast(
        "_Compiled",
        kernels.cache[(2,)](
            (
                batch["decoded"],
                batch["actions"],
                batch["action_mask"],
                values,
                logps,
                new_lp,
            ),
            rows * horizon,
            num_actions=43,
            block=ExactPPO.Config().block,
            num_warps=ExactPPO.Config().num_warps,
            **LAUNCH_OPTIONS,
        ),
    )
    ptx = cache.asm["ptx"]
    census = {
        name: ptx.count(name)
        for name in (
            "ex2.approx.f32",
            "ex2.approx.ftz",
            "lg2.approx",
            "div.rn.f32",
            "div.full.f32",
            "fma.rn.f32",
        )
    }
    print("exact log_probs PTX:", census)  # noqa: T201 -- The census is read from the job log.
    assert census["ex2.approx.f32"] > 0
    assert census["ex2.approx.ftz"] == 0
    assert census["lg2.approx"] > 0
    # The kernel's only contraction is ``max + __logf(sum)`` as one fma; with fp
    # fusion off, ptxas adds no other.
    assert census["fma.rn.f32"] == 1
    assert census["div.full.f32"] == 0


@pytest.fixture
def interpreted_kernels(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run the module's Triton kernels through Triton's CPU interpreter, uncached."""
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    monkeypatch.setattr(
        jit,
        "_normalize_ty",
        partial(_normalize_bare_constexpr, normalize=jit._normalize_ty),
    )
    monkeypatch.setattr(
        interpreter,
        "_patch_lang_tensor",
        partial(_patch_scalar_index, patch=interpreter._patch_lang_tensor),
    )
    # Triton jitted its library functions at import, compiled-mode, which the
    # interpreter refuses to call.
    for module in (triton.language, triton.language.standard):
        for name, value in inspect.getmembers(module, _is_jit):
            monkeypatch.setattr(
                module,
                name,
                interpreter.InterpretedFunction(value.fn),
            )
    # Each library call patches ``triton.language.core``'s builtins and never
    # restores them (``InterpretedFunction.__call__``), so a kernel compiled
    # after this test would trace the interpreter's builtins and fail.
    members: dict[str, object] = dict(vars(triton.language.core))
    for name, value in members.items():
        monkeypatch.setattr(triton.language.core, name, value)
    compat._sum_squares_kernels.cache_clear()
    compat._ordered_sum_kernel.cache_clear()
    try:
        yield
    finally:
        compat._sum_squares_kernels.cache_clear()
        compat._ordered_sum_kernel.cache_clear()


# The interpreter matches a constexpr only by its bare name; the module's postponed
# annotations spell it ``language.constexpr``.
def _normalize_bare_constexpr(
    ty: object,
    *,
    normalize: Callable[[object], str],
) -> str:
    return normalize(
        ty.strip("'").removeprefix("language.") if isinstance(ty, str) else ty,
    )


# The interpreter holds a scalar as a one-element array, which numpy 2.5 no longer
# converts with ``int``; a runtime loop bound such as ``range(passes)`` needs it.
def _patch_scalar_index(
    tensor: type,
    scope: _PatchScope,
    *,
    patch: Callable[[type, _PatchScope], None],
) -> None:
    patch(tensor, scope)
    scope.set_attr(tensor, "__index__", _scalar_index)


def _scalar_index(tensor: _InterpretedTensor) -> int:
    return int(tensor.handle.data.item())


def _is_jit(value: object) -> TypeIs[jit.JITFunction[..., object]]:
    return isinstance(value, jit.JITFunction)


def _jit_global(scope: Mapping[str, object], name: str) -> jit.JITFunction[..., object]:
    helper = scope[name]
    assert _is_jit(helper)
    return helper


class _PatchScope(Protocol):
    def set_attr(self, obj: object, name: str, value: object) -> None: ...


class _InterpretedHandle(Protocol):
    data: NDArray[np.integer]


class _InterpretedTensor(Protocol):
    handle: _InterpretedHandle


def _muon_record(
    optimizer: FusedMuon,
    parameters: dict[str, torch.nn.Parameter],
    *,
    order: list[str],
    steps: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Step once per gradient set and return every clip and tensor state."""
    record: dict[str, Tensor] = {}
    for index, gradients in enumerate(steps):
        for name, parameter in parameters.items():
            parameter.grad = gradients[name]
        clip = clip_coefficient(
            optimizer.norm([gradients[name] for name in order]),
            optimizer.max_grad_norm,
        )
        record[f"step/{index}/clip"] = torch.as_tensor(
            clip,
            dtype=torch.float32,
        ).reshape(())
        optimizer.step()
        state = cast("dict[int, dict[str, Tensor]]", optimizer.state_dict()["state"])
        for position, name in enumerate(order):
            record[f"step/{index}/master_weight/{name}"] = (
                state[position]["master_weight"].detach().clone()
            )
            record[f"step/{index}/momentum_buffer/{name}"] = (
                state[position]["momentum_buffer"].detach().clone()
            )
        for name, parameter in parameters.items():
            record[f"step/{index}/parameter/{name}"] = parameter.detach().clone()
    return record


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
