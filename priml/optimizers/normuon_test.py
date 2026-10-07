"""Tests for the NorMuon optimizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast

from torch import Tensor

import pytest
import torch

from priml.lib.custom_json import ReadError
from priml.optimizers import normuon
from priml.optimizers.normuon import NorMuon


def _parameters(*shapes: tuple[int, ...]) -> list[torch.nn.Parameter]:
    """Parameters of the given shapes, each carrying a gradient."""
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(shape)) for shape in shapes]
    for parameter in params:
        parameter.grad = torch.randn_like(parameter)
    return params


def _state_tensor(optimizer: NorMuon, parameter: Tensor, name: str) -> Tensor:
    """One per-parameter state buffer, narrowed from torch's untyped state."""
    value = cast(dict[str, object], optimizer.state[parameter])[name]
    assert isinstance(value, Tensor)
    return value


def _capture_kernel_state(
    optimizer: NorMuon,
    capture: Callable[[Tensor, Tensor, int], None],
) -> None:
    def update(
        stacked_grads: Tensor,
        stacked_params: Tensor,
        momentum_buffer: Tensor,
        second_moment: Tensor,
        *,
        momentum: Tensor,
        lr: Tensor,
        weight_decay: Tensor,
        beta2: Tensor,
        ns_steps: int,
        reduce_dim: int,
        coefficients: tuple[tuple[float, float, float], ...],
    ) -> None:
        del (
            stacked_grads,
            stacked_params,
            momentum,
            lr,
            weight_decay,
            beta2,
            ns_steps,
            coefficients,
        )
        capture(momentum_buffer, second_moment, reduce_dim)

    optimizer._update = update


def test_by_shape_skips_missing_gradients_and_orders_buckets() -> None:
    parameters = _parameters((2, 3), (4, 2), (2, 3))
    parameters[0].grad = None
    buckets = normuon._by_shape(parameters)
    assert len(buckets) == 2
    assert buckets[0][0] is parameters[2]
    assert buckets[1][0] is parameters[1]


def test_gradient_helper_rejects_a_missing_gradient_with_exact_message() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2, 3))
    with pytest.raises(ValueError, match=r"^Expected grad is not None\.$"):
        normuon._gradient(parameter)


def test_the_update_is_approximately_orthogonal() -> None:
    """The whole point: the step's singular values are all near one.

    An update that merely descended would leave them spread. This is what
    makes the step invariant to the weight matrix's scale, so if the
    polynomial iteration were wrong the optimizer would silently become a
    poorly-tuned SGD.
    """
    params = _parameters((16, 17))
    before = params[0].detach().clone()
    # Eager: these assertions are about the update's algebra, which holds at
    # either, and tracing the step costs 11s per process and is never cached.
    NorMuon(params, lr=1.0, momentum=0.0, weight_decay=0.0, compile=False).step()
    update = before - params[0].detach()
    singular = torch.linalg.svdvals(update)
    assert float(singular.min()) > 0.7
    assert float(singular.max()) < 1.4


def test_it_is_invariant_to_the_gradient_scale() -> None:
    """Scaling the gradient by 1000 must not scale the step.

    Orthogonalization normalizes the update away, which is the property a
    scale-free optimizer is chosen for.

    The tolerance is bf16's, not the algorithm's. The iteration runs in bf16
    deliberately -- it is self-correcting and its matmuls dominate the step --
    and the same iteration measured at three precisions over 20 seeds and four
    matrix sizes gives a worst-case relative deviation of 1.9e-01 at bf16
    against 2.0e-05 at float32 and 1.2e-05 at float64. A tighter bound here
    would be testing the dtype rather than the invariance; a loose one still
    separates this from an optimizer that passed the scale through, which
    would differ by 1000x.
    """
    steps: list[Tensor] = []
    for scale in (1.0, 1000.0):
        params = _parameters((8, 9))
        assert params[0].grad is not None
        params[0].grad *= scale
        before = params[0].detach().clone()
        NorMuon(params, lr=0.1, momentum=0.0, weight_decay=0.0, compile=False).step()
        steps.append(before - params[0].detach())
    # Relative to the step's own size: an optimizer that passed the scale
    # through would differ by 1000x, not by a rounding error.
    relative = (steps[0] - steps[1]).abs().max() / steps[0].abs().max()
    assert float(relative) < 0.25


def test_same_shape_parameters_step_together() -> None:
    """Parameters are batched by shape, so a mixed model must still update.

    Every parameter must move, and one shape's update must not leak into
    another's -- which a wrong stacking would silently do.
    """
    params = _parameters((8, 9), (8, 9), (4, 16))
    before = [p.detach().clone() for p in params]
    NorMuon(params, lr=0.1, momentum=0.0, weight_decay=0.0, compile=False).step()
    for original, updated in zip(before, params, strict=True):
        assert not torch.equal(original, updated.detach())


def test_update_has_exact_bfloat16_state_and_cautious_decay() -> None:
    stacked_grads = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.bfloat16,
    )
    stacked_params = torch.tensor(
        [[[1.0, -2.0], [-3.0, 4.0]]],
        dtype=torch.bfloat16,
    )
    momentum_buffer = torch.zeros_like(stacked_grads)
    # _normuon_update stores one second moment per reduced row or column.
    second_moment = torch.zeros((1, 2, 1), dtype=torch.bfloat16)
    normuon._normuon_update(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_moment,
        momentum=torch.tensor(0.0),
        lr=torch.tensor(0.1),
        weight_decay=torch.tensor(0.2),
        beta2=torch.tensor(0.0),
        ns_steps=1,
        reduce_dim=-1,
        coefficients=((1.0, 0.0, 0.0),),
    )
    assert torch.equal(momentum_buffer, stacked_grads)
    assert torch.equal(
        second_moment,
        torch.tensor([[[0.2393], [0.2393]]], dtype=torch.bfloat16),
    )
    assert torch.equal(
        stacked_params,
        torch.tensor([[[0.9102, -1.9609], [-2.9375, 3.8438]]], dtype=torch.bfloat16),
    )


def test_decay_sign_uses_the_underflowed_product() -> None:
    stacked_grads = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    stacked_params = torch.tensor([[[-1e-23, 0.0], [0.0, 0.0]]])
    momentum_buffer = torch.zeros_like(stacked_grads)
    # _normuon_update stores one second moment per reduced row or column.
    second_moment = torch.zeros((1, 2, 1))
    normuon._normuon_update(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_moment,
        momentum=torch.tensor(0.0),
        lr=torch.tensor(0.1),
        weight_decay=torch.tensor(0.2),
        beta2=torch.tensor(0.0),
        ns_steps=1,
        reduce_dim=-1,
        coefficients=((7e-23, 0.0, 0.0),),
    )
    assert stacked_params.tolist() == [
        [[-9.799805244255315e-24, 0.0], [0.0, -3.6207482954480034e-31]],
    ]


def test_rectangular_update_matches_exact_reference_values() -> None:
    stacked_grads = torch.tensor([[[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]]])
    stacked_params = torch.tensor([[[0.5, -1.0, 2.0], [3.0, 4.0, -2.0]]])
    momentum_buffer = torch.tensor([[[0.25, -0.5, 0.75], [1.0, -1.5, 2.0]]])
    # _normuon_update stores one second moment per reduced row or column.
    second_moment = torch.zeros((1, 2, 1))
    normuon._normuon_update(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_moment,
        momentum=torch.tensor(0.3),
        lr=torch.tensor(0.17),
        weight_decay=torch.tensor(0.23),
        beta2=torch.tensor(0.4),
        ns_steps=2,
        reduce_dim=-1,
        coefficients=NorMuon.Config().coefficients,
    )
    assert stacked_params.tolist() == [
        [
            [0.3564453125, -1.212890625, 2.0400390625],
            [2.8828125, 3.7099609375, -2.2099609375],
        ],
    ]
    assert momentum_buffer.tolist() == [
        [
            [0.7749999761581421, 1.25, 0.22500000894069672],
            [0.30000001192092896, 1.649999976158142, 3.4000000953674316],
        ],
    ]
    assert second_moment.tolist() == [[[0.2272014617919922], [0.6333345174789429]]]


def test_batched_tall_update_matches_exact_reference_values() -> None:
    stacked_grads = torch.tensor(
        [
            [[1.0, 2.0], [0.0, 3.0], [4.0, 1.0]],
            [[2.0, 0.0], [1.0, 3.0], [2.0, 5.0]],
        ],
    )
    stacked_params = torch.tensor(
        [
            [[0.5, -1.0], [2.0, 3.0], [-2.0, 4.0]],
            [[-1.0, 2.0], [3.0, -2.0], [1.0, 0.5]],
        ],
    )
    momentum_buffer = torch.zeros_like(stacked_grads)
    # _normuon_update stores one second moment per reduced row or column.
    second_moment = torch.zeros((2, 3, 1))
    normuon._normuon_update(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_moment,
        momentum=torch.tensor(0.2),
        lr=torch.tensor(0.13),
        weight_decay=torch.tensor(0.15),
        beta2=torch.tensor(0.3),
        ns_steps=2,
        reduce_dim=-1,
        coefficients=NorMuon.Config().coefficients,
    )
    assert stacked_params.tolist() == [
        [
            [0.4501953125, -1.04296875],
            [1.93359375, 2.8896484375],
            [-2.052001953125, 3.89453125],
        ],
        [
            [-1.1435546875, 1.93115234375],
            [2.890625, -2.13671875],
            [0.92578125, 0.35546875],
        ],
    ]


def test_tiny_gradient_matches_exact_reference_values() -> None:
    stacked_grads = torch.tensor([[[1e-6, 2e-6], [3e-6, 5e-6]]])
    stacked_params = torch.zeros_like(stacked_grads)
    momentum_buffer = torch.zeros_like(stacked_grads)
    # _normuon_update stores one second moment per reduced row or column.
    second_moment = torch.zeros((1, 2, 1))
    normuon._normuon_update(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_moment,
        momentum=torch.tensor(0.0),
        lr=torch.tensor(1.0),
        weight_decay=torch.tensor(0.0),
        beta2=torch.tensor(0.0),
        ns_steps=2,
        reduce_dim=-1,
        coefficients=NorMuon.Config().coefficients,
    )
    assert stacked_params.tolist() == [
        [[0.396484375, -0.55078125], [-0.56640625, -0.376953125]],
    ]


def test_large_second_moment_keeps_the_update_norm() -> None:
    stacked_grads = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    stacked_params = torch.zeros_like(stacked_grads)
    momentum_buffer = torch.zeros_like(stacked_grads)
    # _normuon_update stores one second moment per reduced row or column.
    second_moment = torch.full((1, 2, 1), 1_000_000.0)
    normuon._normuon_update(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_moment,
        momentum=torch.tensor(0.0),
        lr=torch.tensor(1.0),
        weight_decay=torch.tensor(0.0),
        beta2=torch.tensor(0.999),
        ns_steps=2,
        reduce_dim=-1,
        coefficients=NorMuon.Config().coefficients,
    )
    assert stacked_params.tolist() == [[[1.09375, -1.25], [-1.0625, -0.328125]]]


def test_low_precision_momentum_uses_gradient_dtype() -> None:
    torch.manual_seed(13)
    stacked_grads = torch.randn((2, 16, 17), dtype=torch.bfloat16)
    stacked_params = torch.zeros_like(stacked_grads)
    momentum_buffer = torch.randn_like(stacked_grads)
    expected_buffer = momentum_buffer.clone()
    expected_buffer.lerp_(stacked_grads, 1 - torch.tensor(0.95).bfloat16())
    wider_buffer = momentum_buffer.clone()
    wider_buffer.lerp_(stacked_grads, 1 - torch.tensor(0.95))
    assert not torch.equal(expected_buffer, wider_buffer)
    # _normuon_update stores one second moment per reduced row or column.
    second_moment = torch.zeros((2, 16, 1), dtype=torch.bfloat16)
    normuon._normuon_update(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_moment,
        momentum=torch.tensor(0.95),
        lr=torch.tensor(0.0),
        weight_decay=torch.tensor(0.0),
        beta2=torch.tensor(0.0),
        ns_steps=1,
        reduce_dim=-1,
        coefficients=((1.0, 0.0, 0.0),),
    )
    assert torch.equal(momentum_buffer, expected_buffer)


def test_square_update_uses_the_wide_branch_at_equal_sides() -> None:
    stacked_grads = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])
    stacked_params = torch.tensor([[[1.0, -1.0], [2.0, 3.0]]])
    momentum_buffer = torch.zeros_like(stacked_grads)
    # _normuon_update stores one second moment per reduced row or column.
    second_moment = torch.zeros((1, 1, 2))
    normuon._normuon_update(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_moment,
        momentum=torch.tensor(0.0),
        lr=torch.tensor(0.25),
        weight_decay=torch.tensor(0.1),
        beta2=torch.tensor(0.5),
        ns_steps=2,
        reduce_dim=-2,
        coefficients=NorMuon.Config().coefficients,
    )
    assert stacked_params.tolist() == [
        [[1.1162109375, -1.1728515625], [1.720458984375, 2.7354736328125]],
    ]


def test_weight_decay_is_cautious() -> None:
    """Decay applies exactly where it agrees with the update, and nowhere else.

    A weight the update is already shrinking must not be decayed too, or the
    two compound into a step neither asked for. With every weight positive,
    "agrees" reduces to "the update is also positive", so the set decay
    touches is predictable independently of the optimizer -- which is what
    makes this a check rather than a restatement.
    """

    def step(*, weight_decay: float) -> Tensor:
        params = _parameters((8, 9))
        with torch.no_grad():
            params[0].copy_(torch.ones(8, 9))
        before = params[0].detach().clone()
        NorMuon(
            params,
            lr=0.1,
            momentum=0.0,
            weight_decay=weight_decay,
            compile=False,
        ).step()
        return before - params[0].detach()

    without = step(weight_decay=0.0)
    with_decay = step(weight_decay=0.5)
    # ``without`` IS the orthogonalized update, up to the learning rate, so a
    # positive entry is one that decreases a positive weight.
    decayed = with_decay != without
    assert torch.equal(decayed, without > 0)
    # And where it applies, it pushes further in the update's own direction.
    assert bool((with_decay[decayed] > without[decayed]).all())


def test_row_rescaling_redistributes_without_resizing() -> None:
    """NorMuon's correction moves step budget between rows, not into them.

    Driven from a SKEWED second moment, because the correction is near-inert
    otherwise: an orthogonalized square update already has near-equal row
    energies, so on a first step there is nothing to redistribute (measured
    row-norm spread 0.0066 against a mean of 1.0223). Halving one row's
    recorded energy is what makes the effect observable.

    The invariant is the whole design: rows move relative to each other, and
    the total norm does not -- dropping the renormalization keeps the rows and
    changes the norm instead.
    """

    def update(*, skew: bool) -> Tensor:
        params = _parameters((16, 17))
        before = params[0].detach().clone()
        optimizer = NorMuon(
            params,
            lr=1.0,
            momentum=0.0,
            weight_decay=0.0,
            compile=False,
        )
        optimizer.step()
        if skew:
            second_moment = _state_tensor(optimizer, params[0], "second_moment")
            second_moment[:8] *= 0.25
        assert params[0].grad is not None
        params[0].grad = torch.randn_like(params[0])
        optimizer.step()
        return before - params[0].detach()

    plain, skewed = update(skew=False), update(skew=True)
    rows_moved = (plain.norm(dim=-1) - skewed.norm(dim=-1)).abs().max()
    assert float(rows_moved) > 0.01
    assert float(skewed.norm()) == pytest.approx(float(plain.norm()), rel=0.05)


def test_a_vector_is_rejected() -> None:
    """Orthogonalizing a rank-1 tensor is undefined.

    It must not be routed here silently.
    """
    params = _parameters((8,))
    with pytest.raises(
        ValueError,
        match=r"^NorMuon requires ndim >= 2; got shape \(8,\)\.$",
    ):
        NorMuon(params, lr=0.1, compile=False).step()


def test_eligibility_names_the_rank_rule() -> None:
    """The recipe routes by this, so it states the algorithm's own constraint."""
    matrix = torch.nn.Parameter(torch.zeros(4, 5))
    vector = torch.nn.Parameter(torch.zeros(4))
    assert NorMuon.eligible_tensor("w", matrix)
    assert not NorMuon.eligible_tensor("b", vector)


_Build = Callable[[list[torch.nn.Parameter]], NorMuon]

_INVALID: Final[list[tuple[str, _Build, str]]] = [
    ("lr", lambda p: NorMuon(p, lr=-1.0), "Learning rate"),
    ("momentum", lambda p: NorMuon(p, momentum=1.0), "Momentum"),
    ("beta2", lambda p: NorMuon(p, beta2=1.0), "Beta2"),
    ("weight_decay", lambda p: NorMuon(p, weight_decay=-1.0), "Weight decay"),
    ("ns_steps", lambda p: NorMuon(p, ns_steps=99), "ns_steps"),
]


@pytest.mark.parametrize(
    ("build", "message"),
    [(build, message) for _, build, message in _INVALID],
    ids=[name for name, _, _ in _INVALID],
)
def test_invalid_hyperparameters_are_rejected(
    build: _Build,
    message: str,
) -> None:
    """Each bound would otherwise fail deep in an update, or not at all."""
    with pytest.raises(ValueError, match=message):
        build(_parameters((4, 4)))


def test_config_builds_a_constructor_awaiting_parameters() -> None:
    """A config tree has no parameters, so ``make`` cannot return an optimizer."""
    build = NorMuon.Config(lr=0.5, compile=False).make()
    optimizer = build(_parameters((4, 4)))
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_scalar_dtype_ignores_the_global_default_dtype() -> None:
    params = _parameters((2, 3))
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        optimizer = NorMuon(params, compile=False)
    finally:
        torch.set_default_dtype(original_dtype)
    assert all(value.dtype == torch.float32 for value in optimizer._scalars.values())


def test_optimizer_state_uses_the_parameter_device() -> None:
    params = _parameters((2, 3))
    optimizer = NorMuon(params, compile=False)
    captured: list[tuple[Tensor, Tensor, int]] = []
    _capture_kernel_state(
        optimizer,
        lambda momentum_buffer, second_moment, reduce_dim: captured.append(
            (momentum_buffer, second_moment, reduce_dim),
        ),
    )
    with torch.device("meta"):
        optimizer.step()
    assert len(captured) == 1
    assert captured[0][0].device == torch.device("cpu")
    assert captured[0][1].device == torch.device("cpu")


@pytest.mark.parametrize("name", ["momentum", "lr", "weight_decay", "beta2"])
def test_invalid_parameter_group_scalar_has_exact_error(name: str) -> None:
    params = _parameters((2, 3))
    optimizer = NorMuon(params, compile=False)
    optimizer.param_groups[0][name] = None
    with pytest.raises(ReadError):
        optimizer.step()


def test_optimizer_defaults_and_inclusive_lower_bounds() -> None:
    params = _parameters((3, 5))
    optimizer = NorMuon(params, compile=False)
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.04
    assert group["momentum"] == 0.95
    assert group["beta2"] == 0.95
    assert group["weight_decay"] == 0.2
    assert group["ns_steps"] == 5
    assert optimizer._scalars.keys() == {"momentum", "lr", "weight_decay", "beta2"}
    assert all(value.dtype == torch.float32 for value in optimizer._scalars.values())

    NorMuon(
        params,
        lr=0.0,
        momentum=0.0,
        beta2=0.0,
        weight_decay=0.0,
        ns_steps=1,
        compile=False,
    )


def test_a_tall_matrix_orthogonalizes_over_its_shorter_side() -> None:
    """A tall update is orthogonal along its columns, its only full-rank side.

    Its singular values sit at ``sqrt(rows / cols)`` rather than one: the step
    is scaled up so a tall matrix's per-element magnitude matches a square one.
    """
    params = _parameters((16, 8))
    before = params[0].detach().clone()
    NorMuon(params, lr=1.0, momentum=0.0, weight_decay=0.0, compile=False).step()
    update = before - params[0].detach()
    singular = torch.linalg.svdvals(update) / (16 / 8) ** 0.5
    assert singular.shape == (8,)
    assert float(singular.min()) > 0.7
    assert float(singular.max()) < 1.4


def test_step_evaluates_and_returns_the_closure() -> None:
    params = _parameters((4, 4))
    calls: list[int] = []

    def closure() -> float:
        calls.append(1)
        return 2.5

    loss = NorMuon(params, momentum=0.0, compile=False).step(closure)
    assert loss == 2.5
    assert calls == [1]


def test_a_parameter_without_a_gradient_is_left_alone() -> None:
    params = _parameters((4, 4), (4, 4))
    params[0].grad = None
    before = [parameter.detach().clone() for parameter in params]
    NorMuon(params, lr=0.1, momentum=0.0, compile=False).step()
    assert torch.equal(params[0].detach(), before[0])
    assert not torch.equal(params[1].detach(), before[1])


def test_shape_setup_preserves_dtype_and_reduces_over_the_short_axis() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 4, 3, dtype=torch.bfloat16))
    parameter.grad = torch.ones_like(parameter)
    optimizer = NorMuon(
        [parameter],
        lr=0.3,
        momentum=0.4,
        beta2=0.6,
        weight_decay=0.1,
        compile=False,
    )
    calls: list[tuple[Tensor, Tensor, int, int]] = []

    def capture(
        stacked_grads: Tensor,
        stacked_params: Tensor,
        momentum_buffer: Tensor,
        second_moment: Tensor,
        *,
        momentum: Tensor,
        lr: Tensor,
        weight_decay: Tensor,
        beta2: Tensor,
        ns_steps: int,
        reduce_dim: int,
        coefficients: tuple[tuple[float, float, float], ...],
    ) -> None:
        del (
            stacked_grads,
            stacked_params,
            momentum,
            lr,
            weight_decay,
            beta2,
            coefficients,
        )
        calls.append((momentum_buffer, second_moment, reduce_dim, ns_steps))

    optimizer._update = capture
    optimizer.step()

    assert len(calls) == 1
    assert calls[0][0].shape == (1, 2, 4, 3)
    assert calls[0][0].dtype == torch.bfloat16
    # One second moment per row of EACH matrix in the batch, not shared.
    assert calls[0][1].shape == (1, 2, 4, 1)
    assert calls[0][1].dtype == torch.bfloat16
    assert calls[0][2] == -1
    assert calls[0][3] == 5
    torch.testing.assert_close(
        optimizer._scalars["lr"],
        torch.tensor(0.3 * (4 / 3) ** 0.5),
    )


def test_wide_matrix_uses_columns_for_moments() -> None:
    parameter = torch.nn.Parameter(torch.ones(3, 4))
    parameter.grad = torch.ones_like(parameter)
    optimizer = NorMuon([parameter], lr=0.1, compile=False)
    captured: list[tuple[Tensor, Tensor, int]] = []
    _capture_kernel_state(
        optimizer,
        lambda momentum_buffer, second_moment, reduce_dim: captured.append(
            (momentum_buffer, second_moment, reduce_dim),
        ),
    )
    optimizer.step()
    assert len(captured) == 1
    assert captured[0][0].shape == (1, 3, 4)
    assert captured[0][1].shape == (1, 1, 4)
    assert captured[0][2] == -2


def test_compile_is_on_by_default_and_wraps_the_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reference compiles; matching its numerics means issuing that graph."""
    compiled: list[object] = []

    def fake_compile(fn: object, *, dynamic: bool) -> object:
        compiled.append((fn, dynamic))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    normuon._compiled_update.cache_clear()
    try:
        assert NorMuon.Config().compile is True
        optimizer = NorMuon(_parameters((4, 4)))
        assert compiled == [(normuon._normuon_update, False)]
        assert optimizer._update is normuon._normuon_update
    finally:
        normuon._compiled_update.cache_clear()


def test_bucket_membership_changes_do_not_move_optimizer_state() -> None:
    """Each parameter keeps its own history when a bucket-mate drops out.

    Buckets are rebuilt every step from the parameters holding a gradient, so
    a parameter missing one step changes who shares its bucket. Stepping two
    same-shape parameters through that must match stepping each alone.
    """
    shared = _parameters((2, 3), (2, 3))
    alone = [torch.nn.Parameter(p.detach().clone()) for p in shared]
    joint = NorMuon(shared, lr=0.1, compile=False)
    solo = [NorMuon([p], lr=0.1, compile=False) for p in alone]
    torch.manual_seed(1)
    grads = [[torch.randn(2, 3) for _ in shared] for _ in range(3)]
    for step, step_grads in enumerate(grads):
        for index, (a, b, grad) in enumerate(
            zip(shared, alone, step_grads, strict=True),
        ):
            dropped = step == 1 and index == 0
            a.grad = None if dropped else grad.clone()
            b.grad = None if dropped else grad.clone()
        joint.step()
        for optimizer, parameter in zip(solo, alone, strict=True):
            if parameter.grad is not None:
                optimizer.step()
    for a, b in zip(shared, alone, strict=True):
        assert torch.equal(a.detach(), b.detach())


def test_a_batched_matrix_parameter_steps_through_the_real_kernel() -> None:
    """A rank-3 parameter keeps one row moment per matrix in its batch."""
    parameter = torch.nn.Parameter(torch.randn(2, 4, 3))
    parameter.grad = torch.randn_like(parameter)
    optimizer = NorMuon([parameter], lr=0.1, compile=False)
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(parameter.detach(), before)
    second_moment = _state_tensor(optimizer, parameter, "second_moment")
    momentum_buffer = _state_tensor(optimizer, parameter, "momentum_buffer")
    assert second_moment.shape == (2, 4, 1)
    assert momentum_buffer.shape == (2, 4, 3)


_NAN: Final = float("nan")

_NON_FINITE: Final[list[tuple[str, _Build, str]]] = [
    (
        "lr_nan",
        lambda p: NorMuon(p, lr=_NAN, compile=False),
        "Learning rate must be finite",
    ),
    (
        "lr_inf",
        lambda p: NorMuon(p, lr=float("inf"), compile=False),
        "Learning rate must be finite",
    ),
    (
        "momentum_nan",
        lambda p: NorMuon(p, momentum=_NAN, compile=False),
        "Momentum must be finite",
    ),
    (
        "momentum_inf",
        lambda p: NorMuon(p, momentum=float("inf"), compile=False),
        "Momentum must be finite",
    ),
    (
        "beta2_nan",
        lambda p: NorMuon(p, beta2=_NAN, compile=False),
        "Beta2 must be finite",
    ),
    (
        "beta2_inf",
        lambda p: NorMuon(p, beta2=float("inf"), compile=False),
        "Beta2 must be finite",
    ),
    (
        "weight_decay_nan",
        lambda p: NorMuon(p, weight_decay=_NAN, compile=False),
        "Weight decay must be finite",
    ),
    (
        "weight_decay_inf",
        lambda p: NorMuon(p, weight_decay=float("inf"), compile=False),
        "Weight decay must be finite",
    ),
]


@pytest.mark.parametrize(
    ("name", "build", "message"),
    _NON_FINITE,
    ids=[name for name, _, _ in _NON_FINITE],
)
def test_non_finite_hyperparameter_is_rejected(
    name: str,
    build: _Build,
    message: str,
) -> None:
    """Non-finite values are rejected before they poison optimizer state."""
    del name
    with pytest.raises(ValueError, match=message):
        build(_parameters((2, 3)))


def test_a_prefix_of_the_coefficients_is_a_shorter_iteration() -> None:
    """``ns_steps`` selects a prefix, so fewer steps means a coarser factor.

    Pinned because the coefficients are jointly tuned for the full schedule:
    a change that reordered them would leave every other test green.
    """
    updates: list[Tensor] = []
    for ns_steps in (1, 5):
        params = _parameters((16, 17))
        before = params[0].detach().clone()
        NorMuon(
            params,
            lr=1.0,
            momentum=0.0,
            weight_decay=0.0,
            ns_steps=ns_steps,
            compile=False,
        ).step()
        updates.append(before - params[0].detach())
    coarse = torch.linalg.svdvals(updates[0])
    fine = torch.linalg.svdvals(updates[1])
    # The full schedule lands closer to every singular value being 1.
    assert float((fine - 1).abs().max()) < float((coarse - 1).abs().max())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
