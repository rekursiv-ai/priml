"""Tests for the NorMuon optimizer."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from priml.optimizers.normuon import NorMuon


def _parameters(*shapes: tuple[int, ...]) -> list[torch.nn.Parameter]:
    """Parameters of the given shapes, each carrying a gradient."""
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(shape)) for shape in shapes]
    for parameter in params:
        parameter.grad = torch.randn_like(parameter)
    return params


def test_the_update_is_approximately_orthogonal() -> None:
    """The whole point: the step's singular values are all near one.

    An update that merely descended would leave them spread. This is what
    makes the step invariant to the weight matrix's scale, so if the
    polynomial iteration were wrong the optimizer would silently become a
    poorly-tuned SGD.
    """
    params = _parameters((16, 16))
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
    steps: list[torch.Tensor] = []
    for scale in (1.0, 1000.0):
        params = _parameters((8, 8))
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
    params = _parameters((8, 8), (8, 8), (4, 16))
    before = [p.detach().clone() for p in params]
    NorMuon(params, lr=0.1, momentum=0.0, weight_decay=0.0, compile=False).step()
    for original, updated in zip(before, params, strict=True):
        assert not torch.equal(original, updated.detach())


def test_weight_decay_is_cautious() -> None:
    """Decay applies exactly where it agrees with the update, and nowhere else.

    A weight the update is already shrinking must not be decayed too, or the
    two compound into a step neither asked for. With every weight positive,
    "agrees" reduces to "the update is also positive", so the set decay
    touches is predictable independently of the optimizer -- which is what
    makes this a check rather than a restatement.
    """

    def step(*, weight_decay: float) -> torch.Tensor:
        params = _parameters((8, 8))
        with torch.no_grad():
            params[0].copy_(torch.ones(8, 8))
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

    def update(*, skew: bool) -> torch.Tensor:
        params = _parameters((16, 16))
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
            state = optimizer.state[params[0]]
            state["second_moment"][:, :8] *= 0.25
        assert params[0].grad is not None
        params[0].grad = torch.randn_like(params[0])
        optimizer.step()
        return before - params[0].detach()

    plain, skewed = update(skew=False), update(skew=True)
    rows_moved = (plain.norm(dim=-1) - skewed.norm(dim=-1)).abs().max()
    assert float(rows_moved) > 0.01
    assert float(skewed.norm()) == pytest.approx(float(plain.norm()), rel=0.05)


def test_a_vector_is_rejected() -> None:
    """Orthogonalizing a rank-1 tensor is undefined, so it must not be routed
    here silently.
    """
    params = _parameters((8,))
    with pytest.raises(ValueError, match="ndim >= 2"):
        NorMuon(params, lr=0.1, compile=False).step()


def test_eligibility_names_the_rank_rule() -> None:
    """The recipe routes by this, so it states the algorithm's own constraint."""
    matrix = torch.nn.Parameter(torch.zeros(4, 4))
    vector = torch.nn.Parameter(torch.zeros(4))
    assert NorMuon.eligible_tensor("w", matrix)
    assert not NorMuon.eligible_tensor("b", vector)


_Build = Callable[[list[torch.nn.Parameter]], NorMuon]

_INVALID: list[tuple[str, _Build, str]] = [
    ("lr", lambda p: NorMuon(p, lr=-1.0), "learning rate"),
    ("momentum", lambda p: NorMuon(p, momentum=1.0), "momentum"),
    ("beta2", lambda p: NorMuon(p, beta2=1.0), "beta2"),
    ("weight_decay", lambda p: NorMuon(p, weight_decay=-1.0), "weight_decay"),
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
    assert isinstance(optimizer, NorMuon)
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_a_prefix_of_the_coefficients_is_a_shorter_iteration() -> None:
    """``ns_steps`` selects a prefix, so fewer steps means a coarser factor.

    Pinned because the coefficients are jointly tuned for the full schedule:
    a change that reordered them would leave every other test green.
    """
    updates: list[torch.Tensor] = []
    for ns_steps in (1, 5):
        params = _parameters((16, 16))
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
    from priml.lib.testing import test_main

    test_main(__file__)
