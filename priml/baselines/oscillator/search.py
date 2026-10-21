"""Derivative-free searches that fit the oscillator's simulator coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import math

from configgle import Fig


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")  # ~1050 ms; only the least-squares fit needs it.


# Brent's machine epsilon as the reference fminbound spells it, not float64's
# exact epsilon; only this value reproduces its iterates bit for bit.
_SQRT_EPS: Final = float(2.2e-16**0.5)
_GOLDEN: Final = 0.5 * (3 - float(5**0.5))


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchResult[X]:
    """The best point a search evaluated, and whether a stopping rule ended it."""

    x: X
    success: bool


class BoundedScalarSearch:
    """Minimize a function of one variable on an interval without derivatives."""

    class Config(Fig["BoundedScalarSearch"]):
        bounds: tuple[float, float] = (0.0, 1.0)
        """Interval searched; its endpoints are never evaluated."""

        xatol: float = 1e-6
        """Absolute tolerance on the minimizer's location."""

        max_evaluations: int = 64
        """Objective evaluations before the search reports failure."""

    def __init__(self, config: Config) -> None:
        low, high = config.bounds
        if not (math.isfinite(low) and math.isfinite(high)) or low > high:
            raise ValueError("Search bounds must be finite and ordered.")
        self.config = config

    def __call__(self, fun: Callable[[float], float], /) -> SearchResult[float]:
        """Search by golden-section and parabolic steps, keeping the best point.

        Args:
          fun: Objective to minimize.

        Returns:
          result: The lowest evaluated point. It is unsuccessful when the
            evaluation budget runs out first or the lowest value is NaN.

        References:
          https://maths-people.anu.edu.au/~brent/pub/pub011.html
            Brent, Algorithms for Minimization without Derivatives (1973), ch. 5.

        """
        low, high = self.config.bounds
        x = low + _GOLDEN * (high - low)
        fx = fun(x)
        bracket = _Bracket(low=low, high=high, x=x, w=x, fx=fx, fw=fx)
        evaluations = 1
        while not bracket.converged(self.config.xatol):
            if evaluations >= self.config.max_evaluations:
                return SearchResult(x=bracket.x, success=False)
            u = bracket.propose(self.config.xatol)
            bracket.update(u, fu=fun(u))
            evaluations += 1
        return SearchResult(x=bracket.x, success=not math.isnan(bracket.fx))


class BoundedLeastSquares:
    """Fit parameters inside a box by Levenberg-Marquardt."""

    class Config(Fig["BoundedLeastSquares"]):
        bounds: tuple[float, float] = (0.0, 2.0)
        """Lower and upper limit shared by every parameter."""

        max_evaluations: int = 40
        """Residual evaluations before failing; Jacobian columns are not counted."""

        ftol: float = 1e-9
        """Stop once a step lowers the cost by less than this fraction of it."""

        xtol: float = 1e-9
        """Stop once a step is shorter than this fraction of the parameters."""

        gtol: float = 1e-9
        """Stop once every free gradient component is smaller than this."""

    def __init__(self, config: Config) -> None:
        low, high = config.bounds
        if math.isnan(low) or math.isnan(high) or low >= high:
            raise ValueError("Least-squares bounds must be ordered numbers.")
        self.config = config

    def __call__(
        self,
        fun: Callable[[Tensor], Tensor],
        x0: Tensor,
        /,
    ) -> SearchResult[Tensor]:
        """Minimize half the squared residual norm from a start inside the box.

        A parameter on a bound whose gradient points out of the box is held
        there; the damped Gauss-Newton step moves the rest and is clipped to
        the box. Jacobians are forward differences, taken backward where a
        forward step would leave the box.

        Args:
          fun: Residuals as a function of the parameters.
          x0: Starting parameters, within the bounds.

        Returns:
          result: The lowest-cost parameters accepted. It is unsuccessful when
            the evaluation budget runs out before a tolerance is met, as it
            does from a start whose residuals are not finite.

        References:
          https://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf
            Madsen, Nielsen and Tingleff, Methods for Non-Linear Least
            Squares Problems (2004), Algorithm 3.16.

        """
        cfg = self.config
        low, high = cfg.bounds
        x, residuals = x0, fun(x0)
        cost = float(residuals.square().sum()) / 2
        evaluations = 1
        jacobian = _jacobian(fun, x, residuals=residuals, high=high)
        damping, growth = 1e-3 * float(jacobian.square().sum(dim=0).max()), 2.0
        while True:
            gradient = jacobian.T @ residuals
            held = ((x <= low) & (gradient > 0)) | ((x >= high) & (gradient < 0))
            if float(gradient.masked_fill(held, 0).abs().max()) < cfg.gtol:
                return SearchResult(x=x, success=True)
            if evaluations >= cfg.max_evaluations:
                return SearchResult(x=x, success=False)
            free = jacobian.masked_fill(held, 0)
            trial = x + _damped_step(free, residuals=residuals, damping=damping)
            trial = trial.clamp(low, high)
            trial_residuals = fun(trial)
            evaluations += 1
            trial_cost = float(trial_residuals.square().sum()) / 2
            step = trial - x
            predicted = -float(gradient @ step + (jacobian @ step).square().sum() / 2)
            ratio = (cost - trial_cost) / predicted if predicted > 0 else 0.0
            accepted = trial_cost < cost
            small_gain = cost - trial_cost < cfg.ftol * cost and ratio > 0.25
            small_step = float(step.norm()) < cfg.xtol * (cfg.xtol + float(x.norm()))
            if small_gain or small_step:
                return SearchResult(x=trial if accepted else x, success=True)
            if accepted:
                x, residuals, cost = trial, trial_residuals, trial_cost
                jacobian = _jacobian(fun, x, residuals=residuals, high=high)
                damping *= max(1 / 3, 1 - (2 * ratio - 1) ** 3)
                growth = 2.0
            else:
                damping *= growth
                growth *= 2


@dataclass(slots=True, kw_only=True)
class _Bracket:
    """Brent's interval, its lowest point x, the runner-up w, and the one before v."""

    low: float
    high: float
    x: float
    w: float
    fx: float
    fw: float
    # The first update copies w into v before any parabola reads it.
    v: float = math.nan
    fv: float = math.nan
    step: float = 0.0
    previous: float = 0.0

    def converged(self, xatol: float) -> bool:
        tol = _SQRT_EPS * abs(self.x) + xatol / 3.0
        return abs(self.x - 0.5 * (self.low + self.high)) <= (
            2.0 * tol - 0.5 * (self.high - self.low)
        )

    def propose(self, xatol: float) -> float:
        """Take a parabolic step through x, w, and v when acceptable, else golden.

        Args:
          xatol: Absolute tolerance; no step is shorter than about a third of it.

        Returns:
          u: The next point to evaluate, strictly inside the interval.

        """
        tol = _SQRT_EPS * abs(self.x) + xatol / 3.0
        middle = 0.5 * (self.low + self.high)
        if abs(self.previous) <= tol or not self._parabola(tol, middle=middle):
            self.previous = (self.low if self.x >= middle else self.high) - self.x
            self.step = _GOLDEN * self.previous
        if self.step >= 0:
            return self.x + max(abs(self.step), tol)
        return self.x - max(abs(self.step), tol)

    def update(self, u: float, *, fu: float) -> None:
        """Shrink the interval around the lowest point and rank the new one.

        Args:
          u: The point just evaluated.
          fu: Its objective value.

        """
        # Propose moved right of x exactly when its step was nonnegative.
        right = self.step >= 0
        if fu <= self.fx:
            if right:
                self.low = self.x
            else:
                self.high = self.x
            self.v, self.fv = self.w, self.fw
            self.w, self.fw = self.x, self.fx
            self.x, self.fx = u, fu
            return
        if right:
            self.high = u
        else:
            self.low = u
        if fu <= self.fw or self.w == self.x:
            self.v, self.fv = self.w, self.fw
            self.w, self.fw = u, fu
        elif fu <= self.fv or self.v in {self.x, self.w}:
            self.v, self.fv = u, fu

    def _parabola(self, tol: float, *, middle: float) -> bool:
        """Step to the vertex through x, w, and v when it lies well inside."""
        r = (self.x - self.w) * (self.fx - self.fv)
        q = (self.x - self.v) * (self.fx - self.fw)
        p = (self.x - self.v) * q - (self.x - self.w) * r
        q = 2.0 * (q - r)
        limit, self.previous = self.previous, self.step
        # The vertex is at x - p / q. Scaling the interval by q, rather than
        # dividing by it, keeps the reference's rounding for either sign of q.
        ends = (q * (self.low - self.x), q * (self.high - self.x))
        if abs(p) < abs(0.5 * q * limit) and min(ends) < -p < max(ends):
            self.step = -p / q
            u = self.x + self.step
            if u - self.low < 2.0 * tol or self.high - u < 2.0 * tol:
                self.step = tol if middle >= self.x else -tol
            return True
        return False


def _jacobian(
    fun: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    residuals: Tensor,
    high: float,
) -> Tensor:
    """Difference each parameter forward, or backward where forward leaves the box."""
    step = torch.finfo(x.dtype).eps ** 0.5 * x.abs().clamp(min=1)
    step = step.where(x + step <= high, -step)
    return torch.column_stack(
        [
            (fun(x + shift) - residuals) / shift[index]
            for index, shift in enumerate(torch.diag(step))
        ],
    )


def _damped_step(jacobian: Tensor, *, residuals: Tensor, damping: float) -> Tensor:
    """Solve the damped normal equations; a zero column holds its parameter."""
    hessian = jacobian.T @ jacobian
    hessian.diagonal().add_(damping)
    return torch.linalg.solve(hessian, -(jacobian.T @ residuals))
