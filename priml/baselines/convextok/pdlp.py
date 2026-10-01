"""PDLP as cuOpt runs it by default (Stable3): reflected Halpern PDHG with restarts.

The program is scaled (``scaling``) and the constant step fixed (``step_size``). Each
iteration then takes a reflected primal-dual step and pulls the result toward the last
restart point with Halpern weight ``(n + 1) / (n + 2)``, ``n`` iterations after that
restart. Termination is checked every 10 iterations below 1,000, every 100 below 10,000
and every 1,000 beyond, and at every 200th iteration, on the unscaled "potential next"
iterate. The 200th iterations also decide restarts by the fixed-point error, and a
restart updates the primal weight with cuOpt's PID controller.

Everything follows cuOpt's operations. The one departure is summation order, and on
the device cuOpt fuses multiply-adds where the port rounds twice, so iterates agree to
the last few digits rather than bitwise.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, NamedTuple

import math

from configgle import Fig
from torch import Tensor

import torch

from priml.baselines.convextok.program import LinearProgram
from priml.baselines.convextok.scaling import (
    ScaledProgram,
    bound_norm,
    scale_program,
)
from priml.baselines.convextok.spmv import CsrMatrix
from priml.baselines.convextok.step_size import initial_step_size


@dataclass(frozen=True, slots=True, kw_only=True)
class PdlpResult:
    """Where PDLP stopped.

    Attributes:
      primal: Unscaled primal iterate at the last check (cuOpt's returned solution).
      dual: Unscaled dual iterate at the last check.
      reduced_cost: Unscaled reduced costs of that iterate.
      current_primal: Scaled current primal iterate (cuOpt's warm-start state).
      current_dual: Scaled current dual iterate.
      iterations: Iterations taken.
      optimal: Whether the tolerances were met, rather than the iteration limit.
      step_size: The constant step size.
      primal_weight: The primal weight at the stop.

    """

    primal: Tensor
    dual: Tensor
    reduced_cost: Tensor
    current_primal: Tensor
    current_dual: Tensor
    iterations: int
    optimal: bool
    step_size: float
    primal_weight: float


class Pdlp:
    """Solve a linear program with cuOpt's default PDLP."""

    class Config(Fig["Pdlp"]):
        """cuOpt's Stable3 constants; every field is always meaningful to PDLP."""

        tolerance: float = 1e-4
        """Absolute and relative tolerance on both residuals and the gap."""

        iteration_limit: int | None = None
        """Stop at the first check at or past this iteration; None runs to optimality."""

        ruiz_iterations: int = 10
        """Ruiz scaling passes."""

        restart_period: int = 200
        """Iterations between restart decisions."""

        sufficient_reduction: float = 0.2
        """Restart once the fixed-point error falls to this fraction of its first value."""

        necessary_reduction: float = 0.8
        """Restart below this fraction if the error rose since the last decision."""

        artificial_restart: float = 0.36
        """Restart once this fraction of all iterations passed since the last one."""

        pid_proportional: float = 0.99
        """Primal-weight controller's proportional gain."""

        pid_integral: float = 0.01
        """Primal-weight controller's integral gain."""

        pid_derivative: float = 0.0
        """Primal-weight controller's derivative gain."""

        pid_integral_smoothing: float = 0.3
        """Decay of the controller's integrated error at each restart."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, program: LinearProgram) -> PdlpResult:
        """Solve ``program`` from the zero point.

        Args:
          program: The program; its matrix, bounds and objective on one device.

        Returns:
          result: The last checked iterate and the solver's state there.

        """
        config = self.config
        scaled = scale_program(program, ruiz_iterations=config.ruiz_iterations)
        step = initial_step_size(scaled)
        problem = scaled.program
        updates = _compiled_updates() if problem.values.is_cuda else _UPDATES
        termination = _Termination(program, scaled)
        weight = _PrimalWeight()
        zeros_primal = torch.zeros_like(problem.objective)
        zeros_dual = torch.zeros_like(problem.row_lower)
        x = torch.maximum(torch.minimum(zeros_primal, problem.upper), problem.lower)
        y = zeros_dual
        anchor_x, anchor_y = zeros_primal, zeros_dual
        next_x, next_y, slack = zeros_primal, zeros_dual, zeros_primal
        fixed_point_error = initial_error = math.nan
        last_trial_error = math.inf
        since_restart = 0
        iteration = 0
        while True:
            major = iteration % config.restart_period == 0
            restarted = False
            if major or iteration % _check_interval(iteration) == 0:
                primal, dual, reduced_cost = scaled_to_original(
                    scaled,
                    next_x,
                    next_y,
                    slack,
                )
                check = termination(primal, dual, reduced_cost)
                optimal = iteration > 1 and check.optimal(config.tolerance)
                limited = (
                    config.iteration_limit is not None
                    and iteration >= config.iteration_limit
                )
                if optimal or limited:
                    return PdlpResult(
                        primal=primal,
                        dual=dual,
                        reduced_cost=reduced_cost,
                        current_primal=x,
                        current_dual=y,
                        iterations=iteration,
                        optimal=optimal,
                        step_size=step,
                        primal_weight=weight.value,
                    )
                next_x, next_y, slack = original_to_scaled(
                    scaled,
                    primal,
                    dual,
                    reduced_cost,
                )
                if major:
                    after_first = iteration > config.restart_period
                    restarted = iteration == config.restart_period or (
                        after_first
                        and (
                            fixed_point_error
                            <= config.sufficient_reduction * initial_error
                            or (
                                fixed_point_error
                                <= config.necessary_reduction * initial_error
                                and fixed_point_error > last_trial_error
                            )
                            or since_restart >= config.artificial_restart * iteration
                        )
                    )
                    last_trial_error = fixed_point_error
                if restarted:
                    weight.update(
                        float(torch.dot(next_x - anchor_x, next_x - anchor_x)),
                        float(torch.dot(next_y - anchor_y, next_y - anchor_y)),
                        check,
                        config,
                    )
                    x = anchor_x = next_x
                    y = anchor_y = next_y
                    since_restart = 0
                    last_trial_error = math.inf

            primal_step = step / weight.value
            dual_step = step * weight.value
            store = (iteration + 1) % config.restart_period == 0 or (
                iteration + 2
            ) % _check_interval(iteration + 2) == 0
            aty = scaled.transpose @ y
            # Only a step before a check keeps the projected iterate; the others
            # write just the reflection, as cuOpt's do.
            primal_args: tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor], float] = (
                x,
                problem.objective,
                aty,
                (problem.lower, problem.upper),
                primal_step,
            )
            if store:
                next_x, slack, reflected_x = updates.primal(*primal_args)
            else:
                reflected_x = updates.primal_reflection(*primal_args)
            dual_args: tuple[Tensor, Tensor, Tensor, Tensor, float] = (
                y,
                scaled.matrix @ reflected_x,
                problem.row_lower,
                problem.row_upper,
                dual_step,
            )
            if store:
                next_y, reflected_y = updates.dual(*dual_args)
            else:
                reflected_y = updates.dual_reflection(*dual_args)
            if (iteration + 1) % config.restart_period == 0 or restarted:
                delta_x, delta_y = reflected_x - x, reflected_y - y
                interaction = float(
                    torch.dot(scaled.transpose @ reflected_y - aty, delta_x),
                )
                movement = (
                    float(torch.dot(delta_x, delta_x)) * weight.value
                    + float(torch.dot(delta_y, delta_y)) / weight.value
                )
                fixed_point_error = max(0.0, movement + 2.0 * interaction * step) ** 0.5
                if restarted:
                    initial_error = fixed_point_error
            halpern = (since_restart + 1) / (since_restart + 2)
            x = updates.halpern(reflected_x, anchor_x, halpern)
            y = updates.halpern(reflected_y, anchor_y, halpern)
            iteration += 1
            since_restart += 1


def scaled_to_original(
    scaled: ScaledProgram,
    primal: Tensor,
    dual: Tensor,
    reduced_cost: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map a scaled iterate back to the original program, as cuOpt unscales.

    Args:
      scaled: The scaling that produced the iterate's space.
      primal: Scaled primal values.
      dual: Scaled dual values.
      reduced_cost: Scaled reduced costs.

    Returns:
      primal: Original primal values.
      dual: Original dual values.
      reduced_cost: Original reduced costs.

    """
    return (
        (primal * scaled.column_scale) * (1.0 / scaled.bound_rescaling),
        (dual * scaled.row_scale) * (1.0 / scaled.objective_rescaling),
        (reduced_cost / scaled.column_scale) * (1.0 / scaled.objective_rescaling),
    )


def original_to_scaled(
    scaled: ScaledProgram,
    primal: Tensor,
    dual: Tensor,
    reduced_cost: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map an original iterate into the scaled space, as cuOpt scales.

    Args:
      scaled: The scaling to apply.
      primal: Original primal values.
      dual: Original dual values.
      reduced_cost: Original reduced costs.

    Returns:
      primal: Scaled primal values.
      dual: Scaled dual values.
      reduced_cost: Scaled reduced costs.

    """
    return (
        (primal / scaled.column_scale) * scaled.bound_rescaling,
        (dual / scaled.row_scale) * scaled.objective_rescaling,
        (reduced_cost * scaled.column_scale) * scaled.objective_rescaling,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Check:
    """cuOpt's convergence information for one iterate of the original program."""

    gap: float
    absolute_objective: float
    primal_residual: float
    dual_residual: float
    bound_norm: float
    objective_norm: float

    def optimal(self, tolerance: float) -> bool:
        """Return whether both residuals and the gap meet ``tolerance``."""
        return (
            self.gap <= tolerance + tolerance * self.absolute_objective
            and self.primal_residual <= tolerance + tolerance * self.bound_norm
            and self.dual_residual <= tolerance + tolerance * self.objective_norm
        )


class _Termination:
    """Evaluate iterates on the original program, as cuOpt's termination check does."""

    def __init__(self, program: LinearProgram, scaled: ScaledProgram) -> None:
        self.program = program
        self.matrix = CsrMatrix(
            program.crow_indices,
            program.col_indices,
            program.values,
        )
        self.transpose = CsrMatrix(
            scaled.transpose.crow_indices,
            scaled.transpose.col_indices,
            program.values[scaled.transpose_order],
        )
        self.bound_norm = float(bound_norm(program.row_lower, program.row_upper))
        self.objective_norm = float(torch.linalg.vector_norm(program.objective))
        self.finite_lower = torch.where(
            torch.isfinite(program.row_lower),
            program.row_lower,
            0.0,
        )
        self.finite_upper = torch.where(
            torch.isfinite(program.row_upper),
            program.row_upper,
            0.0,
        )

    def __call__(self, primal: Tensor, dual: Tensor, reduced_cost: Tensor) -> _Check:
        program = self.program
        activity = self.matrix @ primal
        clamped = torch.maximum(
            program.row_lower,
            torch.minimum(activity, program.row_upper),
        )
        primal_residual = activity - clamped
        bound_value = (
            dual.clamp(min=0.0) * self.finite_lower
            + dual.clamp(max=0.0) * self.finite_upper
        )
        dual_residual = (program.objective - self.transpose @ dual) - reduced_cost
        offset = program.objective_offset
        primal_objective = float(torch.dot(primal, program.objective)) + offset
        dual_objective = (
            float(torch.dot(reduced_cost, primal)) + float(bound_value.sum())
        ) + offset
        return _Check(
            gap=abs(primal_objective - dual_objective),
            absolute_objective=abs(primal_objective) + abs(dual_objective),
            primal_residual=float(torch.linalg.vector_norm(primal_residual)),
            dual_residual=float(torch.linalg.vector_norm(dual_residual)),
            bound_norm=self.bound_norm,
            objective_norm=self.objective_norm,
        )


@dataclass(slots=True, kw_only=True)
class _PrimalWeight:
    """cuOpt's PID-controlled primal weight, updated at each restart on the host."""

    value: float = 1.0
    best: float = 1.0
    error_sum: float = 0.0
    last_error: float = 0.0
    best_gap: float = math.inf

    def update(
        self,
        primal_distance: float,
        dual_distance: float,
        check: _Check,
        config: Pdlp.Config,
    ) -> None:
        """Steer the weight toward balanced primal and dual movement.

        Args:
          primal_distance: Squared distance the primal iterate moved since the last restart.
          dual_distance: Squared distance the dual iterate moved.
          check: Convergence information at this restart.
          config: The controller's gains.

        """
        relative_dual = check.dual_residual / (1.0 + check.objective_norm)
        relative_primal = check.primal_residual / (1.0 + check.bound_norm)
        ratio = math.inf if relative_primal == 0.0 else relative_dual / relative_primal
        primal_norm, dual_norm = primal_distance**0.5, dual_distance**0.5
        if (
            1e-16 < primal_norm < 1e12
            and 1e-16 < dual_norm < 1e12
            and 1e-8 < ratio < 1e8
        ):
            error = math.log(dual_norm) - math.log(primal_norm) - math.log(self.value)
            self.error_sum = self.error_sum * config.pid_integral_smoothing + error
            delta = error - self.last_error
            self.value = (
                math.exp(
                    config.pid_proportional * error
                    + config.pid_integral * self.error_sum
                    + config.pid_derivative * delta,
                )
                * self.value
            )
            self.last_error = error
        else:
            self.value = self.best
            self.error_sum = 0.0
            self.last_error = 0.0
        residual_gap = math.inf if ratio == 0.0 else abs(math.log10(ratio))
        if residual_gap < self.best_gap:
            self.best_gap = residual_gap
            self.best = self.value


def _check_interval(iteration: int) -> int:
    """Return the iterations between termination checks: 10 below 1,000, then tenfold per decade."""
    interval, threshold = 10, 1000
    while iteration >= threshold:
        interval *= 10
        threshold *= 10
    return interval


class _Updates(NamedTuple):
    """The elementwise parts of one PDHG iteration."""

    primal: Callable[
        [Tensor, Tensor, Tensor, tuple[Tensor, Tensor], float],
        tuple[Tensor, Tensor, Tensor],
    ]
    primal_reflection: Callable[
        [Tensor, Tensor, Tensor, tuple[Tensor, Tensor], float],
        Tensor,
    ]
    dual: Callable[[Tensor, Tensor, Tensor, Tensor, float], tuple[Tensor, Tensor]]
    dual_reflection: Callable[[Tensor, Tensor, Tensor, Tensor, float], Tensor]
    halpern: Callable[[Tensor, Tensor, float], Tensor]


def _primal_update(
    x: Tensor,
    objective: Tensor,
    aty: Tensor,
    bounds: tuple[Tensor, Tensor],
    step: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Take a projected primal step; return it, its reduced costs, and its reflection."""
    lower, upper = bounds
    unprojected = x - step * (objective - aty)
    projected = torch.maximum(torch.minimum(unprojected, upper), lower)
    return projected, (projected - unprojected) / step, 2.0 * projected - x


def _dual_update(
    y: Tensor,
    activity: Tensor,
    row_lower: Tensor,
    row_upper: Tensor,
    step: float,
) -> tuple[Tensor, Tensor]:
    """Take a projected dual step; return it and its reflection."""
    shifted = y / step - activity
    clipped = torch.maximum(-row_upper, torch.minimum(shifted, -row_lower))
    dual_next = (shifted - clipped) * step
    return dual_next, 2.0 * dual_next - y


def _primal_reflection(
    x: Tensor,
    objective: Tensor,
    aty: Tensor,
    bounds: tuple[Tensor, Tensor],
    step: float,
) -> Tensor:
    """Take a projected primal step and return only its reflection."""
    return _primal_update(x, objective, aty, bounds, step)[2]


def _dual_reflection(
    y: Tensor,
    activity: Tensor,
    row_lower: Tensor,
    row_upper: Tensor,
    step: float,
) -> Tensor:
    """Take a projected dual step and return only its reflection."""
    return _dual_update(y, activity, row_lower, row_upper, step)[1]


def _halpern_update(reflected: Tensor, anchor: Tensor, weight: float) -> Tensor:
    """Pull a reflected iterate toward the last restart point."""
    return weight * reflected + (1.0 - weight) * anchor


_UPDATES: Final = _Updates(
    _primal_update,
    _primal_reflection,
    _dual_update,
    _dual_reflection,
    _halpern_update,
)


@lru_cache(maxsize=1)
def _compiled_updates() -> _Updates:
    """Fuse each update into one kernel; the eager updates remain the CPU reference."""
    return _Updates(
        torch.compile(_primal_update, fullgraph=True),
        torch.compile(_primal_reflection, fullgraph=True),
        torch.compile(_dual_update, fullgraph=True),
        torch.compile(_dual_reflection, fullgraph=True),
        torch.compile(_halpern_update, fullgraph=True),
    )
