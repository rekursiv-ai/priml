"""
Training step utilities and evaluation metrics for Feynman SR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    nmse: float
    exact_recovery: bool
    complexity: int
    expression_str: str
    true_formula: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nmse": self.nmse,
            "exact_recovery": self.exact_recovery,
            "complexity": self.complexity,
            "expression": self.expression_str,
            "true_formula": self.true_formula,
        }


def compute_nmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Normalized Mean Squared Error."""
    pred = pred.flatten() if pred.ndim > 1 else pred
    target = target.flatten() if target.ndim > 1 else target

    if pred.shape != target.shape:
        return float("inf")

    if not np.all(np.isfinite(pred)):
        return float("inf")

    mse = np.mean((pred - target) ** 2)
    var = np.var(target)
    if var == 0:
        return float("inf")
    return mse / var


def check_exact_recovery(
    expr_str: str, true_formula: str, tolerance: float = 0.01
) -> bool:
    """
    Check if predicted expression matches true formula.
    Uses sympy for symbolic equality with constant tolerance.
    """
    try:
        import sympy as sp

        # Parse both expressions
        pred_expr = sp.sympify(expr_str.replace("^", "**"))
        true_expr = sp.sympify(true_formula.replace("^", "**"))

        # Check exact symbolic equality
        if sp.simplify(pred_expr - true_expr) == 0:
            return True

        # Check with constant tolerance (extract constants and compare)
        pred_consts = pred_expr.atoms(sp.Number)
        true_consts = true_expr.atoms(sp.Number)

        # If structure matches but constants differ within tolerance
        pred_structure = pred_expr.replace(
            lambda x: x.is_Number, lambda x: sp.Symbol("C")
        )
        true_structure = true_expr.replace(
            lambda x: x.is_Number, lambda x: sp.Symbol("C")
        )

        if pred_structure == true_structure:
            # Compare constants
            pred_vals = [float(c) for c in pred_consts]
            true_vals = [float(c) for c in true_consts]
            if len(pred_vals) == len(true_vals):
                for pv, tv in zip(pred_vals, true_vals):
                    if tv != 0:
                        if abs(pv - tv) / abs(tv) > tolerance:
                            return False
                    elif abs(pv) > tolerance:
                        return False
                return True

        return False
    except Exception:  # noqa: BLE001
        return False


def evaluate_expression(
    expr, variables: dict[str, np.ndarray], target: np.ndarray, true_formula: str
) -> EvaluationMetrics:
    """Evaluate a single expression and return metrics."""
    pred = expr.evaluate(variables)
    nmse = compute_nmse(pred, target)
    exact = check_exact_recovery(str(expr), true_formula)
    complexity = expr.count_nodes()

    return EvaluationMetrics(
        nmse=nmse,
        exact_recovery=exact,
        complexity=complexity,
        expression_str=str(expr),
        true_formula=true_formula,
    )


def aggregate_results(results: list[EvaluationMetrics]) -> dict[str, Any]:
    """Aggregate metrics across multiple equations."""
    if not results:
        return {"recovery_rate": 0.0, "mean_nmse": float("inf"), "mean_complexity": 0}

    exact_count = sum(1 for r in results if r.exact_recovery)
    recovery_rate = exact_count / len(results)

    valid_nmse = [r.nmse for r in results if np.isfinite(r.nmse)]
    mean_nmse = np.mean(valid_nmse) if valid_nmse else float("inf")

    mean_complexity = np.mean([r.complexity for r in results])

    return {
        "recovery_rate": recovery_rate,
        "exact_recoveries": exact_count,
        "total": len(results),
        "mean_nmse": mean_nmse,
        "mean_complexity": mean_complexity,
        "results": [r.to_dict() for r in results],
    }
