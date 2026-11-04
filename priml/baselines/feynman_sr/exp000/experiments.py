"""Feynman SR Experiments: configgle configurations for exp000+.

Following priml pattern: one change per fork, frozen baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Import our model and data configs
from .data import (
    FeynmanEquation,
    generate_data_for_equation,
    load_feynman_10,
    load_feynman_full,
)
from .model import FeynmanSRModel, GAConfig

# =============================================================================
# BASE CONFIG CLASSES
# =============================================================================


@dataclass
class FeynmanTrainStep:
    """Single training/evaluation step for one equation."""

    model: FeynmanSRModel = field(default_factory=FeynmanSRModel)
    equation: FeynmanEquation | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    target_values: Any = None
    n_samples: int = 1000
    seed: int = 42

    def run(self) -> dict:
        """Execute one equation evaluation."""
        if self.equation is None:
            msg = "No equation set"
            raise ValueError(msg)
        if self.target_values is None:
            self.variables, self.target_values = generate_data_for_equation(
                self.equation,
                n_samples=self.n_samples,
                seed=self.seed,
            )
        best_expr, nmse, info = self.model.solve_equation(
            self.equation,
            self.variables,
            self.target_values,
        )
        # Check exact recovery using model's method
        exact = self.model._check_exact_recovery(
            best_expr,
            self.equation.formula,
        )
        return {
            "expression": str(best_expr),
            "nmse": nmse,
            "complexity": info["complexity"],
            "exact": exact,
        }


@dataclass
class FeynmanTrainLoop:
    """Main training loop: evaluates model on dataset split."""

    step: FeynmanTrainStep = field(default_factory=FeynmanTrainStep)
    dataset_split: str = "subset10"  # "subset10", "easy", "medium", "hard", "all"
    n_samples: int = 1000
    seed: int = 42

    def run(self) -> dict:
        """Run evaluation on full dataset split."""
        if self.dataset_split == "subset10":
            dataset = load_feynman_10()
        else:
            dataset = load_feynman_full().get_split(self.dataset_split)

        results = []
        exact_count = 0

        for i, eq in enumerate(dataset):
            step_cfg = FeynmanTrainStep(
                model=self.step.model,
                equation=eq,
                n_samples=self.n_samples,
                seed=self.seed + i,  # Different seed per equation
            )
            result = step_cfg.run()
            results.append({**result, "eq_id": eq.eq_id, "true_formula": eq.formula})
            if result.get("exact", False):
                exact_count += 1

        recovery_rate = exact_count / len(dataset) if dataset else 0.0
        return {
            "recovery_rate": recovery_rate,
            "exact_recoveries": exact_count,
            "total_equations": len(dataset),
            "results": results,
            "dataset_split": self.dataset_split,
        }


# =============================================================================
# EXPERIMENT CONFIGS (exp000 = baseline, each fork changes ONE thing)
# =============================================================================


def exp000() -> FeynmanTrainLoop:
    """Baseline: Dimensional GA + LM constant optimization.

    - Population: 200, Generations: 50
    - 10-equation subset (fast iteration)
    - Seed: 42 (bit-for-bit reproducible).
    """
    cfg = FeynmanTrainLoop()
    cfg.dataset_split = "subset10"
    cfg.n_samples = 1000
    cfg.seed = 42
    cfg.step.model = FeynmanSRModel(
        ga_config=GAConfig(
            population_size=200,
            max_generations=50,
            cxpb=0.6,
            mutpb=0.3,
            tournsize=3,
            hall_of_fame_size=5,
            seed=42,
            max_nodes=30,
        ),
    )
    cfg.step.n_samples = 1000
    cfg.step.seed = 42
    return cfg


def exp001() -> FeynmanTrainLoop:
    """Fork: Increase generations to 100 for better convergence.

    Hypothesis: More generations → higher recovery rate.
    """
    cfg = exp000()
    cfg.step.model.ga_config.max_generations = 100
    return cfg


def exp002() -> FeynmanTrainLoop:
    """Fork: Increase population to 500 for better exploration.

    Hypothesis: Larger population → better diversity → better recovery.
    """
    cfg = exp000()
    cfg.step.model.ga_config.population_size = 500
    return cfg


def exp003() -> FeynmanTrainLoop:
    """Fork: Run on full Easy split (30 equations).

    Hypothesis: Model generalizes to full Easy set.
    """
    cfg = exp000()
    cfg.dataset_split = "easy"
    return cfg


def exp004() -> FeynmanTrainLoop:
    """Fork: Add MCTS-guided search (future exp004+).

    Hypothesis: Hypothesis pinning + MCTS → better search efficiency.
    """
    return exp000()
    # TODO: Add MCTS config when implemented


def exp005() -> FeynmanTrainLoop:
    """Fork: Neural-guided expression generation (future exp005+).

    Hypothesis: Neural prior → better initial population → faster convergence.
    """
    return exp000()
    # TODO: Add neural guidance config when implemented


# =============================================================================
# EXPERIMENT REGISTRY
# =============================================================================

ALL_EXPERIMENTS: dict[str, Any] = {
    "exp000": exp000,
    "exp001": exp001,
    "exp002": exp002,
    "exp003": exp003,
    "exp004": exp004,
    "exp005": exp005,
}

DEFAULT_EXPERIMENT = "exp000"
