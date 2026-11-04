"""Feynman Symbolic Regression Baseline (exp000).

Dimensional GA + Levenberg-Marquardt constant optimization.
Target: Beat PySR 60% on SRSD Feynman 120 equations.

Requires optional dependencies:
    pip install priml[feynman_sr]

Import experiments directly:
    from priml.baselines.feynman_sr.exp000.experiments import exp000, ALL_EXPERIMENTS, ...
"""

__all__ = [
    "ALL_EXPERIMENTS",
    "DEFAULT_EXPERIMENT",
    "exp000",
    "exp001",
    "exp002",
    "exp003",
    "exp004",
    "exp005",
]
