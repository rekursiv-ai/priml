"""Feynman Symbolic Regression Baseline (exp000).

Dimensional GA + Levenberg-Marquardt constant optimization.
Target: Beat PySR 60% on SRSD Feynman 120 equations.
"""

from .experiments import (
    ALL_EXPERIMENTS,
    DEFAULT_EXPERIMENT,
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
)


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
