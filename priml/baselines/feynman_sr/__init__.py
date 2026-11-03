"""
Feynman Symbolic Regression baselines for priml.

SRSD Feynman benchmark: 120 physics equations (30 Easy, 30 Medium, 60 Hard).
Exact recovery = symbolic equality via sympy with 1% constant tolerance.

Exp000: Dimensional GA + Levenberg-Marquardt constant optimization
"""

from .exp000 import exp000, exp001, exp002, exp003, exp004, exp005, ALL_EXPERIMENTS

__all__ = [
    'exp000', 'exp001', 'exp002', 'exp003', 'exp004', 'exp005',
    'ALL_EXPERIMENTS',
]
