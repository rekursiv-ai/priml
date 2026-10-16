"""Sudoku constraint-solving baseline.

``exp000``-``exp006`` build the tiny recursive model recipe one training
mechanism at a time, from the published baseline to a 0.9624 full-set solver.
``exp007``-``exp013`` fork that recipe at evaluation -- hypothesis-pinning
search, a learned verifier committee, latent hypotheses, an agreement lock --
and reach 422,786/422,786.

The input embedding is a list of additive channels rather than a fixed set, so
a differently-shaped puzzle -- ARC's 30x30 grid, say -- is a different channel
list against the same model and train step.
"""
