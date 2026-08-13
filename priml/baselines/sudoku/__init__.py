"""Sudoku constraint-solving baseline.

``exp000`` trains a plain post-norm transformer over the 81-cell grid. The
ladder then varies two independent axes one at a time: which block mixes the
tokens (transformer or MLP-mixer) and whether the solver runs a recurrence with
adaptive computation time.

The input embedding is a list of additive channels rather than a fixed set, so
a differently-shaped puzzle -- ARC's 30x30 grid, say -- is a different channel
list against the same model and train step.
"""
