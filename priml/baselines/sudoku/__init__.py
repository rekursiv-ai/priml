"""Sudoku constraint-solving baseline.

Two ladders share the dataset. ``exp000``-``exp003`` vary two independent axes
one at a time over a generic solver: which block mixes the tokens (transformer
or MLP-mixer) and whether it runs a recurrence with adaptive computation time.

``exp004``-``exp010`` build the tiny recursive model recipe one mechanism at a
time, from the published baseline to a 0.9624 full-set solver.
``exp011``-``exp014`` fork that recipe at evaluation -- search, agreement
lock, verifier sieve, seed ensemble -- and reach 422,786/422,786.

The input embedding is a list of additive channels rather than a fixed set, so
a differently-shaped puzzle -- ARC's 30x30 grid, say -- is a different channel
list against the same model and train step.
"""
