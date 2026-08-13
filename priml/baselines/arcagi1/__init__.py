"""ARC-AGI abstract-reasoning baseline.

Each task is a rule demonstrated by a handful of input/output grid pairs; the
model must infer the rule and apply it to a held-out input. Scoring is
whole-grid: a single wrong cell fails the puzzle.

The solver, recurrence, adaptive computation time, and train step are the
sudoku baseline's, filled with different values -- a 30x30 grid, twelve colors,
and a learned per-task prefix. ARC's own code is what names ARC: its dataset,
its pass@K metric, and the experiment ladder.
"""
