"""Maze-Hard: mark the shortest path through a 30x30 maze, scored by exact match.

``exp000`` is the reference tiny recursive model recipe on one H100 for one
hour. Each later rung is a campaign checkpoint that raised the record:
canonical labels with the Universal Reasoning Model's single-latent recurrence
(``exp001``), a deeper single pass (``exp002``, the one-hour best), and a
four-hour run (``exp003``). The solver, pool, and step are shared with the ARC
and sudoku baselines; this package adds the maze's data, its boundary loss, and
the recipes.
"""
