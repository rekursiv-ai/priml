"""MLS-Bench harness: score a code edit against a task's published leaderboard.

A task is a directory of editable files, settings (scripts run in a pinned
container) and a leaderboard. This package parses the task (:mod:`.task`),
materializes a workspace with one edit applied and guards the edit surface
(:mod:`.workspace`). Nothing from upstream is vendored: tasks are read from a
clone at the logical path ``/mlsbench`` beneath ``base_dir`` (upstream pin
``80cf5c5c``). Importing this package has no side effects.
"""
