"""MLS-Bench tasks, re-implemented as native priml baselines.

Each subpackage ports one upstream task: its editable function becomes a config
slot, ``exp000`` is the pristine recipe end to end, and each idea is a fork
that changes only that slot. A capture script proves every port bit for bit
against the pristine upstream script (pinned commit ``b8739449``). Nothing from
upstream is vendored.
"""
