"""The ARC-AGI-2 reference implementation, kept only for bit-exact parity.

The source trained its ARC-AGI-2 recipe with its own step, loader, and metric.
They are ported here bit for bit, with tests that replay the source's
single-rank and two-rank records. No rung trains on them: the rungs run
:class:`~priml.baselines.arcagi.train_step.PooledActTrainStep`, whose golden
pins the same recipe. Nothing outside this package imports it, so deleting it
leaves the baseline intact.
"""
