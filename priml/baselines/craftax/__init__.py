"""Craftax reinforcement-learning baseline in PyTorch.

``exp000`` reproduces the published 1M-interaction PPO recipe on full symbolic
Craftax; later experiments change the budget and the architecture one step at a
time. The simulator, the policy, and the learning algorithm are all PyTorch --
the environment is stepped with a batched leading environment axis rather than
a vectorizing transform, so one step is one ordinary tensor program.

The primary metric is mean episodic return as a percentage of the environment's
maximum reward, 226.
"""
