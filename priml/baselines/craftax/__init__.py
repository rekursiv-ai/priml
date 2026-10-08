"""PufferLib's Craftax trainer, ported bit-for-bit to Numba, torch and Triton.

``exp000`` reproduces PufferLib's Craftax baseline (pin ``6ffa5b10``,
``config/craftax.ini``) with no PufferLib and no C. It keeps PufferLib's design
-- the game steps on CPU threads while the policy and learner run on the GPU --
and each component is checked bit for bit against PufferLib's trainer (README,
"Parity").

The primary metric is the mean achievement return as a percentage of the
maximum, 226.
"""
