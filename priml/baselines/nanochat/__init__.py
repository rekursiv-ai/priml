"""NanoChat: a small decoder-only language model trained to a time budget.

``exp000`` pretrains a transformer on ClimbMix and scores validation bits per
byte. What makes the baseline unusual is its stop condition: training runs to a
wall-clock BUDGET rather than a step count, so an experiment that makes a step
cheaper is rewarded with more steps rather than a shorter run. Every schedule
is therefore driven by elapsed budget fraction, not by step index.

The architecture is a plain pre-norm transformer plus four things measured to
matter at this scale: rotary positions, parameter-free RMS norm on every
sublayer input, a squared-ReLU feed-forward, and alternating value embeddings
that let a layer read the token table directly. Each is a value in a slot, so
removing one is a fork rather than an edit.
"""
