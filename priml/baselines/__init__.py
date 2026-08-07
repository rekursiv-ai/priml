"""Reference training pipelines built entirely from ``priml`` components.

Each subpackage is one dataset end to end -- data, model, train step, and a
family of experiments -- written in the same shape so that reading one is
enough to navigate the rest. Every package exposes ``exp000``: the best
straightforward recipe for its dataset, frozen once measured, so downstream
work has a stable control to fork from. Later experiments (``exp001`` and on)
each apply one named, sourced change on top of a named parent.
"""
