"""ConvexTok: a vocabulary chosen by a linear-programming relaxation.

A port of ConvexTok (Tempus et al., 2026) that fits the same vocabulary as the
upstream implementation from the same text. Pretokens become one shortest-path
flow graph each; a linear program chooses which substrings to keep under a
vocabulary budget; PDLP solves it on the GPU; rounding turns the fractional
choice into tokens. The equivalence contract with upstream and the
measurements behind it are in this baseline's ``README.md``.
"""
