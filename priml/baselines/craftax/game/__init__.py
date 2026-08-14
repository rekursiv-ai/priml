"""The Craftax game: rules, world, and what a player can see.

An open-ended 2D survival game. A player mines wood and stone, crafts tools,
fights creatures, eats, drinks, sleeps, and descends through nine floors. There
is no single goal -- 67 achievements form a tech tree, and the score rewards
breadth rather than depth.

This package is the game and nothing else. It knows nothing about policies,
optimizers, or training: it advances a world one action at a time and encodes
what the player sees. The learning code lives one directory up and depends on
this package; nothing here depends on it.

References:
    https://arxiv.org/abs/2402.16801
        Matthews et al. 2024. Craftax: a lightning-fast benchmark for
        open-ended reinforcement learning.

"""

from __future__ import annotations
