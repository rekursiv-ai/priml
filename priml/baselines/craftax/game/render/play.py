"""Play the game, or watch a policy play it.

Two ways to drive the same world. ``play`` opens a window and takes the
keyboard, which is the fastest way to find out whether an achievement the
agent never unlocks is even reachable. ``record`` writes an mp4 from a policy,
which is the only way to see WHY a return is what it is -- a number cannot
show you an agent that drowned on step 40 of every episode.

Both drive a batch of one. Watching sixty-four worlds at once shows nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import imageio_ffmpeg
import numpy as np
import pygame
import torch

from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.game import (
    step as game_step,
    world_gen,
)
from priml.baselines.craftax.game.constants import Action
from priml.baselines.craftax.game.render.pixels import Renderer


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.craftax.game.state import EnvState


KEYS: dict[int, Action] = {
    pygame.K_a: Action.LEFT,
    pygame.K_d: Action.RIGHT,
    pygame.K_w: Action.UP,
    pygame.K_s: Action.DOWN,
    pygame.K_SPACE: Action.DO,
    pygame.K_TAB: Action.SLEEP,
    pygame.K_r: Action.PLACE_STONE,
    pygame.K_t: Action.PLACE_TABLE,
    pygame.K_f: Action.PLACE_FURNACE,
    pygame.K_p: Action.PLACE_PLANT,
    pygame.K_1: Action.MAKE_WOOD_PICKAXE,
    pygame.K_2: Action.MAKE_STONE_PICKAXE,
    pygame.K_3: Action.MAKE_IRON_PICKAXE,
    pygame.K_4: Action.MAKE_WOOD_SWORD,
    pygame.K_5: Action.MAKE_STONE_SWORD,
    pygame.K_6: Action.MAKE_IRON_SWORD,
    pygame.K_e: Action.REST,
    pygame.K_COMMA: Action.DESCEND,
    pygame.K_PERIOD: Action.ASCEND,
    pygame.K_y: Action.MAKE_ARROW,
    pygame.K_u: Action.SHOOT_ARROW,
    pygame.K_i: Action.CAST_FIREBALL,
    pygame.K_o: Action.CAST_ICEBALL,
    pygame.K_z: Action.PLACE_TORCH,
    pygame.K_x: Action.ENCHANT_SWORD,
    pygame.K_c: Action.ENCHANT_ARMOUR,
    pygame.K_b: Action.READ_BOOK,
    pygame.K_n: Action.LEVEL_UP_DEXTERITY,
    pygame.K_m: Action.LEVEL_UP_STRENGTH,
    pygame.K_k: Action.LEVEL_UP_INTELLIGENCE,
}
"""Keyboard to action. WASD moves, space acts, digits craft."""


class Policy(Protocol):
    """Anything that scores actions from an observation."""

    def __call__(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Return action logits and a value estimate."""
        ...


def play(
    *,
    seed: int = 0,
    block_pixels: int = 64,
    asset_dir: Path | None = None,
) -> EnvState:
    """Open a window and play one world from the keyboard.

    The world advances only when a key is pressed, so there is no clock to
    lose to: this is for inspecting a situation, not for reflexes.

    Args:
      seed: World seed.
      block_pixels: Tile size, and therefore window size.
      asset_dir: Where sprites are cached; defaults to the user cache.

    Returns:
      state: The world as it stood when the window closed.

    """
    pygame.init()
    generator = torch.Generator().manual_seed(seed)
    state = world_gen.generate_world(
        num_envs=1,
        generator=generator,
        device=torch.device("cpu"),
    )
    renderer = Renderer(block_pixels=block_pixels, asset_dir=asset_dir)
    frame = renderer.render(state)
    screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
    pygame.display.set_caption(f"Craftax (seed {seed})")

    running = True
    while running:
        _show(screen, renderer.render(state))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                action = KEYS.get(event.key)
                if action is None:
                    continue
                state, _ = game_step.step(
                    state,
                    torch.tensor([int(action)]),
                    generator=generator,
                )
                if bool(game_step.is_done(state)[0]):
                    state = world_gen.generate_world(
                        num_envs=1,
                        generator=generator,
                        device=torch.device("cpu"),
                    )
    pygame.quit()
    return state


def record(
    policy: Policy,
    path: Path | str,
    *,
    seed: int = 0,
    max_steps: int = 1_000,
    fps: int = 10,
    block_pixels: int = 64,
    asset_dir: Path | None = None,
) -> int:
    """Write an mp4 of one episode played by ``policy``.

    Args:
      policy: Network returning action logits for one observation.
      path: Where to write the video.
      seed: World and sampling seed.
      max_steps: Cap on episode length.
      fps: Frames per second in the output.
      block_pixels: Tile size in the output.
      asset_dir: Where sprites are cached; defaults to the user cache.

    Returns:
      steps: How many steps the episode ran.

    Raises:
      ValueError: The geometry is not positive.

    """
    if max_steps <= 0 or fps <= 0:
        raise ValueError("Replay geometry must be positive")

    config = CraftaxEnv.Config()
    config.num_envs = 1
    config.device = "cpu"
    config.seed = seed
    env = config.make()
    observation = env.reset()

    renderer = Renderer(block_pixels=block_pixels, asset_dir=asset_dir)
    generator = torch.Generator().manual_seed(seed)
    frames: list[np.ndarray] = []
    steps = 0
    with torch.no_grad():
        for _ in range(max_steps):
            frames.append(renderer.render(env.state))
            logits, _ = policy(observation)
            action = torch.multinomial(
                logits.softmax(-1),
                1,
                generator=generator,
            ).squeeze(-1)
            transition = env.step(action)
            observation = transition.observation
            steps += 1
            if bool(transition.done[0]):
                break

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[0], frames[0].shape[1]
    # 16 is what ``imageio.v3.imwrite`` used, so the written geometry is
    # unchanged by the switch to the lower-level writer. It only matters off
    # the default board: at ``block_pixels=64`` the render is 576x704, already
    # aligned, so every value agrees. Do NOT "improve" this to 1 -- an odd
    # dimension then yields a file ffmpeg cannot read back, and the writer
    # returns without raising, so the corruption surfaces far from here.
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        macro_block_size=16,
    )
    # Prime the generator so it reaches its first ``yield`` and spawns ffmpeg;
    # ``next`` rather than ``send(None)`` because the two are the same operation
    # and only the former types as one.
    next(writer)
    try:
        for frame in frames:
            _ = writer.send(frame.tobytes())
    finally:
        writer.close()
    return steps


def _show(screen: pygame.Surface, frame: np.ndarray) -> None:
    """Blit one rendered frame to the window."""
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
