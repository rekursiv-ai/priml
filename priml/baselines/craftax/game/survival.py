"""Movement, and the meters that run down while the player is alive.

Two things happen every step regardless of what the player chose to do: they
face and possibly enter a tile, and their body advances one tick toward hunger,
thirst, exhaustion, and either healing or dying of it.

The meters are driven by hidden accumulators rather than dropping a point per
step. Hunger rises continuously and costs a food point only when it crosses a
threshold, which is what makes dexterity able to slow the whole clock smoothly
instead of in whole steps.
"""

from __future__ import annotations

from torch import Tensor

import torch

from priml.baselines.craftax.game import constants, mechanics
from priml.baselines.craftax.game.constants import Achievement, Action
from priml.baselines.craftax.game.state import EnvState


def move_player(state: EnvState, action: Tensor) -> EnvState:
    """Turn to face a direction and step there if the tile allows it.

    A blocked move still turns the player, which is what lets them mine or
    place against a wall they cannot walk into.

    Args:
      state: The current world.
      action: The chosen action per environment, ``[envs]``.

    Returns:
      state: The world with the player moved and facing.

    """
    step = constants.DIRECTIONS.to(state.device)[action.long()]
    proposed = state.player_position + step
    land = torch.tensor([False, True, True], device=state.device).expand(
        state.num_envs,
        3,
    )
    allowed = mechanics.can_walk_on(state, proposed, land)
    state.player_position = state.player_position + allowed[:, None].int() * step

    is_movement = step.abs().sum(-1) != 0
    state.player_direction = torch.where(
        is_movement,
        action.int(),
        state.player_direction,
    )
    return state


def update_intrinsics(state: EnvState, action: Tensor) -> EnvState:
    """Advance sleep, rest, and every survival meter by one step.

    Sleeping halves the rate at which hunger and thirst accrue and reverses
    fatigue, which is why it is worth the vulnerability. The boss floor
    suspends starvation entirely so the fight is decided by combat.

    Args:
      state: The current world.
      action: The chosen action per environment, ``[envs]``.

    Returns:
      state: The world one tick older.

    """
    state = _update_sleep_and_rest(state, action)
    starves = ~mechanics.is_fighting_boss(state)
    # Dexterity slows the entire body clock; at the cap the meters run half
    # as fast, which is what makes it worth levelling.
    decay = 1.0 - 0.125 * (state.player_dexterity - 1).float()
    asleep = state.is_sleeping

    state.player_hunger, state.player_food = _tick_meter(
        accumulator=state.player_hunger + torch.where(asleep, 0.5, 1.0) * decay,
        meter=state.player_food,
        threshold=25.0,
        starves=starves,
    )
    state.player_thirst, state.player_drink = _tick_meter(
        accumulator=state.player_thirst + torch.where(asleep, 0.5, 1.0) * decay,
        meter=state.player_drink,
        threshold=20.0,
        starves=starves,
    )
    state = _tick_fatigue(state, decay=decay, starves=starves)
    state = _tick_health(state, starves=starves)
    return _tick_mana(state)


def _update_sleep_and_rest(state: EnvState, action: Tensor) -> EnvState:
    """Start and end sleeping and resting according to the meters."""
    starts_sleep = (action == int(Action.SLEEP)) & (
        state.player_energy < mechanics.max_energy(state)
    )
    state.is_sleeping = state.is_sleeping | starts_sleep

    wakes = (state.player_energy >= mechanics.max_energy(state)) & state.is_sleeping
    state.is_sleeping = state.is_sleeping & ~wakes
    state.achievements = mechanics.unlock_achievement(
        state,
        torch.full(
            (state.num_envs,),
            int(Achievement.WAKE_UP),
            device=state.device,
        ),
        wakes,
    )

    starts_rest = (action == int(Action.REST)) & (
        state.player_health < mechanics.max_health(state)
    )
    state.is_resting = state.is_resting | starts_rest
    # Resting ends on recovery, but also on an empty stomach: it must not be
    # a way to sit out starvation.
    stops_rest = state.is_resting & (
        (state.player_health >= mechanics.max_health(state))
        | (state.player_food <= 0)
        | (state.player_drink <= 0)
    )
    state.is_resting = state.is_resting & ~stops_rest
    return state


def _tick_meter(
    *,
    accumulator: Tensor,
    meter: Tensor,
    threshold: float,
    starves: Tensor,
) -> tuple[Tensor, Tensor]:
    """Spend one point of ``meter`` each time ``accumulator`` crosses over."""
    crossed = accumulator > threshold
    spent = (meter - starves.int()).clamp(min=0)
    return (
        torch.where(crossed, torch.zeros_like(accumulator), accumulator),
        torch.where(crossed, spent, meter),
    )


def _tick_fatigue(state: EnvState, *, decay: Tensor, starves: Tensor) -> EnvState:
    """Accrue tiredness while awake and pay it back while asleep."""
    fatigue = torch.where(
        state.is_sleeping,
        (state.player_fatigue - 1).clamp(max=0.0),
        state.player_fatigue + decay,
    )
    exhausted = fatigue > 30
    energy = torch.where(
        exhausted,
        (state.player_energy - starves.int()).clamp(min=0),
        state.player_energy,
    )
    fatigue = torch.where(exhausted, torch.zeros_like(fatigue), fatigue)

    rested = fatigue < -10
    energy = torch.where(
        rested,
        (energy + 1).minimum(mechanics.max_energy(state)),
        energy,
    )
    state.player_fatigue = torch.where(rested, torch.zeros_like(fatigue), fatigue)
    state.player_energy = energy
    return state


def _tick_health(state: EnvState, *, starves: Tensor) -> EnvState:
    """Heal while fed, watered and rested; bleed out otherwise."""
    sustained = (
        (state.player_food > 0)
        & (state.player_drink > 0)
        & ((state.player_energy > 0) | state.is_sleeping)
    )
    gain = torch.where(state.is_sleeping, 2.0, 1.0)
    loss = torch.where(state.is_sleeping, -0.5, -1.0) * starves.float()
    recover = state.player_recover + torch.where(sustained, gain, loss)

    healed = recover > 25
    health = torch.where(
        healed,
        (state.player_health + 1).minimum(mechanics.max_health(state).float()),
        state.player_health,
    )
    recover = torch.where(healed, torch.zeros_like(recover), recover)

    hurt = recover < -15
    state.player_health = torch.where(hurt, health - 1, health)
    state.player_recover = torch.where(hurt, torch.zeros_like(recover), recover)
    return state


def _tick_mana(state: EnvState) -> EnvState:
    """Refill mana, faster while asleep and for a more intelligent player."""
    rate = 1 + 0.25 * (state.player_intelligence - 1).float()
    recover = (
        state.player_recover_mana + torch.where(state.is_sleeping, 2.0, 1.0)
    ) * rate
    restored = recover > 30
    state.player_mana = torch.where(
        restored,
        state.player_mana + 1,
        state.player_mana,
    )
    state.player_recover_mana = torch.where(
        restored,
        torch.zeros_like(recover),
        recover,
    )
    return state
