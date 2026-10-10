# PufferLib Craftax vs original Craftax: every difference

The port reproduces PufferLib Craftax bit for bit. This catalog lists every
way PufferLib differs from original Craftax (Craftax-Symbolic-v1). For each
difference it gives the class, the citations on both sides, and the Config
option that selects the original behaviour. Every option defaults to
PufferLib's behaviour, so exp000's bits do not move.

## Sources

- **PufferLib**: `6ffa5b10`: `ocean/craftax/craftax.h`, `constants.h`,
  `config/craftax.ini`, `src/pufferl.cu`.
- **lockstep**: PufferLib `6ffa5b10`: `ocean/craftax/craftax_parity.h` and
  `tests/craftax_parity.py`.
- **JAX**: `craftax/...` of the `craftax` 1.6.1 package (PyPI).
- **port**: `priml/baselines/craftax/{env.py,game/}` in this tree.
- **parity**: the port's parity suite, which reads fixtures that are not
  distributed and so is kept beside them, outside this package. Its checks are
  named below: the original-setup test (the port on JAX's own states, one
  transition at a time), the world-generation test, the trajectory test
  (random play, compared in distribution), the lockstep diff test and the
  undefined-spawn test.

Paths below are relative to those roots. `craftax.h:2378` is PufferLib, and
`game_logic.py:3073` is JAX (`craftax/craftax/game_logic.py`); both are pinned,
so their line numbers hold. The port is still changing, so it is cited by
function: `step.py::score_numba` is `game/step.py`'s `score_numba`, and
a test checks that every such name exists.
The pin's source is a checkout of PufferLib at `6ffa5b10`,
and the `craftax` 1.6.1 sources come from the uv cache.

## Method

1. Diff `craftax.h` against `craftax_parity.h`. Their header comments say the
   lockstep port runs "JAX Craftax-Symbolic-v1 ... Threefry RNG matches the
   Python env" while PufferLib has the "same rules, simpler RNG".
   The lockstep diff test makes that checkable. It rewrites
   both headers modulo their RNGs: each draw keeps only the arguments that
   set its distribution, and statements that only move random state are
   dropped. The 28 blocks that still differ are pinned by content, each with
   a reviewed reason (D9, D13, D14, a same-distribution identity, rendering,
   a rename or a refactor), and so are the RNG helpers.
2. Take the exceptions that `tests/craftax_parity.py:1-16` states: sleep and
   rest are collapsed (D4), and rewards are not compared (D1).
3. The two C headers share everything outside the diff, including the reward,
   terminal, reset pool, mask and observation code. Read that code in PufferLib
   against JAX: `craftax_symbolic_env.py`, `environment_bases.py`,
   `game_logic.py`, `renderer.py`, `world_gen.py`, `constants.py`.
4. Measure what is cheap to measure. The achievement reward weights were
   computed from JAX's `achievement_mapping` and compared with PufferLib's
   table.

Classes:

- **RNG**: same distribution, different random stream. It matters only for a
  lockstep comparison.
- **rule**: the game's transition or termination differs.
- **setup**: the interface to the learner differs: reward, action mask, what
  one step is, reset distribution, or observation encoding. The underlying
  game stays the same.
- **numerics**: same formula, different float representation or library.

## Summary

| ID | Difference | Class | Option: `CraftaxEnv.Config.rules` field, or its `restart_numba` slot (default = PufferLib) |
|---|---|---|---|
| D1 | reward function | setup | `original_reward = False` |
| D2 | beating the necromancer ends the episode | rule | `end_on_boss_defeat = False` |
| D3 | episode timeout | none | `max_timesteps = 100_000` (equal) |
| D4 | sleep and rest are one decision | setup | `collapse_sleep = True` |
| D5 | illegal-action mask in the sampler | setup | `action_mask = True` |
| D6 | resets draw from a pool of worlds | setup | `restart = WorldPool.Config()` (8,192 worlds); `FreshWorlds.Config()` is original |
| D7 | autoreset | none | -- |
| D8 | observation encoding (843 vs 8,268) | setup | `symbolic_observation = False` |
| D9 | JAX mob-channel scatter (wrap, drift) | rule (obs) | none (proposed) |
| D10 | light map stored as uint8 | numerics, rule | -- (tolerated) |
| D11 | libm vs XLA transcendentals | numerics | -- (tolerated) |
| D12 | random streams (rand_r vs Threefry) | RNG | none (proposed) |
| D13 | draw counts per call site | RNG | none (with D12) |
| D14 | out-of-range spawn-index read | rule (RNG) | none (with D12) |
| D15 | episode statistics | none | -- (logging only) |
| D16 | a new projectile's `health` value | none | -- (never read) |

The built options are the fields of `CraftaxEnv.Config.rules`, named as
`game/step.py`'s `Rules` names them, plus the `restart_numba` slot; the env passes
D1-D6 and D8 to `Rules`, whose defaults are PufferLib's. D6 is
`restart = FreshWorlds.Config()`, which sets `Rules.fresh_worlds`. exp002 sets
D1, D2, D4, D5, D6 and D8 to original. D5 has the env write an all-ones mask,
which leaves the sampler unchanged. D8 is `rules.symbolic_observation = True`;
since `MinGRUPolicy`'s multi-hot embedding reads packed ids, exp002 pairs it
with `DenseObservation`, a first stage that hands the 8,268 floats to
`proj_in` unchanged, and the step refuses an env whose observation width its
policy's first stage does not read.

### Proposed options (not built)

- `jax_mob_wrap` (D9): reproduce JAX's mob-channel scatter. It is a JAX bug
  in what the agent sees; the D8 test emulates it instead.
- `float_light` (D10): keep light as float32 with JAX's 0.05 threshold. It
  changes the state layout, world generation and observation for a sliver of
  behaviour, so D10 is tolerated instead; it can be built on request.
- `rng = "threefry"` (D12, with D13 and D14): JAX's key splitting, for an
  exact lockstep with JAX. The proof chain makes it unnecessary for
  verification.

## D1. Reward function

- PufferLib: the achievement reward is the weighted sum of achievements
  unlocked this step. PufferLib adds the change in equipped armour, and any step
  that leaves health at or below 0 gets exactly -1 (`craftax.h:2381-2388`;
  port `step.py::score_numba`).
- JAX: the same weighted sum plus 0.1 times the change in health
  (`game_logic.py:3073-3079`).
- The weights are identical. MEASURED: all 67 entries of JAX's
  `achievement_mapping` (`constants.py:523-536`) equal `constants.h:254-264`,
  and both sum to 226.
- Under the sleep collapse (D4), PufferLib's reward covers every tick of the
  decision.

## D2. Beating the necromancer ends the episode

- PufferLib: `done = health <= 0 || timestep >= DEFAULT_MAX_TIMESTEPS`
  (`craftax.h:2378`; port `rules.py::tick_numba`, and the option in
  `step.py::play_numba`). The DEFEAT_NECROMANCER
  achievement is set when `boss_progress >= NUM_LEVELS - 1`
  (`craftax.h:2105`), and the episode continues.
- JAX: `is_game_over` also ends the episode on `has_beaten_boss`
  (`game_logic.py:4-9`, `util/game_logic_utils.py:28-29`), which is the same
  `boss_progress` test.
- No PufferLib episode has reached floor 8 in training: the 3.493B run's eval
  and the 20B checkpoint's eval both show `floor_8 = 0` (oracle-v1
  `MANIFEST.json`). MEASURED instead on floor-8 JAX states
  with the necromancer marked beaten (`boss_progress = 8`; the original-setup test):
  with the option the port ends the episode wherever JAX's `is_game_over`
  does, and without it a surviving player plays on.

## D3. Episode timeout (no difference)

- Both end at 100,000 ticks: `DEFAULT_MAX_TIMESTEPS` (`constants.h:30`) and
  `EnvParams.max_timesteps` (`craftax_state.py:113`), each counted in
  `state.timestep` ticks. With the sleep collapse (D4) one PufferLib decision
  can cover many ticks, so the decision count at timeout differs, but the
  tick count does not.

## D4. Sleep and rest are one decision

- PufferLib: `puf_step` repeats NOOP ticks while the player is asleep or
  resting, until they wake, the episode ends, or they are hit
  (`craftax.h:1655-1661, 2379`; port `step.py::play_numba`). The learner gets one
  observation, one summed reward and one decision.
  `episode_length` counts decisions (`craftax.h:2396`).
- JAX: one tick per `env.step`. The action is replaced by NOOP while
  sleeping or resting (`game_logic.py:3011-3012`), but the learner observes,
  is rewarded and acts on every tick.
- The states at decision points are identical. The pin's own parity test
  replays JAX with NOOP until the player wakes (`tests/craftax_parity.py:
  282-300`). What differs is the discount per step, the number of recurrent
  updates, the length units and the intermediate observations.

## D5. Illegal-action mask in the sampler

- PufferLib: `config/craftax.ini:20` sets `action_mask = 1`.
  `compute_action_mask_numba` (`craftax.h:1387-1451`; port
  `observation.py::compute_action_mask_numba`) marks the actions that would do
  something, and `sample_logits` samples only from those
  (`pufferl.cu:592-704`). The env still executes any action it is given
  (`puf_step`).
- JAX: no mask. The action space is `Discrete(43)`
  (`craftax_symbolic_env.py:169-170`), and an illegal action does nothing.
  For example, sleep starts only below maximum energy
  (`game_logic.py:1832`), rest only below maximum health (`1851`), and
  crafting checks its inputs, tier and table (`do_crafting`, from `522`).
- MEASURED (the original-setup test, 26,062 masked state-action pairs on JAX states
  from all floors, one tick each): every masked action gives NOOP's next
  state, with one exception. REST is masked at full health
  (`craftax.h:1405`), but the rest check reads health after `update_mobs`
  (`game_logic.py:1851`). A REST taken on a tick in which a creature hits the
  player therefore starts a rest. This happened in 69 pairs, and the player
  was hit in every one. So PufferLib's mask removes one action with an effect:
  the preemptive REST.

## D6. Resets draw from a fixed pool of worlds

- PufferLib: `config/craftax.ini:19` sets `reset_pool_size = 8192`. Worlds
  0..8191 are generated once, world `i` from `rand_r` seed `i`
  (`craftax.h:2455-2461, 2471-2477`; port `world_gen.py::build_pool_numba`). Every
  reset copies `levels[rand_r % 8192]` (`craftax.h:1631-1632, 2422-2423`;
  port `step.py::restart_numba`).
- JAX: every reset generates a new world from the reset key
  (`craftax_symbolic_env.py:146-152`, `environment_bases.py:33-43`).
- PufferLib already has the original behaviour: with `num_levels == 0` it calls
  `generate_world` on the env's own stream (`craftax.h:1633-1635,
  2424-2426`). The port ports that branch: `CraftaxEnv.Config.restart =
  FreshWorlds.Config()` sets `Rules.fresh_worlds`, and `restart_numba` generates
  each world into a zeroed template (`step.py::restart_numba`,
  `env.py::CraftaxEnv.__init__`).
- MEASURED: fresh port worlds match JAX's resets in distribution, 378
  per-floor counts over 384 worlds a side (the world-generation test). A fresh world
  is `generate_world` on the env's stream, exactly (the original-setup test). Under
  a random policy, one world for every episode is rejected, while PufferLib's
  8,192-world pool is not (the trajectory test). The pool is visible only to a
  policy that can tell worlds apart.

## D7. Autoreset (no difference)

- Both reset in place when the episode ends. The returned observation is the
  new episode's first, while the reward and done flag belong to the transition
  that ended (`craftax.h:2393-2429`; `environment_bases.py:33-43`). Both treat
  the timeout as terminal: JAX's discount is 0 at any terminal
  (`environment_bases.py:78-80`), and PufferLib sets `terminals = 1`
  (`craftax.h:2394`). Only which world comes next differs (D6).

## D8. Observation encoding

- PufferLib: 843 floats. Each of the 9x11 tiles has 8 channels: block id,
  item id + 1, visible, and five mob-class channels holding the mob type + 1.
  They are followed by 51 scalars (`craftax.h:1453-1546`; port
  `observation.py::compute_observations_numba`).
- JAX: 8,268 floats. Each tile is one-hot: 37 blocks, 5 items, 5x8 mob
  (class, type) bits and visible. The same 51 scalars follow
  (`renderer.py:9-197`, `craftax_symbolic_env.py:18-37`).
- The 51 scalars use the same formulas in the same order
  (`renderer.py:131-198`, `craftax.h:1495-1544`). Out-of-map tiles are all
  zero in both, because JAX pads the light map with 0 (`renderer.py:117-125`).
- `pack_symbolic_obs` (`tests/craftax_parity.py:242-262`) maps JAX to PufferLib
  exactly, except when two mobs of one class share a tile: JAX keeps both
  type bits, while PufferLib keeps the type of the last slot written. The port's
  symbolic option keeps both bits, as JAX does: each live slot sets its own
  bit (`observation.py::_write_symbolic_mob_obs_numba`; read, not exercised by the
  fixture). So the caveat applies only to the packed layout.
- MEASURED (the original-setup test): the port's symbolic observation equals JAX's
  jitted renderer bit for bit on 1,426 JAX states from all nine floors. Two
  documented differences are emulated exactly: D9 (9 erased bits) and D11
  (XLA's reciprocal division in the `/ 10` scalars).

## D9. JAX mob-channel scatter (index wrap and drifting empty slots)

- JAX scatters every mob slot's on-screen flag at its view-relative position,
  inactive and off-screen slots included (`renderer.py:42-58`). The write is
  0 for those slots, so it can erase a visible mob's (class, type) bit on the
  same tile. A position less than one view height above, or one view width
  left of, the view is negative, and JAX wraps it like NumPy. Positions
  beyond that are dropped.
- JAX also moves every projectile slot every tick, inactive ones included:
  `position = proposed_position` has no mask (`game_logic.py:1655, 1799`, in
  `_move_mob_projectile` and `_move_player_projectile`, `1610-1712` and
  `1713-1828`). Only its effects are masked. An empty slot therefore drifts along its
  last direction (MEASURED, oracle `harness/lockstep_state_probe.py`, job
  30486: seed 62's empty mob-projectile slots at (18,18), (19,19), ...
  (24,24) over steps 17-23), and its zero write lands wherever it drifts.
  PufferLib's empty slots stay where they died.
- The lockstep port reproduces the scatter at the packed level
  (`craftax_parity.h:872-903`) but not the drift. PufferLib writes only
  on-screen, lit, active mobs (`craftax.h:734-757`; port
  `observation.py::_write_mob_obs_numba`).
- MEASURED (the pin's own lockstep parity test): in 12 of 128 seeds x 1,000 random actions the
  lockstep port and JAX differ in exactly one mob-projectile observation bit.
  In the three seeds traced, dynamics were identical up to the terminal step
  and the bit was erased by a drifting empty slot. This is a JAX bug that
  changes only what the agent sees.

## D10. Light map stored as uint8 (tolerated)

- PufferLib: light is stored as `(unsigned char)(light * 255)`, truncated at
  world generation (`craftax.h:497`; port `world_gen.py::_generate_smooth_level_numba`)
  and again after every torch (`craftax.h:1963-1971`; port `rules.py::place_numba`).
  A tile is
  visible when the stored value is above 12 (`constants.h:32`).
- JAX: light stays float32 (`world_gen/world_gen.py:507-561`,
  `game_logic.py:956-971`), and a tile is visible when its light is above 0.05
  (`renderer.py:122`).
- Effects: a tile lit in (0.05, 13/255 = 0.0510) is visible in JAX and dark in
  PufferLib. Each torch placement can lose up to 1/255 more. The light map
  affects only visibility (the observation); mob spawning reads the global
  daylight `light_level`, not the map (`game_logic.py:2210`).

## D11. libm vs XLA transcendentals (tolerated)

- Daylight uses the same formula on both sides, `1 - |cos(pi * (t / 300 mod 1
  + 0.3))|^3`, but PufferLib computes it with glibc `cosf`/`powf`
  (`craftax.h:2375-2376`) and JAX with XLA's `cos` and `** 3`
  (`util/game_logic_utils.py:349-351`). World-generation noise and the
  light convolution also use different kernels (`util/noise.py:13-87` and
  `world_gen/world_gen.py:507-561` vs `craftax.h:262-322, 342-497`).
- MEASURED: in lockstep world generation the raw
  normalized noise differs from JAX's by at most 1 ulp. A block flips only
  when a value sits on a threshold. Chain link 3 met this twice in 128 seeds:
  seed 37's `tree_noise` is exactly 0.7f in C (not `> 0.7`) and 1 ulp above in
  JAX (lava); seed 105's `mountain` straddles the 0.85 cave threshold by 4 ulps
  (a path cell). No option: the port keeps glibc's results, and verification
  tolerates such flips. The distributional test sees no shift (D6).
- MEASURED (the original-setup test): JAX's jitted functions apply XLA's own
  rewrites, which PufferLib's C does not. A division by a constant becomes a
  multiplication by its reciprocal: the observation's `/ 10.0` scalars
  (inventory square roots, meters, attributes, floor) differ by one ulp,
  with JAX at 9 / 10 = 0.90000004 and PufferLib at 0.9. The original reward's
  `achievements + 0.1 * health gained` becomes one fused multiply-add. It
  differs from the port's two roundings in 12 of 713 JAX transitions, by less
  than 1e-6. The original-setup test emulates both exactly.
  The pin's own parity test hides the first with `atol = 1e-5`.

## D12. Random streams

- PufferLib: one `rand_r` stream per env, seeded `seed_offset + i`
  (`craftax.h:2445, 2487`), with `rng_f32_numba`, `rng_int_numba`, `choice_valid_numba` and
  `choose_weighted_numba` (`craftax.h:164-222`; port `rng.py::rand_r_step_numba`,
  `rng.py::rng_f32_numba`, `rng.py::rng_int_numba`, `rng.py::choice_valid_numba`,
  `rng.py::choose_weighted_numba`). World generation hashes a per-cell seed
  (`rng_f32_at_numba`, `craftax.h:178-182`; port `rng.py::rng_f32_at_numba`).
- JAX: Threefry2x32 keys split at every call site. `craftax_parity.h:142-285`
  reproduces `jax.random.split`, `uniform` and `randint` bit for bit.
- Every draw site keeps its distribution. The call sites (all in the diff):
  Perlin angles and the tree, ore and rare-ore thresholds; diamond and ladder
  cells; dungeon rooms and chests; the potion permutation (Fisher-Yates,
  `craftax.h:697-705`, vs a sort of random keys, `world_gen/world_gen.py:644`); chest
  loot; sapling drops; mob movement; spawn chance and cell; the book's spell;
  the enchant target.

## D13. Draw counts per call site

- Only a lockstep needs these to match. For example, PufferLib draws the armour
  enchant target only when enchanting armour (`craftax.h:2081-2084`), while
  JAX always splits a key and draws (`craftax_parity.h:2283, 2293`). The chest
  loot consumes an extra `randint` in JAX (`craftax_parity.h:2072-2074`).
  They would follow the proposed `rng` option.

## D14. Out-of-range spawn-index read

- PufferLib: `rng_int(lo, hi)` returns `lo + (int)(u * span)`, where
  `u = rand_r * (1 / (RAND_MAX + 1.0f))` can round to 1.0f. The result is then
  `hi`, and the spawn code reads `spawn_rows[n]` one past the end
  (`craftax.h:168-174, 2209-2240`; the port emulates it, `rng.py::rng_int_numba`,
  `mobs.py::spawn_mobs_numba`, `mobs.py::select_spawn_cell_numba`). MEASURED: 5 events
  in 2.05e9 transitions, deterministic (the undefined-spawn test).
- JAX and the lockstep port cannot hit it: `pick_spawn_cell` clamps
  (`craftax_parity.h:1337-1352`). The proposed `rng = "threefry"` would remove
  it. Under `rand_r` its rate is too small to move any distributional test.

## D15. Episode statistics (logging only)

- PufferLib logs `score_numba`, `perf` and the achievement rate from the
  achievement-only return (`craftax.h:2398-2419`). JAX's `info` reports
  `achievements * 100` at done (`envs/common.py:5-11`). These are not
  dynamics.

## D16. A new projectile's `health` value (never read)

- PufferLib stores the projectile's damage sum in `health` when it spawns (2.0
  for an arrow; `craftax.h:1003-1024`). JAX's `spawn_projectile` leaves the
  slot's `health` as it was, 1.0 from world generation
  (`util/game_logic_utils.py:180-230`). MEASURED: the only state difference
  in the traced seeds.
- Neither side reads a projectile's `health`. Both compute projectile damage
  from its type (`craftax.h:1026-1110`; `game_logic.py:1678-1683`). Not an
  option.

## Verification status

| Row | Status | Evidence |
|---|---|---|
| D1 | option; exact vs JAX up to D11's FMA | the original-setup test |
| D2 | option; exact vs JAX's `is_game_over` | the original-setup test |
| D3 | equal (100,000 ticks) | read both sides |
| D4 | option; exact (one tick, action ignored while asleep) | the original-setup test |
| D5 | option; masked equals NOOP, but REST under attack | the original-setup test |
| D6 | option; exact generator, JAX distribution | the world-generation test, the original-setup test, the trajectory test |
| D7 | equal | read both sides |
| D8 | option; exact vs JAX up to D9 and D11 | the original-setup test |
| D9 | documented; emulated in the D8 test; no option | the pin's lockstep test, the original-setup test |
| D10 | tolerated; no fixture light in the gap | the original-setup test |
| D11 | tolerated; sources: noise, reciprocal, FMA | the lockstep world generation, the original-setup test |
| D12-D14 | RNG; not built (the chain covers it) | the proof chain |
| D15 | logging only | -- |
| D16 | never read | the pin's lockstep test |
| all | exp002's setup plays like JAX under a random policy | the trajectory test |

The last row is the trajectory test: 11,200 random
episodes a side, 99 pre-stated tests (episode length and return, death rate,
unlock rates, per-action outcome rates), family-wise alpha 1e-3. None
rejects, and the sample detects a 0.047 CDF gap and a 0.037 unlock-rate
difference at power 0.9. The same test rejects PufferLib's reward (D1),
collapsed sleep (D4) and a single world (D6), and passes JAX against a
second JAX sample.

## Checked and equal

- Achievement reward weights (D1), measured.
- Timeout and day length: 100,000 and 300 ticks (`constants.h:30-31`;
  `craftax_state.py:113-114`).
- Autoreset and discount at terminal (D7).
- The 51 scalar observations and the out-of-map tiles (D8).
- Sleep and rest start conditions, and the crafting inputs the mask checks
  (D5).
- Everything else in the rules, by a chain of measurements:
  the port equals PufferLib bit for bit (README, "Parity"). Modulo their RNGs,
  PufferLib and the lockstep port differ only in 28 reviewed blocks:
  D9 (4), D13 (1), D14 (4), same-distribution identities (10), rendering
  (5), declarations (2), one rename and one refactor. The RNG helpers they
  call are pinned with the distributions they draw
  (the lockstep diff test). That test rejects every rule
  change tried on the changed lines: 8 named edits, 152 operator flips and
  1,073 name swaps, of which the test's previous, token-labelling version
  accepted 5, 144 and 980. The lockstep port matches JAX, MEASURED:
  the pin's own parity test (`tests/craftax_parity.py:547-660`, world
  generation plus packed observations, defaults `663-664`) passes 16 seeds
  x 200 random actions.
  Over 128 seeds x 1,000 actions, 112 seeds pass. Of the 16 that fail, 12
  differ in one D9 observation bit, 2 fail only the test's own structure
  assertions (identically in JAX), and 2 differ in one world-generation cell
  each, both D11 threshold flips. With exp002's options
  the port matches JAX transition by transition on JAX's own states
  (the original-setup test) and in distribution under a random policy
  (the trajectory test).

## Out of scope

The learning algorithm and its hyperparameters (`config/craftax.ini [train]`)
are chosen by an `exp` fork, not by environment options.
