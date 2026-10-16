# Ghost site data, format `craftax-ghosts/4`

`build.py` writes one directory of data files; the page adds its own html,
js and sprites beside them. `layout.py` is the reference decoder of every file
below, and `testdata/fixture/` holds a real three-episode site with the values
it decodes to (`expected.json`, last section).

## Conventions

- Every integer in a binary file is little-endian. `u8` and `u32` are unsigned.
- A `.gz` file is one gzip member (level 9, mtime 0); decompress it whole.
- Floors are 0-8 (0 the surface, 8 the graveyard). Tiles are `(row, col)`,
  0-47, row-major; row grows downward.
- Facing and movement use the game's action ids: 1 LEFT `(0, -1)`, 2 RIGHT
  `(0, +1)`, 3 UP `(-1, 0)`, 4 DOWN `(+1, 0)`, as `(drow, dcol)`.
- Time is the decision index. `s_t` is the game before decision `t`; an
  episode of `T` decisions has states `s_0 .. s_T`, and `s_T` is its end (the
  death tile, the timeout tile, or the tile of its win). At time `t` an episode
  is live while `t < T` and ended from `t >= T` on, when only its end marker is
  drawn.
- Every episode of every tier starts from the same world and the same start
  (`manifest.start`).
- A ghost ends at its first DEFEAT_NECROMANCER (achievement 49): `T` is that
  decision plus one, and its outcome is `win`, whatever the episode did after.
- Outcomes: `death`, `timeout` (the game's 100,000-tick clock), `win`, and
  `truncated` (a capped capture cut the episode, alive, at its cap; no set of
  this format holds one, as `short` takes only ended episodes).

## Directory

```
manifest.json
world.bin.gz
<tier>/<set>/g<k>/episodes.json
<tier>/<set>/g<k>/players.bin.gz
<tier>/<set>/g<k>/events.bin.gz
<tier>/<set>/g<k>/creatures-w<j>.bin.gz      j = 0 .. windows - 1
<tier>/<set>/timeline-n<count>.bin.gz        one per count of the set, but wins
<tier>/<wins|unbroken>/g<k>/keeps.bin.gz     a time-mapped set's time maps
<tier>/<wins|unbroken>/g<k>/keeps-sleep.bin.gz   its time maps when sleep shows
<tier>/<wins|unbroken>/g<k>/sleep.bin.gz     its episodes' sleeps, tick by tick
```

Every tier has two sets, `all` and `short`, and a tier that wins a third,
`wins` (next section). Groups nest: with a
set's `counts = [100, 250, 500, 1000]`, group `g0` holds its episodes `[0,
100)`, `g1` `[100, 250)`, `g2` `[250, 500)` and `g3` `[500, 1000)`. Showing `n =
counts[k]` episodes loads groups `0..k`. Each group is self-contained: its
offsets point into its own files only.

## Sets

- `all`: the main capture's episodes by capture ordinal. Every episode started
  at the same moment, so the first `n` are an unbiased sample of the tier.
- `short`: every episode, of any of the tier's captures, that ends (death or
  win) within `manifest.short_decisions` (10,000) decisions; then wins longer
  than that, shortest first, until wins make up at least their natural share
  of the set, the share of wins among the deaths and wins of the uncapped
  captures (or every win, if there are fewer). The set is ordered by capture
  ordinal, then by capture (`source`), so its first `n` are an unbiased
  sample of it. A tier with fewer than 1,000 such episodes has fewer: its
  `counts` are the standard ones below its size, then its size.
- `wins`: every win of every one of the tier's captures, by capture ordinal,
  then capture: an unbiased order, as in `short`, but for one win the build
  pins by its sampling seed (`build.py --unbroken TIER=SEED`), which comes
  first so every count shows it: the page follows it, and it plays unbroken
  (`time_map.unbroken`). Its `counts` are the standard ones below its size,
  then its size, so its last count is every win (1,284 for the boss tier of
  world 15). Each win has a time map (`keeps.bin.gz`) and the set has no
  timelines: the time maps replace the quiet-stretch skip.
- `unbroken`, built only with `build.py --unbroken-set COUNT:STEPS` and only
  for a tier with a pinned win: `COUNT` wins that each play every decision,
  for a figure of whole games (`sets.unbroken_set`). The pinned win comes
  first; the others are, of the tier's other wins of at most `STEPS`
  decisions sorted by decisions, then ordinal, then capture, those at ranks
  `round(k (n - 1) / (COUNT - 2))` for `k = 0 .. COUNT - 2` (the shortest, the
  longest and evenly between, so their ends spread over the timeline), by
  ordinal, then capture. If fewer than `COUNT - 1` fit, it takes them all and
  the shortest longer wins, whose time maps then compress them to `STEPS`.
  Time-mapped like `wins`, with `time_map.rule.steps = STEPS`; every win that
  fits has the identity map (`time_map.whole`). The Craftax post's figure is
  this set, 50 wins of at most 6,000 decisions.

A set's `composition` shows how it was chosen: `run` (episodes considered: the
main capture's for `all`, every capture's for `short` and `wins`), `qualified` (those
meeting the rule, before the set's cap of `counts[-1]`), `deaths`, `wins`,
`timeouts`, `truncated` (in the set), `added_wins` (wins longer than
`short_decisions` added for the share), `natural_win_share`, and
`death_decisions` and `win_decisions` (`[shortest, longest]` in the set, or
`[]`).

## manifest.json

| Field | Meaning |
|---|---|
| `format` | `"craftax-ghosts/4"`: 3 with a fourth byte, the facing, in every creature of a sample. `/3` was 2 plus the `wins` set, its time maps and the `time_map` set field. |
| `git_commit` | Commit the builder ran from. |
| `game_package_digest` | Digest of the game's sources (`game.jit.package_digest`). |
| `platform` | `<os>-<machine>-<libc>`: the libm the replay and world ran under. |
| `world_seed` | The shared world W. |
| `start` | `[floor, row, col, facing]` of `s_0`; `[0, 24, 24, 3]`. |
| `creature_stride` | Decisions between creature samples (4). |
| `sleep_stride` | Ticks between a sleep's samples (4): one display step per 4 ticks when sleep shows. |
| `window_decisions` | Decisions per creature window file (8,192). A multiple of the stride. |
| `counts` | The standard cumulative group counts, `[100, 250, 500, 1000]`; each set lists its own. |
| `short_decisions` | The `short` set's cap (10,000; the fixture uses 160). |
| `quiet` | The timelines' rule: `{"per_live": 50, "min_run": 64, "keep": 8}` (timelines below; the fixture uses 1, 16 and 4). |
| `achievement_rewards` | 67 ints: each achievement's reward; an episode's return is the sum over its unlocked ones. |
| `tiers` | One entry per tier, below. |
| `files` | Every file but the manifest, path relative to the directory, to its sha256 (hex). |
| `sizes` | The same paths to their byte counts. |

A tier: `name` (its directory), `arm` (capture arm), `sources`, and `sets`
(`all` first, then `short`, then `wins` if the tier wins).

A source is one capture the tier's episodes came from, the main one first:
`root`, `capped` (whether it cut episodes at a decision cap), `episodes` (of
this tier's arm in it), and `provenance` (checkpoint path and sha256, sampler
seed). An episode's `source` indexes this list.

A set: `name`, `rule` (one sentence), `counts`, `episodes` (= `counts[-1]`),
`decisions` (sum of `T`), `max_decisions` (largest `T`), `composition`,
`groups`, `stats`, `timelines` (empty for `wins`), `time_map` (`null` but
for a time-mapped set) and `sleep_map` (the same for the view that shows
sleep, `keeps-sleep.bin.gz`; its `L` counts each kept decision and each sleep
sample, and `samples` how many samples the set's sleeps hold).

A `time_map`: `rule` (`{"steps", "levels"}`, below), `steps` (the longest
time map, `L` of the set's longest win to play: the length of its timeline),
`kept` (sum of `L`), `shortest` and `median` (`L`), `levels` (how many
compressed wins used each level of the rule, then how many needed progress
thinned), `unbroken` (`0`, the index of the pinned win, whose time map is
the identity, one run `[0, T)` with `L = T`; `null` when no win is pinned),
`whole` (how many wins have the identity map: the pinned one, and in
`unbroken` every win that fits), and `samples` (sleep samples its steps
include: 0 for `time_map`). A pinned win longer than `steps` fails the build.

A group: `path` (`<tier>/<set>/g<k>`), `first` (index of its first episode),
`count`, `decisions` (sum of its `T`), `windows` (`ceil(max T in the group /
window_decisions)`).

`stats` has one entry per count, over the set's first `count` episodes:
`count`, `mean_return`, `mean_decisions`, `reached` (9 ints: episodes that
stood on each floor), `deaths` (9 ints: deaths per end floor), `timeouts`,
`wins`, `escapes` (escape rows; 0 means every player byte decoded by rule).

A timeline: `count`, `path`, `decisions` (`D`, the largest `T` of the first
`count` episodes), `kept` (decisions playback keeps when it skips quiet
stretches) and `segments` (how many kept ranges).

## world.bin.gz

62,244 bytes after decompression, the world at its reset:

| Bytes | Content |
|---|---|
| 20,736 | `block[9][48][48]` u8: block ids 0-36 (`game.state.BlockType`). |
| 20,736 | `item[9][48][48]` u8: 0 none, 1 torch, 2 ladder down, 3 ladder up, 4 ladder down blocked. |
| 20,736 | `light[9][48][48]` u8: 0-255. Only the reset's; torches placed later light tiles the files do not track. |
| 18 | `down_ladders[9][2]` u8: each floor's down ladder `(row, col)`. |
| 18 | `up_ladders[9][2]` u8: each floor's up ladder `(row, col)`. |

A ladder entry means something only on floors that have that ladder (the
surface has no up ladder, the graveyard no down ladder). In `sprites.png` a
block draws as sprite `block` and an item as sprite `42 + item - 1`, as the
single-game viewer does.

## episodes.json

`{"tier", "set", "group", "first", "episodes": [...]}`, one object per episode
in index order:

| Field | Meaning |
|---|---|
| `index` | Index within the set, `first + i`. |
| `source` | Index of its capture in the tier's `sources`. |
| `ordinal` | Capture ordinal within that capture. |
| `sampling_seed` | The receipt's sampling seed, a decimal string (a u64). |
| `decisions` | `T`. |
| `outcome` | `"death"`, `"timeout"`, `"win"` or `"truncated"`. |
| `end` | `[floor, row, col, facing]` of `s_T`, the decoded path's last entry. |
| `achievement_return` | The summed reward of the achievements unlocked by `T`. |
| `floor_first` | 9 ints: the first `t` with the player on that floor in `s_t` (0 for the surface), or -1. |
| `players` | Offset of the episode's `T` bytes in `players.bin`. |
| `map`, `achievements`, `escapes` | `[start, count]`: the episode's rows in each table of `events.bin`. |

## players.bin.gz

The group's episodes' player bytes back to back; episode `i` owns
`[players, players + decisions)`. Byte `b_t` describes decision `t`, the step
from `s_t` to `s_(t+1)`:

| Bits | Meaning |
|---|---|
| 0-5 | `a_t`, the effective action (0-42; NOOP when the decision began asleep or resting). |
| 6 | `MOVED` (0x40): the player stepped one tile in `a_t`'s direction. |
| 7 | `FLOOR` (0x80): the player took the ladder `a_t` names. |

Decode `s_(t+1)` from `s_t = (floor, row, col, facing)`:

1. If an escape row has decision `t`, `s_(t+1)` is that row's floor, row,
   column and facing. Stop.
2. If `a_t` is 1-4, `facing = a_t` (a movement action turns the player whether
   or not it moves).
3. If `MOVED`, add `a_t`'s step to `(row, col)`.
4. Else if `FLOOR` and `a_t` is 18 (DESCEND): `floor += 1`, `(row, col) =
   up_ladders[floor]`.
5. Else if `FLOOR` (`a_t` is 19, ASCEND): `floor -= 1`, `(row, col) =
   down_ladders[floor]`.

The builder writes an escape for every decision these rules would get wrong,
so decoding is exact by construction; none has been seen in real play.

The interaction target of decision `t` is the tile one step from `s_t`'s
position in `s_t`'s facing, for `a_t` in {5 DO, 7 PLACE_STONE, 8 PLACE_TABLE, 9
PLACE_FURNACE, 10 PLACE_PLANT, 24 SHOOT_ARROW, 26 CAST_FIREBALL, 27
CAST_ICEBALL, 28 PLACE_TORCH}. It may lie off the map.

## events.bin.gz

Three tables back to back: map events, achievement events, escapes. A table
is a `u32` row count `n`, then each column in order as `n` values of its type,
each column zero-padded to a multiple of 4 bytes. Every table and column
therefore starts 4-byte aligned, so a column is a typed-array view.

| Table | Columns |
|---|---|
| map | `decision` u32, `floor` u8, `row` u8, `col` u8, `block` u8, `item` u8 |
| achievements | `decision` u32, `achievement` u8 |
| escapes | `decision` u32, `floor` u8, `row` u8, `col` u8, `facing` u8 |

An episode's rows are contiguous (`[start, count]` in `episodes.json`), in
decision order; map rows within a decision go by floor, then row, then column.

- A map row says that decision `t` left tile `(floor, row, col)` holding
  `(block, item)`: it holds them in `s_(t+1)` and after, until a later row for
  that tile. The map of `s_t` is the world plus every row with `decision < t`.
  Rows cover every floor: a decision changes only the floor it starts on, the
  one it ends on, and the surface, where sown plants ripen wherever the player
  is, and the builder checks all nine floors against the rows every 256
  decisions and at the episode's end.
- An achievement row says decision `t` unlocked it. The return at `s_t` sums
  `achievement_rewards` over rows with `decision < t`.
- An escape row overrides the player byte's decoding of decision `t` (above).

## timeline-n&lt;count&gt;.bin.gz

The activity of a set's first `count` episodes and the decisions playback keeps
when it skips quiet stretches. Two tables, as in `events.bin`:

| Table | Columns |
|---|---|
| activity | `active` u16: one row per decision `t` in `[0, D)` |
| segments | `start` u32, `stop` u32: the kept ranges `[start, stop)`, in order |

`active[t]` counts the shown episodes active at decision `t`: the decision
leaves the player on a tile that episode never stood on before, changes a
block or item, unlocks an achievement, changes the floor, or is the episode's
last. Decision `t` is quiet when `active[t] < max(1, ceil(live[t] / per_live))`,
`live[t]` being the shown episodes with `T > t`: fewer than 2% of the live
episodes do anything, and none at all when fewer than 51 are live. A decision at
which a shown episode dies or wins (its last decision, outcome `death` or `win`)
is never quiet, so skipping never jumps over a death or a win; a timeout's end
can be skipped. Every run of
at least `min_run` (64) quiet decisions collapses to its first and last `keep`
(8) decisions, so the motion into and out of it stays visible; the segments
are everything else. With skipping on, playback steps through the kept
decisions only and jumps over each gap.

## keeps.bin.gz (the `wins` set's time maps)

A win of `T` decisions plays as `L <= time_map.rule.steps` display steps: step
`u` shows `s_(d(u))`, `d` its time map, and from `u = L` on the win is over
and only its end marker is drawn, as at `t >= T` elsewhere. Two tables, as in
`events.bin`:

| Table | Columns |
|---|---|
| runs | `runs` u32: one row per episode of the group, in index order |
| keeps | `start` u32, `stop` u32: every episode's kept runs back to back |

Episode `i`'s kept runs are the `runs[i]` rows after the first `runs[0] + ... +
runs[i - 1]`: increasing, disjoint, non-empty `[start, stop)` decision ranges,
the first starting at 0 and the last stopping at `T`. `d(u)` is the `u`-th
decision they cover, in order, so `L` is their total length, `d(0) = 0` and
`d(L - 1) = T - 1`, the win.

The builder (`sets.keep_runs`) keeps every progress decision, one whose
activity is a first visit to a tile, a block or item change, an achievement, a
floor change, the end, a change in what the player holds (collected, crafted,
placed, shot, drank or read), or a fight (a creature on the player's floor
hurt or killed, a kill counted or the boss hurt), with `context` decisions on
each side; of every other stretch it keeps `idle` decisions spread evenly over
it, its first among them. Decision 0, floor changes and the last decision are
always kept. It tries the rule's `levels` of `[context, idle]` in order,
`[[2, 8], [2, 4], [1, 4], [1, 2], [0, 2], [0, 1], [0, 0]]`, and uses the first
whose `L` fits `steps` (5,000); if none does, it keeps the always-kept
decisions and as many progress decisions as fit, spread evenly.

## sleep.bin.gz and keeps-sleep.bin.gz (a time-mapped set's sleeps)

The capture's rules collapse a sleep into one decision: the step of a SLEEP
that puts the player to sleep plays every tick until they wake or the
episode ends (`game/step.py`, `Rules.collapse_sleep`), so the states between
decisions never show it. The builder plays each sleep again a tick at a time,
from the world before its decision, through the same game code with
`collapse_sleep` off (NOOP each tick while the player sleeps or rests, as the
collapsed loop plays), and checks that it ends on the very bytes, stream and
tick count of the collapsed step; any difference fails the build. It samples
the world after every `sleep_stride`-th tick that the player still sleeps
through: a sleep of `k` ticks has `floor((k - 1) / sleep_stride)` samples.
Five tables, as in `events.bin`:

| Table | Columns |
|---|---|
| sleeps per episode | `sleeps` u32: one row per episode of the group |
| sleeps | `decision` u32, `ticks` u32: every episode's sleeps back to back |
| samples | `run` u32, `changes` u32: every sleep's samples back to back: where its creature run starts in `runs`, and how many `changes` rows it has |
| changes | `floor` u8, `row` u8, `col` u8, `block` u8, `item` u8: every sample's back to back: each tile of the player's floor and floor 0 that differs from before the sleep's decision, as it is at the sample |
| runs | `byte` u8: the samples' creature runs back to back, each as one creature sample (below): the creatures on the player's floor |

A sample's run ends where the next one starts (the last, at the end of
`runs`). The player stays on one tile through a sleep, asleep.

`keeps-sleep.bin.gz` is a time map file as `keeps.bin.gz`, for the view that
shows sleep: the same rule, but every sleep's decision kept as the
always-kept ones are (the pinned win, and every win of `unbroken` that fits,
stay the identity). Display steps: each kept decision `d` is one step showing
`s_d`, and a kept sleep's decision is followed by one step per sample of it,
in order, each showing the sample (`layout.displayed_sleep`). With steps
added, `L` may exceed `rule.steps`.

## creatures-w&lt;j&gt;.bin.gz

Window `j` holds the creature samples of decisions `[j W, (j + 1) W)`, `W =
window_decisions`: `u32 n` (the group's episode count), `u32 offsets[n + 1]`,
then the body. Episode `i`'s run is `body[offsets[i] : offsets[i + 1]]`; the
body starts at byte `4 (n + 2)`.

A run is the samples at `t = j W, j W + S, j W + 2S, ...` with `t < min((j + 1)
W, T)`, `S = creature_stride`; an episode with `T <= j W` has an empty run.
Each sample is `u8 k`, then `k` creatures of 4 bytes: `u8 kind`, `u8 row`, `u8
col`, `u8 facing`, with `class = kind >> 4` and `species = kind & 15`. A
projectile's (class 3 or 4) `facing` is the move action of the tile it flies
each tick (the game's `mob_projectile_dirs` and
`player_projectile_directions`): 1 LEFT `(0, -1)`, 2 RIGHT `(0, +1)`, 3 UP
`(-1, 0)`, 4 DOWN `(+1, 0)`; any other creature's is 0. A sample lists the
creatures of `s_t` on `s_t`'s floor, by class and then slot. Creatures frozen
on floors the player has left are not sampled.

| Class | Creature | Sprite (single-game viewer's `mapSprite`) |
|---|---|---|
| 0 | melee mob | `80 + species` |
| 1 | passive mob | `88 + species` |
| 2 | ranged mob | `91 + species` |
| 3 | mob projectile | `[51, 99, 100, 101, 51, 102, 100, 101][species]`, turned to its facing |
| 4 | player projectile | as class 3 |

A projectile's sprite points up; it is drawn turned as original Craftax's
renderer turns it (`craftax/renderer.py`): flipped top to bottom when it
flies down or right, then, flying left or right, transposed (rows for
columns). Sprite 51 is the arrow pointing up (50 points down).

## The fixture: testdata/fixture/

`site/` is a real site written by `fixture.py` on world 1 of the platform its
manifest names: a tier, `fixture`, of three episodes, in as few files as the
format allows, and a tier, `fixture-wins`, of the same three won at their
first wooden sword, which two of them make: its `wins` set is one group of
those two under a 120-step rule of levels `[[1, 2], [0, 1], [0, 0]]`, the
118-decision win of sampling seed 2 pinned first and unbroken, the
122-decision one compressed. Its `all` set is one group of all three (`counts = [3]`); its
`short` set, capped at 160 decisions, is one group of the two episodes that
died by then (`counts = [2]`). Each group has one creature window. Its quiet
rule makes a decision quiet unless every live episode is active, so its
timelines collapse runs. Episode 0 walks to the surface's down ladder,
descends, wanders on floor 1, ascends and then plays randomly; episodes 1 and
2 play uniformly random legal actions. Several groups and windows are tested
on the builder's own sites (`build_test.py`) and the page's synthetic one
(`viewer/synthetic.mjs`).

`expected.json` is `{"world_seed", "episodes": [...], "timelines": [...]}` with
what `layout.py` decodes. A timeline entry is `{"tier", "set", "count",
"activity", "segments"}`, the file's two tables as lists. An episode entry, in
tier, set and index order, is a full entry (below) or, for a ghost an earlier
full entry already holds (the same `path`: one episode in two sets or two
tiers), `{"tier", "set", "index", "same_as"}` plus only the fields whose
values differ from that entry's or that it lacks; `same_as` is that entry's
position in `episodes`, and the entry reads as that entry's fields under its
own (`fixture.resolved`). A full entry:

| Field | Meaning |
|---|---|
| `tier`, `set`, `index`, `decisions`, `outcome`, `end`, `achievement_return` | As in `episodes.json`. |
| `path` | `T + 1` entries `[floor, row, col, facing]`: `s_0 .. s_T`. |
| `interactions` | `[t, floor, row, col]` of each interaction target. |
| `final_changes` | `[floor, row, col, block, item]` of every tile whose `s_T` value differs from the world, sorted. |
| `final_map_sha256` | sha256 of `s_T`'s `block[9][48][48]` bytes, then its `item[9][48][48]` bytes. |
| `creatures` | `[t, [[class, species, row, col, facing], ...]]` per sample, every window's in order. |
| `displayed` | For a time-mapped set only: `d(u)` for `u = 0 .. L - 1`. |
| `displayed_sleep` | For a time-mapped set only: `[decision, sample]` per step of the view that shows sleep, `sample` 1-based within its sleep, 0 for `s_decision`. |
| `sleeps` | For a time-mapped set only: `[decision, ticks]` of each sleep. |
| `sleep_creatures` | For a time-mapped set only: each sleep sample's `[[class, species, row, col, facing], ...]`, in order. |
