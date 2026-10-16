# Recurrent memory: one family

Every recurrent layer in this folder updates a memory with the same
recurrence; they differ only in the memory's shape and which gates are on.
Read this first; each linked vignette is one row.

## Slots vs operator

minGRU's memory is a set of slots; the GLA lineage's memory is a linear
operator, a learned key → value map, trained online one pair per token.

```
slots     hₜ = (1 − uₜ) ⊙ hₜ₋₁ + uₜ ⊙ h̃ₜ   ∈ ℝᵇ      # memory is the output; slot i fixed
          yₜ = hₜ
operator  Sₜ = Sₜ₋₁ + kₜ vₜᵀ               ∈ ℝᵈˣᵈ    # each write adds one pair kₜ → vₜ
          yₜ = Sₜᵀ qₜ                      ∈ ℝᵈ      # read = evaluate the map at qₜ
```

| | slots (minGRU) | operator (GLA lineage) |
|---|---|---|
| memory | `b` numbers, one per slot | `d × d` matrix |
| write | blend slot `i` | add or fit one pair `kₜ → vₜ` |
| read | the memory itself | apply it to a query |
| address | fixed by channel | chosen by key, matched by query |

Bigger is not the point: `d × d` can be smaller than `b`. What changes is
that the model can store a vector under a key it picks and fetch it later.

Write rules are ways of fitting the operator to the pairs seen so far
(test-time regression; Wang, Shi, Fox, arXiv:2501.12352):

| write | fits | interference |
|-------|------|--------------|
| add | nothing; sums pairs | pairs with overlapping keys blend |
| delta | one gradient step on `‖Sᵀ kₜ − vₜ‖₂²` | old pairs drift as new ones fit |
| least squares | all pairs, `argmin Σₛ ‖Sᵀ kₛ − vₛ‖₂²` | minimal for a linear map |

## Notation

```
xₜ                                 ∈ ℝᶜ        # input
kₜ    = K xₜ                       ∈ ℝᵈ        # key: where to write
vₜ    = V xₜ                       ∈ ℝᵈ        # value: what to write
qₜ    = Q xₜ                       ∈ ℝᵈ        # query: where to read
Sₜ                                 ∈ ℝᵈˣᵈ      # memory: row i holds the value stored under key channel i
hₜ                                 ∈ ℝᵇ        # memory when it is a vector (minGRU)
```

## The general update (GDN-2)

```
Sₜ    = (I − kₜ (εₜ ⊙ kₜ)ᵀ) Diag(αₜ) Sₜ₋₁ + kₜ (ωₜ ⊙ vₜ)ᵀ   ∈ ℝᵈˣᵈ
yₜ    = Sₜᵀ qₜ                                            ∈ ℝᵈ
```

| gate | shape | job | minGRU analogue |
|------|-------|-----|-----------------|
| `αₜ` | `(0,1)ᵈ` | fade: shrink each row before writing | `1 − uₜ` |
| `εₜ` | `[0,1]ᵈ` | erase: remove what is stored at `kₜ` | none (slots are disjoint) |
| `ωₜ` | `[0,1]ᵈ` | write strength, per value channel | `uₜ` |

## Every model is a setting of the gates

| model | memory | `αₜ` (fade) | `εₜ` (erase) | `ωₜ` (write) | file |
|-------|--------|-------------|--------------|--------------|------|
| minGRU | `ℝᵇ` | `1 − uₜ` | `0` | `uₜ` (via `kₜ`) | [mingru.md](mingru.md) |
| linear attention | `ℝᵈˣᵈ` | `1` | `0` | `1` | [gla.md](gla.md) |
| GLA | `ℝᵈˣᵈ` | `Diag(αₜ)` | `0` | `1` | [gla.md](gla.md) |
| Mamba-2 | `ℝᵈˣᵈ` | scalar `αₜ` | `0` | `δₜ` | [mamba.md](mamba.md) |
| DeltaNet | `ℝᵈˣᵈ` | `1` | `βₜ` | `βₜ` | [deltanet.md](deltanet.md) |
| Gated DeltaNet | `ℝᵈˣᵈ` | scalar `αₜ` | `βₜ` | `βₜ` | [deltanet.md](deltanet.md) |
| KDA | `ℝᵈˣᵈ` | `Diag(αₜ)` | `βₜ` | `βₜ` | [kda.md](kda.md) |
| GDN-2 | `ℝᵈˣᵈ` | `Diag(αₜ)` | `εₜ` | `ωₜ` | [kda.md](kda.md) |

## Reductions (checked numerically, max diff 0 to 1e-15, 20 steps)

```
GDN-2, εₜ = 0, ωₜ = 1                      = GLA
GDN-2, αₜ = 1, εₜ = ωₜ = βₜ 1, ‖kₜ‖₂ = 1    = DeltaNet
GLA, one value dim, vₜ = 1,
     αₜ = 1 − uₜ, kₜ = uₜ ⊙ h̃ₜ             = minGRU
```

So minGRU is the operator family with one value column and no erase: the
operator degenerates to a vector of slots that fade and get written.

## Two axes

```
                    no erase           erase at kₜ
scalar fade         Mamba-2            Gated DeltaNet
per-channel fade    GLA  (⊃ minGRU)    KDA  →  GDN-2 (split erase/write)
```

- **Fade** controls time: how fast old items weigh less.
- **Erase** controls collisions: a new value replaces, not piles onto,
  what is stored under the same key.

## Write rules beyond one step

The rows above take one fitting step per token. Two rules fit harder:

| rule | update | file |
|------|--------|------|
| delta (one gradient step) | `S += β k (v − Sᵀ k)ᵀ` | [deltanet.md](deltanet.md) |
| Kalman / least squares | gain from carried uncertainty | KLA, KDN, Gated KalmaNet |

Toy (T = 256 random items, 4,096-number memory): additive write recall
0.39, least-squares write 0.85. The least-squares write is the ceiling for
a linear memory of that size.

## What is kept

Every memory decides which past tokens survive, and whether exactly or
blended into a state:

| keeps | exact (cache) | blended (state) |
|---|---|---|
| everything | attention | linear attention, no fade |
| recent | sliding window: hard cutoff after `L` | minGRU, GLA, Mamba: fade |
| chosen | H2O, LoLA, EM-LLM: by importance or surprise | DeltaNet, KDA: erase by key |

- Columns: a kept token is stored itself, or mixed into one state.
- Rows: all tokens survive, the most recent, or ones a rule picks.
- Caches: [lola.md](lola.md). States that are networks: [ttt.md](ttt.md).
- Raven (arXiv:2607.25357) writes each token to the top-`K` of `M` slots by a
  learned router and freezes the rest: a blended state with chosen writes.

## Limits shared by all rows

- All are affine in `Sₜ₋₁` with gates computed from `xₜ`, so all train with
  the same scan ([scan.md](scan.md)) and share its state-tracking limit.
- A linear operator with `d`-dim keys holds about `d` pairs cleanly; beyond that recall degrades
  by crowding (no fade) or forgetting (fade or erase). Keeping every item
  needs a cache: [attention.md](attention.md).
