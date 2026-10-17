# Parallel scan

Blelloch, "Prefix Sums and Their Applications", CMU-CS-90-190, 1990.
Heinsen, "Parallelization of an Ubiquitous Sequential Computation", arXiv:2311.06281.

## Notation

```
t                                  ∈ 1..n      # time step; n = sequence length
aₜ                                 ∈ ℝᵇ        # decay; known before the scan
bₜ                                 ∈ ℝᵇ        # input; known before the scan
vₜ                                 ∈ ℝᵇ        # state; v₀ given
⊙                                              # elementwise product
```

## Recurrence

```
vₜ    = aₜ ⊙ vₜ₋₁ + bₜ             ∈ ℝᵇ        # n steps in order
```

## Scan

Each step is the affine map `v ↦ a ⊙ v + b`. Composing two steps is another
affine map, and composition is associative:

```
(a, b) ∘ (a′, b′) = (a′ ⊙ a, a′ ⊙ b + b′)      # apply (a, b), then (a′, b′)
```

So all `n` prefixes take `O(log n)` rounds of parallel compositions (Blelloch).
Unrolled:

```
γₜ    = ∏ₛ₌₁ᵗ aₛ                   ∈ ℝᵇ        # cumprod; γ₀ = 1
vₜ    = γₜ ⊙ (v₀ + Σₛ₌₁ᵗ bₛ / γₛ)  ∈ ℝᵇ        # cumsum
```

`γₛ → 0` underflows, so compute in log space (Heinsen); needs `bₛ > 0`:

```
log vₜ = log γₜ + logcumsumexpₛ₌₀ᵗ (log bₛ − log γₛ) # b₀ = v₀
```

## Chunked

Matrix states (`Sₜ ∈ ℝᵈˣᵈ`) make each composition cost a matmul. Instead:

1. split `n` into chunks of length `C`.
2. inside a chunk: one `C × C` masked matmul (tensor cores).
3. across chunks: carry the state, `n / C` serial steps.

Used by Mamba-2, GLA, DeltaNet, KDA.

## Users

| model    | `aₜ`                 | `bₜ`            | state  | file                       |
|----------|----------------------|-----------------|--------|----------------------------|
| minGRU   | `1 − uₜ`             | `uₜ ⊙ h̃ₜ`      | `ℝᵇ`   | [mingru.md](mingru.md)     |
| minLSTM  | `f′ₜ`                | `i′ₜ ⊙ h̃ₜ`     | `ℝᵇ`   | [minlstm.md](minlstm.md)   |
| Mamba    | `exp(A Diag(δₜ))`    | `kₜ (δₜ ⊙ xₜ)ᵀ` | `ℝᵖˣᶜ` | [mamba.md](mamba.md)       |
| GLA      | `Diag(αₜ)`           | `kₜ vₜᵀ`        | `ℝᵈˣᵈ` | [gla.md](gla.md)           |
| DeltaNet | `αₜ (I − βₜ kₜ kₜᵀ)` | `βₜ kₜ vₜᵀ`     | `ℝᵈˣᵈ` | [deltanet.md](deltanet.md) |

DeltaNet's `aₜ` is a full matrix (applied by matmul, not `⊙`); still affine, so
still a scan, but chunked with a WY form instead of elementwise.

## Limit

Brandoit et al., arXiv:2609.32447: at finite precision, every affine update
`vₜ = aₜ vₜ₋₁ + bₜ` (every row above) tracks at most definite automata, so
it fails general state tracking at long lengths. Their NFSM keeps a scan
but drops the affine form; synthetic results only.
