# LoLA (and sparse caches)

A sparse cache keeps some past `(kₛ, vₛ)` pairs exactly and reads them with
softmax. The methods differ only in which pairs they keep; LoLA keeps the ones
its linear state cannot recall.

## Notation (one head)

```
t                                  ∈ 1..n      # time step; n = sequence length
xₜ                                 ∈ ℝᶜ        # input
d                                              # head dim
softmaxₛ(aⱼ : j ∈ J) = exp(aₛ) / Σ(exp(aⱼ) : j ∈ J)  ∈ (0,1)  # s ∈ J
Q,K,V                              ∈ ℝᵈˣᶜ      # Linear operators
qₜ    = Q xₜ                       ∈ ℝᵈ        # query
kₜ    = K xₜ                       ∈ ℝᵈ        # key
vₜ    = V xₜ                       ∈ ℝᵈ        # value
Cₜ    ⊆ {1..t}                                 # indices kept at time t
λ                                              # cache size: |Cₜ| ≤ λ
L                                              # window size
```

## Read (shared by every row)

```
wₜₛ   = softmaxₛ(qₜᵀ kⱼ / √d : j ∈ Cₜ) ∈ (0,1)  # s ∈ Cₜ
yₜ    = Σ(wₜₛ vₛ : s ∈ Cₜ)          ∈ ℝᵈ        # attention over the kept set
```

Attention, but: `s ≤ t` becomes `s ∈ Cₜ`. Everything below is a rule for `Cₜ`.

## Write: which pairs survive

```
Cₜ    = top-λ of scoreₜ(s) over s ∈ Cₜ₋₁ ∪ {t}   # evict the lowest-scoring pair
```

| method | `scoreₜ(s)` | keeps | file |
|--------|-------------|-------|------|
| attention | `1` | every pair (`λ = t`) | [attention.md](attention.md) |
| sliding window | `s` (newest wins) | last `L` | |
| H2O | `Σ(wᵣₛ : r ≤ t)`, attention received so far | heavy hitters + recent | |
| LoLA | `‖m(kₛ) − vₛ‖₂`, self-recall error | what the state can't recall | below |
| EM-LLM | events cut where `−log p(xₜ \| x<ₜ) > μ + γ σ` | surprising episodes | below |
| Memorizing Transformers | keep all; read top-k by `qₜᵀ kₛ` | everything, read sparsely | |

## LoLA

McDermott, Heath, Parhi, "LoLA: Low-Rank Linear Attention With Sparse Caching", arXiv:2505.23666, §3.

Three stores. Pairs leaving the window are either cached or folded into a linear state.

```
φ                                  : ℝᵈ → ℝᴰ   # learned feature map; D = 2d
Hₜ    = Hₜ₋₁ + Σ(φ(k) vᵀ : (k,v) ∈ Sₜ) ∈ ℝᴰˣᵈ    # linear attention state (gla.md)
sₜ    = sₜ₋₁ + Σ(φ(k)    : (k,v) ∈ Sₜ) ∈ ℝᴰ      # its normalizer
m(k)  = Hₜᵀ φ(k) / sₜᵀ φ(k)          ∈ ℝᵈ        # what the state recalls for k
SRE(k,v) = ‖m(k) − v‖₂               ∈ ℝ≥0       # self-recall error

Eₜ    = Gₜ₋₁ ∪ {(kₜ₋ₗ, vₜ₋ₗ)}                    # cache + pair leaving the window
Gₜ    = top-λ of SRE over Eₜ                     # sparse cache: hardest to recall
Sₜ    = Eₜ \ Gₜ                                  # the rest go into Hₜ

yₜ    = [ φ(qₜ)ᵀ Hₜ + Σ(eᵢ vᵢ : i ∈ Gₜ) + Σⱼ₌ₜ₋ₗ₊₁ᵗ eⱼ vⱼ ]
      / [ φ(qₜ)ᵀ sₜ + Σ(eᵢ    : i ∈ Gₜ) + Σⱼ₌ₜ₋ₗ₊₁ᵗ eⱼ    ]   ∈ ℝᵈ
eᵢ    = exp(qₜᵀ kᵢ / √d)
```

Sliding window + linear attention, but:

1. A pair leaving the window is cached if the linear state recalls it worst, else folded in.
2. Cached pairs are re-scored each update; one that the state now recalls is folded in.
3. One normalizer shared by all three stores.

- Training-free: applied at inference to a distilled window + linear model (LoLCATs).
- Updated once per chunk; `L = λ = 64` on pass-key: 0.6% → 97.4% (Llama-3.1 8B base).

## EM-LLM

Fountas et al., "Human-inspired Episodic Memory for Infinite Context LLMs", arXiv:2407.09450, §3.

```
surpₜ = −log p(xₜ | x₁..ₜ₋₁)        ∈ ℝ≥0       # the LM's own surprise
bₜ    = [surpₜ > μ + γ σ]           ∈ {0,1}     # boundary; μ, σ over the last τ tokens
events = runs of tokens between boundaries, refined by key-graph modularity
```

Retrieval: the `k` events most similar to `qₜ`, plus their temporal neighbours; read with
softmax beside the local window. No training; the whole past stays in (host) memory.

## Against the other memories

| keeps | exact (cache) | blended (state) |
|---|---|---|
| everything | attention | linear attention, no fade |
| recent | sliding window | minGRU, GLA, Mamba: fade |
| chosen | H2O, LoLA, EM-LLM | DeltaNet, KDA: erase by key |

LoLA sits in both columns: hard pairs cached, the rest blended.
