# Attention

Vaswani et al., "Attention Is All You Need", arXiv:1706.03762.

## Notation

```
t                                  ∈ 1..n      # time step; n = sequence length
xₜ                                 ∈ ℝᶜ        # input
d                                              # head dim
```

## One head (causal)

```
Q,K,V                              ∈ ℝᵈˣᶜ      # Linear operators
qₜ    = Q xₜ                       ∈ ℝᵈ        # query
kₜ    = K xₜ                       ∈ ℝᵈ        # key
vₜ    = V xₜ                       ∈ ℝᵈ        # value
wₜₛ   = exp(qₜᵀ kₛ/√d) / Σᵣ₌₁ᵗ exp(qₜᵀ kᵣ/√d) ∈ (0,1) # s ≤ t; Σₛ wₜₛ = 1
oₜ    = Σₛ₌₁ᵗ wₜₛ vₛ               ∈ ℝᵈ        # weighted average of past values
```

Each query picks its own weights over every past token. minGRU's weights
are fixed by the gates as they pass ([mingru.md](mingru.md)).

## Multi-head

```
m                                              # heads; c = m · d
Q⁽ⁱ⁾,K⁽ⁱ⁾,V⁽ⁱ⁾                     ∈ ℝᵈˣᶜ      # head i = 1..m
oₜ⁽ⁱ⁾ = one head with Q⁽ⁱ⁾, K⁽ⁱ⁾, V⁽ⁱ⁾ ∈ ℝᵈ
Y                                  ∈ ℝᶜˣᶜ      # output projection
yₜ    = Y [oₜ⁽¹⁾, …, oₜ⁽ᵐ⁾]        ∈ ℝᶜ        # concat heads, then mix
```

- heads attend to different positions with different query–key matches.
- GQA: `g` groups of heads share one `K, V`; cache `2 g d`. MQA: `g = 1`.

## Block

```
x′ₜ   = xₜ + MHA(LN(x))ₜ           ∈ ℝᶜ        # pre-norm residual (resnet.md)
yₜ    = x′ₜ + MLP(LN(x′ₜ))         ∈ ℝᶜ
```

## Cost (per layer)

| layer        | train      | decode / token | cache / token      | file             |
|--------------|------------|----------------|--------------------|------------------|
| MHA          | `O(n² c)`  | `O(n c)`       | `2c`               |                  |
| GQA          | `O(n² c)`  | `O(n c)`       | `2gd`              |                  |
| MLA          | `O(n² c)`  | `O(n r)`       | `r + dᴿ`           | [mla.md](mla.md) |
| linear / GLA | `O(n c d)` | `O(c d)`       | none; `m d²` state | [gla.md](gla.md) |

## Linear attention

Drop the softmax: `oₜ = Σₛ (qₜᵀ kₛ) vₛ = Sₜᵀ qₜ` with `Sₜ = Sₜ₋₁ + kₜ vₜᵀ`.
A fixed-size state replaces the growing cache: [gla.md](gla.md).
