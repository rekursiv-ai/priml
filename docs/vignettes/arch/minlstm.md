# minLSTM

Feng et al., "Were RNNs All We Needed?", arXiv:2410.01201v1, §3.2 and App. B.2.2.

Notation, `g`, stacking, and training as in [mingru.md](mingru.md).

## LSTM (sequential)

```
F,I,C,O                            ∈ ℝᵇˣ⁽ᶜ⁺ᵇ⁾  # Linear operators; biases omitted
fₜ    =    σ(F [xₜ, hₜ₋₁])         ∈ (0,1)ᵇ    # forget gate: share of cₜ₋₁ kept
iₜ    =    σ(I [xₜ, hₜ₋₁])         ∈ (0,1)ᵇ    # input gate: share of c̃ₜ written
c̃ₜ    = tanh(C [xₜ, hₜ₋₁])         ∈ (−1,1)ᵇ   # candidate, needs hₜ₋₁
oₜ    =    σ(O [xₜ, hₜ₋₁])         ∈ (0,1)ᵇ    # output gate
cₜ    = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ        ∈ ℝᵇ        # cell (memory)
hₜ    = oₜ ⊙ tanh(cₜ)              ∈ (−1,1)ᵇ   # output; n steps in order
```

## minLSTM (parallel)

```
F,I,H                              ∈ ℝᵇˣᶜ      # LSTM's F, I, C (C → H) minus the hₜ₋₁ columns
fₜ    = σ(F xₜ)                    ∈ (0,1)ᵇ    # all t at once
iₜ    = σ(I xₜ)                    ∈ (0,1)ᵇ    # all t at once
f′ₜ   = fₜ / (fₜ + iₜ)             ∈ (0,1)ᵇ    # f′ₜ + i′ₜ = 1
i′ₜ   = iₜ / (fₜ + iₜ)             ∈ (0,1)ᵇ
h̃ₜ    = g(H xₜ)                    ∈ (0,∞)ᵇ    # all t at once
hₜ    = f′ₜ ⊙ hₜ₋₁ + i′ₜ ⊙ h̃ₜ      ∈ ℝᵇ        # minGRU's update with 1 − uₜ → f′ₜ, uₜ → i′ₜ
```

LSTM, but:

1. `fₜ`, `iₜ`, `h̃ₜ` read `xₜ` only, not `hₜ₋₁`.
2. del both `tanh`s.
3. normalize `fₜ, iₜ` so `f′ₜ + i′ₜ = 1`; else `cₜ` grows with `t`
   (e.g. `fₜ, iₜ → 1` gives `cₜ → c₀ + Σ c̃ₛ`).
4. del `oₜ` and `cₜ`; with no output gate, `hₜ = cₜ`.

## Log-space scan

Same scan as minGRU, with these coefficients (App. B.2.2):

```
log f′ₜ            = −softplus(softplus(−F xₜ) − softplus(−I xₜ))
log(i′ₜ ⊙ h̃ₜ)      = −softplus(softplus(−I xₜ) − softplus(−F xₜ)) + log g(H xₜ)
log γₜ             = Σₛ₌₁ᵗ log f′ₛ                  # cumsum; γ₀ = 1
log hₜ             = log γₜ + logcumsumexpₛ₌₀ᵗ (log(i′ₛ ⊙ h̃ₛ) − log γₛ) # s = 0 term: log h₀
```

## vs minGRU

|                                | minGRU               | minLSTM                |
|--------------------------------|----------------------|------------------------|
| keep / write                   | `1 − uₜ`, `uₜ`       | `f′ₜ`, `i′ₜ`           |
| weights                        | `U,H ∈ ℝᵇˣᶜ` (`2bc`) | `F,I,H ∈ ℝᵇˣᶜ` (`3bc`) |
| selective copy, `ℓ = 3`        | 99.5 ± 0.2           | 96.0 ± 2.8             |
| speedup vs original, `n = 512` | 175×                 | 235×                   |

Paper: two independent gates pull the keep/write ratio in different
directions during training, so minLSTM is less stable than minGRU (§4.1).
