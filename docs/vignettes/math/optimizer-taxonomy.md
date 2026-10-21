# A Taxonomy of First-Order Methods

Every first-order method in this collection is one update rule with four
choices in it. The rule is regularized follow-the-leader on linearized,
transformed losses:

  ```
  x_{t+1} = argmin_x  Σ_{s≤t} w_{t,s} ⋅ ⟨ψ(g_s), x⟩  +  R_t(x)
  ```

- `ψ` acts on each draw's gradient before it is summed: influence.
- `w_{t,s}` weights past steps: momentum (direction) and memory (scale).
- `R_t` is the penalty on moving, a norm or Bregman divergence possibly
  learned from the past: geometry.

The objective (which functional the loss's minimizer estimates) sits above
the rule, and the oracle (values, gradients, Hessians) below it. This
vignette derives the methods from the rule and places them on the axes.

Let `g_t` be a stochastic gradient at `x_t`, possibly an average over a batch
of per-draw gradients `g_{t,i}`, and,

  ```
  ψ        a map on gradients, applied per draw before averaging
  w_{t,s}  ≥ 0, weights on past rounds
  R_t      a strongly convex regularizer, R_t(x) = (1/2η) ‖x − x_t‖²_t or
           (1/η) D_φ(x, x_t), with ‖·‖_t possibly built from g₁..g_t
  ```

First, the one-step case is proximal descent. Take `w_{t,s} = 𝟙[s = t]`
(only the current gradient), `ψ = id`, and `R_t = (1/2η)‖x − x_t‖²`.
Completing the square (gd-as-proximal.md),

  ```
  x_{t+1} = argmin_x ⟨g_t, x⟩ + (1/2η) ‖x − x_t‖²
          = x_t − η g_t
  ```

Gradient descent. Each axis below changes one choice from this baseline.

Second, geometry: the penalty `R_t`. With a norm ball instead of a quadratic,
the step is the norm's linear minimization oracle,

  ```
  argmin_{‖d‖ ≤ 1} ⟨g, d⟩    =  −g/‖g‖₂            ℓ₂: normalized GD
                             =  −sign(g)           ℓ∞: sign descent
                             =  −U Vᵀ              spectral: Muon (muon.md)
  argmin_d ⟨g, d⟩ + ½‖d‖²_H  =  −H⁻¹ g             Newton (newton-as-proximal.md)
  argmin_x ⟨g, x⟩ + D_φ(x,x_t)/η                   mirror descent (mirror-descent.md)
  ```

where the ℓ∞ case is coordinatewise, and the spectral case uses
`G = U Σ Vᵀ`. The norm picks which directions are cheap.

Third, memory: learning `R_t` from the past. Take
`R_t(x) = (1/2η) ‖x − x_t‖²_{A_t}` with `A_t` built from squared gradients,

  ```
  A_t                                     update
  diag( √Σ_{s≤t} g_s² )                   AdaGrad
  diag( √EMA_β₂(g²) )                     RMSProp; with momentum, Adam (adam.md)
  (Σ G_s G_sᵀ)^{1/4} ⊗ (Σ G_sᵀ G_s)^{1/4} Shampoo
  (G_t G_tᵀ)^{1/4} ⊗ (G_tᵀ G_t)^{1/4}     Muon (current gradient only)
  ```

The last row checks: `(G Gᵀ)^{−1/4} G (Gᵀ G)^{−1/4}
= U Σ^{−½} Uᵀ ⋅ U Σ Vᵀ ⋅ V Σ^{−½} Vᵀ = U Vᵀ`. So

  ```
  memory     vector (diagonal)   matrix (Kronecker)
  none       sign descent        Muon
  EMA        Adam                SOAP, EMA-Shampoo
  full sum   AdaGrad             Shampoo
  ```

and `Muon : Shampoo :: sign descent : AdaGrad`. Zero memory is a fixed norm;
full memory is a norm learned from data; the square root in AdaGrad and
Shampoo is what makes them regret-optimal over diagonal and Kronecker
preconditioners (Duchi–Hazan–Singer 2011; Gupta–Koren–Singer 2018).

Fourth, momentum: the weights on past linear losses. With a fixed quadratic
`R` centered at `x₀` and `w_{t,s} = 1`, the rule is dual averaging,

  ```
  x_{t+1} = x₀ − η Σ_{s≤t} g_s
  ```

With `w_{t,s} = (1 − β) β^{t−s}` and the regularizer re-centered at `x_t`,

  ```
  x_{t+1} = x_t − η m_t,    m_t = (1 − β) Σ_{s≤t} β^{t−s} g_s
  ```

heavy-ball momentum. Nesterov (nesterov.md) evaluates the gradient at an
extrapolated point instead, an estimate sequence that makes the weights grow
like `t`. Momentum reweights which losses enter the model; it does not
change the norm.

Fifth, influence: `ψ` before averaging. Robbins–Monro with a nonlinearity
(polyak-tsypkin.md) has asymptotic variance `𝔼[ψ²]/𝔼[ψ′]²`, minimized by the
noise score and, over a contamination neighborhood, by Huber's clip
(least-favorable-density.md). The same `ψ` can act at any granularity,

  ```
  one draw                      ψ                     bounds influence of
  a residual coordinate         Huber / z cap         outlying coordinates
  a sample's gradient           per-sample clip       outlying samples
  a batch's gradient            global norm clip      outlying batches
  ```

and only the granularity at which the outliers live is protected. Anything
applied after the average (a global clip, a learning rate, Adam's `√v`) acts
on the sum and controls step size, not which draws choose the direction.

Hence the rule with its four choices,

  ```
  x_{t+1} = argmin_x  Σ_{s≤t} w_{t,s} ⟨ψ(g_s), x⟩  +  R_t(x)
            ─────────────────────────────────────    ─────
            momentum (w), influence (ψ)               geometry (R), memory (R_t)
  ```

recovers GD, sign descent, Newton, mirror descent, AdaGrad, Adam, Shampoo,
Muon, heavy ball and dual averaging, and Huber-robust Robbins–Monro, and a
new method is a choice not yet made.

Examples.

- Adam: `ψ = id`, `w` an EMA (`β₁`), `R_t` diagonal from an EMA of `g²`
  (`β₂`). Its scale-free bounded step comes from `R_t`; its lack of robustness
  from `ψ = id`.
- Muon's successors. NorMuon restores a diagonal `R_t` after the spectral
  one (memory); MuonClip and spectral clipping put a `ψ` on logits or
  singular values (influence); Dion approximates the spectral `R_t` at low
  rank (geometry). Each moves along an axis Muon left at zero.
- A training blow-up from rare far residuals under a Gaussian NLL. The fix
  lives on the influence axis at the coordinate granularity (a z cap); a
  global clip, a smaller rate, a longer warmup, or a faster `β₂` move other
  axes and do not bound it.

Regularity: the derivations use a linear model of the loss at each step
(Newton's quadratic model is the exception, placed on the geometry axis via
`‖·‖_H`). Regret bounds for the memory axis need convex losses and bounded
gradients; the Polyak–Tsypkin variance needs the stochastic-approximation
conditions there. Muon's and Shampoo's matrix forms act per weight matrix.
