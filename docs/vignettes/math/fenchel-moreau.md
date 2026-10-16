# Fenchel–Moreau

Fenchel–Moreau follows from two facts:

1. Fenchel–Young gives `f** ≤ f` for any proper `f`, and,
2. a closed convex epigraph is cut off from any outside point by a hyperplane.

Let,

  ```
  f : ℝⁿ → ℝ ∪ {+∞}            proper, closed (lower semicontinuous), convex
  epi f  ≝ { (x,t) ∈ ℝⁿ × ℝ : f(x) ≤ t }       nonempty, closed, convex
  f*(y)  ≝ sup_x ( ⟨x,y⟩ − f(x) )
  f**(x) ≝ sup_y ( ⟨x,y⟩ − f*(y) )
  ```

The separation theorem used: if `C ⊂ ℝᵏ` is nonempty, closed, and convex and
`p ∉ C`, there are `a ∈ ℝᵏ` and `c ∈ ℝ` with `⟨a,z⟩ ≤ c < ⟨a,p⟩` for all
`z ∈ C`. (Take `a = p − P_C p`, with `P_C` the Euclidean projection.)

First, `f** ≤ f`. By Fenchel–Young, `⟨x,y⟩ − f*(y) ≤ f(x)` for every `y`,

  ```
  f**(x) = sup_y ( ⟨x,y⟩ − f*(y) )
         ≤ f(x)
  ```

Second, separate a point below the epigraph. Suppose `f**(x₀) < r < f(x₀)`
for some `x₀` and `r`. Then `(x₀, r) ∉ epi f`, so there are `(y, β)` and `c`
with

  ```
  ⟨y,x⟩ + β t ≤ c < ⟨y,x₀⟩ + β r        for all (x,t) ∈ epi f
  ```

Since `t` may grow without bound in `epi f`, `β ≤ 0`.

Third, the case `β < 0`. Rescale so `β = −1`. Put `t ≝ f(x)` for each `x` in
`dom f`; then `⟨y,x⟩ − f(x) ≤ c`, so `f*(y) ≤ c`, and

  ```
  f**(x₀) ≥ ⟨y,x₀⟩ − f*(y)
          ≥ ⟨y,x₀⟩ − c
          > r
  ```

This contradicts `f**(x₀) < r`.

Fourth, `f*` is somewhere finite. Pick `x₁ ∈ dom f` and separate the point
`(x₁, f(x₁) − 1)` from `epi f`. Its `β` cannot be `0`: putting `x = x₁` would
give `⟨y,x₁⟩ ≤ c < ⟨y,x₁⟩`. So `β < 0`, and the third step's argument gives
some `y₀` with `f*(y₀) < ∞`.

Fifth, the case `β = 0`. Then `⟨y,x⟩ ≤ c < ⟨y,x₀⟩` for all `x ∈ dom f`. For
`λ > 0`, tilt `y₀` by `λy`,

  ```
  f*(y₀ + λy) = sup_{x ∈ dom f} ( ⟨y₀,x⟩ − f(x) + λ⟨y,x⟩ )
              ≤ sup_{x ∈ dom f} ( ⟨y₀,x⟩ − f(x) ) + λ c
              = f*(y₀) + λ c
  ```

so

  ```
  f**(x₀) ≥ ⟨y₀ + λy, x₀⟩ − f*(y₀ + λy)
          ≥ ⟨y₀,x₀⟩ − f*(y₀) + λ ( ⟨y,x₀⟩ − c )
          → +∞                                   as λ → ∞
  ```

since `⟨y,x₀⟩ − c > 0`. This again contradicts `f**(x₀) < r`.

Hence no `x₀` has `f**(x₀) < f(x₀)`, and with the first step,

  ```
  f** = f
  ```

The hypotheses are needed. For any proper `f` with an affine minorant,
`f**` is the largest closed convex function below `f`, so `f** = f` exactly
when `f` is closed and convex. For example, `f = 1_{x ≠ 0}` on `ℝ` has
`f** = 0`.

Regularity: the proof needs `f` proper (so `epi f ≠ ∅` and `f* > −∞`),
`f` closed (so `epi f` is closed and strict separation applies), and `f`
convex (so `epi f` is convex). It works in `ℝⁿ`; in a Hilbert space the same
proof holds verbatim.
