# Conformal Coverage

Conformal prediction covers the truth with probability `1 − α` for any model
and any distribution, from one fact: under exchangeability, the test point's
rank among the scores is uniform.

Let,

  ```
  (X₁,Y₁), …, (Xₙ,Yₙ), (Xₙ₊₁,Yₙ₊₁)   exchangeable
  S(x,y)                             any score fixed before seeing the data
  Sᵢ   ≝ S(Xᵢ,Yᵢ),  i = 1, …, n+1
  k    ≝ ⌈(1 − α)(n + 1)⌉
  q̂    ≝ k-th smallest of S₁, …, Sₙ
  C(x) ≝ { y : S(x,y) ≤ q̂ }
  R    ≝ rank of Sₙ₊₁ among S₁, …, Sₙ₊₁
  ```

Assume no ties among the `Sᵢ` and `k ≤ n`.

First, the rank is uniform. Swapping `Sᵢ` and `Sₙ₊₁` does not change the
joint law, so every index is equally likely to hold rank `j`. Exactly one
index holds it,

  ```
  1 = Σᵢ P(Sᵢ has rank j)
    = (n + 1) ⋅ P(Sₙ₊₁ has rank j)
  ```

so `P(R = j) = 1/(n + 1)` for each `j = 1, …, n+1`.

Second, coverage is a rank event. `Sₙ₊₁ ≤ q̂` iff at most `k − 1`
calibration scores lie below `Sₙ₊₁`, i.e., iff `R ≤ k`,

  ```
  P(Yₙ₊₁ ∈ C(Xₙ₊₁)) = P(Sₙ₊₁ ≤ q̂)
                    = P(R ≤ k)
                    = k / (n + 1)
  ```

Third, bound `k` from both sides using `x ≤ ⌈x⌉ < x + 1`,

  ```
  1 − α ≤ k / (n + 1) < 1 − α + 1/(n + 1)
  ```

Hence,

  ```
  1 − α ≤ P(Yₙ₊₁ ∈ C(Xₙ₊₁)) < 1 − α + 1/(n + 1)
  ```

The coverage is exact up to `1/(n + 1)`. Nothing about the model, the score,
or the distribution entered: only the invariance of the joint law under
permutation. A bad score costs width of `C`, never coverage.

Fine print: with ties, break them uniformly at random; the lower bound still
holds and the upper bound may fail. If `k > n`, set `q̂ ≝ ∞`, giving trivial
coverage `1`. The guarantee is marginal, averaged over the calibration draw
and the test point; it does not hold conditionally on `Xₙ₊₁`, and exact
conditional coverage is impossible in general (Vovk 2012; Barber et al.
2021). Full conformal refits the score on all `n + 1` points for each
candidate `y`; the same rank argument applies because the fit is symmetric
in its inputs.

Regularity: the proof needs the `n + 1` pairs exchangeable (iid suffices)
and the score `S` chosen independently of the calibration and test data,
e.g., trained on a separate split.
