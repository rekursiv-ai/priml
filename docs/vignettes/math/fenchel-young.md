# Fenchel–Young

Fenchel–Young follows from the definition of `f*` as a supremum, and its
equality case is the subgradient inequality read backwards.

Let,

  ```
  f : ℝⁿ → ℝ ∪ {+∞}            proper, i.e., f > −∞ and f(x) < ∞ for some x
  f*(y) ≝ sup_z ( ⟨z,y⟩ − f(z) )
  ∂f(x) ≝ { y : f(z) ≥ f(x) + ⟨y, z − x⟩  for all z }    (x with f(x) < ∞)
  ```

Note `f*(y) > −∞`, since some `z` has `f(z) < ∞`.

First, the inequality. If `f(x) = +∞` it is trivial. Otherwise take `z = x`
in the supremum,

  ```
  f*(y) = sup_z ( ⟨z,y⟩ − f(z) )
        ≥ ⟨x,y⟩ − f(x)
  ```

so `f(x) + f*(y) ≥ ⟨x,y⟩` for all `x, y`.

Second, the equality case. For `f(x) < ∞`, each line below is equivalent to
the next,

  ```
  y ∈ ∂f(x)
  ⇔ f(z)         ≥ f(x) + ⟨y, z − x⟩        for all z
  ⇔ ⟨x,y⟩ − f(x) ≥ ⟨z,y⟩ − f(z)             for all z
  ⇔ ⟨x,y⟩ − f(x) ≥ sup_z ( ⟨z,y⟩ − f(z) )
  ⇔ ⟨x,y⟩ − f(x) ≥ f*(y)
  ⇔ f(x) + f*(y) = ⟨x,y⟩
  ```

where the second step rearranges terms, and the last step combines `≤` with
the first step's `≥`.

Hence,

  ```
  f(x) + f*(y) ≥ ⟨x,y⟩,   with equality iff y ∈ ∂f(x)
  ```

Equality holds exactly on the graph of `∂f`. In words, `x` attains the
supremum defining `f*(y)`.

For example, let `f(x) = ½ xᵀAx` with `A ≻ 0`. The supremum in `f*` is a
concave quadratic in `z`, maximized where `y − Az = 0`, i.e., `z = A⁻¹y`,

  ```
  f*(y) = ⟨A⁻¹y, y⟩ − ½ (A⁻¹y)ᵀ A (A⁻¹y)
        = yᵀA⁻¹y − ½ yᵀA⁻¹y
        = ½ yᵀA⁻¹y
  ```

The gap is a completed square,

  ```
  f(x) + f*(y) − ⟨x,y⟩ = ½ xᵀAx − xᵀy + ½ yᵀA⁻¹y
                       = ½ (x − A⁻¹y)ᵀ A (x − A⁻¹y)
                       ≥ 0
  ```

with equality iff `x = A⁻¹y`, i.e., `y = Ax = ∇f(x)`, matching
`∂f(x) = {Ax}`. With `A = I` this is `½‖x‖² + ½‖y‖² ≥ ⟨x,y⟩`.

Regularity: the proof needs only `f` proper. Neither convexity nor
closedness of `f` is used.
