# TRM (Tiny Recursive Model)

Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks", arXiv:2510.04871.

## Notation

```
x                                  ∈ ℝᶜ        # per cell: embedded puzzle (9×9 or 30×30)
y                                  ∈ ℝᶜ        # per cell: current answer, embedded
z                                  ∈ ℝᶜ        # per cell: latent reasoning
f                                              # one 2-layer net: attention or token-MLP, SwiGLU
O                                  ∈ ℝᵛˣᶜ      # output head; v = vocab
Q                                  ∈ ℝ¹ˣᶜ      # halting head
```

## One recursion

```
z     ← f(x + y + z)                           # repeat n = 6 times; sees the question
y     ← f(y + z)                               # update answer; no x
```

## One supervision step

```
T − 1 = 2 recursions, no gradient
1 recursion with gradient                      # backprop through all n + 1 calls
o     = O y                                    # logits per cell
q     = σ(Q y)                                 # P(answer is correct)
loss  = CE(o, target) + BCE(q, argmax o = target)
```

- deep supervision: detach `(y, z)`, repeat up to 16 steps.
- training may stop early on `q` (ACT); test runs all 16.
- depth per step: 2 layers × (n + 1) × T = 42.
- EMA of weights, 0.999.

HRM, but (Sudoku-Extreme ablations, final = 87.4%):

1. one net, not two (82.4).
2. backprop through the full recursion, not a 1-step gradient (56.5).
3. 2 layers, not 4 (79.5).
4. halting by BCE, no second forward pass (86.1).
5. weight EMA (79.9 without).
6. MLP over cells instead of attention on small grids (74.7 with attention).

## Results

| task      | TRM       | HRM (27M) |
|-----------|-----------|-----------|
| Sudoku    | 87.4 (5M) | 55.0      |
| Maze-Hard | 85.3 (7M) | 74.5      |
| ARC-AGI-1 | 44.6 (7M) | 40.3      |
| ARC-AGI-2 | 7.8 (7M)  | 5.0       |

An RNN over thinking steps, not time: one `f` reused, state `(y, z)`,
inputs merged by `+`, no gate (compare [mingru.md](mingru.md)).
