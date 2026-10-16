# What Makes Good Math

Good results turn something you can't compute into something you can.
Almost every proof here does it in two moves: carve the problem into easy
pieces, then bound what's left.

## Carve

Representation: change coordinates until the problem falls apart.
Eigenvectors turn matrix powers into scalar powers. Fourier turns sums of
random variables into products. Logs turn products into sums. Duality
describes a convex function by its tangents. Projection splits a vector into
a fitted part and an orthogonal residual. Symmetry tells you which
coordinates to pick: anything that commutes with a group is diagonal in that
group's basis.

Recursion: solve one step, then repeat. The chain rule gives backprop. The
principle of optimality gives Bellman, LQR and Kalman. Telescoping a
one-step inequality gives convergence rates. Recursion is legal when the
present summarizes the past, and cheap because steps share subproblems.

## Bound

Convexity: the average of a convex function is at least the function of the
average. Jensen gives Gibbs, the ELBO, Hölder and Rao–Blackwell.

Uniformity: bound one point, then count. Concentration plus a union bound
over a net gives PAC, Johnson–Lindenstrauss, VC and DKW.

Indistinguishability: if two worlds look alike in the data, no method can
tell them apart. This gives floors: Le Cam, Fano, no free lunch. Upper
bounds say a method works; floors say when to stop improving it.

## The Pattern

A typical proof uses one carve and one bound:

- Cramér–Rao projects the estimator onto the score; the residual's
  nonnegative variance gives the bound.
- Gradient descent recurses one step at a time; convexity bounds each step.
- Johnson–Lindenstrauss uses rotation invariance to reduce to one fixed
  vector, then a union bound over pairs.

The results worth recording are the ones where the carve is exact and the
bound is tight: an equality case shows nothing was lost.
