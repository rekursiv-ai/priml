# MLS-Bench `Opt` area: TODO

1 of 13 ported (linked). TODO: port the other 12 in-scope tasks below as native priml
baselines, one subpackage each, the way [`../dl/`](../dl/) does: the
task's editable function becomes a config slot, `exp000` is the pristine
recipe, each published baseline is a fork of that slot, and
`scripts/capture_reference.py` proves parity against upstream commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`. Tasks built on one upstream
package share one subpackage of common code, as
[`../dl/vision/`](../dl/vision/README.md) does for the DL area.

| Task | Upstream package | In scope |
|---|---|---|
| `optimization-bilevel` | hanshen95/penalized-bilevel-gradient-descent | yes |
| `optimization-convex-concave` | TrueNobility303/RAIN | yes |
| `optimization-diagonal-net` | TrueNobility303/RAIN | yes |
| `optimization-dp-sgd` | custom | yes |
| `optimization-evolution-strategy` | DEAP/deap | yes |
| `optimization-gradient-compression` | custom | yes; ported in [`gradient_compression/`](gradient_compression/README.md), slot `step.compressor` |
| `optimization-hyperparameter-search` | custom | yes |
| `optimization-multi-objective` | DEAP/deap | yes |
| `optimization-nas` | automl/naslib | yes |
| `optimization-online-bandit` | SMPyBandits/SMPyBandits | yes |
| `optimization-pac-bayes-bound` | mperezortiz/PBB | yes |
| `optimization-parity` | pytorch/examples | yes |
| `optimization-variance-reduction` | custom | yes |
