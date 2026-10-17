# MLS-Bench `Sci` area: TODO

TODO: port the 10 in-scope tasks below as native priml
baselines, one subpackage each, the way [`../dl/`](../dl/) does: the
task's editable function becomes a config slot, `exp000` is the pristine
recipe, each published baseline is a fork of that slot, and
`scripts/capture_reference.py` proves parity against upstream commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`. Tasks built on one upstream
package share one subpackage of common code, as
[`../dl/vision/`](../dl/vision/README.md) does for the DL area.

| Task | Upstream package | In scope |
|---|---|---|
| `ai4bio-mutation-effect-prediction` | OATML-Markslab/ProteinGym | yes |
| `ai4bio-protein-inverse-folding` | A4Bio/ProteinInvBench | yes |
| `ai4bio-protein-structure-repr` | a-r-j/ProteinWorkshop | yes |
| `ai4sci-climate-emulation` | leap-stc/ClimSim | yes |
| `ai4sci-inverse-diffusion-algo` | devzhk/InverseBench | yes |
| `ai4sci-mol-property-prediction` | deepmodeling/Uni-Mol | yes |
| `ai4sci-pla-binding-affinity` | guaguabujianle/EHIGN_PLA | yes |
| `ai4sci-vs-contrastive-scoring` | jianhuiwemi/HypSeek | yes |
| `ai4sci-weather-forecast-aggregation` | microsoft/ClimaX | yes |
| `pde-design-solver` | thuml/Neural-Solver-Library | yes |
