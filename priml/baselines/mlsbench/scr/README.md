# MLS-Bench `SCR` area: TODO

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
| `causal-discovery-discrete` | py-why/causal-learn | yes |
| `causal-observational-linear-gaussian` | py-why/causal-learn | yes |
| `causal-observational-linear-non-gaussian` | py-why/causal-learn | yes |
| `causal-observational-nonlinear` | py-why/causal-learn | yes |
| `causal-treatment-effect` | custom | yes |
| `graph-generation` | pyg-team/pytorch_geometric | yes |
| `graph-graph-classification` | pyg-team/pytorch_geometric | yes |
| `graph-link-prediction` | custom | yes |
| `graph-node-classification` | pyg-team/pytorch_geometric | yes |
| `graph-signal-propagation` | ivam-he/ChebNetII | yes |
