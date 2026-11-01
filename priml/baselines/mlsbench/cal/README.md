# MLS-Bench `CAL` area: TODO

TODO: port the 14 in-scope tasks below as native priml
baselines, one subpackage each, the way [`../dl/`](../dl/) does: the
task's editable function becomes a config slot, `exp000` is the pristine
recipe, each published baseline is a fork of that slot, and
`scripts/capture_reference.py` proves parity against upstream commit
`b873944969c53b01e4ee6e843c77767b8c702485`. Tasks built on one upstream
package share one subpackage of common code, as
[`../dl/vision/`](../dl/vision/README.md) does for the DL area.

| Task | Upstream package | In scope |
|---|---|---|
| `meta-fewshot-classification` | sicara/easy-few-shot-learning | yes |
| `meta-inner-loop-optimizer` | learnables/learn2learn | yes |
| `ml-active-learning` | JordanAsh/badge | yes |
| `ml-anomaly-detection` | custom | yes |
| `ml-calibration` | custom | yes |
| `ml-clustering-algorithm` | custom | yes |
| `ml-continual-regularization` | GMvandeVen/continual-learning | yes |
| `ml-dimensionality-reduction` | custom | yes |
| `ml-ensemble-boosting` | custom | yes |
| `ml-federated-aggregation` | adap/flower | yes |
| `ml-missing-data-imputation` | custom | yes |
| `ml-selective-deferral` | custom | yes |
| `ml-subgroup-calibration-shift` | custom | yes |
| `ml-symbolic-regression` | trevorstephens/gplearn | yes |
