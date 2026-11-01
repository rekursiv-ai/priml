# MLS-Bench `TS` area: TODO

TODO: port the 10 in-scope tasks below as native priml
baselines, one subpackage each, the way [`../dl/`](../dl/) does: the
task's editable function becomes a config slot, `exp000` is the pristine
recipe, each published baseline is a fork of that slot, and
`scripts/capture_reference.py` proves parity against upstream commit
`b873944969c53b01e4ee6e843c77767b8c702485`. Tasks built on one upstream
package share one subpackage of common code, as
[`../dl/vision/`](../dl/vision/README.md) does for the DL area.

| Task | Upstream package | In scope |
|---|---|---|
| `quant-concept-drift` | microsoft/qlib | yes |
| `quant-graph-stock` | microsoft/qlib | yes |
| `quant-stock-prediction` | microsoft/qlib | yes |
| `stf-traffic-forecast` | GestaltCogTeam/BasicTS | yes |
| `ts-anomaly-detection` | thuml/Time-Series-Library | yes |
| `ts-classification` | thuml/Time-Series-Library | yes |
| `ts-exogenous-forecast` | thuml/Time-Series-Library | yes |
| `ts-imputation` | thuml/Time-Series-Library | yes |
| `ts-long-term-forecast` | thuml/Time-Series-Library | yes |
| `ts-short-term-forecast` | thuml/Time-Series-Library | yes |
