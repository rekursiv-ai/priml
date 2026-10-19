# MLS-Bench `TL` area: TODO

4 of 8 ported (linked). TODO: port the other 4 in-scope tasks below as native priml
baselines, one subpackage each, the way [`../dl/`](../dl/) does: the
task's editable function becomes a config slot, `exp000` is the pristine
recipe, each published baseline is a fork of that slot, and
`scripts/capture_reference.py` proves parity against upstream commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`. Tasks built on one upstream
package share one subpackage of common code, as
[`../dl/vision/`](../dl/vision/README.md) does for the DL area.

| Task | Upstream package | In scope |
|---|---|---|
| `security-adversarial-attack-black-box-score` | Harry24k/adversarial-attacks-pytorch | yes |
| `security-adversarial-attack-sparse-l0` | Harry24k/adversarial-attacks-pytorch | yes |
| `security-adversarial-attack-white-box-linf` | Harry24k/adversarial-attacks-pytorch | yes |
| `security-adversarial-training` | Harry24k/adversarial-attacks-pytorch | yes |
| [`security-backdoor-defense`](backdoor_defense/README.md) | custom | yes; `defense` |
| [`security-machine-unlearning`](machine_unlearning/README.md) | custom | yes; `method` |
| [`security-membership-inference-defense`](membership_inference_defense/README.md) | custom | yes; `step.loss` |
| [`security-poison-robust-learning`](poison_robust_learning/README.md) | custom | yes; `step.loss` |
