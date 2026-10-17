# MLS-Bench `Rob` area: TODO

TODO: port the 12 in-scope tasks below as native priml
baselines, one subpackage each, the way [`../dl/`](../dl/) does: the
task's editable function becomes a config slot, `exp000` is the pristine
recipe, each published baseline is a fork of that slot, and
`scripts/capture_reference.py` proves parity against upstream commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`. Tasks built on one upstream
package share one subpackage of common code, as
[`../dl/vision/`](../dl/vision/README.md) does for the DL area.

| Task | Upstream package | In scope |
|---|---|---|
| `jepa-planning` | facebookresearch/eb_jepa | yes |
| `jepa-prediction-loss` | facebookresearch/eb_jepa | yes |
| `jepa-regularizer` | facebookresearch/eb_jepa | yes |
| `robo-diffusion-guidance` | CleanDiffuserTeam/CleanDiffuser | yes |
| `robo-diffusion-policy` | CleanDiffuserTeam/CleanDiffuser | yes |
| `robo-diffusion-sampling-method` | CleanDiffuserTeam/CleanDiffuser | yes |
| `robo-humanoid-sim2real-algo` | roboterax/humanoid-gym | yes |
| `robomimic-bc-loss` | ARISE-Initiative/robomimic | yes |
| `robomimic-iql-vf` | ARISE-Initiative/robomimic | yes |
| `robomimic-obs-encoder` | ARISE-Initiative/robomimic | yes |
| `tdmpc2-planning` | nicklashansen/tdmpc2 | yes |
| `tdmpc2-simnorm` | nicklashansen/tdmpc2 | yes |
