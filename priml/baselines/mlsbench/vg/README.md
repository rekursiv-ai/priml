# MLS-Bench `V&G` area: TODO

TODO: port the 11 in-scope tasks below as native priml
baselines, one subpackage each, the way [`../dl/`](../dl/) does: the
task's editable function becomes a config slot, `exp000` is the pristine
recipe, each published baseline is a fork of that slot, and
`scripts/capture_reference.py` proves parity against upstream commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`. Tasks built on one upstream
package share one subpackage of common code, as
[`../dl/vision/`](../dl/vision/README.md) does for the DL area.

| Task | Upstream package | In scope |
|---|---|---|
| `cv-3dgs-densification` | nerfstudio-project/gsplat | yes |
| `cv-3dgs-regularizer` | nerfstudio-project/gsplat | yes |
| `cv-dbm-sampler` | thu-ml/DiffusionBridge | yes |
| `cv-dbm-scheduler` | thu-ml/DiffusionBridge | yes |
| `cv-diffusion-architecture` | huggingface/diffusers | yes |
| `cv-diffusion-cfg` | CFGpp-diffusion/CFGpp | yes |
| `cv-diffusion-conditioning` | huggingface/diffusers | yes |
| `cv-diffusion-efficiency` | CFGpp-diffusion/CFGpp | yes |
| `cv-diffusion-prediction` | huggingface/diffusers | yes |
| `cv-meanflow-perceptual-loss` | snap-research/alphaflow | yes |
| `cv-vae-loss` | huggingface/diffusers | yes |
