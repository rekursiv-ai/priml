# MLS-Bench `optimization-gradient-compression`, as priml experiments

The task's editable `Compressor` -- applied to every parameter's gradient
between the backward pass and the optimizer step -- becomes the slot
`step.compressor`. Pinned to MLS-Bench commit
`b873944969c53b01e4ee6e843c77767b8c702485`.

```
exp000  ResNet-20, CIFAR-10, no compression        (resnet20-cifar10)
  +-- exp001  top-1% with error feedback           (topk_ef)
  +-- exp002  QSGD, 256 levels, clipped to norm 1  (qsgd)
  +-- exp003  scaled sign with error feedback      (signsgd)
  +-- exp004  VGG-11, CIFAR-100, rate 0.05         (vgg11-cifar100)
  +-- exp005  ResNet-56, CIFAR-10                  (resnet56-cifar10, hidden)
```

## What is reused from `dl/vision`

The script is not the `cv-classification-loss` one, so only pieces whose
arithmetic and draws are identical are imported:

| Piece | Reused | Why |
|---|---|---|
| ResNet-20/56 | `vision.model.ResNet`, initializers set to `keep_default` | Same layers and order; the script keeps torch's default init |
| VGG-11 | no: local `model.VGG` | Vision's VGG-16 has a two-layer classifier |
| Loader | `vision.data.Cifar100Data`, 2 streams, test batch 256 | Same transforms; `num_workers=2` |
| Step | `VisionTrainStep` subclassed as `CompressedTrainStep` | Compression sits between backward and step |
| Schedule | no: local `warmup_cosine` | Linear warmup then cosine to 1% |
| Loop | `train.epoch_loop.EpochTrainLoop` with `vision.train_loop.VisionLoop`, narrowed as `CompressionTrainLoop` | Evaluates after epochs picked by the `eval_epochs` slot |
| Test loss | no: local `MeanCrossEntropy` | Summed CE over the image count, a scored column |

| File | Holds |
|---|---|
| `compressors.py` | `identity`, `top_k`, `scaled_sign`, `qsgd`, `ErrorFeedback` |
| `model.py` | `VGG` (VGG-11-BN) |
| `metrics.py` | `MeanCrossEntropy` |
| `train_step.py` | `CompressedTrainStep`, `warmup_cosine` |
| `train_loop.py` | `CompressionTrainLoop`, vision's `VisionLoop` on the compressing step |
| `experiments.py` | `exp000` to `exp005` |
| `reference.py`, `upstream.py` | Port-side and script-side runs of the capture |
| `scripts/capture_reference.py` | The bit-for-bit proof |

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.activation.scripts.prepare_data
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.opt.gradient_compression.experiments.exp000
```

## Proof against upstream

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/opt/gradient_compression/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/opt/gradient_compression/testdata
```

It imports the pristine script and each baseline's edit from the clone and
compares with exact equality: every compressor on three rounds of fixed
gradients (clipped, unclipped and zero), with the error-feedback residuals
and QSGD's draws; each setting's full network state, forward and generator
position; the 200 rates and optimizer of each setting; the evaluated epochs
and reported accuracy; both test metrics; the test pipelines; every pixel of
both datasets' loaders; and every parameter after three updates of the
script's own `train` per compressor, through warmup and into the cosine.
`reference_test.py` replays the stored values in the default tier.
