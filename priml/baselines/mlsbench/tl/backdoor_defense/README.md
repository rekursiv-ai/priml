# MLS-Bench `security-backdoor-defense`, as priml experiments

The task's editable `BackdoorDefense` -- a scorer fit on the victim's
penultimate features that flags poisoned training images -- becomes the slot
`defense` of `BackdoorTrainLoop`. Pinned to MLS-Bench commit
`b873944969c53b01e4ee6e843c77767b8c702485`.

```
exp000  max softmax confidence                     resnet20-cifar10-badnets, 5%
  +-- exp001  confidence z-score per predicted class  (confidence_filter)
  +-- exp002  spectral signature                      (spectral_signature)
  +-- exp003  activation clustering                   (activation_clustering)
  +-- exp004  per-class feature z-score               (zscore_outlier)
  +-- exp005  vgg16bn-cifar100-blend, 1%
  +-- exp006  mobilenetv2-fmnist-badnets, 8% (hidden upstream)
```

One run: train the victim on the poisoned split, take its features over the
split in order through the training transform, remove the top
`1.5 * poison_fraction` scores, retrain a fresh network on the rest, report
`clean_acc`, `asr`, `poison_recall` and `defense_score`. Both trainings are the
[`vision`](../../dl/vision/README.md) recipe for 100 epochs; the networks and
settings are its `ResNet`, `VGG16BN` and `MobileNetV2`.

| File | Holds |
|---|---|
| `triggers.py` | `badnets`, `blend` |
| `poison.py` | `PoisonedData`: positions, relabeling, trigger after augmentation, the analysis pass |
| `defenses.py` | The five scorers |
| `harness.py` | Features, the filter, clean accuracy, ASR |
| `train_loop.py` | `BackdoorTrainLoop` |
| `experiments.py` | `exp000` to `exp006` |
| `reference.py`, `upstream.py` | Port-side and driver-side runs of the capture |
| `scripts/capture_reference.py` | The bit-for-bit proof |

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data --dataset cifar10
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.tl.backdoor_defense.experiments.exp000
```

## Differences from the driver

- The driver evaluates the test split after epoch 1, every tenth and the last;
  so does the port. Those evaluations are diagnostic: no reported number reads
  them.
- `TrainLoop` salts its seed, so whole-run draws differ from
  `torch.manual_seed(42)`; the retrained network restarts from the seed rather
  than continuing the victim's generator. Poisoned positions use a Python
  generator seeded 42, the driver's.
- The driver's `select_poison_indices` reads every image through the training
  transform to get its label, burning three torch draws per image before the
  network is built; the port reads labels directly.

## Proof against upstream

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/tl/backdoor_defense/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/tl/backdoor_defense/testdata
```

It imports the driver once per defense from the clone and compares with exact
equality: poisoned positions and Python's generator after them; both triggers'
pixels; every defense's scores on fixed features; `main` run whole at each
setting with training skipped (poisoned images, evaluation epochs, rates,
optimizer, generators) and, for every defense, its scores, kept images and
recall on `main`'s own features; each setting's shallowest network built,
featurized and measured; each full network's state and forward; every image
of the poisoned and analysis loaders; and every parameter after two updates.
`reference_test.py` replays the stored values in the default tier.
