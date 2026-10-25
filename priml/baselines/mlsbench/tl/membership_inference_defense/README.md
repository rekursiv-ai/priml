# MLS-Bench `security-membership-inference-defense`, as priml experiments

The task's editable `MembershipDefense.compute_loss` -- the training loss --
becomes the slot `step.loss`. The training split is shuffled with Python's
generator seeded 42 and halved: the network trains on the members, and after
300 epochs a confidence-threshold attack scores how well its largest softmax
probability separates members from non-members. Pinned to MLS-Bench commit
`b873944969c53b01e4ee6e843c77767b8c702485`.

```
exp000  resnet20-cifar10, cross-entropy
  +-- exp001  erm                     (the same cross-entropy)
  +-- exp002  label smoothing 0.1     (label_smoothing)
  +-- exp003  confidence penalty 0.1  (confidence_penalty)
  +-- exp004  RelaxLoss, alpha 1      (relaxloss)
  +-- exp005  vgg16bn-cifar100        (exp006-exp009: the four, alpha 0.5)
  +-- exp010  mobilenetv2-fmnist      (exp011-exp014: the four, alpha 1)
```

Upstream picks RelaxLoss's level from the class count at call time; here it is
the `alpha` field of each setting's fork.

| File | Holds |
|---|---|
| `data.py` | `MembershipData`: the member split, and evaluation over test, members, non-members |
| `defenses.py` | `confidence_penalty`, `relaxloss` |
| `metrics.py` | `MembershipInference`: accuracy, AUC, gap, privacy score |
| `schedules.py` | `multistep`, the step decay at epochs 150 and 225 |
| `experiments.py` | `exp000` to `exp014` |
| `reference.py`, `upstream.py` | Port-side and driver-side runs of the capture |
| `scripts/capture_reference.py` | The bit-for-bit proof |

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.activation.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.tl.membership_inference_defense.experiments.exp000
```

## Proof against upstream

```bash
/opt/scratch/scripts/capture.sh priml/baselines/mlsbench/tl/membership_inference_defense
```

It imports the driver with its pristine defense and with each baseline's edit
and compares with exact equality: every defense and its gradient on fixed
logits in both RelaxLoss phases, the attack on fixed logits with tied scores,
the training pipelines with and without `--augment` and the test pipeline, the
300 rates and the optimizer; per setting the member split of 50,000 images
and the full network's initial state, generator position and forward
(ResNet-20, VGG-16-BN, and MobileNetV2 with this driver's ReLU); and per
defense a whole run of the driver's `main` on the shallowest ResNet: every
loss, logit, label, epoch and rate, every parameter after each update, and the
attack's scores and metrics. `reference_test.py` replays the stored values in
the default tier.

Not proven here: whole runs at the `vgg16bn-cifar100` and `mobilenetv2-fmnist`
settings, and Fashion-MNIST's data path, which the vision scaffold's capture
proves.
