# MLS-Bench `RL` area: TODO

TODO: port the 13 in-scope tasks below as native priml
baselines, one subpackage each, the way [`../dl/`](../dl/) does: the
task's editable function becomes a config slot, `exp000` is the pristine
recipe, each published baseline is a fork of that slot, and
`scripts/capture_reference.py` proves parity against upstream commit
`b873944969c53b01e4ee6e843c77767b8c702485`. Tasks built on one upstream
package share one subpackage of common code, as
[`../dl/vision/`](../dl/vision/README.md) does for the DL area.

| Task | Upstream package | In scope |
|---|---|---|
| `marl-centralized-critic` | uoe-agents/epymarl | yes |
| `meta-rl` | katerakelly/oyster | yes |
| `meta-rl-algorithm` | katerakelly/oyster | yes |
| `rl-intrinsic-exploration` | vwxyzjn/cleanrl | yes |
| `rl-offline-adroit` | corl-team/CORL | yes |
| `rl-offline-continuous` | corl-team/CORL | yes |
| `rl-offline-off2on` | corl-team/CORL | yes |
| `rl-offpolicy-continuous` | vwxyzjn/cleanrl | yes |
| `rl-onpolicy-continuous` | vwxyzjn/cleanrl | yes |
| `rl-reward-learning` | HumanCompatibleAI/imitation | yes |
| `rl-value-atari` | vwxyzjn/cleanrl | yes |
| `rl-value-discrete` | vwxyzjn/cleanrl | yes |
| `safe-rl` | PKU-Alignment/omnisafe | yes |
