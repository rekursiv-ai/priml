# MLS-Bench

The baselines of [MLS-Bench](https://github.com/Imbernoulli/MLS-Bench), 140
machine-learning research tasks, reproduced as priml experiments. A task asks
for one edit to a small codebase: a sampler, a loss, an optimizer. It scores
the edit by running the task's own scripts in a pinned container, once per
setting, and the published leaderboard gives the score of each baseline
edit. An experiment here is one leaderboard row: a task, a baseline and a
seed. It applies that baseline's edit, runs every setting and records the
numbers under the leaderboard's column names, so they can be checked against
the published row.

Upstream code is never copied into priml. Tasks are read from a clone of
MLS-Bench at the pinned commit `80cf5c5c` when the run starts.

## Run it

```bash
git clone https://github.com/Imbernoulli/MLS-Bench.git /opt/scratch/mlsbench
git -C /opt/scratch/mlsbench checkout 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.tasks.llm_dllm_demask_strategy.exp002
```

The clone is about 1.4 GB and needs no GPU. A different commit can change
tasks under the registry, the goldens and the eval-script pins, so stay on the
pinned one. Data a task ships in `tasks/<task>/data` needs no staging: each
setting's container mounts the task directory read-only. The third command
runs one row, here `llm-dllm-demask-strategy` with baseline `klass`.

Every setting runs in its upstream image, `bohanlyu2022/mlsbench-harbor-<package>`,
pinned by digest in `testdata/image_lock.json`. Docker must be running. The first run of a
task pulls its image before any setting's time budget starts, and all of them together need about 80 GB of disk. The
images are linux/amd64 only. GPU needs are declared upstream (`compute` per
setting in the task's `config.json`). The settings of a task run one after another.

`--override PATH=VALUE` adapts a run to the machine it lands on:

```bash
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.tasks.llm_dllm_demask_strategy.exp002 --override base_dir=/data/me --override mlsbench_root=/clones/mls-bench
```

Path fields are *logical*. `mlsbench_root` (the clone, `/mlsbench`) and
`data_root` (dataset binds, `/datasets/mlsbench`) resolve beneath the run's
`base_dir` (`/opt/scratch` by default). Pass `/clones/mls-bench`, not the full on-disk
path. To move the whole tree at once, override `base_dir`. Task, baseline and
seed are not overridable: each row is its own experiment.

To check an installation without the clone, Docker or a GPU, run `exp_smoke`.
It is a synthetic three-setting task with a fake executor that returns canned
logs:

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.experiments.exp_smoke
```

Runs land in `<base_dir>/runs/mlsbench/<task>/<baseline>/seed<SEED>/metrics.json`.
The verdict against the published row:

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.leaderboard --task llm-dllm-demask-strategy --row klass --runs /opt/scratch
uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.leaderboard --all --runs /opt/scratch --include-missing
```

Each row gets a verdict: `ok`, `diverged` or `unverified`, the worst over the
row's published cells. A published cell that is missing or was never measured
is never `ok`. A measured column the row does not publish (a hidden split's
metric) is listed but does not affect the verdict. With `--all`, the exit code
is 1 if any row diverged, and `--include-missing` counts clone tasks that have
no run as `unverified`.

## Experiments

There are no hand-written per-task experiments. `registry.json` is a committed
snapshot of the clone, covering every task's baselines, settings, seeds and
hidden settings. `scripts/generate_experiments.py` turns it into one module per
task, `tasks/<task_module>.py`, where the module is the task name with `-`
replaced by `_`. Each module holds one `expNNN()` per (baseline, seed), numbered
in registry baseline order and then by seed. Numbering is local to the task.
Each docstring carries a **Hypothesis** (the published row should be reproduced),
**References** (the upstream leaderboard row and pin) and **Results** (`TBD`
until the row has run on reference hardware). `INDEX` maps `"expNNN"` to
`(task, baseline, seed)`.

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.generate_experiments list --task llm-dllm-demask-strategy
```

| Experiment | Task | Baseline | Seed |
|---|---|---|---|
| `tasks.llm_dllm_demask_strategy.exp000` | `llm-dllm-demask-strategy` | `topk_margin` | 42 |
| `tasks.llm_dllm_demask_strategy.exp001` | `llm-dllm-demask-strategy` | `confidence_greedy` | 42 |
| `tasks.llm_dllm_demask_strategy.exp002` | `llm-dllm-demask-strategy` | `klass` | 42 |
| `experiments.exp_smoke` | synthetic | `base_a` | 42 |

There are 563 generated experiments over 139 tasks. `cv-classification-loss`
is ported natively instead, in [`pytorch_vision/`](pytorch_vision/README.md).
Most tasks publish seed 42 only, and 13 publish several. A seed the task does
not publish is refused. Every setting runs, hidden ones included.

A new edit of a task is an experiment of its own. Write the edit script (the
format of the task's `edits/`) beneath `base_dir`, then add a function that
returns `candidate_experiment`:

```python
def exp_my_idea() -> MlsbenchEval:
    """Hypothesis, References, Results, as for any experiment."""
    return candidate_experiment(
        "llm-dllm-demask-strategy", "my-idea", "/edits/my-idea.edit.py"
    )
```

It runs every setting as a baseline row does, guarded the same way: only the
task's editable ranges may change. It lands in
`<base_dir>/runs/mlsbench/<task>/my-idea/seed42/`. `--candidate` sets its
numbers against a published row. There, `diverged` only means the numbers
differ from the row. Read the deltas against the task's metric directions.

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.leaderboard --task llm-dllm-demask-strategy --row klass --candidate my-idea --runs /opt/scratch
```

After an upstream bump, re-pin and regenerate:

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.generate_experiments --registry
uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.generate_experiments --write
uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.pin_eval_scripts
uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.pin_images --list
uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.pin_images --package <package> --repin
```

## Layout

| File | Contents |
|---|---|
| `task.py` | Parse a task from the clone: settings, baselines, leaderboard |
| `data.py` | One `metric_only` batch per setting |
| `train_step.py` | `NullStep`: the score comes from the metric, not a model |
| `metric.py` | `MlsbenchSuite`: materialize, run each setting, record its numbers |
| `setting.py` | A setting's env, data mounts and columns; the upstream parser |
| `executor.py` | `LocalDocker` (the real run) and `FakeExecutor` (tests, `exp_smoke`) |
| `workspace.py` | Apply an edit and guard the edit surface |
| `harbor.py` | Read Harbor bundles; guard eval scripts against their copy and the pins |
| `experiments.py` | `MlsbenchEval`, `mlsbench_experiment(task, baseline, seed=)`, `exp_smoke` |
| `tasks/` | The generated per-task modules; do not edit |
| `registry.json` | Snapshot of the clone the generator reads |
| `eval_script_pins.json` | Hashes of every task's eval scripts |
| `harbor_adapted.json` | Tasks whose eval scripts Harbor ships patched |
| `testing.py` | Synthetic MLS-Bench-shaped tasks |
| `scripts/prepare_data.py` | Build the `exp_smoke` clone |
| `scripts/leaderboard.py` | Verdicts against the published rows; whole-benchmark aggregate |
| `scripts/pin_eval_scripts.py` | Hashes in `eval_script_pins.json` |
| `scripts/generate_experiments.py` | Registry and per-task modules |
| `scripts/pin_images.py` | Image digests in `testdata/image_lock.json` |

The run is a `TrainLoop` with `eval_only=True`, and the metric does the work, as in
the craftax baseline. `SettingBatches` names one setting per batch.
`MlsbenchSuite.update` then materializes a workspace with the baseline's edit
applied and checks that only the declared lines changed and that the eval
scripts match their pins. It runs the setting's script through the injected
executor, parses the log with the task's own parser and records the
leaderboard's columns. A timeout, crash or missing input leaves the column
absent, never NaN.

Each setting gets the package environment from the task's Harbor bundle. The
seed, the run directories and the data binds always win. Each setting writes
`logs/<label>.request.json`, which records the seed, environment and binds,
including any skipped binds. Held-out inputs are generated in a throwaway
container with no network. A failure there is `missing_input`, never a silent
run. `LocalDocker`'s network follows the bundle's `allow_internet` setting. To
force isolation, write a fork with `LocalDocker.Config(network="none")`.

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/mlsbench
```

These run on CPU in about 20 seconds with no clone, Docker or network.
Each task's `exp000` has a pprint golden (`testdata/tasks/<task>/exp000.txt`);
its other baselines and seeds derive from it. `testdata/exp_smoke.txt` freezes
the metrics the smoke run writes, exactly.
Regenerate the goldens only when a config change is intended:

```bash
uv --quiet run --frozen pytest priml/baselines/mlsbench --regenerate-golden
```

The opt-in tier checks the harness against a real clone. It parses,
materializes and guards all 140 tasks, and checks the registry against the
clone. The container tests run `ml-symbolic-regression` in its real image:

```bash
MLSBENCH_ROOT=/opt/scratch/mlsbench uv --quiet run --frozen pytest priml/baselines/mlsbench -m ''
```

Not yet verified: reproduction of any published row. Only `gplearn`
(`ml-symbolic-regression`) has run in its real image, and no GPU run exists.
