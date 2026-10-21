# Name That LLM

`exp000` rewrites the public Qwen8B/SVM/centred-head recipe in Priml.
Qwen is frozen; classifiers fit its embeddings and text features.
`exp001` adds a 300-second budget. `exp002` changes validation to two prompt
folds. Runtime and score for the rewrite are unmeasured.

[Name That LLM](https://www.kaggle.com/competitions/name-that-llm) is a $50,000
Featured competition to identify which model generated a passage. Final
submissions close 14 January 2027 at 23:59 UTC; authenticated Kaggle metadata
was checked on 16 October 2026.

## Acquire and prepare

[DATA.md](DATA.md) gives the one-time commands and describes the small test
fixtures.

Run from the repository root after Kaggle authentication and rule acceptance.
Preparation preserves CSV bytes and records input digests, row identifiers
and the sorted 18-label vocabulary. It requires all 2,699 labelled passages
and 150 prompt identifiers.

```bash
export UV_NO_SYNC=1
export WORKERS_DEFAULT=1
uv --quiet run --frozen kaggle competitions download -c name-that-llm -p /opt/scratch/datasets/name-that-llm/raw
unzip /opt/scratch/datasets/name-that-llm/raw/name-that-llm.zip -d /opt/scratch/datasets/name-that-llm/raw
uv --quiet run --frozen kaggle models instances versions download qwen-lm/qwen-3/transformers/8b-base/1 -p /opt/scratch/datasets/name-that-llm/qwen-3-8b-base-v1 --untar
uv --quiet run --frozen python -m priml priml.baselines.name_that_llm.scripts.prepare_data.prepare_data
```

## Run

Use one H100/H200 and fresh output directories. Downloads precede the budget;
tokenization, embedding extraction, classifier fitting, validation, final
fitting and output publication belong inside it.

```bash
/usr/bin/time -p uv --quiet run --frozen python -m priml priml.baselines.name_that_llm.experiments.exp000
/usr/bin/time -p uv --quiet run --frozen python -m priml priml.baselines.name_that_llm.experiments.exp001
uv --quiet run --frozen python -m priml priml.baselines.name_that_llm.experiments.exp002
uv --quiet run --frozen python -m priml priml.baselines.name_that_llm.experiments.exp_smoke
```

`exp_smoke` uses six real prompts and shorter token dimensions with the full
pretrained network and every classifier branch. Its score checks setup.

Each run writes `result.json`, held-out `validation.npz`, test
`predictions.npz`, and `submission.parquet`. The nested `budget` config uses
shared `train.budget.WallBudget`; `exp000` is uncapped and `exp001` sets only
`budget.seconds=300`. The receipt envelope records `status`, `budget_seconds`,
`elapsed_seconds`, and an `error` on failure. Its `metadata` contains config,
dependencies, accelerator, input digests and metrics. The optional literal
`budget.receipt_path` defaults to `result.json` in the resolved run directory.
Cold preparation, fitting, evaluation, export and durable receipt publication
count toward the allowance. The run holds `budget.interruptible()`, so the
deadline raises `TimeoutError` at the next Python bytecode and still seals a
failure receipt; one uninterrupted native fit finishes first. Late
publication also seals a failure receipt and raises `TimeoutError`; retain
outer process timing too.

Fork the named parent experiment and change one component. Output paths
resolve beneath `base_dir` from `study_name` and `experiment_name`; input
directories are logical paths beneath the same `base_dir`. Keep
prompt-exclusive folds, labels, balanced accuracy and the budget fixed across
candidate comparisons. Two-fold and five-fold scores use different protocols.

## Public recipe

The reference is qasminumber (Zaphyer Modal)'s
[name-that-llm-v5, script version 358565580](https://www.kaggle.com/code/qasminumber/name-that-llm-v5?scriptVersionId=358565580),
published 15 October 2026 at 05:55 UTC, kernel 138797198, version 1.
It scored 0.475 publicly and 0.513400447427293 on grouped CV; it was the
highest scored public code retrieved on 15 October 2026.

| Stage | Recipe |
| --- | --- |
| Encoder | Frozen `qwen-lm/qwen-3/transformers/8b-base/1`; fp16, batch 8, right padding, no special tokens, length 512; longest-first batches |
| Pooling | Hidden layers 4, 6, 8, 22; float32 masked mean; bf16 retry for nonfinite rows |
| Plain heads | Four dense standardizers and multinomial logistic heads; C=0.002, L-BFGS-B, 5000 updates; mean probabilities |
| Centred heads | Continuing RNG seed 42; chunk sizes 2, 3, 4, 6; per-chunk means; four standardized heads, 3000 updates; plain probabilities for singleton clusters |
| Text | Case-sensitive character TF-IDF 1..5, capped at 300000; whitespace word TF-IDF 1..2; sublinear TF and min_df=2; all 25 standardized style features weighted 0.3 |
| SVM | L2 squared-hinge dual solver, C=1, 5000 sweeps, seed 42; sigmoid calibration, three grouped inner folds, ensemble=False |
| Topics | English-stopword TF-IDF; clipped cosine distance, average linkage, cutoff 0.9; validation chunks of three, continuing RNG seed 43 |
| Blend | 0.4 log(SVM) + 0.3 log(plain) + 0.3 log(centred); clip probabilities to [1e-12, 1] |
| Evaluation | Five prompt-exclusive outer folds, balanced accuracy; final fit on every row and branch; id/model Parquet submission |

Outer folds run sequentially; the notebook uses joblib for outer SVM fits.
The native classifier implements the dual solver selected by this feature
geometry. `torch.optim.LBFGS` fits the dense heads and sigmoid calibration in
place of scikit-learn's L-BFGS-B. Native NumPy sparse rows and average linkage
replace SciPy's; average linkage keeps its merge order and tie-breaking.
Transformers supplies tokenizer I/O.

The original notebook SHA256 is
`509018093b5264c507f15333aa2c78d2baeab46efb727c1742157f3af6a6b139`;
its code-cell SHA256 is
`946c7138b4e8685fb36327c0a6c4d1ef50a21acdb62746c46c44b99a228ec1aa`.

## Numerical validation

```bash
nice -n 19 env UV_NO_SYNC=1 WORKERS_DEFAULT=1 uv --quiet run --frozen pytest -n 0 -p no:randomly --durations=0 priml/baselines/name_that_llm
```

Every `.pt` golden is minted in-tree from this package: the test that reads
it writes it under `--regenerate-b4b` and compares values rounded to float32,
each in its own dtype. Torch computations run inside `host_agnostic_numerics`;
the logistic heads' NumPy products sum in float64 and round once.

`exp000.pt` holds every fitted state and output of the three exp000 branches
on an 18-row miniature, after three updates of each solver: three classes, six
prompt groups, all four saved layers and hidden width two. `encoder.pt` holds
the pooled states of a native Qwen with four blocks, width seven, intermediate
width eleven, four query and two key/value heads of width six and vocabulary
seventeen, in float32, float16 and bfloat16. Component tests check
hand-derived values on literal inputs.

The Qwen golden injects `SdpaNaive`; default fused CUDA attention, full-data
scores and five-minute admission remain unmeasured. Qwen receives no training
updates in this recipe.
