# Name That LLM data

Accept the [competition rules](https://www.kaggle.com/competitions/name-that-llm)
and authenticate with Kaggle. From the repository root, download the complete
competition CSVs and pinned Qwen weights, then prepare the data once:

```sh
UV_NO_SYNC=1 uv --quiet run --frozen kaggle competitions download \
  -c name-that-llm -p /opt/scratch/datasets/name-that-llm/raw
unzip /opt/scratch/datasets/name-that-llm/raw/name-that-llm.zip \
  -d /opt/scratch/datasets/name-that-llm/raw
UV_NO_SYNC=1 uv --quiet run --frozen kaggle models instances versions download \
  qwen-lm/qwen-3/transformers/8b-base/1 \
  -p /opt/scratch/datasets/name-that-llm/qwen-3-8b-base-v1 --untar
UV_NO_SYNC=1 uv --quiet run --frozen python -m priml \
  priml.baselines.name_that_llm.scripts.prepare_data.prepare_data
```

Preparation requires all 2,699 labelled passages and 150 prompt identifiers.
It preserves the input CSV bytes and records digests and row identifiers. Tests
write two labelled CSV rows and one test row under `tmp_path`; numerical tests
use small checked-in `.pt` records. Running pytest needs neither Kaggle data nor
Qwen weights.
