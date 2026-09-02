"""Tests for the nanochat experiment ladder.

Each test asserts the DELTA a fork applies, which is what enforces one change
per experiment: a fork that quietly moved a second knob fails here rather than
producing a result nobody can attribute.

Every test builds configs only -- no data, no device, no training -- so the
ladder stays checkable on any machine.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import json
import math
import pickle

from configgle import PartialConfig
from configgle.pprinting import pformat
from pyarrow import parquet

import numpy as np
import pyarrow as pa
import pytest
import tiktoken

from priml.baselines.nanochat import experiments
from priml.baselines.nanochat.data import token_bytes_fingerprint
from priml.baselines.nanochat.experiments import NanoChatLoop
from priml.baselines.nanochat.flash3 import Flash3Attention
from priml.baselines.nanochat.train_step import (
    NanoChatTrainStep,
    nanochat_optimizer,
)
from priml.metrics.bits_per_byte import BitsPerByte
from priml.model.attention.value_gated_attention import ValueGatedAttention
from priml.model.narrow_embedding import NarrowEmbedding
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.normuon import NorMuon
from priml.runtime import SingleProcess
from priml.train.checkpointing import Checkpointer
from priml.train.parallelism import NoParallel
from priml.train.tracker import AsyncTracker, TrackerList, WandbTracker


LADDER: list[tuple[str, Callable[[], NanoChatLoop.Config]]] = [
    ("exp000", experiments.exp000),
    ("exp001", experiments.exp001),
    ("exp002", experiments.exp002),
    ("exp003", experiments.exp003),
    ("exp_smoke", experiments.exp_smoke),
]

# exp000 pins a kernel that builds only for SM90, so anything CONSTRUCTING a
# model excludes it: the rung is unbuildable on a laptop and on most CI, which
# is the point of exp001 existing. Its config-level fields are still checked.
PORTABLE: list[tuple[str, Callable[[], NanoChatLoop.Config]]] = LADDER[1:]


def _pattern(cfg: NanoChatLoop.Config) -> str:
    """The window pattern the stack's attention layers carry."""
    attention = cfg.step.model.template.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    return attention.window_pattern


@pytest.mark.parametrize(("name", "factory"), LADDER, ids=[n for n, _ in LADDER])
def test_every_experiment_finalizes(
    name: str,
    factory: Callable[[], NanoChatLoop.Config],
) -> None:
    """A config must build without a dataset or a GPU."""
    config = factory().copy_tree().finalize()
    assert config.experiment_name == name
    assert config.study_name == "nanochat"


def test_exp000_pins_the_reference_kernel() -> None:
    """The reproduction rung must issue the kernel the reference measured on.

    A fused attention reduces in a different order than a masked SDPA, so a
    rung matching every hyperparameter and swapping the kernel produces a
    different number -- and could not settle whether the port is faithful,
    which is the only question exp000 exists to answer.
    """
    attn = experiments.exp000().step.model.template.attn
    assert isinstance(attn, ValueGatedAttention.Config)
    assert isinstance(attn.kernel, Flash3Attention.Config)


def test_exp001_changes_only_the_kernel() -> None:
    """The portable rung is the reference recipe, kernel aside."""
    base, fork = experiments.exp000(), experiments.exp001()
    base_attn, fork_attn = base.step.model.template.attn, fork.step.model.template.attn
    assert isinstance(base_attn, ValueGatedAttention.Config)
    assert isinstance(fork_attn, ValueGatedAttention.Config)
    assert isinstance(base_attn.kernel, Flash3Attention.Config)
    assert not isinstance(fork_attn.kernel, Flash3Attention.Config)
    assert _pattern(fork) == _pattern(base)
    assert fork.step.model.value_embedding_stride == (
        base.step.model.value_embedding_stride
    )
    assert fork.step.train_budget_sec == base.step.train_budget_sec
    assert fork.seed == base.seed


def test_exp001_reports_to_wandb_asynchronously() -> None:
    """The portable parent owns non-blocking dashboard delivery."""
    config = experiments.exp001()

    assert isinstance(config.tracker, TrackerList.Config)
    assert list(config.tracker.trackers) == ["wandb"]
    wrapper = config.tracker.trackers["wandb"]
    assert isinstance(wrapper, AsyncTracker.Config)
    assert isinstance(wrapper.tracker, WandbTracker.Config)
    assert wrapper.tracker.project == "nanochat"


def test_exp000_turns_both_mechanisms_on() -> None:
    """The baseline is the recipe to reproduce, not the plain control.

    Both mechanisms belong here because the forks REMOVE them one at a time;
    if exp000 shipped either one off, the rung meant to price it would be
    measuring against an unstated recipe instead.
    """
    cfg = experiments.exp000()
    assert _pattern(cfg) == "SSSL"
    assert cfg.step.model.value_embedding_stride == 2


@pytest.mark.parametrize(
    ("name", "factory"),
    LADDER[:-1],
    ids=[name for name, _ in LADDER[:-1]],
)
def test_every_scored_experiment_selects_cuda(
    name: str,
    factory: Callable[[], NanoChatLoop.Config],
) -> None:
    """Every scored rung must consume its requested GPU allocation."""
    config = factory()
    assert isinstance(config.step.parallelism, NoParallel.Config), name
    assert config.step.parallelism.device == "cuda", name
    assert config.dataset.device == "cuda", name
    assert isinstance(config.runtime, SingleProcess.Config), name
    assert config.runtime.device == "cuda", name


def test_smoke_keeps_automatic_device_selection() -> None:
    config = experiments.exp_smoke()

    assert isinstance(config.step.parallelism, NoParallel.Config)
    assert config.step.parallelism.device is None
    assert config.dataset.device == "auto"
    assert isinstance(config.runtime, SingleProcess.Config)
    assert config.runtime.device == "auto"


def test_exp002_removes_only_the_value_embeddings() -> None:
    base, fork = experiments.exp001(), experiments.exp002()
    assert base.step.model.value_embedding_stride == 2
    assert fork.step.model.value_embedding_stride == 0
    assert _pattern(fork) == _pattern(base)
    assert fork.step.train_budget_sec == base.step.train_budget_sec
    assert fork.step.model.channels_in == base.step.model.channels_in


def test_exp003_removes_the_window_too() -> None:
    base, fork = experiments.exp002(), experiments.exp003()
    assert _pattern(base) == "SSSL"
    assert _pattern(fork) == "L"
    assert fork.step.model.value_embedding_stride == (
        base.step.model.value_embedding_stride
    )
    assert fork.step.train_budget_sec == base.step.train_budget_sec


def test_the_value_embedding_stride_follows_a_changed_depth() -> None:
    """A stride survives a fork that changes the depth; indices would not.

    Computing the layer list in the factory snapshots whatever ``num_layers``
    was at that moment, so a fork narrowing the model carries indices for a
    stack that no longer exists -- and the model rejects them.
    """
    cfg = experiments.exp000()
    cfg.step.model.num_layers = 4
    final = cfg.copy_tree().finalize()
    assert final.step.model.value_embedding_layers == [1, 3]


@pytest.mark.parametrize(("name", "factory"), LADDER, ids=[n for n, _ in LADDER])
def test_no_rung_resumes_a_checkpoint(
    name: str,
    factory: Callable[[], NanoChatLoop.Config],
) -> None:
    """Every rung must start fresh, and state it in the config rather than a flag.

    Two independent reasons, either alone sufficient: the run is a wall-clock
    BUDGET, so a resumed run measures a different budget than the one it
    reports; and the packed token stream cannot be positioned mid-corpus, so
    the dataset refuses the restore outright. Carried on ``exp000`` and
    inherited, since every rung forks it -- a per-launch ``--override`` is a
    property of one command line, and the rung it protects is launched from
    several.
    """
    checkpointing = factory().checkpointing
    assert isinstance(checkpointing, Checkpointer.Config), name
    assert checkpointing.resume is False, name


def test_the_budget_and_the_schedule_horizon_agree() -> None:
    """A schedule annealing past the stop wastes the tail; short of it, the
    run trains its last steps at a rate the recipe never intended.
    """
    for name, factory in LADDER:
        config = factory()
        assert config.max_time == config.step.train_budget_sec, name
        assert config.max_time_kind == "train", name


def test_the_loop_reads_the_steps_budget_clock() -> None:
    """Stop condition, reported time, and schedules must share one clock.

    ``TrainLoop`` rebases after one step; this baseline excludes a configured
    warmup. If the loop kept its own clock the run would anneal against one
    budget and stop on another, differing by the whole warmup.
    """
    loop = NanoChatLoop.__new__(NanoChatLoop)
    step = NanoChatTrainStep.__new__(NanoChatTrainStep)
    step.elapsed_sec = 12.5
    loop.step = step
    assert loop._train_elapsed() == 12.5


def test_the_dataset_batch_follows_the_steps_pass_size() -> None:
    """Two places naming the same number silently disagree; one propagates."""
    config = experiments.exp000()
    config.step.rows_per_pass = 8
    assert config.copy_tree().finalize().dataset.batch_size == 8


@pytest.mark.parametrize(("name", "factory"), PORTABLE, ids=[n for n, _ in PORTABLE])
@pytest.mark.compute_training
def test_every_experiments_eval_geometry_is_constructible(
    name: str,
    factory: Callable[[], NanoChatLoop.Config],
    tmp_path: Path,
) -> None:
    """A shipped experiment must survive building its dataset, not just its config.

    Finalizing proves the tree is coherent; it does not run the validation that
    lives in ``__init__``. A cap that is not a whole number of eval batches, or
    a batch wider than the split, therefore passes every config-only test and
    fails the moment a run reaches for data.

    ``exp000`` is excluded because it CONSTRUCTS a model: its pinned kernel
    builds only for SM90, so this would assert the runner's hardware rather
    than the experiment's geometry. It shares every geometry field with
    ``exp001``, which is covered here.
    """
    config = factory()
    model = config.step.model
    # Shrunk on every axis that costs time and none that bears on the question:
    # the subject is EVAL BATCHING, so the context, width, depth, and vocabulary
    # are cut to what a CPU runner can step in milliseconds.
    model.channels_in = 32
    model.num_layers = 1
    model.max_seq_len = 16
    model.vocab_size = 32
    attention = model.template.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    attention.channels_head = 32
    config.step.parallelism = NoParallel.Config(device="cpu")
    config.step.dtype_autocast = None
    config.step.compile = None
    # The optimizers are constructed and never stepped -- this asks about eval
    # batching. Leaving them compiled charges the test ``torch.compile``'s own
    # import of inductor, 1.4 of its 1.5 seconds, for a kernel it never issues.
    config.step.optimizer = nanochat_optimizer(compile=False)
    # Autocast is off above, so the narrow tables the recipe declares would
    # hand a half-precision stream to a float32 projection and the matmul would
    # refuse -- a dtype error rather than an answer about eval batching. Widened
    # HERE, beside the autocast it pairs with, rather than defaulted away on the
    # model: the pairing is the invariant, and hiding half of it makes the other
    # half look arbitrary.
    embedding = model.embedding
    assert isinstance(embedding, NarrowEmbedding.Config)
    embedding.dtype = None
    model.rope.dtype = None

    # A corpus at the vocabulary this experiment declares. Two shards, since
    # the validation one is pinned and excluded from training.
    _prepared(tmp_path, vocab=model.vocab_size)
    config.base_dir = "/"
    config.dataset.working_dir = str(tmp_path)
    config.dataset.device = "cpu"
    config.dataset.num_train_shards = 1
    config.dataset.val_shard = 1
    # Two batches, so the stream's extent is exercised without the recipe's
    # 20.97M tokens. The width is left as the experiment declares it, since
    # that is the field under test.
    config.dataset.eval_tokens = 2 * config.dataset.eval_batch_size * model.max_seq_len

    final = config.copy_tree().finalize()
    built = final.dataset.make()
    step = final.step.make()
    # Consumed one at a time, NOT collected into a list first: the stream yields
    # views of one reused buffer, so a list of them holds the same tensor twice
    # and every assertion after it would read the last batch. Each is stepped
    # where it is drawn, which is also how the training loop takes them.
    drawn = 0
    for batch in built.eval_dataloader():
        # RUN it: drawing proves the batch exists, not that the model can
        # consume it. A token id the embedding rejects passes the first check
        # and fails the second.
        step.eval_loss(**step.preprocess_batch(batch))
        drawn += 1
    assert drawn == 2, name


def _prepared(root: Path, *, vocab: int) -> None:
    """Write a corpus and a vocabulary of exactly ``vocab`` tokens.

    Single-character merges so a document's token count is its length, which is
    what lets the packer fill a row out of a handful of short documents.
    """
    reserved = tuple(f"<|reserved_{index}|>" for index in range(16))
    merges = vocab - len(reserved)
    assert merges > 0, "the vocabulary must exceed its reserved tokens"
    alphabet = [bytes([ord("a") + index]) for index in range(merges)]
    encoding = tiktoken.Encoding(
        name="test",
        pat_str=r".",
        mergeable_ranks={token: rank for rank, token in enumerate(alphabet)},
        special_tokens={name: merges + index for index, name in enumerate(reserved)},
    )
    directory = root / "tokenizer"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "tokenizer.pkl").open("wb") as file:
        pickle.dump(encoding, file)
    special = set(reserved)
    lengths = np.array(
        [
            0 if (text := encoding.decode([token])) in special else len(text.encode())
            for token in range(vocab)
        ],
        dtype=np.int32,
    )
    np.save(directory / "token_bytes.npy", lengths)
    (directory / "tokenizer_recipe.json").write_text(
        json.dumps(
            {
                "bos_token": reserved[0],
                "token_bytes_sha256": token_bytes_fingerprint(lengths),
            },
        ),
    )
    documents = [
        bytes(alphabet[index % merges][0] for _ in range(index + 1)).decode()
        for index in range(8)
    ]
    for shard in range(2):
        parquet.write_table(
            pa.table({"text": documents}),
            root / f"shard_{shard:05d}.parquet",
        )


def test_the_dataset_inherits_the_models_geometry() -> None:
    """The model declares the geometry; the dataset verifies data against it.

    Without the push the two are independent copies agreeing only by coincident
    defaults, so ``exp_smoke`` -- which narrows the model -- would load rows of
    a width nothing checked.
    """
    config = experiments.exp_smoke().copy_tree().finalize()
    assert config.dataset.max_seq_len == config.step.model.max_seq_len
    assert config.dataset.vocab_size == config.step.model.vocab_size


def test_the_score_is_bits_per_byte() -> None:
    """A per-token score would rank a coarser tokenizer better for free."""
    assert isinstance(experiments.exp000().metrics["val"], BitsPerByte.Config)


def test_smoke_is_small_on_every_costly_axis() -> None:
    """It answers "does this run", so anything not bearing on that is cut."""
    smoke, base = experiments.exp_smoke(), experiments.exp000()
    assert smoke.step.train_budget_sec < base.step.train_budget_sec
    assert smoke.step.model.channels_in < base.step.model.channels_in
    assert smoke.step.model.num_layers < base.step.model.num_layers
    assert smoke.step.model.max_seq_len < base.step.model.max_seq_len
    assert not smoke.step.compile
    # A finite bound: exp000 stops on its time budget and leaves max_steps at
    # infinity, against which any value would compare smaller.
    assert smoke.max_steps < 100
    assert math.isinf(base.max_steps)


def test_smoke_differs_from_exp001_only_in_size() -> None:
    """The test artifacts use smoke to guard exp001; that rests on this.

    Everything a golden could catch -- the windowing, the value embeddings, the
    optimizer partition, the precision -- must be the SAME object in both, or
    the test artifacts freeze a model the ladder does not run. Only sizes and
    budgets may differ.
    """
    smoke, base = experiments.exp_smoke(), experiments.exp001()
    smoke_model, base_model = smoke.step.model, base.step.model
    assert smoke_model.value_embedding_stride == base_model.value_embedding_stride
    assert repr(smoke_model.embedding) == repr(base_model.embedding)
    assert smoke_model.rope.dtype == base_model.rope.dtype
    assert repr(smoke_model.lm_head) == repr(base_model.lm_head)
    # Compared by ``repr``, not by ``==``: a PartialConfig raises on
    # ``parent_class`` when equality reaches it, and both of these slots hold
    # one. ``repr`` is declared on object, so it also needs no narrowing of the
    # ``Makeable`` the field is typed as -- which ``pprint`` would.
    # Taken BEFORE finalize, since finalize rescales the Adam rates by the
    # model width and the two rungs are deliberately different widths.
    assert repr(smoke.step.optimizer) == repr(base.step.optimizer)
    assert repr(smoke.step.schedule) == repr(base.step.schedule)
    assert smoke.step.dtype_autocast == base.step.dtype_autocast
    smoke_attn, base_attn = smoke_model.template.attn, base_model.template.attn
    assert isinstance(smoke_attn, ValueGatedAttention.Config)
    assert isinstance(base_attn, ValueGatedAttention.Config)
    assert smoke_attn.window_pattern == base_attn.window_pattern
    assert smoke_attn.norm_qk == base_attn.norm_qk
    assert type(smoke_attn.kernel) is type(base_attn.kernel)


def test_the_models_compile_switch_leaves_the_optimizers_alone() -> None:
    """A compiled optimizer step and an eager one are different arithmetic.

    ``exp_smoke`` turns the model's compile off to start quickly. If that
    reached the optimizer, the rung would stop reproducing the reference --
    whose kernels are compiled -- and the test artifacts minted over it would
    freeze numbers no shipped experiment computes.
    """
    for factory in (experiments.exp_smoke, experiments.exp001):
        optimizer = factory().copy_tree().finalize().step.optimizer
        assert isinstance(optimizer, CompositeOptimizer.Config)
        for member in optimizer.optimizers:
            if isinstance(member, PartialConfig):
                assert member._kwargs.get("compile", True) is True, factory.__name__
            else:
                assert isinstance(member, NorMuon.Config)
                assert member.compile is True, factory.__name__


@pytest.mark.compute_large_fixture
def test_exp000_matches_its_golden_config(request: pytest.FixtureRequest) -> None:
    """Pin the WHOLE finalized ``exp000`` as readable text.

    ``exp000`` is the control every fork is measured against, so a change to
    it invalidates published numbers. A digest would say only that something
    moved; this golden says WHICH field, from what, to what.
    ``hide_default_values=False`` so a field that changes only because a
    library default changed still shows up here.

    Marked rather than shrunk: the claim IS the whole tree, so every lever that
    would bring the render under the unit budget -- a narrower model, fewer
    layers, hiding defaults -- pins a config no experiment runs. Measured 0.25s
    on colossus, essentially all of it inside ``pformat``, which walks the
    finalized tree and tokenizes each rendered node to find replacements
    outside string literals.

    Refresh ``testdata/exp000.txt`` with ``--golden-overwrite`` after reading
    the diff.
    """
    golden = Path(__file__).resolve().parent / "testdata" / "exp000.txt"
    rendered = pformat(
        experiments.exp000().copy_tree().finalize(), hide_default_values=False
    )
    if request.config.getoption("--golden-overwrite", default=False):
        golden.parent.mkdir(parents=True, exist_ok=True)
        _ = golden.write_text(rendered + "\n", encoding="utf-8")
    assert golden.read_text(encoding="utf-8") == rendered + "\n", (
        "exp000 changed; read the diff, then rerun with --golden-overwrite "
        "if the change is intended."
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
