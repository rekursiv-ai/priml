"""The nanochat experiment ladder.

Every experiment trains to the same wall-clock BUDGET and is scored on the same
validation stream, so a change is measured by what it achieves in fixed time
rather than in fixed steps. A change that makes a step cheaper is rewarded with
more steps; one that makes a step better is rewarded directly. Both show up in
one number.

``exp000`` REPRODUCES a published recipe rather than stating one of ours, down
to the attention kernel, and is never edited: it is the number every other rung
is measured against, so an edit here silently reprices the whole ladder. Each
rung below is a fork removing exactly one thing, so it answers "what does this
part earn?" rather than "does adding it help?" -- the ladder descends from the
reference to a plain transformer::

    exp000  the published five-minute recipe, on its own FlashAttention-3
      +-- exp001  without the pinned kernel: portable attention
            +-- exp002  without the value embeddings
                  +-- exp003  without the windowing too: a plain transformer

Only ``exp000`` pins FA3, so only it requires SM90. ``exp001`` is the rung to
fork for ordinary work: the same recipe, and portable to any GPU.

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --num-train-shards 7
    time CUDA_VISIBLE_DEVICES=1 uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp001

The four experiments above retain their 300-second defaults. ``exp004`` starts
a separate cumulative sequence from ``exp000``, checkpointing the major
improvements from the research campaigns. It adds n-gram memory, output norms,
residual and attention gates, layer pooling, nonuniform FFNs, sparse table
optimization, and fused attention kernels. ``exp020`` switches to prepared
16K Unigram tokens with byte-matched reference evaluation.

``exp004`` through ``exp022`` default to 525 charged training seconds for
H-series GPUs, excluding compilation warmup and evaluation. Each factory
includes a commented 300-second budget for B200. In this added sequence,
``exp004`` through ``exp015`` also use Hopper-only FA3 and need an
attention-backend change for B200. ``exp016`` onward use FlashAttention-4 and
Triton kernels. ``exp_smoke`` uses portable attention and a small model to
check an installation.

For prepared Unigram experiments, run the preparation script without the BPE
flags, then use its ``--stage train --experiment exp022`` entry. It verifies
the prepared inputs and binds their identities before launching training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from functools import partial
from typing import TYPE_CHECKING, Self, cast, override


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")  # ~1050 ms; experiment setup needs it.

from configgle import Makes, PartialConfig

from priml.baselines.nanochat.attention import (
    CausalAttention,
    Flash3Attention,
    Flash4Attention,
)
from priml.baselines.nanochat.data import NanoChatData, ReferenceEvaluation
from priml.baselines.nanochat.model import (
    GatedResidualMix,
    MemoryNanoChatLM,
    OutputNormFeedForward,
    ScaledSoftCap,
    SourceReuseTransformerBlock,
    thresholded_relu_squared,
)
from priml.baselines.nanochat.ngram import HashedNgramTables, NgramEmbedding
from priml.baselines.nanochat.optimizers import (
    BiasCorrectedRMSProp,
    FFNScaledNorMuon,
    ScheduledOptimizerUpdate,
    WeightDecayPulse,
)
from priml.baselines.nanochat.train_step import (
    BoundedTokenCrossEntropy,
    NanoChatTrainStep,
    NgramTrainStep,
    ReferenceBitsPerByte,
)
from priml.math.schedules import trapezoidal
from priml.metrics.bits_per_byte import BitsPerByte
from priml.model import softcap
from priml.model.attention.rope import HuggingFaceFrequencies
from priml.model.attention.value_gated_attention import (
    ValueGatedAttention,
    sdpa_attention,
)
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.narrow_embedding import NarrowEmbedding
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLUReluSquared
from priml.optimizers.composite import CompositeOptimizer, matching
from priml.optimizers.fused_adamw import FusedAdamW
from priml.runtime import SingleProcess
from priml.train.checkpointing import Checkpointer
from priml.train.parallelism import NoParallel
from priml.train.tracker import (
    AsyncTracker,
    FileTracker,
    TrackerList,
    WandbTracker,
)
from priml.train.train_loop import TrainLoop

import priml.model.transformer.block


class NanoChatLoop(TrainLoop):
    """A training loop whose budget clock is the train step's.

    ``TrainLoop`` rebases its own pure-train clock after ONE step, which is
    right when only the first step carries a compile. This baseline excludes a
    configured number of warmup steps instead, and the stop condition, the
    reported elapsed time, and the schedules must all read the SAME clock --
    otherwise the run anneals its learning rate against one budget while
    stopping on another, and the two disagree by the warmup.
    """

    class Config(Makes["NanoChatLoop"], TrainLoop.Config):
        """A loop with the nanochat step and dataset already in place.

        Narrowing the two slots here rather than at each call site lets a
        factory reach ``cfg.step.model`` directly, with no ``isinstance``
        narrow before a field it is about to set.
        """

        step: NanoChatTrainStep.Config = field(
            default_factory=NanoChatTrainStep.Config,
        )
        """Model, optimization, and the budget the schedules anneal over."""

        dataset: NanoChatData.Config = field(
            default_factory=NanoChatData.Config,
        )
        """The corpus, packed into rows as the reference packs them."""

        @override
        def finalize(self) -> Self:
            # How many rows fit in one pass is the step's decision -- it
            # follows device memory -- so the dataset takes its batch size from
            # there rather than the two being set to agree by hand.
            self.dataset.batch_size = self.step.rows_per_pass
            # Geometry is declared ONCE, on the model, and pushed here so the
            # dataset can verify the fitted vocabulary against it at load. Two
            # independently-typed copies would agree only by coincidence, and
            # disagree deep inside a forward pass.
            self.dataset.vocab_size = self.step.model.vocab_size
            self.dataset.max_seq_len = self.step.model.max_seq_len
            return super().finalize()

    @override
    def _train_elapsed(self) -> float:
        """Return the step's budget-counted seconds (warmup excluded)."""
        step = self.step
        assert isinstance(step, NanoChatTrainStep)
        return step.elapsed_sec

    # Its ``next(train_loader)`` runs inside the region its clock brackets
    # (train.py:550, between 543 and 573). Ours happens here, outside the step, so the
    # budget would otherwise buy free steps: measured at 0.160 of 1.683 s/step on a
    # 5090, a tenth of the run.
    @override
    def _on_batch_ready(self, fetch_time: float) -> None:
        """Charge loading to the budget, as the reference charges it."""
        step = self.step
        assert isinstance(step, NanoChatTrainStep)
        step.charge_budget(fetch_time)


def exp000() -> NanoChatLoop.Config:
    """Return the reference recipe.

    - Eight layers, width 512, and alternating value embeddings.
    - Hopper-only FlashAttention-3 with SSSL attention windows.
    - BF16 embedding tables and rotary factors; TF32 matrix multiplication.
    - 300-second training budget; evaluate once after training.
    - Seed 42; single-GPU execution without checkpoint resumption.
    - Save file metrics and report to W&B asynchronously.

    Operationally, this recipe reports to W&B through the shared asynchronous
    tracker wrapper. Reporting is not a scientific treatment.

    Hypothesis:
      Porting the reference's architecture, optimizer partition, data
      protocol, and kernel into this package reproduces its published score,
      so every fork measures a recipe change rather than a porting artifact.

    References:
      https://github.com/karpathy/autoresearch
        Karpathy. autoresearch, commit
        b11d6f283f866eb7e10fb776a4b8553fef873fd5.
      https://arxiv.org/abs/2109.08668
        So et al. Primer: Searching for Efficient Transformer for Language
        Modeling.
      https://kellerjordan.github.io/posts/muon/
        Jordan et al. 2024. Muon: an optimizer for hidden layers.
      https://arxiv.org/abs/2004.05150
        Beltagy et al. Longformer: The Long-Document Transformer.
      https://arxiv.org/abs/2410.17897
        Zhou et al. Value Residual Learning.

    Results:
      H200, 525s: 0.973908 mean BPB (3 seeds).
      H200, 300s: 1.005432 mean BPB (3 seeds).

    """
    cfg = NanoChatLoop.Config()
    cfg.study_name = "nanochat"
    cfg.experiment_name = "exp000"

    # The kernel the reference measured on, pinned by revision. Stated HERE
    # rather than inherited from a portable rung, because it is part of the
    # recipe being reproduced rather than a deviation from one: exp000 is the
    # statement, and every other rung is a diff against it.
    block = cfg.step.model.template
    assert isinstance(block.attn, ValueGatedAttention.Config)
    block.attn.kernel = Flash3Attention.Config()

    # The reference normalizes with a bare ``F.rms_norm(x, shape)``, which
    # leaves eps to torch -- the dtype's own epsilon, ~1.19e-7 in float32.
    # priml's RMSNorm defaults to 1e-6, an order of magnitude larger, and that
    # difference reaches the residual stream at every sublayer of every layer.
    for norm in (block.norm1, block.norm2, block.attn.norm_qk):
        assert isinstance(norm, RMSNorm.Config)
        norm.eps = torch.finfo(torch.float32).eps

    # Half precision, stated as two knobs because they are two decisions the
    # reference made and one default cannot carry: the tables are HELD narrow
    # (``train.py:177-179``) and the rotary factors are ROUNDED narrow
    # (``train.py:189``), so every product inside the rotation accumulates
    # there rather than being promoted and rounded once. Both are set here
    # rather than defaulted on the model, because a narrowed table makes the
    # model runnable only under autocast -- which this recipe's loop supplies
    # and a bare ``model(tokens)`` does not.
    embedding = cfg.step.model.embedding
    assert isinstance(embedding, NarrowEmbedding.Config)
    embedding.dtype = torch.bfloat16
    cfg.step.model.rope.dtype = torch.bfloat16

    cfg.step.model.channels_in = 512
    cfg.step.model.num_layers = 8
    block.attn.window_pattern = "SSSL"
    # Alternating layers. A stride rather than the indices it implies, so a
    # fork that changes the depth still gets alternating layers rather than
    # indices computed against a stack that no longer exists.
    cfg.step.model.value_embedding_stride = 2

    # The budget the schedules anneal over and the budget the loop stops on:
    # equal, or the learning rate lands short of zero or decays past the end.
    cfg.step.train_budget_sec = 300.0
    cfg.max_time = cfg.step.train_budget_sec
    cfg.max_time_kind = "train"

    cfg.metrics["val"] = BitsPerByte.Config()
    # One eval, at the end, as the reference does (train.py:611, after its
    # loop). A mid-run eval is not charged to the budget, so it would not cost
    # steps -- but it holds the GPU for ~23 (5090) seconds each.
    cfg.num_steps_eval = -1

    # Print every step.
    cfg.num_steps_log = 1
    cfg.early_train_log_steps = 0

    # Every row is a full context by construction, so a pass over the data is
    # not a meaningful boundary -- the budget is what ends the run.
    cfg.eval_every_epoch = False

    # Pinned, not left to the loop's ``None`` default, which draws from OS
    # entropy: two runs of this factory would then differ in initialization
    # before any code changed, and a comparison between them would measure the
    # draw rather than the recipe. It seeds initialization alone -- the data
    # order is the corpus's own, packed deterministically and never shuffled.
    cfg.seed = 42
    # Avoid checkpoint resuming because the job is single-shot.
    checkpointing = cfg.checkpointing
    assert isinstance(checkpointing, Checkpointer.Config)
    checkpointing.resume = False

    cfg.step.parallelism = NoParallel.Config()
    cfg.step.parallelism.device = "cuda"
    cfg.dataset.device = "cuda"
    cfg.runtime = SingleProcess.Config()
    cfg.runtime.device = "cuda"
    # TF32 matmuls. The reference recipe's throughput assumes them, and a
    # run left at torch's default reduces in a different order, so a score
    # measured here would not be comparable to one measured there.
    cfg.runtime.float32_matmul_precision = "high"

    dashboard = WandbTracker.Config()
    dashboard.project = "nanochat"
    wrapper = AsyncTracker.Config()
    wrapper.tracker = dashboard
    cfg.tracker = TrackerList.Config()
    cfg.tracker.trackers = {
        "metrics": FileTracker.Config(),
        "wandb": wrapper,
    }

    return cfg


def exp001() -> NanoChatLoop.Config:
    """Fork exp000 with the following changes.

    - Replace FlashAttention-3 with PyTorch attention.

    The scientific deviation is the backend. ``exp000`` pins FlashAttention-3,
    which builds only for SM90, so it refuses to construct anywhere else.
    This rung takes the CUDA backend torch dispatches for the allocated GPU
    and therefore runs across supported CUDA architectures. Fork this for
    ordinary work: reproducing the reference kernel requires Hopper, while
    a portable experiment should use the backend available on its GPU.

    The FA3 path in ``exp000`` cannot be tested off Hopper. This rung lets us
    check the port with a common kernel: the reference implementation is given
    the same portable backend and stepped beside it. The
    ``scripts/karpathy_parity.py`` script compares parameters and gradients,
    so the comparison measures recipe differences with the backend held fixed.

    Hypothesis:
      The recipe's score comes from its architecture and optimizer rather
      than its kernel, so the portable backend reproduces exp000 within
      run-to-run noise. Kernel throughput can change the number of steps
      completed within the fixed training budget.

    Results:
      H200, 525s: 1.011005 mean BPB (3 seeds).
      H200, 300s: 1.056776 mean BPB (3 seeds).

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"

    # The following are tuned for rtx5090.
    block = cfg.step.model.template
    assert isinstance(block.attn, ValueGatedAttention.Config)
    block.attn.kernel = PartialConfig(sdpa_attention)

    # Karpathy uses 32 but we can get 15.7% more speed @ 64.
    # Theoretically up to 76.
    # cfg.step.rows_per_pass = 64
    # Karpathy uses 128 but we can actually fit 512.
    # cfg.dataset.eval_batch_size = 512
    # However it doesn't seem to affect performance...dataloader bottleneck?

    return cfg


def exp002() -> NanoChatLoop.Config:
    """Fork exp001 with the following changes.

    - Remove value embeddings.

    Hypothesis:
      A deep stack's residual stream is increasingly processed, so a layer
      wanting the raw token identity must reconstruct it. Alternating layers
      read a dedicated token embedding through a per-head gate to supply it
      directly. Removing those tables and gates may cost accuracy but makes
      each step cheaper. This rung wins if the extra steps within the fixed
      budget recover more accuracy than the value embeddings supplied.

    References:
      https://arxiv.org/abs/2410.17897
        Zhou et al. Value Residual Learning.

    Results:
      H200, 525s: 1.028092 mean BPB (3 seeds).
      H200, 300s: 1.074704 mean BPB (3 seeds).

    """
    cfg = exp001()
    cfg.experiment_name = "exp002"
    cfg.step.model.value_embedding_stride = 0
    return cfg


def exp003() -> NanoChatLoop.Config:
    """Fork exp002 with the following changes.

    - Use full-context attention in every layer.

    Hypothesis:
      Most layers resolve local structure, so restricting their attention
      to recent history costs little accuracy while making each step
      cheaper. Restoring full context buys back accuracy at a higher cost
      per step. Under a fixed budget it loses if the steps it gives up were
      worth more than the attention it regains.

    References:
      https://arxiv.org/abs/2004.05150
        Beltagy et al. Longformer: The Long-Document Transformer.

    Results:
      H200, 525s: 0.995873 mean BPB (3 seeds).
      H200, 300s: 1.030769 mean BPB (3 seeds).

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    window_attn = cfg.step.model.template.attn
    assert isinstance(window_attn, ValueGatedAttention.Config)
    window_attn.window_pattern = "L"
    return cfg


class NgramTrainLoop(NanoChatLoop):
    """Run memory-augmented models on the training step's budget clock."""

    class Config(NanoChatLoop.Config):
        """Bind memory-table optimization to the packed training dataset."""

        step: NanoChatTrainStep.Config = field(default_factory=NgramTrainStep.Config)
        """Model, n-gram gradients, optimizer policy and training budget."""

        @override
        def finalize(self) -> Self:
            if self.dataset.reference_evaluation is not None:
                self.metrics["val"] = ReferenceBitsPerByte.Config()
            return super().finalize()


def exp004() -> NgramTrainLoop.Config:
    """Fork exp000 to start the campaign checkpoint sequence.

    - Use the memory-capable model, train step, and loop.
    - Increase the training budget from 300 to 525 seconds for Hopper GPUs.
    - Set the campaign run-directory template.

    The longer Hopper budget approximates the step count of a 300-second B200
    run; see "Reproducing the blog post" in README.md.

    Results:
      Campaign H200, 525s: 0.972782 mean BPB (3 seeds).
      Campaign H200, 300s: 1.002721 mean BPB (3 seeds).

    """
    config = NgramTrainLoop.Config().update(exp000())
    config.step = NgramTrainStep.Config().update(config.step)
    config.step.model = MemoryNanoChatLM.Config().update(config.step.model)
    config.study_name = "nanochat"
    config.experiment_name = "exp004"
    config.step.train_budget_sec = 525.0
    config.max_time = config.step.train_budget_sec
    config.seed = 42
    config.working_dir = "/runs/{study_name}/{experiment_name}"
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # config.max_time = config.step.train_budget_sec = 300.0 # Seconds.
    return config


def exp005() -> NgramTrainLoop.Config:
    """Fork exp004 with the following changes.

    - Reduce depth from eight layers to five.
    - Increase width from 512 to 768.
    - Increase microbatches to 256 rows.
    - Set local attention windows to 384 tokens.

    Results:
      Campaign H200, 525s: 0.961005 mean BPB (3 seeds).
      Campaign H200, 300s: 0.988665 mean BPB (3 seeds).

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    cfg.working_dir = "/runs/{study_name}/{experiment_name}"
    cfg.step.rows_per_pass = 256
    cfg.step.model.channels_in = 768
    cfg.step.model.num_layers = 5
    attention = cfg.step.model.template.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    attention.window = 384
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp006() -> NgramTrainLoop.Config:
    """Fork exp005 with the following changes.

    - Add a zero-initialized, 1,048,576-row input bigram table.
    - Use hash multipliers (1, 257) and embedding scale 0.25.

    Results:
      Campaign H200, 525s: 0.957160 mean BPB (3 seeds).
      Campaign H200, 300s: 0.969540 mean BPB (3 seeds).

    """
    cfg = exp005()
    cfg.experiment_name = "exp006"
    embedding = NgramEmbedding.Config().update(cfg.step.model.embedding)
    context = embedding.contexts["bigram"] = NgramEmbedding.Config()
    context.multipliers = (1, 257)
    context.scale = 0.25
    context.num_embeddings = 1_048_576
    context.inner = Embedding.Config(init_weight=torch.nn.init.zeros_)
    cfg.step.model.embedding = embedding
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp007() -> NgramTrainLoop.Config:
    """Fork exp006 with the following changes.

    - Change bigram hash multipliers to (1, 1,000,003).
    - Add a zero-initialized, 1,048,576-row input trigram table.
    - Use trigram multipliers (1, 257, 66,049) and scale 0.125.
    - Separate context-table optimization from token embeddings.
    - Set value-embedding and context-table weight decay to 0.01.

    Results:
      Campaign H200, 525s: 0.949300 mean BPB (3 seeds).
      Campaign H200, 300s: 0.960311 mean BPB (3 seeds).

    """
    cfg = exp006()
    cfg.experiment_name = "exp007"
    embedding = cfg.step.model.embedding
    assert isinstance(embedding, NgramEmbedding.Config)
    context = embedding.contexts["bigram"]
    assert isinstance(context, NgramEmbedding.Config)
    context.multipliers = (1, 1_000_003)
    optimizer = cfg.step.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    value = optimizer.optimizers[2]
    token = optimizer.optimizers[1]
    assert isinstance(value, PartialConfig)
    assert isinstance(token, PartialConfig)
    value.weight_decay = 0.01
    context = cast(PartialConfig[torch.optim.Optimizer], token.copy_tree())
    context.weight_decay = 0.01
    optimizer.select[1] = matching("embed.inner")
    optimizer.select.append(matching("embed.contexts.bigram.inner"))
    optimizer.optimizers.append(context)
    trigram = embedding.contexts["trigram"] = NgramEmbedding.Config()
    trigram.multipliers = (1, 257, 66_049)
    trigram.scale = 0.125
    trigram.num_embeddings = 1_048_576
    trigram.inner = Embedding.Config(init_weight=torch.nn.init.zeros_)
    optimizer.select[6] = matching(
        "embed.contexts.bigram.inner", "embed.contexts.trigram.inner"
    )
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp008() -> NgramTrainLoop.Config:
    """Fork exp007 with the following changes.

    - Double trigram capacity to 2,097,152 rows.
    - Split bigram and trigram optimizer groups; set distinct learning rates.
    - Increase bigram/trigram scales to 0.35/0.175.
    - Materialize per-layer blocks with FFN expansion -1.
    - Increase the RoPE base to 50,000.
    - Set head learning rate to 0.0045 before multiplying Adam rates by 1.32.
    - Set schedule flat/final fractions to 0.4/0.05.
    - Extend momentum warmup to 200 steps.

    Results:
      Campaign H200, 525s: 0.938595 mean BPB (3 seeds).
      Campaign H200, 300s: 0.956233 mean BPB (3 seeds).

    """
    cfg = exp007()
    cfg.experiment_name = "exp008"
    embedding = cfg.step.model.embedding
    assert isinstance(embedding, NgramEmbedding.Config)
    embedding.contexts["trigram"].num_embeddings = 2_097_152
    optimizer = cfg.step.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    context = optimizer.optimizers[6]
    assert isinstance(context, PartialConfig)
    context.lr = 0.8
    trigram = cast(PartialConfig[torch.optim.Optimizer], context.copy_tree())
    trigram.lr = 1.2
    optimizer.optimizers.append(trigram)
    optimizer.select[6] = matching("embed.contexts.bigram.inner")
    optimizer.select.append(matching("embed.contexts.trigram.inner"))
    template = cfg.step.model.template
    blocks: list[priml.model.transformer.block.TransformerBlock.Config] = []
    for _ in range(cfg.step.model.num_layers):
        block = template.copy_tree()
        assert isinstance(block.ffn, SwiGLUReluSquared.Config)
        block.ffn.expansion = -1
        blocks.append(block)
    cfg.step.model.block = blocks
    context = embedding.contexts["bigram"]
    triple = embedding.contexts["trigram"]
    assert isinstance(context, NgramEmbedding.Config)
    assert isinstance(triple, NgramEmbedding.Config)
    context.scale = 0.35
    triple.scale = 0.175
    cfg.step.model.rope.frequencies = HuggingFaceFrequencies.Config(base=50_000.0)
    head = optimizer.optimizers[0]
    assert isinstance(head, PartialConfig)
    head.lr = 0.0045
    for member in optimizer.optimizers:
        if isinstance(member, PartialConfig):
            member.lr *= 1.2
            member.lr *= 1.1
    assert isinstance(cfg.step.schedule, PartialConfig)
    schedule = cast(PartialConfig[Callable[[float], float]], cfg.step.schedule)
    schedule.flat = 0.4
    schedule.final = 0.05
    cfg.step.momentum_warmup_steps = 200
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp009() -> NgramTrainLoop.Config:
    """Fork exp008 with the following changes.

    - Increase depth from five layers to eight.
    - Use native BF16 weights, meta initialization, and no autocast.
    - Use 512-token windows; layers 3 and 7 attend globally.
    - Reduce microbatches to 72 rows; update once per microbatch.
    - Compile static full graphs with autotuning; disable CUDA graphs.
    - Disable checkpointing.

    Results:
      Campaign H200, 525s: 0.934975 mean BPB (3 seeds).
      Campaign H200, 300s: 0.962389 mean BPB (3 seeds).

    """
    cfg = exp008()
    cfg.experiment_name = "exp009"
    cfg.checkpointing = None
    model = cfg.step.model
    assert isinstance(model, MemoryNanoChatLM.Config)
    model.num_layers = 8
    model.dtype = torch.bfloat16
    assert isinstance(model.block, list)
    template = model.block[0]
    blocks: list[priml.model.transformer.block.TransformerBlock.Config] = []
    for layer in range(model.num_layers):
        block = template.copy_tree()
        assert isinstance(block, priml.model.transformer.block.TransformerBlock.Config)
        assert isinstance(block.attn, ValueGatedAttention.Config)
        block.attn.window = model.max_seq_len if layer in (3, 7) else 512
        blocks.append(block)
    model.block = blocks
    cfg.step.model = model
    cfg.step.device_init = "meta"
    cfg.step.dtype_autocast = None
    cfg.step.rows_per_pass = 72
    cfg.step.tokens_per_optimizer_step = cfg.step.rows_per_pass * model.max_seq_len
    cfg.step.compile = PartialConfig(
        torch.compile, dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs"
    )
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp010() -> NgramTrainLoop.Config:
    """Fork exp009 with the following changes.

    - Increase the RoPE base to 3,000,000.
    - Use a scaled readout with output cap 15.
    - Normalize attention and FFN outputs.
    - Use squared ReLU with threshold 0.75 and FFN expansion 4.
    - Use dtype-dependent RMSNorm epsilon throughout the blocks.

    Results:
      Campaign H200, 525s: 0.934865 mean BPB (3 seeds).
      Campaign H200, 300s: 0.961917 mean BPB (3 seeds).

    """
    cfg = exp009()
    cfg.experiment_name = "exp010"
    model = cfg.step.model
    model.rope.frequencies = HuggingFaceFrequencies.Config(base=3_000_000)
    assert isinstance(model.lm_head, softcap.SoftCap.Config)
    model.lm_head = ScaledSoftCap.Config().update(model.lm_head)
    model.lm_head.output_cap = 15
    assert isinstance(model.block, list)
    for block in model.block:
        assert isinstance(block, priml.model.transformer.block.TransformerBlock.Config)
        attention = CausalAttention.Config().update(block.attn)
        attention.norm_qk = RMSNorm.Config(eps=None)
        attention.norm_out = RMSNorm.Config(eps=None)
        block.attn = attention
        ffn = OutputNormFeedForward.Config().update(block.ffn)
        ffn.expansion = 4
        ffn.act = partial(thresholded_relu_squared, threshold=0.75)
        ffn.norm_out = RMSNorm.Config(eps=None)
        block.ffn = ffn
        for norm in (block.norm1, block.norm2):
            assert isinstance(norm, RMSNorm.Config)
            norm.eps = None
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp011() -> NgramTrainLoop.Config:
    """Fork exp010 with the following changes.

    - Add gated residual mixing and attention-head gates.
    - Pool the last two layers with learned weights.
    - Use 16 input channels for attention gates; zero-initialize head gates.
    - Route residual gates into the skip optimizer group.
    - Add pooling optimization at 0.15 times the skip learning rate.

    Results:
      Campaign H200, 525s: 0.927435 mean BPB (3 seeds).
      Campaign H200, 300s: 0.955083 mean BPB (3 seeds).

    """
    cfg = exp010()
    cfg.experiment_name = "exp011"
    model = cfg.step.model
    assert isinstance(model, MemoryNanoChatLM.Config)
    model.mix = GatedResidualMix.Config()
    model.num_pool_layers = 2
    assert isinstance(model.block, list)
    for block in model.block:
        assert isinstance(block.attn, CausalAttention.Config)
        block.attn.gate_channels = 16
        gate = block.attn.head_gate = Linear.Config()
        gate.bias = False
        gate.init_weight = torch.nn.init.zeros_
    optimizer = cfg.step.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    optimizer.select[4] = matching("mix.original", "mix.gate_scales")
    skip = optimizer.optimizers[4]
    assert isinstance(skip, PartialConfig)
    pooling = cast(PartialConfig[torch.optim.Optimizer], skip.copy_tree())
    pooling.lr *= 0.15
    optimizer.optimizers.append(pooling)
    optimizer.select.append(matching("pool_weights"))
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp012() -> NgramTrainLoop.Config:
    """Fork exp011 with the following changes.

    - Expand training to fourteen shards, excluding validation shard seven.
    - Select the train14 dataset directory.

    Results:
      Campaign H200, 525s: 0.927194 mean BPB (3 seeds).
      Campaign H200, 300s: 0.955053 mean BPB (3 seeds).

    """
    cfg = exp011()
    cfg.experiment_name = "exp012"
    cfg.dataset.working_dir = "/datasets/nanochat/train14"
    cfg.dataset.train_shard_indices = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp013() -> NgramTrainLoop.Config:
    """Fork exp012 with the following changes.

    - Replace input n-gram embeddings with per-layer attention-value memories.
    - Add bigram memories at layers 1, 3, 5, and 7.
    - Add trigram memories at layers 1, 5, and 7.
    - Use two hashes per table, generated on CPU with Torch seed 0.
    - Size each table to 128 times the vocabulary.
    - Rebuild optimizer groups for memories, pooling, residuals, and block matrices.
    - Use RMSProp for memories and FFN-scaled NorMuon for block matrices.
    - Add learning-rate, momentum, beta, and memory-ramp schedules.
    - Add three weight-decay pulses; extend momentum warmup to 300 steps.

    Results:
      Campaign H200, 525s: 0.910094 mean BPB (3 seeds).
      Campaign H200, 300s: 0.946772 mean BPB (3 seeds).

    """
    cfg = exp012()
    cfg.experiment_name = "exp013"
    model = cfg.step.model
    assert isinstance(model, MemoryNanoChatLM.Config)
    model.embedding = NarrowEmbedding.Config().update(
        model.embedding, skip_missing=True
    )
    # Pin both generator and draws to CPU: device contexts must not change hashes.
    rng = torch.Generator(device="cpu").manual_seed(0)
    for tables, order, layers in (
        (model.bigrams, 2, (1, 3, 5, 7)),
        (model.trigrams, 3, (1, 5, 7)),
    ):
        for layer in layers:
            table = tables[str(layer)] = HashedNgramTables.Config()
            table.num_embeddings = model.vocab_size * 128
            table.hash_multipliers = tuple(
                tuple(
                    2
                    * int(
                        torch.randint(
                            table.num_embeddings // 2,
                            (),
                            generator=rng,
                            device="cpu",
                            dtype=torch.int64,
                        )
                    )
                    + 1
                    for _ in range(order)
                )
                for _ in range(2)
            )
    assert isinstance(model.block, list)
    for layer, block in enumerate(model.block):
        assert isinstance(block.attn, CausalAttention.Config)
        block.attn.bigram = str(layer) in model.bigrams
        block.attn.trigram = str(layer) in model.trigrams
    step = cfg.step
    assert isinstance(step, NgramTrainStep.Config)
    step.optimizer_update = ScheduledOptimizerUpdate.Config()
    step.optimizer_update.optimizer_recompile_limit = 32
    step.schedule = PartialConfig(trapezoidal, flat=0.5)
    step.momentum_warmup_steps = 300
    step.optimizer_update.muon_warmdown = 0.95
    step.optimizer_update.adam_warmdown = 0.6
    step.optimizer_update.final_lr_fraction = 0.025
    step.optimizer_update.momentum_final = 0.85
    step.optimizer_update.muon_beta2_final = 0.98
    step.optimizer_update.adam_beta1_final = 0.4
    step.optimizer_update.ngram_beta2_final = 0.99995
    step.optimizer_update.ngram_ramp_fraction = 1.0
    step.optimizer_update.skip_member = 4
    step.optimizer_update.adam_beta1_members = (0, 1, 2)
    step.optimizer_update.weight_decay_pulses = (
        WeightDecayPulse(center=0.015, half_width=0.005, multiplier=3.0),
        WeightDecayPulse(center=0.03, half_width=0.01, multiplier=5.0),
        WeightDecayPulse(center=0.80, half_width=0.025, multiplier=6, triangular=True),
    )
    model = step.model
    optimizer = CompositeOptimizer.Config()
    optimizer.select = [
        matching(name)
        for name in (
            "lm_head",
            "embed.inner",
            "value_embeds",
            "mix.running",
            "mix.original",
            "bigrams",
            "trigrams",
            "pool_weights",
            "blocks",
        )
    ]
    optimizer.select[4] = matching("mix.original", "mix.gate_scales")
    for rate, scaled, betas, decay in (
        (0.004, True, (0.8, 0.95), 0.0),
        (0.6, True, (0.8, 0.95), 0.0),
        (0.6, True, (0.8, 0.95), 0.0),
        (0.008, False, (0.8, 0.95), 0.0),
        (1.2, False, (0.96, 0.95), 0.002),
    ):
        optimizer.optimizers.append(
            PartialConfig(
                FusedAdamW,
                lr=rate,
                betas=betas,
                eps=1e-10,
                weight_decay=decay,
                width_scaled=scaled,
            )
        )
    ngram_rate = 0.6 / (model.channels_in / 768) ** 0.5
    optimizer.optimizers.extend(
        [
            BiasCorrectedRMSProp.Config(
                lr=ngram_rate, beta2=0.999, eps=1e-10, compile=True
            ),
            BiasCorrectedRMSProp.Config(
                lr=ngram_rate, beta2=0.999, eps=1e-10, compile=True
            ),
            FusedAdamW.Config(lr=0.06, betas=(0.96, 0.95), eps=1e-10),
        ]
    )
    matrices = FFNScaledNorMuon.Config(channels_in=model.channels_in)
    matrices.optimizer.weight_decay = 0.1
    optimizer.optimizers.append(matrices)
    step.optimizer = optimizer
    cfg.step = step
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp014() -> NgramTrainLoop.Config:
    """Fork exp013 with the following changes.

    - Set per-layer FFN expansions to (2, 2, 3, 3, 5, 5, 6, 6).

    Results:
      Campaign H200, 525s: 0.907300 mean BPB (3 seeds).
      Campaign H200, 300s: 0.943577 mean BPB (3 seeds).

    """
    cfg = exp013()
    cfg.experiment_name = "exp014"
    model = cfg.step.model
    assert isinstance(model.block, list)
    for block, expansion in zip(model.block, (2, 2, 3, 3, 5, 5, 6, 6), strict=True):
        assert isinstance(block, priml.model.transformer.block.TransformerBlock.Config)
        assert isinstance(block.ffn, OutputNormFeedForward.Config)
        block.ffn.expansion = expansion
    # Inherited FFNScaledNorMuon now applies sqrt(4 / expansion) to FFN inputs.
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp015() -> NgramTrainLoop.Config:
    """Fork exp014 with the following changes.

    - Reduce width from 768 to 640.
    - Reduce microbatches to 64 rows; update once per microbatch.
    - Shorten local attention windows from 512 to 256 tokens.
    - Use row-wise RMSProp statistics and adjust memory learning rates for width.
    - Apply a 1.25 FFN learning-rate multiplier.
    - Reduce the training buffer to 256 documents.

    Results:
      Campaign H200, 525s: 0.906465 mean BPB (3 seeds).
      Campaign H200, 300s: 0.935564 mean BPB (3 seeds).

    """
    cfg = exp014()
    cfg.experiment_name = "exp015"
    model = cfg.step.model
    model.channels_in = 640
    cfg.step.rows_per_pass = 64
    cfg.step.tokens_per_optimizer_step = cfg.step.rows_per_pass * model.max_seq_len
    assert isinstance(model.block, list)
    for layer, block in enumerate(model.block):
        assert isinstance(block.attn, CausalAttention.Config)
        block.attn.window = model.max_seq_len if layer in (3, 7) else 256
    optimizer = cfg.step.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    for index in (5, 6):
        member = optimizer.optimizers[index]
        assert isinstance(member, BiasCorrectedRMSProp.Config)
        member.lr = 0.6 / (model.channels_in / 768) ** 0.5
        member.rowwise = True
    matrices = optimizer.optimizers[8]
    assert isinstance(matrices, FFNScaledNorMuon.Config)
    matrices.channels_in = model.channels_in
    matrices.ffn_lr_multiplier = 1.25
    cfg.dataset.train_buffer_size = 256
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp016() -> NgramTrainLoop.Config:
    """Fork exp015 with the following changes.

    - Replace FlashAttention-3 with FlashAttention-4.
    - Enable CUDA graphs with max-autotune compilation.

    Results:
      Campaign H200, 525s: 0.906208 mean BPB (3 seeds).
      Campaign H200, 300s: 0.935105 mean BPB (3 seeds).

    """
    cfg = exp015()
    cfg.experiment_name = "exp016"
    assert isinstance(cfg.step.compile, PartialConfig)
    compiler = cast(PartialConfig[Callable[..., object]], cfg.step.compile)
    compiler.mode = "max-autotune"
    assert isinstance(cfg.step.model.block, list)
    for block in cfg.step.model.block:
        assert isinstance(block.attn, CausalAttention.Config)
        block.attn.kernel = Flash4Attention.Config()
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp017() -> NgramTrainLoop.Config:
    """Fork exp016 with the following changes.

    - Fuse n-gram accumulation.
    - Fuse query/key normalization and rotary embeddings.

    Results:
      Campaign H200, 525s: 0.903610 mean BPB (3 seeds).
      Campaign H200, 300s: 0.931969 mean BPB (3 seeds).

    """
    cfg = exp016()
    cfg.experiment_name = "exp017"
    model = cfg.step.model
    assert isinstance(model, MemoryNanoChatLM.Config)
    model.fused_ngram = True
    assert isinstance(model.block, list)
    for block in model.block:
        assert isinstance(block.attn, CausalAttention.Config)
        block.attn.fused_qk_rope = True
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp018() -> NgramTrainLoop.Config:
    """Fork exp017 with the following changes.

    - Increase width from 640 to 768.
    - Increase microbatches to 96 rows; update once per microbatch.
    - Adjust memory learning rates and FFN scaling for the wider model.
    - Apply batch learning-rate scaling and its correction in the recorded order.

    Results:
      Campaign H200, 525s: 0.900618 mean BPB (3 seeds).
      Campaign H200, 300s: 0.936150 mean BPB (3 seeds).

    """
    cfg = exp017()
    cfg.experiment_name = "exp018"
    model = cfg.step.model
    assert isinstance(model, MemoryNanoChatLM.Config)
    previous_width = model.channels_in
    model.channels_in = 768
    batch_scale = (96 / cfg.step.rows_per_pass) ** 0.5
    batch_correction = (cfg.step.rows_per_pass / 96) ** 0.5
    cfg.step.rows_per_pass = 96
    cfg.step.tokens_per_optimizer_step = cfg.step.rows_per_pass * model.max_seq_len
    optimizer = cfg.step.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    for index in (0, 1, 2):
        member = optimizer.optimizers[index]
        assert isinstance(member, PartialConfig)
        member.lr *= batch_scale
    for index in (5, 6):
        member = optimizer.optimizers[index]
        assert isinstance(member, BiasCorrectedRMSProp.Config)
        member.lr *= (previous_width / model.channels_in) ** 0.5 * batch_scale
    # Preserve multiplication order: folding these factors changes float LR bits.
    for index in (0, 1, 2, 5, 6):
        member = optimizer.optimizers[index]
        assert isinstance(member, (PartialConfig, BiasCorrectedRMSProp.Config))
        member.lr *= batch_correction
    matrices = optimizer.optimizers[8]
    assert isinstance(matrices, FFNScaledNorMuon.Config)
    matrices.channels_in = model.channels_in
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp019() -> NgramTrainLoop.Config:
    """Fork exp018 with the following changes.

    - Reuse layer-4 attention inputs in layers 5, 6, and 7.
    - Select the dataset with reordered training documents.

    Results:
      Campaign H200, 525s: 0.897725 mean BPB (3 seeds).
      Campaign H200, 300s: 0.932429 mean BPB (3 seeds).

    """
    cfg = exp018()
    cfg.experiment_name = "exp019"
    model = cfg.step.model
    assert isinstance(model, MemoryNanoChatLM.Config)
    model.attention_source_layers = (5, 6, 7)
    model.attention_source_after_layer = 4
    assert isinstance(model.block, list)
    model.block = [
        SourceReuseTransformerBlock.Config().update(block) for block in model.block
    ]
    cfg.dataset.working_dir = "/datasets/nanochat/augmented"
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp020() -> NgramTrainLoop.Config:
    """Fork exp019 with the following changes.

    - Switch to a 16,384-token Unigram vocabulary and prepared input manifests.
    - Scale memory-table capacity with the vocabulary.
    - Evaluate on byte-matched reference rows.
    - Restrict compiled matrix multiplication to ATen; retain CUDA graphs.

    Results:
      Campaign H200, 525s: 0.894827 mean BPB (3 seeds).
      Campaign H200, 300s: 0.930738 mean BPB (3 seeds).

    """
    cfg = exp019()
    cfg.experiment_name = "exp020"
    model = cfg.step.model
    assert isinstance(model, MemoryNanoChatLM.Config)
    previous_vocab_size = model.vocab_size
    model.vocab_size = 16_384
    # Torch 2.11/Triton 3.6 overflows large GEMM indexing on H200: the
    # backward can illegal-access; a 128-row evaluation can silently return
    # zero logits. Retain max-autotune's other options but use ATen GEMMs.
    cfg.step.compile = PartialConfig(
        torch.compile,
        dynamic=False,
        fullgraph=True,
        options={
            "max_autotune": True,
            "triton.cudagraphs": True,
            "coordinate_descent_tuning": True,
            "max_autotune_gemm_backends": "ATEN",
        },
    )
    for table in (*model.bigrams.values(), *model.trigrams.values()):
        table.num_embeddings = (
            table.num_embeddings * model.vocab_size // previous_vocab_size
        )
    cfg.dataset.prepared_train_manifest = (
        "/datasets/nanochat/unigram16k/prepared/train/PREPARED_MANIFEST.json"
    )
    cfg.dataset.prepared_eval_manifest = (
        "/datasets/nanochat/unigram16k/prepared/eval/PACKED_EVAL_MANIFEST.json"
    )
    cfg.dataset.reference_evaluation = ReferenceEvaluation.Config()
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp021() -> NgramTrainLoop.Config:
    """Fork exp020 with the following changes.

    - Clear only touched n-gram gradient rows.
    - Use sparse-row RMSProp updates.
    - Use bounded-logit cross-entropy with BF16 loss computation.

    Results:
      Campaign H200, 525s: 0.888283 mean BPB (3 seeds).
      Campaign H200, 300s: 0.923441 mean BPB (3 seeds).

    """
    cfg = exp020()
    cfg.experiment_name = "exp021"
    model = cfg.step.model
    assert isinstance(model, MemoryNanoChatLM.Config)
    model.ngram_dirty_clear = True
    assert isinstance(model.lm_head, ScaledSoftCap.Config)
    loss = cfg.step.loss = BoundedTokenCrossEntropy.Config()
    loss.logit_upper_bound = model.lm_head.output_cap
    loss.dtype = torch.bfloat16
    optimizer = cfg.step.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    for member in optimizer.optimizers:
        if isinstance(member, BiasCorrectedRMSProp.Config):
            member.sparse_rows = True
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp022() -> NgramTrainLoop.Config:
    """Fork exp021 with the following changes.

    - Zero-initialize memory tables after consuming their original initialization draws.

    Results:
      Campaign H200, 525s: 0.887457 mean BPB (10 seeds).
      Campaign H200, 300s: 0.922382 mean BPB (10 seeds).
      Campaign B200, 300s: 0.887791 mean BPB (10 seeds).
      Standalone Priml H200, 300s: 0.923656 mean BPB (3 seeds).

    """
    cfg = exp021()
    cfg.experiment_name = "exp022"
    model = cfg.step.model
    assert isinstance(model, MemoryNanoChatLM.Config)
    # Widening the coefficient range changes the mapping, even with the same seed.
    hash_capacity = model.vocab_size * 64
    rng = torch.Generator(device="cpu").manual_seed(0)
    for table in (*model.bigrams.values(), *model.trigrams.values()):
        table.hash_multipliers = tuple(
            tuple(
                2
                * int(
                    torch.randint(
                        hash_capacity // 2,
                        (),
                        generator=rng,
                        device="cpu",
                        dtype=torch.int64,
                    )
                )
                + 1
                for _ in row
            )
            for row in table.hash_multipliers
        )
    for tables in (model.bigrams, model.trigrams):
        for table in tables.values():
            table.init_after = torch.nn.init.zeros_
    # Uncomment for B200's 300-second budget; leave commented for H-series.
    # cfg.max_time = cfg.step.train_budget_sec = 300.0 # Seconds.
    return cfg


def exp_smoke() -> NanoChatLoop.Config:
    """Fork exp001 for an unscored installation check.

    - Automatically select the available device.
    - Reduce vocabulary/width/depth/context to 16/32/2/8.
    - Reduce attention-head width to 16 and microbatches to two rows.
    - Use 32 tokens per optimizer step; disable compilation and warmup.
    - Stop after four steps or ten seconds.
    - Evaluate two batches of four rows.
    """
    cfg = exp001()
    cfg.experiment_name = "exp_smoke"
    # A smoke run validates the installation it is executed on. Scored rungs
    # pin CUDA so a scheduler mistake cannot silently produce a CPU result;
    # this unscored rung keeps the runtime's documented best-device fallback.
    parallelism = cfg.step.parallelism
    assert isinstance(parallelism, NoParallel.Config)
    parallelism.device = None
    cfg.dataset.device = "auto"
    runtime = cfg.runtime
    assert isinstance(runtime, SingleProcess.Config)
    runtime.device = "auto"
    cfg.step.model.vocab_size = 16
    # The value gate reads a fixed 32 channels_in of its input, so a model
    # narrower than 32 has a gate of a different shape than the reference's.
    cfg.step.model.channels_in = 32
    cfg.step.model.num_layers = 2
    cfg.step.model.max_seq_len = 8
    # Half the width, so there are two heads: at one head every head-axis
    # reshape is the identity and a split on the wrong axis still agrees.
    attention = cfg.step.model.template.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    attention.channels_head = 16
    cfg.step.compile = None
    cfg.step.rows_per_pass = 2
    cfg.step.tokens_per_optimizer_step = 32
    cfg.step.budget_warmup_steps = 0
    cfg.step.train_budget_sec = 10.0
    cfg.max_time = cfg.step.train_budget_sec
    cfg.max_steps = 4
    cfg.num_steps_eval = 2
    # Both, together: the scored token count must be a whole number of eval
    # batches, and the recipe's 20.97M tokens is a full evaluation rather than
    # the two batches a smoke run wants.
    cfg.dataset.eval_batch_size = 4
    cfg.dataset.eval_tokens = 2 * 4 * cfg.step.model.max_seq_len
    return cfg
