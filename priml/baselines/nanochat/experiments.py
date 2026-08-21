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

    uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data
    time CUDA_VISIBLE_DEVICES=1 uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp001 --override checkpointing.resume=False
"""

from __future__ import annotations

from dataclasses import field
from typing import Self, override

from configgle import Makes, PartialConfig

import torch

from priml.baselines.nanochat.data import NanoChatData
from priml.baselines.nanochat.flash3 import Flash3Attention
from priml.baselines.nanochat.train_step import NanoChatTrainStep
from priml.metrics.bits_per_byte import BitsPerByte
from priml.model.narrow_embedding import NarrowEmbedding
from priml.model.norm import RMSNorm
from priml.model.value_gated_attention import (
    ValueGatedAttention,
    sdpa_attention,
)
from priml.runtime import SingleProcess
from priml.train.parallelism import NoParallel
from priml.train.tracker import AsyncTracker, TrackerList, WandbTracker
from priml.train.train_loop import TrainLoop


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

    @override
    def _on_batch_ready(self, fetch_time: float) -> None:
        """Charge loading to the budget, as the reference charges it.

        Its ``next(train_loader)`` runs inside the region its clock brackets
        (train.py:550, between 543 and 573). Ours happens here, outside the
        step, so the budget would otherwise buy free steps: measured at 0.160
        of 1.683 s/step on a 5090, a tenth of the run.
        """
        step = self.step
        assert isinstance(step, NanoChatTrainStep)
        step.charge_budget(fetch_time)


def exp000() -> NanoChatLoop.Config:
    """The published five-minute recipe, reproduced on its own kernel.

    Frozen, and the only rung that reproduces rather than proposes. It pins
    FlashAttention-3 because that is what the reference measured on: a fused
    kernel reduces in a different order than a masked
    ``scaled_dot_product_attention``, so a run that matches every
    hyperparameter and swaps the kernel produces a different number and cannot
    settle whether the port is faithful.

    That pin carries a hardware requirement -- FA3 builds for SM90, so this
    rung runs on Hopper and refuses elsewhere. :func:`exp001` is the same
    recipe with the backend resolved from the device; fork that one.

    Hypothesis:
      Porting the reference's architecture, optimizer partition, data
      protocol, and kernel into this package reproduces its published score,
      so the port is faithful and every fork below measures a real change
      rather than an artifact of the port.

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
      TBD.

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
    cfg.step.parallelism = NoParallel.Config()
    cfg.step.parallelism.device = "cuda"
    cfg.dataset.device = "cuda"
    cfg.runtime = SingleProcess.Config()
    cfg.runtime.device = "cuda"
    # TF32 matmuls. The reference recipe's throughput assumes them, and a
    # run left at torch's default reduces in a different order, so a score
    # measured here would not be comparable to one measured there.
    cfg.runtime.float32_matmul_precision = "high"

    return cfg


def exp001() -> NanoChatLoop.Config:
    """exp000 with the attention kernel resolved from the device.

    The one deviation is the backend. ``exp000`` pins FlashAttention-3, which
    builds only for SM90, so it refuses to construct anywhere else; this rung
    takes the CUDA backend torch dispatches for the allocated GPU and therefore
    runs across supported CUDA architectures. Fork THIS for ordinary work -- a
    rung that reproduces a published number has to refuse a GPU that cannot
    reproduce it, which is the wrong behaviour for everything except that one
    job.

    ``exp000`` is the faithful reproduction and is untestable off a Hopper
    card, so this rung is where the reproduction is actually PROVEN: the
    reference implementation is given this same portable kernel and stepped
    beside it, and ``scripts/karpathy_parity.py`` reports whether every
    parameter and gradient agrees. The kernel is then common to both sides,
    so what the comparison measures is the recipe rather than the backend.

    Operationally, this portable rung also reports to W&B through the shared
    asynchronous tracker wrapper. Reporting is not a scientific treatment.

    Hypothesis:
      The recipe's score comes from its architecture and its optimizer rather
      than from its kernel, so the portable backend reproduces exp000 to
      within run-to-run noise.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"

    # The following are tuned for rtx5090.
    block = cfg.step.model.template
    assert isinstance(block.attn, ValueGatedAttention.Config)
    block.attn.kernel = PartialConfig(sdpa_attention)

    dashboard = WandbTracker.Config()
    dashboard.project = "nanochat"
    wrapper = AsyncTracker.Config()
    wrapper.tracker = dashboard
    cfg.tracker = TrackerList.Config()
    cfg.tracker.trackers = {
        "wandb": wrapper,
    }

    # Karpathy uses 32 but we can get 15.7% more speed @ 64.
    # Theoretically up to 76.
    # cfg.step.rows_per_pass = 64
    # Karpathy uses 128 but we can actually fit 512.
    # cfg.dataset.eval_batch_size = 512
    # However it doesn't seem to affect performance...dataloader bottleneck?

    return cfg


def exp002() -> NanoChatLoop.Config:
    """exp001 without the value embeddings.

    Hypothesis:
      A deep stack's residual stream is increasingly processed, so a layer
      wanting the raw token identity must reconstruct it; alternating layers
      read a dedicated embedding of the input tokens through a per-head gate
      to supply that directly. Removing it costs accuracy, and the drop is
      what those four tables are worth. It also returns their parameters and
      their gate, so a cheaper step buys more steps under the same budget --
      this rung wins if the accuracy given up is smaller than the accuracy
      those extra steps buy.

    References:
      https://arxiv.org/abs/2410.17897
        Zhou et al. Value Residual Learning.

    Results:
      TBD.

    """
    cfg = exp001()
    cfg.experiment_name = "exp002"
    cfg.step.model.value_embedding_stride = 0
    return cfg


def exp003() -> NanoChatLoop.Config:
    """exp002 with every layer attending over the full context.

    Both mechanisms are now off, so this is the plain transformer -- the floor
    the other rungs are priced against.

    Hypothesis:
      Most layers of a language model resolve local structure, so restricting
      their attention to recent history costs little accuracy while making
      each step cheaper. Restoring the full context therefore buys back that
      accuracy at a higher price per step, and under a fixed budget it loses
      if the steps it gives up were worth more than the attention it regains.

    References:
      https://arxiv.org/abs/2004.05150
        Beltagy et al. Longformer: The Long-Document Transformer.

    Results:
      TBD.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    window_attn = cfg.step.model.template.attn
    assert isinstance(window_attn, ValueGatedAttention.Config)
    window_attn.window_pattern = "L"
    return cfg


def exp_smoke() -> NanoChatLoop.Config:
    """exp001 at minimum size, for verifying an installation end to end.

    Forks the portable rung rather than ``exp000``: a smoke test that refused
    to run wherever FlashAttention-3 is unavailable would fail on exactly the
    machines it exists to check.

    Not a result. It answers one question -- is the data prepared and does the
    loop run -- so every axis costing time without bearing on that answer is
    cut: a few seconds of budget, and a network narrow enough to finish in
    them. The score will be poor, which is expected.

    Every delta from ``exp001`` is a size or a budget, so the goldens minted
    over this rung guard the real one: a change to the architecture, the
    optimizer partition, the schedules, or the precision reaches this config
    too.
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
