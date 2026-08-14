"""The nanochat experiment ladder.

Every experiment trains to the same wall-clock BUDGET and is scored on the same
validation stream, so a change is measured by what it achieves in fixed time
rather than in fixed steps. A change that makes a step cheaper is rewarded with
more steps; one that makes a step better is rewarded directly. Both show up in
one number.

``exp000`` is the naive recipe and is never edited; improvements are forks, so
a number measured against it stays comparable::

    exp000  transformer + NorMuon/AdamW, 5-minute budget
      +-- exp001  windowed attention on three layers in four
            +-- exp002  value embeddings on alternating layers

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp000
"""

from __future__ import annotations

from dataclasses import field
from typing import Self, override

from configgle import Makes

from priml.baselines.nanochat.data import NanoChatData
from priml.baselines.nanochat.metric import BitsPerByte
from priml.baselines.nanochat.model import ValueGatedAttention
from priml.baselines.nanochat.train_step import NanoChatTrainStep
from priml.runtime import SingleProcess
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

        dataset: NanoChatData.Config = field(default_factory=NanoChatData.Config)
        """Prepared token rows, served from device memory."""

        @override
        def finalize(self) -> Self:
            # How many rows fit in one pass is the step's decision -- it
            # follows device memory -- so the dataset takes its batch size from
            # there rather than the two being set to agree by hand.
            self.dataset.batch_size = self.step.rows_per_pass
            # Geometry is declared ONCE, on the model, and pushed here so the
            # dataset can verify the prepared arrays against it at load. Two
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


def exp000() -> NanoChatLoop.Config:
    """Pre-norm transformer trained to a five-minute budget.

    The baseline every other experiment forks, and the only one stating a
    recipe rather than a change. Frozen: improvements belong in a fork, so a
    result measured against it stays comparable.

    Every layer attends over the full context and none reads a value
    embedding, so the two mechanisms the ladder tests are both OFF here --
    a plain transformer is the bar each has to clear.

    Hypothesis:
      A pre-norm transformer with rotary positions, a squared-ReLU
      feed-forward, and orthogonalized updates on the matrices is the
      strongest recipe using nothing exotic -- the bar windowed attention and
      value embeddings must clear to earn their complexity.

    References:
      https://arxiv.org/abs/2109.08668
        So et al. Primer: Searching for Efficient Transformer for Language
        Modeling.
      https://kellerjordan.github.io/posts/muon/
        Jordan et al. 2024. Muon: an optimizer for hidden layers.

    Results:
      TBD.

    """
    cfg = NanoChatLoop.Config()
    cfg.study_name = "nanochat"
    cfg.experiment_name = "exp000"

    cfg.step.model.channels = 512
    cfg.step.model.num_layers = 8
    cfg.step.model.window_pattern = "L"
    cfg.step.model.value_embedding_layers = []

    # The budget the schedules anneal over and the budget the loop stops on:
    # equal, or the learning rate lands short of zero or decays past the end.
    cfg.step.time_budget_sec = 300.0
    cfg.max_time = cfg.step.time_budget_sec
    cfg.max_time_kind = "train"

    cfg.metrics["val"] = BitsPerByte.Config()
    cfg.num_steps_eval = 200
    cfg.num_steps_log = 50
    # Every row is a full context by construction, so a pass over the data is
    # not a meaningful boundary -- the budget is what ends the run.
    cfg.eval_every_epoch = False
    cfg.runtime = SingleProcess.Config()
    return cfg


def exp001() -> NanoChatLoop.Config:
    """exp000 with three layers in four attending over half the context.

    Hypothesis:
      Most layers of a language model resolve local structure, so restricting
      their attention to recent history should cost little accuracy while
      making each step cheaper. Under a fixed budget the saved time becomes
      more steps, so this wins if the accuracy it gives up is smaller than the
      accuracy those extra steps buy.

    References:
      https://arxiv.org/abs/2004.05150
        Beltagy et al. Longformer: The Long-Document Transformer.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.step.model.window_pattern = "SSSL"
    return cfg


def exp002() -> NanoChatLoop.Config:
    """exp001 plus a gated value embedding on alternating layers.

    Hypothesis:
      A deep stack's residual stream is increasingly processed, so a layer
      wanting the raw token identity must reconstruct it. Letting alternating
      layers read a dedicated embedding of the input tokens -- admitted
      through a per-head gate, so a head can decline it -- supplies that
      directly. The gate initializes to pass the embedding through unchanged,
      so the model must learn to attenuate rather than to attend.

    References:
      https://arxiv.org/abs/2410.17897
        Zhou et al. Value Residual Learning.

    Results:
      TBD.

    """
    cfg = exp001()
    cfg.experiment_name = "exp002"
    # Alternating layers, counting BACK from the last so the deepest one gets
    # a table: that is where the residual stream is most processed and a path
    # to the raw tokens is worth the most.
    cfg.step.model.value_embedding_layers = sorted(
        range(cfg.step.model.num_layers - 1, -1, -2),
    )
    return cfg


def exp_smoke() -> NanoChatLoop.Config:
    """exp000 at minimum size, for verifying an installation end to end.

    Not a result. It answers one question -- is the data prepared and does the
    loop run -- so every axis costing time without bearing on that answer is
    cut: a few seconds of budget, and a network narrow enough to finish in
    them. The score will be poor, which is expected.
    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.step.model.channels = 64
    cfg.step.model.num_layers = 2
    cfg.step.model.max_seq_len = 128
    # The head width is not derived from the model width, so narrowing one
    # without the other leaves a model that cannot be built at all.
    attention = cfg.step.model.block.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    attention.channels_head = 32
    cfg.step.compile = False
    cfg.step.rows_per_pass = 2
    cfg.step.tokens_per_optimizer_step = 256
    cfg.step.budget_warmup_steps = 0
    cfg.step.time_budget_sec = 10.0
    cfg.max_time = cfg.step.time_budget_sec
    cfg.max_steps = 4
    cfg.num_steps_eval = 2
    cfg.dataset.num_eval_rows = 4
    return cfg
