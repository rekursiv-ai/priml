"""Released inference (`exp000`) and independent cold training (`train_exp000`)."""

from __future__ import annotations

from configgle import PartialConfig
from torch import nn

import torch

from priml.baselines.rsna_knee.imaging import (
    anatomical_mirror,
    unchanged,
)
from priml.baselines.rsna_knee.inference import (
    PublicRecipe,
    RankGroup,
    RankMember,
    ReleasedState,
    accept_logits,
)
from priml.baselines.rsna_knee.model import ReleasedVNet
from priml.baselines.rsna_knee.train_loop import ColdRun, ProbabilityAuc
from priml.loss.simple_loss import SimpleLoss, bce_with_logits
from priml.model.coatnet import CoAtNet
from priml.model.resnet import ResNet, ResNetBlock, ResNetStage
from priml.model.special import Dropout2d
from priml.optimizers.composite import CompositeOptimizer
from priml.runtime import SingleProcess
from priml.train.tracker import FileTracker


def exp000() -> PublicRecipe.Config:
    """Run every released inference stage through independent native components.

    Hypothesis:
      The two Raptor checkpoints and four R7 folds, including original state
      and fluid stackers, provide the published comparison baseline.

    References:
      https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-957-new-one-you-can-use

    Results:
      Tiny-width CPU model goldens minted in-tree replay exactly.
      Complete learned native GPU inference remains unmeasured. A historical
      source-runtime run took 168.29 seconds on H200; it does not qualify this
      rewrite. Public score 0.957 is reported by the source; official test
      labels are unavailable locally.
      Original private weak targets and full checkpoint-training histories are
      unavailable, so this experiment reproduces released-checkpoint inference.

    Returns:
      config: The complete native inference pipeline, within the nine-hour session.

    """
    config = PublicRecipe.Config()
    config.study_name = "rsna-knee"
    config.experiment_name = "exp000"
    config.budget.seconds = 9 * 3_600
    config.budget.reserve_evaluation_sec = config.budget.seconds - 8.3 * 3_600

    config.raptor384.checkpoint.file.path = (
        "/datasets/rsna-knee-raptor-swa-v1/"
        "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
    )
    config.raptor384.checkpoint.file.sha256 = (
        "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
    )
    config.raptor384.checkpoint.key = "model"
    config.raptor384.checkpoint.weights_only = False
    assert isinstance(config.raptor384.model.backbone, CoAtNet.Config)
    config.raptor384.model.backbone.image_size = (320, 320)
    config.raptor384.model.chunk_size = 16
    config.raptor384.views = [unchanged, anatomical_mirror]
    config.raptor384.time_limit = 2 * 3_600
    config.raptor384.cudnn_benchmark = True
    config.raptor384.allow_tf32_matmul = True

    config.raptor224.checkpoint.file.path = (
        "/datasets/rsna-knee-raptor-swa-v1/"
        "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt"
    )
    config.raptor224.checkpoint.file.sha256 = (
        "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"
    )
    config.raptor224.checkpoint.key = "model"
    config.raptor224.checkpoint.weights_only = False
    assert isinstance(config.raptor224.model.backbone, CoAtNet.Config)
    config.raptor224.model.backbone.image_size = (224, 224)
    config.raptor224.windows.crop_size = 307
    config.raptor224.score.dtypes_autocast = [torch.float16, None]
    config.raptor224.score.check_logits = accept_logits
    config.raptor224.time_limit = 2 * 3_600
    config.raptor224.cudnn_benchmark = True
    config.raptor224.allow_tf32_matmul = True

    config.r7.checkpoints = [ReleasedState.Config() for _ in range(4)]
    for fold, (checkpoint, digest) in enumerate(
        zip(
            config.r7.checkpoints,
            (
                "ca8a76e024828793e0663e5a2a13461de17e820bed6789eafe3f155072d3612f",
                "fb432cf544f218608dbaecf7a8dd3c8b6a26b34a3dfe2dc198172c3e222d5910",
                "a9718ae111d6307492e724eb1767f0b5a77397d9a4d82c50c1e146b046523bf7",
                "073c22c71a9946c1457e37e56fec145f073a921da106b55a10884da01a40fc94",
            ),
            strict=True,
        ),
    ):
        checkpoint.file.path = f"/datasets/rsna-knee-r7-v2/E_lrprior_s0_f{fold}.pt"
        checkpoint.file.sha256 = digest
    config.r7.anatomy = anatomy = ReleasedVNet.Config(channels_in=1, channels_out=4)
    for stage in (*anatomy.down[2:], *anatomy.up[:2]):
        stage.dropout = Dropout2d.Config(p=0.0)
    config.r7.side = ResNet.Config(
        channels_in=1,
        channels_out=1,
        stage=ResNetStage.Config(
            block=ResNetBlock.Config(init_scale_residual=nn.init.zeros_),
        ),
    )
    config.r7.anatomy_checkpoint.file.path = (
        "/datasets/rsna-knee-orthofoundation-v1/anatomy.pt"
    )
    config.r7.anatomy_checkpoint.file.sha256 = (
        "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"
    )
    config.r7.anatomy_checkpoint.key = "state_dict"
    config.r7.side_checkpoint.file.path = "/datasets/rsna-knee-r7-v2/side_cls.pt"
    config.r7.side_checkpoint.file.sha256 = (
        "ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a"
    )
    config.r7.postprocess.fluid_file.path = (
        "/datasets/rsna-knee-r7-v2/fluid_stacker.json"
    )
    config.r7.postprocess.fluid_file.sha256 = (
        "be5a1664077300db1fdd0af8daaf44a3ebc41f91152e4f0ec124f805a7a51efb"
    )
    config.r7.postprocess.state_file.path = (
        "/datasets/rsna-knee-r7-v2/state_stacker.json"
    )
    config.r7.postprocess.state_file.sha256 = (
        "7b54a3f774dad3f842d68aaccbb03771806c26c0a3ca3a07fea8575a21f3ead3"
    )
    config.r7.postprocess.spec_file.path = "/datasets/rsna-knee-r7-v2/state_spec.json"
    config.r7.postprocess.spec_file.sha256 = (
        "2c976e70773b745cb357b992ba04946c157aa478e65d82f2883039c038b6ab3c"
    )
    config.r7.score.dtype_autocast = torch.bfloat16

    raptor, r7 = config.blend.groups = [RankGroup.Config(), RankGroup.Config()]
    raptor.weight = r7.weight = 0.5
    arm384, arm224 = raptor.members = [RankMember.Config(), RankMember.Config()]
    arm384.name, arm384.weight = "Raptor", 0.65
    arm224.name, arm224.weight = "Raptor-224", 0.35
    (fold_mean,) = r7.members = [RankMember.Config()]
    fold_mean.name = "R7"
    fold_mean.max_failure_fraction = 0.5
    return config


def train_exp000() -> ColdRun.Config:
    """Train an ImageNet-initialized study classifier within the complete cold budget.

    Hypothesis:
      ConvNeXt-tiny and finding attention can learn from gold labels in 300
      seconds on one H100/H200, including rendering, construction and export.

    References:
      https://pytorch.org/vision/stable/models/convnext.html

    Results:
      Current cold GPU timing is unmeasured. Historical 200.29-second H200
      measurements describe an earlier implementation, not this protocol.

    Returns:
      config: Printable cold job around a real dataset, step and training loop.

    """
    config = ColdRun.Config()
    loop = config.cold_loop
    loop.study_name = "rsna_knee_training"
    loop.experiment_name = "exp000"

    loop.step.model.head.dropout.p = 0.0
    loop.step.compile = None
    loop.step.dtype_autocast = torch.bfloat16
    loop.step.loss = SimpleLoss.Config()
    loop.step.loss.loss_fn = bce_with_logits
    loop.step.optimizer = CompositeOptimizer.Config()
    adamw = PartialConfig(torch.optim.AdamW)
    adamw.lr = 1e-4
    adamw.weight_decay = 1e-2
    loop.step.optimizer.optimizers = [adamw]

    loop.max_steps = 10_000
    loop.max_time = loop.step.train_budget_sec = 180.0
    loop.max_eval_time = config.budget.reserve_evaluation_sec
    loop.num_steps_eval = -1
    loop.eval_every_epoch = False
    loop.checkpointer = None
    loop.metrics_eval = {"": ProbabilityAuc.Config()}
    loop.tracker = FileTracker.Config()
    loop.runtime = SingleProcess.Config()
    loop.runtime.device = "cuda"
    return config


def train_exp001() -> ColdRun.Config:
    """train_exp000 with four studies per optimizer batch.

    Hypothesis:
      Larger batches improve GPU throughput under the same cold wall budget.

    References:
      https://pytorch.org/vision/stable/models/convnext.html

    Results:
      Unmeasured.

    Returns:
      config: The named parent with only batch size changed.

    """
    config = train_exp000()
    config.cold_loop.experiment_name = "exp001"
    config.cold_loop.dataset.batch_size = 4
    return config
