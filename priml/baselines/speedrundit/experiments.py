"""SpeedrunDiT experiments: REG/SPRINT SiT on autoencoder latents of ImageNet.

    exp000  REG/SPRINT SiT-B/1 on INVAE latents (the reference)
      +-- exp001  shared, simpler floating-point arithmetic
      +-- exp002  VTP-Large latents
      +-- exp003  RAE DINOv2-B latents, stored float16
            +-- exp004  the same latents stored as uint8 Lloyd-Max indices

Each experiment's ``dataset.source`` names the autoencoder and storage codec its
corpus was made with; prepare that corpus once, then launch::

    uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --experiment exp002 --source /datasets/imagenet
    uv --quiet run --frozen python -m priml priml.baselines.speedrundit.experiments.exp002
"""

from __future__ import annotations

from dataclasses import field
from typing import Self, override

from configgle import Makes

import torch

from priml.baselines.speedrundit.data import SpeedrunImageNetData
from priml.baselines.speedrundit.latent_codec import FloatCodec, ScalarTableCodec
from priml.baselines.speedrundit.optimizers import speedrundit_optimizer
from priml.baselines.speedrundit.train_step import SpeedrunTrainStep
from priml.model.vision_ae.rae import rae_dinov2_base
from priml.model.vision_ae.vtp import vtp_large
from priml.runtime import MultiProcess, SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.parallelism import DataParallel, NoParallel
from priml.train.train_loop import TrainLoop


class SpeedrunTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[SpeedrunTrainStep.Config, SpeedrunImageNetData.Config],
):
    """Bind the SpeedrunDiT train step to its paired ImageNet dataset."""

    step: SpeedrunTrainStep.Config = field(default_factory=SpeedrunTrainStep.Config)
    """Model, teacher, objective, precision, and optimizer recipe."""

    dataset: SpeedrunImageNetData.Config = field(
        default_factory=SpeedrunImageNetData.Config,
    )
    """Processed ImageNet images paired with autoencoder latents."""

    @override
    def finalize(self) -> Self:
        # The dataset's autoencoder states the latent the model reads. Its published
        # normalizer fills an unset step slot; a mismatched geometry is an error, not
        # an overwrite, since the model's widths are the experiment's own choice.
        autoencoder = self.dataset.source.autoencoder
        if self.step.latent_norm is None:
            self.step.latent_norm = autoencoder.latent_norm
        channels, height, width = autoencoder.latent_shape()
        model = self.step.model
        if (model.in_channels, model.input_size, model.input_size) != (
            channels,
            height,
            width,
        ):
            raise ValueError(
                f"step.model reads {model.in_channels}x{model.input_size}x"
                f"{model.input_size} latents; dataset.source.autoencoder produces "
                f"{channels}x{height}x{width}.",
            )
        return super().finalize()


def exp000() -> SpeedrunTrainLoop:
    """REG/SPRINT SiT-B/1 with source Muon arithmetic and MLP scaling.

    Hypothesis:
      Reproduce the later REG branch's Muon optimizer, RMS normalization,
      value residual, contrastive flow loss, and depth-dependent MLP widths.

    References:
      https://github.com/SwayStar123/REG/tree/invae-sprint-rms-rope-valres-cfm-muon-layerwisescaling

    Results:
      TBD.

    """
    config = SpeedrunTrainLoop()
    config.study_name = "speedrundit"
    config.experiment_name = "exp000"
    config.max_steps = config.step.train_budget_steps = 400_000
    config.num_steps_eval = float(
        "inf",
    )  # The reference evaluates generated images separately.
    config.eval_every_epoch = False
    config.seed = 0
    config.runtime = MultiProcess.Config(float32_matmul_precision="high")
    config.step.parallelism = DataParallel.Config()
    assert isinstance(config.checkpointer, Checkpointer.Config)
    config.checkpointer.save_every = 10_000
    return config


def exp001() -> SpeedrunTrainLoop:
    """Keep exp000's algorithm with shared, simpler floating-point arithmetic.

    The fixed position table is computed in float32 and Muon uses the shared
    fused Newton-Schulz update. These are small numerical changes only.
    """
    config = exp000()
    config.experiment_name = "exp001"
    config.step.model.position_compute_dtype = torch.float32
    config.step.model.reference_rope = False
    config.step.optimizer = speedrundit_optimizer(reference_numerics=False)
    return config


def exp002() -> SpeedrunTrainLoop:
    """exp000 trained on VTP-Large latents instead of INVAE's.

    The latent space is the one change; the model's input width moves with it
    (64 channels instead of 32 at the same 16x16 grid), which is inseparable.
    Stored in float32 (about 84 GB for ImageNet-1k); the normalizer is VTP's
    published per-channel statistics, filled at finalize.

    Hypothesis:
      VTP's tokenizer is pretrained jointly for reconstruction, CLIP alignment,
      and self-supervision, which its authors report makes latents easier to
      generate from as the tokenizer scales. If that holds, the same SiT and
      recipe reach a lower loss and better samples from VTP latents than from
      INVAE's reconstruction-first latents.

    References:
      https://arxiv.org/abs/2512.13687
        Yao et al. 2025. Towards Scalable Pre-training of Visual Tokenizers for
        Generation.

    Results:
      TBD.

    """
    config = exp000()
    config.experiment_name = "exp002"
    source = config.dataset.source
    source.autoencoder = vtp_large()
    source.latent_subdir = "vtp-large"
    config.step.model.in_channels = source.autoencoder.latent_shape()[0]
    return config


def exp003() -> SpeedrunTrainLoop:
    """exp000 trained on RAE DINOv2-B latents, stored float16.

    The latent space is the change; the model's input width moves with it (768
    channels at the same 16x16 grid). Storage moves with it too: float32 would
    be about 1 TB for ImageNet-1k, so the corpus is float16 (504 GB), and
    exp004 isolates the storage question. The normalizer is RAE's published
    per-element variance, filled at finalize. The SiT's width equals the latent
    width here; the paper's wide DiT^DH head is not added.

    Hypothesis:
      Latents that are a frozen representation encoder's tokens are
      semantically organized, which RAE's authors report lets a diffusion
      transformer converge faster and sample better than on a reconstruction
      VAE's latents. The REG alignment target is DINOv2 as well, so the student
      is asked to align with features close to its own input. Float16 rounding
      of these bounded LayerNorm outputs (about 2**-11 relative) is assumed
      negligible; exp004 tests the coarser uint8 storage against it.

    References:
      https://arxiv.org/abs/2510.11690
        Zheng et al. 2025. Diffusion Transformers with Representation
        Autoencoders.

    Results:
      TBD.

    """
    config = exp000()
    config.experiment_name = "exp003"
    source = config.dataset.source
    source.autoencoder = rae_dinov2_base()
    source.latent_subdir = "rae-dinov2-base-f16"
    source.codec = FloatCodec.Config(dtype=torch.float16)
    config.step.model.in_channels = source.autoencoder.latent_shape()[0]
    return config


def exp004() -> SpeedrunTrainLoop:
    """exp003 with its latents stored as uint8 Lloyd-Max indices.

    One byte per scalar against a fitted per-channel table of 256 levels: half
    of exp003's float16 corpus (252 GB). Everything the model sees is otherwise
    unchanged; the same images, autoencoder, and normalizer.

    Hypothesis:
      At 8 bits a per-channel minimum-squared-error quantizer leaves about
      0.6% RMS error per scalar for a Gaussian-like channel (Panter-Dite, ~44
      dB), far below both the flow-matching noise at all but the smallest
      timesteps and the noise RAE's decoder was trained to tolerate (sigma up
      to 0.8). If so, the trained model is indistinguishable from exp003's
      within seed variance. Run benchmark_codec.py first; it measures the
      latent- and pixel-space error this relies on.

    References:
      S. P. Lloyd. Least squares quantization in PCM. IEEE Trans. Inf. Theory
        28(2):129-137, 1982.
      J. Max. Quantizing for minimum distortion. IRE Trans. Inf. Theory
        6(1):7-12, 1960.

    Results:
      TBD.

    """
    config = exp003()
    config.experiment_name = "exp004"
    source = config.dataset.source
    source.codec = ScalarTableCodec.Config()
    source.latent_subdir = "rae-dinov2-base-u8"
    return config


def exp_smoke() -> SpeedrunTrainLoop:
    """Five quick single-device updates with the same training mechanisms."""
    config = exp000()
    config.experiment_name = "exp_smoke"
    config.runtime = SingleProcess.Config()
    config.step.parallelism = NoParallel.Config()
    config.max_steps = config.step.train_budget_steps = 5
    config.step.model.hidden_size = 32
    config.step.model.num_heads = 4
    config.step.model.depth = 6
    config.step.model.projector_hidden = 64
    config.step.model.projection_depths = (2, 3, 6)
    config.dataset.batch_size = 2
    config.checkpointer = None
    config.dataset.num_workers = 0
    return config
