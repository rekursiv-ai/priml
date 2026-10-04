#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# ruff: noqa: E402 -- Every import after cap_math_threads() must follow it.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Prove each vision autoencoder port bit-identical to the reference it ports.

For INVAE, RAE, and VTP: clones the reference at its pinned commit, imports it
UNMODIFIED, gives both sides one set of weights, and runs both encode/decode
paths on the same inputs. Following ``docs/SKILL.bit-for-bit.md``:

1. Inventory. Every function, method, and class in the reference files on the
   reconstruction path is PAIRED with the port code that mirrors it, or NOT
   PORTED with a reason. A definition in neither table fails the run.
2. Randomness. ``torch.randn`` is pinned on both sides to the same stored
   draws, and each call site is counted; every other random primitive raises.
3. Bisection. Forward hooks record every submodule's outputs under its module
   name on both sides, and every name both sides share is compared with
   ``torch.equal``. All mismatches are collected into one report.
4. Coverage. ``coverage`` measures branches over both sides. Every paired
   function must run, no not-ported one may, and every branch in paired code
   must go both ways unless an allowlist names it with a reason.
5. Goldens. ``--mint`` writes ``testdata/rae_reference.pt`` and
   ``testdata/vtp_reference.pt`` from the reference side. Without it, the
   run re-derives both and requires the checked-in bits.

Initialization is compared too (bit-for-bit checkpoint 1): each side is built
from the same seed and the state dicts must match in names, order, and bits.

Exactly these things are supplied or changed on the reference side:

- RAE: its Hugging Face encoder is switched to eager attention through HF's
  own ``set_attn_implementation``, the kernel the port fixes.
- INVAE: REG's ``models/invae.py`` imports ``dictdot`` for one return wrapper;
  a stub supplies a dict with attribute access.
- VTP: ``vtp/__init__`` imports the CLIP side's dependencies, so ``vtp`` and
  ``vtp.models`` are registered as bare namespace packages and only the
  reconstruction modules import. The bfloat16 case autocasts on the CPU where
  the reference's evaluation autocasts on CUDA.

INVAE has no checked-in reference golden: REG hardcodes 32 GroupNorm groups,
so the smallest model it builds is far over the 32,768-byte golden ceiling.
This run is its reference proof; ``testdata/invae.pt`` guards the port.

Examples:
  reference_parity.py
  reference_parity.py --model rae vtp --mint

'''
# fmt: on

from __future__ import annotations

from priml.conftest import cap_math_threads


# Before torch loads: the goldens this mints must come from the environment
# pytest replays them in.
cap_math_threads()

from collections import Counter, defaultdict
from collections.abc import (
    Callable,
    Generator,
    Mapping,
    Sequence,
)
from contextlib import (
    ExitStack,
    contextmanager,
    nullcontext,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, Self, cast

import argparse
import ast
import dataclasses
import importlib
import inspect
import json
import subprocess
import sys
import tempfile
import types

from safetensors.torch import save_file
from torch import Tensor, nn
from torchvision.transforms import Normalize
from torchvision.transforms.functional import (
    to_tensor,
)

import coverage
import torch

from priml.hub import get_cache_dir
from priml.model.vision_ae.checkpoint import (
    LocalFile,
)
from priml.model.vision_ae.custom_types import (
    posterior_mode,
)
from priml.model.vision_ae.invae import INVAE
from priml.model.vision_ae.latent_norm import (
    ElementwiseLatentStats,
)
from priml.model.vision_ae.rae import (
    RAE,
    Dinov2WithRegisters,
    GeneralDecoder,
)
from priml.model.vision_ae.vtp import VTP
from priml.testing.bfb import (
    host_agnostic_numerics,
    load_golden,
)
from priml.testing.golden import (
    mismatches,
    read_tensors,
    write_tensors,
)

import priml


_PRIML: Final = Path(priml.__file__).resolve().parent
_TESTDATA: Final = _PRIML / "model" / "vision_ae" / "testdata"


# What this run reads of each untyped reference, as karpathy_parity types its own.
class _Posterior(Protocol):
    mean: Tensor
    logvar: Tensor
    std: Tensor
    var: Tensor

    def sample(self) -> Tensor: ...

    def mode(self) -> Tensor: ...


class _Decoded(Protocol):
    sample: Tensor


class _TheirInvae(Protocol):
    def encode(self, x: Tensor, /) -> _Posterior: ...

    def decode(self, z: Tensor, /) -> _Decoded: ...


class _InvaeModule(Protocol):
    VAE_F16D32: Callable[[], nn.Module]


class _HfEncoder(Protocol):
    def set_attn_implementation(self, name: str, /) -> None: ...


class _TheirDinov2(Protocol):
    encoder: _HfEncoder


class _TheirRae(Protocol):
    encoder: _TheirDinov2
    decoder: nn.Module

    def encode(self, x: Tensor, /) -> Tensor: ...

    def decode(self, z: Tensor, /) -> Tensor: ...


class _Stage1Module(Protocol):
    RAE: Callable[..., nn.Module]


class _ViTMAEConfig(Protocol):
    hidden_size: int
    image_size: int


class _DecoderUtilsModule(Protocol):
    ViTMAEConfig: Callable[..., _ViTMAEConfig]


class _DecoderModule(Protocol):
    GeneralDecoder: Callable[..., nn.Module]
    DinoV3PixelDecoder: Callable[..., nn.Module]


class _TheirVtp(Protocol):
    def get_reconstruction_latents(self, image: Tensor, /) -> Tensor: ...

    def get_latents_decoded_images(self, latents: Tensor, /) -> Tensor: ...


class _VtpHfModule(Protocol):
    VTPModel: Callable[[object], nn.Module]
    VTPConfig: Callable[..., object]


@dataclass(frozen=True, slots=True, kw_only=True)
class Reference:
    """One reference repository, pinned, and its inventory against the port.

    Keys are ``"<file>::<Qualified.name>"``: reference files relative to the
    clone, port files relative to the ``priml`` package.
    """

    url: str
    """Repository to clone."""

    commit: str
    """The revision this comparison is against."""

    files: tuple[str, ...]
    """Reference files on the reconstruction path; every definition is inventoried."""

    paired: Mapping[str, tuple[str, ...]]
    """Reference definition -> the port definitions that mirror it."""

    not_ported: Mapping[str, str]
    """Reference definition (a class covers its methods) -> why nothing mirrors it."""

    reference_unreached: Mapping[str, str]
    """``"<file>::<source line>"`` of a reference branch no input drives both ways -> why."""

    port_unreached: Mapping[str, str]
    """The same, for branches in the port's ``model/vision_ae`` code."""

    extra_outputs: Mapping[str, str]
    """Module names whose calls return more tensors on one side -> why."""

    uncompared: Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])
    """Module names whose outputs differ in form and are compared elsewhere -> why."""

    reference_calls: Mapping[str, tuple[tuple[int, ...], str]] = dataclasses.field(
        default_factory=dict[str, tuple[tuple[int, ...], str]],
    )
    """Module names the reference calls more often -> its calls the port mirrors, and why."""


_INVAE: Final = Reference(
    url="https://github.com/SwayStar123/REG.git",
    commit="3c51606c801dd9e87ee9ef778782766ab7c379ca",
    files=("models/invae.py",),
    paired={
        **{
            f"models/invae.py::{name}": (f"model/vision_ae/invae.py::{name}",)
            for name in (
                "nonlinearity",
                "Normalize",
                "Upsample",
                "Upsample.__init__",
                "Upsample.forward",
                "Downsample",
                "Downsample.__init__",
                "Downsample.forward",
                "ResnetBlock",
                "ResnetBlock.__init__",
                "ResnetBlock.forward",
                "AttnBlock",
                "AttnBlock.__init__",
                "AttnBlock.forward",
                "Encoder",
                "Encoder.__init__",
                "Encoder.forward",
                "Decoder",
                "Decoder.__init__",
                "Decoder.forward",
                "DiagonalGaussianDistribution",
                "DiagonalGaussianDistribution.__init__",
                "DiagonalGaussianDistribution.sample",
                "DiagonalGaussianDistribution.mode",
            )
        },
        "models/invae.py::AutoencoderKL": ("model/vision_ae/invae.py::INVAE",),
        "models/invae.py::AutoencoderKL.__init__": (
            "model/vision_ae/invae.py::INVAE.__init__",
        ),
        "models/invae.py::AutoencoderKL.encode": (
            "model/vision_ae/invae.py::INVAE.posterior",
            "model/vision_ae/invae.py::INVAE.encode",
            "model/vision_ae/custom_types.py::posterior_sample",
            "model/vision_ae/custom_types.py::posterior_mode",
        ),
        "models/invae.py::AutoencoderKL.decode": (
            "model/vision_ae/invae.py::INVAE.decode",
        ),
        "models/invae.py::VAE_F16D32": ("model/vision_ae/invae.py::INVAE.Config",),
    },
    not_ported={
        "models/invae.py::DiagonalGaussianDistribution.kl": (
            "A training loss; nothing on the encode/decode path calls it."
        ),
        "models/invae.py::DiagonalGaussianDistribution.nll": (
            "A training loss; nothing on the encode/decode path calls it."
        ),
        "models/invae.py::AutoencoderKL.forward": (
            "The training round trip: encode, sample, decode, each paired above."
        ),
        "models/invae.py::VAE_F8D4": (
            "The f8d4 geometry is INVAE.Config values (channels_latent=4, "
            "channel_multipliers=(1, 2, 4, 4)); no checkpoint of it is published."
        ),
    },
    reference_unreached={
        "models/invae.py::if self.with_conv:": (
            "Every published model resamples with a convolution."
        ),
        "models/invae.py::if temb is not None:": "INVAE has no time embedding.",
        "models/invae.py::if temb_channels > 0:": "INVAE has no time embedding.",
        "models/invae.py::if self.use_conv_shortcut:": (
            "The encoder and decoder build every ResnetBlock without it."
        ),
        "models/invae.py::if self.deterministic:": (
            "AutoencoderKL.encode never builds a deterministic posterior."
        ),
        "models/invae.py::if not self.use_variational:": ("VAE_F16D32 is variational."),
        "models/invae.py::if self.give_pre_end:": "Never set by AutoencoderKL.",
    },
    port_unreached={
        "model/vision_ae/invae.py::if self.with_conv:": (
            "The port builds every resampler with a convolution."
        ),
        "model/vision_ae/invae.py::if temb is not None:": (
            "INVAE has no time embedding."
        ),
        "model/vision_ae/invae.py::if temb_channels > 0:": (
            "INVAE has no time embedding."
        ),
        "model/vision_ae/invae.py::if self.use_conv_shortcut:": (
            "The encoder and decoder build every ResnetBlock without it."
        ),
        "model/vision_ae/invae.py::if self.give_pre_end:": "INVAE never sets it.",
    },
    extra_outputs={},
)


_RAE_DECODER: Final = "src/stage1/decoders/decoder.py"
_RAE_MAE_ENCODER: Final = (
    "The MAE encoder's; GeneralDecoder builds only the decoder half."
)

_RAE: Final = Reference(
    url="https://github.com/bytetriper/RAE.git",
    commit="a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041",
    files=(
        "src/stage1/rae.py",
        "src/stage1/encoders/dinov2.py",
        _RAE_DECODER,
        "src/stage1/decoders/utils.py",
    ),
    paired={
        "src/stage1/rae.py::RAE": ("model/vision_ae/rae.py::RAE",),
        "src/stage1/rae.py::RAE.__init__": ("model/vision_ae/rae.py::RAE.__init__",),
        "src/stage1/rae.py::RAE.encode": (
            "model/vision_ae/rae.py::RAE.encode",
            "model/vision_ae/latent_norm.py::ElementwiseLatentStats.__init__",
            "model/vision_ae/latent_norm.py::ElementwiseLatentStats.normalize",
            "math/pixel.py::rgb2float",
        ),
        "src/stage1/rae.py::RAE.decode": (
            "model/vision_ae/rae.py::RAE.decode",
            "model/vision_ae/latent_norm.py::ElementwiseLatentStats.denormalize",
        ),
        "src/stage1/encoders/dinov2.py::Dinov2withNorm": (
            "model/vision_ae/rae.py::Dinov2WithRegisters",
        ),
        "src/stage1/encoders/dinov2.py::Dinov2withNorm.__init__": (
            "model/vision_ae/rae.py::Dinov2WithRegisters.__init__",
        ),
        "src/stage1/encoders/dinov2.py::Dinov2withNorm.dinov2_forward": (
            "model/vision_ae/rae.py::Dinov2WithRegisters.forward",
        ),
        "src/stage1/encoders/dinov2.py::Dinov2withNorm.forward": (
            "model/vision_ae/rae.py::Dinov2WithRegisters.forward",
        ),
        **{
            f"{_RAE_DECODER}::{name}": (f"model/vision_ae/rae.py::{name}",)
            for name in (
                "ViTMAESelfAttention",
                "ViTMAESelfAttention.__init__",
                "ViTMAESelfAttention.forward",
                "ViTMAESelfOutput",
                "ViTMAESelfOutput.__init__",
                "ViTMAESelfOutput.forward",
                "ViTMAEAttention",
                "ViTMAEAttention.__init__",
                "ViTMAEAttention.forward",
                "ViTMAEIntermediate",
                "ViTMAEIntermediate.__init__",
                "ViTMAEIntermediate.forward",
                "ViTMAEOutput",
                "ViTMAEOutput.__init__",
                "ViTMAEOutput.forward",
                "ViTMAELayer",
                "ViTMAELayer.__init__",
                "ViTMAELayer.forward",
                "GeneralDecoder",
                "GeneralDecoder.__init__",
                "GeneralDecoder.unpatchify",
                "GeneralDecoder.forward",
            )
        },
        f"{_RAE_DECODER}::ViTMAESelfAttention.transpose_for_scores": (
            "model/vision_ae/rae.py::ViTMAESelfAttention._heads",
        ),
        f"{_RAE_DECODER}::GeneralDecoder.initialize_weights": (
            "model/vision_ae/rae.py::GeneralDecoder.__init__",
        ),
        f"{_RAE_DECODER}::GeneralDecoder.set_trainable_cls_token": (
            "model/vision_ae/rae.py::GeneralDecoder.__init__",
        ),
        f"{_RAE_DECODER}::GeneralDecoder.interpolate_latent": (
            "model/vision_ae/rae.py::GeneralDecoder.forward",
        ),
        **{
            f"{_RAE_DECODER}::{name}": (
                "math/position_embedding.py::sincos_position_table",
            )
            for name in (
                "get_2d_sincos_pos_embed",
                "get_2d_sincos_pos_embed_from_grid",
                "get_1d_sincos_pos_embed_from_grid",
            )
        },
        "src/stage1/decoders/utils.py::ViTMAEConfig": (
            "model/vision_ae/rae.py::GeneralDecoder.Config",
        ),
        "src/stage1/decoders/utils.py::ViTMAEConfig.__init__": (
            "model/vision_ae/rae.py::GeneralDecoder.Config",
        ),
    },
    not_ported={
        "src/stage1/rae.py::Stage1Protocal": (
            "A typing Protocol; rae.py's PatchEncoder states the same contract."
        ),
        "src/stage1/rae.py::RAE.noising": (
            "A decoder-training augmentation; encode runs it only in training "
            "mode with noise_tau > 0, and the port never trains."
        ),
        "src/stage1/rae.py::RAE.forward": "Encode then decode, both paired.",
        f"{_RAE_DECODER}::ViTMAEModelOutput": _RAE_MAE_ENCODER,
        f"{_RAE_DECODER}::ViTMAEForPreTrainingOutput": _RAE_MAE_ENCODER,
        f"{_RAE_DECODER}::ViTMAEEmbeddings": _RAE_MAE_ENCODER,
        f"{_RAE_DECODER}::ViTMAEPatchEmbeddings": _RAE_MAE_ENCODER,
        f"{_RAE_DECODER}::ViTMAEDecoderOutput": (
            "An output record; the port's GeneralDecoder returns the logits."
        ),
        f"{_RAE_DECODER}::ViTMAESdpaSelfAttention": (
            "Selected by an sdpa attention config; the decoder config the "
            "published RAE uses selects the eager class."
        ),
        f"{_RAE_DECODER}::GeneralDecoder.interpolate_pos_encoding": (
            "Resizes a latent grid the decoder was not built for; the port "
            "raises instead."
        ),
    },
    reference_unreached={
        "src/stage1/rae.py::if len(keys.missing_keys) > 0:": (
            "A load report; the port loads strictly, so no key is ever missing."
        ),
        "src/stage1/rae.py::if self.training and self.noise_tau > 0:": (
            "Training-time noising; both sides only evaluate."
        ),
        "src/stage1/rae.py::if self.reshape_to_2d:": (
            "The published RAE reshapes; the port always does."
        ),
        "src/stage1/encoders/dinov2.py::if normalize:": (
            "The published RAE strips the final norm's affine; the port always does."
        ),
        f"{_RAE_DECODER}::if config.hidden_size % config.num_attention_heads != 0 "
        'and not hasattr(config, "embedding_size"):': (
            "A misconfiguration guard; the port's matching one is tested in rae_test."
        ),
        f"{_RAE_DECODER}::if head_mask is not None:": "No caller passes a head mask.",
        f"{_RAE_DECODER}::if isinstance(config.hidden_act, str):": (
            "Every ViTMAE config names its activation; the port injects the function."
        ),
        f"{_RAE_DECODER}::if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:": (
            "A shape guard the decoder's own output always satisfies."
        ),
        f"{_RAE_DECODER}::if drop_cls_token:": "RAE.decode passes drop_cls_token=False.",
        f"{_RAE_DECODER}::if l == self.num_patches:": (
            "The latent always has the decoder's grid; on another the port "
            "raises, which rae_test drives."
        ),
        f"{_RAE_DECODER}::if not return_dict:": "RAE.decode reads the returned record.",
        f"{_RAE_DECODER}::if output_attentions:": "RAE.decode asks for no attentions.",
        f"{_RAE_DECODER}::if interpolate_pos_encoding:": (
            "RAE.decode never asks; the port raises on another grid instead."
        ),
        f"{_RAE_DECODER}::if output_hidden_states:": (
            "RAE.decode asks for no hidden states."
        ),
        f"{_RAE_DECODER}::if self.gradient_checkpointing and self.training:": (
            "Training-only."
        ),
        f"{_RAE_DECODER}::if add_cls_token:": (
            "initialize_weights always asks for the CLS row."
        ),
        f"{_RAE_DECODER}::if embed_dim % 2 != 0:": (
            "A misconfiguration guard; the port's GeneralDecoder.Config refuses "
            "a width the table cannot split."
        ),
    },
    port_unreached={
        "model/vision_ae/latent_norm.py::if config.stats is None:": (
            "An error guard; latent_norm_test drives it."
        ),
        "model/vision_ae/latent_norm.py::if var is None:": (
            "An error guard; latent_norm_test drives it."
        ),
        "model/vision_ae/rae.py::if config.checkpoint is not None:": (
            "The comparison loads both checkpoints; rae_test builds without them."
        ),
        "model/vision_ae/rae.py::if tokens.shape[1] != self.num_patches:": (
            "Raises where the reference resizes; rae_test drives it."
        ),
        "model/vision_ae/rae.py::if channels % num_heads:": (
            "A misconfiguration guard; valid configs never take it."
        ),
    },
    extra_outputs={
        "encoder.encoder": (
            "The reference asks HF for every hidden state (output_hidden_states=True); "
            "the last one, the latent, is compared."
        ),
        "encoder.encoder.encoder": (
            "The reference asks HF for every hidden state (output_hidden_states=True); "
            "the last one, the latent, is compared."
        ),
    },
)


_VTP_LAYERS: Final = "vtp/models/layers"
_VTP_TEXT: Final = "The CLIP text and alignment side; reconstruction never reads it."
_VTP_TRAINING: Final = "Training-only; the evaluation forward never runs it."
_VTP_OTHER_BLOCK: Final = (
    "Another architecture's block; the trunk and decoder use none."
)

_VTP: Final = Reference(
    url="https://github.com/MiniMax-AI/VTP.git",
    commit="5ce1eb67010fff3c1eed483352483be6a1838556",
    files=(
        "vtp/models/vtp_hf/modeling_vtp.py",
        "vtp/models/vtp_hf/configuration_vtp.py",
        "vtp/models/encoders/vision_transformer_bottleneck.py",
        "vtp/models/encoders/vision_transformer.py",
        "vtp/models/decoders/pixel_decoder.py",
        f"{_VTP_LAYERS}/attention.py",
        f"{_VTP_LAYERS}/block.py",
        f"{_VTP_LAYERS}/embeddings.py",
        f"{_VTP_LAYERS}/ffn.py",
        f"{_VTP_LAYERS}/normalization.py",
        f"{_VTP_LAYERS}/misc.py",
        f"{_VTP_LAYERS}/activation.py",
    ),
    paired={
        "vtp/models/vtp_hf/modeling_vtp.py::VTPPreTrainedModel": (
            "model/vision_ae/vtp.py::VTP",
        ),
        "vtp/models/vtp_hf/modeling_vtp.py::VTPPreTrainedModel._init_weights": (
            "model/vision_ae/vtp.py::init_weights_post",
        ),
        "vtp/models/vtp_hf/modeling_vtp.py::VTPModel": ("model/vision_ae/vtp.py::VTP",),
        "vtp/models/vtp_hf/modeling_vtp.py::VTPModel.__init__": (
            "model/vision_ae/vtp.py::VTP.__init__",
        ),
        "vtp/models/vtp_hf/modeling_vtp.py::VTPModel._init_vision_components": (
            "model/vision_ae/vtp.py::VTP.__init__",
        ),
        "vtp/models/vtp_hf/modeling_vtp.py::VTPModel.get_reconstruction_latents": (
            "model/vision_ae/vtp.py::VTP.encode",
            "math/pixel.py::rgb2float",
        ),
        "vtp/models/vtp_hf/modeling_vtp.py::VTPModel.get_latents_decoded_images": (
            "model/vision_ae/vtp.py::VTP.decode",
        ),
        "vtp/models/vtp_hf/modeling_vtp.py::VTPModel._patch_tokens_to_4d": (
            "model/vision_ae/vtp.py::DinoVisionTransformerWithBottleneck.forward",
        ),
        "vtp/models/vtp_hf/configuration_vtp.py::VTPConfig": (
            "model/vision_ae/vtp.py::VTP.Config",
        ),
        "vtp/models/vtp_hf/configuration_vtp.py::VTPConfig.__init__": (
            "model/vision_ae/vtp.py::VTP.Config",
        ),
        **{
            f"vtp/models/encoders/{file}::{name}": (
                f"model/vision_ae/vtp.py::DinoVisionTransformerWithBottleneck{method}",
            )
            for file, name, method in (
                (
                    "vision_transformer_bottleneck.py",
                    "DinoVisionTransformerWithBottleneck",
                    "",
                ),
                (
                    "vision_transformer_bottleneck.py",
                    "DinoVisionTransformerWithBottleneck.__init__",
                    ".__init__",
                ),
                (
                    "vision_transformer_bottleneck.py",
                    "DinoVisionTransformerWithBottleneck.init_weights",
                    ".__init__",
                ),
                (
                    "vision_transformer_bottleneck.py",
                    "DinoVisionTransformerWithBottleneck._apply_feature_bottleneck",
                    ".forward",
                ),
                (
                    "vision_transformer_bottleneck.py",
                    "DinoVisionTransformerWithBottleneck.forward_features",
                    ".forward",
                ),
                (
                    "vision_transformer_bottleneck.py",
                    "DinoVisionTransformerWithBottleneck._process_output_dict",
                    ".forward",
                ),
                ("vision_transformer.py", "DinoVisionTransformer", ""),
                (
                    "vision_transformer.py",
                    "DinoVisionTransformer.__init__",
                    ".__init__",
                ),
                (
                    "vision_transformer.py",
                    "DinoVisionTransformer.init_weights",
                    ".__init__",
                ),
                (
                    "vision_transformer.py",
                    "DinoVisionTransformer.prepare_tokens_with_masks",
                    ".forward",
                ),
                (
                    "vision_transformer.py",
                    "DinoVisionTransformer.forward_features_list",
                    ".forward",
                ),
                (
                    "vision_transformer.py",
                    "DinoVisionTransformer.forward_features",
                    ".forward",
                ),
                ("vision_transformer.py", "DinoVisionTransformer.forward", ".forward"),
            )
        },
        "vtp/models/encoders/vision_transformer.py::init_weights_vit": (
            "model/vision_ae/vtp.py::init_weights_vit",
        ),
        "vtp/models/decoders/pixel_decoder.py::DinoV3PixelDecoder": (
            "model/vision_ae/vtp.py::DinoV3PixelDecoder",
        ),
        "vtp/models/decoders/pixel_decoder.py::DinoV3PixelDecoder.__init__": (
            "model/vision_ae/vtp.py::DinoV3PixelDecoder.__init__",
        ),
        "vtp/models/decoders/pixel_decoder.py::DinoV3PixelDecoder.init_weights": (
            "model/vision_ae/vtp.py::DinoV3PixelDecoder.__init__",
        ),
        "vtp/models/decoders/pixel_decoder.py::DinoV3PixelDecoder.forward": (
            "model/vision_ae/vtp.py::DinoV3PixelDecoder.forward",
        ),
        **{
            f"{_VTP_LAYERS}/{file}::{name}": (f"model/vision_ae/vtp.py::{port}",)
            for file, name, port in (
                ("attention.py", "rope_rotate_half", "rope_rotate_half"),
                ("attention.py", "rope_apply", "rope_apply"),
                ("attention.py", "SelfAttention", "SelfAttention"),
                ("attention.py", "SelfAttention.__init__", "SelfAttention.__init__"),
                (
                    "attention.py",
                    "SelfAttention.apply_rope",
                    "SelfAttention.apply_rope",
                ),
                ("attention.py", "SelfAttention.forward", "SelfAttention.forward"),
                (
                    "attention.py",
                    "SelfAttention.compute_attention",
                    "SelfAttention.forward",
                ),
                ("block.py", "SelfAttentionBlock", "SelfAttentionBlock"),
                (
                    "block.py",
                    "SelfAttentionBlock.__init__",
                    "SelfAttentionBlock.__init__",
                ),
                (
                    "block.py",
                    "SelfAttentionBlock._forward_list",
                    "SelfAttentionBlock.forward",
                ),
                (
                    "block.py",
                    "SelfAttentionBlock.forward",
                    "SelfAttentionBlock.forward",
                ),
                ("embeddings.py", "make_2tuple", "PatchEmbed.__init__"),
                ("embeddings.py", "PatchEmbed", "PatchEmbed"),
                ("embeddings.py", "PatchEmbed.__init__", "PatchEmbed.__init__"),
                ("embeddings.py", "PatchEmbed.forward", "PatchEmbed.forward"),
                (
                    "embeddings.py",
                    "PatchEmbed.reset_parameters",
                    "PatchEmbed.reset_parameters",
                ),
                ("embeddings.py", "RopePositionEmbedding", "RopePositionEmbedding"),
                (
                    "embeddings.py",
                    "RopePositionEmbedding.__init__",
                    "RopePositionEmbedding.__init__",
                ),
                (
                    "embeddings.py",
                    "RopePositionEmbedding.forward",
                    "RopePositionEmbedding.forward",
                ),
                (
                    "embeddings.py",
                    "RopePositionEmbedding._init_weights",
                    "RopePositionEmbedding.reset_parameters",
                ),
                ("ffn.py", "SwiGLUFFN", "SwiGLUFFN"),
                ("ffn.py", "SwiGLUFFN.__init__", "SwiGLUFFN.__init__"),
                ("ffn.py", "SwiGLUFFN.forward", "SwiGLUFFN.forward"),
                ("normalization.py", "RMSNorm", "RMSNorm"),
                ("normalization.py", "RMSNorm.__init__", "RMSNorm.__init__"),
                (
                    "normalization.py",
                    "RMSNorm.reset_parameters",
                    "RMSNorm.reset_parameters",
                ),
                ("normalization.py", "RMSNorm._norm", "RMSNorm.forward"),
                ("normalization.py", "RMSNorm.forward", "RMSNorm.forward"),
            )
        },
    },
    not_ported={
        **{
            f"vtp/models/vtp_hf/modeling_vtp.py::VTPModel.{name}": _VTP_TEXT
            for name in (
                "_init_text_components",
                "get_clip_image_feature",
                "get_clip_text_feature",
                "get_clip_logits",
                "_forward_clip",
            )
        },
        **{
            f"vtp/models/vtp_hf/modeling_vtp.py::VTPModel.{name}": (
                "A representation probe for linear probing, not the autoencoder."
            )
            for name in (
                "get_last_layer_feature",
                "get_intermediate_layers_feature",
                "_forward_feature",
            )
        },
        "vtp/models/vtp_hf/modeling_vtp.py::VTPModel.forward": (
            "The training forward; reconstruction's two calls are paired."
        ),
        "vtp/models/vtp_hf/modeling_vtp.py::VTPModel._forward_reconstruction": (
            "The training forward; reconstruction's two calls are paired."
        ),
        "vtp/models/vtp_hf/configuration_vtp.py::VTPConfig.from_vtp_yaml": (
            "Reads a training YAML; vtp_small, vtp_base, and vtp_large state "
            "the published configurations."
        ),
        "vtp/models/encoders/vision_transformer_bottleneck.py::DinoVisionTransformerWithBottleneck.get_intermediate_layers": (
            "A representation probe, not the autoencoder."
        ),
        "vtp/models/encoders/vision_transformer_bottleneck.py::DinoVisionTransformerWithBottleneck.encode": (
            "An entry point hardcoding a 16-pixel patch; reconstruction goes "
            "through get_reconstruction_latents."
        ),
        "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer._get_intermediate_layers_not_chunked": (
            "A representation probe, not the autoencoder."
        ),
        "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.get_intermediate_layers": (
            "A representation probe, not the autoencoder."
        ),
        **{
            f"vtp/models/encoders/vision_transformer.py::{name}": (
                "A backbone preset; VTPModel builds its trunk directly, and "
                "vtp_small, vtp_base, and vtp_large state the published sizes."
            )
            for name in (
                "vit_small",
                "vit_base",
                "vit_large",
                "vit_so400m",
                "vit_huge2",
                "vit_giant2",
                "vit_7b",
            )
        },
        **{
            f"vtp/models/decoders/pixel_decoder.py::{name}": (
                "A decoder preset; VTPModel builds its decoder directly."
            )
            for name in (
                "dinov3_pixel_decoder_small",
                "dinov3_pixel_decoder_base",
                "dinov3_pixel_decoder_large",
            )
        },
        f"{_VTP_LAYERS}/attention.py::LinearKMaskedBias": (
            "A masked-key-bias projection another configuration selects; VTP's "
            "qkv is a plain biased Linear."
        ),
        f"{_VTP_LAYERS}/attention.py::SelfAttention.forward_list": (
            "The list-batched forward; every block calls forward."
        ),
        f"{_VTP_LAYERS}/attention.py::CausalSelfAttention": _VTP_TEXT,
        f"{_VTP_LAYERS}/attention.py::Attention": _VTP_TEXT,
        f"{_VTP_LAYERS}/attention.py::AttentionalPooler": _VTP_TEXT,
        f"{_VTP_LAYERS}/block.py::get_branges_scales": _VTP_TRAINING,
        f"{_VTP_LAYERS}/block.py::clear_sampling_cache": _VTP_TRAINING,
        f"{_VTP_LAYERS}/block.py::print_sampling_cache_info": _VTP_TRAINING,
        f"{_VTP_LAYERS}/block.py::SelfAttentionBlock._maybe_index_rope": (
            "Indexes the rotary table for dropped patches; training-only."
        ),
        f"{_VTP_LAYERS}/block.py::SelfAttentionBlock._forward": (
            "The unbatched path; forward wraps a single tensor into _forward_list."
        ),
        f"{_VTP_LAYERS}/block.py::CausalSelfAttentionBlock": _VTP_TEXT,
        f"{_VTP_LAYERS}/block.py::ResidualAttentionBlock": _VTP_TEXT,
        f"{_VTP_LAYERS}/block.py::CustomResidualAttentionBlock": _VTP_TEXT,
        f"{_VTP_LAYERS}/embeddings.py::PatchEmbed.flops": (
            "A FLOP estimate; the port's cost() prices the patch convolution."
        ),
        **{
            f"{_VTP_LAYERS}/embeddings.py::{name}": (
                "Absolute position tables; the trunk and decoder use rotary positions."
            )
            for name in (
                "get_2d_sincos_pos_embed",
                "get_2d_sincos_pos_embed_from_grid",
                "get_1d_sincos_pos_embed_from_grid",
                "interpolate_pos_embed",
            )
        },
        f"{_VTP_LAYERS}/ffn.py::ListForwardMixin": (
            "List dispatch the blocks never use: they call the FFN on a tensor."
        ),
        f"{_VTP_LAYERS}/ffn.py::Mlp": "The GELU MLP; VTP's configurations choose SwiGLU.",
        f"{_VTP_LAYERS}/normalization.py::LayerNorm": _VTP_TEXT,
        f"{_VTP_LAYERS}/normalization.py::LayerNormFp32": _VTP_TEXT,
        f"{_VTP_LAYERS}/misc.py::LayerScale": (
            "Identity at VTP's layerscale_init=None; the block builds nn.Identity."
        ),
        f"{_VTP_LAYERS}/misc.py::PatchDropout": _VTP_TRAINING,
        f"{_VTP_LAYERS}/activation.py::QuickGELU": _VTP_TEXT,
    },
    reference_unreached={
        "vtp/models/vtp_hf/modeling_vtp.py::elif isinstance(module, nn.Embedding):": "VTP's reconstruction modules hold no embedding tables.",
        "vtp/models/vtp_hf/modeling_vtp.py::if N != feat_h * feat_w:": "A shape guard the trunk's own tokens always satisfy.",
        "vtp/models/vtp_hf/modeling_vtp.py::if config.train_clip:": "Reconstruction builds VTPModel without the CLIP side.",
        "vtp/models/vtp_hf/modeling_vtp.py::if config.train_reconstruction:": "Reconstruction always builds the pixel decoder.",
        "vtp/models/vtp_hf/modeling_vtp.py::if self.pixel_decoder is None:": "Reconstruction always builds the pixel decoder.",
        "vtp/models/encoders/vision_transformer_bottleneck.py::if vit_feature_bottleneck is None:": "Every published trunk has a bottleneck.",
        "vtp/models/encoders/vision_transformer_bottleneck.py::if self.vit_feature_bottleneck != self.original_embed_dim:": "Every published bottleneck narrows; the port refuses an equal width.",
        "vtp/models/encoders/vision_transformer_bottleneck.py::if self.feature_bottleneck is not None:": "Every published trunk has a bottleneck.",
        "vtp/models/encoders/vision_transformer_bottleneck.py::if self.feature_bottleneck is None or not use_bottleneck:": "get_reconstruction_latents always asks for the bottleneck.",
        "vtp/models/encoders/vision_transformer_bottleneck.py::if isinstance(output, list):": "Reconstruction encodes one tensor, not a crop list.",
        "vtp/models/encoders/vision_transformer_bottleneck.py::for item in output:": "Reconstruction encodes one tensor, not a crop list.",
        "vtp/models/encoders/vision_transformer.py::if len(ignored_kwargs) > 0:": "A warning for unknown constructor keywords; none are passed.",
        "vtp/models/encoders/vision_transformer.py::if self.n_storage_tokens > 0:": "VTP's trunk has no register tokens.",
        "vtp/models/encoders/vision_transformer.py::if untie_cls_and_patch_norms:": "VTP ties the CLS and patch norms.",
        "vtp/models/encoders/vision_transformer.py::if untie_global_and_local_cls_norm:": "VTP ties the global and local CLS norms.",
        "vtp/models/encoders/vision_transformer.py::if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:": "VTP ties both norm pairs.",
        "vtp/models/encoders/vision_transformer.py::if self.untie_global_and_local_cls_norm and self.training and idx == 1:": "VTP ties both norm pairs, so this branch is never reached.",
        "vtp/models/encoders/vision_transformer.py::elif self.untie_cls_and_patch_norms:": "VTP ties both norm pairs, so this branch is never reached.",
        "vtp/models/encoders/vision_transformer.py::if masks is not None:": "Masked-token pretraining; reconstruction masks nothing.",
        "vtp/models/encoders/vision_transformer.py::if self.rope_embed is not None:": "VTP always uses rotary positions.",
        "vtp/models/encoders/vision_transformer.py::if isinstance(x, torch.Tensor):": "Reconstruction encodes one tensor, not a crop list.",
        "vtp/models/encoders/vision_transformer.py::if is_training:": "get_reconstruction_latents always asks for the token dict.",
        "vtp/models/encoders/vision_transformer.py::if isinstance(module, LayerScale):": "LayerScale is identity at VTP's layerscale_init=None.",
        "vtp/models/decoders/pixel_decoder.py::if len(ignored_kwargs) > 0:": "A warning for unknown constructor keywords; none are passed.",
        "vtp/models/decoders/pixel_decoder.py::if isinstance(m, nn.Conv2d):": "Its only modules are the two convolutions.",
        "vtp/models/decoders/pixel_decoder.py::if m.bias is not None:": "Both convolutions are biased.",
        "vtp/models/layers/attention.py::if rope is not None:": "VTP always uses rotary positions.",
        "vtp/models/layers/block.py::elif isinstance(x_or_x_list, list):": "The trunk and decoder pass one tensor.",
        "vtp/models/layers/block.py::if rope_or_rope_list is None:": "The trunk and decoder pass one tensor.",
        "vtp/models/layers/block.py::if rope_list is not None:": "Reached only for a list input; the blocks get one tensor.",
        "vtp/models/layers/block.py::if self.training and effective_drop_ratio > 0.0:": "Training-only stochastic depth.",
        "vtp/models/layers/embeddings.py::if isinstance(x, tuple):": "Every published patch size is an int.",
        "vtp/models/layers/embeddings.py::if not self.flatten_embedding:": "VTP flattens its patch embedding.",
        "vtp/models/layers/embeddings.py::if self.proj.bias is not None:": "The patch projection is biased.",
        "vtp/models/layers/embeddings.py::if (base is None and not both_periods) or (base is not None and both_periods):": "A misconfiguration guard; VTP configures a base.",
        "vtp/models/layers/embeddings.py::if self.base is not None:": "VTP configures a base, not explicit periods.",
        'vtp/models/layers/embeddings.py::if self.normalize_coords == "max":': "VTP normalizes each side separately, the one mode the port implements.",
        'vtp/models/layers/embeddings.py::elif self.normalize_coords == "min":': "VTP normalizes each side separately, the one mode the port implements.",
        'vtp/models/layers/embeddings.py::elif self.normalize_coords == "separate":': "VTP normalizes each side separately, the one mode the port implements.",
        "vtp/models/layers/embeddings.py::if self.training and self.shift_coords is not None:": "Training-only coordinate augmentation.",
        "vtp/models/layers/embeddings.py::if self.training and self.jitter_coords is not None:": "Training-only coordinate augmentation.",
        "vtp/models/layers/embeddings.py::if self.training and self.rescale_coords is not None:": "Training-only coordinate augmentation.",
    },
    port_unreached={
        "model/vision_ae/vtp.py::elif isinstance(module, nn.Embedding):": "Mirrors the reference's branch; VTP holds no embedding tables.",
        "model/vision_ae/vtp.py::if conv.bias is not None:": "Mirrors the reference's guard; both convolutions are biased.",
        "model/vision_ae/vtp.py::if self.proj.bias is not None:": "Mirrors the reference's guard; the patch projection is biased.",
        "model/vision_ae/vtp.py::if embed_dim % (4 * num_heads):": "A misconfiguration guard; valid configs never take it.",
        "model/vision_ae/vtp.py::if height % self.patch_size or width % self.patch_size:": "Raises on a side off the patch grid; vtp_test drives it.",
    },
    extra_outputs={},
    uncompared={
        "trunk": (
            "The reference trunk returns a dict of tokens; its patch tokens, "
            "reshaped, are the compared latent."
        ),
    },
    reference_calls={
        "trunk.feature_bottleneck": (
            (1,),
            (
                "The reference also projects the CLS token first, which "
                "reconstruction discards; its second call, the patches, is the port's."
            ),
        ),
    },
)


@dataclass(frozen=True, slots=True)
class Definition:
    """One function, method, or class, and the body lines that are its own."""

    key: str
    path: Path
    function: bool
    lines: frozenset[int]


def definitions(root: Path, relative: str) -> list[Definition]:
    """Return every definition in a file, nested ones included.

    Args:
      root: Directory ``relative`` resolves against.
      relative: The file, as inventory keys spell it.

    Returns:
      definitions: One per ``def`` and ``class``, keyed ``"<relative>::<qualname>"``.

    """
    path = root / relative
    found: list[Definition] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(
                child,
                ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
            ):
                continue
            name = f"{prefix}{child.name}"
            end = child.end_lineno or child.lineno
            own = set(range(child.body[0].lineno, end + 1))
            for inner in ast.walk(child):
                if inner is not child and isinstance(
                    inner,
                    ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
                ):
                    own -= set(
                        range(inner.lineno, (inner.end_lineno or inner.lineno) + 1),
                    )
            found.append(
                Definition(
                    key=f"{relative}::{name}",
                    path=path,
                    function=not isinstance(child, ast.ClassDef),
                    lines=frozenset(own),
                ),
            )
            walk(child, f"{name}.")

    walk(ast.parse(path.read_text()), "")
    return found


def inventory_problems(
    reference: Reference,
    *,
    clone: Path,
    measured: coverage.Coverage,
) -> list[str]:
    """Check the inventory against the reference files and the measured run.

    Args:
      reference: The pinned reference and its tables.
      clone: The reference's clone.
      measured: Coverage of one run of both sides, with branches.

    Returns:
      problems: Every gap, stale entry, and unexplained branch, one per line.

    """
    problems: list[str] = []
    theirs = {
        d.key: d for relative in reference.files for d in definitions(clone, relative)
    }
    ours: dict[str, Definition] = {}
    for keys in reference.paired.values():
        for key in keys:
            relative = key.split("::", 1)[0]
            ours.update({d.key: d for d in definitions(_PRIML, relative)})

    def not_ported(key: str) -> bool:
        return any(key == k or key.startswith(f"{k}.") for k in reference.not_ported)

    problems.extend(
        f"unpaired reference definition: {key}"
        for key in theirs
        if key not in reference.paired and not not_ported(key)
    )
    problems.extend(
        f"inventory names a definition the reference lacks: {key}"
        for key in (*reference.paired, *reference.not_ported)
        if key not in theirs
    )
    for keys in reference.paired.values():
        problems.extend(
            f"inventory names a port definition that does not exist: {key}"
            for key in keys
            if key not in ours
        )

    data = measured.get_data()

    def ran(definition: Definition) -> bool:
        return bool(definition.lines & set(data.lines(str(definition.path)) or ()))

    for key, definition in theirs.items():
        if not definition.function:
            continue
        if key in reference.paired and not ran(definition):
            problems.append(f"paired but never ran: {key}")
        if not_ported(key) and ran(definition):
            problems.append(f"not ported but ran, so it is on the path: {key}")
    port_keys = {key for keys in reference.paired.values() for key in keys}
    problems.extend(
        f"port definition never ran: {key}"
        for key in sorted(port_keys & ours.keys())
        if ours[key].function and not ran(ours[key])
    )

    problems += _branch_problems(
        [theirs[k] for k in reference.paired if k in theirs and theirs[k].function],
        root=clone,
        allowed=reference.reference_unreached,
        measured=measured,
    )
    problems += _branch_problems(
        [
            ours[k]
            for k in sorted(port_keys & ours.keys())
            if ours[k].function and k.startswith("model/vision_ae/")
        ],
        root=_PRIML,
        allowed=reference.port_unreached,
        measured=measured,
    )
    return problems


def _branch_problems(
    functions: Sequence[Definition],
    *,
    root: Path,
    allowed: Mapping[str, str],
    measured: coverage.Coverage,
) -> list[str]:
    """Report each branch in ``functions`` not taken both ways, less the allowlist."""
    problems: list[str] = []
    used: set[str] = set()
    for definition in functions:
        source = definition.path.read_text().splitlines()
        relative = definition.path.relative_to(root).as_posix()
        for line, (total, taken) in measured.branch_stats(str(definition.path)).items():
            if line not in definition.lines or taken >= total:
                continue
            text = source[line - 1].strip()
            if f"{relative}::{text}" in allowed:
                used.add(f"{relative}::{text}")
                continue
            problems.append(
                f"{definition.key}:{line} takes {taken} of {total} exits: {text}",
            )
    problems.extend(
        f"allowlisted branch now goes both ways, or no longer exists: {key}"
        for key in allowed
        if key not in used
    )
    return problems


class PinnedRandom:
    """Serve ``torch.randn`` from stored draws; refuse every other random primitive."""

    _REFUSED: Final = (
        "rand",
        "rand_like",
        "randn_like",
        "randint",
        "randint_like",
        "randperm",
        "normal",
        "bernoulli",
        "multinomial",
        "poisson",
    )
    _REFUSED_METHODS: Final = (
        "normal_",
        "uniform_",
        "bernoulli_",
        "random_",
        "exponential_",
        "geometric_",
        "log_normal_",
        "cauchy_",
    )

    def __init__(self, draws: Sequence[Tensor]) -> None:
        self._draws = list(draws)
        self.sites: Counter[str] = Counter()
        self._stack = ExitStack()

    def __enter__(self) -> Self:
        """Patch the primitives."""
        self._patch(torch, "randn", self._randn)
        for name in self._REFUSED:
            self._patch(torch, name, _refuse(name))
        for name in self._REFUSED_METHODS:
            self._patch(torch.Tensor, name, _refuse(name))
        return self

    def __exit__(self, *exc: object) -> None:
        """Restore the primitives and require every draw consumed."""
        self._stack.close()
        if self._draws and exc[0] is None:
            raise AssertionError(f"{len(self._draws)} pinned draws were never taken")

    def _patch(self, owner: object, name: str, value: object) -> None:
        original = cast("object", getattr(owner, name))
        setattr(owner, name, value)
        self._stack.callback(setattr, owner, name, original)

    def _randn(self, *size: object, **kwargs: object) -> Tensor:
        caller = inspect.currentframe()
        caller = caller.f_back if caller is not None else None
        if caller is not None:
            self.sites[f"{Path(caller.f_code.co_filename).name}:{caller.f_lineno}"] += 1
        shape = tuple(cast("Sequence[int]", size[0])) if len(size) == 1 else size
        if not self._draws:
            raise AssertionError(f"an unpinned torch.randn{shape} was drawn")
        draw = self._draws.pop(0)
        if tuple(draw.shape) != shape:
            raise AssertionError(
                f"torch.randn{shape} drawn; the pinned draw is {tuple(draw.shape)}",
            )
        return draw.to(device=cast("torch.device | str | None", kwargs.get("device")))


def _refuse(name: str) -> Callable[..., object]:
    """Return a stand-in that fails naming the primitive."""

    def refused(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError(f"unpinned random primitive torch.{name} was called")

    return refused


type Record = dict[str, list[list[Tensor]]]
"""Module name -> one entry per call -> that call's output tensors, in order."""


@contextmanager
def recorded(model: nn.Module) -> Generator[Record]:
    """Record every submodule's tensor outputs, per call, by module name."""
    outputs: Record = defaultdict(list)
    handles = [
        module.register_forward_hook(_recorder(outputs[name]))
        for name, module in model.named_modules()
        if name
    ]
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def _recorder(into: list[list[Tensor]]) -> Callable[[nn.Module, object, object], None]:
    def hook(module: nn.Module, args: object, output: object) -> None:
        del module, args
        into.append([t.detach().clone() for t in _tensors(output)])

    return hook


def _tensors(value: object) -> list[Tensor]:
    """Flatten a module output into its tensors, in order."""
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [
            t
            for item in cast("Mapping[str, object]", value).values()
            for t in _tensors(item)
        ]
    if isinstance(value, list | tuple):
        return [t for item in cast("Sequence[object]", value) for t in _tensors(item)]
    return []


def compare(
    label: str,
    theirs: Record,
    ours: Record,
    *,
    allowances: Reference,
) -> tuple[list[str], int]:
    """Compare every name both records share, exactly.

    Args:
      label: Prefix for each reported line.
      theirs: The reference's record.
      ours: The port's record.
      allowances: The reference whose ``extra_outputs``, ``uncompared``, and
        ``reference_calls`` name each structural difference, with a reason.

    Returns:
      problems: One line per differing call count, output count, dtype,
        shape, or bits.
      compared: How many names both sides recorded.

    """
    problems: list[str] = []
    shared = sorted(theirs.keys() & ours.keys())
    for name in shared:
        if name in allowances.uncompared:
            continue
        want_calls, got_calls = theirs[name], ours[name]
        if name in allowances.reference_calls:
            keep, _ = allowances.reference_calls[name]
            want_calls = [want_calls[i] for i in keep if i < len(want_calls)]
        if len(want_calls) != len(got_calls):
            problems.append(
                f"{label} {name}: {len(got_calls)} calls vs {len(want_calls)}",
            )
            continue
        for call, (want, got) in enumerate(zip(want_calls, got_calls, strict=True)):
            if len(want) != len(got) and name not in allowances.extra_outputs:
                problems.append(
                    f"{label} {name} call {call}: {len(got)} outputs vs {len(want)}",
                )
            for index, (a, b) in enumerate(zip(want, got, strict=False)):
                where = f"{label} {name} call {call} output {index}"
                if a.dtype != b.dtype or a.shape != b.shape:
                    problems.append(
                        f"{where}: {b.dtype}{list(b.shape)} vs {a.dtype}{list(a.shape)}",
                    )
                elif not torch.equal(a, b):
                    problems.append(
                        f"{where}: {int((a != b).sum())}/{a.numel()} differ",
                    )
    return problems, len(shared)


def state_problems(label: str, theirs: nn.Module, ours: nn.Module) -> list[str]:
    """Require identical state dicts: the same names, in order, with the same bits."""
    want, got = theirs.state_dict(), ours.state_dict()
    if list(want) != list(got):
        return [
            f"{label}: state keys differ: {list(got)[:4]}... vs {list(want)[:4]}...",
        ]
    return [
        f"{label}: {key} differs"
        for key in want
        if want[key].dtype != got[key].dtype or not torch.equal(want[key], got[key])
    ]


def clone_upstream(reference: Reference, root: Path) -> Path:
    """Fetch the reference at its pinned commit, or verify an existing clone.

    Args:
      reference: What to fetch.
      root: Directory the clone lives in.

    Returns:
      path: The clone's path.

    Raises:
      RuntimeError: An existing clone is dirty or at another commit, so what
        it contains is no longer the reference this comparison names.

    """
    if not (root / ".git").is_dir():
        root.mkdir(parents=True, exist_ok=True)
        for arguments in (
            ("init", "--quiet"),
            ("fetch", "--quiet", "--depth", "1", reference.url, reference.commit),
            ("checkout", "--quiet", "FETCH_HEAD"),
        ):
            _ = _git(root, *arguments)
    head = _git(root, "rev-parse", "HEAD")
    if head != reference.commit:
        raise RuntimeError(f"{root} is at {head}, expected {reference.commit}")
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise RuntimeError(f"{root} has local modifications:\n{dirty}")
    return root.resolve()


def _git(root: Path, *arguments: str) -> str:
    """Run a git command in the clone and return its output."""
    return subprocess.run(  # noqa: S603 -- The parity harness runs fixed git subcommands against the pinned reference.
        ["git", *arguments],  # noqa: S607 -- The parity harness runs fixed git subcommands against the pinned reference.
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@dataclass(slots=True, kw_only=True)
class Outcome:
    """What one model's comparison found."""

    problems: list[str]
    compared: int = 0
    sites: Counter[str] | None = None
    golden: dict[str, Tensor] | None = None


def run_invae(clone: Path, work: Path) -> Outcome:
    """Compare INVAE with REG's ``AutoencoderKL`` at the published f16d32 architecture.

    Args:
      clone: REG's clone.
      work: Scratch directory for the shared checkpoint.

    Returns:
      outcome: Mismatches, the modules compared, and the randn call sites.

    """
    _ = sys.modules.setdefault("dictdot", _dictdot_stub())
    sys.path.insert(0, str(clone))
    upstream = cast("_InvaeModule", importlib.import_module("models.invae"))
    outcome = Outcome(problems=[], sites=Counter())

    torch.manual_seed(0)
    theirs = upstream.VAE_F16D32()
    their_invae = cast("_TheirInvae", theirs)
    torch.manual_seed(0)
    unloaded = INVAE.Config(checkpoint=None).make()
    outcome.problems += state_problems("invae init", theirs, unloaded)

    checkpoint = work / "invae.pt"
    torch.save(theirs.state_dict(), checkpoint)
    ours = INVAE.Config(checkpoint=LocalFile.Config(path=checkpoint)).make()
    theirs.eval()
    generator = torch.Generator().manual_seed(0)
    for index in range(2):
        image = torch.randint(
            0,
            256,
            (2, 3, 32, 32),
            generator=generator,
            dtype=torch.uint8,
        )
        for draw in range(3):
            noise = torch.randn(2, 32, 2, 2, generator=generator)
            label = f"invae image {index} draw {draw}"
            with torch.no_grad(), host_agnostic_numerics():
                with PinnedRandom([noise]) as pinned, recorded(theirs) as their_modules:
                    # REG's own preprocessing: preprocessing/encoders.py, InvaeEncoder.encode.
                    upstream_posterior = their_invae.encode(
                        image.to(torch.float32) / 127.5 - 1,
                    )
                    their_sample = upstream_posterior.sample()
                    # REG's decode: generate.py, ``(samples + 1) / 2`` then a clamp.
                    their_decoded = (
                        (their_invae.decode(their_sample).sample + 1) / 2
                    ).clamp(0, 1)
                assert outcome.sites is not None
                outcome.sites.update(pinned.sites)
                with PinnedRandom([noise]) as pinned, recorded(ours) as our_modules:
                    our_sample = ours.encode(image)
                    our_decoded = ours.decode(our_sample)
                outcome.sites.update(pinned.sites)
                our_posterior = ours.posterior(image)
            fields = ("mean", "logvar", "std", "var")
            problems, compared = compare(
                label,
                {
                    **their_modules,
                    **{f: [[getattr(upstream_posterior, f)]] for f in fields},
                    "mode": [[upstream_posterior.mode()]],
                    "sample": [[their_sample]],
                    "decoded": [[their_decoded]],
                },
                {
                    **our_modules,
                    **{f: [[getattr(our_posterior, f)]] for f in fields},
                    "mode": [[posterior_mode(our_posterior)]],
                    "sample": [[our_sample]],
                    "decoded": [[our_decoded]],
                },
                allowances=_INVAE,
            )
            outcome.problems += problems
            outcome.compared = compared
    return outcome


def _dictdot_stub() -> types.ModuleType:
    """Supply ``dictdot.dictdot``: a dict whose keys read as attributes."""
    module = types.ModuleType("dictdot")

    class dictdot(dict[str, object]):  # noqa: N801 -- The name REG imports.
        def __getattr__(self, name: str) -> object:
            return self[name]

    module.__dict__["dictdot"] = dictdot
    return module


def run_rae(clone: Path, work: Path) -> Outcome:
    """Compare RAE with the reference ``stage1.RAE`` on its tiny parity fixture.

    The fixture is ``rae_test.tiny``'s geometry: width 8, a 2x2 latent grid,
    8px out, with a 16px input that drives the bicubic resize and an 8px one
    that skips it. Latent normalization is compared with and without a mean.

    Args:
      clone: The RAE clone.
      work: Scratch directory for the tiny checkpoints and configs.

    Returns:
      outcome: Mismatches, the modules compared, and the golden record.

    """
    sys.path.insert(0, str(clone / "src"))
    stage1 = cast("_Stage1Module", importlib.import_module("stage1"))
    upstream_decoder = cast(
        "_DecoderModule",
        importlib.import_module("stage1.decoders.decoder"),
    )
    utils = cast(
        "_DecoderUtilsModule",
        importlib.import_module("stage1.decoders.utils"),
    )
    import transformers  # noqa: PLC0415 -- Heavy; only this model needs it.

    outcome = Outcome(problems=[], sites=Counter())
    encoder_dir, decoder_dir = work / "enc", work / "dec"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    decoder_dir.mkdir(parents=True, exist_ok=True)

    hf = transformers.Dinov2WithRegistersModel(
        transformers.Dinov2WithRegistersConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            mlp_ratio=4,
            hidden_act="gelu",
            layer_norm_eps=1e-6,
            image_size=12,
            patch_size=4,
            num_channels=3,
            qkv_bias=True,
            layerscale_value=1.0,
            num_register_tokens=4,
            use_swiglu_ffn=False,
        ),
    )
    _randomize(hf, seed=1)
    hf.save_pretrained(encoder_dir)
    _ = (encoder_dir / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "crop_size": {"height": 224, "width": 224},
                "do_center_crop": True,
                "do_convert_rgb": True,
                "do_normalize": True,
                "do_rescale": True,
                "do_resize": True,
                "image_mean": [0.485, 0.456, 0.406],
                "image_processor_type": "BitImageProcessor",
                "image_std": [0.229, 0.224, 0.225],
                "resample": 3,
                "rescale_factor": 0.00392156862745098,
                "size": {"shortest_edge": 256},
            },
        ),
    )
    decoder_json = cast(
        "dict[str, object]",
        json.loads((clone / "configs/decoder/ViTXL/config.json").read_text()),
    )
    decoder_json.update(
        decoder_hidden_size=8,
        decoder_intermediate_size=16,
        decoder_num_attention_heads=2,
        decoder_num_hidden_layers=1,
        patch_size=4,
    )
    _ = (decoder_dir / "config.json").write_text(json.dumps(decoder_json))

    def build(decoder: Path | None, stats: Path | None) -> nn.Module:
        return stage1.RAE(
            encoder_cls="Dinov2withNorm",
            encoder_config_path=str(encoder_dir),
            encoder_input_size=8,
            encoder_params={"dinov2_path": str(encoder_dir), "normalize": True},
            decoder_config_path=str(decoder_dir),
            decoder_patch_size=4,
            pretrained_decoder_path=None if decoder is None else str(decoder),
            noise_tau=0.0,
            reshape_to_2d=True,
            normalization_stat_path=None if stats is None else str(stats),
        )

    bootstrap = cast("_TheirRae", build(None, None))
    _randomize(bootstrap.decoder, seed=2)
    torch.save(bootstrap.decoder.state_dict(), work / "decoder.pt")

    config = _rae_tiny()
    assert isinstance(config.encoder, Dinov2WithRegisters.Config)
    config.encoder.checkpoint = LocalFile.Config(path=encoder_dir / "model.safetensors")
    config.decoder.checkpoint = LocalFile.Config(path=work / "decoder.pt")
    ours = config.make()

    mean = torch.randn(8, 2, 2, generator=torch.Generator().manual_seed(3))
    var = torch.rand(8, 2, 2, generator=torch.Generator().manual_seed(4)) + 0.5
    torch.save({"mean": mean, "var": var}, work / "stats.pt")
    torch.save({"mean": None, "var": var}, work / "stats_var.pt")

    generator = torch.Generator().manual_seed(0)
    large = torch.randint(
        0,
        256,
        (2, 3, 16, 16),
        generator=generator,
        dtype=torch.uint8,
    )
    exact = torch.randint(
        0,
        256,
        (2, 3, 8, 8),
        generator=torch.Generator().manual_seed(5),
        dtype=torch.uint8,
    )
    for stats in (None, work / "stats.pt", work / "stats_var.pt"):
        theirs = build(work / "decoder.pt", stats).eval()
        # The one reference-side change: the kernel the port fixes, through HF's setter.
        their_rae = cast("_TheirRae", theirs)
        their_rae.encoder.encoder.set_attn_implementation("eager")
        outcome.problems += state_problems("rae weights", theirs, ours)
        norm = None
        if stats is not None:
            norm = ElementwiseLatentStats.Config(
                stats=LocalFile.Config(path=stats),
            ).make()
        for name, image in (("16px", large), ("8px", exact)):
            label = f"rae {name} stats={None if stats is None else stats.name}"
            with torch.no_grad(), host_agnostic_numerics(), PinnedRandom([]):
                with recorded(theirs) as their_modules:
                    their_latent = their_rae.encode(image.float() / 255)
                    their_decoded = their_rae.decode(their_latent)
                with recorded(ours) as our_modules:
                    raw = ours.encode(image)
                    our_latent = raw if norm is None else norm.normalize(raw)
                    our_decoded = ours.decode(
                        our_latent if norm is None else norm.denormalize(our_latent),
                    )
            problems, compared = compare(
                label,
                {
                    **their_modules,
                    "latent": [[their_latent]],
                    "decoded": [[their_decoded.clamp(0, 1)]],
                },
                {**our_modules, "latent": [[our_latent]], "decoded": [[our_decoded]]},
                allowances=_RAE,
            )
            outcome.problems += problems
            outcome.compared = max(outcome.compared, compared)
            if stats is None and name == "16px":
                outcome.golden = {
                    **{f"state.{k}": v for k, v in ours.state_dict().items()},
                    "image": image,
                    "latent": their_latent.float(),
                    "decoded": their_decoded.float(),
                }

    upstream_config = utils.ViTMAEConfig(
        **{
            k: v
            for k, v in decoder_json.items()
            if not k.startswith("_")
            and k
            not in {
                "architectures",
                "model_type",
                "torch_dtype",
                "transformers_version",
            }
        },
    )
    upstream_config.hidden_size = 8
    upstream_config.image_size = 8
    torch.manual_seed(7)
    their_decoder = upstream_decoder.GeneralDecoder(upstream_config, num_patches=4)
    torch.manual_seed(7)
    our_decoder = GeneralDecoder.Config(
        channels_in=8,
        num_patches=4,
        channels_hidden=8,
        channels_hidden_mlp=16,
        num_layers=1,
        heads=2,
        patch_size=4,
    ).make()
    outcome.problems += state_problems("rae decoder init", their_decoder, our_decoder)
    return outcome


def _rae_tiny() -> RAE.Config:
    """Return ``rae_test.tiny``'s geometry, which the checked-in golden is minted at."""
    config = RAE.Config()
    encoder = config.encoder
    assert isinstance(encoder, Dinov2WithRegisters.Config)
    encoder.channels_hidden = 8
    encoder.num_layers = 1
    encoder.heads = 2
    encoder.patch_size = 4
    encoder.image_size = 12
    config.encoder_image_size = 8
    config.decoder.channels_hidden = 8
    config.decoder.channels_hidden_mlp = 16
    config.decoder.num_layers = 1
    config.decoder.heads = 2
    config.decoder.patch_size = 4
    config.image_size = 8
    return config


def _randomize(module: nn.Module, *, seed: int) -> None:
    """Draw every parameter from ``N(0, 0.3 ** 2)``: weights far from any init."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)


def run_vtp(clone: Path, work: Path) -> Outcome:
    """Compare VTP with ``VTPModel``'s reconstruction path on ``vtp_test.tiny``'s geometry.

    Args:
      clone: The VTP clone.
      work: Scratch directory for the shared checkpoint.

    Returns:
      outcome: Mismatches, the modules compared, and the golden record.

    """
    for name, path in (
        ("vtp", clone / "vtp"),
        ("vtp.models", clone / "vtp" / "models"),
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    sys.path.insert(0, str(clone))
    decoders = cast("_DecoderModule", importlib.import_module("vtp.models.decoders"))
    hf = cast("_VtpHfModule", importlib.import_module("vtp.models.vtp_hf"))
    outcome = Outcome(problems=[], sites=Counter())

    def their_decoder() -> nn.Module:
        # VTPModel builds its decoder at ffn ratio 4 and upscale 16; the same
        # constructor at the golden's ratio 1 and upscale 4.
        return decoders.DinoV3PixelDecoder(
            in_chans=4,
            embed_dim=16,
            num_heads=2,
            depth=1,
            ffn_ratio=1.0,
            ffn_layer="swiglu",
            norm_layer="layernorm",
            upscale_factor=4,
        )

    # Initialization, at a geometry VTPModel builds natively: its decoder's
    # ffn ratio is 4 and its upscale the patch, 16.
    torch.manual_seed(0)
    their_init = hf.VTPModel(
        hf.VTPConfig(
            train_clip=False,
            image_size=32,
            vision_patch_size=16,
            vision_embed_dim=16,
            vision_depth=1,
            vision_num_heads=2,
            vision_mlp_ratio=4.0,
            vision_feature_bottleneck=4,
            decoder_embed_dim=16,
            decoder_depth=1,
            decoder_num_heads=2,
        ),
    )
    native = VTP.Config()
    native.trunk.channels_hidden = 16
    native.trunk.num_layers = 1
    native.trunk.heads = 2
    native.trunk.channels_out = 4
    native.pixel_decoder.channels_hidden = 16
    native.pixel_decoder.num_layers = 1
    native.pixel_decoder.heads = 2
    native.image_size = 32
    native.checkpoint = None
    torch.manual_seed(0)
    outcome.problems += state_problems("vtp init", their_init, native.make())

    golden = load_golden(_TESTDATA / "vtp.pt")
    state = golden["state_dict"]
    save_file(
        {k: v.contiguous() for k, v in state.items()},
        str(work / "vtp.safetensors"),
    )
    theirs = hf.VTPModel(
        hf.VTPConfig(
            train_clip=False,
            image_size=8,
            vision_patch_size=4,
            vision_embed_dim=16,
            vision_depth=1,
            vision_num_heads=2,
            vision_mlp_ratio=1.0,
            vision_feature_bottleneck=4,
            decoder_embed_dim=16,
            decoder_depth=1,
            decoder_num_heads=2,
        ),
    )
    theirs.pixel_decoder = their_decoder()
    _ = theirs.load_state_dict(state, strict=True)
    theirs.eval()
    their_vtp = cast("_TheirVtp", theirs)
    config = _vtp_tiny()
    config.checkpoint = LocalFile.Config(path=work / "vtp.safetensors")
    ours = config.make()
    config.dtype_autocast = torch.bfloat16
    ours_bf16 = config.make()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = Normalize(mean, std)
    denormalize = Normalize(
        [-m / s for m, s in zip(mean, std, strict=True)],
        [1 / s for s in std],
    )

    def theirs_round_trip(
        image: Tensor,
        autocast: torch.dtype | None,
    ) -> dict[str, Tensor]:
        # The reference's evaluation, tools/test_reconstruction_hf.py: ToTensor and
        # Normalize in, the encoder under autocast, the inverse Normalize and a clamp out.
        pixels_in = torch.stack(
            [normalize(to_tensor(img.permute(1, 2, 0).numpy())) for img in image],
        )
        cast_context = (
            nullcontext() if autocast is None else torch.autocast("cpu", dtype=autocast)
        )
        with cast_context:
            latent = their_vtp.get_reconstruction_latents(pixels_in)
        pixels = their_vtp.get_latents_decoded_images(latent.float())
        return {
            "latent": latent,
            "pixels": pixels,
            "image": torch.clamp(denormalize(pixels), 0, 1),
        }

    golden_input = cast("Tensor", golden["input"])
    square = torch.randint(
        0,
        256,
        (2, 3, 8, 8),
        generator=torch.Generator().manual_seed(1),
        dtype=torch.uint8,
    )
    for label, image, model, autocast in (
        ("vtp golden input", golden_input, ours, None),
        ("vtp square input", square, ours, None),
        ("vtp bfloat16 autocast", golden_input, ours_bf16, torch.bfloat16),
    ):
        numerics = host_agnostic_numerics() if autocast is None else nullcontext()
        with torch.no_grad(), numerics, PinnedRandom([]):
            with recorded(theirs) as their_modules:
                expected = theirs_round_trip(image, autocast)
            with recorded(model) as our_modules:
                latent = model.encode(image)
                got = {"latent": latent, "image": model.decode(latent.float())}
        # The raw pixels are the recorded ``pixel_decoder`` output on both sides.
        problems, compared = compare(
            label,
            {
                **their_modules,
                "latent": [[expected["latent"]]],
                "image": [[expected["image"]]],
            },
            {**our_modules, **{k: [[v]] for k, v in got.items()}},
            allowances=_VTP,
        )
        outcome.problems += problems
        outcome.compared = max(outcome.compared, compared)
        if label == "vtp golden input":
            commit = _VTP.commit.encode()
            outcome.golden = {
                "source_commit": torch.frombuffer(bytearray(commit), dtype=torch.uint8),
                **{k: v.float() for k, v in expected.items()},
            }
    return outcome


def _vtp_tiny() -> VTP.Config:
    """Return ``vtp_test.tiny``'s geometry, which the checked-in golden is minted at."""
    config = VTP.Config()
    config.trunk.patch_size = 4
    config.trunk.channels_hidden = 16
    config.trunk.num_layers = 1
    config.trunk.heads = 2
    config.trunk.expansion = 1.0
    config.trunk.channels_out = 4
    config.pixel_decoder.channels_hidden = 16
    config.pixel_decoder.num_layers = 1
    config.pixel_decoder.heads = 2
    config.pixel_decoder.expansion = 1.0
    config.image_size = 8
    config.checkpoint = None
    return config


_MODELS: Final = {
    "invae": (_INVAE, run_invae, None),
    "rae": (_RAE, run_rae, "rae_reference.pt"),
    "vtp": (_VTP, run_vtp, "vtp_reference.pt"),
}


def main() -> int:
    """Run the selected comparisons and print every problem found.

    Returns:
      status: 0 when every selected port matches its reference, else 1.

    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    _ = parser.add_argument(
        "--model",
        nargs="+",
        choices=sorted(_MODELS),
        default=sorted(_MODELS),
        help="Which ports to compare.",
    )
    _ = parser.add_argument(
        "--clone-dir",
        type=Path,
        default=get_cache_dir() / "reference",
        help="Where the pinned reference clones live.",
    )
    _ = parser.add_argument(
        "--mint",
        action="store_true",
        help="Rewrite the checked-in reference goldens instead of checking them.",
    )
    flags = parser.parse_args()
    models = cast("list[str]", flags.model)
    clone_dir = cast("Path", flags.clone_dir)
    mint = cast("bool", flags.mint)

    clones = {
        name: clone_upstream(_MODELS[name][0], clone_dir / name) for name in models
    }
    include = [str(clones[name] / f) for name in models for f in _MODELS[name][0].files]
    include += [
        str(_PRIML / "model" / "vision_ae" / "*.py"),
        str(_PRIML / "math" / "*.py"),
    ]
    measured = coverage.Coverage(
        branch=True,
        data_file=None,
        include=include,
        config_file=False,
    )
    failed = False
    with tempfile.TemporaryDirectory() as scratch:
        for name in models:
            reference, run, golden_name = _MODELS[name]
            work = Path(scratch) / name
            work.mkdir()
            measured.start()
            try:
                outcome = run(clones[name], work)
            finally:
                measured.stop()
            problems = list(outcome.problems)
            if golden_name is not None and outcome.golden is not None:
                path = _TESTDATA / golden_name
                if mint:
                    write_tensors(path, outcome.golden)
                else:
                    problems += [
                        f"{golden_name}: {line}"
                        for line in mismatches(read_tensors(path), outcome.golden)
                    ]
            problems += inventory_problems(
                reference,
                clone=clones[name],
                measured=measured,
            )
            print(
                f"== {name} at {reference.commit[:7]}: {outcome.compared} module outputs compared",
            )
            if outcome.sites:
                print(f"   torch.randn call sites: {dict(outcome.sites)}")
            for problem in problems:
                print(f"   {problem}")
            print(f"   {'FAIL' if problems else 'ok'}: {len(problems)} problems")
            failed |= bool(problems)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
