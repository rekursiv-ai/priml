"""Frozen DINOv2 representation targets for REG alignment."""

from __future__ import annotations

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, elementwise_cost, matmul_cost
from priml.hub import load_torch_hub_distributed
from priml.model.attention.kernel import attention_kernel_cost


class DinoV2Teacher(nn.Module):
    """Expose selected DINOv2 layers as CLS plus spatial patch tokens."""

    class Config(Fig["DinoV2Teacher"]):
        variant: str = "dinov2_vitb14"
        """Torch Hub DINOv2 backbone name."""
        layer_indices: tuple[int, ...] = (12, 12, 12)
        """Reference block indices for the three REG targets."""
        image_size: int = 256
        """Side length of the preprocessed teacher image."""

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Estimate the frozen ViT-B/14 forward through the requested layer."""
            del kwargs
            if self.variant != "dinov2_vitb14":
                raise ValueError(f"no cost model for DINOv2 variant {self.variant}")
            width, heads, mlp_width = 768, 12, 3072
            patches = (self.image_size // 16) ** 2
            tokens = patches + 1
            rows = batch_size * tokens
            block = (
                matmul_cost(
                    channels_in=width,
                    channels_out=3 * width,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                )
                + matmul_cost(
                    channels_in=width,
                    channels_out=width,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                )
                + attention_kernel_cost(
                    seq_len=tokens,
                    batch_size=batch_size,
                    dtype=dtype,
                    num_heads=heads,
                    channels_head=width // heads,
                )
                + matmul_cost(
                    channels_in=width,
                    channels_out=mlp_width,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                )
                + matmul_cost(
                    channels_in=mlp_width,
                    channels_out=width,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=8 * rows * width,
                    adjoint=0,
                    channels=width,
                    rows=rows,
                    dtype=dtype,
                )
            )
            patch = matmul_cost(
                channels_in=3 * 14 * 14,
                channels_out=width,
                bias=True,
                rows=batch_size * patches,
                dtype=dtype,
            )
            layers = min(12, max((1, *(index + 1 for index in self.layer_indices))))
            full = patch + block.tile(layers, copies=layers)
            # The teacher runs under no_grad; keep its parameters but only its
            # forward kernels in the training cost.
            return Cost(
                cells=full.only("primal").cells,
                params=full.params,
                params_active=full.params_active,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.image_size % 16:
            raise ValueError("image_size must be divisible by 16")
        self.config = config
        self.encoder = load_torch_hub_distributed(
            "facebookresearch/dinov2", config.variant
        )
        self.encoder.eval().requires_grad_(False)
        patch_grid = config.image_size // 16
        # Match the reference's 224/448-pixel input and resized DINO position table.
        position = self.encoder.pos_embed.detach()
        source_grid = int((position.shape[1] - 1) ** 0.5)
        patch = (
            position[:, 1:].reshape(1, source_grid, source_grid, -1).permute(0, 3, 1, 2)
        )
        patch = functional.interpolate(
            patch,
            size=(patch_grid, patch_grid),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        resized = torch.cat((position[:, :1], patch.flatten(2).transpose(1, 2)), dim=1)
        self.encoder.pos_embed = nn.Parameter(resized, requires_grad=False)

    @torch.no_grad()
    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        """Return CLS plus patch tokens at the requested DINO block indices."""
        size = 224 * (self.config.image_size // 256)
        if image.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError("teacher image size does not match configuration")
        image = image.float() / 255
        mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        image = functional.interpolate(
            (image - mean) / std,
            size=(size, size),
            mode="bicubic",
        )
        blocks = len(self.encoder.blocks)
        requested = tuple(max(0, min(i, blocks - 1)) for i in self.config.layer_indices)
        unique = sorted(set(requested))
        layers = self.encoder.get_intermediate_layers(
            image,
            n=unique,
            reshape=False,
            return_class_token=True,
        )
        by_layer = {
            index: torch.cat((cls[:, None], patches), dim=1)
            for index, (patches, cls) in zip(unique, layers, strict=True)
        }
        return tuple(by_layer[i] for i in requested)
