"""Quantization strategies for training.

Float8 quantization converts nn.Linear → Float8Linear using torchao.
Requires torchao>=0.9.0 and compute capability >= 8.9 (H100, 5090, etc.).
"""

# ruff: noqa: PLC0415  # Lazy imports for optional torchao dependency

from __future__ import annotations

from collections.abc import Callable

import dataclasses
import logging

from configgle import Fig
from torch import nn


logger = logging.getLogger(__name__)


class NoModelQuantization:
    """No quantization (default)."""

    class Config(Fig["NoModelQuantization"]):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(self, model: nn.Module) -> nn.Module:
        return model


class Float8ModelQuantization:
    """Float8 linear layer quantization via torchao.

    Converts nn.Linear → Float8Linear for memory + compute efficiency.
    Requires H100/5090+ GPUs (SM89+).

    Compatible with torch.compile and FSDP.
    """

    class Config(Fig["Float8ModelQuantization"]):
        recipe: str = "tensorwise"
        """Quantization recipe ("tensorwise", "rowwise", "rowwise_gw_hp")."""
        module_filter: Callable[[nn.Module, str], bool] | None = None
        """Optional filter to select which modules to quantize."""
        enable_fsdp_float8_all_gather: bool = False
        """Use float8 all-gather in FSDP (tensorwise only)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        # Initialize all instance attributes up front so the disabled path
        # leaves a consistent object (no AttributeError on later access).
        self.module_filter = config.module_filter
        self.converted_count = 0
        self.enabled = False

        available, reason = _check_float8_available()
        if not available:
            logger.warning(f"Float8 disabled: {reason}")
            return

        from torchao.float8.config import (
            Float8LinearConfig,
            Float8LinearRecipeName,
        )

        recipe_map = {
            "tensorwise": Float8LinearRecipeName.TENSORWISE,
            "rowwise": Float8LinearRecipeName.ROWWISE,
            "rowwise_gw_hp": Float8LinearRecipeName.ROWWISE_WITH_GW_HP,
        }

        if config.recipe not in recipe_map:
            raise ValueError(
                f"Invalid recipe '{config.recipe}'. "
                f"Choose from: {list(recipe_map.keys())}",
            )

        recipe_name = recipe_map[config.recipe]

        if (
            config.enable_fsdp_float8_all_gather
            and recipe_name != recipe_map["tensorwise"]
        ):
            raise ValueError(
                "enable_fsdp_float8_all_gather only works with recipe='tensorwise'",
            )

        self.float8_config = Float8LinearConfig.from_recipe_name(recipe_name)

        # Override FSDP all-gather setting on the frozen dataclass.
        if config.enable_fsdp_float8_all_gather:
            self.float8_config = dataclasses.replace(
                self.float8_config,
                enable_fsdp_float8_all_gather=True,
            )

        self.enabled = True

        logger.info(
            f"Float8Linear enabled: recipe={config.recipe}, "
            f"fsdp_all_gather={config.enable_fsdp_float8_all_gather}",
        )

    def _default_module_filter(self, mod: nn.Module, _fqn: str) -> bool:
        """Default filter: all nn.Linear with shapes divisible by 16.

        Pure predicate -- conversion counting happens in ``__call__``.
        """
        if not isinstance(mod, nn.Linear):
            return False
        # Float8 matmul requires dimensions divisible by 16 (hardware constraint).
        return mod.weight.shape[0] % 16 == 0 and mod.weight.shape[1] % 16 == 0

    def __call__(self, model: nn.Module) -> nn.Module:
        if not self.enabled:
            return model

        from torchao.float8 import (
            convert_to_float8_training,
        )

        filter_fn = self.module_filter or self._default_module_filter
        # Count conversions by applying the (pure) filter ourselves, so the
        # tally is correct regardless of how many times torchao invokes it.
        self.converted_count = sum(
            1 for name, mod in model.named_modules() if filter_fn(mod, name)
        )

        # No autocast wrapping: torchao's ``Float8Linear.forward`` REQUIRES the
        # surrounding bf16 autocast to stay enabled -- it reads
        # ``torch.is_autocast_enabled()`` and casts its input to the autocast
        # dtype before the fp8 matmul (torchao float8_linear.py). Disabling
        # autocast around it left the input fp32 and dispatched an unsupported
        # ``aten.to.dtype`` onto the Float8 tensor subclass inside ``torch.mm``,
        # crashing the first step. The model already runs under bf16 autocast,
        # which is exactly what the fp8 path expects.
        model = convert_to_float8_training(
            model,
            config=self.float8_config,
            module_filter_fn=filter_fn,
        )

        logger.info(f"Converted {self.converted_count} nn.Linear → Float8Linear")
        return model


def _check_float8_available() -> tuple[bool, str]:
    """Check if float8 training is available."""
    try:
        from torchao.float8 import (
            convert_to_float8_training,  # noqa: F401  # availability check
        )

        import torch
    except ImportError:
        return False, "torchao not installed"

    # Check compute capability (SM89+ required for float8)
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    cc_major, cc_minor = torch.cuda.get_device_capability()
    compute_version = cc_major * 10 + cc_minor

    if compute_version < 89:
        return (
            False,
            f"Float8 requires SM89+ (H100, 5090). Found SM{compute_version}",
        )

    return True, ""
