"""Initialize a model body from another run's checkpoint.

The checkpoint's EMA shadow overlays its live weights -- those are the weights
that run evaluated with. A tensor absent from this model, or shaped
differently, keeps its fresh init: across dataset builds that is the per-task
table, whose rows index a different task vocabulary. Loading reports every
tensor's outcome and refuses a checkpoint that matches nothing, which would
otherwise train silently from random weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import logging

from configgle import Fig

import torch

from priml.lib.codec import from_plain


if TYPE_CHECKING:
    from torch import Tensor, nn


logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True, frozen=True)
class WarmStartReport:
    """Per-tensor outcome of one warm start."""

    loaded: tuple[str, ...] = field(default_factory=tuple)
    """Model tensors taken from the checkpoint."""

    skipped_shape: tuple[str, ...] = field(default_factory=tuple)
    """Checkpoint tensors dropped for a shape mismatch."""

    skipped_missing: tuple[str, ...] = field(default_factory=tuple)
    """Checkpoint tensors this model has no counterpart for."""

    fresh: tuple[str, ...] = field(default_factory=tuple)
    """Model tensors the checkpoint did not cover; they keep their init."""


class WarmStart:
    """Load every matching tensor of a training checkpoint into a model."""

    class Config(Fig["WarmStart"]):
        """Which checkpoint to read."""

        path: Path | str = ""
        """A training checkpoint holding ``{"step": {"model": ..., "ema": ...}}``."""

        rename: Callable[[str], str] | None = None
        """Maps a checkpoint's tensor names to this model's; ``None`` keeps them.

        A checkpoint from an implementation that named its parameters otherwise
        needs this, or every renamed tensor silently keeps its fresh init."""

    def __init__(self, config: Config) -> None:
        self.path = Path(config.path).expanduser()
        self.rename = config.rename

    def __call__(self, model: nn.Module) -> WarmStartReport:
        """Initialize ``model``'s matching tensors in place.

        Args:
          model: Model to initialize; wrapper prefixes on either side are ignored.

        Returns:
          report: Which tensors loaded, were skipped, or stayed fresh.

        Raises:
          ValueError: If no tensor of the checkpoint fits the model.

        """
        # A checkpoint this project wrote; its nested step state needs full unpickling.
        checkpoint = from_plain(
            cast(object, torch.load(self.path, map_location="cpu", weights_only=False)),
            dict[str, object],
        )
        step = from_plain(checkpoint["step"], dict[str, object])
        raw_state = from_plain(step["model"], dict[str, object])
        state = {
            name: from_plain(value, torch.Tensor)
            for name, value in raw_state.items()
            if isinstance(value, torch.Tensor)
        }
        ema = step.get("ema")
        if ema:
            state.update(
                _ema_shadow(from_plain(ema, dict[str, object]), path=self.path),
            )
        own = model.state_dict()
        own_by_bare = {_bare(name): name for name in own}
        loadable: dict[str, Tensor] = {}
        skipped_shape: list[str] = []
        skipped_missing: list[str] = []
        for name, value in state.items():
            bare = _bare(name)
            target = own_by_bare.get(bare if self.rename is None else self.rename(bare))
            if target is None:
                skipped_missing.append(name)
            elif own[target].shape != value.shape:
                skipped_shape.append(name)
            else:
                loadable[target] = value
        if not loadable:
            raise ValueError(
                f"warm start from {self.path} matched no tensors; the checkpoint "
                "layout does not fit this model.",
            )
        model.load_state_dict(loadable, strict=False)
        report = WarmStartReport(
            loaded=tuple(sorted(loadable)),
            skipped_shape=tuple(sorted(skipped_shape)),
            skipped_missing=tuple(sorted(skipped_missing)),
            fresh=tuple(sorted(name for name in own if name not in loadable)),
        )
        logger.info(
            "warm start %s: loaded %d tensor(s); shape-skipped %s; missing %s; fresh %s",
            self.path,
            len(report.loaded),
            list(report.skipped_shape) or "none",
            list(report.skipped_missing) or "none",
            list(report.fresh) or "none",
        )
        return report


# ``EMA.state_dict`` stores ``shadow_params`` (param_dict kind) or ``shadow_model``
# (module kind); older checkpoints stored the shadow mapping flat.
def _ema_shadow(ema: dict[str, object], *, path: Path) -> dict[str, Tensor]:
    """Return the EMA shadow tensors, rejecting an EMA that holds none."""
    shadow = ema.get("shadow_params", ema.get("shadow_model", ema))
    tensors = {
        name: value
        for name, value in from_plain(shadow, dict[str, object]).items()
        if isinstance(value, torch.Tensor)
    }
    if not tensors:
        raise ValueError(
            f"warm start from {path}: the checkpoint's EMA state holds no tensors "
            f"(keys {sorted(ema)}); refusing to evaluate live weights instead.",
        )
    return tensors


_WRAPPER_PREFIXES: Final = ("module.", "_orig_mod.")


def _bare(name: str) -> str:
    """Strip data-parallel and compile wrapper prefixes, in any order and nesting."""
    while name.startswith(_WRAPPER_PREFIXES):
        name = next(
            name.removeprefix(prefix)
            for prefix in _WRAPPER_PREFIXES
            if name.startswith(prefix)
        )
    return name
