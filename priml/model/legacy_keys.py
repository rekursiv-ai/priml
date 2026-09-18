"""Absorb pre-rename state-dict keys at load time."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor, nn


def absorb_legacy_keys(module: nn.Module, renames: Mapping[str, str]) -> None:
    """Register a load hook rewriting ``old`` attribute prefixes to ``new``.

    A state dict is keyed by attribute path, so renaming a child attribute
    orphans every checkpoint minted before the rename. The hook rewrites
    ``<prefix><old>.<rest>`` to ``<prefix><new>.<rest>`` before torch matches
    keys, so ``state_dict()`` writes the new names while ``load_state_dict``
    accepts both.

    Args:
      module: The module whose direct children were renamed.
      renames: ``old_attr -> new_attr`` for each renamed child.

    """

    def hook(
        module: nn.Module,
        state_dict: dict[str, Tensor],
        prefix: str,
        *hook_args: object,
    ) -> None:
        del module, hook_args
        for old, new in renames.items():
            head = f"{prefix}{old}."
            for key in [k for k in state_dict if k.startswith(head)]:
                state_dict[f"{prefix}{new}.{key.removeprefix(head)}"] = state_dict.pop(
                    key,
                )

    module.register_load_state_dict_pre_hook(hook)
