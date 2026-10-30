"""Protocols an autoencoder and the distribution its encoder returns satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    import torch


@runtime_checkable
class Distribution(Protocol):
    """A distribution over codes; a deterministic encoder returns a point mass."""

    def sample(self, *, generator: torch.Generator | None = None) -> Tensor:
        """Return a draw, ``[..., *latent]``; ``None`` reads the global generator."""
        ...

    def mode(self) -> Tensor:
        """Return the most likely code, ``[..., *latent]``."""
        ...


@runtime_checkable
class Encoder(Protocol):
    """Encodes inputs to a distribution over codes.

    Leading dimensions ``...`` are arbitrary and pass through unchanged.
    """

    def encode(self, x: Tensor, /) -> Distribution:
        """Return the code distribution: batch ``[...]``, event ``[*latent]``."""
        ...


@runtime_checkable
class Decoder(Protocol):
    """Decodes codes to inputs.

    Leading dimensions ``...`` are arbitrary and pass through unchanged.
    """

    def decode(self, z: Tensor, /) -> Tensor:
        """Return the reconstruction ``[..., *input]`` of codes ``[..., *latent]``."""
        ...


@runtime_checkable
class Autoencoder(Encoder, Decoder, Protocol):
    """Both an encoder and a decoder."""


class StateDictRemap(Protocol):
    """Renames one half's published tensors to that half's ``state_dict`` names.

    One per checkpoint layout, so a half reads any layout its config names.
    """

    def __call__(
        self,
        state: Mapping[str, Tensor],
        /,
        *,
        half: str,
    ) -> dict[str, Tensor]:
        """Return ``half``'s tensors renamed; keys outside ``half`` are ignored."""
        ...
