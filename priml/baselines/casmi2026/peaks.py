"""Peak cleaning and FPNet input packing shared by training and the neural bank."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from torch import Tensor

import numpy as np
import torch


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class FPNetInputs(TypedDict):
    """The nine FPNet arguments, keyed by name in call order."""

    mz: Tensor
    it: Tensor
    pad: Tensor
    prec: Tensor
    ad: Tensor
    ce: Tensor
    ce_known: Tensor
    merged: Tensor
    mode: Tensor


def prepare_peaks(
    mass: NDArray[np.floating],
    intensity: NDArray[np.floating],
    *,
    precursor: float,
    max_peaks: int,
    floor: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Mirror the reference peak preparation without changing its NumPy arithmetic.

    Args:
      mass: Raw peak mass-to-charge values.
      intensity: Raw peak intensities aligned with ``mass``.
      precursor: Measured precursor m/z; peaks above it by more than two are dropped.
      max_peaks: Maximum number of intensity-selected peaks.
      floor: Relative intensity cutoff after precursor filtering.

    Returns:
      masses: Sorted retained float32 peak masses.
      values: Float32 square-root intensities relative to the eligible maximum.

    """
    masses, values = np.asarray(mass, np.float64), np.asarray(intensity, np.float64)
    allowed = (masses > 0) & (masses <= precursor + 2.0)
    masses, values = masses[allowed], values[allowed]
    if not len(masses) or values.max() <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    values = values / values.max()
    selected = values >= floor
    masses, values = masses[selected], values[selected]
    if len(masses) > max_peaks:
        selected = np.argsort(-values, kind="stable")[:max_peaks]
        masses, values = masses[selected], values[selected]
    order = np.argsort(masses, kind="stable")
    return masses[order].astype(np.float32), np.sqrt(values[order]).astype(np.float32)


def pack_inputs(
    masses: Sequence[NDArray[np.float32]],
    intensities: Sequence[NDArray[np.float32]],
    *,
    precursor: NDArray[np.float32],
    adduct: NDArray[np.int64],
    energy: NDArray[np.float32],
    energy_known: NDArray[np.float32],
    merged_count: NDArray[np.float32],
    mode: NDArray[np.float32],
) -> FPNetInputs:
    """Pack the nine FPNet arguments, keeping one unmasked slot for empty spectra.

    Args:
      masses: Prepared peak masses of each spectrum.
      intensities: Prepared square-root intensities aligned with ``masses``.
      precursor: Precursor mass-to-charge per spectrum.
      adduct: Reference adduct-vocabulary index per spectrum.
      energy: Collision energy per spectrum, zero where unknown.
      energy_known: One where ``energy`` was measured, otherwise zero.
      merged_count: Spectra merged into each item, at most eight.
      mode: Positive one or negative one for ionization polarity.

    Returns:
      inputs: CPU tensors keyed by FPNet argument name, in call order.

    """
    width = max(1, *(len(mass) for mass in masses))
    padded_mass = np.zeros((len(masses), width), np.float32)
    padded_intensity = np.zeros_like(padded_mass)
    padding = np.ones(padded_mass.shape, np.bool_)
    for row, (mass, intensity) in enumerate(zip(masses, intensities, strict=True)):
        count = len(mass)
        padded_mass[row, :count] = mass
        padded_intensity[row, :count] = intensity
        padding[row, : max(1, count)] = False
    return {
        "mz": torch.from_numpy(padded_mass),
        "it": torch.from_numpy(padded_intensity),
        "pad": torch.from_numpy(padding),
        "prec": torch.from_numpy(precursor),
        "ad": torch.from_numpy(adduct),
        "ce": torch.from_numpy(energy),
        "ce_known": torch.from_numpy(energy_known),
        "merged": torch.from_numpy(merged_count) / 4.0,
        "mode": torch.from_numpy(mode),
    }
