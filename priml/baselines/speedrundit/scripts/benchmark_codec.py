#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Measure how much each storage codec degrades an experiment's latents.

Encodes a fitting subset and a disjoint evaluation subset of ImageNet with the
experiment's autoencoder in float32, fits every candidate codec on the first,
and scores it on the second: distortion in the NORMALIZED space the diffusion
model trains in (overall, per channel, in the tails), and optionally after
decoding back to pixels (PSNR, LPIPS) against the undisturbed reconstruction.
It answers whether a codec is safe to materialize a corpus with; whether it
changes the trained model is a downstream experiment (exp003 against exp004).

Examples:
  benchmark_codec.py --experiment exp003 --source /datasets/imagenet --output codecs.json
  benchmark_codec.py --experiment exp003 --source /datasets/imagenet --decode --device cuda

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import json
import logging
import math

from priml.baselines.speedrundit.latent_codec import (
    FittedCodec,
    FloatCodec,
    GaussianFit,
    LinearFit,
    QuantileFit,
    ScalarTableCodec,
    ScaleGroups,
    SharedTable,
    bits_per_scalar,
    entropy_bits,
)
from priml.baselines.speedrundit.scripts.prepare_data import (
    batches,
    dataset_config,
    encode_latents,
    fit_sample_indices,
    records,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from configgle import Makeable
    from torch import Tensor

    import torch

    from priml.baselines.speedrundit.latent_codec import LatentCodec
    from priml.baselines.speedrundit.scripts.prepare_data import Record
    from priml.model.vision_ae.custom_types import Autoencoder, LatentNormalizer
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


logger = logging.getLogger(__name__)


def candidates() -> dict[str, Makeable[LatentCodec]]:
    """Return the codecs compared, keyed by report name.

    Returns:
      candidates: Float rounding, and uint8 tables fitted five ways and shared
        three ways.

    """
    return {
        "float32": FloatCodec.Config(),
        "bfloat16": FloatCodec.Config(dtype=torch.bfloat16),
        "float16": FloatCodec.Config(dtype=torch.float16),
        "uint8_linear_minmax": ScalarTableCodec.Config(fit=LinearFit.Config()),
        "uint8_linear_4sigma": ScalarTableCodec.Config(
            fit=LinearFit.Config(clip_sigmas=4.0),
        ),
        "uint8_quantile": ScalarTableCodec.Config(fit=QuantileFit.Config()),
        "uint8_gaussian": ScalarTableCodec.Config(fit=GaussianFit.Config()),
        "uint8_lloyd_max": ScalarTableCodec.Config(),
        "uint8_lloyd_max_32groups": ScalarTableCodec.Config(
            groups=ScaleGroups.Config(num_groups=32),
        ),
        "uint8_lloyd_max_shared": ScalarTableCodec.Config(groups=SharedTable.Config()),
    }


def latent_metrics(
    raw: Tensor,
    decoded: Tensor,
    normalizer: LatentNormalizer,
) -> dict[str, float]:
    """Score decoded latents against raw ones in diffusion space.

    Args:
      raw: ``[N, C, H, W]`` raw latents.
      decoded: The codec's reconstruction of ``raw``.
      normalizer: Maps raw latents to the space the model trains in.

    Returns:
      metrics: ``nmse``, ``snr_db``, per-channel SNR minimum / 1st percentile /
        median, RMSE where the normalized value exceeds 3, and the worst
        absolute error.

    """
    target = normalizer.normalize(raw).double()
    error = normalizer.normalize(decoded).double() - target
    per_channel = error.pow(2).mean(dim=(0, 2, 3))
    signal = target.pow(2).mean(dim=(0, 2, 3))
    snr = 10 * torch.log10(signal / per_channel.clamp(min=1e-300))
    tail = target.abs() > 3
    nmse = float(error.pow(2).sum() / target.pow(2).sum())
    return {
        "nmse": nmse,
        "snr_db": -10 * math.log10(nmse) if nmse > 0 else math.inf,
        "channel_snr_db_min": float(snr.min()),
        "channel_snr_db_p1": float(snr.quantile(0.01)),
        "channel_snr_db_median": float(snr.median()),
        "tail_rmse": float(error[tail].pow(2).mean().sqrt()) if tail.any() else 0.0,
        "max_abs_error": float(error.abs().max()),
    }


def psnr(reference: Tensor, other: Tensor) -> float:
    """Return the PSNR of ``other`` against ``reference``, both in ``[0, 1]``.

    Args:
      reference: Images in ``[0, 1]``.
      other: Images in ``[0, 1]``, same shape.

    Returns:
      psnr: Decibels; infinite when identical.

    """
    mse = float((other.double() - reference.double()).pow(2).mean())
    return -10 * math.log10(mse) if mse > 0 else math.inf


def evaluate(
    codec_configs: Mapping[str, Makeable[LatentCodec]],
    fit_sample: Tensor,
    eval_sample: Tensor,
    normalizer: LatentNormalizer,
    *,
    decode: Callable[[Tensor], Tensor] | None = None,
    perceptual: Callable[[Tensor, Tensor], float] | None = None,
) -> dict[str, dict[str, float]]:
    """Fit each codec on one sample and score it on another.

    Args:
      codec_configs: Candidates by name.
      fit_sample: ``[N, C, H, W]`` raw latents the fitted codecs learn from.
      eval_sample: ``[M, C, H, W]`` raw latents they are scored on.
      normalizer: The experiment's latent normalizer.
      decode: Maps raw latents to ``[0, 1]`` images; skips image metrics if absent.
      perceptual: Distance between two image batches, e.g. LPIPS.

    Returns:
      report: Metrics per candidate, with storage bits and index entropy.

    """
    reference = decode(eval_sample) if decode is not None else None
    report: dict[str, dict[str, float]] = {}
    for name, config in codec_configs.items():
        codec = config.make()
        if isinstance(codec, FittedCodec):
            codec.fit(fit_sample)
        stored = codec.encode(eval_sample)
        decoded = codec.decode(stored)
        metrics = latent_metrics(eval_sample, decoded, normalizer)
        metrics["bits_per_scalar"] = float(bits_per_scalar(codec))
        if stored.dtype == torch.uint8:
            metrics["index_entropy_bits"] = entropy_bits(stored)
        if decode is not None and reference is not None:
            images = decode(decoded)
            metrics["image_psnr_db"] = psnr(reference, images)
            if perceptual is not None:
                metrics["image_lpips"] = perceptual(reference, images)
        report[name] = metrics
        logger.info("%s: %s", name, metrics)
    return report


def fit_stability(
    fit_sample: Tensor,
    eval_sample: Tensor,
    normalizer: LatentNormalizer,
    sizes: Sequence[int],
) -> dict[int, float]:
    """Return held-out NMSE of the default uint8 codec fitted on growing prefixes.

    Once doubling the fitting images stops lowering the held-out error, more
    images buy nothing; that size is ``num_fit_images``.

    Args:
      fit_sample: ``[N, C, H, W]`` raw latents, largest fitting set.
      eval_sample: Disjoint raw latents to score on.
      normalizer: The experiment's latent normalizer.
      sizes: Fitting-set sizes to try, each at most ``N``.

    Returns:
      nmse: Held-out NMSE per fitting size.

    """
    result: dict[int, float] = {}
    for size in sizes:
        codec = ScalarTableCodec.Config().make()
        codec.fit(fit_sample[:size])
        decoded = codec.decode(codec.encode(eval_sample))
        result[size] = latent_metrics(eval_sample, decoded, normalizer)["nmse"]
    return result


def run(
    experiment: str,
    imagenet: Path,
    *,
    num_fit_images: int,
    num_eval_images: int,
    device: str,
    batch_size: int,
    decode_images: bool,
) -> dict[str, object]:
    """Encode both samples with the experiment's autoencoder and score every codec.

    Args:
      experiment: Factory in :mod:`~priml.baselines.speedrundit.experiments`.
      imagenet: Extracted ImageNet directory.
      num_fit_images: Images the fitted codecs learn from.
      num_eval_images: Disjoint images they are scored on.
      device: Device the autoencoder runs on.
      batch_size: Images per forward.
      decode_images: Also decode to pixels and score PSNR and LPIPS.

    Returns:
      report: The codec table and the fitting-size curve.

    """
    config = dataset_config(experiment)
    root = Path(config.working_dir)
    listed = records(imagenet)
    chosen = fit_sample_indices(len(listed), num_fit_images + num_eval_images)
    fit_records = [listed[i] for i in chosen[0::2][:num_fit_images]]
    eval_records = [listed[i] for i in chosen[1::2][:num_eval_images]]
    autoencoder = config.autoencoder.make()
    if isinstance(autoencoder, torch.nn.Module):
        _ = autoencoder.to(device)
    size = config.autoencoder.image_size
    fit_sample, eval_sample = (
        _encode_all(
            autoencoder,
            chosen_records,
            root=root,
            size=size,
            device=device,
            batch_size=batch_size,
        )
        for chosen_records in (fit_records, eval_records)
    )
    normalizer = config.autoencoder.latent_norm.make()
    decode = None
    perceptual = None
    if decode_images:
        decode = _decoder(autoencoder, device, batch_size)
        perceptual = _lpips(device, batch_size)
    sizes = [n for n in (256, 1_024, 4_096, 16_384) if n <= fit_sample.shape[0]]
    return {
        "experiment": experiment,
        "fit_images": fit_sample.shape[0],
        "eval_images": eval_sample.shape[0],
        "codecs": evaluate(
            candidates(),
            fit_sample,
            eval_sample,
            normalizer,
            decode=decode,
            perceptual=perceptual,
        ),
        "fit_stability_nmse": fit_stability(fit_sample, eval_sample, normalizer, sizes),
    }


def main() -> int:
    """Benchmark the codecs for one experiment and write the report.

    Returns:
      exit_code: Zero on success.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(
        flags.experiment,
        flags.source,
        num_fit_images=flags.fit_images,
        num_eval_images=flags.eval_images,
        device=flags.device,
        batch_size=flags.batch_size,
        decode_images=flags.decode,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if flags.output is None:
        print(text)
    else:
        _ = flags.output.write_text(text + "\n", encoding="utf-8")
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    experiment: str
    source: Path
    output: Path | None
    fit_images: int
    eval_images: int
    device: str
    batch_size: int
    decode: bool


def _encode_all(
    autoencoder: Autoencoder,
    listed: Sequence[Record],
    *,
    root: Path,
    size: int,
    device: str,
    batch_size: int,
) -> Tensor:
    """Encode records to one float32 CPU tensor."""
    return torch.cat(
        [
            encode_latents(autoencoder, images, device)
            for _, images in batches(listed, root, size, batch_size)
        ],
    )


def _decoder(
    autoencoder: Autoencoder,
    device: str,
    batch_size: int,
) -> Callable[[Tensor], Tensor]:
    """Return raw latents to CPU ``[0, 1]`` images, decoded in batches."""

    def decode(latents: Tensor) -> Tensor:
        with torch.inference_mode():
            return torch.cat(
                [
                    autoencoder.decode(chunk.to(device)).float().cpu()
                    for chunk in latents.split(batch_size)
                ],
            )

    return decode


def _lpips(device: str, batch_size: int) -> Callable[[Tensor, Tensor], float]:
    """Return the mean AlexNet LPIPS between two ``[0, 1]`` image batches."""
    import lpips  # noqa: PLC0415 -- Loaded, with its weight download, only when images are scored.

    network = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def distance(reference: Tensor, other: Tensor) -> float:
        scores: list[Tensor] = []
        with torch.inference_mode():
            for left, right in zip(
                reference.split(batch_size),
                other.split(batch_size),
                strict=True,
            ):
                pair = (left.to(device) * 2 - 1, right.to(device) * 2 - 1)
                scores.append(network(*pair).flatten().cpu())
        return float(torch.cat(scores).double().mean())

    return distance


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register benchmark flags."""
    parser.add_argument("--experiment", default="exp003")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fit-images", type=int, default=4_096)
    parser.add_argument("--eval-images", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decode", action="store_true")


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
