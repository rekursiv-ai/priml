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
  benchmark_codec.py --experiment exp003 --source /datasets/imagenet --output /opt/scratch/artifacts/speedrundit/codecs.json
  benchmark_codec.py --experiment exp003 --source /datasets/imagenet --decode --device cuda

'''
# fmt: on

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import json
import logging
import math
import random

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
    FIT_SAMPLE_SEED,
    batches,
    dataset_config,
    encode_latents,
    ensure_image_source,
    fit_sample_indices,
    image_source_identity,
    records,
)
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from configgle import Makeable
    from torch import Tensor

    import lpips
    import torch

    from priml.baselines.speedrundit.latent_codec import LatentCodec
    from priml.baselines.speedrundit.scripts.prepare_data import Record
    from priml.model.vision_ae.custom_types import Autoencoder, LatentNormalizer
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")
    lpips = lazy_import("lpips")


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
    # Unclamped: an exact channel scores inf, an exact zero channel 0/0 = NaN, and a
    # NaN reconstruction stays NaN rather than reading as perfect.
    snr = 10 * torch.log10(signal / per_channel)
    tail = target.abs() > 3
    nmse = float(error.pow(2).sum() / target.pow(2).sum())
    return {
        "nmse": nmse,
        "snr_db": math.inf if nmse == 0 else -10 * math.log10(nmse),
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
      psnr: Decibels; infinite when identical, NaN when ``other`` holds NaN.

    """
    mse = float((other.double() - reference.double()).pow(2).mean())
    return math.inf if mse == 0 else -10 * math.log10(mse)


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
        metrics = latent_metrics(eval_sample, decoded=decoded, normalizer=normalizer)
        metrics["bits_per_scalar"] = float(bits_per_scalar(codec))
        if stored.dtype == torch.uint8:
            metrics["index_entropy_bits"] = entropy_bits(stored)
        if decode is not None and reference is not None:
            images = decode(decoded)
            metrics["image_psnr_db"] = psnr(reference, other=images)
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
        result[size] = latent_metrics(
            eval_sample,
            decoded=decoded,
            normalizer=normalizer,
        )["nmse"]
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
    directory: Path | None = None,
) -> dict[str, object]:
    """Encode both samples with the experiment's autoencoder and score every codec.

    Both samples remain in CPU memory. Their size depends on the requested
    image counts, while ``batch_size`` controls encoding device memory. Each
    float32 RAE latent (768 x 16 x 16) occupies 768 KiB, before fitting and
    metric temporaries; choose image counts that fit the available host RAM.

    Crops are read from, and written to, the corpus root's shared ``images/``,
    binding it to ``imagenet`` as :mod:`prepare_data` does; benchmark another
    source in another ``directory``.

    Args:
      experiment: Factory in :mod:`~priml.baselines.speedrundit.experiments`.
      imagenet: Extracted ImageNet directory.
      num_fit_images: Images the fitted codecs learn from.
      num_eval_images: Disjoint images they are scored on.
      device: Device the autoencoder runs on.
      batch_size: Images per forward.
      decode_images: Also decode to pixels and score PSNR and LPIPS.
      directory: Corpus root whose crops to share, replacing the experiment's.

    Returns:
      report: The codec table and the fitting-size curve.

    Raises:
      ValueError: An image count is not positive, or the source is too small
        for disjoint sets of both sizes.

    """
    if num_fit_images < 1 or num_eval_images < 1:
        raise ValueError("Fitting and evaluation image counts must be positive.")
    config = dataset_config(experiment)
    root = Path(config.working_dir if directory is None else directory)
    listed = records(imagenet)
    chosen = fit_sample_indices(
        len(listed),
        num_images=num_fit_images + num_eval_images,
    )
    if len(chosen) != num_fit_images + num_eval_images:
        raise ValueError(
            "The corpus is too small for disjoint fitting and evaluation sets.",
        )
    size = config.autoencoder.image_size
    ensure_image_source(
        root,
        identity=image_source_identity(imagenet, listed=listed, size=size),
    )
    # Shuffle before partitioning: the source order is grouped by ImageNet class.
    random.Random(FIT_SAMPLE_SEED).shuffle(chosen)  # noqa: S311 -- Selects benchmark images, not secrets.
    fit_records = [listed[i] for i in chosen[:num_fit_images]]
    eval_records = [listed[i] for i in chosen[num_fit_images:]]
    autoencoder = config.autoencoder.make()
    if isinstance(autoencoder, torch.nn.Module):
        _ = autoencoder.to(device)
    fit_sample, eval_sample = (
        _encode_all(
            autoencoder,
            listed=chosen_records,
            root=root,
            size=size,
            device=device,
            batch_size=batch_size,
            latent_shape=config.autoencoder.latent_shape(),
        )
        for chosen_records in (fit_records, eval_records)
    )
    normalizer = config.autoencoder.latent_norm.make()
    decode = None
    perceptual = None
    if decode_images:
        decode = partial(
            _decode,
            autoencoder=autoencoder,
            device=device,
            batch_size=batch_size,
        )
        perceptual = _lpips(device, batch_size=batch_size)
    sizes = [n for n in (256, 1_024, 4_096, 16_384) if n <= fit_sample.shape[0]]
    return {
        "experiment": experiment,
        "fit_images": fit_sample.shape[0],
        "eval_images": eval_sample.shape[0],
        "codecs": evaluate(
            candidates(),
            fit_sample=fit_sample,
            eval_sample=eval_sample,
            normalizer=normalizer,
            decode=decode,
            perceptual=perceptual,
        ),
        "fit_stability_nmse": fit_stability(
            fit_sample,
            eval_sample=eval_sample,
            normalizer=normalizer,
            sizes=sizes,
        ),
    }


def main() -> int:
    """Benchmark the codecs for one experiment and write the report.

    Returns:
      exit_code: Zero on success.

    Raises:
      SystemExit: ``--output`` lies inside ``--source`` or the corpus root.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    directory = (
        None if flags.directory is None else validated_output_path(flags.directory)
    )
    output = None if flags.output is None else validated_output_path(flags.output)
    root = directory or Path(dataset_config(flags.experiment).working_dir)
    if output is not None and any(
        output.resolve().is_relative_to(path.expanduser().resolve())
        for path in (flags.source, root)
    ):
        parser.error("--output must lie outside --source and the corpus root.")
    report = run(
        flags.experiment,
        imagenet=flags.source,
        num_fit_images=flags.fit_images,
        num_eval_images=flags.eval_images,
        device=flags.device,
        batch_size=flags.batch_size,
        decode_images=flags.decode,
        directory=directory,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if output is None:
        print(text)
    else:
        _ = output.write_text(text + "\n", encoding="utf-8")
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    experiment: str
    source: Path
    directory: Path | None
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
    latent_shape: tuple[int, int, int],
) -> Tensor:
    """Encode records to one float32 CPU tensor."""
    return torch.cat(
        [
            encode_latents(
                autoencoder,
                images=images,
                device=device,
                latent_shape=latent_shape,
            )
            for _, images in batches(
                listed,
                root=root,
                size=size,
                batch_size=batch_size,
            )
        ],
    )


def _decode(
    latents: Tensor,
    *,
    autoencoder: Autoencoder,
    device: str,
    batch_size: int,
) -> Tensor:
    """Decode raw latents to CPU images in bounded batches."""
    with torch.inference_mode():
        return torch.cat(
            [
                autoencoder.decode(chunk.to(device)).float().cpu()
                for chunk in latents.split(batch_size)
            ],
        )


def _lpips(device: str, batch_size: int) -> Callable[[Tensor, Tensor], float]:
    """Return the mean AlexNet LPIPS between two ``[0, 1]`` image batches."""
    network = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return partial(
        _lpips_distance,
        network=network,
        device=device,
        batch_size=batch_size,
    )


def _lpips_distance(
    reference: Tensor,
    other: Tensor,
    *,
    network: lpips.LPIPS,
    device: str,
    batch_size: int,
) -> float:
    """Measure mean perceptual distance between ``[0, 1]`` images in bounded batches."""
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


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register benchmark flags."""
    parser.add_argument("--experiment", default="exp003")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--directory",
        type=Path,
        help="Corpus root whose crops to share, replacing the experiment's.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fit-images", type=int, default=4_096)
    parser.add_argument("--eval-images", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decode", action="store_true")


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
