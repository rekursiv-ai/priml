#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Measure this port against the pinned upstream SpeedrunDiT, bit for bit.

Clones the reference at a pinned commit, imports it UNMODIFIED, and compares
with exact tensor equality inside Priml's host-agnostic numeric context: the
ImageNet conversion, the loader, initialization, five optimizer steps of
exp000's own train step (times, noise, losses, every gradient, every weight,
the EMA shadow), an eval forward with and without the sparse path, and the
guided sampler.

Nothing here is a unit test. It needs a network, a git clone, and a minute,
and it establishes the port ONCE against a moving upstream; the bit-for-bit
goldens under ``testdata/`` are what keep it frozen afterwards. So this module
is never imported by the library, and the library never imports it. The
reference imports ``timm``, and its preprocessing ``click`` and ``tqdm``, none of
which Priml depends on, so the run supplies them for its own duration.

The comparison is deliberately one-substitution: both sides run the same
geometry, the same seed, and the same input tensors, and neither is adjusted
to make the other agree. Where a difference is structural rather than
numerical -- parameter names -- it is REPORTED under its own heading rather
than normalized away.

Examples:
  uv --quiet run --frozen --with timm --with click --with tqdm python -m priml.baselines.speedrundit.scripts.parity  # noqa: E501
  uv --quiet run --frozen --with timm --with click --with tqdm python -m priml.baselines.speedrundit.scripts.parity --upstream /path/to/SpeedrunDiT  # noqa: E501

'''
# fmt: on

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypedDict, cast, override

import argparse
import ast
import copy
import importlib
import json
import re
import subprocess
import sys
import tempfile

from torch import Tensor, nn

import numpy as np
import torch

from priml.baselines.speedrundit.data import SpeedrunDiTData
from priml.baselines.speedrundit.experiments import exp000
from priml.baselines.speedrundit.loss import SpeedrunDiTLoss
from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.baselines.speedrundit.sampler import EulerMaruyamaSampler
from priml.baselines.speedrundit.scripts.prepare_data import convert
from priml.baselines.speedrundit.train_step import SpeedrunDiTTrainStep
from priml.testing.bfb import host_agnostic_numerics
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from PIL import Image
    from torch.utils.data import Dataset

    from priml.baselines.speedrundit.loss import VelocityField
else:
    from wrapt import lazy_import

    Image = lazy_import("PIL.Image")  # ~60 ms; only the synthetic ImageNet needs it.


_CWD: Final = Path(__file__).resolve().parent

SOURCE_URL: Final = "https://github.com/SwayStar123/SpeedrunDiT.git"
"""Reference repository."""

SOURCE_REVISION: Final = "c24c2ff25699cce63174ca56c2afcfeeb225e367"
"""Pinned reference commit; every golden in testdata/ was measured against it."""

STEPS: Final = 5
"""Optimizer steps compared."""

# The smallest the reference builds: one head of four (the axial rotary
# ladder's floor), a 2x2 grid, and five layers, since the reference hard-codes
# two dense layers either side of its sparse stage. ``source_init`` is minted
# here; the other goldens shrink the port further.
GEOMETRY: Final[dict[str, object]] = {
    "input_size": 2,
    "patch_size": 1,
    "in_channels": 1,
    "hidden_size": 4,
    "decoder_hidden_size": 4,
    "depth": 5,
    "num_heads": 1,
    "encoder_depth": 2,
    "num_classes": 3,
    "class_dropout_prob": 0.1,
    "use_cfg": True,
    "z_dims": [2],
    "projector_dim": 2,
    "cls_token_dim": 2,
}

BATCH: Final = 3
"""Samples per compared batch."""

INIT_GOLDEN: Final = "source_init.pt"
"""The initialization golden this script alone mints, under ``testdata/``."""

SAMPLER_CLASSES: Final = 1000
"""Classes for the sampler comparison: the reference hard-codes 1000 as its
null label, so a smaller table would index past its end."""

RENAMES: Final = (
    (r"^sprint\.mask_token$", "mask_token"),
    (r"^sprint\.fusion_proj\.", "fusion_proj."),
    (r"\.attn\.value_residual\.weight$", ".attn.v1_lambda"),
    (r"\.ffn\.", ".mlp."),
    (r"\.modulation\.modulation\.", ".adaLN_modulation."),
    (r"^cls_projector\.", "cls_projectors2."),
    (r"^rope\.cos$", "feat_rope.freqs_cos"),
    (r"^rope\.sin$", "feat_rope.freqs_sin"),
)
"""Port state names rewritten to the reference's; every other name agrees."""


class _Objective(Protocol):
    """The reference's ``SILoss`` instance: seven terms per call."""

    def __call__(
        self,
        model: nn.Module,
        images: Tensor,
        model_kwargs: dict[str, Tensor],
        *,
        zs: list[Tensor],
        cls_token: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Apply to the input."""
        ...


class _ReferenceModules(Protocol):
    """The reference modules this script reads, by name, from the clone."""

    SiT: Callable[..., nn.Module]
    SILoss: Callable[..., _Objective]
    CustomDataset: Callable[[str], Dataset[tuple[Tensor, Tensor, Tensor]]]
    euler_maruyama_sampler_path_drop: Callable[..., Tensor]


class _Trace(TypedDict):
    """One optimizer step, as both drivers record it."""

    loss: Tensor
    time: Tensor
    noise: Tensor
    grads: list[tuple[str, Tensor]]
    weights: list[tuple[str, Tensor]]
    ema: list[tuple[str, Tensor]]


class _UpdateEma(Protocol):
    """The reference's ``train.update_ema``."""

    def __call__(
        self,
        ema_model: nn.Module,
        model: nn.Module,
        decay: float = ...,
    ) -> None:
        """Apply to the input."""
        ...


def upstream_module(name: str) -> _ReferenceModules:
    """Import a module of the reference, which is on ``sys.path`` by now.

    Args:
      name: Module path inside the clone, e.g. ``"models.sit"``.

    Returns:
      module: The module, typed as the members this script reads.

    """
    return cast(_ReferenceModules, importlib.import_module(name))


def native_config(geometry: dict[str, object] = GEOMETRY) -> SpeedrunDiT.Config:
    """Build the port at a comparison geometry.

    Args:
      geometry: The reference's constructor arguments.

    Returns:
      cfg: A config matching ``geometry``.

    """
    cfg = SpeedrunDiT.Config()
    cfg.channels_in = cast(int, geometry["in_channels"])
    cfg.channels_hidden = cast(int, geometry["hidden_size"])
    cfg.image_size = cast(int, geometry["input_size"])
    cfg.patch_size = cast(int, geometry["patch_size"])
    cfg.num_layers = cast(int, geometry["depth"])
    cfg.heads = cast(int, geometry["num_heads"])
    cfg.num_classes = cast(int, geometry["num_classes"])
    cfg.label_embedder.dropout = cast(float, geometry["class_dropout_prob"])
    cfg.projector_dims = (cast(list[int], geometry["z_dims"])[0],)
    cfg.projector_hidden = cast(int, geometry["projector_dim"])
    return cfg


def upstream_name(name: str) -> str:
    """Translate a port parameter name to the reference's.

    Args:
      name: A name from the port's ``named_parameters``.

    Returns:
      name: The reference's name for the same tensor.

    """
    for pattern, replacement in RENAMES:
        name = re.sub(pattern, replacement, name)
    return name


def build_pair(
    geometry: dict[str, object],
    seed: int,
) -> tuple[nn.Module, SpeedrunDiT]:
    """Construct the reference and the port under one seed.

    Args:
      geometry: The reference's constructor arguments.
      seed: Global seed both constructions start from.

    Returns:
      reference: The reference model.
      candidate: The port.

    """
    torch.manual_seed(seed)
    upstream = upstream_module("models.sit").SiT(
        qk_norm=True,
        fused_attn=True,
        **geometry,
    )
    torch.manual_seed(seed)
    return upstream, native_config(geometry).make()


def perturb(reference: nn.Module, candidate: SpeedrunDiT) -> None:
    """Add the same noise to both models' weights, matched by name.

    Zero-initialized readouts and modulations make a freshly built model
    output exactly zero, which a wrong trunk would reproduce as well.

    Args:
      reference: The reference model.
      candidate: The port.

    """
    generator = torch.Generator().manual_seed(5)
    by_name = dict(reference.named_parameters())
    with torch.no_grad():
        for name, parameter in candidate.named_parameters():
            noise = 0.05 * torch.randn(parameter.shape, generator=generator)
            parameter.add_(noise)
            by_name[upstream_name(name)].add_(noise)


@contextmanager
def pinned_source(explicit: Path | None) -> Generator[Path]:
    """Yield a checkout of the reference at :data:`SOURCE_REVISION`.

    An explicit path is used as given and NOT verified, so a local working
    copy can be measured; a clone is pinned by SHA and the SHA is read back,
    which catches a moved ref rather than trusting the name.

    Args:
      explicit: An existing checkout, or ``None`` to clone one.

    Yields:
      root: Directory holding the reference.

    Raises:
      RuntimeError: If a fresh clone does not land on the pinned commit.

    """
    if explicit is not None:
        yield explicit.expanduser().resolve()
        return
    with tempfile.TemporaryDirectory(prefix="speedrundit-parity-") as name:
        root = Path(name) / "srdit"
        subprocess.run(  # noqa: S603 -- Fixed executable and a constant URL.
            ["git", "clone", "--quiet", "--no-checkout", SOURCE_URL, str(root)],  # noqa: S607 -- git from PATH, as a developer's clone would run it.
            check=True,
        )
        subprocess.run(  # noqa: S603 -- Fixed executable, constant revision.
            ["git", "-C", str(root), "checkout", "--quiet", SOURCE_REVISION],  # noqa: S607 -- git from PATH, as a developer's clone would run it.
            check=True,
        )
        head = subprocess.run(  # noqa: S603 -- Fixed executable.
            ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607 -- git from PATH, as a developer's clone would run it.
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != SOURCE_REVISION:
            raise RuntimeError(
                f"clone landed on {head}, expected {SOURCE_REVISION}.",
            )
        yield root


def report(label: str, ok: bool, detail: str = "") -> bool:
    """Print one comparison line.

    Args:
      label: What was compared.
      ok: Whether it matched.
      detail: Extra text shown on failure.

    Returns:
      ok: The value passed in, so callers can accumulate.

    """
    mark = "OK  " if ok else "FAIL"
    suffix = f"  {detail}" if detail and not ok else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok


def tensors_equal(left: Tensor, right: Tensor) -> tuple[bool, str]:
    """Compare two tensors exactly, reporting ULPs when they differ.

    Args:
      left: Reference tensor.
      right: Candidate tensor.

    Returns:
      ok: Whether every bit matches.
      detail: A description of the largest disagreement.

    """
    if left.shape != right.shape:
        return False, f"shape {tuple(left.shape)} != {tuple(right.shape)}"
    if left.dtype != right.dtype:
        return False, f"dtype {left.dtype} != {right.dtype}"
    if torch.equal(left, right):
        return True, ""
    diff = (left.double() - right.double()).abs().max().item()
    return False, f"maxabs={diff:.6e}"


def compare_named(
    left: Sequence[tuple[str, Tensor]],
    right: Sequence[tuple[str, Tensor]],
    label: str,
) -> bool:
    """Compare two parameter lists by name, through :data:`RENAMES`.

    By name rather than by position: the reference's root owns ``mask_token``
    and ``pos_embed`` directly, and ``named_parameters`` yields a module's own
    parameters before any child's, while the port keeps the mask token with
    the routing that uses it. That swap involves only the frozen table, which
    never has a gradient; the order of the tensors that do is asserted by
    :func:`compare_trainable_order`.

    Args:
      left: Reference ``(name, tensor)`` pairs.
      right: Candidate pairs.
      label: Heading for the report line.

    Returns:
      ok: Whether every tensor matched.

    """
    reference = dict(left)
    candidate = {upstream_name(name): value for name, value in right}
    unmatched = sorted(set(reference) ^ set(candidate))
    if unmatched:
        return report(label, ok=False, detail=f"unmatched names: {unmatched[:4]}")
    bad: list[str] = []
    for name, value in reference.items():
        ok, detail = tensors_equal(value, candidate[name])
        if not ok:
            bad.append(f"{name}: {detail}")
    if bad:
        return report(label, ok=False, detail=f"{len(bad)} differ; first: {bad[0]}")
    return report(label, ok=True)


def compare_trainable_order(reference: nn.Module, candidate: nn.Module) -> bool:
    """Compare the order of the parameters that receive gradients.

    The gradient clip reduces one norm over the gradients in this order, so
    two orders are two float reductions of the same terms.

    Args:
      reference: The reference model.
      candidate: The port.

    Returns:
      ok: Whether the orders agree.

    """
    left = [n for n, p in reference.named_parameters() if p.requires_grad]
    right = [
        upstream_name(n) for n, p in candidate.named_parameters() if p.requires_grad
    ]
    first = next((a for a, b in zip(left, right, strict=False) if a != b), "")
    return report("trainable parameter order", left == right, f"first: {first}")


def synthetic_corpus(root: Path, *, count: int) -> None:
    """Write a corpus in the reference's on-disk layout.

    Args:
      root: Destination directory.
      count: Samples to write.

    """
    images = root / "images" / "00000"
    latents = root / "vae-in" / "00000"
    images.mkdir(parents=True)
    latents.mkdir(parents=True)
    labels: list[list[object]] = []
    rng = np.random.default_rng(0)
    size = cast(int, GEOMETRY["input_size"])
    channels = cast(int, GEOMETRY["in_channels"])
    for index in range(count):
        stem = f"{index:08d}"
        image = rng.integers(0, 255, (3, size, size), np.uint8)
        np.save(images / f"img{stem}.npy", image)
        latent = rng.standard_normal((1, channels, size, size)).astype(np.float32)
        np.save(latents / f"img-latents-{stem}.npy", latent)
        labels.append([f"00000/img-latents-{stem}.npy", index % 10])
    manifest = {"labels": labels}
    (root / "vae-in" / "dataset.json").write_text(json.dumps(manifest), "utf-8")
    # The port reads its class-token targets from disk where the reference
    # encodes them each step; the reference's loader never looks at this file.
    width = cast(list[int], GEOMETRY["z_dims"])[0]
    np.save(root / "cls_token.npy", rng.standard_normal((count, width), np.float32))


IMAGENET_SPECS: Final = (
    (500, 375, "RGB", "JPEG"),
    (375, 500, "RGB", "JPEG"),
    (1200, 900, "RGB", "JPEG"),
    (2100, 2100, "RGB", "JPEG"),
    (256, 256, "RGB", "JPEG"),
    (200, 150, "RGB", "JPEG"),
    (333, 257, "RGB", "JPEG"),
    (640, 480, "L", "JPEG"),
    (640, 427, "CMYK", "JPEG"),
    (300, 400, "RGB", "PNG"),
    (300, 300, "RGBA", "PNG"),
    (220, 330, "P", "PNG"),
    (220, 330, "LA", "PNG"),
)
"""Size, mode, and container of each synthetic ImageNet image.

Landscape, portrait and square; no, one, and two box halvings, and an
upscale; odd sizes for the bicubic rounding; and every mode the real archive
or ``convert("RGB")`` distinguishes -- ImageNet's train split holds CMYK
JPEGs and a PNG named ``.JPEG``, and every file is named ``.JPEG``."""


def synthetic_imagenet(root: Path) -> None:
    """Write an extracted ImageNet ``train/`` tree covering every conversion path.

    Its synsets are the first of the canonical list, where indexing the
    directories present -- the reference's rule -- and indexing the full
    list -- priml's -- agree. File names are ImageNet's unpadded ones, so
    string order and numeric order differ.

    Args:
      root: Destination; ``train/`` is created beneath it.

    """
    synsets = ("n01440764", "n01443537", "n01484850")
    rng = np.random.default_rng(0)
    for index, (width, height, mode, container) in enumerate(IMAGENET_SPECS):
        synset = synsets[index % len(synsets)]
        (root / "train" / synset).mkdir(parents=True, exist_ok=True)
        rows = np.arange(height)[:, None].repeat(width, 1)
        cols = np.arange(width)[None, :].repeat(height, 0)
        ramp = np.stack([cols, rows, cols + rows], -1) % 192
        pixels = rng.integers(0, 64, (height, width, 3)) + ramp
        image = Image.fromarray(pixels.astype(np.uint8), "RGB")
        if mode == "RGBA":
            alpha = rng.integers(0, 256, (height, width, 1), dtype=np.uint8)
            image = Image.fromarray(np.concatenate([np.asarray(image), alpha], -1))
        elif mode == "P":
            image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
        else:
            image = image.convert(mode)
        path = root / "train" / synset / f"{synset}_{7**index % 10_007}.JPEG"
        if container == "JPEG":
            image.save(path, format="JPEG", quality=90)
        else:
            image.save(
                path,
                format="PNG",
                **({"transparency": 3} if mode == "P" else {}),
            )


def compare_imagenet_convert(upstream: Path) -> bool:
    """Compare the ImageNet conversion with the reference's ``convert``.

    Both run unmodified on one synthetic ImageNet tree; the reference's tool
    runs in its own process, as its README invokes it. Compared are every
    file name, ``dataset.json`` byte for byte, and every PNG byte for byte.

    Args:
      upstream: Reference checkout.

    Returns:
      ok: Whether the two trees are identical.

    """
    print("imagenet convert")
    with tempfile.TemporaryDirectory(prefix="speedrundit-imagenet-") as name:
        root = Path(name)
        synthetic_imagenet(root / "imagenet")
        subprocess.run(  # noqa: S603 -- The running interpreter and fixed arguments.
            [
                sys.executable,
                "dataset_tools.py",
                "convert",
                f"--source={root / 'imagenet' / 'train'}",
                f"--dest={root / 'reference' / 'images'}",
                "--resolution=256x256",
                "--transform=center-crop-dhariwal",
                "--workers=2",
            ],
            cwd=upstream / "preprocessing",
            check=True,
            capture_output=True,
        )
        count = convert(root / "imagenet", root / "port", workers=2)
        left, right = (
            {
                p.relative_to(r).as_posix(): p.read_bytes()
                for p in sorted(r.rglob("*"))
                if p.is_file()
            }
            for r in (root / "reference" / "images", root / "port" / "images")
        )
        ok = report(
            "file names",
            list(left) == list(right),
            f"{len(left)} vs {len(right)}",
        )
        ok &= report(
            "dataset.json",
            left.get("dataset.json") == right.get("dataset.json"),
        )
        differ = [n for n in left if left[n] != right.get(n)]
        return ok & report(f"{count} PNGs", not differ, f"first: {differ[:1]}")


def draw_inputs(generator: torch.Generator, *, classes: int) -> dict[str, Tensor]:
    """Draw one batch at :data:`GEOMETRY`, for both sides to consume.

    Args:
      generator: The stream to draw from.
      classes: Label range.

    Returns:
      batch: Latents, times, labels, class tokens, and alignment targets.

    """
    size = cast(int, GEOMETRY["input_size"])
    channels = cast(int, GEOMETRY["in_channels"])
    width = cast(list[int], GEOMETRY["z_dims"])[0]
    return {
        "media": torch.randn(BATCH, channels, size, size, generator=generator),
        "time": torch.rand(BATCH, generator=generator),
        "label": torch.randint(classes, (BATCH,), generator=generator),
        "cls_token": torch.randn(BATCH, width, generator=generator),
        "features": torch.randn(BATCH, 1 + size * size, width, generator=generator),
    }


def compare_loader(upstream: Path) -> bool:
    """Compare the loader against the reference's dataset.

    Runs both over one synthetic tree in the reference's own layout, so the
    enumeration, the sort, the label join, and the latent scaling are all
    exercised on real files rather than asserted from the source.

    Args:
      upstream: Reference checkout.

    Returns:
      ok: Whether the loaders agree.

    """
    print("loader")
    from torch.utils.data import DataLoader  # noqa: PLC0415 -- Script-local.

    count = 6
    with tempfile.TemporaryDirectory(prefix="speedrundit-corpus-") as name:
        root = Path(name)
        synthetic_corpus(root, count=count)
        sys.path.insert(0, str(upstream))
        dataset = upstream_module("dataset").CustomDataset(str(root))
        loader = DataLoader(dataset, batch_size=count, shuffle=False)
        raw_image, latent, label = next(iter(loader))

        config = SpeedrunDiTData.Config()
        config.base_dir = "/"
        config.working_dir = str(root)
        config.batch_size = count
        config.eval_batch_size = count
        config.device = "cpu"
        config.keep_images = True
        # Compare the loader's own arithmetic, not the tokenizer constant.
        config.latent_scale = 1.0
        data = config.make()
        batch = next(iter(data.eval_dataloader()))

        ok = report("order and labels", *tensors_equal(label, batch["label"]))
        ok &= report("latents", *tensors_equal(latent.squeeze(1), batch["media"]))
        if "raw_image" not in batch:
            raise ValueError("The loader dropped raw_image despite keep_images.")
        return ok & report("images", *tensors_equal(raw_image, batch["raw_image"]))


def build_batches(count: int) -> list[dict[str, Tensor]]:
    """Draw the inputs both sides consume.

    Args:
      count: Batches to build.

    Returns:
      batches: Fixed inputs, identical for both implementations.

    """
    generator = torch.Generator().manual_seed(99)
    classes = cast(int, GEOMETRY["num_classes"])
    return [draw_inputs(generator, classes=classes) for _ in range(count)]


def upstream_update_ema(upstream: Path) -> _UpdateEma:
    """Bind the reference's ``update_ema`` from ``train.py``'s own source.

    ``train.py`` is a script that imports wandb, accelerate, and its encoders at
    module scope, so the function is executed alone rather than imported; the
    text run is theirs, not a retyping.

    Args:
      upstream: Reference checkout.

    Returns:
      update_ema: The reference's EMA step.

    """
    path = upstream / "train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "update_ema"
    )
    namespace: dict[str, object] = {"torch": torch, "OrderedDict": OrderedDict}
    exec(compile(ast.Module([node], []), str(path), "exec"), namespace)  # noqa: S102 -- Executes one function from the pinned reference checkout.
    return cast(_UpdateEma, namespace["update_ema"])


def run_upstream(
    model: nn.Module,
    batches: Sequence[dict[str, Tensor]],
    *,
    update_ema: _UpdateEma,
) -> list[_Trace]:
    """Drive the reference for :data:`STEPS` optimizer steps.

    Mirrors ``train.py``'s loop: its EMA is seeded from the initial weights
    (``update_ema(ema, model, decay=0)``) and stepped after every update.

    Args:
      model: The reference model.
      batches: Fixed inputs.
      update_ema: The reference's own EMA step.

    Returns:
      trace: Per-step losses, gradients, post-step weights, and EMA shadow.

    """
    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    update_ema(ema, model, decay=0)
    objective = upstream_module("loss").SILoss(
        path_type="linear",
        weighting="uniform",
        cfm_weighting="uniform",
        apply_time_shift=True,
        shift_base=4096,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        eps=1e-8,
    )
    model.train()
    trace: list[_Trace] = []
    for batch in batches:
        denoise, proj, time, noise, cls, cfm, _cfm_cls = objective(
            model,
            batch["media"],
            {"y": batch["label"]},
            zs=[batch["features"]],
            cls_token=batch["cls_token"],
        )
        loss = (
            denoise.mean() + 0.5 * proj.mean() + 0.03 * cls.mean() + 0.05 * cfm.mean()
        )
        loss.backward()
        _ = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # After the clip, where the optimizer consumes them -- the same point
        # the port side reads its own.
        grads = [
            (name, p.grad.detach().clone())
            for name, p in model.named_parameters()
            if p.grad is not None
        ]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_ema(ema, model)
        trace.append(
            {
                "loss": loss.detach().clone(),
                "time": time.detach().clone(),
                "noise": noise.detach().clone(),
                "grads": grads,
                "weights": _named(model),
                "ema": _named(ema),
            },
        )
    return trace


def native_step(seed: int) -> SpeedrunDiTTrainStep:
    """Build exp000's own train step under the seed the reference was built at.

    The step constructs its model itself, so its initialization, its optimizer,
    and its EMA seeding are all what exp000 does -- nothing is copied in after
    the fact to paper over a difference. Autocast and compile are off: the
    reference side runs fp32.

    Args:
      seed: Global seed the reference's construction started from.

    Returns:
      step: exp000's train step at the parity geometry.

    """
    cfg = exp000().step
    cfg.model = native_config()
    cfg.parallelism = NoParallel.Config(device="cpu")
    cfg.dtype_autocast = None
    cfg.compile = None
    torch.manual_seed(seed)
    step = cfg.make()
    assert isinstance(step, SpeedrunDiTTrainStep)
    return step


def run_native(
    step: SpeedrunDiTTrainStep,
    batches: Sequence[dict[str, Tensor]],
) -> list[_Trace]:
    """Drive exp000's train step for :data:`STEPS` optimizer steps.

    Args:
      step: exp000's train step, from :func:`native_step`.
      batches: Fixed inputs.

    Returns:
      trace: Per-step losses, gradients, post-step weights, and EMA shadow.

    """
    model = step.model
    probe = _StepProbe(step)
    trace: list[_Trace] = []
    for batch in batches:
        _ = step.train_step(
            media=batch["media"],
            label=batch["label"],
            cls_token=batch["cls_token"],
            features=[batch["features"]],
        )
        result = probe.result
        shadow = step.ema.shadow_model
        if shadow is None:
            raise ValueError("exp000's train step carries no EMA shadow model.")
        trace.append(
            {
                "loss": result.loss.detach().clone(),
                "time": result.time.detach().clone(),
                "noise": result.noise.detach().clone(),
                "grads": probe.grads,
                "weights": _named(model),
                "ema": _named(shadow),
            },
        )
    return trace


class _StepProbe:
    """Records what a train step consumed, from inside it.

    The objective's output and the gradients the optimizer steps on are both
    gone by the time ``train_step`` returns -- it zeroes the gradients -- so
    each is read at the call that consumes it: the objective through a
    forward hook on the model it drives, the gradients through the
    optimizer's own step pre-hook.
    """

    def __init__(self, step: SpeedrunDiTTrainStep) -> None:
        self._model = step.model
        self._objective = step.objective
        self.result: SpeedrunDiTLoss.Output
        self.grads: list[tuple[str, Tensor]] = []
        step.objective = _RecordingObjective(self)
        optimizer = step.optimizer
        assert isinstance(optimizer, torch.optim.Optimizer)
        _ = optimizer.register_step_pre_hook(self._capture)

    def _capture(self, optimizer: object, args: object, kwargs: object) -> None:
        del optimizer, args, kwargs
        self.grads = [
            (name, p.grad.detach().clone())
            for name, p in self._model.named_parameters()
            if p.grad is not None
        ]


def compare_forward(reference: nn.Module, candidate: SpeedrunDiT) -> bool:
    """Compare one eval forward, with the sparse path kept and dropped.

    Eval runs the middle stage dense and ``uncond`` discards it, which are
    the two routes the training steps above never take.

    Args:
      reference: The reference model, perturbed.
      candidate: The port, perturbed identically.

    Returns:
      ok: Whether every output matched.

    """
    print("eval forward")
    batch = draw_inputs(
        torch.Generator().manual_seed(3),
        classes=cast(int, GEOMETRY["num_classes"]),
    )
    media, time, label = batch["media"], batch["time"], batch["label"]
    cls_token = batch["cls_token"]
    reference.eval()
    candidate.eval()
    ok = True
    for uncond in (False, True):
        left = cast(
            tuple[Tensor, list[Tensor], Tensor],
            reference(media, time, label, cls_token=cls_token, uncond=uncond),
        )
        right = candidate(media, time, label, cls_token, uncond=uncond)
        pairs = [(left[0], right.velocity), (left[2], right.cls_velocity)]
        pairs.extend(zip(left[1], right.projections, strict=True))
        for name, (a, b) in zip(("velocity", "cls", "projection"), pairs, strict=False):
            ok &= report(f"uncond={uncond} {name}", *tensors_equal(a, b))
    return ok


def compare_sampler() -> bool:
    """Compare guided sampling with the reference's FID sampler.

    Four steps with guidance on an interval that excludes the first, so the
    guided branch, the unguided branch, and the final deterministic step are
    all reached.

    Returns:
      ok: Whether both streams' samples matched.

    """
    sampler = upstream_module("samplers").euler_maruyama_sampler_path_drop
    print("sampler")
    geometry = {**GEOMETRY, "num_classes": SAMPLER_CLASSES}
    reference, candidate = build_pair(geometry, seed=1234)
    perturb(reference, candidate)
    reference.eval()
    candidate.eval()
    batch = draw_inputs(torch.Generator().manual_seed(8), classes=SAMPLER_CLASSES)
    media, label, cls_token = batch["media"], batch["label"], batch["cls_token"]
    guidance, high = 2.5, 0.85
    args = argparse.Namespace(
        time_shifting=True,
        shift_base=4096,
        cls_cfg_scale=guidance,
    )

    torch.manual_seed(11)
    left = sampler(
        reference,
        media,
        label,
        num_steps=4,
        cfg_scale=guidance,
        guidance_high=high,
        cls_latents=cls_token,
        args=args,
    ).to(torch.float32)
    config = EulerMaruyamaSampler.Config(num_steps=4, guidance=guidance)
    config.guidance_interval = (0.0, high)
    torch.manual_seed(11)
    right = config.make()(candidate, media, label, cls_token)
    return report("guided latents", *tensors_equal(left, right.media))


def initialized_state(build: Callable[[], nn.Module]) -> dict[str, Tensor]:
    """Construct a model exactly as the initialization golden replays it.

    Seed zero on a forked stream, inside host-agnostic numerics, and the
    stream's state afterwards alongside the tensors: a construction that drew
    a different NUMBER of values fails even where the tensors it kept agree.

    Args:
      build: Constructs the model.

    Returns:
      state: The ``state_dict``, plus the RNG state under ``"rng"``.

    """
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.default_generator.manual_seed(0)
        state = dict(build().state_dict())
        state["rng"] = torch.get_rng_state()
    return state


def compare_initialization(golden: Path | None) -> bool:
    """Compare both constructions, and mint the golden from the reference.

    Minted from the REFERENCE, never the port, under the port's state names:
    the golden then freezes what the pinned commit initializes, and the port
    passes it only by agreeing.

    Args:
      golden: Where to write the golden, or ``None`` to compare only.

    Returns:
      ok: Whether every tensor and the RNG state matched.

    """
    build = upstream_module("models.sit").SiT
    reference = initialized_state(
        lambda: build(qk_norm=True, fused_attn=True, **GEOMETRY),
    )
    candidate = initialized_state(lambda: native_config().make())
    ok = compare_named(
        list(reference.items()),
        list(candidate.items()),
        "state and RNG",
    )
    if ok and golden is not None:
        torch.save({name: reference[upstream_name(name)] for name in candidate}, golden)
        print(f"  minted {golden}")
    return ok


def main() -> int:
    """Run every comparison and report.

    Returns:
      status: Zero when every checkpoint matched.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    with pinned_source(flags.upstream) as upstream:
        sys.path.insert(0, str(upstream))
        ok = compare_loader(upstream)
        ok &= compare_imagenet_convert(upstream)

        print("initialization")
        golden = _CWD.parent / "testdata" / INIT_GOLDEN
        ok &= compare_initialization(golden if flags.mint else None)
        reference, candidate = build_pair(GEOMETRY, seed=1234)
        ok &= compare_named(
            list(reference.named_parameters()),
            list(candidate.named_parameters()),
            "parameters after init",
        )
        ok &= compare_trainable_order(reference, candidate)
        # Kept pristine for the forward below; the steps move the originals.
        reference_eval, candidate_eval = (
            copy.deepcopy(reference),
            copy.deepcopy(candidate),
        )

        batches = build_batches(STEPS)
        print(f"{STEPS} training steps, exp000's train step")
        step = native_step(seed=1234)
        ok &= compare_named(
            list(reference.named_parameters()),
            list(step.model.named_parameters()),
            "train step's own initialization",
        )
        torch.manual_seed(777)
        with host_agnostic_numerics():
            left = run_upstream(
                reference,
                batches,
                update_ema=upstream_update_ema(upstream),
            )
        torch.manual_seed(777)
        with host_agnostic_numerics():
            right = run_native(step, batches)

        for index, (a, b) in enumerate(zip(left, right, strict=True), start=1):
            for key in ("time", "noise", "loss"):
                ok &= report(f"step {index} {key}", *tensors_equal(a[key], b[key]))
            ok &= compare_named(a["grads"], b["grads"], f"step {index} gradients")
            ok &= compare_named(a["weights"], b["weights"], f"step {index} weights")
            ok &= compare_named(a["ema"], b["ema"], f"step {index} EMA shadow")

        perturb(reference_eval, candidate_eval)
        with host_agnostic_numerics():
            ok &= compare_forward(reference_eval, candidate_eval)
            ok &= compare_sampler()

    print()
    print("known structural difference, not normalized away:")
    print("  - parameter names differ (see RENAMES), and the root's frozen")
    print("    pos_embed enumerates before the mask token rather than after;")
    print("    tensors are compared by name, trainable order separately.")
    print()
    print("PARITY HOLDS" if ok else "PARITY BROKEN")
    return 0 if ok else 1


class _Flags(Protocol):
    """Parsed command line."""

    upstream: Path | None
    mint: bool


class _RecordingObjective(SpeedrunDiTLoss):
    """The step's objective, keeping its last output on the probe."""

    def __init__(self, probe: _StepProbe) -> None:
        super().__init__(probe._objective.config)  # noqa: SLF001 -- The probe owns the wrapped objective.
        self._probe = probe

    @override
    def __call__(
        self,
        model: VelocityField,
        *,
        media: Tensor,
        label: Tensor,
        cls_token: Tensor,
        features: Sequence[Tensor] = (),
        time: Tensor | None = None,
        noise: Tensor | None = None,
        noise_cls: Tensor | None = None,
    ) -> SpeedrunDiTLoss.Output:
        self._probe.result = super().__call__(
            model,
            media=media,
            label=label,
            cls_token=cls_token,
            features=features,
            time=time,
            noise=noise,
            noise_cls=noise_cls,
        )
        return self._probe.result


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--upstream",
        type=Path,
        default=None,
        help="Existing reference checkout; omit to clone the pinned commit.",
    )
    parser.add_argument(
        "--mint",
        action="store_true",
        help=f"Write the reference's initialization to testdata/{INIT_GOLDEN}.",
    )


def _named(model: nn.Module) -> list[tuple[str, Tensor]]:
    """Snapshot every named parameter."""
    return [(n, p.detach().clone()) for n, p in model.named_parameters()]


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
