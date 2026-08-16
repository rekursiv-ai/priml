#!/bin/sh
# ruff: noqa: EXE003, D300, T201 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Run the reference's own train.py, on this package's portable kernel.

The companion to a plain ``exp001`` launch: one card runs THEIR script, the
other runs OURS, both against the same corpus and the same budget. The parity
scripts already prove the two agree step for step; this asks the question they
cannot, which is what a whole budgeted run produces end to end.

Their script is imported and executed unmodified. Exactly three things are
supplied from here, and each is a property of THIS MACHINE rather than of the
recipe:

* **The kernel.** Their ``train.py`` imports FlashAttention-3, which builds
  only for SM90; this box is SM120, so a ``kernels`` stub hands their own
  ``fa3`` symbol ``exp001``'s portable SDPA kernel. The same substitution
  ``karpathy_parity.py`` makes, for the same reason.
* **The corpus.** Their loader reads ``~/.cache/autoresearch``; both sides are
  pointed at one prepared directory so neither is scored on its own data.
* **The microbatch.** Their ``DEVICE_BATCH_SIZE`` of 128 peaked at 45,058 MiB
  on an H100 and does not fit this card's 31.4 GiB. The TOKEN batch per
  optimizer step is unchanged -- only how many passes reach it -- so the
  gradient is the same and the schedule is untouched.

Nothing else is theirs to lose: their depth, widths, learning rates, schedules,
time budget, and evaluator all run as written.

Examples:
  CUDA_VISIBLE_DEVICES=0 karpathy_race.py
  CUDA_VISIBLE_DEVICES=0 karpathy_race.py --rows 16 --corpus /opt/scratch/datasets/nanochat-priml

Example Output:

````
$ CUDA_VISIBLE_DEVICES=0 karpathy_race.py
upstream:  /opt/scratch/karpathy-autoresearch/train.py
corpus:    /opt/scratch/datasets/nanochat-priml
kernel:    their_attention (SDPA; FA3 needs SM90)
rows/pass: 32 (theirs is 128; 45 GiB on H100)
vocab:     8,192
device:    NVIDIA GeForce RTX 5090
Vocab size: 8,192
Model config: {'sequence_len': 2048, 'vocab_size': 8192, 'n_layer': 8, 'n_head': 4, 'n_kv_head': 4, 'n_embd': 512, 'window_pattern': 'SSSL'}
Parameter counts:
  wte                     : 4,194,304
  value_embeds            : 16,777,216
  lm_head                 : 4,194,304
  transformer_matrices    : 25,166,336
  scalars                 : 16
  total                   : 50,332,176
Estimated FLOPs per token: 2.390784e+08
Scaling AdamW LRs by 1/sqrt(512/768) = 1.224745
Time budget: 300s
Gradient accumulation steps: 8
step 00190 (99.9%) | loss: 3.304897 | lrm: 0.00 | dt: 1678ms | tok/sec: 312,513 | mfu: 7.6% | epoch: 1 | remaining: 0s
---
val_bpb:          1.174512
training_seconds: 301.3
total_seconds:    348.4
peak_vram_mb:     11709.0
mfu_percent:      7.61
total_tokens_M:   100.1
num_steps:        191
num_params_M:     50.3
depth:            8

RESULT: {
  "device": "NVIDIA GeForce RTX 5090",
  "rows_per_pass": 32,
  "steps": 191,
  "training_seconds": 301.2872302532196,
  "val_bpb": 1.17451236968107,
  "wall_seconds": 349.0457171459857
}
```

See Also:
  CUDA_VISIBLE_DEVICES=1 uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp001 --override checkpointing.resume=False

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Any

import argparse
import json
import re
import subprocess
import sys
import time
import types

from torch import Tensor

import torch

from priml.model.value_gated_attention import sdpa_attention


def their_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool,
    window_size: tuple[int, int],
) -> Tensor:
    """``exp001``'s own kernel, behind FlashAttention-3's call signature.

    Args:
      q: ``[B, S, heads, channels_head]`` queries.
      k: Keys, same shape.
      v: Values, same shape.
      causal: Whether the mask is causal; the recipe always passes True.
      window_size: ``(history, future)``; the recipe always passes future 0.

    Returns:
      out: Attention output, same shape as ``q``.

    """
    assert causal, "the recipe attends causally"
    assert window_size[1] == 0, f"unexpected future window {window_size[1]}"
    return sdpa_attention(q, k, v, window=window_size[0])


def clone_upstream(
    root: Path,
    *,
    url: str = "https://github.com/karpathy/autoresearch.git",
    commit: str = "b11d6f283f866eb7e10fb776a4b8553fef873fd5",
) -> Path:
    """Clone the reference at its pinned commit, or verify an existing clone.

    Args:
      root: Directory the clone lives in.
      url: Repository to clone.
      commit: Revision this run is of.

    Returns:
      path: The clone's path.

    Raises:
      RuntimeError: The clone is dirty or at another commit, so what it holds
        is no longer the reference this run claims to be.

    """
    if not (root / ".git").is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603 -- fixed URL and commit from the signature
            ["git", "clone", "--quiet", url, str(root)],  # noqa: S607
            check=True,
        )
        subprocess.run(  # noqa: S603 -- fixed URL and commit from the signature
            ["git", "checkout", "--quiet", commit],  # noqa: S607
            cwd=root,
            check=True,
        )
    head = _git(root, "rev-parse", "HEAD")
    if head != commit:
        raise RuntimeError(f"clone is at {head}, expected {commit}")
    if dirty := _git(root, "status", "--porcelain"):
        raise RuntimeError(f"clone has local modifications:\n{dirty}")
    return root


def main() -> int:
    """Run their script to completion and report what it scored."""
    args = _parse_args()
    root = clone_upstream(args.clone)

    sys.path.insert(0, str(root))
    sys.modules["kernels"] = _kernels_stub()

    # Their corpus location, and the microbatch this card can hold. Both are
    # module-level constants their script reads at import, so they are set
    # before it runs rather than edited into it.
    prepare = _import_prepare(args.corpus)
    source = _resize_microbatch((root / "train.py").read_text(), rows=args.rows)

    print(f"upstream:  {root}/train.py")
    print(f"corpus:    {args.corpus}")
    print(f"kernel:    {their_attention.__qualname__} (SDPA; FA3 needs SM90)")
    print(f"rows/pass: {args.rows} (theirs is 128; 45 GiB on H100)")
    print(f"vocab:     {prepare.Tokenizer.from_directory().get_vocab_size():,}")
    print(f"device:    {torch.cuda.get_device_name(0)}", flush=True)

    started = time.perf_counter()
    # Run as ``__main__`` with its own globals: their file is a script, not a
    # module, and its whole training loop is at module scope.
    result: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(root / "train.py"),
    }
    exec(compile(source, str(root / "train.py"), "exec"), result)  # noqa: S102 -- their own script, from the pinned clone
    elapsed = time.perf_counter() - started

    summary = {
        "val_bpb": result.get("val_bpb"),
        "steps": result.get("step"),
        "training_seconds": result.get("total_training_time"),
        "wall_seconds": elapsed,
        "rows_per_pass": args.rows,
        "device": torch.cuda.get_device_name(0),
    }
    print(f"\nRESULT: {json.dumps(summary, indent=2, sort_keys=True)}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"wrote {args.output}")
    return 0


def _resize_microbatch(source: str, *, rows: int) -> str:
    """Rewrite their ``DEVICE_BATCH_SIZE`` constant to what this card holds.

    Their script assigns it at module scope, so it cannot be injected: any
    value handed in beforehand is overwritten the moment the line runs. It is
    replaced in the source text instead, and the substitution is asserted --
    a silent miss would OOM 128 rows into 31 GiB, which is exactly the failure
    this exists to avoid.

    The TOKEN batch per optimizer step is untouched (their ``TOTAL_BATCH_SIZE``
    is unchanged), so this buys passes, not gradient.

    Args:
      source: Their ``train.py``, verbatim.
      rows: Rows per forward/backward pass.

    Returns:
      source: The same text, with that one constant changed.

    Raises:
      RuntimeError: The constant is not where this expects it, so the clone is
        not the revision this script was written against.

    """
    pattern = re.compile(r"^DEVICE_BATCH_SIZE = \d+", re.MULTILINE)
    patched, count = pattern.subn(f"DEVICE_BATCH_SIZE = {rows}", source)
    if count != 1:
        raise RuntimeError(
            f"expected one DEVICE_BATCH_SIZE assignment in train.py; found {count}.",
        )
    return patched


def _import_prepare(corpus: Path) -> types.ModuleType:
    """Import their ``prepare`` module, pointed at the shared corpus.

    Their file computes both directories at import from ``~/.cache``. They are
    rebound afterwards -- and only they -- so both sides of the race read the
    identical shards and the identical vocabulary.

    ``TOKENIZER_DIR`` is also a DEFAULT ARGUMENT of their
    ``Tokenizer.from_directory``, bound at definition and so unaffected by the
    rebinding; the default is replaced too, since their ``train.py`` calls it
    with no argument.

    Args:
      corpus: Directory holding the shards and ``tokenizer/``.

    Returns:
      module: Their ``prepare`` module.

    """
    import importlib  # noqa: PLC0415 -- imported after sys.path is prepared

    prepare = importlib.import_module("prepare")
    prepare.DATA_DIR = str(corpus)  # ty: ignore[unresolved-attribute] -- dynamically imported module  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module
    prepare.TOKENIZER_DIR = str(corpus / "tokenizer")  # ty: ignore[unresolved-attribute] -- dynamically imported module  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module
    prepare.Tokenizer.from_directory.__func__.__defaults__ = (
        str(corpus / "tokenizer"),
    )
    return prepare


def _kernels_stub() -> types.ModuleType:
    """A ``kernels`` module whose ``get_kernel`` yields the portable kernel."""
    module = types.ModuleType("kernels")

    def get_kernel(name: str) -> types.SimpleNamespace:
        assert "flash-attention-3" in name, name
        return types.SimpleNamespace(
            flash_attn_interface=types.SimpleNamespace(
                flash_attn_func=their_attention,
            ),
        )

    module.get_kernel = get_kernel  # ty: ignore[unresolved-attribute] -- stub module built at runtime  # pyright: ignore[reportAttributeAccessIssue] -- stub module built at runtime
    return module


def _git(root: Path, *arguments: str) -> str:
    """Run a read-only git command in the clone."""
    return subprocess.run(  # noqa: S603 -- fixed read-only subcommands from the caller
        ["git", *arguments],  # noqa: S607
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clone",
        type=Path,
        default=Path("/opt/scratch/karpathy-autoresearch"),
        help="Where the reference is cloned.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("/opt/scratch/datasets/nanochat-priml"),
        help="Prepared shards and tokenizer, read by both sides.",
    )
    # Measured on this card: 32 peaks at 22.9 GiB of 31.4, and 64 OOMs. The
    # token batch per optimizer step is unchanged, so this costs passes rather
    # than gradient.
    parser.add_argument(
        "--rows",
        type=int,
        default=32,
        help="Rows per forward/backward pass; theirs is 128.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the result summary as JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
