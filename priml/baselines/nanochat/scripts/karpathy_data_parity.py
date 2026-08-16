#!/bin/sh
# ruff: noqa: EXE003, D300, T201 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Prove this package's dataloaders emit the reference's exact token stream.

Clones karpathy/autoresearch at the pinned commit, imports its ``prepare.py``
UNMODIFIED, and draws batches from its ``make_dataloader`` beside ours on the
same corpus and the same vocabulary. Every token of every batch is compared with
``torch.equal``; nothing is compared with a tolerance.

This is the companion to ``karpathy_parity.py``, which proves the MODEL and the
OPTIMIZER and deliberately stubs the data out ("None of the three is under
test"). That stub is the reason a data deviation could not be caught there, so
the same argument is made here for the half it excluded: the packing, the
vocabulary the packing reads through, and the evaluation stream's extent.

Nothing about their side is supplied by this script but the directory their
corpus sits in. Their packer reads their own module constants, their tokenizer
is the one on disk, and their ``evaluate_bpb`` is read for the token count it
scores rather than retyped.

Examples:
  karpathy_data_parity.py
  karpathy_data_parity.py --batches 20 --rows 8

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Any

import argparse
import importlib
import subprocess
import sys
import types

from torch import Tensor

import torch

from priml.baselines.nanochat.data import NanoChatData
from priml.baselines.nanochat.scripts.prepare_data import prepare


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
      commit: Revision the comparison is against.

    Returns:
      path: The clone's path.

    Raises:
      RuntimeError: An existing clone is dirty or at another commit, so what it
        contains is no longer the reference this comparison names.

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
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise RuntimeError(f"clone has local modifications:\n{dirty}")
    return root


def load_upstream(root: Path, *, corpus: Path) -> types.ModuleType:
    """Import their ``prepare.py`` as itself, pointed at the shared corpus.

    Their module locates the corpus through two module-level constants computed
    at import, both under ``~/.cache``. Those are rebound afterwards -- and ONLY
    those -- so both sides read the identical shards and the identical
    vocabulary. Their split rule, their packer, their buffer size, and their
    evaluation token count all stay theirs.

    ``TOKENIZER_DIR`` is also a DEFAULT ARGUMENT of their
    ``Tokenizer.from_directory``, bound at definition and therefore unaffected
    by the rebinding; the caller passes the path explicitly for that one.

    Args:
      root: The clone's path.
      corpus: Directory holding the shards and ``tokenizer/``.

    Returns:
      module: Their ``prepare`` module.

    """
    sys.path.insert(0, str(root))
    module = importlib.import_module("prepare")
    module.DATA_DIR = str(corpus)  # ty: ignore[unresolved-attribute] -- dynamically imported module  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module
    module.TOKENIZER_DIR = str(corpus / "tokenizer")  # ty: ignore[unresolved-attribute] -- dynamically imported module  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module
    return module


def build_ours(
    *,
    corpus: Path,
    rows: int,
    device: str,
    num_train_shards: int,
) -> NanoChatData:
    """Prepare the corpus if needed, then build this package's dataset over it.

    Only the corpus location, the batch width, and the device are set: the
    packer's buffer size, its refill granularity, the split rule, and the
    evaluation extent are the recipe, and setting any of them here would be the
    comparison supplying the answer it is meant to check.

    Args:
      corpus: Directory holding the shards and ``tokenizer/``.
      rows: Rows per batch, matched on both sides.
      device: Device batches land on.
      num_train_shards: Shards the training split is made of.

    Returns:
      data: The built dataset.

    """
    # Prepared first: the comparison needs a corpus and a vocabulary, and
    # ``prepare`` skips whatever is already staged -- so this costs a directory
    # listing on a ready corpus and rebuilds a stale or absent one rather than
    # failing with an instruction the caller then has to run by hand.
    prepare(corpus, num_train_shards=num_train_shards)
    config = NanoChatData.Config()
    config.base_dir = "/"
    config.working_dir = str(corpus)
    config.device = device
    config.batch_size = rows
    config.eval_batch_size = rows
    config.num_train_shards = num_train_shards
    config.val_shard = num_train_shards
    return config.make()


def compare(label: str, theirs: Tensor, ours: Tensor) -> str | None:
    """Describe how two token tensors differ, or None when identical.

    Args:
      label: Name reported with a difference.
      theirs: The reference's tensor.
      ours: Ours.

    Returns:
      problem: A description naming the first differing position, or None.

    """
    if theirs.shape != ours.shape:
        return f"{label}: SHAPE {tuple(theirs.shape)} vs {tuple(ours.shape)}"
    if torch.equal(theirs, ours):
        return None
    differing = (theirs != ours).nonzero()
    first = tuple(int(index) for index in differing[0])
    return (
        f"{label}: DIFFERS at {first} "
        f"({int(theirs[first])} vs {int(ours[first])}), "
        f"{len(differing)}/{theirs.numel()} positions"
    )


def compare_stream(
    theirs: Any,
    ours: Any,
    *,
    batches: int,
    tag: str,
) -> list[str]:
    """Draw from both streams together and compare every token.

    Both the inputs and the targets are compared, not only the inputs: the two
    are one row offset by one, so a packer that cut rows at a different place
    could still agree on the inputs of the first batch.

    Args:
      theirs: The reference's generator, yielding ``(x, y, epoch)``.
      ours: An iterator over this package's batches.
      batches: Batches to draw.
      tag: Prefix for each reported difference.

    Returns:
      problems: One description per differing batch, inputs before targets.

    """
    problems: list[str] = []
    for index in range(1, batches + 1):
        their_media, their_label, _ = next(theirs)
        our_batch = next(ours)
        for name, a, b in (
            ("media", their_media, our_batch["media"]),
            ("label", their_label, our_batch["label"]),
        ):
            found = compare(f"{tag}[{index}] {name}", a.cpu(), b.cpu())
            if found:
                problems.append(found)
    return problems


def main() -> int:
    """Draw from both implementations' loaders and report every difference."""
    args = _parse_args()
    root = clone_upstream(args.clone)
    upstream = load_upstream(root, corpus=args.corpus)
    print(f"upstream: {upstream.__file__}")
    print(f"corpus:   {args.corpus}")

    ours = build_ours(
        corpus=args.corpus,
        rows=args.rows,
        device=args.device,
        num_train_shards=args.num_train_shards,
    )
    # Their tokenizer, loaded by their own loader from the same directory ours
    # read. A vocabulary difference would otherwise surface as a token
    # difference on every position and say nothing about the packing.
    their_tokenizer = upstream.Tokenizer.from_directory(str(args.corpus / "tokenizer"))
    if their_tokenizer.get_vocab_size() != ours.tokenizer.vocab_size:
        raise RuntimeError(
            f"vocabularies differ: theirs {their_tokenizer.get_vocab_size()}, "
            f"ours {ours.tokenizer.vocab_size}",
        )
    if their_tokenizer.get_bos_token_id() != ours.tokenizer.bos_token_id:
        raise RuntimeError(
            f"document-start token differs: theirs "
            f"{their_tokenizer.get_bos_token_id()}, ours "
            f"{ours.tokenizer.bos_token_id}",
        )
    print(f"vocab:    {ours.tokenizer.vocab_size} tokens, identical on both sides")

    failures = 0
    for split, our_stream in (
        ("train", iter(ours.train_dataloader())),
        ("val", iter(ours.eval_dataloader())),
    ):
        their_stream = upstream.make_dataloader(
            their_tokenizer,
            args.rows,
            ours.config.max_seq_len,
            split,
        )
        problems = compare_stream(
            their_stream,
            our_stream,
            batches=args.batches,
            tag=split,
        )
        failures += len(problems)
        print(f"\n[{split}] {args.batches} batches: {len(problems)} differ")
        for line in problems[:8]:
            print(f"    {line}")

    # The extent of the evaluation stream, not only its contents: a loader
    # agreeing token for token still reports a different number if it stops
    # somewhere else, and their count is read off their own module rather than
    # retyped here.
    their_batches = upstream.EVAL_TOKENS // (args.rows * ours.config.max_seq_len)
    our_batches = len(ours.eval_dataloader())
    if their_batches != our_batches:
        failures += 1
        print(f"\neval extent: theirs {their_batches} batches, ours {our_batches}")
    else:
        print(f"\neval extent: {our_batches} batches on both sides")

    verdict = "BIT-IDENTICAL" if not failures else f"{failures} DIFFERENCE(S)"
    print(f"\n{args.batches} batches per split: {verdict}")
    return 1 if failures else 0


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
        default=Path("/opt/scratch/datasets/nanochat"),
        help="Prepared shards and tokenizer, read by both sides.",
    )
    parser.add_argument("--batches", type=int, default=8, help="Batches per split.")
    # Small because the packer's state is per row: a difference in which
    # document was chosen shows up in the first row that chooses one, and a
    # wider batch only makes the same answer slower.
    parser.add_argument("--rows", type=int, default=4, help="Rows per batch.")
    parser.add_argument(
        "--num-train-shards",
        type=int,
        default=7,
        help="Shards forming the training split; the next one is validation.",
    )
    # CUDA, because their loader stages into a ``device="cuda"`` buffer
    # unconditionally (``prepare.py:283``) and so cannot run anywhere else.
    parser.add_argument("--device", default="cuda", help="Device batches land on.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
