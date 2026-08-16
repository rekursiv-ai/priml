#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Download the corpus and fit the vocabulary it is read through.

Run once before the first experiment. Re-running costs nothing: a shard already
present is left alone, and a vocabulary whose recorded recipe matches the
request is reused. One that cannot be matched is refitted, since these
artifacts are derived and this is what derives them.

Two stages, in order:

1. **Download.** Parquet text shards, fetched once and reused. The last one is
   the validation shard, so every candidate is scored on text no run trained on.
2. **Fit the vocabulary.** A byte-pair vocabulary fitted on a prefix of the
   training text. The vocabulary is part of the recipe -- it decides what a
   token IS, and therefore what the score's denominator counts -- so it is built
   here and frozen, never refitted per run.

Rows are NOT packed here. The reference packs at read time out of a document
buffer, and that packing is what
:mod:`priml.baselines.nanochat.data` reproduces; writing rows to disk would
freeze one arrangement of the tokens and call it the data.

The tokenizer trainer is imported HERE and nowhere else. Preparation happens
once, on a machine with a network; training reads shards and a pickled encoding,
so a published install needs no BPE trainer.

Examples:
  prepare_data.py
  prepare_data.py --directory /datasets/my-nanochat

References:
    https://github.com/karpathy/autoresearch
      ``prepare.py``, commit b11d6f283f866eb7e10fb776a4b8553fef873fd5.

'''
# fmt: on

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import argparse
import json
import logging
import os
import shutil
import tempfile
import urllib.request

import numpy as np

from priml.baselines.nanochat.data import NanoChatData, token_bytes_fingerprint
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    import tiktoken


logger = logging.getLogger(__name__)

SOURCE_URL: Final = (
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve"
)
"""Base URL of the source shards.

Fetched over plain HTTP rather than through a Hugging Face client: a handful of
files, nothing a dependency would add. Keeping it stdlib means preparing data
needs no optional extra."""

SOURCE_REVISION: Final = "main"
"""Revision the shards are fetched at.

NOT a pin: this tracks the branch, so a corpus that moves upstream produces
different shards under the same name. What IS pinned is the vocabulary those
shards produce -- ``tokenizer_recipe.json`` records the byte-length table by
digest, so a score measured against one preparation cannot be silently compared
with a score measured against another. Digest-pinning the shards themselves
would mean listing every one of them here."""


def main() -> int:
    """Prepare the corpus and vocabulary; return the process exit code."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare(
        args.directory,
        num_train_shards=args.num_train_shards,
        vocab_size=args.vocab_size,
        tokenizer_train_chars=args.tokenizer_train_chars,
        tokenizer_doc_cap=args.tokenizer_doc_cap,
    )
    return 0


def default_directory() -> Path:
    """Return the corpus directory a default ``TrainLoop`` would resolve."""
    config = NanoChatData.Config()
    config.base_dir = TrainLoop.Config().base_dir
    return Path(config.copy_tree().finalize().working_dir)


def prepare(
    directory: Path | str | None = None,
    *,
    num_train_shards: int = 7,
    vocab_size: int = 8_192,
    tokenizer_train_chars: int = 2_000_000_000,
    tokenizer_doc_cap: int = 10_000,
    download: bool = True,
) -> Path:
    """Fetch the shards and fit the vocabulary if they are not already there.

    Args:
      directory: Destination; ``None`` uses :func:`default_directory`.
      num_train_shards: Shards forming the training split. The shard after them
        is the validation split.
      vocab_size: Vocabulary size, including the reserved tokens.
      tokenizer_train_chars: Characters the vocabulary is fitted on.
      tokenizer_doc_cap: Characters one document may contribute to that fit.
        Capped so a handful of long documents cannot dominate the merges.
      download: Fetch missing shards. False expects them staged already, which
        is what lets a test build the pipeline hermetically.

    Returns:
      directory: Where the corpus and vocabulary live.

    """
    if num_train_shards <= 0:
        raise ValueError(f"num_train_shards must be positive; got {num_train_shards}.")
    if vocab_size <= len(RESERVED_TOKENS):
        raise ValueError(
            f"vocab_size must exceed the {len(RESERVED_TOKENS)} reserved tokens; "
            f"got {vocab_size}.",
        )
    if tokenizer_train_chars <= 0 or tokenizer_doc_cap <= 0:
        raise ValueError(
            "tokenizer_train_chars and tokenizer_doc_cap must be positive; got "
            f"{tokenizer_train_chars} and {tokenizer_doc_cap}.",
        )
    out = Path(directory) if directory is not None else default_directory()
    out.mkdir(parents=True, exist_ok=True)
    shards = (
        _download(out, count=num_train_shards + 1)
        if download
        else _staged(out, count=num_train_shards + 1)
    )
    _fit_vocabulary(
        shards[:num_train_shards],
        out=out / "tokenizer",
        vocab_size=vocab_size,
        train_chars=tokenizer_train_chars,
        doc_cap=tokenizer_doc_cap,
    )
    logger.info("nanochat corpus ready at %s", out)
    return out


def _download(out: Path, *, count: int) -> list[Path]:
    """Fetch the source shards, skipping any already present."""
    paths: list[Path] = []
    for index in range(count):
        name = f"shard_{index:05d}.parquet"
        path = out / name
        if not path.is_file():
            url = f"{SOURCE_URL}/{SOURCE_REVISION}/{name}"
            logger.info("downloading %s", url)
            handle, staged = tempfile.mkstemp(dir=out, prefix=f".{name}.")
            os.close(handle)
            staging = Path(staged)
            # Stream rather than read whole: a shard is hundreds of MB.
            with (
                urllib.request.urlopen(url) as response,  # noqa: S310 -- fixed https URL from pinned constants
                staging.open("wb") as file,
            ):
                shutil.copyfileobj(response, file)
            staging.replace(path)
        paths.append(path)
    return paths


def _staged(out: Path, *, count: int) -> list[Path]:
    """Return shards already present, refusing a short corpus.

    Raises:
      FileNotFoundError: Fewer shards are staged than the splits need.

    """
    paths = [out / f"shard_{index:05d}.parquet" for index in range(count)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{out} is missing {missing}.")
    return paths


def _fit_vocabulary(
    shards: list[Path],
    *,
    out: Path,
    vocab_size: int,
    train_chars: int,
    doc_cap: int,
) -> None:
    """Fit a byte-pair vocabulary, or verify the one already fitted.

    Args:
      shards: Training shards supplying the text.
      out: Directory receiving ``tokenizer.pkl`` and its byte table.
      vocab_size: Vocabulary size, including the reserved tokens.
      train_chars: Characters to fit on.
      doc_cap: Characters one document may contribute.

    """
    # Imported here, not at module scope: training reads a pickled encoding, so
    # a published install must not need a BPE trainer to import this package.
    import rustbpe  # noqa: PLC0415 -- preparation-only dependency
    import tiktoken  # noqa: PLC0415 -- preparation-only dependency

    out.mkdir(parents=True, exist_ok=True)
    pickled = out / "tokenizer.pkl"
    recipe_path = out / "tokenizer_recipe.json"
    # The recipe that produced the vocabulary, recorded beside it. A vocabulary
    # fitted on different text IS a different tokenizer even at the same size,
    # so comparing only its length would serve a stale one whose merges came
    # from a corpus this run never saw -- and every token id would mean
    # something else.
    recipe: dict[str, Any] = {
        "vocab_size": vocab_size,
        "train_chars": train_chars,
        "doc_cap": doc_cap,
        "shards": [shard.name for shard in shards],
        "split_pattern": SPLIT_PATTERN,
        "bos_token": BOS_TOKEN,
    }
    # A vocabulary is REUSED only when it can prove it matches this request:
    # one fitted on different text is a different tokenizer at the same size,
    # so every token id would mean something else. Anything unprovable is
    # refitted rather than refused -- the artifacts are derived, this function
    # is how they are derived, and stopping to make a caller delete a file by
    # hand serves nobody. Only what THIS function writes is replaced; the
    # downloaded shards beside it are never touched.
    recorded = (
        json.loads(recipe_path.read_text())
        if pickled.is_file() and recipe_path.is_file()
        else None
    )
    # Compared on the REQUESTED keys only: the written recipe also carries the
    # byte table's fingerprint, which no request states.
    if (
        recorded is not None
        and not _differing(recorded, recipe)
        and (out / "token_bytes.npy").is_file()
    ):
        logger.info("nanochat vocabulary already fitted at %s", out)
        return
    if pickled.is_file():
        logger.info(
            "refitting the nanochat vocabulary at %s: %s",
            out,
            "no recipe recorded beside it"
            if recorded is None
            else f"recipe differs {_differing(recorded, recipe)}",
        )
        for stale in out.iterdir():
            if stale.is_file():
                stale.unlink()

    logger.info("fitting a %d-token vocabulary", vocab_size)
    trainer = rustbpe.Tokenizer()
    trainer.train_from_iterator(
        _documents(shards, max_chars=train_chars, doc_cap=doc_cap),
        vocab_size - len(RESERVED_TOKENS),
        pattern=SPLIT_PATTERN,
    )
    ranks = {bytes(token): rank for token, rank in trainer.get_mergeable_ranks()}
    encoding = tiktoken.Encoding(
        name="rustbpe",
        pat_str=trainer.get_pattern(),
        mergeable_ranks=ranks,
        special_tokens={
            name: len(ranks) + index for index, name in enumerate(RESERVED_TOKENS)
        },
    )
    probe = "Hello world! Numbers: 123. Unicode: 你好"
    if encoding.decode(encoding.encode_ordinary(probe)) != probe:
        raise RuntimeError(
            "the fitted vocabulary does not round-trip its own probe text, so "
            "it cannot be trusted to encode the corpus.",
        )

    token_bytes = _token_bytes(encoding)
    np.save(out / "token_bytes.npy", token_bytes)
    # Written LAST, and staged: the loader reads the recipe to decide whether
    # the artifact is usable, so an interruption must not leave one that claims
    # a byte table it does not have.
    import pickle  # noqa: PLC0415 -- serialization for the prepared artifact

    staging = pickled.with_suffix(".pkl.partial")
    with staging.open("wb") as file:
        pickle.dump(encoding, file)
    staging.replace(pickled)
    recipe["token_bytes_sha256"] = token_bytes_fingerprint(token_bytes)
    recipe_staging = recipe_path.with_suffix(".json.partial")
    recipe_staging.write_text(json.dumps(recipe, sort_keys=True))
    recipe_staging.replace(recipe_path)
    logger.info("nanochat vocabulary fitted: %d tokens", encoding.n_vocab)


def _differing(
    recorded: dict[str, Any],
    requested: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Requested keys whose recorded value differs, as ``(recorded, wanted)``."""
    return {
        key: (recorded.get(key), value)
        for key, value in requested.items()
        if recorded.get(key) != value
    }


def _token_bytes(encoding: tiktoken.Encoding) -> np.ndarray:
    """UTF-8 byte length of every token id; reserved tokens count as zero.

    The bits-per-byte score divides by these, so a reserved token contributing
    zero is what keeps document boundaries out of the denominator.

    ``decode`` is deliberate, and NOT interchangeable with
    ``decode_single_token_bytes``. The two disagree on tokens that are not valid
    UTF-8 on their own -- a lone high byte decodes to U+FFFD, whose re-encoding
    is three bytes rather than one -- so they are two different denominators,
    and a score is comparable only against others using the same one. This
    spelling is the reference's, and every recorded result was measured under
    it.

    Changing it is therefore a protocol change, not a bug fix: it needs a new
    ``token_bytes_sha256``, which is what stops the two being confused.
    """
    reserved = set(RESERVED_TOKENS)
    lengths = [
        0 if (text := encoding.decode([token])) in reserved else len(text.encode())
        for token in range(encoding.n_vocab)
    ]
    return np.array(lengths, dtype=np.int32)


def _documents(
    shards: list[Path],
    *,
    max_chars: int,
    doc_cap: int,
) -> Iterator[str]:
    """Yield capped documents from parquet shards, stopping after ``max_chars``.

    Each document is truncated to ``doc_cap`` BEFORE it is counted, so the
    budget is spent on many documents rather than on a few long ones -- which is
    what keeps the merges representative of the corpus rather than of its
    outliers.
    """
    from pyarrow import parquet  # noqa: PLC0415 -- preparation-only dependency

    seen = 0
    for path in shards:
        shard = parquet.ParquetFile(path)
        for group in range(shard.num_row_groups):
            column: list[Any] = shard.read_row_group(group).column("text").to_pylist()
            for document in column:
                text = str(document)[:doc_cap]
                seen += len(text)
                yield text
                if seen >= max_chars:
                    return


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--directory",
        default=None,
        help=f"destination (default: {default_directory()})",
    )
    parser.add_argument(
        "--num-train-shards",
        type=int,
        default=7,
        help="shards forming the training split",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8_192,
        help="vocabulary size, including reserved tokens",
    )
    parser.add_argument(
        "--tokenizer-train-chars",
        type=int,
        default=2_000_000_000,
        help="characters the vocabulary is fitted on",
    )
    parser.add_argument(
        "--tokenizer-doc-cap",
        type=int,
        default=10_000,
        help="characters one document may contribute to the fit",
    )


RESERVED_TOKENS: Final = tuple(f"<|reserved_{index}|>" for index in range(16))
"""Tokens appended after the byte-pair merges.

Sixteen rather than one because a vocabulary cannot be extended after the fact
without renumbering every id: a later task needing a turn separator or a tool
marker takes one of these instead of invalidating every checkpoint."""

BOS_TOKEN: Final = RESERVED_TOKENS[0]
"""Marks a document's start; every packed row and every document begins with it."""

SPLIT_PATTERN: Final = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
"""Regex splitting text before the byte-pair merges are applied.

Caps numbers at two digits and keeps a leading space with its word, both of
which bound how much the vocabulary spends on rare literals."""


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
