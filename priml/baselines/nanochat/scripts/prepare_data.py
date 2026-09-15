#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Download the corpus and fit the vocabulary it is read through.

Original BPE preparation:
    Run once before the first BPE experiment. Select this workflow with
    --num-train-shards, --vocab-size, --tokenizer-train-chars, or
    --tokenizer-doc-cap. A shard already present is left alone. A vocabulary
    whose recorded recipe matches the request is reused; a missing or
    mismatched recipe triggers a refit of the derived tokenizer artifacts.

    Two stages, in order:

    1. Download parquet text shards, fetched once and reused. The shard after
       the training prefix is validation, so every candidate is scored on
       text it did not train on.
    2. Fit a byte-pair vocabulary on a prefix of the training text. The
       vocabulary is part of the recipe: it defines the tokens and their
       byte lengths for scoring. It is built here and frozen, never refitted
       per run.

    This workflow does not pack rows. The reference packs at read time from
    a document buffer, as :mod:`priml.baselines.nanochat.data` reproduces.
    Saving those rows would freeze one arrangement of the tokens.

    Tokenizer fitting belongs to the preparation modules. Preparation needs
    network access to fetch missing inputs; BPE training reads local shards,
    a pickled encoding, and the byte-length table without running a trainer.

Added prepared Unigram workflow:
    With no BPE flags, the default recipe prepares inputs for exp020-exp022.
    It fetches pinned source shards, fits the reference BPE vocabulary, moves
    selected training documents, fits a 16K Unigram vocabulary, and writes
    packed training and evaluation rows. Tokenizer fitting excludes validation
    text. Reference evaluation re-encodes the bytes selected by the BPE
    evaluator, preserving document boundaries and excluding padding.

    These experiments read the prepared arrays instead of packing online.
    Use a fresh output directory: completed prepared outputs are not
    overwritten. The original BPE workflow's reuse/refit behavior above does
    not apply to the whole prepared pipeline.

    --print-config shows the full recipe without preparing data. --stage
    selects one operation; its default, all, runs preparation in order.
    --stage verify checks prepared inputs, and --stage dump bundles them.
    --stage train binds the prepared paths and launches through Priml, using
    exp022 unless --experiment selects another recipe. Metrics and the
    resolved configuration are saved in --run-directory; --save-checkpoint
    also retains the final model state. --directory relocates the inputs.

Examples:
    uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --num-train-shards 7
    uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --num-train-shards 7 --directory /opt/scratch/datasets/my-nanochat
    uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --print-config
    uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --directory /opt/scratch/datasets/nanochat-unigram --stage all
    uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --directory /opt/scratch/datasets/nanochat-unigram --stage train --experiment exp022 --seed 42 --run-directory /opt/scratch/runs/nanochat-exp022-42

References:
    https://github.com/karpathy/autoresearch
        Karpathy. prepare.py, commit
        b11d6f283f866eb7e10fb776a4b8553fef873fd5.

'''
# fmt: on

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, Self, cast, override

import argparse
import fcntl
import hashlib
import importlib
import io
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

from numpy import (
    arange,
    array,
    array_equal,
    concatenate,
    count_nonzero,
    flatnonzero,
    full,
    int64,
    load,
    ndarray,
    save,
    savez,
    stack,
    uint16,
)
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

import numpy as np
import rustbpe
import tiktoken
import tokenizers


if TYPE_CHECKING:
    from pyarrow import Table, parquet

    import pyarrow.parquet
    import torch
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    pyarrow = lazy_import("pyarrow")
    torch = lazy_import("torch")

from configgle import Fig

from priml.baselines.nanochat.data import (
    NanoChatData,
    PackedTokenStream,
    PreparedTokenRows,
    Tokenizer,
    token_bytes_fingerprint,
)
from priml.baselines.nanochat.experiments import NgramTrainLoop
from priml.baselines.nanochat.scripts.prepare_tokenizer import (
    ByteLevelTokenizer,
    SamplePreparation,
    UnigramPreparation,
    byte_alphabet,
    document_rows,
    read_mapping,
    write_mapping,
)
from priml.lib.custom_json import DictCodec, IntCodec
from priml.paths import validated_output_path
from priml.train.checkpointing import Checkpointer
from priml.train.tracker import FileTracker, TrackerList
from priml.train.train_loop import TrainLoop


logger = logging.getLogger(__name__)


class _TokenizerToken(Protocol):
    id: int


class _TokenizerModel(Protocol):
    def tokenize(self, text: str) -> list[_TokenizerToken]: ...


_CWD: Final = Path(__file__).resolve().parent


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


class BpePreparation:
    """Fit the baseline byte-pair vocabulary on the original training prefix."""

    class Config(Fig["BpePreparation"]):
        """Configure the baseline fitting corpus and character budget."""

        raw_dir: Path = Path("/opt/scratch/datasets/nanochat/raw")
        """Original shards; the vocabulary is written into their tokenizer directory."""

        num_train_shards: int = 7
        """Training prefix; the following shard is validation."""

        vocab_size: int = 8_192
        """Vocabulary size including sixteen reserved tokens."""

        train_chars: int = 2_000_000_000
        """Fitting budget in document-capped Unicode characters."""

        doc_cap: int = 10_000
        """Maximum fitting characters per document."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def build(self) -> None:
        """Fit from staged shards without changing their revision or heldout split."""
        prepare(
            self.config.raw_dir,
            num_train_shards=self.config.num_train_shards,
            vocab_size=self.config.vocab_size,
            tokenizer_train_chars=self.config.train_chars,
            tokenizer_doc_cap=self.config.doc_cap,
            download=False,
        )


def default_directory() -> Path:
    """Return the corpus directory a default ``TrainLoop`` would resolve.

    Returns:
      result: The Path.

    """
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
                cast(io.BufferedIOBase, urllib.request.urlopen(url)) as response,  # noqa: S310 -- The benchmark fetches a URL supplied by its controlled dataset manifest.
                staging.open("wb") as file,
            ):
                shutil.copyfileobj(response, file)
            staging.replace(path)
        paths.append(path)
    return paths


def _staged(out: Path, *, count: int) -> list[Path]:
    """Return shards already present, refusing a short corpus."""
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
    """Fit a byte-pair vocabulary, or verify the one already fitted."""
    out.mkdir(parents=True, exist_ok=True)
    pickled = out / "tokenizer.pkl"
    recipe_path = out / "tokenizer_recipe.json"
    # The recipe that produced the vocabulary, recorded beside it. A vocabulary
    # fitted on different text IS a different tokenizer even at the same size,
    # so comparing only its length would serve a stale one whose merges came
    # from a corpus this run never saw -- and every token id would mean
    # something else.
    recipe: dict[str, object] = {
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
        dict(
            DictCodec.coerce(
                cast(object, json.loads(recipe_path.read_text())), default=None
            )
        )
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
    # A missing receipt makes an incomplete refit unusable. The other owned
    # artifacts are replaced below; unrelated directory contents are untouched.
    recipe_path.unlink(missing_ok=True)

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
    recorded: dict[str, object],
    requested: dict[str, object],
) -> dict[str, tuple[object, object]]:
    """Return requested keys whose recorded value differs, as ``(recorded, wanted)``."""
    return {
        key: (recorded.get(key), value)
        for key, value in requested.items()
        if recorded.get(key) != value
    }


# The bits-per-byte score divides by these, so a reserved token contributing zero is
# what keeps document boundaries out of the denominator.
#
# ``decode`` is deliberate, and NOT interchangeable with ``decode_single_token_bytes``.
# The two disagree on tokens that are not valid UTF-8 on their own -- a lone high byte
# decodes to U+FFFD, whose re-encoding is three bytes rather than one -- so they are two
# different denominators, and a score is comparable only against others using the same
# one. This spelling is the reference's, and every recorded result was measured under
# it.
#
# Changing it is therefore a protocol change, not a bug fix: it needs a new
# ``token_bytes_sha256``, which is what stops the two being confused.
def _token_bytes(encoding: tiktoken.Encoding) -> np.ndarray:
    """UTF-8 byte length of every token id; reserved tokens count as zero."""
    reserved = set(RESERVED_TOKENS)
    lengths = [
        0 if (text := encoding.decode([token])) in reserved else len(text.encode())
        for token in range(encoding.n_vocab)
    ]
    return np.array(lengths, dtype=np.int32)


# Each document is truncated to ``doc_cap`` BEFORE it is counted, so the budget is spent
# on many documents rather than on a few long ones -- which is what keeps the merges
# representative of the corpus rather than of its outliers.
def _documents(
    shards: list[Path],
    *,
    max_chars: int,
    doc_cap: int,
) -> Iterator[str]:
    """Yield capped documents from parquet shards, stopping after ``max_chars``."""
    seen = 0
    for path in shards:
        shard = parquet.ParquetFile(path)
        for group in range(shard.num_row_groups):
            column = cast(
                list[str], shard.read_row_group(group).column("text").to_pylist()
            )
            for document in column:
                text = document[:doc_cap]
                seen += len(text)
                yield text
                if seen >= max_chars:
                    return


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


def fetch_file(url: str, *, destination: Path) -> None:
    """Download one HTTPS input, reusing only a copy from the same source.

    Args:
      url: Immutable HTTPS input URL.
      destination: Local input cache file.

    Raises:
      ValueError: The URL is not HTTPS or the cached source identity does not match.

    """
    if not url.startswith("https://"):
        raise ValueError("Preparation downloads require HTTPS.")
    destination = validated_output_path(destination)
    source = destination.with_name(destination.name + ".source-url")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode: unlinking it lets waiters and newcomers lock different
    # files and publish a receipt from one source beside another source's bytes.
    with destination.with_name(destination.name + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.is_file():
            if not source.is_file() or source.read_text(encoding="utf-8") != url:
                raise ValueError(
                    f"Cached source does not match {url}: {destination}. "
                    "Use a fresh download directory for this source.",
                )
            return
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix="nanochat-download-"
        ) as temporary:
            staged = Path(temporary) / destination.name
            with (
                cast(
                    io.BufferedIOBase,
                    urllib.request.urlopen(url, timeout=120),  # noqa: S310 -- HTTPS checked above.
                ) as response,
                staged.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            source.write_text(url, encoding="utf-8")
            staged.replace(destination)


def copy_input(source: Path, *, destination: Path) -> None:
    """Copy an artifact while protecting the original from overwrite.

    Args:
      source: Original artifact.
      destination: Independent destination.

    """
    destination = validated_output_path(destination, protected=[source])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix="nanochat-copy-"
    ) as temporary:
        staged = Path(temporary) / destination.name
        shutil.copyfile(source, staged)
        staged.replace(destination)


class CorpusPreparation:
    """Fetch immutable shards and move the frozen original donor documents."""

    class Config(Fig["CorpusPreparation"]):
        """Declare original inputs, donor selection, and serialization."""

        repository: str = "karpathy/climbmix-400b-shuffle"
        """Public Hugging Face dataset repository."""

        revision: str = "915333b4f8b8684f39aeaafea600fea6f43fb703"
        """Immutable upstream source revision."""

        donor_source_ids: tuple[str, ...] = ()
        """Ordered source shard:row identities moved into earlier training shards."""

        raw_dir: Path = Path("/opt/scratch/datasets/nanochat/raw")
        """Original unmodified corpus used for tokenizer fitting."""

        working_dir: Path = Path("/opt/scratch/datasets/nanochat/donor-original")
        """Rebuilt corpus used for model training."""

        train_shard_indices: tuple[int, ...] = (*range(7), *range(8, 15))
        """Ordered training shards; excludes the heldout shard."""

        val_shard: int = 7
        """Unmodified heldout shard."""

        donor_destination_shards: int = 5
        """First shards receiving round-robin donor subsequences."""

        row_group_size: int = 1_024
        """Parquet row-group boundaries, also used by tokenization refills."""

        compression: str = "snappy"
        """Parquet compression matching the packing recipe."""

    def __init__(self, config: Config) -> None:
        self.config = config
        if config.val_shard in config.train_shard_indices:
            raise ValueError("Validation must be excluded from training.")
        if len(set(config.train_shard_indices)) != len(config.train_shard_indices):
            raise ValueError("Training shard identities must be unique.")

    def fetch(self) -> None:
        """Download every source shard at the configured revision."""
        base = (
            f"https://huggingface.co/datasets/{self.config.repository}/resolve/"
            f"{self.config.revision}"
        )
        for shard in sorted((*self.config.train_shard_indices, self.config.val_shard)):
            name = f"shard_{shard:05d}.parquet"
            logger.info("Fetching source %s", name)
            fetch_file(
                base + "/" + name,
                destination=self.config.raw_dir / name,
            )

    def build(self) -> None:
        """Reproduce stable literal deduplication and the ordered donor moves."""
        identities = self.config.donor_source_ids
        if len(set(identities)) != len(identities):
            raise ValueError("The donor selection contains repeated identities.")
        selected = set(identities)
        texts_donor = _donor_texts(self.config.raw_dir, identities=selected)
        output = validated_output_path(
            self.config.working_dir, protected=[self.config.raw_dir]
        )
        output.mkdir(parents=True, exist_ok=True)
        seen: dict[bytes, tuple[str, str]] = {}
        for shard in self.config.train_shard_indices:
            name = f"shard_{shard:05d}.parquet"
            canonical, _ = unique_rows(
                document_rows(self.config.raw_dir / name, shard=shard), seen=seen
            )
            retained = [
                (identity, text)
                for identity, text in canonical
                if identity not in selected
            ]
            additions = (
                identities[shard :: self.config.donor_destination_shards]
                if shard < self.config.donor_destination_shards
                else ()
            )
            rows = interleave_rows(
                retained, additions=[(key, texts_donor[key]) for key in additions]
            )
            destination = validated_output_path(
                output / name, protected=[self.config.raw_dir / name]
            )
            with tempfile.TemporaryDirectory(
                dir=output, prefix="nanochat-shard-"
            ) as temporary:
                staged = Path(temporary) / name
                parquet.write_table(
                    Table.from_pydict({"text": [text for _, text in rows]}),
                    staged,
                    row_group_size=self.config.row_group_size,
                    compression=self.config.compression,
                )
                staged.replace(destination)
            logger.info("Prepared corpus shard %s: %d rows", shard, len(rows))
        val_name = f"shard_{self.config.val_shard:05d}.parquet"
        copy_input(
            self.config.raw_dir / val_name,
            destination=output / val_name,
        )
        if selected - {identity for identity, _ in seen.values()}:
            raise ValueError("A selected donor is not a canonical training document.")


def unique_rows(
    rows: Iterator[tuple[str, str]], *, seen: dict[bytes, tuple[str, str]]
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    """Keep the first exact text occurrence across ordered shards.

    Args:
      rows: Literal source rows.
      seen: Earlier SHA-256 identities, retained across shards.

    Returns:
      kept: Canonical source rows.
      dropped: Duplicate identities with their earlier source.

    """
    kept: list[tuple[str, str]] = []
    dropped: list[dict[str, str]] = []
    for identity, text in rows:
        digest = hashlib.sha256(text.encode()).digest()
        if digest in seen:
            first_id, first_text = seen[digest]
            if text != first_text:
                raise ValueError("SHA-256 collision between distinct literal texts.")
            dropped.append(
                {
                    "source_id": identity,
                    "first_source_id": first_id,
                }
            )
        else:
            seen[digest] = identity, text
            kept.append((identity, text))
    return kept, dropped


def interleave_rows(
    original: list[tuple[str, str]], *, additions: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Insert donors at the original producer's integer-spaced positions.

    Args:
      original: Canonical rows with selected donors removed.
      additions: Ordered donor subsequence for this shard.

    Returns:
      rows: Interleaved rows preserving both input orders.

    """
    slots: dict[int, list[tuple[str, str]]] = {}
    for index, row in enumerate(additions):
        position = (index + 1) * len(original) // (len(additions) + 1)
        slots.setdefault(position, []).append(row)
    output: list[tuple[str, str]] = []
    for index in range(len(original) + 1):
        output.extend(slots.get(index, []))
        if index < len(original):
            output.append(original[index])
    return output


def _donor_texts(directory: Path, *, identities: set[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for shard in sorted({int(key.split(":")[0]) for key in identities}):
        output.update(
            {
                key: text
                for key, text in document_rows(
                    directory / f"shard_{shard:05d}.parquet", shard=shard
                )
                if key in identities
            }
        )
    if set(output) != identities:
        raise ValueError("A donor identity is absent from its original shard.")
    return output


class RowPreparation:
    """Pack token arrays and generate portable loader manifests."""

    class Config(Fig["RowPreparation"]):
        """Declare the full training and evaluation input contract."""

        tokenizer: ByteLevelTokenizer.Config = field(
            default_factory=ByteLevelTokenizer.Config
        )
        """Fitted encoder and reserved IDs."""

        raw_dir: Path = Path("/opt/scratch/datasets/nanochat/donor-original")
        """Donor-original corpus, including unchanged validation shard."""

        working_dir: Path = Path("/opt/scratch/datasets/nanochat/unigram16k/prepared")
        """Destination with separate training and evaluation subdirectories."""

        train_shard_indices: tuple[int, ...] = (*range(7), *range(8, 15))
        """Ordered source shards used by training."""

        val_shard: int = 7
        """Heldout evaluation source."""

        train_batches: int = 3_200
        """Prepared batch supply; training refuses exhaustion."""

        batch_size: int = 96
        """Rows per optimizer update."""

        eval_batches: int = 80
        """Frozen number of validation batches."""

        eval_batch_size: int = 128
        """Rows per validation batch."""

        max_seq_len: int = 2_048
        """Input tokens per row; packing includes one additional target token."""

        train_buffer_size: int = 256
        """Training document-buffer low-water mark."""

        buffer_size: int = 1_000
        """Evaluation document-buffer low-water mark."""

        documents_per_refill: int = 128
        """Training document slices, reset at every source row-group boundary."""

    def __init__(self, config: Config) -> None:
        self.config = config
        if config.val_shard in config.train_shard_indices:
            raise ValueError("Validation must be excluded from training.")

    def build(self) -> None:
        """Write training and evaluation arrays with their loader manifests."""
        output = validated_output_path(
            self.config.working_dir,
            protected=[self.config.raw_dir, self.config.tokenizer.path],
        )
        output.mkdir(parents=True, exist_ok=False)
        (output / "train").mkdir()
        (output / "eval").mkdir()
        encoder = self.config.tokenizer.make()
        self._training(output / "train", encoder=encoder)
        self._evaluation(output / "eval", encoder=encoder)
        self._manifests(output, encoder=encoder)
        self.verify()

    def verify(self) -> PreparedTokenRows:
        """Validate the rebuilt dataset through the NanoChat reader.

        Returns:
          data: Memory-mapped training and evaluation arrays.

        """
        config = NanoChatData.Config()
        config.prepared_train_manifest = (
            self.config.working_dir / "train/PREPARED_MANIFEST.json"
        )
        config.prepared_eval_manifest = (
            self.config.working_dir / "eval/PACKED_EVAL_MANIFEST.json"
        )
        config.batch_size = self.config.batch_size
        config.eval_batch_size = self.config.eval_batch_size
        config.max_seq_len = self.config.max_seq_len
        config.eval_tokens = (
            self.config.eval_batches
            * self.config.eval_batch_size
            * self.config.max_seq_len
        )
        config.train_shard_indices = self.config.train_shard_indices
        config.val_shard = self.config.val_shard
        config.buffer_size = self.config.buffer_size
        config.train_buffer_size = self.config.train_buffer_size
        return PreparedTokenRows(config)

    def _manifests(self, output: Path, *, encoder: ByteLevelTokenizer) -> None:
        """Write portable loader geometry and byte-table metadata."""
        rows: NDArray[np.uint16] = cast(
            NDArray[np.uint16], load(output / "train/train_rows.npy", mmap_mode="r")
        )
        targets: NDArray[np.uint16] = cast(
            NDArray[np.uint16], load(output / "eval/eval_y.npy", mmap_mode="r")
        )
        write_mapping(
            output / "train/PREPARED_MANIFEST.json",
            value={
                "vocab_size": encoder.vocab_size,
                "bos_id": encoder.bos_token_id,
                "train": {
                    "batch_size": self.config.batch_size,
                    "seq_len": self.config.max_seq_len,
                    "buffer_size": self.config.train_buffer_size,
                    "documents_per_refill": self.config.documents_per_refill,
                    "train_shard_indices": self.config.train_shard_indices,
                    "total_rows": len(rows),
                },
            },
        )
        write_mapping(
            output / "eval/PACKED_EVAL_MANIFEST.json",
            value={
                "protocol": "standard-packed-shard7-tokensub-v1",
                "vocab_size": encoder.vocab_size,
                "bos_token_id": encoder.bos_token_id,
                "eval_batch_size": self.config.eval_batch_size,
                "max_seq_len": self.config.max_seq_len,
                "physical_positions": int(targets.size),
                "val_shard": self.config.val_shard,
                "buffer_size": self.config.buffer_size,
                "rows": len(targets),
                "batches": self.config.eval_batches,
                "byte_tables": {
                    "scored_positions": int(
                        count_nonzero(encoder.token_bytes[targets])
                    ),
                    **{
                        name: {
                            "file": f"token_bytes_{name}.npy",
                            "total_on_eval_y": int(
                                cast(NDArray[np.int32], table)[targets].sum()
                            ),
                        }
                        for name, table in (
                            ("primary", encoder.token_bytes),
                            ("literal", encoder.token_bytes_literal),
                        )
                    },
                },
            },
        )

    def _training(self, output: Path, *, encoder: ByteLevelTokenizer) -> None:
        rows = open_memmap(
            output / "train_rows.npy",
            mode="w+",
            dtype=uint16,
            shape=(
                self.config.train_batches * self.config.batch_size,
                self.config.max_seq_len + 1,
            ),
        )
        documents = _document_batches(self.config)
        buffer: list[list[int]] = []
        lengths: list[int] = []
        for index, row in enumerate(cast(Iterator[NDArray[np.uint16]], rows)):
            position = 0
            while position < len(row):
                while len(buffer) < self.config.train_buffer_size:
                    try:
                        texts = next(documents)
                    except StopIteration as error:
                        raise RuntimeError(
                            "Training preparation exhausted the source corpus; wrapping is forbidden."
                        ) from error
                    encoded = encoder.encode_batch(texts)
                    buffer.extend(encoded)
                    lengths.extend(map(len, encoded))
                position = pack_row(
                    row, buffer=buffer, lengths=lengths, position=position
                )
            if index % 20_000 == 0:
                logger.info("Prepared %d/%d training rows", index, len(rows))
        rows.flush()

    def _evaluation(self, output: Path, *, encoder: ByteLevelTokenizer) -> None:
        stream = PackedTokenStream(
            paths=[self.config.raw_dir / f"shard_{self.config.val_shard:05d}.parquet"],
            tokenizer=encoder,
            token_bytes=torch.from_numpy(encoder.token_bytes),
            batch_size=self.config.eval_batch_size,
            max_seq_len=self.config.max_seq_len,
            buffer_size=self.config.buffer_size,
            device=torch.device("cpu"),
            max_batches=self.config.eval_batches,
        )
        inputs: list[ndarray] = []
        targets: list[ndarray] = []
        for batch in stream:
            inputs.append(batch["media"].clone().numpy().astype(uint16))
            targets.append(batch["label"].clone().numpy().astype(uint16))
        for name, values in (
            ("eval_x", concatenate(inputs)),
            ("eval_y", concatenate(targets)),
        ):
            save(output / (name + ".npy"), values)
        save(output / "token_bytes_primary.npy", encoder.token_bytes)
        save(output / "token_bytes_literal.npy", encoder.token_bytes_literal)


def pack_row(
    row: ndarray, *, buffer: list[list[int]], lengths: list[int], position: int
) -> int:
    """Apply the native largest-fit, otherwise shortest-crop packing rule.

    Args:
      row: Destination row including its extra target token.
      buffer: Mutable ordered document buffer.
      lengths: Matching document lengths.
      position: Next output position.

    Returns:
      position: Position after the selected document or cropped prefix.

    """
    remaining = len(row) - position
    best_index = -1
    best_length = 0
    for index, length in enumerate(lengths):
        if best_length < length <= remaining:
            best_index, best_length = index, length
    if best_index < 0:
        best_index = min(range(len(lengths)), key=lengths.__getitem__)
        best_length = remaining
    document = buffer.pop(best_index)
    lengths.pop(best_index)
    row[position : position + best_length] = document[:best_length]
    return position + best_length


def _document_batches(config: RowPreparation.Config) -> Iterator[list[str]]:
    for shard in config.train_shard_indices:
        parquet = pyarrow.parquet.ParquetFile(
            config.raw_dir / f"shard_{shard:05d}.parquet"
        )
        for group in range(parquet.num_row_groups):
            texts = cast(
                list[str], parquet.read_row_group(group).column("text").to_pylist()
            )
            for start in range(0, len(texts), config.documents_per_refill):
                yield texts[start : start + config.documents_per_refill]


class Preparation:
    """Compose corpus, tokenizer, and packing stages."""

    class Config(Fig["Preparation"]):
        """Keep the entire preparation recipe inspectable without doing I/O."""

        working_dir: Path = Path("/opt/scratch/datasets/nanochat")
        """Root for original sources and all reproduced artifacts."""

        corpus: CorpusPreparation.Config = field(
            default_factory=CorpusPreparation.Config
        )
        """Pinned source acquisition and donor-original transformation."""

        baseline: BpePreparation.Config = field(default_factory=BpePreparation.Config)
        """Original BPE vocabulary for the baseline and reference-byte selection."""

        sample: SamplePreparation.Config = field(
            default_factory=SamplePreparation.Config
        )
        """Original-corpus vocabulary fitting sample."""

        tokenizer: UnigramPreparation.Config = field(
            default_factory=UnigramPreparation.Config
        )
        """Tokenizer fitting procedure."""

        rows: RowPreparation.Config = field(default_factory=RowPreparation.Config)
        """Training geometry and separate packed evaluation contract."""

        tokenizer_name: str = "unigram16k"
        """Artifact directory name; algorithm selection lives in the tokenizer slot."""

        experiment_corpora: dict[str, str] = field(default_factory=dict[str, str])
        """Online experiment corpus subdirectories; unlisted experiments use raw."""

        @override
        def finalize(self) -> Self:
            self.corpus.raw_dir = self.working_dir / "raw"
            self.corpus.working_dir = self.working_dir / "donor-original"
            self.baseline.raw_dir = self.corpus.raw_dir
            self.sample.val_shard = self.corpus.val_shard
            self.sample.raw_dir = self.corpus.raw_dir
            self.sample.working_dir = self.working_dir / "sample"
            self.tokenizer.sample_dir = self.sample.working_dir
            self.tokenizer.working_dir = self.working_dir / self.tokenizer_name
            self.rows.raw_dir = self.corpus.working_dir
            self.rows.tokenizer.path = self.tokenizer.working_dir / "tokenizer.json"
            self.rows.working_dir = self.tokenizer.working_dir / "prepared"
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, stage: str) -> None:
        """Execute one stage, or the complete preparation in dependency order.

        Args:
          stage: CLI preparation stage.

        """
        stages = {
            "fetch": self.config.corpus.make().fetch,
            "baseline": self.config.baseline.make().build,
            "corpus": self.config.corpus.make().build,
            "sample": self.config.sample.make().build,
            "tokenizer": self.config.tokenizer.make().build,
            "rows": self.config.rows.make().build,
            "verify": self.config.rows.make().verify,
            "reference": self.reference,
        }
        selected = (
            ("fetch", "baseline", "corpus", "sample", "tokenizer", "rows", "reference")
            if stage == "all"
            else (stage,)
        )
        for name in selected:
            stages[name]()

    def reference(self) -> None:
        """Build both reference-byte replay archives from the original heldout shard."""
        source = (
            self.config.corpus.raw_dir
            / f"shard_{self.config.corpus.val_shard:05d}.parquet"
        )
        tokenizer_dir = self.config.corpus.raw_dir / "tokenizer"
        encoder = Tokenizer.from_directory(tokenizer_dir)
        stream = PackedTokenStream(
            paths=[source],
            tokenizer=encoder,
            token_bytes=torch.from_numpy(encoder.token_bytes),
            batch_size=self.config.rows.eval_batch_size,
            max_seq_len=self.config.rows.max_seq_len,
            buffer_size=self.config.rows.buffer_size,
            device=torch.device("cpu"),
            max_batches=self.config.rows.eval_batches,
        )
        inputs: list[ndarray] = []
        targets: list[ndarray] = []
        for batch in stream:
            inputs.append(batch["media"].clone().numpy().astype(uint16))
            targets.append(batch["label"].clone().numpy().astype(uint16))
        reference = self.config.working_dir / "reference-bpe"
        reference.mkdir(parents=True, exist_ok=False)
        save(reference / "eval_x.npy", concatenate(inputs))
        save(reference / "eval_y.npy", concatenate(targets))
        pickled = tokenizer_dir / "tokenizer.pkl"
        copy_input(
            pickled,
            destination=reference / "tokenizer.pkl",
        )
        manifest = reference / "PACKED_EVAL_MANIFEST.json"
        write_mapping(
            manifest,
            value={
                "eval_batch_size": self.config.rows.eval_batch_size,
            },
        )
        build_reference_eval(
            reference_dir=reference,
            unigram_path=self.config.rows.tokenizer.path,
            output=self.config.working_dir / "reference-eval",
        )

    def dump(self, destination: Path) -> None:
        """Bundle prepared inputs, fitting text, and preparation source.

        Args:
          destination: New uncompressed tar archive below the artifact directory.

        """
        files = {
            "data/" + str(path.relative_to(self.config.rows.working_dir)): path
            for path in self.config.rows.working_dir.rglob("*")
            if path.is_file()
        }
        files.update(
            {
                "tokenizer/" + path.name: path
                for path in self.config.tokenizer.working_dir.iterdir()
                if path.is_file()
            }
        )
        files.update(
            {
                "fitting/" + path.name: path
                for path in self.config.sample.working_dir.iterdir()
                if path.is_file()
            }
        )
        for directory in ("reference-eval", "reference-bpe", "raw/tokenizer"):
            files.update(
                {
                    directory + "/" + path.name: path
                    for path in (self.config.working_dir / directory).iterdir()
                    if path.is_file()
                }
            )
        source = _CWD
        files.update(
            {
                "preparation/" + str(path.relative_to(source)): path
                for path in source.glob("prepare_*.py")
            }
        )
        destination = validated_output_path(destination, protected=files.values())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        manifest = {
            "files": sorted(files),
            "config": self.config.pformat(hide_default_values=False),
        }
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix="nanochat-archive-"
        ) as temporary:
            staged = Path(temporary) / destination.name
            with tarfile.open(staged, mode="w") as archive:
                for name, path in sorted(files.items()):
                    archive.add(path, arcname=name)
                info = tarfile.TarInfo("MANIFEST.json")
                info.size = len(payload)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(payload))
            staged.replace(destination)

    def training_config(
        self, name: str, *, run_directory: Path, seed: int
    ) -> NgramTrainLoop.Config:
        """Bind an experiment to locally rebuilt inputs and local-only reporting.

        Args:
          name: Public experiment factory in experiments.py.
          run_directory: New directory for training outputs.
          seed: Training seed.

        Returns:
          config: Original model and training recipe with portable input locations.

        """
        module = importlib.import_module("priml.baselines.nanochat.experiments")
        if name not in {f"exp{index:03d}" for index in range(4, 24)}:
            raise ValueError(f"Unknown NanoChat experiment: {name}.")
        factory = cast(Callable[[], object], getattr(module, name))
        config = factory()
        if not isinstance(config, NgramTrainLoop.Config):
            raise TypeError("The experiment must return NgramTrainLoop.Config.")
        config.base_dir = "/"
        config.working_dir = validated_output_path(run_directory)
        config.seed = seed
        config.dataset.working_dir = (
            self.config.working_dir / self.config.experiment_corpora.get(name, "raw")
        )
        config.dataset.tokenizer_dir = self.config.corpus.raw_dir / "tokenizer"
        if config.dataset.prepared_train_manifest:
            self.config.rows.make().verify()
            config.dataset.prepared_train_manifest = (
                self.config.rows.working_dir / "train/PREPARED_MANIFEST.json"
            )
            config.dataset.prepared_eval_manifest = (
                self.config.rows.working_dir / "eval/PACKED_EVAL_MANIFEST.json"
            )
            if config.dataset.reference_evaluation is not None:
                config.dataset.reference_evaluation.path = (
                    self.config.working_dir / "reference-eval/unigram.npz"
                )
        config.tracker = TrackerList.Config()
        config.tracker.trackers = {"metrics": FileTracker.Config()}
        return config


def build_reference_eval(
    *,
    reference_dir: Path,
    unigram_path: Path,
    output: Path,
) -> None:
    """Build byte-matched replays from reference rows and fitted tokenizers.

    Args:
      reference_dir: Packed reference arrays, manifest, and tokenizer.pkl.
      unigram_path: Existing fitted tokenizer JSON, never retrained here.
      output: New output directory for bpe.npz and unigram.npz.

    """
    manifest = read_mapping(reference_dir / "PACKED_EVAL_MANIFEST.json")
    pickled = (reference_dir / "tokenizer.pkl").read_bytes()
    reference = cast(object, pickle.loads(pickled))  # noqa: S301 -- The artifact is a trusted tokenizer file created by this pipeline.
    if not isinstance(reference, tiktoken.Encoding):
        raise TypeError("Reference pickle does not contain a tiktoken encoding.")
    inputs, targets = (
        cast(NDArray[np.int64], load(reference_dir / name, allow_pickle=False))
        for name in ("eval_x.npy", "eval_y.npy")
    )
    unigram = tokenizers.Tokenizer.from_file(str(unigram_path))
    destination = validated_output_path(output, protected=[reference_dir, unigram_path])
    destination.mkdir(parents=True, exist_ok=False)
    for name, tokenizer in (("bpe", None), ("unigram", unigram)):
        arrays = prepare_reference_rows(
            inputs,
            targets=targets,
            reference=reference,
            tokenizer=tokenizer,
            batch_size=IntCodec.coerce(manifest["eval_batch_size"], default=None),
        )
        with (destination / f"{name}.npz").open("xb") as stream:
            savez(stream, allow_pickle=False, **arrays)


def encode_fragment(raw: bytes, *, tokenizer: tokenizers.Tokenizer) -> list[int]:
    """Encode a fragment, preserving a possibly incomplete UTF-8 suffix.

    Valid text uses the unchanged tokenizer. An incomplete trailing character has
    no string encoding; tokenize only its raw ByteLevel suffix as a separate piece
    group. Never repair, discard, or extend the reference-selected bytes.

    Args:
      raw: Selected bytes, possibly ending inside a UTF-8 character.
      tokenizer: Frozen ordinary ByteLevel Unigram tokenizer.

    Returns:
      ids: Ordinary tokens reconstructing exactly the selected bytes.

    """
    suffix = b""
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data" or error.end != len(raw):
            raise ValueError(
                "Reference fragment has non-terminal invalid UTF-8."
            ) from error
        text = raw[: error.start].decode("utf-8", errors="strict")
        suffix = raw[error.start :]
    ids: list[int] = tokenizer.encode(text, add_special_tokens=False).ids
    alphabet = byte_alphabet()
    if suffix:
        ids.extend(
            token.id
            for token in cast(_TokenizerModel, tokenizer.model).tokenize(
                "".join(alphabet[value] for value in suffix)
            )
        )
    inverse = {char: value for value, char in alphabet.items()}
    pieces = [tokenizer.id_to_token(token) for token in ids]
    if any(piece is None for piece in pieces):
        raise ValueError("Unigram emitted an unknown token.")
    decoded = bytes(
        inverse[char] for piece in pieces if piece is not None for char in piece
    )
    if decoded != raw:
        raise ValueError("Unigram replay changed the reference-selected bytes.")
    return ids


def prepare_reference_rows(
    inputs: NDArray[np.int64],
    *,
    targets: NDArray[np.int64],
    reference: tiktoken.Encoding,
    tokenizer: tokenizers.Tokenizer | None,
    batch_size: int,
) -> dict[str, ndarray]:
    """Build reference or Unigram rows with identical source-byte accounting.

    Args:
      inputs: Original reference input rows, including each leading BOS.
      targets: Original shifted target rows, including cropped final tokens.
      reference: Frozen BPE tokenizer used to select the rows.
      tokenizer: Frozen Unigram tokenizer; None preserves the reference IDs.
      batch_size: Reference batch size, retained for BPE reduction parity.

    Returns:
      arrays: Saveable replay archive with per-window byte accounting.

    """
    bos = reference.encode_single_token("<|reserved_0|>")
    if (
        inputs.shape != targets.shape
        or inputs.ndim != 2
        or len(inputs) == 0
        or not array_equal(inputs[:, 1:], targets[:, :-1])
        or np.any(cast(NDArray[np.bool_], inputs[:, 0] != bos))
        or batch_size <= 0
    ):
        raise ValueError("Reference rows are not BOS-aligned shifted targets.")
    if tokenizer is not None and (
        cast(object, tokenizer.normalizer) is not None
        or cast(object, tokenizer.post_processor) is not None
        or tokenizer.get_vocab_size()
        != tokenizer.get_vocab_size(with_added_tokens=False)
    ):
        raise ValueError("Unigram replay requires an unnormalized ordinary tokenizer.")
    pieces = [reference.decode_single_token_bytes(i) for i in range(bos)]
    historical: NDArray[np.int64] = array(
        [len(reference.decode([i]).encode()) for i in range(bos)]
        + [0] * (reference.n_vocab - bos),
        dtype=int64,
    )
    output_bos = int(tokenizer.get_vocab_size()) if tokenizer is not None else bos
    output_vocab = output_bos + 16 if tokenizer is not None else reference.n_vocab
    width = int(cast(int, inputs.shape[1]))
    output: list[tuple[ndarray, ndarray, ndarray]] = []
    reference_counts: list[int] = []
    literal_counts: list[int] = []
    original_rows: list[int] = []
    fragment_lengths: list[int] = []
    incomplete_suffixes = 0
    for row_index, row in enumerate(cast(Iterator[NDArray[np.int64]], targets)):
        sequence: list[int] = []
        literal_count = 0
        start = 0
        ends: list[int] = [
            *cast(
                list[int],
                cast(
                    NDArray[np.int64],
                    flatnonzero(cast(NDArray[np.bool_], row == bos)),
                ).tolist(),
            ),
            len(row),
        ]
        for end in ends:
            raw = b"".join(
                pieces[int(token)] for token in cast(list[int], row[start:end].tolist())
            )
            fragment_lengths.append(len(raw))
            literal_count += len(raw)
            try:
                raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                incomplete_suffixes += 1
            if tokenizer is not None:
                sequence.extend(
                    [output_bos, *encode_fragment(raw, tokenizer=tokenizer)]
                )
            start = int(end) + 1
        if tokenizer is None:
            output.append((inputs[row_index], row, historical[row] > 0))
        else:
            if len(sequence) > width + 1:
                raise ValueError(
                    f"Reference row {row_index} exceeds the model context: "
                    f"{len(sequence) - 1} targets > {width}."
                )
            output.append(_pad_row(sequence, bos=output_bos, width=width))
        reference_counts.append(int(historical[row].sum()))
        literal_counts.append(literal_count)
        original_rows.append(row_index)
    return {
        "inputs": stack([row[0] for row in output]).astype(int64),
        "targets": stack([row[1] for row in output]).astype(int64),
        "score_mask": stack([row[2] for row in output]),
        "reference_bytes": array(reference_counts, dtype=int64),
        "literal_bytes": array(literal_counts, dtype=int64),
        "reference_rows": array(original_rows, dtype=int64),
        "fragment_lengths": array(fragment_lengths, dtype=int64),
        "batch_size": array(batch_size),
        "vocab_size": array(output_vocab),
        "bos_token_id": array(output_bos),
        "incomplete_utf8_suffixes": array(incomplete_suffixes),
        "protocol": array("karpathy-reference-bytes-v1"),
        "token_bytes": historical
        if tokenizer is None
        else array(
            [
                len(tokenizer.decode([i], skip_special_tokens=False).encode())
                for i in range(output_bos)
            ]
            + [0] * 16,
            dtype=int64,
        ),
    }


def _pad_row(
    sequence: list[int],
    *,
    bos: int,
    width: int,
) -> tuple[ndarray, ndarray, ndarray]:
    """Pad a complete reference row, excluding padding and document markers."""
    window = array(sequence, dtype=int64)
    inputs = full(width, bos, dtype=int64)
    targets = inputs.copy()
    inputs[: len(window) - 1] = window[:-1]
    targets[: len(window) - 1] = window[1:]
    mask: NDArray[np.bool_] = cast(
        NDArray[np.bool_], (arange(width) < len(window) - 1) & (targets != bos)
    )
    return inputs, targets, mask


def donor_unigram16k() -> Preparation.Config:
    """Build the pinned donor-corpus and Unigram preparation recipe.

    Returns:
      config: Source selection, tokenizer fitting, and packed-row geometry.

    """
    config = Preparation.Config()
    config.corpus.donor_source_ids = (
        "12:48743",
        "14:4698",
        "14:67018",
        "13:63092",
        "12:57421",
        "12:49021",
        "14:73652",
        "14:12791",
        "14:30212",
        "13:46681",
        "13:64668",
        "12:70133",
        "12:5769",
        "12:60475",
        "12:76162",
        "14:23477",
        "14:56575",
        "14:56210",
        "13:10181",
        "13:78411",
        "12:43956",
        "12:68321",
        "12:80480",
        "12:74662",
        "14:74201",
        "14:77566",
        "13:50594",
        "13:25008",
        "12:47286",
        "12:51268",
        "12:12238",
        "12:4463",
        "14:54798",
        "14:4782",
        "13:79636",
        "13:69586",
        "12:15212",
        "12:17246",
        "14:69535",
        "14:18221",
        "13:63793",
        "13:72043",
        "12:35374",
        "12:59896",
        "14:60975",
        "14:22796",
        "13:33206",
        "13:79612",
        "13:45517",
        "12:36797",
        "12:31715",
        "13:59142",
        "13:77960",
        "12:11784",
        "12:55314",
        "14:67363",
        "14:58022",
        "14:16330",
        "13:4021",
        "13:58187",
        "12:78416",
        "12:4524",
        "14:59645",
        "14:22642",
        "13:47812",
        "13:61019",
        "12:41521",
        "12:75893",
        "14:63140",
        "14:51140",
        "13:69224",
        "13:10290",
        "12:61876",
        "12:4095",
        "12:24267",
        "14:22238",
        "14:65978",
        "13:71257",
        "13:84199",
        "12:34475",
        "12:82465",
        "14:17174",
        "14:10890",
        "14:59521",
        "13:62698",
        "13:60867",
        "12:18159",
        "12:25210",
        "14:63322",
        "14:48614",
        "13:60183",
        "13:60472",
        "12:41609",
        "12:60826",
        "14:8193",
        "14:44770",
        "14:21876",
        "13:34910",
        "12:74093",
        "12:50495",
        "12:13918",
        "13:17350",
        "13:1890",
        "12:59635",
        "12:82734",
        "14:43733",
        "14:39063",
        "14:62320",
        "13:70409",
        "13:73836",
        "13:75389",
        "13:38879",
        "12:43452",
        "12:14153",
        "14:52498",
        "14:64055",
        "14:4059",
        "13:75577",
        "13:37682",
        "13:75711",
        "12:51267",
        "12:43387",
        "12:44039",
        "14:54991",
        "14:61975",
        "13:63012",
        "13:51408",
        "13:65390",
        "13:57779",
        "12:71078",
        "12:78320",
        "14:37809",
        "14:30767",
        "13:14661",
        "13:69411",
        "12:20414",
        "12:79710",
        "14:33217",
        "14:53056",
        "13:9233",
        "13:77374",
        "12:2928",
        "12:77597",
        "14:29046",
        "14:27463",
        "14:5598",
        "13:77035",
        "13:60223",
        "12:65539",
        "14:3727",
        "13:78187",
        "13:48459",
        "12:69902",
        "12:11860",
        "14:15764",
        "14:34957",
        "14:60118",
        "13:56018",
        "13:42317",
        "12:10702",
        "12:7026",
        "14:79330",
        "14:82632",
        "13:28210",
        "13:41634",
        "12:38938",
        "12:41671",
        "14:40346",
        "14:58218",
        "13:3109",
        "13:29078",
        "12:77166",
        "12:40530",
        "12:79486",
        "14:5819",
        "14:34778",
        "13:51958",
        "13:50984",
        "12:3642",
        "12:17814",
        "14:73837",
        "14:4389",
        "14:38322",
        "13:70223",
        "13:49203",
        "12:36143",
        "12:61769",
        "14:1070",
        "14:50986",
        "13:30981",
        "13:62933",
        "12:23357",
        "12:17996",
        "12:51369",
        "14:17146",
        "14:81013",
        "13:47884",
        "13:33650",
        "12:28032",
        "14:77367",
        "13:10243",
        "13:52182",
        "12:7346",
        "12:12162",
        "12:63994",
        "14:28802",
        "14:1555",
        "13:15452",
        "13:1506",
        "13:48853",
        "13:12947",
        "12:21021",
        "12:540",
        "12:61793",
        "14:52843",
        "14:57311",
        "13:79210",
        "13:59953",
        "13:10996",
        "13:73790",
        "12:41186",
        "12:6395",
        "14:66290",
        "14:36834",
        "13:32031",
        "13:2915",
        "13:25107",
        "13:22763",
        "12:55121",
        "12:52204",
        "14:38396",
        "14:16257",
        "13:7438",
        "13:24013",
        "12:83014",
        "12:42924",
        "14:84499",
        "14:82133",
        "13:12477",
        "13:7006",
        "12:75375",
        "12:290",
        "12:31637",
        "14:40025",
        "14:17459",
        "13:83817",
        "13:15496",
        "14:33817",
        "14:81236",
        "13:9480",
        "13:37099",
        "12:29089",
        "12:42740",
        "12:83617",
        "14:15998",
        "14:25139",
        "13:70950",
        "13:6451",
        "12:43309",
        "12:68494",
        "14:32450",
        "14:32534",
        "13:47690",
        "13:84769",
        "12:65199",
        "12:21443",
        "14:59708",
        "14:732",
        "13:83463",
        "13:70850",
        "13:61034",
        "12:18517",
        "12:78962",
        "14:10261",
        "14:453",
        "13:44845",
        "13:70131",
        "12:74624",
        "12:59550",
        "12:33317",
        "14:81092",
        "14:15395",
        "13:10170",
        "13:45284",
        "12:28269",
        "12:17402",
        "14:51515",
        "14:40048",
        "13:73733",
        "13:5448",
        "12:14444",
        "12:43731",
        "12:24291",
        "14:62135",
        "13:18719",
        "13:53114",
        "13:48684",
        "14:32376",
        "14:22609",
        "13:21664",
        "13:3853",
        "13:11323",
        "12:43902",
        "12:24958",
        "14:78024",
        "14:24897",
        "14:82784",
        "14:14213",
        "13:27230",
        "13:63727",
        "12:79599",
        "12:78319",
        "12:65663",
        "14:37239",
        "14:51533",
        "14:80760",
        "13:78705",
        "13:67004",
        "13:28354",
        "12:35990",
        "12:4914",
        "14:42696",
        "14:50699",
        "14:7861",
        "14:23779",
        "13:75739",
        "13:1499",
        "12:65553",
        "12:6951",
        "14:40462",
        "14:51136",
        "13:44516",
        "13:38243",
        "12:71753",
        "12:63601",
        "14:29104",
        "14:42830",
        "13:29607",
        "13:29740",
        "12:71008",
        "12:31283",
        "12:36533",
        "14:47025",
        "14:56772",
        "13:43797",
        "12:52372",
        "14:8452",
        "14:54067",
        "13:82897",
        "13:15286",
        "12:58169",
        "12:39316",
        "12:24716",
        "14:19205",
        "14:61224",
        "13:53144",
        "13:30158",
        "12:76523",
        "12:48985",
        "14:13733",
        "14:9348",
        "13:64693",
        "13:10394",
        "12:29828",
        "12:69272",
        "14:3120",
        "14:38599",
        "14:21281",
        "13:31222",
        "13:58400",
        "12:43888",
        "12:25805",
        "14:5791",
        "14:33384",
        "13:14321",
        "13:78388",
        "12:30573",
        "12:51744",
        "12:53382",
        "14:44192",
        "14:74055",
        "13:54744",
        "13:48437",
        "12:51125",
        "12:46238",
        "14:11237",
        "14:71333",
        "13:30745",
        "13:73543",
        "13:63274",
        "12:12650",
        "12:73544",
        "14:39697",
        "14:49415",
        "13:16060",
        "12:23327",
        "14:23625",
        "14:72812",
        "13:65985",
        "13:50133",
        "13:84380",
        "12:67992",
        "12:36485",
        "12:12135",
        "14:25666",
        "14:80585",
        "14:5265",
        "13:77587",
        "13:78979",
        "13:14412",
        "12:1008",
        "12:12225",
        "12:45030",
        "14:10876",
        "14:72759",
        "14:38670",
        "13:39765",
        "13:47393",
        "12:35657",
        "12:8702",
        "12:69239",
        "14:29738",
        "14:67923",
        "14:24347",
        "13:25813",
        "13:33719",
        "12:51667",
        "12:27685",
        "14:65562",
        "14:63743",
        "13:32595",
        "13:43894",
        "12:82350",
        "12:77335",
        "14:50084",
        "14:60416",
        "13:76376",
        "13:53472",
        "13:66044",
        "12:27845",
        "12:44545",
        "14:57455",
        "13:58629",
        "12:54489",
        "12:50057",
        "14:17667",
        "14:79868",
        "13:39928",
        "13:23890",
        "13:60651",
        "12:22868",
        "12:48614",
        "14:3009",
        "14:59848",
        "13:3878",
        "13:45851",
        "12:61560",
        "12:7774",
        "14:14370",
        "14:45732",
        "13:54692",
        "13:76031",
        "12:83477",
        "12:70914",
        "14:80490",
        "14:929",
        "14:39582",
        "13:46706",
        "13:27944",
        "12:78795",
        "12:31048",
        "14:67410",
        "14:50935",
        "13:69487",
        "13:59533",
        "13:39101",
        "12:68470",
        "12:14358",
        "14:80828",
        "14:75511",
        "13:9260",
        "13:75771",
        "12:78733",
        "12:56254",
        "14:4391",
        "14:4811",
        "14:43701",
        "13:35159",
        "13:56796",
        "12:37043",
    )
    config.experiment_corpora = {"exp019": "donor-original"}
    return config


def launch_training(
    config: NgramTrainLoop.Config, *, save_checkpoint: bool = False
) -> None:
    """Record the prepared config and launch it through the canonical Priml process.

    Args:
      config: Bound experiment with a new output directory.
      save_checkpoint: Retain the final training state without periodic saves.

    """
    config = config.copy_tree()
    if save_checkpoint:
        config.checkpointing = Checkpointer.Config()
        config.checkpointing.save_every = sys.maxsize
        config.checkpointing.resume = False
    directory = validated_output_path(config.working_dir)
    directory.mkdir(parents=True, exist_ok=False)
    recipe = directory / "prepared_experiment.py"
    recipe.write_text(
        "from priml.baselines.nanochat.experiments import NgramTrainLoop\n\n"
        "def experiment() -> NgramTrainLoop.Config:\n"
        '    """Run the prepared NanoChat experiment."""\n'
        f"    return NgramTrainLoop.Config.deserialize({config.serialize()!r})\n"
    )
    (directory / "prepared_config.txt").write_text(
        config.pformat(hide_default_values=False) + "\n"
    )
    subprocess.run(
        [sys.executable, "-m", "priml", "prepared_experiment.experiment"],
        cwd=directory,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        check=True,
    )


def main() -> int:
    """Run an explicit preparation factory and return the process exit code.

    Returns:
      exit_code: Zero after the selected operation succeeds.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    args = cast(_Arguments, parser.parse_args())
    module, _, name = args.factory.rpartition(".")
    factory = cast(Callable[[], object], getattr(importlib.import_module(module), name))
    config = factory()
    if not isinstance(config, Preparation.Config):
        raise TypeError("The factory must return Preparation.Config.")
    config.working_dir = args.directory.expanduser().absolute()
    if args.print_config:
        config.pprint(hide_default_values=False)
        return 0
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if any(
        value is not None
        for value in (
            args.num_train_shards,
            args.vocab_size,
            args.tokenizer_train_chars,
            args.tokenizer_doc_cap,
        )
    ):
        prepare(
            args.directory,
            num_train_shards=args.num_train_shards
            if args.num_train_shards is not None
            else 7,
            vocab_size=args.vocab_size if args.vocab_size is not None else 8_192,
            tokenizer_train_chars=args.tokenizer_train_chars
            if args.tokenizer_train_chars is not None
            else 2_000_000_000,
            tokenizer_doc_cap=args.tokenizer_doc_cap
            if args.tokenizer_doc_cap is not None
            else 10_000,
        )
        return 0
    preparation = config.make()
    if args.stage == "dump":
        preparation.dump(args.output)
    elif args.stage == "train":
        training = preparation.training_config(
            args.experiment, run_directory=args.run_directory, seed=args.seed
        )
        training.pprint(hide_default_values=False)
        launch_training(training, save_checkpoint=args.save_checkpoint)
    else:
        preparation.run(args.stage)
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the preparation factory, stage, and output paths."""
    parser.add_argument(
        "factory",
        nargs="?",
        default="priml.baselines.nanochat.scripts.prepare_data.donor_unigram16k",
    )
    parser.add_argument(
        "--directory", type=Path, default=Path("/opt/scratch/datasets/nanochat")
    )
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "fetch",
            "baseline",
            "reference",
            "corpus",
            "sample",
            "tokenizer",
            "rows",
            "verify",
            "dump",
            "train",
        ),
        default="all",
    )
    parser.add_argument("--num-train-shards", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--tokenizer-train-chars", type=int, default=None)
    parser.add_argument("--tokenizer-doc-cap", type=int, default=None)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--experiment", default="exp022")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("/opt/scratch/runs/nanochat-reproduction"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/opt/scratch/artifacts/nanochat/prepared-inputs.tar"),
    )


class _Arguments(Protocol):
    factory: str
    directory: Path
    stage: str
    num_train_shards: int | None
    vocab_size: int | None
    tokenizer_train_chars: int | None
    tokenizer_doc_cap: int | None
    print_config: bool
    experiment: str
    seed: int
    save_checkpoint: bool
    run_directory: Path
    output: Path


if __name__ == "__main__":
    # Factories must share the importable Config identity rather than __main__'s.
    raise SystemExit(
        cast(
            Callable[[], int],
            importlib.import_module(
                "priml.baselines.nanochat.scripts.prepare_data"
            ).main,
        )()
    )
# vim: ft=python
