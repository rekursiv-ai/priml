"""Golden-file assertions and compact tensor records for Priml tests.

A tensor golden is a flat name-to-tensor dict saved with plain ``torch.save``.
Its size is dominated by per-storage archive overhead, not by the bytes of its
small tensors, so :func:`write_tensors` packs every tensor of one dtype into
one storage and :func:`put_steps` stacks per-step values under one key.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, Final, cast

import math
import os
import re
import zlib

from torch import Tensor

import torch

from priml.lib.custom_json import DictCodec


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping

    import pytest


_INDEX: Final = "index"


def assert_text_golden(
    request: pytest.FixtureRequest,
    *,
    test_file: str,
    name: str,
    rendered: str,
) -> None:
    """Assert rendered text matches its test-local golden.

    Args:
      request: Active pytest request carrying ``--golden-overwrite``.
      test_file: ``__file__`` of the owning test module.
      name: Golden filename without its extension.
      rendered: Snapshot text without a trailing newline.

    """
    golden = Path(test_file).resolve().parent / "testdata" / f"{name}.txt"
    missing = not golden.exists()
    if missing or request.config.getoption("--golden-overwrite", default=False):
        golden.parent.mkdir(parents=True, exist_ok=True)
        _ = golden.write_text(rendered + "\n", encoding="utf-8")
    if missing:
        raise AssertionError(
            f"Missing golden regenerated at {golden}; inspect it, then rerun the test.",
        )
    if golden.read_text(encoding="utf-8") != rendered + "\n":
        raise AssertionError(
            f"{name} changed; read the diff, then rerun with --golden-overwrite "
            "if the change is intended.",
        )


# A pickled dict of tensors costs ~100 bytes per entry however small the tensor, so a
# record of hundreds of scalars is mostly pickle. The file instead holds one flat
# tensor per dtype plus a zlib-compressed index of ``name<TAB>dtype<TAB>shape`` lines:
# a plain ``torch.save`` of a few tensors, whatever the key count. Only the index is
# compressed -- its key names repeat -- so ``torch.load`` still reads every file.
def write_tensors(path: Path, record: Mapping[str, Tensor]) -> None:
    """Save a flat name-to-tensor record as one flat tensor per dtype and an index.

    Args:
      path: Destination ``.pt``.
      record: Flat name-to-tensor record.

    """
    torch.save(pack(record), path)


def read_tensors(path: Path) -> dict[str, Tensor]:
    """Load a golden written by :func:`write_tensors`.

    Args:
      path: Source ``.pt``.

    Returns:
      record: Flat name-to-tensor record, in the order it was written.

    Raises:
      TypeError: The file is not an index and its per-dtype tensors.

    """
    raw = cast(object, torch.load(path, weights_only=True))
    try:
        return unpack(raw)
    except TypeError as error:
        raise TypeError(
            f"{path} is not a tensor golden written by write_tensors.",
        ) from error


def pack(record: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return ``record`` as one flat tensor per dtype plus a zlib index.

    Args:
      record: Flat name-to-tensor record.

    Returns:
      packed: Per-dtype tensors and the ``index`` bytes; :func:`unpack` inverts it.

    Raises:
      ValueError: A name holds a tab or newline, which the index reserves.

    """
    lines: list[str] = []
    parts: dict[str, list[Tensor]] = {}
    for key, value in record.items():
        if "\t" in key or "\n" in key:
            raise ValueError(f"Golden key {key!r} holds a tab or newline.")
        dtype = str(value.dtype).removeprefix("torch.")
        shape = ",".join(str(size) for size in value.shape)
        lines.append(f"{key}\t{dtype}\t{shape}")
        parts.setdefault(dtype, []).append(value.detach().reshape(-1).cpu())
    index = bytearray(zlib.compress("\n".join(lines).encode(), level=9))
    payload = {name: torch.cat(values) for name, values in parts.items()}
    payload[_INDEX] = torch.frombuffer(index, dtype=torch.uint8).clone()
    return payload


def unpack(packed: object) -> dict[str, Tensor]:
    """Invert :func:`pack`.

    Args:
      packed: A value :func:`pack` returned, typically fresh from ``torch.load``.

    Returns:
      record: Flat name-to-tensor record, in the order it was packed.

    Raises:
      TypeError: ``packed`` is not an index and its per-dtype tensors.

    """
    payload = DictCodec.coerce(packed, Tensor, default=None)
    if _INDEX not in payload or len(payload) != len(DictCodec.coerce(packed)):
        raise TypeError("Not a record written by pack.")
    index = zlib.decompress(payload.pop(_INDEX).numpy().tobytes()).decode()
    offsets = dict.fromkeys(payload, 0)
    record: dict[str, Tensor] = {}
    for line in index.split("\n") if index else []:
        key, dtype, shape = line.split("\t")
        sizes = [int(size) for size in shape.split(",")] if shape else []
        start = offsets[dtype]
        offsets[dtype] = start + math.prod(sizes)
        record[key] = payload[dtype][start : offsets[dtype]].reshape(sizes)
    return record


def mismatches(
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
) -> list[str]:
    """Return every key whose presence, dtype, shape, or bits differ.

    Args:
      expected: Reference record.
      actual: Candidate record.

    Returns:
      report: One line per mismatch, all of them.

    """
    report = [f"missing {k}" for k in sorted(expected.keys() - actual.keys())]
    report += [f"unexpected {k}" for k in sorted(actual.keys() - expected.keys())]
    for key in sorted(expected.keys() & actual.keys()):
        want, got = expected[key], actual[key]
        if want.dtype != got.dtype or want.shape != got.shape:
            report.append(
                f"{key}: {got.dtype}{list(got.shape)} vs {want.dtype}{list(want.shape)}",
            )
        elif not torch.equal(want, got):
            report.append(f"{key}: {(want != got).sum().item()}/{want.numel()} differ")
    return report


@contextmanager
def expect_golden_mismatch(match: str | Pattern[str]) -> Generator[None]:
    """Require a matching golden assertion failure without regeneration.

    Args:
      match: Regular expression matched against the assertion message.

    Yields:
      nothing: No value; used as a context manager around a golden assertion.

    """
    prior = os.environ.pop("BFB_REGENERATE", None)
    mismatch = False
    try:
        try:
            yield
        except AssertionError as error:
            if not re.search(match, str(error)):
                raise
            mismatch = True
    finally:
        if prior is not None:
            os.environ["BFB_REGENERATE"] = prior
    if not mismatch:
        raise AssertionError("Expected a golden mismatch, but the comparison passed.")


def assert_tensor_golden(path: Path, record: Mapping[str, Tensor]) -> None:
    """Require ``record`` to equal the tensor golden at ``path``.

    A missing golden is written and the call fails, forcing review before the
    next run accepts it; ``BFB_REGENERATE=1`` rewrites an existing one.

    Args:
      path: The ``.pt`` golden.
      record: Flat name-to-tensor record.

    Raises:
      AssertionError: The golden was missing, or ``record`` differs from it.

    """
    missing = not path.is_file()
    if missing or os.environ.get("BFB_REGENERATE", "0") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        write_tensors(path, record)
    if missing:
        raise AssertionError(f"Missing golden minted at {path}; inspect it, rerun.")
    report = mismatches(read_tensors(path), record)
    if report:
        raise AssertionError(f"{len(report)} mismatches:\n" + "\n".join(report))


def stored(value: Tensor) -> Tensor:
    """Return a detached copy of ``value``, small whole numbers as bytes.

    Args:
      value: A recorded tensor.

    Returns:
      copy: The same values; a non-scalar int32, int64, or float32 tensor of
        whole numbers in [0, 256) -- token grids, counters -- narrowed to uint8,
        which holds them exactly. Anything else, including a float ``-0.0``,
        keeps its dtype, so no bit is lost, and a value that stops being whole
        changes the stored dtype, which a comparison reports.

    """
    copy = value.detach().clone()
    whole = copy.dtype in {torch.int32, torch.int64} or (
        copy.dtype == torch.float32
        and bool((copy == copy.round()).all())
        and not bool(copy.signbit().any())
    )
    narrow = copy.numel() > 1 and whole and bool(((copy >= 0) & (copy < 256)).all())
    return copy.to(torch.uint8) if narrow else copy


def put_steps(
    out: dict[str, Tensor],
    prefix: str,
    records: list[dict[str, Tensor]],
) -> None:
    """Store per-step records stacked on a leading step axis.

    Args:
      out: Record to extend.
      prefix: Key prefix; a uniform quantity lands at ``prefix/<key>``.
      records: One name-to-tensor dict per step.

    A quantity missing from some step, or changing shape or dtype, is stored
    per step under ``prefix/<step>/<key>`` (1-based) instead.

    """
    for key in sorted({key for record in records for key in record}):
        values = [record[key] for record in records if key in record]
        uniform = len({(value.shape, value.dtype) for value in values}) == 1
        if len(values) == len(records) and uniform:
            out[f"{prefix}/{key}"] = stored(torch.stack(values))
            continue
        for index, record in enumerate(records, start=1):
            if key in record:
                out[f"{prefix}/{index}/{key}"] = stored(record[key])


def rng_fingerprint() -> Tensor:
    """Return the global generator's next 8 draws without advancing it.

    The raw state is 5 KB of Mersenne Twister words; the next draws pin the
    same position for a hundredth of the bytes.

    Returns:
      draws: The next 8 ``randint`` draws, as int64.

    """
    generator = torch.Generator()
    generator.set_state(torch.get_rng_state())
    return torch.randint(0, 2**31 - 1, (8,), generator=generator)


# A trained weight is the product of every step before it, so its bits already pin the
# whole trajectory; a few elements per tensor catch any divergence without storing the
# model.
def leading(state: Mapping[str, Tensor], *, count: int = 4) -> dict[str, Tensor]:
    """Return detached copies of the first ``count`` elements of every tensor."""
    return {k: v.detach().flatten()[:count].clone() for k, v in state.items()}


# Evenly spaced rather than leading: an output's regions come from different code
# (a prefix, the grid, a padded tail), and the first elements see only one of them.
def spread(value: Tensor, *, count: int = 128) -> Tensor:
    """Return ``value`` flattened, or ``count`` evenly spaced samples of it.

    Args:
      value: A recorded tensor.
      count: Most elements kept.

    Returns:
      sample: Every element when there are at most ``count``, else elements
        ``0, n/count, 2n/count, ...`` of the flattened tensor.

    """
    flat = value.detach().reshape(-1)
    if flat.numel() <= count:
        return flat.clone()
    index = torch.arange(count, device=flat.device) * flat.numel() // count
    return flat[index].clone()


def heads(tensors: Iterable[Tensor], *, count: int = 4) -> Tensor:
    """Return the first ``count`` elements of each tensor, joined into one.

    One key instead of one per tensor keeps a golden's index short.

    Args:
      tensors: Tensors in a fixed order.
      count: Elements kept from each.

    Returns:
      joined: The kept elements in order, under ``torch.cat`` type promotion.
        Floats widen exactly; an integer joined with a float is converted,
        which is lossy past the float's mantissa, so join like dtypes.

    """
    parts = [t.detach().flatten()[:count] for t in tensors]
    return torch.cat(parts) if parts else torch.zeros(0)
