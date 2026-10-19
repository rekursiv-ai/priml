"""Golden-file assertions and compact tensor records for Priml tests.

A tensor golden is a flat name-to-tensor dict saved with plain ``torch.save``.
Its size is dominated by per-storage archive overhead, not by the bytes of its
small tensors, so :func:`write_tensors` packs every tensor of one dtype into
one storage and :func:`put_steps` stacks per-step values under one key.

Replay contexts observe native draws or pin explicit test streams. The broad
sampler stream and dropout-only stream retain distinct formulas and counters.
Production training uses native samplers.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from difflib import unified_diff
from functools import cache
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, Final, cast, override

import fcntl
import inspect
import math
import os
import re
import tempfile
import threading
import traceback
import zlib

from torch import Tensor
from torch._C import Tag
from torch.utils._python_dispatch import TorchDispatchMode
from treekle import from_plain

import torch

from priml.math.seed import get_rng_state
from priml.testing import regenerate


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Generator, Iterable, Mapping

    from configgle.fig import Maker
    from torch import nn
    from torch._ops import OpOverload

    from priml.math.seed import RngState


_INDEX: Final = "index"
_THIS: Final = Path(__file__).resolve()


class NativeDraws(TorchDispatchMode):
    """Observe complete native sampler outputs and pre-draw generator states."""

    def __init__(self) -> None:
        super().__init__()
        self.draws: list[tuple[str, RngState]] = []
        self.outputs: dict[str, Tensor] = {}
        self.generator_states: dict[int, tuple[torch.device, Tensor]] = {}
        """Explicit generator devices/states before draws, indexed from one."""

    @override
    def __torch_dispatch__(
        self,
        func: OpOverload[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        stochastic = _is_stochastic(func, args)
        if stochastic:
            self.draws.append((str(func), get_rng_state()))
            generator = (kwargs or {}).get("generator")
            if isinstance(generator, torch.Generator):
                self.generator_states[len(self.draws)] = (
                    generator.device,
                    generator.get_state().clone(),
                )
        output = func(*args, **(kwargs or {}))
        if stochastic:
            leaves = (
                (output,)
                if isinstance(output, Tensor)
                else cast(tuple[Tensor, ...], output)
            )
            for index, tensor in enumerate(leaves):
                self.outputs[f"draw/{len(self.draws)}/{index}"] = (
                    tensor.detach().cpu().clone()
                )
        return output


class PinnedRandom(TorchDispatchMode):
    """Pin ATen samplers while retaining init, dropout, and DataLoader algorithms."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: Counter[str] = Counter()
        """Legacy receipt keys; numerical offsets depend only on the total."""
        self.site_calls: Counter[str] = Counter()
        """Directory-preserving call-site counts for coverage assertions."""

    @override
    def __torch_dispatch__(
        self,
        func: OpOverload[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        kwargs = kwargs or {}
        if not _is_stochastic(func, args):
            return func(*args, **kwargs)
        kind = _primitive_kind(func)
        if kind == "dropout":
            raise AssertionError(f"Unpinned random primitive: {func}.")
        frame_info = None
        caller = None
        for frame, line in traceback.walk_stack(None):
            path = Path(frame.f_code.co_filename).resolve()
            if (
                caller is None
                and path != _THIS
                and not path.is_relative_to(Path(torch.__file__).resolve().parent)
            ):
                caller = (frame, line)
            if frame.f_code.co_name in (
                "reset_parameters",
                "forward",
                "__iter__",
                "_get_iterator",
            ):
                frame_info = (frame, line)
                break
        site = (
            "factory"
            if frame_info is None
            else f"{Path(frame_info[0].f_code.co_filename).name}:{frame_info[1]}"
        )
        self.calls[f"{func}/{site}"] += 1
        location = frame_info or caller
        if location is None:
            raise ValueError("Cannot identify the random primitive's caller.")
        site_frame, site_line = location
        site_path = Path(site_frame.f_code.co_filename).resolve()
        self.site_calls[f"{func}/{site_path}:{site_line}"] += 1
        offset = sum(self.calls.values())
        if kind == "randperm":
            count = cast(int, args[0])
            output = kwargs.get("out")
            dtype = (
                output.dtype
                if isinstance(output, Tensor)
                else kwargs.get("dtype", torch.int64)
            )
            device = (
                output.device
                if isinstance(output, Tensor)
                else kwargs.get("device", "cpu")
            )
            values = torch.arange(
                count,
                dtype=cast(torch.dtype, dtype),
                device=cast(torch.device, device),
            ).roll(offset % count if count else 0)
            if isinstance(output, Tensor):
                return output.resize_(count).copy_(values)
            return values
        tensor = args[0]
        assert isinstance(tensor, Tensor)
        if kind == "random_":
            low = cast(int, args[1]) if len(args) > 2 else 0
            high = args[2] if len(args) > 2 else (args[1] if len(args) > 1 else None)
            if high is None:
                if tensor.dtype == torch.bool:
                    high = 2
                elif tensor.is_floating_point():
                    high = int(2 / torch.finfo(tensor.dtype).eps) + 1
                else:
                    high = torch.iinfo(tensor.dtype).max + 1
            assert isinstance(high, int)
            if high <= low:
                raise ValueError(
                    "Integer draw upper bound must exceed its lower bound.",
                )
            return tensor.fill_(low + offset % (high - low))
        values = (
            (
                torch.arange(tensor.numel(), device=tensor.device).reshape(tensor.shape)
                + offset
            )
            % 97
        ).to(tensor.dtype) / 97
        if kind == "bernoulli_":
            probability = cast(float, args[1]) if len(args) > 1 else 0.5
            return tensor.copy_(values < probability)
        low = cast(float, args[1]) if len(args) > 1 else 0.0
        high = cast(float, args[2]) if len(args) > 2 else 1.0
        return tensor.copy_(values * (high - low) + low)


class PinnedDropout(TorchDispatchMode):
    """Supply fixed Bernoulli inputs while observing every stochastic call site.

    Args:
      model: Model whose module names identify dropout call sites.
      draw_offset: Number of prior draws in the dropout-only stream.

    """

    def __init__(self, model: nn.Module, *, draw_offset: int = 0) -> None:
        super().__init__()
        self.calls = 0
        self.draw_offset = draw_offset
        self.sites: list[str] = []
        self.names = {id(module): name for name, module in model.named_modules()}

    @override
    def __torch_dispatch__(
        self,
        func: OpOverload[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        if not _is_stochastic(func, args):
            return func(*args, **(kwargs or {}))
        if func is not torch.ops.aten.bernoulli_.float:
            raise ValueError(func)
        self.calls += 1
        frame = inspect.currentframe()
        site = "functional"
        while frame is not None:
            module = frame.f_locals.get("self")
            if id(module) in self.names:
                site = self.names[id(module)]
                break
            frame = frame.f_back
        del frame
        self.sites.append(site)
        output = args[0]
        assert isinstance(output, Tensor)
        # ATen omits the schema's default p=0.5 from positional arguments.
        probability = cast(float, args[1]) if len(args) > 1 else 0.5
        uniform = (
            (
                torch.arange(output.numel(), device=output.device)
                + 7 * (self.calls - 1 + self.draw_offset)
            )
            % 20
        ).reshape(output.shape) / 20
        return output.copy_((uniform < probability).to(dtype=output.dtype))


def assert_pprint_golden(
    *,
    test_file: str,
    name: str,
    config: Maker[object],
    normalize: Callable[[str], str] = str,
    width: int = 80,
) -> None:
    """Assert a config's full finalized pprint matches its test-local golden.

    A missing golden is written, then raises for inspection;
    ``--regenerate-golden`` rewrites an existing one.

    Args:
      test_file: ``__file__`` of the owning test module.
      name: Golden filename without its extension.
      config: Config to finalize and render with every field visible.
      normalize: Maps rendered text to its host-independent form.
      width: Maximum line width of the render.

    Raises:
      AssertionError: The golden was missing, or the render differs; the
        message holds the diff.

    """
    golden = Path(test_file).resolve().parent / "testdata" / f"{name}.txt"
    rendered = normalize(
        config.pformat(
            finalize=True,
            mask_memory_addresses=True,
            hide_default_values=False,
            width=width,
        )
        + "\n",
    )
    missing = not golden.exists()
    if missing or regenerate.golden():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(rendered.encode())
    if missing:
        raise AssertionError(
            f"Missing golden regenerated at {golden}; inspect it, then rerun the test.",
        )
    expected = golden.read_bytes().decode()
    if expected == rendered:
        return
    diff = "".join(
        unified_diff(
            expected.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=str(golden),
            tofile=f"{name} (rendered)",
        ),
    )
    raise AssertionError(
        f"{name} changed; rerun with --regenerate-golden if intended.\n{diff}",
    )


def assert_text_golden(*, test_file: str, name: str, rendered: str) -> None:
    """Assert rendered text matches its test-local golden.

    A missing golden is written, then raises for inspection;
    ``--regenerate-golden`` rewrites an existing one.

    Args:
      test_file: ``__file__`` of the owning test module.
      name: Golden filename without its extension.
      rendered: Snapshot text without a trailing newline.

    Raises:
      AssertionError: The golden was missing, or ``rendered`` differs from it.

    """
    golden = Path(test_file).resolve().parent / "testdata" / f"{name}.txt"
    missing = not golden.exists()
    if missing or regenerate.golden():
        golden.parent.mkdir(parents=True, exist_ok=True)
        _ = golden.write_text(rendered + "\n", encoding="utf-8")
    if missing:
        raise AssertionError(
            f"Missing golden regenerated at {golden}; inspect it, then rerun the test.",
        )
    if golden.read_text(encoding="utf-8") != rendered + "\n":
        raise AssertionError(
            f"{name} changed; read the diff, then rerun with --regenerate-golden "
            "if the change is intended.",
        )


# A pickled dict of tensors costs ~100 bytes per entry however small the tensor, so a
# record of hundreds of scalars is mostly pickle. The file instead holds one flat
# tensor per dtype plus a zlib-compressed index of ``name<TAB>dtype<TAB>shape`` lines:
# a plain ``torch.save`` of a few tensors, whatever the key count. Only the index is
# compressed -- its key names repeat -- so ``torch.load`` still reads every file.
def write_tensors(path: Path, record: Mapping[str, Tensor]) -> None:
    """Save a flat name-to-tensor record as one flat tensor per dtype and an index.

    The file is written to a uniquely named file beside ``path`` and renamed over
    it, so a failed write leaves the previous golden whole and concurrent writers
    never share a staged file.

    Args:
      path: Destination ``.pt``.
      record: Flat name-to-tensor record.

    """
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        staged = Path(handle.name)
    try:
        torch.save(pack(record), staged)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


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
    payload = from_plain(packed, dict[str, Tensor])
    if _INDEX not in payload:
        raise TypeError("Not a record written by pack.")
    try:
        index = zlib.decompress(payload.pop(_INDEX).numpy().tobytes()).decode()
    except (zlib.error, UnicodeDecodeError) as error:
        raise TypeError(
            "Not a record written by pack: invalid compressed index.",
        ) from error
    offsets = dict.fromkeys(payload, 0)
    record: dict[str, Tensor] = {}
    try:
        for line in index.split("\n") if index else []:
            key, dtype, shape = line.split("\t")
            sizes = [int(size) for size in shape.split(",")] if shape else []
            start = offsets[dtype]
            offsets[dtype] = start + math.prod(sizes)
            record[key] = payload[dtype][start : offsets[dtype]].reshape(sizes)
    except (ValueError, KeyError, RuntimeError) as error:
        raise TypeError(
            "Not a record written by pack: invalid tensor index.",
        ) from error
    return record


def mismatches(
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
    *,
    ulps: bool = False,
) -> list[str]:
    """Return every key whose presence, dtype, shape, or bits differ.

    Args:
      expected: Reference record.
      actual: Candidate record.
      ulps: Include ULP diagnostics for real floating tensors.

    Returns:
      report: One line per mismatch, all of them. Optional ULP distances never
        narrow the dtype. Signed zeros remain bit mismatches at zero ULP;
        NaNs report no numeric distance.

    """
    report = [f"missing {k}" for k in sorted(expected.keys() - actual.keys())]
    report += [f"unexpected {k}" for k in sorted(actual.keys() - expected.keys())]
    for key in sorted(expected.keys() & actual.keys()):
        want, got = expected[key], actual[key]
        if want.dtype != got.dtype or want.shape != got.shape:
            report.append(
                f"{key}: {got.dtype}{list(got.shape)} vs {want.dtype}{list(want.shape)}",
            )
        elif not tensor_bits_equal(want, got):
            different = (
                (_tensor_bytes(want) != _tensor_bytes(got))
                .reshape(
                    want.numel(),
                    want.element_size(),
                )
                .any(dim=1)
            )
            diagnostic = (
                f"; max_ulp={_max_ulp(want, got)}"
                if ulps and want.is_floating_point()
                else ""
            )
            report.append(
                f"{key}: {different.sum().item()}/{want.numel()} differ{diagnostic}",
            )
    return report


def tensor_bits_equal(a: Tensor, b: Tensor) -> bool:
    """Return whether two tensors have identical shapes, dtypes, and stored bits."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(_tensor_bytes(a), _tensor_bytes(b))


@contextmanager
def expect_golden_mismatch(match: str | Pattern[str]) -> Generator[None]:
    """Require a matching golden assertion failure without regeneration.

    Args:
      match: Regular expression matched against the assertion message.

    Yields:
      nothing: No value; used as a context manager around a golden assertion.

    """
    mismatch = False
    with regenerate.suppressed():
        try:
            yield
        except AssertionError as error:
            if not re.search(match, str(error)):
                raise
            mismatch = True
    if not mismatch:
        raise AssertionError("Expected a golden mismatch, but the comparison passed.")


def assert_tensor_golden(path: Path, record: Mapping[str, Tensor]) -> None:
    """Require ``record`` to equal the tensor golden at ``path``.

    A missing golden is written and the call fails, forcing review before the
    next run accepts it; ``--regenerate-b4b`` rewrites an existing one.

    Args:
      path: The ``.pt`` golden.
      record: Flat name-to-tensor record.

    Raises:
      AssertionError: The golden was missing, or ``record`` differs from it.

    """
    missing = not path.is_file()
    if missing or regenerate.b4b():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_tensors(path, record)
    if missing:
        raise AssertionError(f"Missing golden minted at {path}; inspect it, rerun.")
    report = mismatches(read_tensors(path), record)
    if report:
        raise AssertionError(f"{len(report)} mismatches:\n" + "\n".join(report))


def assert_tensor_golden_part(
    path: Path,
    prefix: str,
    record: Mapping[str, Tensor],
) -> None:
    """Require ``record`` to equal the keys under ``prefix`` of the golden at ``path``.

    One golden file holds several tests' records, each under its own prefix, so
    each test replays only its part. A missing part is written and the call
    fails; ``--regenerate-b4b`` rewrites the part and keeps the others. Every
    read and write holds :func:`locked_directory` on the golden's directory, so
    concurrent tests of one golden never drop each other's parts.

    Args:
      path: The ``.pt`` golden.
      prefix: The part's key prefix, ending in ``/``.
      record: The part; every key starts with ``prefix``.

    Raises:
      ValueError: A key of ``record`` lies outside ``prefix``.
      AssertionError: The part was missing, or ``record`` differs from it.

    """
    outside = [key for key in record if not key.startswith(prefix)]
    if outside:
        raise ValueError(f"Golden part {prefix} holds keys outside it: {outside}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked_directory(path.parent):
        current = read_tensors(path) if path.is_file() else {}
        part = {k: v for k, v in current.items() if k.startswith(prefix)}
        missing = not part
        if missing or regenerate.b4b():
            kept = {k: v for k, v in current.items() if not k.startswith(prefix)}
            write_tensors(path, kept | dict(record))
            part = dict(record)
    if missing:
        raise AssertionError(f"Missing golden part {prefix} minted in {path}; rerun.")
    report = mismatches(part, record)
    if report:
        raise AssertionError(f"{len(report)} mismatches:\n" + "\n".join(report))


# ``write_tensors`` renames a new file over a golden, so a lock on the golden would
# guard the replaced inode; a lock file would be owned by whoever made it first,
# locking out every other user of a shared host. The directory is neither.
@contextmanager
def locked_directory(directory: Path) -> Generator[None]:
    """Hold an exclusive lock on ``directory`` for the block, across processes.

    The holder may take it again inside the block: a regeneration that holds a
    golden directory's lock mints parts into it under that same lock.

    Args:
      directory: An existing directory, typically a golden's; it is not created.

    Yields:
      nothing: Control, with the lock held.

    """
    key = directory.resolve()
    held = _held_directories.paths
    if key in held:
        yield
        return
    descriptor = os.open(key, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        held.add(key)
        try:
            yield
        finally:
            held.discard(key)
    finally:
        os.close(descriptor)


class _HeldDirectories(threading.local):
    """The directories this thread holds :func:`locked_directory` on."""

    def __init__(self) -> None:
        self.paths: set[Path] = set()


# ``flock`` locks per open file, so a second descriptor in the same thread would wait
# on the first forever; each thread's held directories are taken only once.
_held_directories: Final = _HeldDirectories()


def stored(value: Tensor) -> Tensor:
    """Return a detached copy of ``value``, small whole numbers as bytes.

    Args:
      value: A recorded tensor.

    Returns:
      copy: The same values; an int32, int64, or float32 tensor with more than
        one element narrows to uint8 when every value is a whole number in
        [0, 256). Anything else, including a float ``-0.0``,
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

    Raises:
      ValueError: Generated keys collide with each other or with ``out``.
        The original output remains unchanged.

    """
    entries: list[tuple[str, Tensor]] = []
    for key in sorted({key for record in records for key in record}):
        values = [record[key] for record in records if key in record]
        uniform = len({(value.shape, value.dtype) for value in values}) == 1
        if len(values) == len(records) and uniform:
            entries.append((f"{prefix}/{key}", stored(torch.stack(values))))
            continue
        for index, record in enumerate(records, start=1):
            if key in record:
                entries.append((f"{prefix}/{index}/{key}", stored(record[key])))
    pending: dict[str, Tensor] = {}
    for key, value in entries:
        if key in out or key in pending:
            raise ValueError(f"Duplicate golden key: {key!r}.")
        pending[key] = value
    out.update(pending)


def flatten_record(
    value: object,
    *,
    prefix: str,
    skip: Collection[object] = (),
) -> dict[str, Tensor]:
    """Flatten nested tensors and numbers into one name-to-tensor record.

    Args:
      value: Tensors, numbers, and ``None`` inside nested mappings, tuples, and
        lists, such as an optimizer ``state_dict``.
      prefix: Key of ``value``; each child appends ``/<key or index>``.
      skip: Mapping keys whose values are dropped at any depth.

    Returns:
      record: Detached tensors in traversal order; a number becomes a scalar
        tensor and ``None`` leaves no key.

    Raises:
      TypeError: A leaf is none of those types.

    """
    if isinstance(value, Tensor):
        return {prefix: value.detach().clone()}
    if isinstance(value, (int, float, bool)):
        return {prefix: torch.tensor(value)}
    record: dict[str, Tensor] = {}
    if isinstance(value, dict):
        for key, item in cast("Mapping[object, object]", value).items():
            if key not in skip:
                record |= flatten_record(item, prefix=f"{prefix}/{key}", skip=skip)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(cast("list[object]", value)):
            record |= flatten_record(item, prefix=f"{prefix}/{index}", skip=skip)
    elif value is not None:
        raise TypeError(f"Unsupported record value: {type(value).__name__}")
    return record


def rng_fingerprint() -> Tensor:
    """Return the global generator's next 8 draws without advancing it.

    The raw state is 5 KB of Mersenne Twister words; the next draws pin the
    same position for a hundredth of the bytes.

    Returns:
      draws: The next 8 ``randint`` draws, as int64.

    """
    generator = torch.Generator()
    generator.set_state(torch.get_rng_state())
    return torch.randint(2**31 - 1, (8,), generator=generator)


# Every element, never a sample: a check that keeps the first few, or evenly spaced
# ones, cannot see a regression confined to the elements it dropped.
def joined(tensors: Iterable[Tensor]) -> Tensor:
    """Return every element of each tensor, flattened and joined into one.

    One key instead of one per tensor keeps a golden's index short.

    Args:
      tensors: Tensors in a fixed order.

    Returns:
      joined: A detached copy of every element in order, under ``torch.cat``
        type promotion. Floats widen exactly; an integer joined with a float is
        converted, which is lossy past the float's mantissa, so join like dtypes.

    """
    parts = [t.detach().reshape(-1) for t in tensors]
    return torch.cat(parts) if parts else torch.zeros(0)


def _is_stochastic(func: OpOverload[..., object], args: tuple[object, ...]) -> bool:
    if func is torch.ops.aten.dropout.default and (
        args[2] is False or args[1] in (0, 1)
    ):
        return False
    if func is torch.ops.aten.native_dropout.default and args[2] is False:
        return False
    return Tag.nondeterministic_seeded in func.tags


@cache
def _primitive_kind(func: OpOverload[..., object]) -> str:
    """Classify seeded ATen metadata without caching any draw or argument."""
    name = str(func)
    if name == "aten.dropout.default":
        return "dropout"
    for token in ("uniform_", "bernoulli_", "randperm", "random_"):
        if token in name:
            return token
    raise AssertionError(f"Unpinned random primitive: {name}.")


def _max_ulp(a: Tensor, b: Tensor) -> int | str:
    kind = {
        torch.float64: torch.int64,
        torch.float32: torch.int32,
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
    }.get(a.dtype)
    if kind is None:
        return "n/a"
    if bool(a.isnan().any() or b.isnan().any()):
        return "nan"
    # Python integers avoid overflow for opposite-sign binary64 endpoints.
    left = from_plain(_tensor_bytes(a).view(kind).tolist(), list[int])
    right = from_plain(_tensor_bytes(b).view(kind).tolist(), list[int])
    floor = torch.iinfo(kind).min
    return max(
        (
            abs((floor - x if x < 0 else x) - (floor - y if y < 0 else y))
            for x, y in zip(left, right, strict=True)
        ),
        default=0,
    )


def _tensor_bytes(value: Tensor) -> Tensor:
    # A lazy conjugate or negation bit is not in the stored bytes, and a byte
    # view of such a tensor is refused.
    flat = value.detach().cpu().resolve_conj().resolve_neg().contiguous().reshape(-1)
    # Contiguous singleton and empty tensors may still have stride zero.
    # A byte view requires stride one when its element size changes.
    return flat.as_strided(flat.shape, (1,)).view(torch.uint8)
