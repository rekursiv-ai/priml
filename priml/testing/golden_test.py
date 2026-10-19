"""Tests for golden-file assertions and compact tensor records."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast, override

import contextlib
import fcntl
import functools
import math
import subprocess
import sys
import textwrap
import threading
import zlib

from configgle.fig import Fig, Maker
from torch import nn
from treekle import ReadError

import pytest
import torch

from priml.model.linear import Linear
from priml.testing import golden, regenerate
from priml.testing.golden import (
    NativeDraws,
    PinnedDropout,
    PinnedRandom,
    assert_pprint_golden,
    assert_tensor_golden,
    assert_tensor_golden_part,
    assert_text_golden,
    expect_golden_mismatch,
    joined,
    mismatches,
    put_steps,
    read_tensors,
    rng_fingerprint,
    stored,
    write_tensors,
)


class _Load(Protocol):
    def __call__(self, f: Path, *, weights_only: bool) -> object: ...


_torch_load: _Load = torch.load


if TYPE_CHECKING:
    from torch import Tensor


@pytest.fixture(autouse=True)
def compare_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert, not regenerate, whatever flags this run was started with."""
    regenerate.override(monkeypatch, golden=False, b4b=False)


class _DefaultsOnly:
    class Config(Fig["_DefaultsOnly"]):
        default_value: int = 3
        """A value retained in the full golden."""

        long_default: str = "a default long enough to force dataclass pprint dispatch"
        """A long default retained in the full golden."""

    def __init__(self, config: Config) -> None:
        del config


class _Example:
    class Config(Fig["_Example"]):
        inherited: int = -1
        """A value filled during finalization."""

        @override
        def finalize(self) -> Self:
            self.inherited = 7
            return super().finalize()

    def __init__(self, config: Config) -> None:
        del config


class _PformatSpy:
    """Stand in for ``Maker.pformat`` and record the keyword arguments it got."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def __call__(self, **kwargs: object) -> str:
        self.kwargs = kwargs
        return "rendered"


def test_assert_pprint_golden_reads_full_finalized_config(tmp_path: Path) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    path = testdata / "example.txt"
    path.write_text(
        _Example.Config().pformat(hide_default_values=False) + "\n",
        encoding="utf-8",
    )

    assert_pprint_golden(
        test_file=str(test_file),
        name="example",
        config=_Example.Config(),
    )

    assert "inherited=7" in path.read_text(encoding="utf-8")


def test_assert_pprint_golden_pins_rendering_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spy = _PformatSpy()
    monkeypatch.setattr(Maker, "pformat", spy)
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    (testdata / "example.txt").write_text("rendered\n", encoding="utf-8")

    assert_pprint_golden(
        test_file=str(test_file),
        name="example",
        config=_Example.Config(),
    )

    assert spy.kwargs == {
        "finalize": True,
        "hide_default_values": False,
        "mask_memory_addresses": True,
        "width": 80,
    }


def test_assert_pprint_golden_renders_at_the_requested_width(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spy = _PformatSpy()
    monkeypatch.setattr(Maker, "pformat", spy)
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    (testdata / "example.txt").write_text("rendered\n", encoding="utf-8")

    assert_pprint_golden(
        test_file=str(test_file),
        name="example",
        config=_Example.Config(),
        width=81,
    )

    assert spy.kwargs["width"] == 81


def test_assert_pprint_golden_normalizes_rendered_text(tmp_path: Path) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    (testdata / "example.txt").write_text(
        "_EXAMPLE.CONFIG(INHERITED=7)\n",
        encoding="utf-8",
    )

    assert_pprint_golden(
        test_file=str(test_file),
        name="example",
        config=_Example.Config(),
        normalize=str.upper,
    )


def test_assert_pprint_golden_writes_full_unchanged_defaults(tmp_path: Path) -> None:
    test_file = tmp_path / "nested" / "owner_test.py"

    with pytest.raises(AssertionError, match="Missing golden regenerated"):
        assert_pprint_golden(
            test_file=str(test_file),
            name="defaults",
            config=_DefaultsOnly.Config(),
        )

    rendered = (tmp_path / "nested" / "testdata" / "defaults.txt").read_text(
        encoding="utf-8",
    )
    assert "default_value=3" in rendered
    assert "long_default='a default long enough" in rendered


def test_assert_pprint_golden_reports_mismatch_without_rewriting(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    path = testdata / "example.txt"
    path.write_text("stale\n", encoding="utf-8")

    with pytest.raises(AssertionError) as exc_info:
        assert_pprint_golden(
            test_file=str(test_file),
            name="example",
            config=_Example.Config(),
        )

    assert str(exc_info.value) == (
        "example changed; rerun with --regenerate-golden if intended.\n"
        f"--- {path}\n"
        "+++ example (rendered)\n"
        "@@ -1 +1 @@\n"
        "-stale\n"
        "+_Example.Config(inherited=7)\n"
    )
    assert path.read_text(encoding="utf-8") == "stale\n"


@pytest.mark.cli_python_subprocess
def test_assert_pprint_golden_rejects_mismatch_under_optimized_python(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    (testdata / "example.txt").write_text("stale\n", encoding="utf-8")
    script = textwrap.dedent(
        f"""
        from configgle.fig import Fig
        from priml.testing.golden import assert_pprint_golden

        class Example:
            class Config(Fig["Example"]):
                value: int = 1

            def __init__(self, config: Config) -> None:
                del config

        assert_pprint_golden(
            test_file={str(test_file)!r},
            name="example",
            config=Example.Config(),
        )
        """,
    )

    result = subprocess.run(  # noqa: S603 -- The test invokes a fixed helper command with controlled fixture arguments.
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "AssertionError" in result.stderr


def test_assert_pprint_golden_regenerates_under_the_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    path = testdata / "example.txt"
    path.write_text("stale\n", encoding="utf-8")
    regenerate.override(monkeypatch, golden=True)

    assert_pprint_golden(
        test_file=str(test_file),
        name="example",
        config=_Example.Config(),
    )

    assert "inherited=7" in path.read_text(encoding="utf-8")


def test_assert_text_golden_reads_testdata(tmp_path: Path) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    (testdata / "example.txt").write_text("value\n", encoding="utf-8")

    assert_text_golden(test_file=str(test_file), name="example", rendered="value")


def test_assert_text_golden_regenerates_missing_then_fails(tmp_path: Path) -> None:
    test_file = tmp_path / "owner_test.py"

    with pytest.raises(AssertionError, match="Missing golden regenerated"):
        assert_text_golden(test_file=str(test_file), name="example", rendered="value")

    assert (tmp_path / "testdata" / "example.txt").read_text() == "value\n"


def test_assert_text_golden_creates_missing_ancestor_directories(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "absent" / "owner_test.py"
    with pytest.raises(AssertionError, match="Missing golden regenerated"):
        assert_text_golden(test_file=str(test_file), name="example", rendered="value")
    assert (tmp_path / "absent" / "testdata" / "example.txt").read_text() == "value\n"


def test_assert_text_golden_fails_a_changed_render_as_an_assertion(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    (testdata / "example.txt").write_text("value\n", encoding="utf-8")

    with pytest.raises(AssertionError) as error:
        assert_text_golden(test_file=str(test_file), name="example", rendered="other")
    assert str(error.value) == (
        "example changed; read the diff, then rerun with --regenerate-golden "
        "if the change is intended."
    )


def _mint_then_signal(path: Path, done: threading.Event) -> None:
    """Mint a golden part, then set ``done``."""
    with contextlib.suppress(AssertionError):
        assert_tensor_golden_part(path, "a/", {"a/x": torch.tensor([1.0])})
    done.set()


def _torn_save(obj: object, f: Path) -> None:
    """Write a truncated file where ``torch.save`` was asked to, then fail."""
    del obj
    Path(f).write_bytes(b"torn")
    raise OSError("torn")


def _record() -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "a": torch.randn(3, 2, generator=generator),
        "b": torch.randn(5, generator=generator),
        "c": torch.arange(4, dtype=torch.int64),
        "d": torch.tensor(1.5, dtype=torch.float64),
    }


def test_write_tensors_round_trips_every_key(tmp_path: Path) -> None:
    record = _record()
    write_tensors(tmp_path / "g.pt", record)
    loaded = read_tensors(tmp_path / "g.pt")
    assert loaded.keys() == record.keys()
    for key, value in record.items():
        assert loaded[key].dtype == value.dtype
        assert loaded[key].shape == value.shape
        assert torch.equal(loaded[key], value)


def test_write_tensors_overhead_does_not_grow_with_key_count(tmp_path: Path) -> None:
    few = {"k0": torch.arange(300, dtype=torch.float64)}
    many = {
        f"metric/key_{i}": torch.tensor(float(i), dtype=torch.float64)
        for i in range(300)
    }
    write_tensors(tmp_path / "few.pt", few)
    write_tensors(tmp_path / "many.pt", many)
    index = sum(len(key) + 24 for key in many)
    growth = (tmp_path / "many.pt").stat().st_size - (
        tmp_path / "few.pt"
    ).stat().st_size
    assert growth < index


def test_write_tensors_keeps_scalars_empty_and_every_dtype(tmp_path: Path) -> None:
    record = {
        "scalar": torch.tensor(3.5, dtype=torch.float64),
        "empty": torch.zeros(0, 3),
        "flag": torch.tensor([True, False]),
        "half": torch.tensor([[0.5, 1.5]], dtype=torch.bfloat16),
        "bytes": torch.tensor([1, 255], dtype=torch.uint8),
        "a/b\\c": torch.arange(3, dtype=torch.int32),
    }
    write_tensors(tmp_path / "g.pt", record)
    loaded = read_tensors(tmp_path / "g.pt")
    assert list(loaded) == list(record)
    for key, value in record.items():
        assert loaded[key].dtype == value.dtype, key
        assert loaded[key].shape == value.shape, key
        assert torch.equal(loaded[key], value), key


def test_write_tensors_rejects_a_key_the_index_reserves(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tab or newline"):
        write_tensors(tmp_path / "g.pt", {"a\tb": torch.zeros(1)})


def test_read_tensors_rejects_a_foreign_file(tmp_path: Path) -> None:
    torch.save({"a": torch.zeros(1)}, tmp_path / "g.pt")
    with pytest.raises(TypeError):
        read_tensors(tmp_path / "g.pt")


@pytest.mark.parametrize("invalid_utf8", [False, True])
def test_read_tensors_wraps_a_corrupt_compressed_index(
    tmp_path: Path,
    invalid_utf8: bool,
) -> None:
    index = (
        torch.frombuffer(bytearray(zlib.compress(b"\xff")), dtype=torch.uint8)
        if invalid_utf8
        else torch.arange(3)
    )
    path = tmp_path / "foreign.pt"
    torch.save({"index": index}, path)
    with pytest.raises(TypeError, match=r"foreign\.pt is not a tensor golden"):
        read_tensors(path)


def test_read_tensors_rejects_a_non_tensor_value(tmp_path: Path) -> None:
    index = torch.frombuffer(bytearray(zlib.compress(b"")), dtype=torch.uint8)
    torch.save({"index": index, "a": 1}, tmp_path / "g.pt")
    with pytest.raises(TypeError):
        read_tensors(tmp_path / "g.pt")


@pytest.mark.parametrize(
    "index",
    [b"bad", b"x\tmissing\t2", b"x\tfloat32\ts", b"x\tfloat32\t2"],
)
def test_malformed_tensor_indices_keep_the_format_error_and_file_path(
    tmp_path: Path,
    index: bytes,
) -> None:
    path = tmp_path / "foreign.pt"
    packed = {
        "index": torch.frombuffer(bytearray(zlib.compress(index)), dtype=torch.uint8),
        "float32": torch.ones(1),
    }
    torch.save(packed, path)
    with pytest.raises(TypeError, match="invalid tensor index") as error:
        golden.unpack(packed)
    assert isinstance(error.value.__cause__, (ValueError, KeyError, RuntimeError))
    with pytest.raises(TypeError, match=r"foreign\.pt is not a tensor golden"):
        read_tensors(path)


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (torch.tensor([0, 3, 255], dtype=torch.int64), torch.uint8),
        (torch.tensor([0, 3, 255], dtype=torch.int32), torch.uint8),
        (torch.tensor([0.0, 3.0, 255.0]), torch.uint8),
        (torch.tensor([0, 256], dtype=torch.int64), torch.int64),
        (torch.tensor([-1, 3], dtype=torch.int64), torch.int64),
        (torch.tensor([0.5, 3.0]), torch.float32),
        (torch.tensor([0.0, 3.0], dtype=torch.float64), torch.float64),
        (torch.tensor(3, dtype=torch.int64), torch.int64),
        (torch.tensor([-0.0, 3.0]), torch.float32),
    ],
)
def test_stored_narrows_only_exact_small_whole_numbers(
    value: Tensor,
    dtype: torch.dtype,
) -> None:
    copy = stored(value)
    assert copy.dtype == dtype
    assert torch.equal(copy.to(value.dtype), value)
    assert torch.equal(copy.to(value.dtype).signbit(), value.signbit())


def test_stored_detaches_and_copies() -> None:
    value = torch.ones(2, requires_grad=True) * 0.5
    copy = stored(value)
    assert not copy.requires_grad
    assert copy.data_ptr() != value.data_ptr()


def test_put_steps_stacks_uniform_quantities() -> None:
    out: dict[str, Tensor] = {}
    put_steps(
        out,
        "train",
        [{"loss": torch.tensor([0.5 * i, 1.0])} for i in range(3)],
    )
    assert list(out) == ["train/loss"]
    assert out["train/loss"].shape == (3, 2)


def test_put_steps_keeps_ragged_quantities_per_step() -> None:
    out: dict[str, Tensor] = {}
    put_steps(
        out,
        "pool",
        [
            {"x": torch.zeros(2) + 0.5, "y": torch.zeros(1) + 0.5},
            {"x": torch.zeros(3) + 0.5},
        ],
    )
    assert sorted(out) == ["pool/1/x", "pool/1/y", "pool/2/x"]


def test_rng_fingerprint_does_not_advance_the_generator() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        first = rng_fingerprint()
        assert torch.equal(first, rng_fingerprint())
        torch.rand(1)
        assert not torch.equal(first, rng_fingerprint())


def test_joined_keeps_every_element_in_order_and_widens_exactly() -> None:
    """A golden compares whole tensors; joining one keeps its last element too."""
    first = torch.arange(6.0).view(2, 3)
    value = joined(
        [
            first,
            torch.tensor([0.5], dtype=torch.bfloat16),
            torch.tensor([1.25, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64),
        ],
    )
    assert value.dtype == torch.float64
    expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.25, 2.0, 3.0, 4.0, 5.0]
    assert value.tolist() == expected
    value[0] = 99.0
    assert first[0, 0] == 0.0


def test_joined_detaches_and_accepts_no_tensors() -> None:
    assert not joined([torch.ones(2, 3, requires_grad=True)]).requires_grad
    assert joined([]).numel() == 0


def test_golden_offers_no_truncating_helper() -> None:
    """Sampling a compared tensor hides the elements it drops from the check."""
    assert not {"leading", "spread", "heads"} & set(vars(golden))


def test_mismatches_sees_presence_dtype_shape_and_bits() -> None:
    base = {"x": torch.arange(6, dtype=torch.float32)}
    variants = [
        base["x"].to(torch.float64),
        base["x"].reshape(2, 3),
        base["x"].clone().index_fill_(0, torch.tensor([5]), 6.0),
    ]
    reports = [
        "x: torch.float64[6] vs torch.float32[6]",
        "x: torch.float32[2, 3] vs torch.float32[6]",
        "x: 1/6 differ",
    ]
    assert [mismatches(base, {"x": v}) for v in variants] == [[r] for r in reports]
    assert mismatches(base, {}) == ["missing x"]
    assert mismatches({}, base) == ["unexpected x"]
    assert not mismatches(base, {"x": base["x"].clone()})


def test_mismatches_compares_signed_zeros_and_nan_payloads_by_bits() -> None:
    assert mismatches({"x": torch.tensor([0.0])}, {"x": torch.tensor([-0.0])}) == [
        "x: 1/1 differ",
    ]
    nan = torch.tensor([float("nan")])
    assert not mismatches({"x": nan}, {"x": nan.clone()})


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_mismatches_reports_one_ulp_without_downcasting(dtype: torch.dtype) -> None:
    want = torch.tensor([1.0, -1.0, 0.0], dtype=dtype)
    got = torch.nextafter(want, torch.full_like(want, math.inf))
    assert mismatches({"x": want}, {"x": got}, ulps=True) == [
        "x: 3/3 differ; max_ulp=1",
    ]


def test_mismatches_binary64_distance_does_not_overflow_signed_int64() -> None:
    maximum = torch.finfo(torch.float64).max
    want = torch.tensor([-maximum], dtype=torch.float64)
    got = -want
    assert mismatches({"x": want}, {"x": got}, ulps=True) == [
        f"x: 1/1 differ; max_ulp={2 * 0x7FEFFFFFFFFFFFFF}",
    ]


def test_mismatches_reports_nan_payload_changes_without_numeric_distance() -> None:
    want = torch.tensor([0x7FF8000000000001], dtype=torch.int64).view(torch.float64)
    got = torch.tensor([0x7FF8000000000002], dtype=torch.int64).view(torch.float64)
    assert mismatches({"x": want}, {"x": got}, ulps=True) == [
        "x: 1/1 differ; max_ulp=nan",
    ]
    assert mismatches({"x": want}, {"x": got}) == ["x: 1/1 differ"]
    assert not mismatches({"x": want}, {"x": want.clone()}, ulps=True)


def test_ulp_diagnostics_preserve_signed_zero_and_metadata_mismatches() -> None:
    want = {"x": torch.tensor([0.0])}
    assert mismatches(want, {"x": torch.tensor([-0.0])}, ulps=True) == [
        "x: 1/1 differ; max_ulp=0",
    ]
    assert mismatches(want, {}, ulps=True) == ["missing x"]
    assert mismatches({}, want, ulps=True) == ["unexpected x"]
    assert mismatches(want, {"x": want["x"].double()}, ulps=True) == [
        "x: torch.float64[1] vs torch.float32[1]",
    ]
    assert mismatches(want, {"x": want["x"].reshape(())}, ulps=True) == [
        "x: torch.float32[] vs torch.float32[1]",
    ]


def test_ulp_diagnostics_report_unsupported_float_formats_without_coercion() -> None:
    want = torch.tensor([1.0], dtype=torch.float8_e4m3fn)
    got = torch.tensor([2.0], dtype=torch.float8_e4m3fn)
    assert mismatches({"x": want}, {"x": got}, ulps=True) == [
        "x: 1/1 differ; max_ulp=n/a",
    ]


def test_mismatches_counts_elements_with_multiple_different_bytes() -> None:
    expected = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    actual = torch.tensor([0.0, 0.5, 1.0, -2.25, 3.0, -4.5])
    assert mismatches({"values": expected}, {"values": actual}) == [
        "values: 2/6 differ",
    ]


@pytest.mark.parametrize("size", [0, 1])
def test_mismatches_accepts_contiguous_zero_stride_views(size: int) -> None:
    expanded = torch.tensor(1, dtype=torch.int64).expand(size)
    assert expanded.is_contiguous()
    assert expanded.stride() == (0,)
    assert not mismatches({"x": expanded}, {"x": torch.ones(size, dtype=torch.int64)})
    if size:
        assert mismatches({"x": expanded}, {"x": torch.tensor([2])}) == [
            "x: 1/1 differ",
        ]


def test_mismatches_compares_conjugate_and_negative_views_by_value() -> None:
    values = torch.tensor([1 + 2j, -3 - 4j], dtype=torch.complex64)
    # Lazy conjugate and negative views refuse a byte view until resolved.
    assert not mismatches({"z": values.conj()}, {"z": values.conj().resolve_conj()})
    imaginary = values.conj().imag
    assert imaginary.is_neg()
    assert not mismatches({"y": imaginary}, {"y": imaginary.resolve_neg()})
    assert mismatches({"y": imaginary}, {"y": values.imag}) == ["y: 2/2 differ"]


def test_assert_tensor_golden_mints_missing_then_compares(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "testdata" / "g.pt"
    record = _record()
    with pytest.raises(AssertionError, match="Missing golden minted"):
        assert_tensor_golden(path, record)
    assert_tensor_golden(path, record)
    changed = {**record, "d": torch.tensor(2.5, dtype=torch.float64)}
    with pytest.raises(AssertionError, match="1 mismatches"):
        assert_tensor_golden(path, changed)
    regenerate.override(monkeypatch, b4b=True)
    assert_tensor_golden(path, changed)
    assert torch.equal(read_tensors(path)["d"], changed["d"])


def test_a_golden_part_is_minted_once_then_replayed_and_regenerated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing part mints and fails; a stored one replays; a regen keeps the rest."""
    path = tmp_path / "nested" / "testdata" / "golden.pt"
    first = {"a/x": torch.tensor([1.0, 2.0]), "a/z": torch.tensor([0])}
    with pytest.raises(AssertionError, match=r"^Missing golden part a/ minted in "):
        assert_tensor_golden_part(path, "a/", first)
    assert_tensor_golden_part(path, "a/", first)
    with pytest.raises(AssertionError, match=r"^Missing golden part b/"):
        assert_tensor_golden_part(path, "b/", {"b/y": torch.tensor([3])})
    assert read_tensors(path).keys() == {"a/x", "a/z", "b/y"}
    changed = {"a/x": torch.tensor([1.0, 5.0]), "a/z": torch.tensor([1])}
    report = r"^2 mismatches:\na/x: 1/2 differ\na/z: 1/1 differ$"
    with pytest.raises(AssertionError, match=report):
        assert_tensor_golden_part(path, "a/", changed)
    with pytest.raises(ValueError, match=r"^Golden part a/ holds keys outside it"):
        assert_tensor_golden_part(path, "a/", {"b/y": torch.tensor([3])})
    regenerate.override(monkeypatch, b4b=True)
    assert_tensor_golden_part(path, "a/", changed)
    record = read_tensors(path)
    assert record.keys() == {"a/x", "a/z", "b/y"}
    assert torch.equal(record["a/x"], changed["a/x"])
    assert torch.equal(record["b/y"], torch.tensor([3]))


@pytest.mark.parametrize("part", [False, True])
def test_a_failed_rewrite_leaves_the_golden_as_it_was(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    part: bool,
) -> None:
    path = tmp_path / "g.pt"
    write_tensors(path, {"a/x": torch.tensor([1.0])})
    monkeypatch.setattr(torch, "save", _torn_save)
    regenerate.override(monkeypatch, b4b=True)
    rewrite = (
        functools.partial(assert_tensor_golden_part, path, "a/")
        if part
        else functools.partial(assert_tensor_golden, path)
    )
    with pytest.raises(OSError, match=r"^torn$"):
        rewrite({"a/x": torch.tensor([2.0])})
    monkeypatch.undo()
    assert torch.equal(read_tensors(path)["a/x"], torch.tensor([1.0]))
    assert [p.name for p in tmp_path.iterdir()] == ["g.pt"]


def test_a_golden_part_waits_for_the_lock_another_writer_holds(tmp_path: Path) -> None:
    path = tmp_path / "g.pt"
    done = threading.Event()
    writer = threading.Thread(
        target=_mint_then_signal,
        args=(path, done),
        daemon=True,
    )
    with golden._part_lock().open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        writer.start()
        assert not done.wait(0.05)
    writer.join(timeout=5)
    assert done.is_set()
    assert read_tensors(path).keys() == {"a/x"}


def test_expect_golden_mismatch_blocks_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "testdata" / "g.pt"
    original = _record()
    changed = {**original, "d": torch.tensor(2.5, dtype=torch.float64)}
    with pytest.raises(AssertionError, match="Missing golden minted"):
        assert_tensor_golden(path, original)
    before = path.read_bytes()

    regenerate.override(monkeypatch, b4b=True)
    assert_tensor_golden(path, changed)
    assert path.read_bytes() != before

    write_tensors(path, original)
    before = path.read_bytes()
    with expect_golden_mismatch(match=r"1 mismatches"):
        assert_tensor_golden(path, changed)
    assert path.read_bytes() == before


def test_expect_golden_mismatch_requires_the_message_to_match(
    tmp_path: Path,
) -> None:
    path = tmp_path / "testdata" / "g.pt"
    record = _record()
    with pytest.raises(AssertionError, match="Missing golden minted"):
        assert_tensor_golden(path, record)
    with (
        pytest.raises(AssertionError, match="different failure"),
        expect_golden_mismatch(match=r"1 mismatches"),
    ):
        raise AssertionError("different failure")


def test_assert_text_golden_overwrites_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_file = tmp_path / "nested" / "owner_test.py"
    path = tmp_path / "nested" / "testdata" / "named.txt"
    path.parent.mkdir(parents=True)
    path.write_text("old\n", encoding="utf-8")
    regenerate.override(monkeypatch, b4b=True)
    with pytest.raises(AssertionError, match="named changed"):
        assert_text_golden(test_file=str(test_file), name="named", rendered="new")
    assert path.read_text(encoding="utf-8") == "old\n"
    regenerate.override(monkeypatch, golden=True)
    assert_text_golden(test_file=str(test_file), name="named", rendered="new")
    assert path.read_text(encoding="utf-8") == "new\n"


def test_assert_tensor_golden_creates_nested_parent_and_reports_every_diff(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deep" / "nested" / "g.pt"
    original = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    with pytest.raises(AssertionError, match="Missing golden minted"):
        assert_tensor_golden(path, original)

    changed = {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])}
    with pytest.raises(AssertionError) as error:
        assert_tensor_golden(path, changed)
    assert str(error.value) == "2 mismatches:\na: 1/1 differ\nb: 1/1 differ"


def test_assert_tensor_golden_regenerates_only_under_the_b4b_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "g.pt"
    record = {"x": torch.tensor([1.0, 2.0])}
    with pytest.raises(AssertionError, match="Missing golden minted"):
        assert_tensor_golden(path, record)
    before = path.read_bytes()
    regenerate.override(monkeypatch, golden=True)
    with pytest.raises(AssertionError, match="mismatches"):
        assert_tensor_golden(path, {"x": torch.tensor([3.0, 4.0])})
    assert path.read_bytes() == before
    regenerate.override(monkeypatch, b4b=True)
    assert_tensor_golden(path, {"x": torch.tensor([3.0, 4.0])})
    assert torch.equal(read_tensors(path)["x"], torch.tensor([3.0, 4.0]))


def test_pack_records_dtype_prefix_shape_and_compression_level() -> None:
    record = {
        "matrix": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "scalar": torch.tensor(5, dtype=torch.int64),
    }
    packed = golden.pack(record)
    index = zlib.decompress(packed["index"].numpy().tobytes()).decode()
    assert index == "matrix\tfloat32\t2,2\nscalar\tint64\t"
    assert list(packed) == ["float32", "int64", "index"]


def test_pack_rejects_newline_and_tab_in_keys() -> None:
    for key in ("line\nbreak", "tab\tkey"):
        with pytest.raises(ValueError, match="Golden key"):
            golden.pack({key: torch.ones(2)})


def test_put_steps_stores_uniform_and_ragged_values_without_skipping() -> None:
    records = [
        {"a": torch.tensor([1, 2]), "b": torch.tensor([3])},
        {"a": torch.tensor([4, 5]), "b": torch.tensor([6, 7])},
    ]
    output: dict[str, Tensor] = {}
    put_steps(output, "step", records)
    assert list(output) == ["step/a", "step/1/b", "step/2/b"]
    assert torch.equal(output["step/a"], torch.tensor([[1, 2], [4, 5]]))
    assert torch.equal(output["step/1/b"], torch.tensor([3]))
    assert torch.equal(output["step/2/b"], torch.tensor([6, 7]))


@pytest.mark.parametrize("existing", [False, True])
def test_put_steps_rejects_collisions_before_extending_output(existing: bool) -> None:
    original = torch.tensor([19, 23])
    output = {"step/1/x" if existing else "untouched": original}
    records = [
        {"x": torch.tensor([1, 2]), "1/x": torch.tensor([6, 7, 8, 9])},
        {"x": torch.tensor([3, 4, 5]), "1/x": torch.tensor([10, 11, 12, 13])},
    ]
    with pytest.raises(ValueError, match="Duplicate golden key"):
        put_steps(output, "step", records)
    assert list(output) == ["step/1/x" if existing else "untouched"]
    assert next(iter(output.values())) is original


def test_read_tensors_requests_weights_only_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "golden.pt"
    torch.save(golden.pack({"x": torch.tensor([1, 2])}), path)

    kwargs_seen: list[dict[str, object]] = []

    def load(actual_path: Path, **kwargs: object) -> object:
        kwargs_seen.append(kwargs)
        weights_only = kwargs.get("weights_only")
        assert isinstance(weights_only, bool)
        return _torch_load(actual_path, weights_only=weights_only)

    monkeypatch.setattr(torch, "load", load)
    assert torch.equal(read_tensors(path)["x"], torch.tensor([1, 2]))
    assert kwargs_seen == [{"weights_only": True}]


def test_rng_fingerprint_is_exact_next_eight_draws_without_advancing() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(314)
        state = torch.get_rng_state()
        expected = torch.randint(
            0,
            2**31 - 1,
            (8,),
            generator=torch.Generator().set_state(state),
        )
        before = torch.get_rng_state()
        actual = rng_fingerprint()
        assert actual.dtype == torch.int64
        assert actual.shape == (8,)
        assert torch.equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), before)


def test_stored_narrows_two_exact_whole_numbers() -> None:
    value = torch.tensor([0, 255], dtype=torch.int64)
    result = stored(value)
    assert result.dtype == torch.uint8
    assert torch.equal(result.to(torch.int64), value)


def test_tensor_bits_equal_rejects_shape_or_dtype_independently() -> None:
    value = torch.arange(6, dtype=torch.float32)
    assert not golden.tensor_bits_equal(value, value.reshape(2, 3))
    assert not golden.tensor_bits_equal(value, value.to(torch.float64))


def test_unpack_rejects_payload_without_index() -> None:
    with pytest.raises(TypeError, match=r"^Not a record written by pack\.$"):
        golden.unpack({"float32": torch.ones(2)})


def test_unpack_rejects_a_non_mapping() -> None:
    with pytest.raises(ReadError):
        golden.unpack("not a record")


def test_unpack_empty_index_returns_empty_record() -> None:
    assert golden.unpack(golden.pack({})) == {}


def test_text_golden_uses_utf8_for_disk_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_file = tmp_path / "owner_test.py"
    golden_path = tmp_path / "testdata" / "disk.txt"
    golden_path.parent.mkdir()
    golden_path.write_text("value\n", encoding="utf-8")
    writes: list[dict[str, object]] = []
    reads: list[dict[str, object]] = []
    write_text = Path.write_text
    read_text = Path.read_text

    def track_write(path: Path, data: str, **kwargs: object) -> int:
        writes.append(kwargs)
        encoding = kwargs.get("encoding")
        assert encoding is None or isinstance(encoding, str)
        return write_text(path, data, encoding=encoding)

    def track_read(path: Path, **kwargs: object) -> str:
        reads.append(kwargs)
        encoding = kwargs.get("encoding")
        assert encoding is None or isinstance(encoding, str)
        return read_text(path, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", track_write)
    monkeypatch.setattr(Path, "read_text", track_read)

    regenerate.override(monkeypatch, golden=True)
    assert_text_golden(
        test_file=str(test_file),
        name="disk",
        rendered="value",
    )
    assert writes == [{"encoding": "utf-8"}]
    assert reads == [{"encoding": "utf-8"}]


def test_pack_requests_maximum_index_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compress = zlib.compress
    levels: list[int] = []

    def track_compress(data: bytes, *, level: int) -> bytes:
        levels.append(level)
        return compress(data, level=level)

    monkeypatch.setattr(zlib, "compress", track_compress)
    golden.pack({"x": torch.ones(2)})
    assert levels == [9]


def test_read_tensors_wraps_invalid_pack_error_with_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "foreign.pt"
    torch.save({"float32": torch.ones(2)}, path)
    with pytest.raises(
        TypeError,
        match=r"^.*foreign\.pt is not a tensor golden written by write_tensors\.$",
    ):
        read_tensors(path)


def test_pinned_random_preserves_draw_order_counters_and_global_rng() -> None:
    before = torch.get_rng_state()
    with PinnedRandom() as pins:
        uniform = torch.empty(2, 3).uniform_(-2, 3)
        bernoulli = torch.empty(5).bernoulli_(0.05)
        integer = torch.empty((), dtype=torch.int64).random_()
        permutation = torch.randperm(7)
    assert torch.equal(uniform, (torch.arange(6).reshape(2, 3) + 1) / 97 * 5 - 2)
    assert torch.equal(bernoulli, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))
    assert integer == 3
    assert torch.equal(permutation, torch.arange(7).roll(4))
    assert sum(pins.calls.values()) == 4
    assert len(pins.calls) == 4
    assert all(key.endswith("/factory") for key in pins.calls)
    assert torch.equal(before, torch.get_rng_state())


def test_pinned_random_records_model_initializer_sites() -> None:
    with PinnedRandom() as pins:
        nn.Linear(2, 3)
    assert sum(pins.calls.values()) == 2
    assert all("/linear.py:" in key for key in pins.calls)


def test_pinned_random_preserves_distinct_direct_call_sites() -> None:
    with PinnedRandom() as pins:
        torch.empty(2).uniform_()
        torch.empty(3).uniform_()
    assert sum(pins.calls.values()) == sum(pins.site_calls.values()) == 2
    assert len(pins.calls) == 1
    assert len(pins.site_calls) == 2


def test_pinned_random_preserves_directories_for_matching_basenames() -> None:
    with PinnedRandom() as pins:
        nn.Linear(3, 4)
        Linear.Config(3, 4, bias=True, init_bias=None).make()
    assert sum(pins.site_calls.values()) == sum(pins.calls.values()) == 4
    assert any("/torch/nn/modules/linear.py:" in site for site in pins.site_calls)
    assert any("/priml/model/linear.py:" in site for site in pins.site_calls)


def test_pinned_random_rejects_unrecognized_stochastic_draws() -> None:
    with (
        PinnedRandom(),
        pytest.raises(AssertionError, match="Unpinned random primitive"),
    ):
        torch.randn(2, 3)


def test_pinned_random_accepts_empty_permutations() -> None:
    with PinnedRandom():
        actual = torch.randperm(0)
    assert actual.shape == (0,)
    assert actual.dtype == torch.int64


@pytest.mark.parametrize("bounds", [(0, 1), (-3, -1), (4, 5)])
def test_pinned_random_keeps_integer_draws_inside_the_bounds(
    bounds: tuple[int, int],
) -> None:
    low, high = bounds
    with PinnedRandom():
        actual = torch.empty(2, 3, dtype=torch.int64).random_(low, high)
    assert bool(((actual >= low) & (actual < high)).all())


@pytest.mark.parametrize("dtype", [torch.int8, torch.int64, torch.float32, torch.bool])
def test_pinned_random_default_integer_draw_preserves_counter_values(
    dtype: torch.dtype,
) -> None:
    with PinnedRandom():
        actual = torch.empty(2, 3, dtype=dtype).random_()
    assert torch.equal(actual, torch.ones(2, 3, dtype=dtype))


def test_pinned_random_one_sided_integer_draws_and_invalid_bounds() -> None:
    with PinnedRandom():
        assert torch.empty((), dtype=torch.int64).random_(1) == 0
        assert torch.empty((), dtype=torch.int64).random_(4, None) == 6
        with pytest.raises(ValueError, match="upper bound"):
            torch.empty((), dtype=torch.int64).random_(1, 1)


@pytest.mark.parametrize("count", [0, 7])
def test_pinned_random_populates_permutation_output_storage(count: int) -> None:
    output = torch.full((count,), -1, dtype=torch.int32)
    with PinnedRandom():
        actual = torch.randperm(count, out=output)
    assert actual.data_ptr() == output.data_ptr()
    assert actual.dtype == torch.int32
    assert torch.equal(output.sort().values, torch.arange(count, dtype=torch.int32))


def test_pinned_random_evaluation_and_training_composite_dropout() -> None:
    value = torch.ones(2, 3)
    with PinnedRandom() as pins:
        actual = pins.__torch_dispatch__(
            torch.ops.aten.dropout.default,
            (torch.Tensor,),
            (value, 0.25, False),
        )
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, value)
        with pytest.raises(AssertionError, match="Unpinned random primitive"):
            pins.__torch_dispatch__(
                torch.ops.aten.dropout.default,
                (torch.Tensor,),
                (value, 0.25, True),
            )
    assert not pins.calls


def test_native_draws_captures_before_states_and_complete_outputs() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        before = torch.get_rng_state()
        with NativeDraws() as record:
            uniform = torch.rand(2, 3)
            between = torch.get_rng_state()
            normal = torch.randn(5)
        assert [name for name, _ in record.draws] == [
            "aten.rand.default",
            "aten.randn.default",
        ]
        assert torch.equal(record.draws[0][1]["torch"], before)
        assert torch.equal(record.draws[1][1]["torch"], between)
        assert torch.equal(record.outputs["draw/1/0"], uniform)
        assert torch.equal(record.outputs["draw/2/0"], normal)
        uniform.zero_()
        assert bool(record.outputs["draw/1/0"].any())


def test_native_draws_captures_every_native_dropout_output() -> None:
    with torch.random.fork_rng(devices=[]), NativeDraws() as record:
        output, mask = cast(
            "tuple[torch.Tensor, torch.Tensor]",
            torch.ops.aten.native_dropout.default(torch.ones(2, 3), 0.25, True),
        )
    assert len(record.draws) == 1
    assert torch.equal(record.outputs["draw/1/0"], output)
    assert torch.equal(record.outputs["draw/1/1"], mask)


def test_native_draws_captures_explicit_and_global_generator_states() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        generator = torch.Generator().manual_seed(31)
        explicit_before = generator.get_state()
        global_before = torch.get_rng_state()
        with NativeDraws() as record:
            output = torch.rand(2, 3, generator=generator)
        assert torch.equal(record.draws[0][1]["torch"], global_before)
        device, state = record.generator_states[1]
        assert device == generator.device
        assert torch.equal(state, explicit_before)
        replay = torch.rand(
            2,
            3,
            generator=torch.Generator(device=device).set_state(state),
        )
        assert torch.equal(replay, output)
        assert torch.equal(record.outputs["draw/1/0"], output)


def test_evaluation_dropout_has_no_native_draw_receipt() -> None:
    model = nn.Dropout(0.25).eval()
    value = torch.ones(2, 3)
    before = torch.get_rng_state()
    with torch.inference_mode(), NativeDraws() as record:
        actual = model(value)
    assert torch.equal(actual, value)
    assert torch.equal(torch.get_rng_state(), before)
    assert not record.draws
    assert not record.outputs


def test_native_evaluation_dropout_has_no_random_receipt() -> None:
    value = torch.ones(2, 3)
    before = torch.get_rng_state()
    with NativeDraws() as record:
        output, mask = cast(
            "tuple[torch.Tensor, torch.Tensor]",
            torch.ops.aten.native_dropout.default(value, 0.25, False),
        )
    assert torch.equal(output, value)
    assert bool(mask.all())
    assert torch.equal(torch.get_rng_state(), before)
    assert not record.draws
    assert not record.outputs
    assert not record.generator_states


@pytest.mark.parametrize("pin_all", [False, True])
def test_native_evaluation_dropout_passes_both_pin_modes(pin_all: bool) -> None:
    value = torch.ones(2, 3)
    before = torch.get_rng_state()
    mode = PinnedRandom() if pin_all else PinnedDropout(nn.Identity())
    with mode as pins:
        output, mask = cast(
            "tuple[torch.Tensor, torch.Tensor]",
            torch.ops.aten.native_dropout.default(value, 0.25, False),
        )
    assert torch.equal(output, value)
    assert bool(mask.all())
    assert torch.equal(torch.get_rng_state(), before)
    assert not pins.calls
    if isinstance(pins, PinnedRandom):
        assert not pins.site_calls
    else:
        assert not pins.sites


@pytest.mark.parametrize("probability", [0.0, 1.0])
@pytest.mark.parametrize("observer", [NativeDraws, PinnedRandom])
def test_composite_dropout_endpoints_leave_no_draw_receipts(
    probability: float,
    observer: type[NativeDraws | PinnedRandom],
) -> None:
    model = nn.Dropout(probability).train()
    value = torch.ones(2, 3)
    before = torch.get_rng_state()
    with torch.inference_mode(), observer() as record:
        output = model(value)
    assert torch.equal(output, value * (1 - probability))
    assert torch.equal(torch.get_rng_state(), before)
    if isinstance(record, NativeDraws):
        assert not record.draws
        assert not record.outputs
    else:
        assert not record.calls
        assert not record.site_calls


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_composite_dropout_endpoints_pass_dropout_pins(probability: float) -> None:
    model = nn.Dropout(probability).train()
    value = torch.ones(2, 3)
    before = torch.get_rng_state()
    with torch.inference_mode(), PinnedDropout(model) as pins:
        output = model(value)
    assert torch.equal(output, value * (1 - probability))
    assert torch.equal(torch.get_rng_state(), before)
    assert pins.calls == 0
    assert not pins.sites


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_native_training_dropout_endpoints_still_consume_a_draw(
    probability: float,
) -> None:
    with torch.random.fork_rng(devices=[]):
        before = torch.get_rng_state()
        with NativeDraws() as record:
            output, mask = cast(
                "tuple[torch.Tensor, torch.Tensor]",
                torch.ops.aten.native_dropout.default(
                    torch.ones(2, 3),
                    probability,
                    True,
                ),
            )
        assert not torch.equal(torch.get_rng_state(), before)
        assert len(record.draws) == 1
        assert torch.equal(record.outputs["draw/1/0"], output)
        assert torch.equal(record.outputs["draw/1/1"], mask)


def test_pinned_dropout_preserves_offset_sites_and_native_rng() -> None:
    model = nn.Sequential(nn.Dropout(0.25), nn.Dropout(0.5))
    before = torch.get_rng_state()
    with PinnedDropout(model, draw_offset=2) as pins:
        actual = model(torch.ones(4, 5))
    first = ((torch.arange(20) + 14) % 20).reshape(4, 5) / 20 < 0.75
    second = ((torch.arange(20) + 21) % 20).reshape(4, 5) / 20 < 0.5
    assert torch.equal(actual, first.float().div_(0.75) * second.float().div_(0.5))
    assert pins.calls == 2
    assert pins.sites == ["0", "1"]
    assert torch.equal(torch.get_rng_state(), before)


def test_pinned_dropout_accepts_evaluation_in_inference_mode() -> None:
    model = nn.Dropout(0.25).eval()
    value = torch.ones(2, 3)
    before = torch.get_rng_state()
    with torch.inference_mode(), PinnedDropout(model) as pins:
        actual = model(value)
    assert torch.equal(actual, value)
    assert torch.equal(torch.get_rng_state(), before)
    assert pins.calls == 0
    assert not pins.sites


def test_pinned_dropout_passes_deterministic_ops_and_rejects_other_random_ops() -> None:
    with PinnedDropout(nn.Identity()) as pins:
        assert torch.equal(torch.arange(3) + 1, torch.tensor([1, 2, 3]))
        with pytest.raises(ValueError, match="rand"):
            torch.rand(2, 3)
    assert pins.calls == 0
    assert pins.sites == []


def test_pinned_dropout_handles_default_probability_at_a_functional_site() -> None:
    with PinnedDropout(nn.Identity()) as pins:
        actual = torch.empty(2, 3).bernoulli_()
    assert torch.equal(actual, torch.ones(2, 3))
    assert pins.calls == 1
    assert pins.sites == ["functional"]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
