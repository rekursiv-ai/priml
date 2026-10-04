"""Tests for golden-file assertions and compact tensor records."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import zlib

import pytest
import torch

from priml.testing import golden
from priml.testing.golden import (
    assert_tensor_golden,
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


def test_assert_text_golden_reads_testdata(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    (testdata / "example.txt").write_text("value\n", encoding="utf-8")

    assert_text_golden(
        request,
        test_file=str(test_file),
        name="example",
        rendered="value",
    )


def test_assert_text_golden_regenerates_missing_then_fails(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "owner_test.py"

    with pytest.raises(AssertionError, match="Missing golden regenerated"):
        assert_text_golden(
            request,
            test_file=str(test_file),
            name="example",
            rendered="value",
        )

    assert (tmp_path / "testdata" / "example.txt").read_text() == "value\n"


def test_assert_text_golden_creates_missing_ancestor_directories(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "absent" / "owner_test.py"
    with pytest.raises(AssertionError, match="Missing golden regenerated"):
        assert_text_golden(
            request,
            test_file=str(test_file),
            name="example",
            rendered="value",
        )
    assert (tmp_path / "absent" / "testdata" / "example.txt").read_text() == "value\n"


def test_assert_text_golden_fails_a_changed_render_as_an_assertion(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "owner_test.py"
    testdata = tmp_path / "testdata"
    testdata.mkdir()
    (testdata / "example.txt").write_text("value\n", encoding="utf-8")

    with pytest.raises(AssertionError) as error:
        assert_text_golden(
            request,
            test_file=str(test_file),
            name="example",
            rendered="other",
        )
    assert str(error.value) == (
        "example changed; read the diff, then rerun with --golden-overwrite "
        "if the change is intended."
    )


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


def test_read_tensors_rejects_a_non_tensor_value(tmp_path: Path) -> None:
    index = torch.frombuffer(bytearray(zlib.compress(b"")), dtype=torch.uint8)
    torch.save({"index": index, "a": 1}, tmp_path / "g.pt")
    with pytest.raises(TypeError):
        read_tensors(tmp_path / "g.pt")


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
    monkeypatch.delenv("BFB_REGENERATE", raising=False)
    path = tmp_path / "testdata" / "g.pt"
    record = _record()
    with pytest.raises(AssertionError, match="Missing golden minted"):
        assert_tensor_golden(path, record)
    assert_tensor_golden(path, record)
    changed = {**record, "d": torch.tensor(2.5, dtype=torch.float64)}
    with pytest.raises(AssertionError, match="1 mismatches"):
        assert_tensor_golden(path, changed)
    monkeypatch.setenv("BFB_REGENERATE", "1")
    assert_tensor_golden(path, changed)
    assert torch.equal(read_tensors(path)["d"], changed["d"])


def test_expect_golden_mismatch_blocks_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "testdata" / "g.pt"
    original = _record()
    changed = {**original, "d": torch.tensor(2.5, dtype=torch.float64)}
    monkeypatch.delenv("BFB_REGENERATE", raising=False)
    with pytest.raises(AssertionError, match="Missing golden minted"):
        assert_tensor_golden(path, original)
    before = path.read_bytes()

    monkeypatch.setenv("BFB_REGENERATE", "1")
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
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_file = tmp_path / "nested" / "owner_test.py"
    golden = tmp_path / "nested" / "testdata" / "named.txt"
    golden.parent.mkdir(parents=True)
    golden.write_text("old\n", encoding="utf-8")
    options: list[tuple[str, dict[str, object]]] = []

    def getoption(name: str, *args: object, **kwargs: object) -> bool:
        options.append((name, {"args": args, **kwargs}))
        return True

    monkeypatch.setattr(request.config, "getoption", getoption)
    assert_text_golden(
        request,
        test_file=str(test_file),
        name="named",
        rendered="new",
    )
    assert options == [("--golden-overwrite", {"args": (), "default": False})]
    assert golden.read_text(encoding="utf-8") == "new\n"


def test_assert_tensor_golden_creates_nested_parent_and_reports_every_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BFB_REGENERATE", raising=False)
    path = tmp_path / "deep" / "nested" / "g.pt"
    original = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    with pytest.raises(AssertionError, match="Missing golden minted"):
        assert_tensor_golden(path, original)

    changed = {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])}
    with pytest.raises(AssertionError) as error:
        assert_tensor_golden(path, changed)
    assert str(error.value) == "2 mismatches:\na: 1/1 differ\nb: 1/1 differ"


def test_assert_tensor_golden_regeneration_uses_only_exact_one_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "g.pt"
    record = {"x": torch.tensor([1.0, 2.0])}
    with pytest.raises(AssertionError, match="Missing golden minted"):
        assert_tensor_golden(path, record)
    before = path.read_bytes()
    monkeypatch.setenv("BFB_REGENERATE", "true")
    with pytest.raises(AssertionError, match="mismatches"):
        assert_tensor_golden(path, {"x": torch.tensor([3.0, 4.0])})
    assert path.read_bytes() == before
    monkeypatch.setenv("BFB_REGENERATE", "1")
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
    with pytest.raises(TypeError, match="cannot coerce"):
        golden.unpack("not a record")


def test_unpack_empty_index_returns_empty_record() -> None:
    assert golden.unpack(golden.pack({})) == {}


def test_text_golden_uses_utf8_for_disk_io(
    request: pytest.FixtureRequest,
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

    def getoption(name: str, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        return name == "--golden-overwrite"

    monkeypatch.setattr(request.config, "getoption", getoption)
    assert_text_golden(
        request,
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
