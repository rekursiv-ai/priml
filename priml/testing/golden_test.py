"""Tests for readable golden-file assertions."""

from __future__ import annotations

from pathlib import Path

import pytest

from priml.testing.golden import assert_text_golden


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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
