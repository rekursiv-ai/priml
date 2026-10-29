"""Filter reasons: appending, decode-failure short-circuit, and reading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import logging

from treekle.types import ReadError

import pytest

from priml.data.filtering import add_filter_reason, read_filter_reasons


if TYPE_CHECKING:
    from priml.data.custom_types import FilterableSample


def test_add_appends_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    sample: FilterableSample = {}
    with caplog.at_level(logging.DEBUG, logger="priml.data.filtering"):
        add_filter_reason(sample, "Small", "too_small")
        add_filter_reason(sample, "Blur", "blurry")
    assert sample["filter_reasons"] == ["Small:too_small", "Blur:blurry"]
    assert "Filter: Blur:blurry" in caplog.text


def test_add_copies_shared_list() -> None:
    shared = ["A:x"]
    first: FilterableSample = {"filter_reasons": shared}
    add_filter_reason(first, "B", "y")
    assert shared == ["A:x"]
    assert first["filter_reasons"] == ["A:x", "B:y"]


def test_decode_failure_logs_nothing_and_blocks_later_reasons(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sample: FilterableSample = {}
    with caplog.at_level(logging.DEBUG, logger="priml.data.filtering"):
        add_filter_reason(sample, "CropDuringDecodeImage", "decode_failed")
        add_filter_reason(sample, "Other", "late")
    assert sample["filter_reasons"] == ["CropDuringDecodeImage:decode_failed"]
    assert caplog.text == ""


def test_read_keeps_only_strings() -> None:
    assert read_filter_reasons({"filter_reasons": ["a", 1, "b", None]}) == ["a", "b"]


def test_read_missing_is_empty_and_malformed_raises() -> None:
    assert read_filter_reasons({}) == []
    with pytest.raises(ReadError, match="cannot read"):
        read_filter_reasons({"filter_reasons": "abc"})


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
