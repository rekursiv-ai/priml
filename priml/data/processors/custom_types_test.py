"""Tests for the media-sample schema."""

from __future__ import annotations

from priml.data.processors.custom_types import Sample


def test_sample_fields_are_all_optional() -> None:
    sample: Sample = {}
    assert Sample.__required_keys__ == frozenset()
    assert {"key", "media_tensor", "filter_reasons"} <= Sample.__optional_keys__
    assert sample == {}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
