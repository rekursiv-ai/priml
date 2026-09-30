from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import Fig

import pytest

from priml.data.processors.profiling import ProfiledProcessor


if TYPE_CHECKING:
    from collections.abc import Iterator


class Identity:
    class Config(Fig["Identity"]): ...

    def __init__(self, config: Config) -> None:
        del config

    def __call__(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        yield from samples


def test_missing_processor_is_rejected() -> None:
    with pytest.raises(ValueError, match="Must specify"):
        ProfiledProcessor.Config().make()


def test_cpu_profile_is_transparent() -> None:
    processor = ProfiledProcessor.Config(processor=Identity.Config()).make()
    sample: dict[str, object] = {"x": 1}
    assert list(processor(iter([sample]))) == [sample]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
