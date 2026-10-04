from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import Fig

import pytest
import torch

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
    with pytest.raises(ValueError, match=r"^Must specify `processor`\.$"):
        ProfiledProcessor.Config().make()


def test_cpu_profile_is_transparent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    processor = ProfiledProcessor.Config(processor=Identity.Config()).make()
    sample: dict[str, object] = {"x": 1}
    assert processor.use_cuda is False
    assert list(processor(iter([sample]))) == [sample]


def test_cuda_profile_records_and_logs_each_sample(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing

        def record(self) -> None:
            pass

        def elapsed_time(self, end_event: object) -> float:
            assert isinstance(end_event, FakeEvent)
            return 2.5

    synchronized: list[bool] = []
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronized.append(True))
    processor = ProfiledProcessor.Config(processor=Identity.Config()).make()
    processor.use_cuda = True
    sample: dict[str, object] = {"x": 1}

    with caplog.at_level("DEBUG"):
        result = list(processor(iter([sample])))

    assert result == [sample]
    assert synchronized == [True]
    assert len(caplog.records) == 1
    assert caplog.records[0].msg == "%s: %.1fms"
    assert caplog.records[0].args == ("Identity", 2.5)
    assert caplog.records[0].getMessage() == "Identity: 2.5ms"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
