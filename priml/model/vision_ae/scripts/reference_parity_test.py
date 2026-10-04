"""The parity harness's own checks: randomness, inventory, branches, and git."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import importlib
import sys

from torch.nn import functional

import pytest
import torch

from priml.model.vision_ae.scripts import reference_parity
from priml.model.vision_ae.scripts.reference_parity import (
    Outcome,
    PinnedRandom,
    Reference,
    clone_upstream,
    definitions,
    inventory_problems,
    measure,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor


_THIS: Final = Path(__file__).resolve()
_PORT_ROOT: Final = _THIS.parents[3]


def test_module_imports_without_test_only_or_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wheel ships no ``conftest``, and ``coverage`` comes with the test group."""
    for name in ("priml.conftest", "coverage", "transformers", "safetensors.torch"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, reference_parity.__name__)
    _ = importlib.import_module(reference_parity.__name__)


def test_pinned_random_refuses_a_draw_inside_a_native_kernel() -> None:
    with PinnedRandom([]), pytest.raises(AssertionError, match="unpinned random op"):
        _ = functional.dropout(torch.ones(8), p=0.5, training=True)
    with PinnedRandom([]), pytest.raises(AssertionError, match=r"aten\.rand"):
        _ = cast("Tensor", torch.ops.aten.rand.default([2]))


def test_pinned_random_lets_attention_without_dropout_run() -> None:
    """Fused attention is seeded for its dropout; at zero it draws nothing."""
    q = torch.ones(1, 1, 2, 4)
    with PinnedRandom([]):
        _ = functional.scaled_dot_product_attention(q, q, q)
    with PinnedRandom([]), pytest.raises(AssertionError, match="unpinned random op"):
        _ = functional.scaled_dot_product_attention(q, q, q, dropout_p=0.5)


def test_pinned_random_serves_randn_however_it_is_spelled() -> None:
    first, second = torch.arange(2.0), torch.arange(6.0).view(2, 3)
    with PinnedRandom([first, second]) as pinned:
        assert torch.equal(torch.randn(2), first)
        drawn = torch.randn((2, 3), dtype=torch.bfloat16)
    assert drawn.dtype == torch.bfloat16
    assert torch.equal(drawn, second.bfloat16())
    assert sum(pinned.sites.values()) == 2
    assert {site.split(":")[0] for site in pinned.sites} == {_THIS.name}


def test_pinned_random_requires_every_draw_taken() -> None:
    with (
        pytest.raises(AssertionError, match="1 pinned draws were never taken"),
        PinnedRandom([torch.zeros(1)]),
    ):
        pass


def test_definitions_reach_into_compound_statements(tmp_path: Path) -> None:
    _ = (tmp_path / "module.py").write_text(
        "if True:\n"
        "    def guarded():\n"
        "        return 1\n"
        "try:\n"
        "    class Tried:\n"
        "        def method(self):\n"
        "            return 2\n"
        "except ImportError:\n"
        "    pass\n"
        "with open(__file__):\n"
        "    def opened():\n"
        "        return 3\n",
    )
    keys = [d.key for d in definitions(tmp_path, relative="module.py")]
    assert keys == [
        "module.py::guarded",
        "module.py::Tried",
        "module.py::Tried.method",
        "module.py::opened",
    ]


def test_inventory_checks_branches_in_paired_math_helpers(tmp_path: Path) -> None:
    """``rgb2float`` is port code like any ``model/vision_ae`` function."""
    pixel = _PORT_ROOT / "math" / "pixel.py"
    line = _line_of(pixel, text="if unit_interval:")
    reference = _reference(tmp_path, paired=("math/pixel.py::rgb2float",))
    measured = _Measured({str(pixel): {line: (2, 1)}})
    problems = inventory_problems(reference, clone=tmp_path, measured=measured)
    assert problems == [
        f"math/pixel.py::rgb2float:{line} takes 1 of 2 exits: if unit_interval:",
    ]


def test_an_allowlist_entry_silences_only_its_own_definition(tmp_path: Path) -> None:
    """``Upsample`` tests ``if self.with_conv:`` twice; allowing one keeps the other."""
    invae = _PORT_ROOT / "model" / "vision_ae" / "invae.py"
    built = _line_of(invae, text="if self.with_conv:")
    run = _line_of(invae, text="if self.with_conv:", after=built + 1)
    reference = _reference(
        tmp_path,
        paired=(
            "model/vision_ae/invae.py::Upsample.__init__",
            "model/vision_ae/invae.py::Upsample.forward",
        ),
        port_unreached={
            "model/vision_ae/invae.py::Upsample.__init__::if self.with_conv:": "Why.",
        },
    )
    measured = _Measured({str(invae): {built: (2, 1), run: (2, 1)}})
    problems = inventory_problems(reference, clone=tmp_path, measured=measured)
    assert problems == [
        (
            f"model/vision_ae/invae.py::Upsample.forward:{run} takes 1 of 2 exits: "
            "if self.with_conv:"
        ),
    ]


def test_each_comparison_is_measured_on_its_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One model's run cannot satisfy another's inventory."""
    monkeypatch.setattr(
        reference_parity,
        "coverage",
        SimpleNamespace(Coverage=_Coverage),
    )
    first, second = _reference(tmp_path / "a"), _reference(tmp_path / "b")
    _, measured_first = measure(first, run=_run, clone=tmp_path / "a", work=tmp_path)
    _, measured_second = measure(second, run=_run, clone=tmp_path / "b", work=tmp_path)
    assert isinstance(measured_first, _Coverage)
    assert isinstance(measured_second, _Coverage)
    assert measured_first is not measured_second
    assert measured_first.events == measured_second.events == ["start", "run", "stop"]
    assert str(tmp_path / "a" / "ref.py") in measured_first.include
    assert str(tmp_path / "a" / "ref.py") not in measured_second.include


@pytest.mark.cli_git
def test_a_failed_git_command_reports_its_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    (tmp_path / "clone" / ".git").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="not a git repository"):
        _ = clone_upstream(_reference(tmp_path), root=tmp_path / "clone")


def _line_of(path: Path, text: str, *, after: int = 1) -> int:
    """Return the first line at or after ``after`` whose stripped text is ``text``."""
    lines = path.read_text().splitlines()
    return next(i for i in range(after, len(lines) + 1) if lines[i - 1].strip() == text)


def _reference(
    clone: Path,
    *,
    paired: tuple[str, ...] = (),
    port_unreached: Mapping[str, str] = MappingProxyType({}),
) -> Reference:
    """Return a one-function reference in ``clone`` paired with ``paired``."""
    clone.mkdir(parents=True, exist_ok=True)
    _ = (clone / "ref.py").write_text("def f():\n    return 1\n")
    return Reference(
        url="https://example.com/ref.git",
        commit="0" * 40,
        files=("ref.py",),
        paired={"ref.py::f": paired} if paired else {},
        not_ported={} if paired else {"ref.py::f": "Unused here."},
        reference_unreached={},
        port_unreached=port_unreached,
        extra_outputs={},
    )


class _Measured:
    """A measured run whose branches are stated per file."""

    def __init__(self, branches: dict[str, dict[int, tuple[int, int]]]) -> None:
        self._branches = branches

    def get_data(self) -> _Lines:
        return _Lines()

    def branch_stats(self, morf: str) -> dict[int, tuple[int, int]]:
        return self._branches.get(morf, {})


class _Lines:
    """Every file ran every line."""

    def lines(self, filename: str) -> list[int] | None:
        del filename
        return list(range(1, 10_000))


class _Coverage:
    """Records what one measurement was asked to include and when it ran."""

    events: list[str]
    include: list[str]
    running: _Coverage | None = None

    def __init__(self, *, include: list[str], **kwargs: object) -> None:
        del kwargs
        self.include = include
        self.events = []

    def start(self) -> None:
        self.events.append("start")
        type(self).running = self

    def stop(self) -> None:
        self.events.append("stop")
        type(self).running = None


def _run(clone: Path, work: Path) -> Outcome:
    """Mark the running measurement, as a comparison would exercise code under it."""
    del clone, work
    assert _Coverage.running is not None
    _Coverage.running.events.append("run")
    return Outcome(problems=[])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
