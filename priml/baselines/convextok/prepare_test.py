"""Preparation must turn raw shards into upstream ConvexTok's tokenizer, end to end.

On the eight-document fixture corpus the prepared tokenizer's learned pieces must be
upstream's ``det`` vocabulary, and held-out text must segment into upstream's token
strings.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import json

from configgle import Fig
from pyarrow import parquet
from tokenizers import Tokenizer, pre_tokenizers
from torch import Tensor

import pyarrow as pa
import pytest
import torch

from priml.baselines.convextok import prepare
from priml.baselines.convextok.candidates import count_candidates
from priml.baselines.convextok.prepare import (
    ConvexTokPreparation,
    donor_convextok16k,
)
from priml.baselines.convextok.presolver.presolve import Presolved, presolve
from priml.baselines.convextok.pretokens import count_pretokens
from priml.baselines.convextok.program import LinearProgram
from priml.baselines.nanochat.scripts.prepare_data import donor_unigram16k
from priml.baselines.nanochat.scripts.prepare_tokenizer import document_rows
from priml.lib.custom_json import DictCodec, IntCodec, ListCodec
from priml.paths import validated_output_path


_CWD: Final = Path(__file__).resolve().parent


# The whole preparation, parquet to tokenizer.json: 0.3 s even with the kernels
# interpreted, past the unit tier's 100 ms.
@pytest.mark.compute_large_fixture
def test_fixture_corpus_yields_upstream_tokenizer(tmp_path: Path) -> None:
    corpus = _read_json("corpus.json")
    config = _fixture_config(tmp_path)
    config.make().build()

    tokenizer = Tokenizer.from_file(str(config.working_dir / "tokenizer.json"))
    learned = _learned_pieces(tokenizer)
    alphabet = set(pre_tokenizers.ByteLevel.alphabet())
    golden = set(
        ListCodec.coerce(_read_json("vocab.json").get("det"), str, default=None),
    )
    assert learned == golden - alphabet
    held_out = [
        tokenizer.encode(text, add_special_tokens=False).tokens
        for text in ListCodec.coerce(corpus.get("heldout"), str, default=None)
    ]
    assert held_out == [
        ListCodec.coerce(row, str, default=None)
        for row in ListCodec.coerce(
            _read_json("heldout_tokens.json").get("tokens"),
            object,
            default=None,
        )
    ]


# The same pipeline as above, with a stand-in solver instead of PDLP and presolve.
def test_any_solver_fills_the_solver_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every variable at 1 ties every indicator, so rounding keeps candidate order."""

    def patched_count_pretokens(
        texts: Sequence[str],
        *,
        split_pattern: str,
        num_workers: int,
    ) -> dict[str, int]:
        if num_workers != 1:
            pytest.fail(f"count_pretokens num_workers={num_workers}")
        return count_pretokens(
            texts,
            split_pattern=split_pattern,
            num_workers=num_workers,
        )

    def patched_count_candidates(
        pretokens: dict[str, int],
        *,
        num_workers: int,
    ) -> dict[str, int]:
        if num_workers != 1:
            pytest.fail(f"count_candidates num_workers={num_workers}")
        return count_candidates(pretokens, num_workers=num_workers)

    monkeypatch.setattr(prepare, "count_pretokens", patched_count_pretokens)
    monkeypatch.setattr(prepare, "count_candidates", patched_count_candidates)
    config = _fixture_config(tmp_path)
    config.working_dir = tmp_path / "new-parent" / "nested" / "convextok"
    config.device = "meta"
    config.presolve = None
    config.solver = _AllOnes.Config()
    original_solver = _AllOnes.__call__
    solver_devices: list[torch.device] = []

    def inspect_solver(self: _AllOnes, program: LinearProgram, /) -> _Solution:
        solver_devices.append(program.objective.device)
        return original_solver(self, program)

    monkeypatch.setattr(_AllOnes, "__call__", inspect_solver)

    def check_protected_output(path: Path, *, protected: Sequence[Path] = ()) -> Path:
        assert list(protected) == [config.raw_dir]
        return validated_output_path(path, protected=protected)

    monkeypatch.setattr(prepare, "validated_output_path", check_protected_output)
    shards: list[tuple[Path, int]] = []

    def record_shard(path: Path, *, shard: int) -> Iterator[tuple[str, str]]:
        shards.append((path, shard))
        return document_rows(path, shard=shard)

    monkeypatch.setattr(prepare, "document_rows", record_shard)
    rounding_calls: list[tuple[Tensor, Sequence[str], int]] = []

    def record_rounding(
        indicators: Tensor,
        candidates: Sequence[str],
        /,
        *,
        budget: int,
    ) -> Tensor:
        assert indicators.dtype == torch.float64
        assert indicators.shape == (len(candidates),)
        rounding_calls.append((indicators, candidates, budget))
        return torch.arange(min(budget, len(candidates)))

    config.rounding = record_rounding
    config.make().build()

    tokenizer = Tokenizer.from_file(str(config.working_dir / "tokenizer.json"))
    candidates = ListCodec.coerce(
        _read_json("candidates.json").get("tokens"),
        str,
        default=None,
    )
    budget = IntCodec.coerce(_read_json("corpus.json").get("budget"), default=None)
    assert _learned_pieces(tokenizer) == set(candidates[:budget])
    assert shards == [(config.raw_dir / "shard_00000.parquet", 0)]
    assert solver_devices == [torch.device("meta")]
    assert len(rounding_calls) == 1
    assert rounding_calls[0][1] == candidates
    assert rounding_calls[0][2] == budget
    assert (tmp_path / "new-parent").is_dir()
    with pytest.raises(FileExistsError) as error:
        config.make().build()
    assert str(error.value) == f"[Errno 17] File exists: '{config.working_dir}'"


def test_build_forwards_the_presolved_program_and_solution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture_config(tmp_path)
    config.working_dir = tmp_path / "presolved" / "convextok"
    config.device = "meta"
    config.solver = _AllOnes.Config()
    original_presolve = presolve
    presolve_calls: list[tuple[LinearProgram, torch.device]] = []

    def record_presolve(program: LinearProgram, device: torch.device) -> Presolved:
        presolve_calls.append((program, device))
        return original_presolve(program, torch.device("cpu"))

    config.presolve = record_presolve
    original_solver = _AllOnes.__call__
    solver_devices: list[torch.device] = []

    def inspect_solver(self: _AllOnes, program: LinearProgram, /) -> _Solution:
        solver_devices.append(program.objective.device)
        return original_solver(self, program)

    monkeypatch.setattr(_AllOnes, "__call__", inspect_solver)
    rounding_dtypes: list[torch.dtype] = []

    def record_rounding(
        indicators: Tensor,
        candidates: Sequence[str],
        /,
        *,
        budget: int,
    ) -> Tensor:
        assert indicators.shape == (len(candidates),)
        rounding_dtypes.append(indicators.dtype)
        return torch.arange(min(budget, len(candidates)))

    config.rounding = record_rounding
    config.make().build()

    assert len(presolve_calls) == 1
    assert presolve_calls[0][1] == torch.device("meta")
    assert solver_devices == [torch.device("meta")]
    assert rounding_dtypes == [torch.float64]


def test_vocabulary_must_leave_room_for_every_byte() -> None:
    config = ConvexTokPreparation.Config()
    config.vocab_size = 256
    with pytest.raises(
        ValueError,
        match=r"\AThe ordinary vocabulary must contain every byte\.\Z",
    ):
        config.make()

    config.vocab_size = 266
    preparation = config.make()
    assert preparation.config.vocab_size == 266


def test_donor_recipe_fits_convextok_on_its_raw_shards(tmp_path: Path) -> None:
    """The baseline's recipe with ConvexTok in its tokenizer slot and its ten IDs."""
    config = donor_convextok16k()
    config.working_dir = tmp_path
    finalized = config.copy_tree().finalize()
    tokenizer = finalized.tokenizer
    assert isinstance(tokenizer, ConvexTokPreparation.Config)
    assert tokenizer.raw_dir == tmp_path / "raw"
    assert tokenizer.working_dir == tmp_path / "convextok16k"
    assert finalized.rows.tokenizer.path == tmp_path / "convextok16k/tokenizer.json"
    assert finalized.rows.tokenizer.reserved_count == 10
    unigram = donor_unigram16k()
    unigram.working_dir = tmp_path
    assert finalized.corpus == unigram.copy_tree().finalize().corpus


def _fixture_config(tmp_path: Path) -> ConvexTokPreparation.Config:
    """Write the fixture corpus as one raw shard and point a CPU preparation at it."""
    corpus = _read_json("corpus.json")
    raw = tmp_path / "raw"
    raw.mkdir()
    parquet.write_table(
        pa.table(
            {"text": ListCodec.coerce(corpus.get("texts"), str, default=None)},
        ),
        raw / "shard_00000.parquet",
    )
    config = ConvexTokPreparation.Config()
    config.raw_dir = raw
    config.shard_indices = [0]
    config.working_dir = tmp_path / "convextok"
    config.vocab_size = IntCodec.coerce(corpus.get("vocab_size"), default=None)
    config.num_workers = 1
    config.device = "cpu"
    return config


def _learned_pieces(tokenizer: Tokenizer) -> set[str | None]:
    """Return the tokenizer's pieces after its 256 bytes."""
    return {
        tokenizer.id_to_token(index) for index in range(256, tokenizer.get_vocab_size())
    }


def _read_json(name: str) -> dict[str, object]:
    raw = cast(object, json.loads((_CWD / "testdata" / name).read_text()))
    return dict(DictCodec.coerce(raw, default=None))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Solution:
    primal: Tensor


class _AllOnes:
    """A stand-in solver that is not PDLP: every variable at its upper bound, 1."""

    class Config(Fig["_AllOnes"]):
        """No settings."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, program: LinearProgram, /) -> _Solution:
        return _Solution(primal=torch.ones(program.num_columns, dtype=torch.float32))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
