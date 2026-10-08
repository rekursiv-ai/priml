"""Check model bundles: the viewer's frame layout, annotations, and payload files."""

from pathlib import Path
from typing import Final

import dataclasses
import gzip
import hashlib
import math
import shutil
import struct
import subprocess

import pytest
import torch

from priml.baselines.craftax.world_model.batch import Segment
from priml.baselines.craftax.world_model.dream import Rollout
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.session import Mark, Stream
from priml.baselines.craftax.world_model.viewer.bundle import (
    Manifest,
    annotations,
    frame_records,
    stream_of,
    token_records,
    write_bundle,
)
from priml.baselines.craftax.world_model.viewer.exact import read_manifest


RECORD: Final = 7858
_CWD: Final = Path(__file__).resolve().parent


def aux_row(**values: int) -> torch.Tensor:
    """Return one frame's aux values: attributes 1, the rest 0 unless given."""
    merged = {"dexterity": 1, "strength": 1, "intelligence": 1} | values
    names = craftax_schema().scalar_names
    return torch.tensor([merged.get(name, 0) for name in names], dtype=torch.int16)


def two_decisions() -> Stream:
    """Return a stream whose second decision ends the episode."""
    cells = torch.randint(0, 2, (3, 99, 8), generator=torch.Generator().manual_seed(0))
    first = aux_row(
        wood=5, health=180, strength=2, mana=4, facing_down=1, floor=1, food=3,
        drink=4, energy=5, sapling=2, pickaxe=2, armour_2=1, potion_cyan=7, books=2,
        learned_iceball=1, sword_enchantment=2, sapphire=3, ruby=6, bow=1, arrows=9,
    )  # fmt: skip
    second = aux_row(health=160, mana=3, floor=2, facing_left=1, sleeping=1)
    return Stream(
        starts_episode=True,
        cells=cells.to(torch.uint8),
        aux=torch.stack([first, second, aux_row(health=180)]),
        frame_marks=torch.zeros(3, 150, dtype=torch.uint8),
        frame_logp=torch.zeros(3, 150),
        action=torch.tensor([3, 5], dtype=torch.uint8),
        reward=torch.tensor([2, 1], dtype=torch.int16),
        done=torch.tensor([False, True]),
        decision_marks=torch.tensor(
            [[Mark.FORCED, Mark.MODEL, Mark.MODEL], [Mark.DATA] * 3],
            dtype=torch.uint8,
        ),
        decision_logp=torch.tensor(
            [[-0.5, math.log(0.25), math.log(0.9)], [0.0, 0.0, 0.0]],
        ),
    )


def unpack(records: bytes, index: int, fmt: str, offset: int) -> tuple[object, ...]:
    """Read little-endian ``fmt`` at ``offset`` of frame ``index``."""
    return struct.unpack_from("<" + fmt, records, index * RECORD + offset)


def test_frames_fill_the_policy_view_layout_from_tokens() -> None:
    stream = two_decisions()
    records = frame_records(stream)
    assert len(records) == 3 * RECORD
    step, tick_before, tick_after = unpack(records, 0, "3I", 0)
    assert (step, tick_before, tick_after) == (0, 0, 0)
    head = unpack(records, 0, "8BH", 12)
    # `action`, floor before/after, row, col, direction (facing down), sleeping,
    # terminal, score.
    assert head == (3, 1, 2, 0, 0, 4, 0, 0, 2)
    assert unpack(records, 0, "2f2h", 22) == (9.0, 8.0, 4, 3)
    observation = records[34 : 34 + 792]
    assert observation == stream.cells[0].numpy().tobytes()
    # Maximum health 8 + 2, mana 6 + 3, need 7 + 2; food, drink, energy; no
    # achievement count.
    assert unpack(records, 0, "7B", 826) == (10, 9, 9, 3, 4, 5, 0)
    inventory = unpack(records, 0, "24H", 833)
    assert inventory[:17] == (5, 0, 0, 0, 0, 2, 2, 0, 1, 9, 0, 0, 1, 0, 0, 6, 3)
    assert inventory[17:] == (0, 0, 0, 0, 7, 0, 2)
    assert unpack(records, 0, "4B", 881) == (0, 1, 2, 0)
    assert records[885:RECORD] == bytes(RECORD - 885)


def test_terminal_decision_keeps_its_own_after_values_and_score() -> None:
    records = frame_records(two_decisions())
    head = unpack(records, 1, "8BH", 12)
    assert head == (5, 2, 2, 0, 0, 1, 1, 1, 3)
    assert unpack(records, 1, "2f2h", 22) == (8.0, 8.0, 3, 3)
    # The final frame starts the next episode yet carries the last decision's
    # score and terminal flag, as an exact replay's final frame does.
    assert unpack(records, 2, "8BH", 12) == (255, 0, 0, 0, 0, 0, 0, 1, 3)
    assert frame_records(two_decisions(), start=1) == records[RECORD:]


def test_score_restarts_after_an_episode_ends_and_never_goes_negative() -> None:
    stream = two_decisions()
    longer = Stream(
        starts_episode=True,
        cells=torch.cat([stream.cells, stream.cells[:2]]),
        aux=torch.cat([stream.aux, stream.aux[:2]]),
        frame_marks=torch.zeros(5, 150, dtype=torch.uint8),
        frame_logp=torch.zeros(5, 150),
        action=torch.tensor([3, 5, 1, 2], dtype=torch.uint8),
        reward=torch.tensor([2, 1, 4, -1], dtype=torch.int16),
        done=torch.tensor([False, True, False, False]),
        decision_marks=torch.zeros(4, 3, dtype=torch.uint8),
        decision_logp=torch.zeros(4, 3),
    )
    records = frame_records(longer)
    scores = [unpack(records, i, "H", 20)[0] for i in range(5)]
    assert scores == [2, 3, 4, 3, 3]
    negative = dataclasses.replace(
        longer,
        reward=torch.tensor([-1, 0, 0, 0], dtype=torch.int16),
    )
    assert unpack(frame_records(negative), 0, "H", 20) == (0,)


def test_annotations_mark_sources_probabilities_and_invalid_cells() -> None:
    stream = two_decisions()
    stream.cells[0, 7] = torch.tensor([2, 1, 1, 0, 0, 0, 0, 0], dtype=torch.uint8)
    stream.cells[1, 7] = torch.tensor([0, 0, 0, 0, 3, 0, 0, 0], dtype=torch.uint8)
    stream.frame_marks[2] = Mark.DATA
    # A frame's source is its least mark, so an override of slot 0 leaves it data.
    stream.frame_marks[2, [0, 99 + 22]] = Mark.OVERRIDE
    notes = annotations(stream)
    assert notes.action_marks == ["forced", "data"]
    assert notes.frame_source == ["model", "model", "data"]
    assert notes.overrides == [[], [], [0, 121]]
    assert 7 in notes.invalid[1]
    assert 7 not in notes.invalid[0]
    reward_prob, done_prob = notes.reward_prob, notes.done_prob
    assert reward_prob[1] is None
    assert done_prob[1] is None
    assert reward_prob[0] == pytest.approx(0.25)
    # The first decision did not end the episode with probability 0.9.
    assert done_prob[0] == pytest.approx(0.1)
    assert notes.reward == [2, 1]
    assert notes.done == [False, True]
    tail = annotations(stream, start=1)
    assert tail.invalid == notes.invalid[1:]
    assert tail.reward == [1]


def test_stream_of_rollout_marks_a_real_first_frame_as_data() -> None:
    frames, decisions = 3, 2
    generator = torch.Generator().manual_seed(3)
    rollout = Rollout(
        cells=torch.randint(0, 2, (2, frames, 99, 8), generator=generator).to(
            torch.uint8,
        ),
        aux=torch.zeros(2, frames, 51, dtype=torch.int16),
        starts=torch.tensor([[True, False, False], [False, False, True]]),
        frame_logp=-torch.rand(2, frames, 150, generator=generator),
        invalid=torch.zeros(2, frames, 99, dtype=torch.bool),
        action=torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8),
        reward=torch.tensor([[0, 1], [2, 0]], dtype=torch.int16),
        done=torch.tensor([[False, False], [False, True]]),
        action_logp=-torch.rand(2, decisions, generator=generator),
        reward_logp=-torch.rand(2, decisions, generator=generator),
        done_logp=-torch.rand(2, decisions, generator=generator),
    )
    stream = stream_of(rollout, row=1, action_mark=Mark.FORCED)
    assert not stream.starts_episode
    assert torch.equal(stream.cells, rollout.cells[1])
    assert stream.frame_marks[:, 0].tolist() == [Mark.DATA, Mark.MODEL, Mark.MODEL]
    assert stream.decision_marks[:, 0].tolist() == [Mark.FORCED] * 2
    assert bool((stream.decision_marks[:, 1:] == Mark.MODEL).all())
    assert torch.equal(stream.decision_logp[:, 2], rollout.done_logp[1])
    assert torch.equal(stream.frame_logp, rollout.frame_logp[1])
    assert stream_of(rollout, row=0).frame_marks[0, 0] == Mark.MODEL


def test_token_records_hold_cells_then_little_endian_aux() -> None:
    stream = two_decisions()
    payload = token_records(stream.cells, stream.aux)
    assert len(payload) == 3 * 894
    assert payload[894 : 894 + 792] == stream.cells[1].numpy().tobytes()
    assert struct.unpack_from("<51h", payload, 894 + 792) == tuple(
        stream.aux[1].tolist(),
    )


def test_write_bundle_records_payload_digests_and_absent_fields(
    tmp_path: Path,
) -> None:
    stream = two_decisions()
    real = stream.cells.clone()
    real[2, 5, 0] = 9
    reference = Segment(
        cells=real,
        aux=stream.aux,
        actions=stream.action,
        reward=stream.reward,
        done=stream.done,
        starts_episode=True,
    )
    output = tmp_path / "bundle"
    write_bundle(stream, output, title="Dream", reference=reference)
    manifest = read_manifest((output / "manifest.json").read_text(), Manifest)
    assert (manifest.schema_name, manifest.source) == ("craftax-model-game/v1", "model")
    assert {"tick", "map", "mobs", "position"} <= set(manifest.absent)
    assert (manifest.frames, manifest.actions, manifest.score) == (3, 2, 3)
    assert (manifest.tick, manifest.achievements) == (0, 0)
    assert manifest.annotations == annotations(stream)
    for name, raw, packed in (
        ("frames", manifest.frames_sha256, manifest.gzip_sha256),
        ("tokens", manifest.tokens_sha256, manifest.tokens_gzip_sha256),
        ("reference", manifest.reference_sha256, manifest.reference_gzip_sha256),
    ):
        compressed = (output / f"{name}.bin.gz").read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == packed
        assert hashlib.sha256(gzip.decompress(compressed)).hexdigest() == raw
    frames = gzip.decompress((output / "frames.bin.gz").read_bytes())
    assert frames == frame_records(stream)
    real_tokens = gzip.decompress((output / "reference.bin.gz").read_bytes())
    assert real_tokens == token_records(real, stream.aux)
    with pytest.raises(FileExistsError):
        write_bundle(stream, output, title="Again")


@pytest.mark.cli_node
@pytest.mark.skipif(shutil.which("node") is None, reason="The viewer builds with Node.")
def test_viewer_builder_accepts_a_model_bundle(tmp_path: Path) -> None:
    stream = two_decisions()
    write_bundle(stream, tmp_path / "bundle", title="Dream")
    write_bundle(stream, tmp_path / "second", title="Second")
    page = tmp_path / "page.html"
    node = shutil.which("node")
    assert node is not None
    command = [node, str(_CWD / "games.mjs"), "build", "--output", str(page)]
    command += [str(tmp_path / "bundle"), str(tmp_path / "second")]
    subprocess.run(command, check=True, capture_output=True, text=True)  # noqa: S603 -- Fixed argv: resolved node, our games.mjs.
    html = page.read_text()
    assert '"kind":"model"' in html
    assert '"tokensGzip":"' in html
    (tmp_path / "second" / "tokens.bin.gz").write_bytes(gzip.compress(b"x"))
    broken = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603 -- Fixed argv: resolved node, our games.mjs.
    assert broken.returncode != 0
    assert "Tokens payload hash mismatch" in broken.stderr


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
