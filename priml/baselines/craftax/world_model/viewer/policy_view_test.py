"""Check policy-view bundles: exact frames cut before their map with their facings, and their manifest.

The episode is ``testing``'s: four decisions of the game, played as Python
(``eager``) on the tiny world, a nine-tick sleep among them; a bundle keeps a
prefix of three.
"""

from collections.abc import Generator
from pathlib import Path
from typing import Final

import gzip
import hashlib

import numpy as np
import pytest

from priml.baselines.craftax.eager import eager
from priml.baselines.craftax.game import jit
from priml.baselines.craftax.game.jit import package_digest, platform_key
from priml.baselines.craftax.lib.arrays import int_rows
from priml.baselines.craftax.world_model.archive import Record
from priml.baselines.craftax.world_model.viewer import testing
from priml.baselines.craftax.world_model.viewer.exact import (
    Replayed,
    frame_dtype,
    read_frame,
    read_manifest,
    replay_episode,
)
from priml.baselines.craftax.world_model.viewer.policy_view import (
    Manifest,
    write_bundle,
)


_PREFIX: Final = len(testing.SCRIPT) - 1


@pytest.fixture(scope="module")
def record() -> Record:
    with eager(world=testing.world):
        return testing.record()


@pytest.fixture(scope="module")
def prefix(record: Record) -> Replayed:
    with eager(world=testing.world):
        return replay_episode(record, limit=_PREFIX)


@pytest.fixture(autouse=True)
def tiny() -> Generator[None]:
    with eager(world=testing.world):
        yield


def test_a_bundle_keeps_every_frame_up_to_its_map_and_its_facings(
    tmp_path: Path,
    record: Record,
    prefix: Replayed,
) -> None:
    output = tmp_path / "view"
    manifest = write_bundle(
        prefix,
        output,
        record=record,
        archive=tmp_path,
        episode="val/arm0/w0/shard-000000/3",
    )
    packed = (output / "policy-view.bin.gz").read_bytes()
    raw = gzip.decompress(packed)
    fields = frame_dtype().fields
    assert fields is not None
    # policy_view.js and hud.js read a cut frame at the exact frame's offsets,
    # then its 99 facings.
    assert fields["map"][1] == 885
    assert manifest.record_bytes == 885 + 99
    assert manifest.schema_name == "craftax-policy-view/v2"
    frames = prefix.frames.view(np.uint8).reshape(_PREFIX + 1, 7_858)
    records = np.frombuffer(raw, np.uint8).reshape(_PREFIX + 1, 984)
    assert records[:, :885].tobytes() == frames[:, :885].tobytes()
    assert records[:, 885:].tobytes() == prefix.facings.tobytes()
    assert (manifest.frames_sha256, manifest.gzip_sha256) == (
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(packed).hexdigest(),
    )
    assert read_manifest((output / "manifest.json").read_text(), Manifest) == (manifest)
    with pytest.raises(FileExistsError):
        write_bundle(prefix, output, record=record, archive=tmp_path, episode="")


def test_a_bundle_with_sleep_frames_cuts_them_alike_and_indexes_them(
    tmp_path: Path,
    record: Record,
    prefix: Replayed,
) -> None:
    slept = replay_episode(record, limit=_PREFIX, sleep_stride=4)
    assert len(slept.sleeps)
    output = tmp_path / "view"
    manifest = write_bundle(
        slept,
        output,
        record=record,
        archive=tmp_path,
        episode="val/arm0/w0/shard-000000/3",
        sleep_stride=4,
    )
    packed = (output / "policy-sleep.bin.gz").read_bytes()
    raw = gzip.decompress(packed)
    frames = slept.sleep_frames.view(np.uint8).reshape(len(slept.sleep_frames), 7_858)
    records = np.frombuffer(raw, np.uint8).reshape(len(slept.sleep_frames), 984)
    assert records[:, :885].tobytes() == frames[:, :885].tobytes()
    assert records[:, 885:].tobytes() == slept.sleep_facings.tobytes()
    assert (manifest.sleep_stride, manifest.sleep_frames) == (
        4,
        len(slept.sleep_frames),
    )
    assert manifest.sleeps == int_rows(slept.sleeps)
    assert (manifest.sleep_frames_sha256, manifest.sleep_gzip_sha256) == (
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(packed).hexdigest(),
    )
    # Without sleep frames the bundle is as before.
    plain = write_bundle(
        prefix,
        tmp_path / "plain",
        record=record,
        archive=tmp_path,
        episode="val/arm0/w0/shard-000000/3",
    )
    assert (plain.sleeps, plain.sleep_frames, plain.sleep_stride) == ([], 0, None)
    assert not (tmp_path / "plain" / "policy-sleep.bin.gz").exists()


def test_the_manifest_names_the_record_the_build_and_the_end(
    tmp_path: Path,
    record: Record,
    prefix: Replayed,
) -> None:
    manifest = write_bundle(
        prefix,
        tmp_path / "view",
        record=record,
        archive=tmp_path,
        episode="val/arm0/w0/shard-000000/3",
    )
    receipt = record.receipt
    assert manifest.schema_name == "craftax-policy-view/v2"
    assert (manifest.decisions, manifest.record_decisions) == (_PREFIX, _PREFIX + 1)
    assert (manifest.archive, manifest.episode) == (
        str(tmp_path),
        "val/arm0/w0/shard-000000/3",
    )
    assert (manifest.world_seed, manifest.sampling_seed) == (4, "2")
    assert manifest.initial_state_hash == str(receipt.initial_state_hash)
    assert (manifest.arm, manifest.split) == (receipt.arm, receipt.split)
    assert (
        manifest.actions_sha256
        == hashlib.sha256(record.actions.numpy().tobytes()).hexdigest()
    )
    assert (
        manifest.hashes_sha256
        == hashlib.sha256(record.hashes.numpy().tobytes()).hexdigest()
    )
    assert manifest.game == package_digest(Path(jit.__file__).parent)
    assert manifest.platform == platform_key()
    final = read_frame(prefix.frames, -1)
    assert final.action == 255
    assert manifest.end == (
        final.floor_before,
        final.row,
        final.col,
        final.direction,
    )
    assert (manifest.tick, manifest.score) == (final.tick_before, final.score)
    # A 64-bit seed stays exact as a decimal string in JSON.
    text = (tmp_path / "view" / "manifest.json").read_text()
    assert f'"initialStateHash": "{receipt.initial_state_hash}"' in text


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
