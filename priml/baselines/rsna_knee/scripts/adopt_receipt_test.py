"""Verify transferred pixels against the original receipt before adopting a root."""

from __future__ import annotations

from pathlib import Path

import csv
import hashlib
import json
import zlib

from treekle import json_loads, to_plain

import pytest

from priml.baselines.rsna_knee.scripts.adopt_receipt import adopt_receipt
from priml.baselines.rsna_knee.train_loop import (
    ORIGINAL_RECEIPT,
    RECEIPT,
    SELECTED_FILES,
    SERIES_FILES,
    TRANSFER,
    VERIFIED_FILES,
    verify_inventory,
)


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "missing",
        "changed",
        "extra",
        "manifest",
        "receipt",
        "crc",
        "bytes",
        "series",
    ],
)
def test_transfer_requires_complete_original_evidence_and_all_pixels(
    tmp_path: Path,
    fault: str,
) -> None:
    config = adopt_receipt()
    config.base_dir = None
    config.input_dir = tmp_path / "raw"
    config.source_dir = tmp_path / "original"
    config.working_dir = tmp_path / "adopted"
    config.source_dir.mkdir()
    rows: list[list[str | int]] = []
    for name in (
        "train_images/study/series/a.dcm",
        "train_images/study/series/b.dcm",
        "test_images/other/series/c.dcm",
    ):
        path = config.input_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = name.encode()
        path.write_bytes(payload)
        rows.append(
            [
                name,
                len(payload),
                zlib.crc32(payload),
                hashlib.sha256(payload).hexdigest(),
            ],
        )
    if fault == "crc":
        rows[-1][2] = 0
    for filename in (SELECTED_FILES, VERIFIED_FILES):
        with (config.source_dir / filename).open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "bytes", "crc32", "sha256"])
            writer.writerows(rows)
    (config.source_dir / SERIES_FILES).write_text(
        "series,files\ntrain_images/study/series,2\ntest_images/other/series,1\n",
    )
    receipt = {
        "complete": fault != "receipt",
        "input_dir": "/original/host/raw",
        "files": 3,
        "bytes": sum(int(row[1]) for row in rows) + (fault == "bytes"),
        "series": 2 + (fault == "series"),
        "inventory_sha256": hashlib.sha256(
            (config.source_dir / SELECTED_FILES).read_bytes(),
        ).hexdigest(),
        "verified_sha256": hashlib.sha256(
            (config.source_dir / VERIFIED_FILES).read_bytes(),
        ).hexdigest(),
    }
    original = json.dumps(to_plain(receipt)).encode()
    (config.source_dir / RECEIPT).write_bytes(original)
    unrelated_test = config.input_dir / "test_images/other/series/c.dcm"
    if fault == "missing":
        unrelated_test.unlink()
    elif fault == "changed":
        unrelated_test.write_bytes(b"wrong")
    elif fault == "extra":
        (unrelated_test.parent / "extra.dcm").write_bytes(b"extra")
    elif fault == "manifest":
        (config.source_dir / VERIFIED_FILES).write_text("modified")
    if fault == "none":
        config.make().run()
        verify_inventory(
            config.input_dir,
            acquisition_dir=config.working_dir,
            selected={Path("train_images/study/series")},
        )
        assert (config.working_dir / ORIGINAL_RECEIPT).read_bytes() == original
        adopted = json_loads(
            (config.working_dir / RECEIPT).read_text(),
            dict[str, object],
        )
        assert adopted["input_dir"] == str(config.input_dir.resolve())
        assert (
            adopted["transfer_source_receipt_sha256"]
            == hashlib.sha256(original).hexdigest()
        )
        provenance = json_loads(
            (config.working_dir / TRANSFER).read_text(),
            dict[str, object],
        )
        assert provenance["source_input_dir"] == "/original/host/raw"
        assert provenance["verified_files"] == 3
        with pytest.raises(FileExistsError):
            config.make().run()
    else:
        with pytest.raises((ValueError, FileNotFoundError)) as caught:
            config.make().run()
        if fault == "receipt":
            assert str(caught.value) == "DICOM acquisition is incomplete."
        assert not (config.working_dir / RECEIPT).exists()
    assert (config.source_dir / RECEIPT).read_bytes() == original
    with pytest.raises(ValueError, match="configuration fields"):
        config.make().run("unexpected")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
