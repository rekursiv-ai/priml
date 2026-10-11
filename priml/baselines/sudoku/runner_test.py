from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hashlib
import json

import numpy as np
import pytest
import torch

from priml.baselines.sudoku.runner import (
    checkpoint_payload,
    eval_model,
    file_sha256,
    fill_template,
    load_eval_weights,
    open_npz,
    prepared,
    release,
    require_population,
    to_device,
    write_json,
)
from priml.baselines.sudoku.testing import tiny_recursion_config


if TYPE_CHECKING:
    from priml.baselines.sudoku.puzzle_data import PuzzleBatch


def test_fill_template_fills_fields_and_keeps_literal_paths(tmp_path: Path) -> None:
    assert (
        fill_template(
            "/runs/{experiment_name}/x-{index}",
            base_dir=tmp_path,
            experiment_name="e",
            index=2,
        ).name
        == "x-2"
    )
    literal = Path("/abs/metrics.json")
    assert fill_template(literal, base_dir=None, experiment_name="unused") is literal


def test_eval_weights_overlay_the_ema_shadow_on_a_full_checkpoint(
    tmp_path: Path,
) -> None:
    live = {"w": torch.zeros(2, 3), "b": torch.ones(3)}
    shadow = {"w": torch.full((2, 3), 5.0)}
    full = tmp_path / "full.pt"
    ema = {"shadow_params": shadow, "shadow_buffers": {}, "global_step": 4}
    torch.save({"step": {"model": live, "ema": ema}}, full)
    weights = load_eval_weights(full)
    assert weights.keys() == {"w", "b"}
    assert torch.equal(weights["w"], shadow["w"])
    assert torch.equal(weights["b"], live["b"])
    assert torch.equal(checkpoint_payload(full, "cpu")["model"]["w"], live["w"])
    # An EMA that has not updated yet saves only its counters: the live weights load.
    torch.save({"step": {"model": live, "ema": {"global_step": 0}}}, full)
    assert "ema" not in checkpoint_payload(full, "cpu")
    assert torch.equal(load_eval_weights(full)["w"], live["w"])


def test_eval_weights_load_a_model_only_file_as_is(tmp_path: Path) -> None:
    live = {"w": torch.arange(6.0).reshape(2, 3)}
    model_only = tmp_path / "model.pt"
    torch.save({"model": live}, model_only)
    assert "ema" not in checkpoint_payload(model_only, "cpu")
    assert torch.equal(load_eval_weights(str(model_only))["w"], live["w"])


def test_eval_model_loads_the_checkpoint_into_a_frozen_model(tmp_path: Path) -> None:
    config = tiny_recursion_config()
    torch.manual_seed(0)
    source = config.make()
    path = tmp_path / "model.pt"
    torch.save({"model": source.state_dict()}, path)
    torch.manual_seed(1)
    model = eval_model(config, path, torch.device("cpu"))
    assert not model.training
    for name, tensor in source.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor)


def test_require_population_rejects_a_short_pass() -> None:
    require_population(3, 3, label="pass")
    with pytest.raises(RuntimeError, match=r"pass evaluated 2 puzzles.*is 3"):
        require_population(2, 3, label="pass")


def test_write_json_creates_the_parent_and_sorts_keys(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "metrics.json"
    write_json(path, {"z": 1, "a": [2, 3]})
    assert path.read_text() == json.dumps({"a": [2, 3], "z": 1}, indent=2) + "\n"
    assert list(path.parent.iterdir()) == [path]
    nested = prepared(tmp_path / "c" / "d.npz")
    assert nested.parent.is_dir()
    assert not nested.exists()


def test_npz_and_digest_helpers_read_what_numpy_wrote(tmp_path: Path) -> None:
    path = tmp_path / "x.npz"
    np.savez(path, a=np.arange(3, dtype=np.int16))
    with open_npz(path) as archive:
        assert archive["a"].tolist() == [0, 1, 2]
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_to_device_moves_each_tensor_and_keeps_the_count() -> None:
    batch: PuzzleBatch = {
        "media": torch.ones(2, 3, dtype=torch.int32),
        "label": torch.full((2, 3), 2, dtype=torch.int32),
        "valid_count": 1,
        "puzzle_identifiers": torch.zeros(2, dtype=torch.int32),
    }
    moved = to_device(batch, torch.device("meta"))
    assert moved["valid_count"] == 1
    assert {moved[key].device.type for key in ("media", "label")} == {"meta"}
    assert moved["puzzle_identifiers"].device.type == "meta"
    release()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
