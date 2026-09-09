from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from isaac.scripts.augment_target_state import augment
from ludo_engine.board import BoardGeometry


def _source(tmp_path: Path, *, place: object = ["track", 29]) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "episode_000000").mkdir()
    np.save(root / "episode_000000" / "states.npy", np.arange(16, dtype=np.float32).reshape(2, 8))
    np.save(root / "episode_000000" / "actions.npy", np.zeros((2, 8), dtype=np.float32))
    (root / "episode_000000" / "wrist.mp4").write_bytes(b"video")
    (root / "raw_manifest.jsonl").write_text(
        json.dumps({"episode_index": 0, "place": place}) + "\n"
    )
    return root


def test_augment_appends_target_and_preserves_action(tmp_path: Path) -> None:
    raw = _source(tmp_path)
    out = tmp_path / "augmented"
    augment(raw, out, BoardGeometry(0.34, 0.15, 0.0))
    states = np.load(out / "episode_000000" / "states.npy")
    actions = np.load(out / "episode_000000" / "actions.npy")
    expected = np.asarray(BoardGeometry(0.34, 0.15, 0.0).world_xy("red", ("track", 29)))
    assert states.shape == (2, 10)
    np.testing.assert_allclose(states[:, 8:], np.broadcast_to(expected, (2, 2)))
    assert actions.shape == (2, 8)
    assert (out / "episode_000000" / "wrist.mp4").read_bytes() == b"video"
    record = json.loads((out / "raw_manifest.jsonl").read_text())
    assert record["state_dim"] == 10
    assert record["target_xy"] == pytest.approx(expected.tolist())


def test_augment_rejects_nonempty_output(tmp_path: Path) -> None:
    raw = _source(tmp_path)
    out = tmp_path / "augmented"
    out.mkdir()
    (out / "sentinel").write_text("keep")
    with pytest.raises(ValueError, match="output must be empty"):
        augment(raw, out, BoardGeometry())


def test_augment_rejects_wrong_state_width(tmp_path: Path) -> None:
    raw = _source(tmp_path)
    np.save(raw / "episode_000000" / "states.npy", np.zeros((2, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="states must have shape"):
        augment(raw, tmp_path / "augmented", BoardGeometry())
