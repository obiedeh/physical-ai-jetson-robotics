"""Decode exact dataset rows on the server for offline, synchronized review."""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def confined_file(root: Path, relative: str) -> Path:
    """Resolve a server-selected artifact without following any symlink outside its root."""
    base = root.resolve()
    candidate = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("artifact path is not relative to its dataset")
    if base not in candidate.resolve().parents:
        raise ValueError("artifact path escapes its dataset")
    for part in (candidate, *candidate.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("artifact paths cannot contain symlinks")
    if not candidate.is_file():
        raise FileNotFoundError("recorded artifact is unavailable")
    return candidate


def encode_rgb(image: Any) -> bytes:
    """Encode one already captured RGB image, without touching a camera source."""
    import cv2

    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("preview requires an RGB uint8 image")
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("image encoding failed")
    return bytes(encoded)


def _plain(value: Any) -> Any:
    """Normalize tensor-backed scalar and vector fields for the browser JSON protocol."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_dataset(repo_id: str, root: Path, episode: int) -> Any:
    """Force upstream's offline mode so incomplete local recordings cannot trigger downloads."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from huggingface_hub import constants
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    constants.HF_HUB_OFFLINE = True
    return LeRobotDataset(
        repo_id, root=root, episodes=[episode], video_backend="pyav", download_videos=False,
    )


class EpisodePlayback:
    """Serve synchronized frame pairs with bounded decoded rows and JPEG memory."""

    def __init__(
        self, *, max_frames: int = 24,
        reader: Callable[[str, Path, int], Any] = _load_dataset,
    ) -> None:
        """Set a finite cache limit and optionally inject an offline fake dataset reader."""
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError("playback cache size must be a positive integer")
        self.max_frames = max_frames
        self.reader = reader
        self._lock = threading.RLock()
        self._identity: tuple[str, str, int, int] | None = None
        self._dataset: Any = None
        self._indices: list[int] = []
        self._cache: OrderedDict[int, tuple[dict[str, Any], dict[str, bytes]]] = OrderedDict()

    def frame(
        self, root: Path, repo_id: str, episode: int, index: int,
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        """Return an exact stored row and both camera JPEGs at the same local frame index."""
        if type(episode) is not int or episode < 0 or type(index) is not int or index < 0:
            raise ValueError("episode and frame indices must be nonnegative integers")
        metadata = confined_file(root, "physical_episode_metadata.jsonl")
        if (root / ".recording_transaction.json").exists():
            raise RuntimeError("dataset save or blocked recovery prevents playback")
        identity = (str(root.resolve()), repo_id, episode, metadata.stat().st_mtime_ns)
        with self._lock:
            if identity != self._identity:
                if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
                    raise ValueError("dataset playback refuses symlinked metadata or media")
                self._dataset = self.reader(repo_id, root, episode)
                self._indices = [
                    i for i, row in enumerate(self._dataset.hf_dataset)
                    if int(row["episode_index"]) == episode
                ]
                self._cache.clear()
                self._identity = identity
            if index >= len(self._indices):
                raise IndexError("frame index is outside the recorded episode")
            if index not in self._cache:
                row = self._dataset[self._indices[index]]
                fields = {
                    key: _plain(value) for key, value in row.items()
                    if not key.startswith("observation.images.")
                }
                fields.update(frame_index=index, frame_count=len(self._indices))
                images = {}
                for name in ("wrist", "front"):
                    pixels = row[f"observation.images.{name}"]
                    if hasattr(pixels, "detach"):
                        pixels = pixels.detach().cpu().numpy()
                    pixels = np.asarray(pixels)
                    if pixels.shape[0] == 3 and pixels.ndim == 3:
                        pixels = np.moveaxis(pixels, 0, -1)
                    if np.issubdtype(pixels.dtype, np.floating):
                        pixels = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
                    images[name] = encode_rgb(pixels)
                self._cache[index] = fields, images
                while len(self._cache) > self.max_frames:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(index)
            return self._cache[index]

    def final_still(self, root: Path, episode: int) -> bytes:
        """Read the native final still identified by this episode's authoritative sidecar."""
        records = confined_file(root, "physical_episode_metadata.jsonl")
        for line in records.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["episode_index"] == episode:
                # Absolute saved paths survive neither trash nor restore; preserve their
                # recorder-owned final_stills suffix instead of trusting arbitrary paths.
                saved = Path(record["final_still"])
                if saved.parent.name != "final_stills":
                    raise ValueError("episode still is not recorder-owned")
                return confined_file(root, f"final_stills/{saved.name}").read_bytes()
        raise FileNotFoundError("episode has no final still record")
