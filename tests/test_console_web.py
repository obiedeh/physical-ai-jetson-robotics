"""Exercise loopback request guards and frame-exact review without device access."""

from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from synria_lerobot.console.playback import EpisodePlayback, confined_file
from synria_lerobot.console.server import MAX_BODY, make_server


class FakeService:
    """Model server-owned state independently of browser request lifetimes."""

    def __init__(self) -> None:
        """Keep mutations visible so refused requests can prove they changed nothing."""
        self.commands: list[Any] = []

    def read(self, route: list[str], query: dict[str, list[str]]) -> Any:
        """Expose inert state and refuse routes that would otherwise look like GET commands."""
        if route != ["state"]:
            raise KeyError("not a read route")
        return {"commands": self.commands, "capture": "recording"}

    def mutate(self, route: list[str], payload: dict[str, Any]) -> Any:
        """Simulate a mutation only after the HTTP guard has admitted it."""
        if route == ["refuse"]:
            raise RuntimeError("preflight refused")
        self.commands.append([route, payload])
        return {"accepted": True}

    def media(self, route: list[str], query: dict[str, list[str]]) -> bytes:
        """Return fixed image bytes without invoking any capture routine."""
        return b"fake-latest-jpeg"


@pytest.fixture
def endpoint() -> Iterator[tuple[FakeService, int, str]]:
    """Run a disposable loopback listener on an ephemeral port without sleeps."""
    service = FakeService()
    token = "test-launch-token-" * 3
    server = make_server(service, token=token)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield service, server.server_port, token
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)
        assert not worker.is_alive()


def request(
    endpoint: tuple[FakeService, int, str], path: str, *, method: str = "GET",
    headers: dict[str, str] | None = None, body: bytes = b"{}",
) -> tuple[int, dict[str, str], bytes]:
    """Issue one bounded loopback HTTP request with explicit browser-like authentication."""
    _, port, token = endpoint
    base = {"Origin": f"http://127.0.0.1:{port}", "X-Console-Token": token,
            "Content-Type": "application/json"}
    base.update(headers or {})
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body if method != "GET" else None, headers=base)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("override", [
    {"Host": "attacker.example"}, {"Origin": "https://attacker.example"}, {"Origin": "null"},
    {"X-Console-Token": "wrong"}, {"X-Console-Token": ""},
])
def test_mutation_requires_token_host_and_origin(
    endpoint: tuple[FakeService, int, str], override: dict[str, str],
) -> None:
    """Cross-site and unauthenticated writes fail before reaching the mutable service."""
    assert request(endpoint, "/api/start", method="POST", headers=override)[0] == 403
    assert endpoint[0].commands == []


def test_page_reload_is_read_only_and_reconstructs_server_state(
    endpoint: tuple[FakeService, int, str],
) -> None:
    """Fresh pages recover active capture state without starting or stopping it."""
    assert request(endpoint, "/api/start", method="POST")[0] == 200
    first = request(endpoint, "/api/state")
    second = request(endpoint, "/api/state")
    assert first[2] == second[2]
    assert json.loads(first[2])["capture"] == "recording"
    assert len(endpoint[0].commands) == 1
    assert "frame-ancestors 'none'" in first[1]["Content-Security-Policy"]
    assert first[1]["Cache-Control"] == "no-store"


@pytest.mark.parametrize("path", ["/../AGENTS.md", "/%2e%2e/AGENTS.md", "/app.js/../../x"])
def test_static_path_confinement(
    endpoint: tuple[FakeService, int, str], path: str,
) -> None:
    """Encoded and plain traversal routes cannot expose repository or workspace files."""
    assert request(endpoint, path)[0] == 400


def test_mutations_are_post_only_and_bodies_bounded(
    endpoint: tuple[FakeService, int, str],
) -> None:
    """Alternative methods, oversized objects and wrong content types never write."""
    assert request(endpoint, "/api/start")[0] == 404
    assert request(endpoint, "/api/start", method="DELETE")[0] == 405
    assert request(endpoint, "/api/start", method="POST", body=b"x" * (MAX_BODY + 1))[0] == 400
    assert request(endpoint, "/api/start", method="POST",
                   headers={"Content-Type": "text/plain"})[0] == 400
    assert request(endpoint, "/api/refuse", method="POST")[0] == 409
    assert endpoint[0].commands == []


def test_packaged_page_and_latest_preview_are_inert(
    endpoint: tuple[FakeService, int, str],
) -> None:
    """Page assets and preview requests never ask the service to capture or move anything."""
    assert b"Synria Teleop Console" in request(endpoint, "/")[2]
    assert request(endpoint, "/media/preview/front")[2] == b"fake-latest-jpeg"
    assert endpoint[0].commands == []


def test_non_loopback_needs_explicit_permission() -> None:
    """A remote listener cannot be accidentally enabled through a changed bind address."""
    with pytest.raises(ValueError, match="allow-remote"):
        make_server(FakeService(), token="t" * 32, host="0.0.0.0")
    with pytest.raises(ValueError, match="IPv4"):
        make_server(FakeService(), token="t" * 32, host="::1")


def test_playback_is_exact_bounded_and_does_not_reencode_on_cache_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both camera images and state/action vectors come from the same stored row."""
    (tmp_path / "physical_episode_metadata.jsonl").write_text("{}\n", encoding="utf-8")
    frames = [{"episode_index": 2, "action": np.arange(7) + index,
               "observation.state": np.arange(7) - index,
               "observation.images.wrist": np.full((3, 4, 5), index / 10),
               "observation.images.front": np.full((3, 4, 5), (index + 1) / 10)}
              for index in range(4)]

    class Dataset:
        """Expose upstream's row interface using only deterministic memory arrays."""

        hf_dataset = frames

        def __getitem__(self, index: int) -> dict[str, Any]:
            """Return an exact synthetic stored row."""
            return frames[index]

    monkeypatch.setattr(
        "synria_lerobot.console.playback.encode_rgb", lambda pixels: pixels.tobytes(),
    )
    playback = EpisodePlayback(max_frames=2, reader=lambda *args: Dataset())
    for index in range(4):
        row, images = playback.frame(tmp_path, "local/fake", 2, index)
        assert row["action"] == (np.arange(7) + index).tolist()
        assert row["observation.state"] == (np.arange(7) - index).tolist()
        assert images["wrist"] == bytes([round(index * 25.5)]) * 60
        assert len(playback._cache) <= 2
    assert playback.frame(tmp_path, "local/fake", 2, 3) is playback._cache[3]
    with pytest.raises(IndexError):
        playback.frame(tmp_path, "local/fake", 2, 4)
    (tmp_path / ".recording_transaction.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="recovery"):
        playback.frame(tmp_path, "local/fake", 2, 0)


def test_artifact_confinement_and_native_still(tmp_path: Path) -> None:
    """Still lookup preserves recorder-owned filenames but refuses arbitrary disk references."""
    (tmp_path / "final_stills").mkdir()
    (tmp_path / "final_stills" / "episode-0.jpg").write_bytes(b"native-frame")
    metadata = tmp_path / "physical_episode_metadata.jsonl"
    metadata.write_text(json.dumps({"episode_index": 0,
                                    "final_still": "/old/root/final_stills/episode-0.jpg"}),
                        encoding="utf-8")
    assert EpisodePlayback().final_still(tmp_path, 0) == b"native-frame"
    with pytest.raises(ValueError):
        confined_file(tmp_path, "../private")
    metadata.write_text(json.dumps({"episode_index": 0, "final_still": "/private/secret"}),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="recorder-owned"):
        EpisodePlayback().final_still(tmp_path, 0)


def test_playback_refuses_symlinked_media_before_reader(tmp_path: Path) -> None:
    """A disk-side media link cannot make the upstream reader decode data outside the dataset."""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "physical_episode_metadata.jsonl").write_text("{}\n", encoding="utf-8")
    target = tmp_path / "private.mp4"
    target.write_bytes(b"private")
    try:
        (root / "linked.mp4").symlink_to(target)
    except OSError:
        pytest.skip("platform does not permit test symlinks")
    playback = EpisodePlayback(reader=lambda *args: pytest.fail("reader reached unsafe media"))
    with pytest.raises(ValueError, match="symlinked"):
        playback.frame(root, "local/fake", 0, 0)


def test_review_pairs_and_selection_ignore_late_browser_responses() -> None:
    """Run the shipped review functions against deterministic reordered image and row promises."""
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("optional JavaScript runtime is unavailable")
    root = Path(__file__).parents[1]
    subprocess.run(
        [executable, str(Path(__file__).with_name("console_review_races.cjs")),
         str(root / "synria_lerobot/console/static/app.js")],
        check=True, timeout=15, capture_output=True, text=True,
    )
