"""Exercise the real local dataset console over HTTP using only synthetic sources."""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.lerobot_dataset")
cv2 = pytest.importorskip("cv2")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from synria_lerobot.act_training import TRAINING_VERSION, validate_config  # noqa: E402
from synria_lerobot.console.app import ConsoleService  # noqa: E402
from synria_lerobot.console.controller import RecordingController  # noqa: E402
from synria_lerobot.console.playback import encode_rgb  # noqa: E402
from synria_lerobot.console.server import make_server  # noqa: E402
from synria_lerobot.quality_gates import (  # noqa: E402
    GateConfig,
    load_episode_records,
    load_limits,
    write_session_artifacts,
)


class Clock:
    """Provide monotonic synthetic sample times without sleeping or touching ROS clocks."""

    def __init__(self) -> None:
        """Start at an arbitrary positive host-clock epoch."""
        self.now = 100.0

    def __call__(self) -> float:
        """Return the test-selected time without advancing it through preview reads."""
        return self.now


class ObservedController(RecordingController):
    """Notify the test when the real owner worker consumes a fake-clock sample boundary."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Keep test synchronization separate from production capture and HTTP behavior."""
        self.progress = threading.Condition()
        self.observed_time = -1.0
        super().__init__(*args, **kwargs)

    def tick(self) -> None:
        """Run the unmodified production tick before publishing a test-only notification."""
        super().tick()
        with self.progress:
            self.observed_time = self.recorder.clock()
            self.progress.notify_all()

    def await_capture(self, when: float, frames: int) -> None:
        """Wait on worker progress rather than using a sleep or invoking capture from HTTP."""
        with self.progress:
            assert self.progress.wait_for(
                lambda: self.observed_time >= when and self.recorder.pending_frame_count >= frames,
                timeout=5,
            ), self.snapshot()


def test_demo_http_records_reviews_curates_and_purges_real_dataset(tmp_path: Path) -> None:
    """Keep synthetic evidence separate while exercising the complete named-session workflow."""
    clock = Clock()
    service = ConsoleService(
        tmp_path / "workspace", demo=True, min_free_bytes=1,
        controller_factory=ObservedController, clock=clock,
    )
    token = "synthetic-loopback-acceptance-" * 2
    server = make_server(service, token=token)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()

    def call(path: str, payload: dict[str, Any] | None = None) -> Any:
        """Use the public HTTP protocol for every operator action and review request."""
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
        headers = {"Origin": f"http://127.0.0.1:{server.server_port}",
                   "Content-Type": "application/json", "X-Console-Token": token}
        try:
            connection.request("GET" if payload is None else "POST", path,
                               body=None if payload is None else json.dumps(payload),
                               headers=headers)
            response = connection.getresponse()
            body = response.read()
            assert response.status == 200, (response.status, body)
            return json.loads(body) if response.getheader("Content-Type", "").startswith(
                "application/json"
            ) else body
        finally:
            connection.close()

    try:
        assert b"Synria Teleop Console" in call("/")
        session = call("/api/sessions", {
            "name": "Synthetic cup collection", "task_id": "die_into_cup", "gripper_type": "50mm",
            "fps": 5, "image_width": 32, "image_height": 24,
            "operator": "synthetic-test", "scene": "Synthetic marked cup and tray",
        })
        identity = session["id"]
        root = Path(session["dataset_path"])
        assert "sessions-demo" in root.parts
        assert session["source_kind"] == "synthetic_demo"
        call(f"/api/session/{identity}/open", {})
        controller = service.controller
        assert isinstance(controller, ObservedController)
        for label in ("success", "failure"):
            start = clock.now
            call("/api/record/start", {})
            for index in range(6):
                clock.now = start + index * 0.20001
                controller.await_capture(clock.now, index + 1)
            clock.now = start + 1.2
            call("/api/record/stop", {})
            call(f"/api/record/{label}", {})
        episodes = call(f"/api/episodes/{identity}")
        assert [row["operator_label"] for row in episodes] == ["success", "failure"]
        assert call("/api/state")["active_session"]["counts"]["saved"] == 2
        assert sum(event["kind"] == "episode_saved"
                   for event in service.catalog.events(identity)) == 2
        call(f"/api/session/{identity}/close", {})
        dataset = LeRobotDataset(session["repo_id"], root=root, video_backend="pyav")
        assert dataset.num_episodes == 2
        for episode in range(2):
            indices = [i for i, row in enumerate(dataset.hf_dataset)
                       if int(row["episode_index"]) == episode]
            for frame, global_index in enumerate(indices):
                stored = dataset[global_index]
                reviewed = call(f"/api/frame/{identity}/{episode}/{frame}")
                for field in ("observation.state", "action", "state_monotonic_s",
                              "action_monotonic_s", "front_monotonic_s", "wrist_monotonic_s"):
                    assert reviewed[field] == stored[field].numpy().tolist()
                for camera in ("wrist", "front"):
                    pixels = stored[f"observation.images.{camera}"].numpy()
                    rgb = np.rint(np.moveaxis(pixels, 0, -1) * 255).astype(np.uint8)
                    assert call(f"/media/frame/{identity}/{episode}/{frame}/{camera}") == (
                        encode_rgb(rgb)
                    )
            still = cv2.imdecode(np.frombuffer(call(f"/media/still/{identity}/{episode}"),
                                              dtype=np.uint8), cv2.IMREAD_COLOR)
            assert still.shape == (480, 640, 3)
            provenance = call(f"/api/provenance/{identity}/{episode}")
            assert provenance["contract"]["recording_purpose"] == "synthetic_demo"
        call(f"/api/episode/{identity}/0/exclude", {
            "reason_code": "capture_fault", "note": "Synthetic curation test",
            "operator": "synthetic-test",
        })
        assert call(f"/api/episodes/{identity}")[0]["excluded"]
        call(f"/api/episode/{identity}/0/restore", {
            "reason_code": "capture_fault", "operator": "synthetic-test",
        })
        assert not call(f"/api/episodes/{identity}")[0]["excluded"]
        contract = json.loads((root / "physical_contract.json").read_text(encoding="utf-8"))
        with pytest.raises(ValueError, match="synthetic demo"):
            validate_config({"version": TRAINING_VERSION, "policy_id": "synthetic-test",
                             "dataset_repo_id": session["repo_id"], "expected_contract": contract},
                            contract)
        with pytest.raises(ValueError, match="synthetic demo"):
            write_session_artifacts(
                session_dir=tmp_path / "refused-summary", dataset_path=root,
                provenance={"recording_purpose": "synthetic_demo"},
                episodes=load_episode_records(root / "physical_quality_records.jsonl"),
                limits=load_limits(Path(__file__).parents[1] / "config/synria_limits.yaml"),
                gate_config=GateConfig(1, 30),
            )
        confirmation = {"confirmation": session["name"], "reason": "Synthetic acceptance cleanup"}
        trashed = call(f"/api/session/{identity}/trash", confirmation)
        assert trashed["status"] == "trashed" and not root.exists()
        restored = call(f"/api/session/{identity}/restore", {})
        assert restored["status"] == "closed" and root.exists()
        reloaded = LeRobotDataset(session["repo_id"], root=root, video_backend="pyav")
        assert reloaded.num_episodes == 2
        call(f"/api/session/{identity}/trash", confirmation)
        purged = call(f"/api/session/{identity}/purge", confirmation)
        assert purged["status"] == "purged"
        assert not Path(purged["dataset_path"]).exists()
        assert purged["tombstone"]["counts"]["saved"] == 2
        assert service.catalog.events(identity)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        service.close(force=True, reason="synthetic test cleanup")
