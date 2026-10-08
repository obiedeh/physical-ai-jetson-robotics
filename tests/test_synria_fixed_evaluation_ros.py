"""Exercise fixed-task evaluation through localhost HTTP and an in-memory ROS graph."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from test_synria_fixed_evaluation import fixed_setup
from test_synria_ros_adapter import setup as fake_ros_setup  # noqa: F401

from synria_lerobot.evaluation import FUNNEL
from synria_lerobot.physical_contract import ImageFrame
from synria_lerobot.policy_server import make_http_server
from synria_lerobot.sessions import run_fixed_evaluation


@pytest.fixture
def evaluation(tmp_path: Path, fake_ros_setup: Any) -> Any:  # noqa: F811
    config, transport, repository = fixed_setup(tmp_path)
    setup = fake_ros_setup
    setup.clock.now = 3.0
    setup.config.update(config)
    setup.settings.update(image_width=4, image_height=4)
    Path(config["adapter_config"]).write_text(json.dumps(setup.settings))
    original_camera = setup.sources.camera
    stored_shapes = []

    def camera(device: str, *, width: int, height: int) -> Any:
        source = original_camera(device, width=width, height=height)

        def read() -> ImageFrame:
            stored = np.full((height, width, 3), (40, 100, 180), np.uint8)
            stored_shapes.append(stored.shape)
            return ImageFrame(
                stored,
                setup.clock(),
                native_resolution=(640, 480),
                source_id=device,
                native_data=np.full((480, 640, 3), (40, 100, 180), np.uint8),
            )

        source.read = read
        return source

    setup.sources.camera = camera
    return SimpleNamespace(
        config=setup.config,
        setup=setup,
        transport=transport,
        repository=repository,
        output=tmp_path / "evaluation",
        stored_shapes=stored_shapes,
    )


@contextmanager
def policy_endpoint(evaluation: Any):
    server = make_http_server(evaluation.transport.service, "127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    evaluation.config["d2_policy"]["endpoint"] = f"http://127.0.0.1:{server.server_port}"
    try:
        yield
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()
        assert not worker.is_alive()


def run(evaluation: Any) -> dict:
    setup = evaluation.setup

    def factory(settings: dict) -> Any:
        setup.config.update(settings)
        return setup.create()

    with policy_endpoint(evaluation):
        return run_fixed_evaluation(
            evaluation.config,
            evaluation.repository,
            evaluation.output,
            enable_motion=True,
            factory=factory,
            # Real sleep reaches or exceeds its deadline; avoid fake rounding below it.
            sleep=lambda duration: setup.clock.sleep(duration + 1e-9),
            clock=setup.clock,
        )


def records(evaluation: Any) -> list[dict]:
    return [
        json.loads(line)
        for line in (evaluation.output / "EvalLog.jsonl").read_text().splitlines()
    ]


def trial_answers(label: str) -> list[str]:
    return ["yes", label, *(["unknown"] * len(FUNNEL))]


def test_twenty_http_trials_share_one_guarded_sink_and_native_camera_evidence(
    evaluation: Any,
) -> None:
    setup = evaluation.setup
    setup.answers.extend(["sync-off", "armed"])
    for index in range(20):
        setup.answers.extend(trial_answers("success" if index < 14 else "failure"))
    setup.answers.append("unchanged")

    stats = run(evaluation)

    assert stats["session_status"] == "completed", stats
    assert stats["threshold_met"] and stats["successes"] == 14
    assert stats["operator_object_successes"] == 14
    assert stats["first_try"]["rate"] == 0.7
    assert stats["funnel"] == dict.fromkeys(FUNNEL)
    assert stats["funnel_unknown_trials"] == dict.fromkeys(FUNNEL, 20)
    assert stats["failure_taxonomy_counts"]["operator-labeled object failure"] == 6
    assert stats["stage_status"] == "planned" and stats["data_kind"] == "synthetic"
    assert setup.ros.publisher_topics == ["/fake_policy_targets"]
    assert setup.ros.action_clients == ["/fake_gripper/follow_joint_trajectory"]
    assert len(setup.ros.goals) == 20
    assert evaluation.transport.model.reset_count == 20
    assert evaluation.stored_shapes and set(evaluation.stored_shapes) == {(4, 4, 3)}

    log = records(evaluation)
    trials = [record for record in log if record["kind"] == "trial"]
    assert len(trials) == 20
    for index, trial in enumerate(trials):
        attempt = trial["attempts"][0]
        assert attempt["execution_status"] == "completed" and attempt["steps"] == 1
        assert attempt["scene_confirmed"] is True
        grade = attempt["operator_grade"]
        assert grade["label"] == ("success" if index < 14 else "failure")
        assert grade["native_resolution"] == [640, 480]
        still = cv2.imread(grade["still"])
        assert still.shape == (480, 640, 3)
        assert still[0, 0].tolist() == pytest.approx([180, 100, 40], abs=2)
    provenance = json.loads((evaluation.output / "provenance.json").read_text())
    stills = [item for item in provenance["source_preflight"]["events"] if "front_still" in item]
    assert len(stills) == 20
    assert all(item["native_resolution_confirmed"] for item in stills)
    assert all(3 <= item["source_monotonic_s"] <= setup.clock() for item in stills)
    assert all(item["source_id"] == setup.settings["front_camera"] for item in stills)
    assert not setup.answers and setup.sources.closed == setup.sources.opened
    assert set(setup.ros.closed) == {"executor", "gripper", "publisher", "node", "context"}


@pytest.mark.parametrize("fault", ["arm", "gripper", "driver", "policy", "direct"])
def test_http_attestation_cannot_bypass_graph_preflight(evaluation: Any, fault: str) -> None:
    setup = evaluation.setup
    if fault in {"arm", "gripper"}:
        setup.ros.servers.remove(setup.settings[fault + "_action"])
    elif fault == "driver":
        setup.ros.nodes.append(("alicia_d_driver_node", "/unexpected_namespace"))
    else:
        topic = "/fake_policy_targets" if fault == "policy" else "/joint_commands"
        setup.ros.external[topic] = [object()]

    stats = run(evaluation)

    assert stats["session_status"] == "failed" and not stats["threshold_met"]
    assert not any(record["kind"] == "trial" for record in records(evaluation))
    assert not setup.sources.opened
    assert not setup.ros.publisher_topics and not setup.ros.action_clients
    assert evaluation.transport.model.reset_count == 0
    assert "node" in setup.ros.closed and "context" in setup.ros.closed


@pytest.mark.parametrize("answer", ["not-confirmed", "abort"])
def test_http_policy_does_not_authorize_motion_without_sync_confirmation(
    evaluation: Any, answer: str
) -> None:
    setup = evaluation.setup
    setup.answers.append(answer)
    if answer != "abort":
        setup.answers.append("unchanged")

    stats = run(evaluation)

    assert stats["session_status"] in {"failed", "aborted"} and not stats["threshold_met"]
    assert not setup.ros.publisher_topics and not setup.ros.action_clients
    assert not setup.ros.goals and evaluation.transport.model.reset_count == 0
    assert setup.sources.closed == setup.sources.opened and not setup.answers
    assert "node" in setup.ros.closed


def test_declining_next_trial_retains_prior_success_and_stops_offers(evaluation: Any) -> None:
    setup = evaluation.setup
    setup.answers.extend(["sync-off", "armed", *trial_answers("success"), "no", "unchanged"])

    stats = run(evaluation)

    assert stats["session_status"] == "aborted" and not stats["threshold_met"]
    assert stats["operator_object_successes"] == 1
    trials = [record for record in records(evaluation) if record["kind"] == "trial"]
    assert len(trials) == 2
    assert trials[0]["attempts"][0]["ok"]
    stopped = trials[1]["attempts"][0]
    assert stopped["execution_status"] == "aborted" and stopped["steps"] == 0
    assert stopped["scene_confirmed"] is False and stopped["operator_grade"] is None
    assert len(setup.ros.goals) == evaluation.transport.model.reset_count == 1
    assert setup.ros.published[-1].position == [setup.sources.position] * 6
    assert setup.sources.closed == setup.sources.opened and not setup.answers


def test_gripper_fault_is_ledgered_and_holds_without_starting_another_trial(
    evaluation: Any,
) -> None:
    setup = evaluation.setup
    setup.ros.result_status = 6
    setup.ros.result_code = -1
    setup.answers.extend(["sync-off", "armed", "yes"])

    stats = run(evaluation)

    assert stats["session_status"] == "failed" and not stats["threshold_met"]
    trials = [record for record in records(evaluation) if record["kind"] == "trial"]
    assert len(trials) == 1
    attempt = trials[0]["attempts"][0]
    assert attempt["execution_status"] == "fault" and not attempt["ok"]
    assert attempt["operator_grade"] is None and attempt["errors"]
    assert len(setup.ros.goals) == evaluation.transport.model.reset_count == 1
    assert setup.ros.published[-1].position == [setup.sources.position] * 6
    assert setup.ros.action_clients == ["/fake_gripper/follow_joint_trajectory"]
    assert setup.ros.cancel_count == 0  # Already-terminal owned goal needs no cancellation.
    assert setup.sources.closed == setup.sources.opened and not setup.answers
    assert "context" in setup.ros.closed
