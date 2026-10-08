"""Fixed-scene roll sessions use synthetic receipts, images and command sinks only."""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from test_synria_policy_server import StubModel, fixture
from test_synria_sessions import FakeIO, ready_config

from synria_lerobot.checkpoint_eval import strict_content_hash
from synria_lerobot.embodiment import SynriaEmbodiment
from synria_lerobot.evaluation import FUNNEL, FixedTaskGrade
from synria_lerobot.perception import GatedPerception
from synria_lerobot.policy_client import GuardedCommandPath, PolicyClient
from synria_lerobot.policy_codec import decode_policy_response, encode_policy_request
from synria_lerobot.policy_server import (
    CheckpointIdentity,
    FixedScenePolicyService,
    make_http_server,
)
from synria_lerobot.quality_gates import load_limits
from synria_lerobot.sessions import bind_fixed_skill, run_roll_skills
from synria_lerobot.task_registry import TASK_IDS
from synria_lerobot.turn_executor import AbortGuardSink, FixedSkillExecutor, OperatorAbort


class Transport:
    def __init__(self, identity: CheckpointIdentity) -> None:
        self.identity = identity
        self.model = StubModel()
        self.service = FixedScenePolicyService(identity, self.model)
        self.calls = []
        self.after_response = lambda: None

    def metadata(self, timeout_s: float) -> dict:
        assert timeout_s == self.identity.metadata()["response_timeout_s"]
        return self.identity.metadata()

    def request(self, payload: dict, timeout_s: float) -> dict:
        self.calls.append(payload)
        result = decode_policy_response(self.service.request(encode_policy_request(payload)))
        self.after_response()
        return result


def roll_config(tmp_path: Path) -> tuple[dict, dict[str, Transport]]:
    config = ready_config(tmp_path)
    transports = {}
    for task in TASK_IDS:
        root = tmp_path / task
        root.mkdir()
        identity, receipt, registry = fixture(root, task)
        manifest = identity.manifest
        manifest["policy_id"] = manifest["configuration"]["policy_id"] = "synthetic-" + task
        (identity.checkpoint / "physical_policy_manifest.json").write_text(json.dumps(manifest))
        receipt.write_text(
            json.dumps(
                {
                    **manifest,
                    "path": str(identity.checkpoint),
                    "evaluation_status": "evaluated",
                    "checkpoint_content_sha256": strict_content_hash(identity.checkpoint),
                }
            )
        )
        identity = CheckpointIdentity.load(identity.checkpoint, receipt, registry)
        transports[task] = Transport(identity)
        config["task_registry"] = str(registry)
        config["command_period_s"] = identity.metadata()["cadence"]["command_period_s"]
        config["roll_skills"][task] = {
            "checkpoint": str(identity.checkpoint),
            "receipt": str(receipt),
            "endpoint": "http://unused.invalid",
            "max_steps": 2,
            "response_timeout_s": identity.metadata()["response_timeout_s"],
        }
    settings = json.loads(Path("config/synria_ros_adapter.json").read_text())
    settings.update(
        arm_action="/fake_arm/follow_joint_trajectory",
        wrist_camera="/dev/v4l/by-id/fake-wrist",
        front_camera="/dev/v4l/by-id/fake-front",
        image_width=4,
        image_height=4,
    )
    adapter = tmp_path / "adapter.json"
    adapter.write_text(json.dumps(settings))
    config["adapter_config"] = str(adapter)
    # Prove the fixed path never needs a board, token scorer or token perception.
    for key in ("calibration", "reachable", "perception_config", "accuracy_reports", "d2_trials"):
        config.pop(key, None)
    return config, transports


class RollIO(FakeIO):
    def __init__(self, root: Path, *, fail: str | None = None) -> None:
        super().__init__(root)
        self.labels = []
        self.pending = False
        self.fail = fail
        self.events = []
        self.confirmations = []

    def confirm_skill(self, task) -> bool:
        assert self.events[-1] == "hold"
        self.confirmations.append(task.task_id)
        return True

    def offer(self, action: tuple) -> None:
        super().offer(action)
        self.events.append("offer")

    def hold(self) -> None:
        super().hold()
        self.events.append("hold")

    def abort_pending(self) -> bool:
        return self.pending

    def grade_skill(self, task) -> FixedTaskGrade:
        assert self.events[-1] == "hold"
        self.events.append("grade")
        self.labels.append(task.task_id)
        return FixedTaskGrade(
            "failure" if task.task_id == self.fail else "success",
            "synthetic operator",
            str(self.still),
            time.monotonic(),
            task,
            (1, 1),
        )

    def capture_front(self, *, require_native: bool = False) -> tuple:
        assert require_native
        return np.zeros((1, 1, 3), np.uint8), str(self.still), time.monotonic()

    def operator_roll(self, *, still: str | None = None) -> int:
        assert still == str(self.still)
        return 4


def run(tmp_path: Path, config: dict, transports: dict, io: RollIO, **kwargs) -> dict:
    return run_roll_skills(
        config,
        Path.cwd(),
        tmp_path / "run",
        factory=lambda _: io,
        transports=transports,
        sleep=lambda _: None,
        **kwargs,
    )


def test_three_skills_share_one_sink_without_board_or_token_path(tmp_path: Path) -> None:
    config, transports = roll_config(tmp_path)
    io = RollIO(tmp_path)
    result = run(tmp_path, config, transports, io, enable_motion=True)
    assert result["status"] == "completed", result
    assert result["roll"]["value"] == 4
    assert io.labels == list(TASK_IDS) and len(io.offers) == 6
    assert io.confirmations == list(TASK_IDS)
    assert io.sinks == io.authorizations == io.preflight_calls == 1 and io.closed
    assert result["d4_qualifying"] is False and result["stage_status"] == "planned"
    records = [
        json.loads(line)
        for line in (tmp_path / "run/skill_attempts.jsonl").read_text().splitlines()
    ]
    assert len(records) == 6
    for task, transport in transports.items():
        assert [call["reset"] for call in transport.calls] == [True, False]
        assert all(
            call["task"] == transport.identity.contract.task_text for call in transport.calls
        )
        record = next(
            row for row in records if row["task_id"] == task and row["status"] == "success"
        )
        assert record["operator_grade"]["funnel"] == dict.fromkeys(FUNNEL)
        assert record["policy"]["checkpoint_content_sha256"] == transport.identity.content_sha256


@pytest.mark.parametrize("failed_index", range(3))
def test_failed_skill_is_ledgered_and_stops_following_skills(
    tmp_path: Path, failed_index: int
) -> None:
    config, transports = roll_config(tmp_path)
    io = RollIO(tmp_path, fail=TASK_IDS[failed_index])
    result = run(tmp_path, config, transports, io, enable_motion=True)
    assert result["status"] == "failed" and result["roll"]["value"] is None
    assert io.labels == list(TASK_IDS[: failed_index + 1]) and io.closed
    for task in TASK_IDS[failed_index + 1 :]:
        assert not transports[task].calls


def test_abort_during_policy_response_prevents_offer_and_retains_attempt(tmp_path: Path) -> None:
    config, transports = roll_config(tmp_path)
    io = RollIO(tmp_path)
    transports[TASK_IDS[0]].after_response = lambda: setattr(io, "pending", True)
    result = run(tmp_path, config, transports, io, enable_motion=True)
    assert result["status"] == "failed" and not io.offers and not io.labels and io.closed
    records = [
        json.loads(line)
        for line in (tmp_path / "run/skill_attempts.jsonl").read_text().splitlines()
    ]
    assert records[-1]["status"] == "aborted" and records[-1]["failure_still"]["sha256"]
    assert all(not transports[task].calls for task in TASK_IDS[1:])


@pytest.mark.parametrize("key", ["task_id", "checkpoint_content_sha256", "response_timeout_s"])
def test_identity_mismatch_refuses_before_any_source_opens(tmp_path: Path, key: str) -> None:
    config, transports = roll_config(tmp_path)
    original = transports[TASK_IDS[-1]].identity.metadata()
    corrupted = copy.deepcopy(original)
    corrupted[key] = "mismatch"
    transports[TASK_IDS[-1]].metadata = lambda _: corrupted
    calls = []
    with pytest.raises(ValueError, match="metadata differs"):
        run_roll_skills(
            config,
            Path.cwd(),
            tmp_path / "run",
            enable_motion=True,
            factory=lambda _: calls.append(True),
            transports=transports,
        )
    assert not calls and not (tmp_path / "run").exists()


@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, float("nan")])
def test_explicit_command_budget_required(tmp_path: Path, value: object) -> None:
    config, transports = roll_config(tmp_path)
    config["roll_skills"][TASK_IDS[0]]["max_steps"] = value
    with pytest.raises(ValueError, match="command budget"):
        bind_fixed_skill(
            config,
            config["roll_skills"][TASK_IDS[0]],
            TASK_IDS[0],
            transport=transports[TASK_IDS[0]],
        )


def test_observe_only_attests_all_policies_but_never_authorizes_or_creates_sink(
    tmp_path: Path,
) -> None:
    config, transports = roll_config(tmp_path)
    io = RollIO(tmp_path)
    result = run(tmp_path, config, transports, io)
    assert result["status"] == "read_only" and io.sinks == io.authorizations == 0 and io.closed
    assert not io.offers and not io.labels


def test_failed_authorization_closes_sources_and_records_refusal(tmp_path: Path) -> None:
    config, transports = roll_config(tmp_path)
    io = RollIO(tmp_path)

    def refused():
        raise RuntimeError("synthetic sync confirmation refused")

    io.authorize_motion = refused
    result = run(tmp_path, config, transports, io, enable_motion=True)
    assert result["status"] == "failed" and io.closed and io.sinks == 0
    assert result["source_preflight"]["synthetic"] and result["power_state_end"]
    assert any("sync confirmation" in error for error in result["errors"])
    assert (tmp_path / "run/roll_result.json").is_file()


def test_roll_primary_fault_and_cleanup_failures_are_both_retained(tmp_path: Path) -> None:
    config, transports = roll_config(tmp_path)
    io = RollIO(tmp_path)

    def primary(task):
        raise RuntimeError("primary synthetic observation fault")

    def failed_close():
        io.closed = True
        raise RuntimeError("secondary synthetic close failure")

    io.observe = primary
    io.close = failed_close
    result = run(tmp_path, config, transports, io, enable_motion=True)
    assert result["status"] == "failed" and io.closed
    assert "primary synthetic" in result["roll"]["error"]
    assert any("secondary synthetic" in error for error in result["errors"])
    assert all((tmp_path / f"run/{task}_latencies.jsonl").is_file() for task in TASK_IDS)
    assert result["power_state_end"]


def test_timeout_cannot_be_replaced_by_a_session_default(tmp_path: Path) -> None:
    config, transports = roll_config(tmp_path)
    config["roll_skills"][TASK_IDS[0]]["response_timeout_s"] *= 2
    with pytest.raises(ValueError, match="per-policy timeout"):
        bind_fixed_skill(
            config,
            config["roll_skills"][TASK_IDS[0]],
            TASK_IDS[0],
            transport=transports[TASK_IDS[0]],
        )


def test_reused_probe_path_keeps_one_current_abort_guard_and_one_hold_forward(
    tmp_path: Path,
) -> None:
    config, transports = roll_config(tmp_path)
    task = TASK_IDS[0]
    binding = bind_fixed_skill(
        config,
        config["roll_skills"][task],
        task,
        enable_motion=True,
        transport=transports[task],
    )
    io = RollIO(tmp_path)
    client = PolicyClient(
        SynriaEmbodiment(binding.contract),
        binding.transport,
        load_limits(Path(config["limits"])),
        binding.safety,
    )
    path = GuardedCommandPath(client, io.command_sink, enable_motion=True)
    guard = None
    calls = []
    pending = False
    for attempt in range(100):

        def abort_pending(index=attempt):
            calls.append(index)
            return pending

        executor = FixedSkillExecutor(
            path,
            io.observe,
            abort_pending,
            max_steps=1,
            period_s=binding.safety.command_period_s,
            sleep=lambda _: None,
        )
        assert isinstance(path.sink, AbortGuardSink) and path.sink.sink is io
        if guard is None:
            guard = path.sink
        assert path.sink is guard
        calls.clear()
        path.sink.offer((0.0,) * 7)
        assert calls == [attempt]
        holds = io.holds
        path.hold()
        assert io.holds == holds + 1 and calls == [attempt]
        if attempt in (0, 1, 99):
            calls.clear()
            assert executor.execute() == 1
            assert calls == [attempt] * 4

    assert guard is not None and io.sinks == 1
    pending = True
    calls.clear()
    offers = len(io.offers)
    with pytest.raises(OperatorAbort, match="before command offer"):
        guard.offer((0.0,) * 7)
    assert calls == [99] and len(io.offers) == offers
    holds = io.holds
    with pytest.raises(OperatorAbort, match="operator aborted"):
        executor.execute()
    assert len(io.offers) == offers and io.holds == holds + 1


def test_full_roll_uses_three_real_local_http_endpoints_and_one_fake_sink(tmp_path: Path) -> None:
    config, transports = roll_config(tmp_path)
    servers = []
    workers = []
    try:
        for task, transport in transports.items():
            server = make_http_server(transport.service, "127.0.0.1", 0)
            servers.append(server)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            workers.append(worker)
            worker.start()
            config["roll_skills"][task]["endpoint"] = f"http://127.0.0.1:{server.server_port}"
        io = RollIO(tmp_path)
        result = run(tmp_path, config, {}, io, enable_motion=True)
        assert result["status"] == "completed", result
        assert io.sinks == 1 and len(io.offers) == 6 and io.labels == list(TASK_IDS)
        assert all(transport.model.reset_count == 1 for transport in transports.values())
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for worker in workers:
            worker.join(2)
            assert not worker.is_alive()


@pytest.mark.parametrize("fault", [KeyboardInterrupt, EOFError, RuntimeError])
def test_mid_skill_fault_holds_and_retains_failure_without_advancing(
    tmp_path: Path, fault: type
) -> None:
    config, transports = roll_config(tmp_path)
    io = RollIO(tmp_path)

    def fail(_: str):
        raise fault("synthetic interruption")

    io.observe = fail
    result = run(tmp_path, config, transports, io, enable_motion=True)
    assert result["status"] == "failed" and io.holds and io.closed and not io.offers
    records = [
        json.loads(line)
        for line in (tmp_path / "run/skill_attempts.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2 and records[-1]["status"] in {"failed", "aborted"}
    assert not io.labels and not transports[TASK_IDS[1]].calls


@pytest.mark.parametrize("key", ["command_period_s", "image_width"])
def test_incompatible_common_physical_configuration_opens_no_source(
    tmp_path: Path, key: str
) -> None:
    config, transports = roll_config(tmp_path)
    if key == "command_period_s":
        config[key] *= 2
    else:
        adapter = Path(config["adapter_config"])
        settings = json.loads(adapter.read_text())
        settings[key] += 1
        adapter.write_text(json.dumps(settings))
    calls = []
    with pytest.raises(ValueError, match="period|dimensions"):
        run_roll_skills(
            config,
            Path.cwd(),
            tmp_path / "run",
            enable_motion=True,
            factory=lambda _: calls.append(True),
            transports=transports,
        )
    assert not calls


def test_declining_second_scene_stops_without_reinterpreting_first_success(tmp_path: Path) -> None:
    config, transports = roll_config(tmp_path)
    io = RollIO(tmp_path)
    io.confirm_skill = lambda task: task.task_id == TASK_IDS[0]
    result = run(tmp_path, config, transports, io, enable_motion=True)
    assert result["status"] == "failed" and io.labels == [TASK_IDS[0]] and len(io.offers) == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "run/skill_attempts.jsonl").read_text().splitlines()
    ]
    assert records[1]["status"] == "success" and records[1]["operator_grade"]["label"] == "success"
    assert records[-1]["status"] == "aborted" and not records[-1]["scene_confirmed"]
    assert records[-1]["offered_steps"] == 0


def test_roll_evidence_cannot_mutate_a_frozen_checkpoint(tmp_path: Path) -> None:
    config, transports = roll_config(tmp_path)
    checkpoint = Path(config["roll_skills"][TASK_IDS[0]]["checkpoint"])
    before = strict_content_hash(checkpoint)
    calls = []
    with pytest.raises(ValueError, match="artifact"):
        run_roll_skills(
            config,
            Path.cwd(),
            checkpoint / "new-report",
            enable_motion=True,
            factory=lambda _: calls.append(True),
            transports=transports,
        )
    assert not calls and strict_content_hash(checkpoint) == before


@pytest.mark.parametrize("value", [True, 0, 7, None, "4", 1.5])
def test_invalid_operator_die_value_is_not_a_completed_roll(tmp_path: Path, value: object) -> None:
    config, transports = roll_config(tmp_path)
    io = RollIO(tmp_path)
    io.operator_roll = lambda **kwargs: value
    result = run(tmp_path, config, transports, io, enable_motion=True)
    assert result["status"] == "failed" and result["roll"]["value"] is None
    assert io.labels == list(TASK_IDS) and io.closed


@pytest.mark.parametrize("failure", ["hold", "capture", "timeout"])
def test_failure_paths_keep_attempt_and_stop_next_skill(tmp_path: Path, failure: str) -> None:
    config, transports = roll_config(tmp_path)
    io = RollIO(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic " + failure)

    if failure == "hold":
        io.hold = fail
    elif failure == "capture":
        io.grade_skill = fail
        io.capture_front = fail
    else:

        def timeout(*args):
            raise TimeoutError("synthetic timeout")

        transports[TASK_IDS[1]].request = timeout
    result = run(tmp_path, config, transports, io, enable_motion=True)
    assert result["status"] == "failed" and io.closed
    assert not transports[TASK_IDS[-1]].calls
    records = [
        json.loads(line)
        for line in (tmp_path / "run/skill_attempts.jsonl").read_text().splitlines()
    ]
    assert records[-1]["status"] == "failed"
    if failure == "capture":
        assert "synthetic capture" in records[-1]["missing_still_reason"]


def test_perception_boundary_preserves_rgb_policy_and_uses_bgr_accuracy_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, transports = roll_config(tmp_path)
    config["fixed_roll"]["read"] = {
        "source": "perception",
        "stable_reads": 1,
        "max_reads": 1,
        "period_s": 0.1,
        "max_age_s": 1.0,
        "image_shape": [3, 4, 4],
    }
    config.update(perception_config="synthetic.json", accuracy_reports={"die": "synthetic.json"})
    io = RollIO(tmp_path)
    pixels = np.full((4, 4, 3), (10, 50, 200), np.uint8)
    io.capture_front = lambda **kwargs: (pixels, str(io.still), time.monotonic())
    seen = []

    class Perception:
        enabled = {"die"}
        synthetic = False
        report_hashes = {"die": "synthetic-report"}

        def require_input_shape(self, component, shape):
            assert component == "die" and shape == (4, 4, 3)

        def die(self, image):
            seen.append(image.copy())
            return 3

    monkeypatch.setattr("synria_lerobot.sessions.GatedPerception", lambda *args: Perception())
    result = run(tmp_path, config, transports, io, enable_motion=True)
    assert result["status"] == "completed", result
    assert np.array_equal(seen[0][0, 0], [200, 50, 10])
    assert np.array_equal(pixels[0, 0], [10, 50, 200])


@pytest.mark.parametrize("shape", [None, [480, 640, 3]])
def test_die_report_missing_or_different_dimensions_refuses_before_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape: object,
) -> None:
    config, transports = roll_config(tmp_path)
    config["fixed_roll"]["read"] = {
        "source": "perception",
        "stable_reads": 1,
        "max_reads": 1,
        "period_s": 0.1,
        "max_age_s": 1,
        "image_shape": [3, 4, 4],
    }
    config.update(perception_config="synthetic.json", accuracy_reports={"die": "synthetic.json"})
    perception = object.__new__(GatedPerception)
    perception.reports = {"die": {"inputs": [{"image_shape_hwc": shape}]}}
    monkeypatch.setattr("synria_lerobot.sessions.GatedPerception", lambda *args: perception)
    calls = []
    with pytest.raises(ValueError, match="dimensions missing or different"):
        run_roll_skills(
            config,
            Path.cwd(),
            tmp_path / "run",
            enable_motion=True,
            factory=lambda _: calls.append(True),
            transports=transports,
        )
    assert not calls
