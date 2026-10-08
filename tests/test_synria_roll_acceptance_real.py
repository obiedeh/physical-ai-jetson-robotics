"""Optional CPU-only recorded-data-to-policy acceptance; every device source is fake."""

from __future__ import annotations

import copy
import json
import subprocess
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from test_lerobot_dataset_writer import writer_config
from test_synria_act_training_real import training_fixture
from test_synria_checkpoint_eval import commit
from test_synria_fixed_evaluation import register_fixed_protocol
from test_synria_policy_server import StubModel
from test_synria_policy_server import fixture as checkpoint_fixture
from test_synria_ros_adapter import setup as ros_setup
from test_task_registry import synthetic_registry

from synria_lerobot.act_training import train
from synria_lerobot.checkpoint_eval import strict_content_hash
from synria_lerobot.evaluation import FUNNEL
from synria_lerobot.physical_contract import (
    ImageFrame,
    PhysicalState,
    StateRateMeasurement,
    StateSourceProvenance,
)
from synria_lerobot.policy_server import (
    CheckpointIdentity,
    FixedScenePolicyService,
    LocalACTModel,
    make_http_server,
)
from synria_lerobot.quality_gates import (
    GateConfig,
    evaluate_episode,
    load_episode_records,
    load_limits,
)
from synria_lerobot.recorder import (
    LeRobotDatasetWriter,
    NextStateActionSource,
    PhysicalEpisodeRecorder,
)
from synria_lerobot.sessions import run_fixed_evaluation, run_roll_skills
from synria_lerobot.task_registry import TASK_IDS


def record_fake_episodes(root: Path, values: tuple[float, ...]) -> None:
    clock = SimpleNamespace(now=100.0, value=0.0, sample=0)
    closed, graph_checks = [], []
    config = replace(
        writer_config(root),
        image_width=32,
        image_height=32,
        state_rate_measurement=StateRateMeasurement(30, 60, 2, 98, 100, 1 / 30),
        state_source_provenance=StateSourceProvenance("standalone_driver", "/joint_states"),
    )

    def frame(name: str) -> ImageFrame:
        shade = 60 + clock.sample * 10 + (40 if name == "front" else 0)
        return ImageFrame(
            np.full((32, 32, 3), shade, np.uint8),
            clock.now,
            native_resolution=(640, 480),
            source_id="synthetic-" + name,
            native_data=np.full((480, 640, 3), shade, np.uint8),
        )

    writer = LeRobotDatasetWriter(config)
    recorder = PhysicalEpisodeRecorder(
        config=config,
        state_source=SimpleNamespace(
            read=lambda: PhysicalState(
                (clock.value,) * 6,
                clock.value / 100,
                clock.now,
                1_800_000_000 + clock.now,
                ros_arrival_stamp_s=1_800_000_000 + clock.now,
            ),
            close=lambda: closed.append("state"),
        ),
        action_source=NextStateActionSource(),
        wrist_source=SimpleNamespace(
            read=lambda: frame("wrist"),
            close=lambda: closed.append("wrist"),
        ),
        front_source=SimpleNamespace(
            read=lambda: frame("front"),
            close=lambda: closed.append("front"),
        ),
        writer=writer,
        clock=lambda: clock.now,
        command_publisher_guard=lambda topics: graph_checks.append(topics),
    )
    try:
        for index, value in enumerate(values):
            clock.now, clock.value = 100.0 + index, value
            recorder.start()
            for sample in range(3):
                clock.sample, clock.now = sample, 100.0 + index + sample / config.fps
                assert recorder.capture_once()
            clock.now = 100.2 + index
            recorder.stop()
            episode = recorder.mark_success()
            assert episode.episode_index == index
            assert len(episode.frames) == 3
            assert cv2.imread(str(episode.final_still_path)).shape == (480, 640, 3)
    finally:
        recorder.close()
    assert closed == ["state", "wrist", "front"] and len(graph_checks) == len(values)


@contextmanager
def serve(identity: CheckpointIdentity, model: Any):
    server = make_http_server(FixedScenePolicyService(identity, model), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
        assert not thread.is_alive()


def fake_device(root: Path) -> Any:
    root.mkdir()
    setup = ros_setup.__wrapped__(root)
    setup.clock.now = 3.0
    setup.settings.update(image_width=32, image_height=32)
    setup.config.update(adapter="never.imported:factory", data_kind="synthetic")
    setup.config["provenance"] = dict.fromkeys(setup.config["provenance"], "synthetic fixture")
    Path(setup.config["adapter_config"]).write_text(json.dumps(setup.settings))
    original = setup.sources.camera

    def camera(device: str, *, width: int, height: int) -> Any:
        source = original(device, width=width, height=height)
        source.read = lambda: ImageFrame(
            np.full((height, width, 3), 90, np.uint8),
            setup.clock(),
            native_resolution=(640, 480),
            source_id=device,
            native_data=np.full((480, 640, 3), 90, np.uint8),
        )
        return source

    setup.sources.camera = camera
    return setup


def io_factory(setup: Any):
    def create(config: dict) -> Any:
        setup.config.update(config)
        return setup.create()

    return create


def specification(identity: CheckpointIdentity, receipt: Path, endpoint: str) -> dict:
    return {
        "checkpoint": str(identity.checkpoint),
        "receipt": str(receipt),
        "endpoint": endpoint,
        "response_timeout_s": identity.metadata()["response_timeout_s"],
        "max_steps": 1,
    }


def test_record_train_serve_evaluate_and_roll_with_no_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, dataset = training_fixture(
        tmp_path,
        monkeypatch,
        episode_values=(0.01, 0.9),
        record_dataset=record_fake_episodes,
    )
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    loaded = LeRobotDataset("local/synria-test", root=dataset, video_backend="pyav")
    assert (loaded.num_episodes, loaded.num_frames) == (2, 6)
    assert tuple(loaded[0]["observation.images.front"].shape) == (3, 32, 32)
    contract = json.loads((dataset / "physical_contract.json").read_text(encoding="utf-8"))
    assert contract == config["expected_contract"]
    assert contract["min_episode_s"] == 0.1 and contract["max_episode_s"] == 1
    records = load_episode_records(dataset / "physical_quality_records.jsonl")
    assert len(records) == 2
    for recorded in records:
        assert recorded.fps == 15 and recorded.duration_s == pytest.approx(0.2)
        gates = evaluate_episode(
            recorded,
            load_limits(Path("config/synria_limits.yaml")),
            GateConfig(contract["min_episode_s"], contract["max_episode_s"]),
        )
        assert gates.failed_gates == []
    config["response_timeout_s"] = 10.0  # Explicit synthetic CPU budget, not a physical default.
    repository = Path(config["repository"])
    selection = repository / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "policy_id": config["policy_id"],
                "selected_step": 2,
                "seed": 7,
                "rule": "Fixed step 2 selected before training; never selected by probe metrics",
                "dataset_content_sha256": strict_content_hash(dataset),
                "probe_sha256": config["probe_sha256"],
                "physical_contract": config["expected_contract"],
            }
        )
    )
    commit(repository, selection)
    selected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = train(
        dataset, dataset / "physical_contract.json", tmp_path / "training", 7, 2, config
    )
    assert manifest["status"] == "completed" and manifest["completed_steps"] == 2
    assert manifest["git_sha"] == selected_head and manifest["upstream_version"] == version(
        "lerobot"
    )
    assert manifest["training_episode_indices"] == (0,) and manifest["held_out_episodes"] == [1]
    assert [entry["step"] for entry in manifest["checkpoints"]] == [1, 2]
    assert all(entry["evaluation_status"] == "evaluated" for entry in manifest["checkpoints"])
    np.testing.assert_allclose(manifest["training_stats"]["action"]["mean"][:6], 0.01, atol=1e-6)
    assert (
        strict_content_hash(dataset) == json.loads(selection.read_text())["dataset_content_sha256"]
    )
    evidence = repository / config["evidence_directory"]
    assert len(list((evidence / "evaluations").glob("*.json"))) == 2
    ledger = (repository / config["policy_ledger"]).read_text()
    assert sum(line.startswith("|") for line in ledger.splitlines()) == 1
    assert "response_timeout_s=10" in ledger

    registry = synthetic_registry(tmp_path / "registry.json", 0.1, 1)
    receipt = evidence / "checkpoint-step-000000002.json"
    identity = CheckpointIdentity.load(Path(manifest["checkpoints"][-1]["path"]), receipt, registry)
    model = LocalACTModel(identity, "cpu")
    with torch.inference_mode():
        restored_mean = model.postprocessor(torch.zeros((1, 6, 7)))
    np.testing.assert_allclose(
        restored_mean[0, 0].numpy(),
        manifest["training_stats"]["action"]["mean"],
        atol=1e-6,
    )
    saved_hash = strict_content_hash(identity.checkpoint)
    with ExitStack() as stack:
        endpoint = stack.enter_context(serve(identity, model))
        evaluation = fake_device(tmp_path / "evaluation-device")
        evaluation.config.update(
            command_period_s=0.4,
            task_registry=str(registry),
            d2_policy=specification(identity, receipt, endpoint),
        )
        register_fixed_protocol(
            evaluation.config,
            identity.metadata(),
            repository,
            selection_rule=f"Fixed step 2: selection.json at {selected_head}; no probe selection",
        )
        assert (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", selected_head, "HEAD"],
                cwd=repository,
                check=False,
            ).returncode
            == 0
        )
        evaluation.answers.extend(["sync-off", "armed"])
        for trial in range(20):
            evaluation.answers.extend(
                [
                    "yes",
                    "success" if trial < 14 else "failure",
                    *(["unknown"] * len(FUNNEL)),
                ]
            )
        evaluation.answers.append("unchanged")
        output = tmp_path / "evaluation"
        stats = run_fixed_evaluation(
            evaluation.config,
            repository,
            output,
            enable_motion=True,
            factory=io_factory(evaluation),
            clock=evaluation.clock,
            sleep=lambda seconds: evaluation.clock.sleep(seconds + 1e-9),
        )
        assert stats["session_status"] == "completed", stats
        assert stats["successes"] == 14 and stats["threshold_met"]
        assert stats["first_try"]["rate"] == 0.7 and stats["data_kind"] == "synthetic"
        assert stats["checkpoint_content_sha256"] == saved_hash
        assert len(evaluation.ros.goals) == 20
        assert evaluation.ros.publisher_topics == ["/fake_policy_targets"]
        assert evaluation.ros.action_clients == ["/fake_gripper/follow_joint_trajectory"]
        assert not evaluation.answers and "context" in evaluation.ros.closed
        trials = [json.loads(line) for line in (output / "EvalLog.jsonl").read_text().splitlines()][
            1:-1
        ]
        for trial in trials:
            grade = trial["attempts"][0]["operator_grade"]
            assert cv2.imread(grade["still"]).shape == (480, 640, 3)

        roll = fake_device(tmp_path / "roll-device")
        roll.config.update(command_period_s=0.4, task_registry=str(registry))
        roll.config["roll_skills"]["die_into_cup"] = specification(identity, receipt, endpoint)
        stub_models = []
        for task in TASK_IDS[1:]:
            directory = tmp_path / task
            directory.mkdir()
            stub_identity, stub_receipt, _ = checkpoint_fixture(directory, task)
            stub_manifest = copy.deepcopy(stub_identity.manifest)
            stub_manifest["policy_id"] = stub_manifest["configuration"]["policy_id"] = (
                task + "-stub"
            )
            for view in ("wrist", "front"):
                stub_manifest["model_feature_shapes"]["observation.images." + view] = [3, 32, 32]
            (stub_identity.checkpoint / "physical_policy_manifest.json").write_text(
                json.dumps(stub_manifest)
            )
            stub_receipt.write_text(
                json.dumps(
                    {
                        **stub_manifest,
                        "path": str(stub_identity.checkpoint),
                        "evaluation_status": "evaluated",
                        "checkpoint_content_sha256": strict_content_hash(stub_identity.checkpoint),
                    }
                )
            )
            stub_identity = CheckpointIdentity.load(
                stub_identity.checkpoint, stub_receipt, registry
            )
            stub = StubModel()
            stub.image_shapes = {
                "observation.images." + view: (32, 32, 3) for view in ("wrist", "front")
            }
            stub_models.append(stub)
            stub_endpoint = stack.enter_context(serve(stub_identity, stub))
            roll.config["roll_skills"][task] = specification(
                stub_identity, stub_receipt, stub_endpoint
            )
        roll.answers.extend(
            [
                "sync-off",
                "armed",
                "yes",
                "success",
                "yes",
                "success",
                "yes",
                "success",
                "4",
                "unchanged",
            ]
        )
        rolled = run_roll_skills(
            roll.config,
            repository,
            tmp_path / "roll",
            enable_motion=True,
            factory=io_factory(roll),
            clock=roll.clock,
            sleep=lambda seconds: roll.clock.sleep(seconds + 1e-9),
        )
        assert rolled["status"] == "completed", rolled
        assert rolled["roll"]["value"] == 4 and not rolled["d4_qualifying"]
        assert rolled["stage_status"] == "planned"
        assert rolled["skills"]["die_into_cup"]["checkpoint_content_sha256"] == saved_hash
        assert [rolled["skills"][task]["task_id"] for task in TASK_IDS] == list(TASK_IDS)
        assert len({rolled["skills"][task]["checkpoint_content_sha256"] for task in TASK_IDS}) == 3
        assert [stub.reset_count for stub in stub_models] == [1, 1]
        assert len(roll.ros.goals) == 3 and roll.ros.publisher_topics == ["/fake_policy_targets"]
        assert roll.ros.action_clients == ["/fake_gripper/follow_joint_trajectory"]
        assert not roll.answers and "context" in roll.ros.closed
    assert strict_content_hash(identity.checkpoint) == saved_hash
