from __future__ import annotations

import errno
import json
import subprocess
import threading
import time
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest
from test_synria_sessions import FakeIO
from test_task_registry import synthetic_task

from synria_lerobot.checkpoint_eval import (
    CheckpointEvaluator,
    PredictionBatch,
    TrialClipCapture,
    WebMWriter,
    load_probes,
    prediction_metrics,
    register_probes,
    render_report,
    run_physical_probes,
    strict_content_hash,
)
from synria_lerobot.physical_contract import ActionSource, PhysicalDatasetContract
from synria_lerobot.policy_client import FakePolicy
from synria_lerobot.sessions import OperatorAbort


def directory_link(path: Path, target: Path) -> None:
    try:
        path.symlink_to(target, target_is_directory=True)
    except NotImplementedError:
        pytest.skip("symbolic links are unsupported on this platform")
    except OSError as error:
        if getattr(error, "winerror", None) == 1314 or error.errno in {
            errno.EACCES,
            errno.EPERM,
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        }:
            pytest.skip("symbolic-link creation is unsupported or not permitted")
        raise


@pytest.mark.parametrize("fault", ["permission", "unsupported", "windows", "unexpected"])
def test_link_fixture_skips_only_platform_limitations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    error = OSError(errno.EIO, "unexpected filesystem error")
    if fault == "permission":
        error = PermissionError(errno.EPERM, "no permission")
    elif fault == "unsupported":
        error = OSError(errno.ENOTSUP, "unsupported")
    elif fault == "windows":
        error.winerror = 1314

    def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(Path, "symlink_to", fail)
    with pytest.raises(OSError if fault == "unexpected" else pytest.skip.Exception):
        directory_link(tmp_path / "alias", tmp_path / "target")


def commit(repository: Path, path: Path) -> None:
    subprocess.run(["git", "add", path.name], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test Operator",
            "-c",
            "user.email=operator@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Freeze synthetic probes",
        ],
        cwd=repository,
        check=True,
    )


@pytest.fixture
def probe_setup(tmp_path: Path) -> Any:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    contract = PhysicalDatasetContract(
        "50mm",
        ActionSource.NEXT_STATE,
        False,
        action_lookahead_steps=2,
        task_id="die_into_cup",
        task_definition=synthetic_task(20, 30),
    )
    (dataset / "physical_contract.json").write_text(json.dumps(contract.as_dict(fps=10)))
    (dataset / "synthetic_frames.bin").write_bytes(b"synthetic dataset fixture")
    probe = repository / "probes.json"
    config = json.loads(Path("config/synria_checkpoint_probes.json").read_text())
    config.update(
        probe_set_id="fixed",
        registered_by="synthetic fixture",
        registered_on="2026-10-08",
        dataset_content_sha256=strict_content_hash(dataset),
        task_id=contract.task_id,
        physical_contract=contract.as_dict(fps=10),
        held_out_episodes=[2],
        held_out_frame_counts={"2": 3},
        capture=dict(fps=30, max_duration_s=2, shutdown_timeout_s=0.5),
        physical_trials=[fixed_probe_trial()],
    )
    probe.write_text(json.dumps(config))
    digest = register_probes(probe)
    commit(repository, probe)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"synthetic checkpoint fixture")

    def reader(root: Path) -> dict[int, int]:
        return {0: 3, 1: 3, 2: 3}

    return NS(
        repository=repository,
        probe=probe,
        digest=digest,
        dataset=dataset,
        checkpoint=checkpoint,
        config=config,
        reader=reader,
    )


def fixed_probe_trial(identifier: str = "trial-a", task_id: str = "die_into_cup") -> dict:
    return {
        "trial_id": identifier,
        "task_id": task_id,
        "scene": dict(
            scene_id="synthetic-fixed",
            cup_mark="fixed cup mark",
            die_start_zone="marked zone",
            landing_tray="fixed tray",
            start_state="operator-declared fixed start",
            die_position_in_zone="centre",
            die_face_up=1,
        ),
    }


def evaluator(fixture: Any, training_ids: tuple[int, ...] = (0, 1)) -> CheckpointEvaluator:
    return CheckpointEvaluator(
        fixture.probe,
        fixture.repository,
        fixture.digest,
        fixture.dataset,
        training_ids,
        metadata_reader=fixture.reader,
    )


@pytest.mark.parametrize("configured", [False, True])
def test_checkpoint_evaluation_refuses_smoke_dataset(probe_setup: Any, configured: bool) -> None:
    from test_task_registry import REGISTRY

    from synria_lerobot.task_registry import load_task_registry

    task = synthetic_task(20, 30) if configured else load_task_registry(REGISTRY)["die_into_cup"]
    contract = PhysicalDatasetContract(
        "50mm",
        ActionSource.NEXT_STATE,
        False,
        task_id=task.task_id,
        task_definition=task,
        recording_purpose="disposable_smoke",
    )
    (probe_setup.dataset / "physical_contract.json").write_text(
        json.dumps(contract.as_dict(fps=10))
    )
    probe_setup.config["dataset_content_sha256"] = strict_content_hash(probe_setup.dataset)
    probe_setup.probe.write_text(json.dumps(probe_setup.config))
    probe_setup.digest = register_probes(probe_setup.probe)
    commit(probe_setup.repository, probe_setup.probe)
    with pytest.raises(ValueError, match="disposable smoke"):
        evaluator(probe_setup)


def batches() -> list[PredictionBatch]:
    actions = np.zeros((3, 2, 7))
    actions[2, 1] = 1000
    return [
        PredictionBatch(
            {"observation.state": np.ones((3, 7))},
            actions,
            np.array([[False, False], [False, False], [False, True]]),
            (2, 2, 2),
            (0, 1, 2),
        )
    ]


def test_probe_template_cannot_be_registered_or_qualify() -> None:
    from synria_lerobot.checkpoint_eval import validate_probes

    with pytest.raises(ValueError):
        validate_probes(json.loads(Path("config/synria_checkpoint_probes.json").read_text()))


@pytest.mark.parametrize(
    "change",
    [
        "legacy",
        "task",
        "square_goal",
        "second_task",
        "varying_geometry",
        "face_boolean",
    ],
)
def test_fixed_probes_refuse_legacy_goal_or_conflicting_task_scene(
    probe_setup: Any,
    change: str,
) -> None:
    import copy

    from synria_lerobot.checkpoint_eval import validate_probes

    config = copy.deepcopy(probe_setup.config)
    if change == "legacy":
        config["version"] = "synria_checkpoint_probes_v1"
    elif change == "task":
        config["task_id"] = "cup_return"
    elif change == "square_goal":
        config["physical_trials"][0]["target_square"] = "B"
    elif change == "second_task":
        config["physical_trials"][0]["task_id"] = "roll_and_dump"
    elif change == "face_boolean":
        config["physical_trials"][0]["scene"]["die_face_up"] = True
    else:
        other = fixed_probe_trial("trial-b")
        other["scene"]["cup_mark"] = "different mark"
        config["physical_trials"].append(other)
    with pytest.raises(ValueError):
        validate_probes(config)


def test_probes_require_committed_unchanged_matching_hash(probe_setup: Any) -> None:
    fixture = probe_setup
    assert load_probes(fixture.probe, fixture.repository, fixture.digest)["held_out_episodes"] == [
        2
    ]
    with pytest.raises(ValueError, match="hash"):
        load_probes(fixture.probe, fixture.repository, "0" * 64)
    fixture.probe.write_text(fixture.probe.read_text() + " ")
    with pytest.raises(ValueError, match="committed"):
        load_probes(fixture.probe, fixture.repository, fixture.digest)


@pytest.mark.parametrize("ids", [(2,), (0,), (0, 1, 999), (), (False, 1)])
def test_training_partition_is_complete_real_and_disjoint(probe_setup: Any, ids: tuple) -> None:
    with pytest.raises(ValueError):
        evaluator(probe_setup, ids)


def test_frozen_frame_counts_match_actual_metadata(probe_setup: Any) -> None:
    probe_setup.reader = lambda root: {0: 3, 1: 3, 2: 4}
    with pytest.raises(ValueError, match="matching lengths"):
        evaluator(probe_setup)


def test_masked_per_axis_metrics_never_mix_joint_and_gripper_units() -> None:
    def predict(observations: dict) -> np.ndarray:
        assert list(observations) == ["observation.state"]
        result = np.full((3, 2, 7), 0.2)
        result[..., 6] = 0.001
        result[2, 1] = -999
        return result

    result = prediction_metrics(batches(), predict, {2: 3})
    assert result["valid_action_count"] == 5 and result["held_out_frames"] == 3
    for index in range(1, 7):
        assert result["axes"][f"Joint{index}"]["mae"] == pytest.approx(0.2)
        assert result["axes"][f"Joint{index}"]["unit"] == "rad"
    assert result["axes"]["Gripper"]["mae"] == pytest.approx(0.001)
    assert result["axes"]["Gripper"]["unit"] == "m"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "padding", "nan", "shape", "labels"])
def test_corrupt_held_out_evaluation_fails(fault: str) -> None:
    values = batches()

    def predict(observations: dict) -> np.ndarray:
        if fault == "nan":
            return np.full((3, 2, 7), np.nan)
        return np.zeros((3, 7) if fault == "shape" else (3, 2, 7))

    if fault == "missing":
        values = []
    elif fault == "duplicate":
        values *= 2
    elif fault == "padding":
        values[0].action_is_pad[0, 0] = True
    elif fault == "labels":
        values[0].observations["action"] = values[0].actions
    with pytest.raises(ValueError):
        prediction_metrics(values, predict, {2: 3})


@pytest.mark.parametrize("fault", ["prediction", "changed_dataset", "changed_probe"])
def test_every_failed_saved_checkpoint_keeps_immutable_record_and_timeline(
    probe_setup: Any,
    tmp_path: Path,
    fault: str,
) -> None:
    evaluation = evaluator(probe_setup)
    output, timeline = tmp_path / "evaluations", tmp_path / "timeline.jsonl"
    if fault == "changed_dataset":
        (probe_setup.dataset / "synthetic_frames.bin").write_bytes(b"changed fixture")
    elif fault == "changed_probe":
        probe_setup.probe.write_text(probe_setup.probe.read_text() + " ")

    def predict(observations: dict) -> np.ndarray:
        return np.full((3, 2, 7), np.nan)

    with pytest.raises(RuntimeError, match="evidence retained"):
        evaluation.evaluate(
            probe_setup.checkpoint, "policy-1", 10, batches(), predict, output, timeline
        )
    record = json.loads(next(output.glob("*.json")).read_text())
    assert record["status"] == "failed" and record["metrics"] is None and record["error"]
    assert len(record["checkpoint_content_sha256"]) == len(record["dataset_content_sha256"]) == 64
    assert (
        record["held_out_episodes"] == [2]
        and record["physical_contract"]["action_lookahead_steps"] == 2
    )
    event = json.loads(timeline.read_text())
    assert event["kind"] == "checkpoint_eval" and event["status"] == "failed"
    with pytest.raises(FileExistsError):
        evaluation.evaluate(
            probe_setup.checkpoint, "policy-1", 10, batches(), predict, output, timeline
        )


def test_missing_empty_and_symlinked_content_never_gets_a_plausible_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        strict_content_hash(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        strict_content_hash(empty)
    (empty / "zero").touch()
    with pytest.raises(ValueError):
        strict_content_hash(empty)
    parent = tmp_path / "actual"
    content = parent / "content"
    content.mkdir(parents=True)
    (content / "data").write_bytes(b"content")
    alias = tmp_path / "alias"
    directory_link(alias, parent)
    with pytest.raises(ValueError):
        strict_content_hash(alias / "content")


class FakeClipWriter:
    def __init__(self, path: Path, fps: float, shape: tuple) -> None:
        self.path, self.fps, self.shape = path, fps, shape
        self.stream = path.open("xb")
        self.closed = False

    def write(self, image: np.ndarray) -> None:
        self.stream.write(image.tobytes())
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()
        self.closed = True


def make_clips(tmp_path: Path, io: FakeIO, **kwargs: Any) -> TrialClipCapture:
    return TrialClipCapture(
        io.observe,
        tmp_path / "raw-clips",
        tmp_path / "repository",
        fps=100,
        max_duration_s=1,
        shutdown_timeout_s=0.2,
        writer_factory=FakeClipWriter,
        **kwargs,
    )


def test_clip_worker_records_both_views_through_operator_wait_with_timestamps(
    tmp_path: Path,
) -> None:
    io = FakeIO(tmp_path)
    capture = make_clips(tmp_path, io)
    capture.start()
    time.sleep(0.05)
    label_stamp = time.monotonic()
    capture.stop()
    record = capture.evidence(label_stamp)
    assert record["complete"] and record["frame_count"] >= 3
    assert record["requested_fps"] == 100 and record["achieved_sample_rate_hz"] > 0
    assert all(record["views"][name]["sha256"] for name in ("wrist", "front"))
    assert all(writer.closed for writer in capture._writers.values())
    assert record["final_frames"]["front"]["source_monotonic_s"] >= label_stamp
    assert record["label_nearest_sample_index"] < record["frame_count"]
    rows = [json.loads(line) for line in Path(record["timestamps_path"]).read_text().splitlines()]
    assert len(rows) == record["frame_count"]
    assert record["achieved_sample_rate_hz"] == pytest.approx(
        (len(rows) - 1) / (rows[-1]["sample_monotonic_s"] - rows[0]["sample_monotonic_s"])
    )


def test_clip_start_failure_closes_first_writer_and_keeps_missing_view_reason(
    tmp_path: Path,
) -> None:
    io = FakeIO(tmp_path)
    opened = []

    def broken_front(path: Path, fps: float, shape: tuple) -> FakeClipWriter:
        if path.name == "front.webm":
            raise RuntimeError("synthetic front encoder failure")
        writer = FakeClipWriter(path, fps, shape)
        opened.append(writer)
        return writer

    capture = TrialClipCapture(
        io.observe,
        tmp_path / "clips",
        tmp_path / "repository",
        fps=30,
        max_duration_s=1,
        shutdown_timeout_s=0.2,
        writer_factory=broken_front,
    )
    with pytest.raises(RuntimeError, match="capture failed"):
        capture.start()
    with pytest.raises(RuntimeError):
        capture.stop()
    evidence = capture.evidence()
    assert not evidence["complete"] and evidence["views"]["front"]["missing_reason"]
    assert opened[0].closed and opened[0].path.exists()


@pytest.mark.parametrize("fault", ["source", "duration"])
def test_clip_fault_and_duration_cap_preserve_partial_video(tmp_path: Path, fault: str) -> None:
    io = FakeIO(tmp_path)
    source = io.observe
    calls = []

    def observe(task: str) -> Any:
        calls.append(task)
        if fault == "source" and len(calls) > 2:
            raise RuntimeError("synthetic observation failure")
        return source(task)

    capture = TrialClipCapture(
        observe,
        tmp_path / "clips",
        tmp_path / "repository",
        fps=100,
        max_duration_s=0.03,
        shutdown_timeout_s=0.2,
        writer_factory=FakeClipWriter,
    )
    capture.start()
    assert capture._finished.wait(0.5)
    with pytest.raises(RuntimeError, match="capture failed"):
        capture.stop()
    evidence = capture.evidence()
    assert not evidence["complete"] and evidence["error"]
    assert all(value["sha256"] for value in evidence["views"].values())


def test_hung_clip_worker_retains_writers_until_bounded_retry(tmp_path: Path) -> None:
    io = FakeIO(tmp_path)
    release = threading.Event()

    def observe(task: str) -> Any:
        release.wait()
        return io.observe(task)

    capture = TrialClipCapture(
        observe,
        tmp_path / "clips",
        tmp_path / "repository",
        fps=100,
        max_duration_s=1,
        shutdown_timeout_s=0.02,
        writer_factory=FakeClipWriter,
    )
    try:
        with pytest.raises(TimeoutError):
            capture.start()
        with pytest.raises(RuntimeError, match="still active"):
            capture.stop()
        assert not capture.stopped and not capture.evidence()["complete"]
    finally:
        release.set()
        capture._thread.join(0.5)
        capture.stop()
    assert capture.stopped


def physical_setup(fixture: Any, tmp_path: Path) -> tuple[dict, Path, FakeIO]:
    from test_synria_roll_skills import RollIO, roll_config

    from synria_lerobot.act_training import validate_config
    from synria_lerobot.policy_server import CheckpointIdentity

    config, _ = roll_config(tmp_path)
    specification = dict(config["roll_skills"]["die_into_cup"], max_steps=1)
    config["probe_policy"] = specification
    fixture.checkpoint = Path(specification["checkpoint"])
    manifest_path = fixture.checkpoint / "physical_policy_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        physical_contract=fixture.config["physical_contract"],
        dataset_content_sha256=fixture.config["dataset_content_sha256"],
        probe_sha256=fixture.digest,
        step=10,
        completed_steps=10,
    )
    manifest["configuration"].update(
        expected_contract=fixture.config["physical_contract"],
        probe_sha256=fixture.digest,
    )
    manifest["cadence"] = validate_config(manifest["configuration"], manifest["physical_contract"])
    manifest_path.write_text(json.dumps(manifest))
    registry = Path(config["task_registry"])
    tasks = json.loads(registry.read_text())
    tasks["tasks"][0] = synthetic_task(20, 30).as_dict()
    registry.write_text(json.dumps(tasks))
    receipt = Path(specification["receipt"])
    receipt.write_text(
        json.dumps(
            {
                **manifest,
                "path": str(fixture.checkpoint),
                "evaluation_status": "evaluated",
                "checkpoint_content_sha256": strict_content_hash(fixture.checkpoint),
            }
        )
    )
    CheckpointIdentity.load(fixture.checkpoint, receipt, registry)
    output = tmp_path / "checkpoint-evaluations"
    saved = evaluator(fixture).evaluate(
        fixture.checkpoint,
        manifest["policy_id"],
        10,
        batches(),
        lambda observations: np.zeros((3, 2, 7)),
        output,
        tmp_path / "timeline.jsonl",
    )
    config["provenance"].update(
        policy_id=manifest["policy_id"], checkpoint_sha256=saved["checkpoint_content_sha256"]
    )
    io = RollIO(tmp_path)
    io.confirm_skill = lambda task: True
    io.recover_skill = lambda task: False
    return config, next(output.glob("*.json")), io


def probe_transport(config: dict, policy: Any = None) -> Any:
    from synria_lerobot.policy_server import CheckpointIdentity

    specification = config["probe_policy"]
    identity = CheckpointIdentity.load(
        Path(specification["checkpoint"]),
        Path(specification["receipt"]),
        Path(config["task_registry"]),
    )
    return NS(
        metadata=lambda timeout: identity.metadata(),
        request=(policy or FakePolicy((0,) * 7)).request,
    )


def physical_run(
    fixture: Any, tmp_path: Path, config: dict, saved: Path, io: FakeIO, **kwargs: Any
) -> dict:
    kwargs["transport"] = probe_transport(config, kwargs.get("transport"))
    return run_physical_probes(
        fixture.probe,
        fixture.digest,
        saved,
        fixture.dataset,
        config,
        fixture.repository,
        tmp_path / "physical-probes",
        tmp_path / "raw-videos",
        enable_motion=True,
        factory=lambda config: io,
        writer_factory=FakeClipWriter,
        **kwargs,
    )


def test_physical_probe_captures_full_trial_label_and_both_external_clips(
    probe_setup: Any,
    tmp_path: Path,
) -> None:
    config, saved, io = physical_setup(probe_setup, tmp_path)
    grade = io.grade_skill

    def delayed_grade(move: Any) -> Any:
        time.sleep(0.08)
        return grade(move)

    io.grade_skill = delayed_grade
    report = physical_run(probe_setup, tmp_path, config, saved, io, transport=FakePolicy((0,) * 7))
    assert report["status"] == "completed" and io.closed
    attempt = json.loads(Path(report["attempts"][0]["path"]).read_text())
    assert attempt["status"] == "success" and attempt["attempted"]
    assert attempt["scene_confirmed"] and attempt["steps"] == 1
    assert attempt["task_id"] == "die_into_cup"
    assert attempt["operator_grade"]["label"] == "success"
    assert attempt["clips"]["frame_count"] >= 3
    assert len(io.offers) == 1
    assert report["endpoint_checkpoint_identity"]["task_id"] == "die_into_cup"
    for media in attempt["clips"]["views"].values():
        assert media["sha256"] and not Path(media["path"]).is_relative_to(probe_setup.repository)
    page = render_report(
        [saved], [tmp_path / "physical-probes/probe_run.json"], tmp_path / "page.html"
    )
    assert "trial-a" in page and "file://" in page


@pytest.mark.parametrize("fault", ["policy", "early_abort", "label_abort", "camera"])
def test_failed_and_aborted_physical_attempts_remain_recorded(
    probe_setup: Any,
    tmp_path: Path,
    fault: str,
) -> None:
    config, saved, io = physical_setup(probe_setup, tmp_path)
    if fault == "early_abort":
        io.abort_pending = lambda: True
    elif fault == "label_abort":

        def aborted(move: Any) -> Any:
            raise OperatorAbort("synthetic label abort")

        io.grade_skill = aborted
    elif fault == "camera":

        def failed(task: str) -> Any:
            raise RuntimeError("synthetic camera failure")

        io.observe = failed
    policy = (
        NS(request=lambda payload, timeout_s: None) if fault == "policy" else FakePolicy((0,) * 7)
    )
    report = physical_run(probe_setup, tmp_path, config, saved, io, transport=policy)
    assert report["status"] in {"failed", "aborted"} and io.closed
    attempt = json.loads(Path(report["attempts"][0]["path"]).read_text())
    assert attempt["status"] == ("aborted" if "abort" in fault else "failed")
    assert attempt["operator_grade"] is None and attempt["errors"]
    assert attempt["object_success_limitation"] == report["object_success_limitation"]
    assert attempt["clips"] or attempt["missing_clips_reason"]


def test_declined_probe_scene_is_kept_without_offering_motion(
    probe_setup: Any, tmp_path: Path
) -> None:
    config, saved, io = physical_setup(probe_setup, tmp_path)
    io.confirm_skill = lambda task: False
    report = physical_run(probe_setup, tmp_path, config, saved, io)
    attempt = json.loads(Path(report["attempts"][0]["path"]).read_text())
    assert report["status"] == attempt["status"] == "aborted"
    assert not attempt["scene_confirmed"] and not attempt["attempted"] and not io.offers
    assert attempt["clips"] and io.closed


def test_diagnostic_record_and_completion_receipt_must_bind_same_task_before_io(
    probe_setup: Any,
    tmp_path: Path,
) -> None:
    config, saved, io = physical_setup(probe_setup, tmp_path)
    record = json.loads(saved.read_text())
    record["physical_contract"]["task_id"] = "cup_return"
    saved.write_text(json.dumps(record))
    calls = []
    with pytest.raises(ValueError, match="policy/checkpoint differs"):
        run_physical_probes(
            probe_setup.probe,
            probe_setup.digest,
            saved,
            probe_setup.dataset,
            config,
            probe_setup.repository,
            tmp_path / "reports",
            tmp_path / "media",
            factory=lambda settings: calls.append(settings),
            transport=probe_transport(config),
        )
    assert not calls and not io.offers and not (tmp_path / "reports").exists()


def test_five_checkpoints_scripted_improvement_keep_every_record_timeline_and_page(
    probe_setup: Any,
    tmp_path: Path,
) -> None:
    evaluation = evaluator(probe_setup)
    output, timeline = tmp_path / "evaluations", tmp_path / "timeline.jsonl"
    for index in range(5):
        (probe_setup.checkpoint / "weights.bin").write_bytes(
            f"synthetic checkpoint {index}".encode()
        )
        error = (5 - index) * 0.01
        evaluation.evaluate(
            probe_setup.checkpoint,
            "policy-1",
            (index + 1) * 10,
            batches(),
            lambda observations, error=error: np.full((3, 2, 7), error),
            output,
            timeline,
        )
    paths = sorted(output.glob("*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    assert len(paths) == len(timeline.read_text().splitlines()) == 5
    for line in timeline.read_text().splitlines():
        event = json.loads(line)
        assert {"t_utc", "kind", "title", "artifact", "numbers"} <= event.keys()
        assert event["numbers"]["held_out_frames"] == 3
        assert event["numbers"]["failed_evaluations"] == 0
    errors = [record["metrics"]["axes"]["Joint1"]["mae"] for record in records]
    assert errors == sorted(errors, reverse=True)
    page = render_report(paths, [], tmp_path / "progress.html")
    assert all(f'"step": {step}' in page for step in (10, 20, 30, 40, 50))
    assert "Diagnostic only" in page and "Prediction error is not object success" in page
    assert 'id="step"' in page and 'id="axis"' in page and 'id="baseline"' in page


def test_report_escapes_failed_records_without_filtering_them(
    probe_setup: Any, tmp_path: Path
) -> None:
    output, timeline = tmp_path / "evaluations", tmp_path / "timeline.jsonl"

    def failed(observations: dict) -> Any:
        raise ValueError('</script><img src=x onerror="alert(1)"> & @@RECORDS@@')

    with pytest.raises(RuntimeError):
        evaluator(probe_setup).evaluate(
            probe_setup.checkpoint, "policy-1", 10, batches(), failed, output, timeline
        )
    page = render_report(list(output.glob("*.json")), [], tmp_path / "failed.html")
    tags = []

    class Inspect(HTMLParser):
        def handle_starttag(self, tag: str, attrs: list) -> None:
            tags.append((tag, dict(attrs)))

    Inspect().feed(page)
    assert sum(tag == "script" for tag, _ in tags) == 2
    assert not any(
        tag in {"img", "iframe", "object"} or "onerror" in attrs or "src" in attrs
        for tag, attrs in tags
    )
    assert '"status": "failed"' in page and "&lt;/script&gt;" in page
    assert "@@RECORDS@@" in page  # User text is not recursively treated as a template.


@pytest.mark.parametrize("target", ["dataset", "checkpoint", "probe", "symlink"])
@pytest.mark.parametrize("artifact", ["output", "timeline"])
def test_evaluation_outputs_cannot_mutate_frozen_inputs(
    probe_setup: Any, tmp_path: Path, target: str, artifact: str
) -> None:
    evaluation = evaluator(probe_setup)
    output, timeline = tmp_path / "evaluations", tmp_path / "timeline.jsonl"
    roots = (probe_setup.dataset, probe_setup.checkpoint)
    before = [strict_content_hash(root) for root in roots]
    probe_before = probe_setup.probe.read_bytes()
    if target == "probe":
        bad = probe_setup.probe
    elif target == "symlink":
        link = tmp_path / "alias"
        directory_link(link, probe_setup.dataset)
        bad = link / "new-output"
    else:
        bad = getattr(probe_setup, target) / "new-output"
    with pytest.raises(ValueError, match="artifact destination"):
        evaluation.evaluate(
            probe_setup.checkpoint,
            "policy-1",
            10,
            batches(),
            lambda observations: np.zeros((3, 2, 7)),
            bad if artifact == "output" else output,
            bad if artifact == "timeline" else timeline,
        )
    assert [strict_content_hash(root) for root in roots] == before
    assert probe_setup.probe.read_bytes() == probe_before
    assert not output.exists() and not timeline.exists()


@pytest.mark.parametrize("target", ["dataset", "checkpoint", "probe", "symlink"])
def test_physical_output_refused_before_adapter_or_input_tree_changes(
    probe_setup: Any, tmp_path: Path, target: str
) -> None:
    config, saved, io = physical_setup(probe_setup, tmp_path)
    roots = (probe_setup.dataset, probe_setup.checkpoint)
    before = [strict_content_hash(root) for root in roots]
    probe_before = probe_setup.probe.read_bytes()
    if target == "probe":
        output = probe_setup.probe
    elif target == "symlink":
        alias = tmp_path / "alias"
        directory_link(alias, probe_setup.dataset)
        output = alias / "reports"
    else:
        output = getattr(probe_setup, target) / "reports"
    calls = []
    with pytest.raises(ValueError, match="artifact destination"):
        run_physical_probes(
            probe_setup.probe,
            probe_setup.digest,
            saved,
            probe_setup.dataset,
            config,
            probe_setup.repository,
            output,
            tmp_path / "media",
            enable_motion=True,
            factory=lambda settings: calls.append(settings),
        )
    assert not calls and not io.offers
    assert [strict_content_hash(root) for root in roots] == before
    assert probe_setup.probe.read_bytes() == probe_before


def test_silent_missing_camera_clip_fails_attempt_but_preserves_operator_label(
    probe_setup: Any, tmp_path: Path
) -> None:
    config, saved, io = physical_setup(probe_setup, tmp_path)

    class MissingViewWriter(FakeClipWriter):
        def __init__(self, path: Path, fps: float, shape: tuple) -> None:
            super().__init__(path, fps, shape)
            self.path = path

        def close(self) -> None:
            super().close()
            if self.path.stem == "front":
                self.path.unlink()

    report = run_physical_probes(
        probe_setup.probe,
        probe_setup.digest,
        saved,
        probe_setup.dataset,
        config,
        probe_setup.repository,
        tmp_path / "physical-probes",
        tmp_path / "raw-videos",
        enable_motion=True,
        factory=lambda settings: io,
        transport=probe_transport(config),
        writer_factory=MissingViewWriter,
    )
    attempt = json.loads(Path(report["attempts"][0]["path"]).read_text())
    assert report["status"] == attempt["status"] == "failed"
    assert attempt["operator_grade"]["label"] == "success"
    assert not attempt["clips"]["complete"]
    assert attempt["clips"]["views"]["front"]["missing_reason"]
    assert attempt["clips"]["views"]["wrist"]["sha256"]


def test_retry_keeps_every_attempt_and_distinct_media(probe_setup: Any, tmp_path: Path) -> None:
    from dataclasses import replace

    config, saved, io = physical_setup(probe_setup, tmp_path)
    config["max_attempts"] = 2
    grade = io.grade_skill
    labels = iter(("failure", "success"))
    io.grade_skill = lambda task: replace(grade(task), label=next(labels))
    io.recover_skill = lambda task: True
    report = physical_run(probe_setup, tmp_path, config, saved, io, transport=FakePolicy((0,) * 7))
    attempts = [json.loads(Path(ref["path"]).read_text()) for ref in report["attempts"]]
    assert [attempt["status"] for attempt in attempts] == ["failure", "success"]
    assert len({attempt["clips"]["views"]["wrist"]["path"] for attempt in attempts}) == 2
    assert all(attempt["clips"]["complete"] for attempt in attempts)


def test_probe_read_only_preflight_creates_no_sink_or_capture(
    probe_setup: Any, tmp_path: Path
) -> None:
    config, saved, io = physical_setup(probe_setup, tmp_path)
    report = run_physical_probes(
        probe_setup.probe,
        probe_setup.digest,
        saved,
        probe_setup.dataset,
        config,
        probe_setup.repository,
        tmp_path / "read-only",
        tmp_path / "media",
        factory=lambda settings: io,
        transport=probe_transport(config),
    )
    assert report["status"] == "read_only" and io.closed
    assert io.preflight_calls == 1 and io.sinks == io.authorizations == 0
    assert not (tmp_path / "media").exists() and not report["attempts"]


def test_playback_refuses_changed_local_media(probe_setup: Any, tmp_path: Path) -> None:
    config, saved, io = physical_setup(probe_setup, tmp_path)
    report = physical_run(probe_setup, tmp_path, config, saved, io, transport=FakePolicy((0,) * 7))
    attempt = json.loads(Path(report["attempts"][0]["path"]).read_text())
    Path(attempt["clips"]["views"]["wrist"]["path"]).write_bytes(b"changed synthetic clip")
    with pytest.raises(ValueError, match="media content changed"):
        render_report(
            [saved], [tmp_path / "physical-probes/probe_run.json"], tmp_path / "page.html"
        )
    assert not (tmp_path / "page.html").exists()


def test_webm_writer_encodes_and_reloads_synthetic_file_only(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "synthetic.webm"
    image = np.full((32, 48, 3), (200, 50, 10), dtype=np.uint8)
    writer = WebMWriter(path, 15, image.shape)
    try:
        for _ in range(3):
            writer.write(image)
    finally:
        writer.close()
    assert path.read_bytes().startswith(b"\x1a\x45\xdf\xa3")
    capture = cv2.VideoCapture(str(path))  # Synthetic file, never a camera/device.
    try:
        ok, bgr = capture.read()
        assert ok and bgr.shape == (32, 48, 3)
        np.testing.assert_allclose(bgr[0, 0], [10, 50, 200], atol=8)
    finally:
        capture.release()


@pytest.mark.parametrize("target", ["dataset", "checkpoint", "probe"])
def test_changed_input_during_prediction_retains_failed_record_without_metrics(
    probe_setup: Any, tmp_path: Path, target: str
) -> None:
    evaluation = evaluator(probe_setup)

    def changing_predictor(observations: dict) -> np.ndarray:
        if target == "probe":
            probe_setup.probe.write_text(probe_setup.probe.read_text() + " ")
        else:
            (getattr(probe_setup, target) / "changed.bin").write_bytes(b"changed")
        return np.zeros((3, 2, 7))

    with pytest.raises(RuntimeError, match="evidence retained"):
        evaluation.evaluate(
            probe_setup.checkpoint,
            "policy-1",
            10,
            batches(),
            changing_predictor,
            tmp_path / "evaluations",
            tmp_path / "timeline.jsonl",
        )
    saved = json.loads(next((tmp_path / "evaluations").glob("*.json")).read_text())
    assert saved["status"] == "failed" and saved["metrics"] is None


def test_capture_failure_during_policy_request_prevents_command_offer(
    probe_setup: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import synria_lerobot.checkpoint_eval as module

    config, saved, io = physical_setup(probe_setup, tmp_path)
    fail_capture = threading.Event()
    observe = io.observe
    workers = []

    def capture_factory(*args: Any, **kwargs: Any) -> TrialClipCapture:
        capture = TrialClipCapture(*args, **kwargs)
        workers.append(capture)
        return capture

    def observation(task: str) -> Any:
        if threading.current_thread().name == "synria-probe-clips" and fail_capture.is_set():
            raise RuntimeError("synthetic camera failed while policy response was pending")
        return observe(task)

    class Transport:
        def request(self, payload: dict, timeout_s: float) -> dict:
            fail_capture.set()
            assert workers[0]._finished.wait(0.1)
            return FakePolicy((0,) * 7).request(payload, timeout_s)

    io.observe = observation
    monkeypatch.setattr(module, "TrialClipCapture", capture_factory)
    report = physical_run(probe_setup, tmp_path, config, saved, io, transport=Transport())
    assert report["status"] == "failed" and io.closed and not io.offers
    attempt = json.loads(Path(report["attempts"][0]["path"]).read_text())
    assert attempt["attempted"] and not attempt["motion_completed"]
    assert not attempt["clips"]["complete"]
    assert "trial camera capture failed" in attempt["errors"]


def test_live_clip_worker_retains_session_sources_and_failed_attempt(
    probe_setup: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import synria_lerobot.checkpoint_eval as module

    config, saved, io = physical_setup(probe_setup, tmp_path)
    release = threading.Event()
    observe = io.observe
    workers = []

    def blocked(task: str) -> Any:
        assert release.wait(3)
        return observe(task)

    def capture_factory(*args: Any, **kwargs: Any) -> TrialClipCapture:
        capture = TrialClipCapture(*args, **kwargs)
        workers.append(capture)
        return capture

    io.observe = blocked
    monkeypatch.setattr(module, "TrialClipCapture", capture_factory)
    try:
        report = physical_run(
            probe_setup, tmp_path, config, saved, io, transport=FakePolicy((0,) * 7)
        )
        assert report["status"] == "failed" and not io.closed
        assert "resources retained" in " ".join(report["errors"])
        attempt = json.loads(Path(report["attempts"][0]["path"]).read_text())
        assert not attempt["attempted"] and not attempt["clips"]["complete"]
    finally:
        release.set()
        for capture in workers:
            capture.stop()
        io.close()


@pytest.mark.parametrize("target", ["dataset", "checkpoint"])
def test_playback_cannot_modify_hashed_inputs(
    probe_setup: Any, tmp_path: Path, target: str
) -> None:
    _, saved, _ = physical_setup(probe_setup, tmp_path)
    root = getattr(probe_setup, target)
    before = strict_content_hash(root)
    with pytest.raises(ValueError, match="frozen inputs"):
        render_report([saved], [], root / "playback.html")
    assert strict_content_hash(root) == before
