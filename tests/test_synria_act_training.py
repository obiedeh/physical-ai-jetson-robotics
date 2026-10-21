"""Training preflight tests do not import a model or touch any robot interface."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_task_registry import synthetic_task

from synria_lerobot.act_training import (
    LIMITS_FILE,
    _save_upstream_checkpoint,
    action_bounds,
    clamp_actions,
    grasp_phase_frame_weights,
    perturb_state,
    smooth_actions,
    train,
    validate_action_limit_margin,
    validate_action_smoothing,
    validate_config,
    validate_dataloader_workers,
    validate_episode_weights,
    validate_grasp_phase,
    validate_state_noise,
)
from synria_lerobot.physical_contract import ActionSource, PhysicalDatasetContract


@pytest.mark.parametrize("signature", ["released", "development", "kwargs", "positional_only"])
def test_checkpoint_save_detects_named_keyword_and_unwraps_once(
    tmp_path: Path, signature: str,
) -> None:
    wrapped, model, optimizer, config, pre, post = (object() for _ in range(6))
    unwrap_calls = []
    calls = []

    def unwrap(value: object) -> object:
        unwrap_calls.append(value)
        return model

    accelerator = SimpleNamespace(unwrap_model=unwrap)

    def released(path, step, cfg, policy, opt, *, preprocessor, postprocessor):
        calls.append((path, step, cfg, policy, opt, preprocessor, postprocessor, {}))

    def development(path, step, cfg, policy, opt, *, preprocessor, postprocessor, accelerator):
        calls.append((path, step, cfg, policy, opt, preprocessor, postprocessor,
                      {"accelerator": accelerator}))

    def kwargs_api(path, step, cfg, policy, opt, *, preprocessor, postprocessor, **kwargs):
        calls.append((path, step, cfg, policy, opt, preprocessor, postprocessor, kwargs))

    def positional_only(
        path, step, cfg, policy, opt, accelerator=None, /, *, preprocessor, postprocessor,
    ):
        assert accelerator is None
        calls.append((path, step, cfg, policy, opt, preprocessor, postprocessor, {}))

    save = {"released": released, "development": development,
            "kwargs": kwargs_api, "positional_only": positional_only}[signature]
    _save_upstream_checkpoint(
        save, tmp_path, 2, config, wrapped, optimizer,
        preprocessor=pre, postprocessor=post, accelerator=accelerator,
    )
    assert unwrap_calls == [wrapped]
    expected = {"accelerator": accelerator} if signature == "development" else {}
    assert calls == [(tmp_path, 2, config, model, optimizer, pre, post, expected)]


def test_checkpoint_save_internal_type_error_is_not_retried_or_replaced(tmp_path: Path) -> None:
    error = TypeError("synthetic failure inside save")
    calls = []

    def save(*args, accelerator, **kwargs):
        calls.append((args, kwargs))
        raise error

    with pytest.raises(TypeError) as caught:
        _save_upstream_checkpoint(
            save, tmp_path, 1, object(), object(), object(), preprocessor=None,
            postprocessor=None, accelerator=SimpleNamespace(unwrap_model=lambda value: value),
        )
    assert caught.value is error
    assert len(calls) == 1


def settings(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = PhysicalDatasetContract("50mm", ActionSource.NEXT_STATE, False,
        task_id="die_into_cup", task_definition=synthetic_task(0.1, 1),
    ).as_dict(fps=15)
    config = json.loads(Path("config/synria_act_training.json").read_text())
    config.update(
        policy_id="synthetic-policy",
        dataset_repo_id="local/synthetic",
        expected_contract=contract,
        repository=str(tmp_path / "repository"),
        probe_set="probes.json",
        probe_sha256="a" * 64,
        evidence_directory="reports/run-1",
        policy_ledger="POLICY_LEDGER.md",
        timeline="timeline.jsonl",
        command_period_s=0.4,
        response_timeout_s=0.3,
        chunk_size=6,
        batch_size=2,
        save_freq=1,
        architecture=dict(
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            latent_dim=8,
            n_vae_encoder_layers=1,
            dropout=0.0,
            kl_weight=1.0,
        ),
    )
    return config, contract


def input_files(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    config, contract = settings(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    path = dataset / "physical_contract.json"
    path.write_text(json.dumps(contract))
    repository = Path(config["repository"])
    repository.mkdir()
    (repository / "POLICY_LEDGER.md").write_text("# Synthetic ledger\n")
    return config, dataset, path


def test_training_cadence_preserves_recorded_lookahead(tmp_path: Path) -> None:
    config, contract = settings(tmp_path)
    cadence = validate_config(config, contract)
    assert cadence["command_stride_steps"] == 6
    assert cadence["action_lookahead_steps"] == 1
    assert "offered now" in cadence["semantics"]
    assert "0,q,2q" in cadence["semantics"]


@pytest.mark.parametrize("configured", [False, True])
def test_training_refuses_disposable_contract_even_when_expected_matches(
    tmp_path: Path, configured: bool,
) -> None:
    from test_task_registry import REGISTRY

    from synria_lerobot.task_registry import load_task_registry

    config, _ = settings(tmp_path)
    task = synthetic_task(20, 30) if configured else load_task_registry(REGISTRY)["die_into_cup"]
    contract = PhysicalDatasetContract(
        "50mm", ActionSource.NEXT_STATE, False, task_id=task.task_id,
        task_definition=task, recording_purpose="disposable_smoke",
    ).as_dict(fps=15)
    config["expected_contract"] = contract
    with pytest.raises(ValueError, match="disposable smoke"):
        validate_config(config, contract)


def simulation_contract(task_id: str = "die_into_cup") -> dict[str, Any]:
    """A dataset contract recorded in the Isaac scene, as the simulation console writes it."""
    return PhysicalDatasetContract(
        "50mm", ActionSource.NEXT_STATE, False, task_id=task_id,
        task_definition=synthetic_task(0.1, 1), recording_purpose="simulation",
    ).as_dict(fps=15)


def test_each_lane_accepts_only_its_own_datasets(tmp_path: Path) -> None:
    """Physical trains qualifying data only; simulation trains simulation data only."""
    config, physical = settings(tmp_path)
    simulated = simulation_contract()
    assert validate_config(config, physical)["command_stride_steps"] == 6
    with pytest.raises(ValueError, match="not eligible"):
        validate_config({**config, "expected_contract": simulated}, simulated)
    sim_config = {**config, "lane": "simulation", "expected_contract": simulated}
    assert validate_config(sim_config, simulated)["command_stride_steps"] == 6
    with pytest.raises(ValueError, match="simulation lane"):
        validate_config({**sim_config, "expected_contract": physical}, physical)
    with pytest.raises(ValueError, match="lane must be"):
        validate_config({**config, "lane": "bench"}, physical)


@pytest.mark.parametrize("limit", [0, True, 2.0, "2"])
def test_episode_limit_must_be_a_positive_integer(tmp_path: Path, limit: Any) -> None:
    """The data-scaling subset size is validated like every other hyperparameter."""
    config, _ = settings(tmp_path)
    simulated = simulation_contract()
    sim_config = {**config, "lane": "simulation", "expected_contract": simulated,
                  "training_episode_limit": limit}
    with pytest.raises(ValueError, match="training_episode_limit"):
        validate_config(sim_config, simulated)


AUGMENTATION = {"enable": True, "max_num_transforms": 3, "random_order": False,
                "transforms": ["brightness", "contrast", "sharpness"]}


def test_augmentation_and_ensembling_blocks_are_validated(tmp_path: Path) -> None:
    """Optional blocks: absent means off; present means well formed and named from the list."""
    config, physical = settings(tmp_path)
    assert validate_config({**config, "image_transforms": AUGMENTATION,
                            "temporal_ensemble_coeff": 0.01}, physical)
    assert validate_config({**config, "image_transforms": None,
                            "temporal_ensemble_coeff": None}, physical)
    for bad in ({**AUGMENTATION, "transforms": ["blur"]}, {**AUGMENTATION, "transforms": []},
                {**AUGMENTATION, "enable": 1}, {**AUGMENTATION, "max_num_transforms": 0},
                {"enable": True}, {**AUGMENTATION, "transforms": ["hue", "hue"]}):
        with pytest.raises(ValueError, match="image|max_num_transforms"):
            validate_config({**config, "image_transforms": bad}, physical)
    for bad in (-0.1, float("nan"), True, "0.01"):
        with pytest.raises(ValueError, match="temporal_ensemble_coeff"):
            validate_config({**config, "temporal_ensemble_coeff": bad}, physical)
    assert validate_dataloader_workers(None) == 0 and validate_dataloader_workers(4) == 4
    assert validate_config({**config, "dataloader_workers": 2}, physical)
    for bad in (-1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="dataloader_workers"):
            validate_config({**config, "dataloader_workers": bad}, physical)
    assert validate_episode_weights(None) is None
    assert validate_episode_weights({"3": 2, "5-7": 0.5}) == {3: 2.0, 5: 0.5, 6: 0.5, 7: 0.5}
    assert validate_config({**config, "episode_weights": {"31-50": 2}}, physical)
    for bad in ({}, [], {"a": 1}, {"3": 0}, {"3": True}, {"3": float("inf")}, {"3": 1, "2-4": 1},
                {3: 1}, {"4-2": 1}):
        with pytest.raises(ValueError, match="episode"):
            validate_config({**config, "episode_weights": bad}, physical)


NOISE = {"joint_rad": 0.02, "gripper_m": 0.002, "drop_probability": 0.1}
PHASE = {"seconds_before_close": 2.0, "seconds_after_close": 1.0, "close_above_m": 0.016,
         "weight": 3.0}


def test_approach_levers_are_validated(tmp_path: Path) -> None:
    """State noise, action smoothing and the grasp phase: absent means off, else well formed."""
    config, physical = settings(tmp_path)
    assert validate_config({**config, "state_noise": NOISE, "action_smoothing_window": 5,
                            "grasp_phase": PHASE}, physical)
    assert validate_state_noise(None) is None and validate_action_smoothing(None) is None
    assert validate_grasp_phase(None) is None
    assert validate_state_noise(NOISE) == NOISE and validate_action_smoothing(3) == 3
    for bad in ({**NOISE, "drop_probability": 1}, {**NOISE, "joint_rad": -1}, {"joint_rad": 0.1},
                {"joint_rad": 0, "gripper_m": 0, "drop_probability": 0}):
        with pytest.raises(ValueError, match="state_noise|drop_probability|joint_rad"):
            validate_config({**config, "state_noise": bad}, physical)
    for bad in (2, 4, 1, True, 3.0):
        with pytest.raises(ValueError, match="action_smoothing_window"):
            validate_config({**config, "action_smoothing_window": bad}, physical)
    for bad in ({**PHASE, "weight": 1}, {**PHASE, "close_above_m": 0}, {"weight": 3},
                {**PHASE, "seconds_before_close": -1}):
        with pytest.raises(ValueError, match="grasp_phase|weight|close_above_m|seconds_before"):
            validate_config({**config, "grasp_phase": bad}, physical)


def test_action_limit_margin_is_validated_and_pulls_the_bounds_in(tmp_path: Path) -> None:
    """Absent means off; a margin in degrees narrows the operator bounds, joints only."""
    import math

    from synria_lerobot.quality_gates import load_limits

    config, physical = settings(tmp_path)
    assert validate_action_limit_margin(None) is None
    assert validate_action_limit_margin(1) == 1.0
    assert validate_config({**config, "action_limit_margin_deg": 0.5}, physical)
    for bad in (0, -1, True, "1", float("nan"), 10.5):
        with pytest.raises(ValueError, match="action_limit_margin_deg"):
            validate_config({**config, "action_limit_margin_deg": bad}, physical)
    limits = load_limits(Path(__file__).resolve().parents[1] / LIMITS_FILE)
    assert limits.verified
    names = [*limits.joint_names, "Gripper"]
    bounds = action_bounds(limits, 1.0, names)
    assert len(bounds) == 6
    for (lower, upper), (raw_lower, raw_upper) in zip(bounds, limits.joint_limits_rad,
                                                      strict=True):
        assert lower == pytest.approx(raw_lower + math.radians(1.0))
        assert upper == pytest.approx(raw_upper - math.radians(1.0))
    with pytest.raises(ValueError, match="joint order"):
        action_bounds(limits, 1.0, names[::-1])
    from dataclasses import replace

    narrow = replace(limits, joint_limits_rad=(
        *limits.joint_limits_rad[:4], (-0.01, 0.01), limits.joint_limits_rad[5]))
    with pytest.raises(ValueError, match="Joint5 no range"):
        action_bounds(narrow, 1.0, names[:6])


def test_clamp_holds_joints_inside_the_bounds_and_keeps_the_gripper() -> None:
    """Joints past a pulled-in bound land on it; joints inside and the gripper are untouched."""
    import math

    from synria_lerobot.quality_gates import load_limits

    torch = pytest.importorskip("torch")
    limits = load_limits(Path(__file__).resolve().parents[1] / LIMITS_FILE)
    bounds = action_bounds(limits, 1.0, limits.joint_names)
    actions = torch.zeros((1, 2, 7))
    actions[0, 0, 1] = math.radians(100.9)  # Joint2 past its 100 degree stop
    actions[0, 1, 2] = math.radians(-35.5)  # Joint3 past its -35 degree stop
    actions[0, :, 6] = 0.025  # the gripper fully open, on its stroke
    clamped = clamp_actions(actions, bounds)
    assert clamped[0, 0, 1].item() == pytest.approx(math.radians(99.0))
    assert clamped[0, 1, 2].item() == pytest.approx(math.radians(-34.0))
    assert clamped[0, 1, 1].item() == 0.0 and clamped[0, :, 6].tolist() == [0.025, 0.025]


def test_grasp_phase_weights_surround_the_final_close_only() -> None:
    """Frames two seconds before to one second after the last close are weighted; others not."""
    # Episode 0: open for 10 frames, closed for 10; at 5 fps the window is frames 0 to 15.
    # Episode 1: never closes. Episode 2: ends open. Episode 3: a brief early close then open,
    # then the final close at frame 8 of 10.
    episodes = [0] * 20 + [1] * 5 + [2] * 6 + [3] * 10
    frames = list(range(20)) + list(range(5)) + list(range(6)) + list(range(10))
    grippers = ([0.0] * 10 + [0.02] * 10 + [0.0] * 5 + [0.02] * 3 + [0.0] * 3
                + [0.0, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02])
    weights, count = grasp_phase_frame_weights(episodes, frames, grippers, 5.0, PHASE)
    assert weights[:16] == [3.0] * 16 and weights[16:20] == [1.0] * 4
    assert weights[20:31] == [1.0] * 11
    assert weights[31:] == [1.0] * 0 + [3.0] * 10
    assert count == 26
    # Shuffled input lines up with its own order.
    order = list(range(len(episodes)))[::-1]
    shuffled, _ = grasp_phase_frame_weights(
        [episodes[i] for i in order], [frames[i] for i in order], [grippers[i] for i in order],
        5.0, PHASE,
    )
    assert shuffled == [weights[i] for i in order]


def test_smoothing_and_state_noise_act_on_the_batch_tensors() -> None:
    """Joints are averaged over the chunk with edge padding, the gripper kept, noise seeded."""
    torch = pytest.importorskip("torch")
    actions = torch.zeros((1, 5, 7))
    actions[0, :, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    actions[0, :, 6] = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    smoothed = smooth_actions(actions, 3)
    assert smoothed[0, :, 0].tolist() == pytest.approx([0.0, 1 / 3, 1 / 3, 1 / 3, 0.0])
    assert smoothed[0, :, 6].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert smooth_actions(actions, 5)[0, 0, 0] == pytest.approx(1 / 5)
    state = torch.full((4, 7), 0.5)
    mean = torch.zeros(7)
    noisy = perturb_state(state, {"joint_rad": 0.01, "gripper_m": 0.001, "drop_probability": 0.0},
                          mean, torch.Generator().manual_seed(1))
    assert noisy.shape == state.shape and not torch.equal(noisy, state)
    assert (noisy - state).abs().max() < 0.1
    again = perturb_state(state, {"joint_rad": 0.01, "gripper_m": 0.001, "drop_probability": 0.0},
                          mean, torch.Generator().manual_seed(1))
    assert torch.equal(noisy, again)
    dropped = perturb_state(state, {"joint_rad": 0.0, "gripper_m": 0.0, "drop_probability": 1.0},
                            mean, torch.Generator().manual_seed(1))
    assert torch.equal(dropped, torch.zeros((4, 7)))


def test_episode_limit_is_refused_outside_the_simulation_lane(tmp_path: Path) -> None:
    """Physical training always uses every episode outside the held-out set."""
    config, physical = settings(tmp_path)
    with pytest.raises(ValueError, match="simulation-lane study setting"):
        validate_config({**config, "training_episode_limit": 5}, physical)


@pytest.mark.parametrize(
    "key,value",
    [
        ("policy_id", None),
        ("policy_id", True),
        ("policy_id", 123),
        ("response_timeout_s", None),
        ("response_timeout_s", False),
        ("response_timeout_s", 0),
        ("response_timeout_s", float("nan")),
        ("command_period_s", True),
        ("command_period_s", 0.41),
        ("command_period_s", 0.01),
        ("chunk_size", 5),
        ("chunk_size", True),
        ("batch_size", 0),
        ("save_freq", False),
        ("learning_rate", float("inf")),
        ("weight_decay", -1),
        ("device", "remote"),
    ],
)
def test_invalid_training_config_refuses_before_model(tmp_path: Path, key: str, value: Any) -> None:
    config, contract = settings(tmp_path)
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config, contract)


@pytest.mark.parametrize(
    "key,value",
    [
        ("state_has_velocity", 1),
        ("action_lookahead_steps", True),
        ("effective_action_lookahead_steps", False),
        ("requested_rate_hz", True),
        ("nominal_action_lookahead_s", False),
        ("action_source", "simulation"),
    ],
)
def test_malformed_contract_fields_fail_closed(tmp_path: Path, key: str, value: Any) -> None:
    config, contract = settings(tmp_path)
    contract[key] = value
    with pytest.raises(ValueError):
        validate_config(config, contract)


def test_expected_contract_cannot_relabel_dataset(tmp_path: Path) -> None:
    config, contract = settings(tmp_path)
    config["expected_contract"] = copy.deepcopy(contract)
    config["expected_contract"]["action_lookahead_steps"] = 2
    with pytest.raises(ValueError, match="must match"):
        validate_config(config, contract)


def test_boolean_expected_lookahead_is_not_equal_to_integer_contract(tmp_path: Path) -> None:
    config, contract = settings(tmp_path)
    config["expected_contract"] = copy.deepcopy(contract)
    config["expected_contract"]["action_lookahead_steps"] = True
    with pytest.raises(ValueError, match="must match"):
        validate_config(config, contract)


@pytest.mark.parametrize("seed,steps", [(True, 2), (1, False), (-1, 2), (2**32, 2), (1, 0)])
def test_invalid_seed_steps_refused_before_reading_inputs(
    tmp_path: Path, seed: Any, steps: Any
) -> None:
    with pytest.raises(ValueError):
        train(
            tmp_path / "missing",
            tmp_path / "missing-contract",
            tmp_path / "output",
            seed,
            steps,
            {},
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "ledger_timeline",
        "owned_manifest",
        "outside_ledger",
        "outside_timeline",
        "existing_output",
        "inside_dataset",
        "inside_repository",
        "contract_mismatch",
    ],
)
def test_unsafe_destinations_fail_before_optional_runtime_or_writes(
    tmp_path: Path, fault: str
) -> None:
    config, dataset, contract = input_files(tmp_path)
    repository = Path(config["repository"])
    output = tmp_path / "output"
    if fault == "ledger_timeline":
        config["timeline"] = config["policy_ledger"]
    elif fault == "owned_manifest":
        config["timeline"] = config["evidence_directory"] + "/training_run.json"
    elif fault.startswith("outside_"):
        config["policy_ledger" if fault == "outside_ledger" else "timeline"] = str(
            tmp_path / "outside"
        )
    elif fault == "existing_output":
        output.mkdir()
    elif fault == "inside_dataset":
        output = dataset / "training"
    elif fault == "inside_repository":
        output = repository / "training"
    else:
        contract = tmp_path / "other.json"
        contract.write_text("{}")
    before = (dataset / "physical_contract.json").read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        train(dataset, contract, output, 1, 2, config)
    assert (dataset / "physical_contract.json").read_bytes() == before
    assert not (repository / config["evidence_directory"]).exists()
    assert (repository / "POLICY_LEDGER.md").read_text() == "# Synthetic ledger\n"


def test_geometric_image_transforms_are_refused(tmp_path: Path) -> None:
    """An affine transform moves the object without moving the label; the trainer refuses it."""
    config, physical = settings(tmp_path)
    with pytest.raises(ValueError, match="geometric image transforms"):
        validate_config({**config, "image_transforms": {**AUGMENTATION,
                                                        "transforms": ["brightness", "affine"]}},
                        physical)


def test_a_server_may_read_a_historical_manifest_with_an_affine_transform(tmp_path: Path) -> None:
    """Serving tolerates the banned transform in a checkpoint's record; training does not."""
    config, physical = settings(tmp_path)
    historical = {**config, "image_transforms": {**AUGMENTATION,
                                                 "transforms": ["brightness", "affine"]}}
    assert validate_config(historical, physical, allow_geometric=True)
    with pytest.raises(ValueError, match="geometric image transforms"):
        validate_config(historical, physical)
