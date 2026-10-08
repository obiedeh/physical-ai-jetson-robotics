"""Training preflight tests do not import a model or touch any robot interface."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_task_registry import synthetic_task

from synria_lerobot.act_training import _save_upstream_checkpoint, train, validate_config
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
