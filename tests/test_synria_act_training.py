"""Training preflight tests do not import a model or touch any robot interface."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from synria_lerobot.act_training import train, validate_config
from synria_lerobot.physical_contract import ActionSource, PhysicalDatasetContract


def settings(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = PhysicalDatasetContract("50mm", ActionSource.NEXT_STATE, False).as_dict(fps=15)
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
