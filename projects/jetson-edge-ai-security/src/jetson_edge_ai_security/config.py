"""Runtime configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    window_size: int = Field(default=50, ge=1)
    step: int = Field(default=10, ge=1)
    replay_delay_seconds: float = Field(default=0.0, ge=0)
    strict_csv: bool = False


class DetectorConfig(BaseModel):
    packet_count_threshold: int = Field(default=500, ge=1)
    event_rate_threshold: float = Field(default=200.0, ge=0)
    unique_source_ip_threshold: int = Field(default=100, ge=1)
    attack_count_threshold: int = Field(default=1, ge=0)
    use_isolation_forest: bool = False


class AlertConfig(BaseModel):
    source: str = "edge-security-runtime"
    default_recommended_action: str = (
        "Review source telemetry, correlate with IDS logs, and isolate affected lab systems if confirmed."
    )


class AppConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = _load_yaml_like(handle.read())
    return AppConfig.model_validate(data)


def _load_yaml_like(text: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml(text)
    return yaml.safe_load(text) or {}


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple nested key/value YAML used by the default config."""

    data: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith((" ", "\t")) and line.endswith(":"):
            section_name = line[:-1].strip()
            current_section = {}
            data[section_name] = current_section
            continue
        if current_section is None or ":" not in line:
            raise ValueError("Fallback config parser only supports one-level nested key/value YAML")
        key, value = line.strip().split(":", 1)
        current_section[key.strip()] = _coerce_scalar(value.strip())
    return data


def _coerce_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value == "":
        return ""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")
