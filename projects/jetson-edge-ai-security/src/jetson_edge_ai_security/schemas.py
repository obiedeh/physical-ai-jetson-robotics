"""Normalized schemas used across the edge security runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _parse_bool_label(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "attack", "malicious", "anomaly"}:
        return True
    if text in {"0", "false", "no", "normal", "benign"}:
        return False
    return None


class TelemetryEvent(BaseModel):
    """A normalized event produced by any defensive telemetry source."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_ip: str = ""
    dest_ip: str = ""
    source_port: int | None = Field(default=None, ge=0, le=65535)
    dest_port: int | None = Field(default=None, ge=0, le=65535)
    protocol: str = "UNKNOWN"
    packet_size: int | None = Field(default=None, ge=0)
    tcp_flags: str | None = None
    icmp_type: str | int | None = None
    flow_id: str | None = None
    source_type: str = "unknown"
    attack_label: bool | None = None
    attack_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value: Any) -> datetime:
        if value is None or value == "":
            return datetime.now(UTC)
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        text = str(value).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        normalized = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    @field_validator("protocol", mode="before")
    @classmethod
    def normalize_protocol(cls, value: Any) -> str:
        if value is None or value == "":
            return "UNKNOWN"
        return str(value).strip().upper()

    @field_validator("attack_label", mode="before")
    @classmethod
    def normalize_attack_label(cls, value: Any) -> bool | None:
        return _parse_bool_label(value)


class FeatureWindow(BaseModel):
    """Aggregated features for one rolling/sliding window."""

    window_start: datetime
    window_end: datetime
    packet_count: int
    mean_packet_size: float
    max_packet_size: int
    protocol_counts: dict[str, int] = Field(default_factory=dict)
    tcp_flag_counts: dict[str, int] = Field(default_factory=dict)
    attack_count: int = 0
    event_rate: float = 0.0
    unique_source_ip_count: int = 0
    unique_dest_ip_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectionResult(BaseModel):
    """Detector output before alert rendering."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_anomaly: bool
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high", "critical"] = "low"
    features: FeatureWindow
    metadata: dict[str, Any] = Field(default_factory=dict)


class Alert(BaseModel):
    """Operational alert emitted by the runtime."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    severity: Literal["low", "medium", "high", "critical"]
    title: str
    description: str
    source: str
    features: dict[str, Any]
    recommended_action: str
    metadata: dict[str, Any] = Field(default_factory=dict)

