"""CSV replay source for Edge-IIoT style defensive telemetry datasets."""

from __future__ import annotations

import csv
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jetson_edge_ai_security.schemas import TelemetryEvent
from jetson_edge_ai_security.sources.base import TrafficSource

LOGGER = logging.getLogger(__name__)


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "frame.time", "frame.time_epoch", "time", "ts", "starttime", "stime"),
    "source_ip": ("source_ip", "ip.src", "ip.src_host", "src_ip", "srcip", "srcaddr", "saddr", "id.orig_h"),
    "dest_ip": ("dest_ip", "ip.dst", "ip.dst_host", "dst_ip", "dstip", "dstaddr", "daddr", "id.resp_h"),
    "source_port": (
        "source_port",
        "tcp.srcport",
        "udp.srcport",
        "src_port",
        "sport",
        "srcport",
        "id.orig_p",
    ),
    "dest_port": (
        "dest_port",
        "tcp.dstport",
        "udp.dstport",
        "dst_port",
        "dport",
        "dstport",
        "id.resp_p",
    ),
    "protocol": ("protocol", "_ws.col.protocol", "proto", "ip.proto", "protocol type"),
    "packet_size": (
        "packet_size",
        "frame.len",
        "length",
        "packet_length",
        "bytes",
        "totbytes",
        "tot size",
        "totalsize",
    ),
    "tcp_flags": ("tcp_flags", "tcp.flags", "flags"),
    "icmp_type": ("icmp_type", "icmp.type"),
    "flow_id": ("flow_id", "flow.id", "uid", "tcp.stream", "udp.stream"),
    "attack_label": ("attack_label", "label", "attack_label", "Attack_label", "class", "target"),
    "attack_type": ("attack_type", "attack_type", "Attack_type", "category", "subcategory", "traffic"),
}


class CsvReplaySource(TrafficSource):
    """Replay an Edge-IIoT style CSV file as normalized telemetry events.

    The adapter accepts common Edge-IIoT, Zeek-like, and generic IDS column names.
    Missing optional fields are filled with safe defaults. Malformed rows are skipped
    unless ``strict`` is enabled.
    """

    name = "csv-replay"

    def __init__(
        self,
        path: str | Path,
        *,
        limit: int | None = None,
        replay_delay_seconds: float = 0.0,
        strict: bool = False,
        source_type: str = "csv-replay",
    ) -> None:
        self.path = Path(path)
        self.limit = limit
        self.replay_delay_seconds = replay_delay_seconds
        self.strict = strict
        self.source_type = source_type
        self._handle: Any | None = None
        self.rows_seen = 0
        self.rows_skipped = 0

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV replay file does not exist: {self.path}")
        self._handle = self.path.open("r", newline="", encoding="utf-8-sig")

    def events(self) -> Iterator[TelemetryEvent]:
        should_close = self._handle is None
        if should_close:
            self.open()
        assert self._handle is not None

        try:
            reader = csv.DictReader(self._handle)
            emitted = 0
            for row in reader:
                self.rows_seen += 1
                if self.limit is not None and emitted >= self.limit:
                    break
                try:
                    event = self._row_to_event(row)
                except (ValueError, ValidationError) as exc:
                    self.rows_skipped += 1
                    if self.strict:
                        raise
                    LOGGER.warning("Skipping malformed CSV row %s: %s", self.rows_seen, exc)
                    continue
                emitted += 1
                if self.replay_delay_seconds > 0:
                    time.sleep(self.replay_delay_seconds)
                yield event
        finally:
            if should_close:
                self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _row_to_event(self, row: dict[str, str]) -> TelemetryEvent:
        payload: dict[str, Any] = {"source_type": self.source_type, "metadata": {"raw_source": self.name}}
        for field, aliases in COLUMN_ALIASES.items():
            value = _first_present(row, aliases)
            if value not in (None, ""):
                payload[field] = value

        payload["source_port"] = _optional_int(payload.get("source_port"))
        payload["dest_port"] = _optional_int(payload.get("dest_port"))
        payload["packet_size"] = _optional_int(payload.get("packet_size"))

        known_columns = {alias.lower() for aliases in COLUMN_ALIASES.values() for alias in aliases}
        payload["metadata"]["unmapped"] = {
            key: value for key, value in row.items() if key.lower() not in known_columns and value != ""
        }
        return TelemetryEvent(**payload)


def _first_present(row: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    lower_map = {key.lower(): value for key, value in row.items()}
    for alias in aliases:
        value = lower_map.get(alias.lower())
        if value not in (None, ""):
            return value
    return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(str(value).strip()))
