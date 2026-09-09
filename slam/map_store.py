"""SLAM map storage and metadata for the Yahboom ROSMASTER M3 Pro.

Maps are saved as PGM/YAML pairs in the standard ROS ``map_server`` format.
This module provides metadata records, an on-disk index, and simple
validation without requiring a live ROS environment.

The map store lives under ``reports/slam/maps/`` by convention. Override the
path for testing via the ``MapStore(store_dir=...)`` argument.

Example::

    from pathlib import Path
    from slam.map_store import MapMetadata, MapRecord, MapStore

    store = MapStore(Path("reports/slam/maps"))
    meta = MapMetadata(
        map_name="lab_room_01",
        captured_at="2026-05-27T10:00:00+00:00",
        robot_id="yahboom-orin-01",
        resolution_m_per_px=0.05,
        origin_x_m=-5.0,
        origin_y_m=-5.0,
        width_px=200,
        height_px=200,
        notes="First SLAM run in open lab area.",
    )
    record = MapRecord(
        metadata=meta,
        pgm_path=Path("reports/slam/maps/lab_room_01.pgm"),
        yaml_path=Path("reports/slam/maps/lab_room_01.yaml"),
    )
    store.add(record)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MapMetadata:
    """Metadata for one saved SLAM map.

    Attributes:
        map_name: Unique map identifier.
        captured_at: ISO-8601 timestamp when the map was saved.
        robot_id: Robot that captured the map.
        resolution_m_per_px: Metres per pixel (ROS YAML ``resolution`` field).
        origin_x_m: X coordinate of the map origin in the world frame.
        origin_y_m: Y coordinate of the map origin in the world frame.
        width_px: Map width in pixels.
        height_px: Map height in pixels.
        notes: Free-text capture context (environment, conditions, etc.).
    """

    map_name: str
    captured_at: str
    robot_id: str
    resolution_m_per_px: float
    origin_x_m: float
    origin_y_m: float
    width_px: int
    height_px: int
    notes: str = ""

    def area_m2(self) -> float:
        """Physical area covered by the map in square metres."""
        return round(
            self.width_px * self.resolution_m_per_px
            * self.height_px * self.resolution_m_per_px,
            3,
        )

    def width_m(self) -> float:
        return round(self.width_px * self.resolution_m_per_px, 3)

    def height_m(self) -> float:
        return round(self.height_px * self.resolution_m_per_px, 3)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MapRecord:
    """Full map record: metadata + optional on-disk file paths."""

    metadata: MapMetadata
    pgm_path: Path | None = None
    yaml_path: Path | None = None

    def is_complete(self) -> bool:
        """True when both PGM and YAML files exist on disk."""
        return (
            self.pgm_path is not None
            and self.pgm_path.exists()
            and self.yaml_path is not None
            and self.yaml_path.exists()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata": self.metadata.to_dict(),
            "pgm_path": str(self.pgm_path) if self.pgm_path else None,
            "yaml_path": str(self.yaml_path) if self.yaml_path else None,
            "complete": self.is_complete(),
        }


class MapStore:
    """File-system store for SLAM map records.

    Maintains a JSON index file (``map_index.json``) in ``store_dir``.
    Records are keyed by ``map_name`` and persist across process restarts.
    """

    _INDEX_FILE = "map_index.json"

    def __init__(self, store_dir: Path) -> None:
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / self._INDEX_FILE
        self._records: dict[str, MapRecord] = {}
        if self._index_path.exists():
            self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: MapRecord) -> None:
        """Register a map record and persist the index."""
        self._records[record.metadata.map_name] = record
        self._save_index()

    def get(self, map_name: str) -> MapRecord | None:
        """Return a record by name, or None if not found."""
        return self._records.get(map_name)

    def list_maps(self) -> list[str]:
        """Return a sorted list of all registered map names."""
        return sorted(self._records)

    def complete_maps(self) -> list[MapRecord]:
        """Return records where both PGM and YAML files exist on disk."""
        return [r for r in self._records.values() if r.is_complete()]

    def remove(self, map_name: str) -> bool:
        """Remove a record from the index (does not delete on-disk files).

        Returns True if the record existed, False otherwise.
        """
        if map_name in self._records:
            del self._records[map_name]
            self._save_index()
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_index(self) -> None:
        index = {
            name: record.to_dict() for name, record in self._records.items()
        }
        self._index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _load_index(self) -> None:
        raw: dict[str, object] = json.loads(
            self._index_path.read_text(encoding="utf-8")
        )
        for name, data in raw.items():
            assert isinstance(data, dict)
            meta_raw = data["metadata"]
            assert isinstance(meta_raw, dict)
            metadata = MapMetadata(**meta_raw)
            pgm = Path(str(data["pgm_path"])) if data.get("pgm_path") else None
            yaml = Path(str(data["yaml_path"])) if data.get("yaml_path") else None
            self._records[name] = MapRecord(
                metadata=metadata, pgm_path=pgm, yaml_path=yaml
            )
