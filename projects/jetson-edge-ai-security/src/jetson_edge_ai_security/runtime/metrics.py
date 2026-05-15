"""Runtime metrics for pipeline executions."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class RuntimeMetrics(BaseModel):
    """Counters and timing captured by the pipeline."""

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    events_seen: int = 0
    windows_seen: int = 0
    detections_seen: int = 0
    alerts_emitted: int = 0

    def finish(self) -> None:
        self.finished_at = datetime.now(UTC)

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or datetime.now(UTC)
        return max((end - self.started_at).total_seconds(), 0.0)

