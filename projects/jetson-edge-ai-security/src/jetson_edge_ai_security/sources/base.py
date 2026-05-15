"""Traffic source abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType

from jetson_edge_ai_security.schemas import TelemetryEvent


class TrafficSource(ABC):
    """Abstract source of normalized defensive telemetry events."""

    name: str

    def __enter__(self) -> "TrafficSource":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @abstractmethod
    def open(self) -> None:
        """Open source resources."""

    @abstractmethod
    def events(self) -> Iterator[TelemetryEvent]:
        """Yield normalized telemetry events."""

    @abstractmethod
    def close(self) -> None:
        """Close source resources."""

