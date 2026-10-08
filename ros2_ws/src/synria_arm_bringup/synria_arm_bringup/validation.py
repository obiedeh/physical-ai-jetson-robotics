from __future__ import annotations


def validate_leader_port(port: str) -> str:
    """Require a stable Linux serial-device identity with no auto-detection."""
    prefix = "/dev/serial/by-id/"
    if not port.startswith(prefix) or len(port) == len(prefix):
        raise ValueError(
            "leader_port is required and must be a /dev/serial/by-id path"
        )
    return port
