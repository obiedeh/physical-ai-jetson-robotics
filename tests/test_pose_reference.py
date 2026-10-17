"""Keep the operator pose reference file well formed and consistent with the limits file."""

from __future__ import annotations

import json
from pathlib import Path

from synria_lerobot.quality_gates import load_limits

REFERENCE = Path("config/synria_pose_reference.json")


def test_pose_reference_is_well_formed_and_attributed() -> None:
    """Every pose carries six joint readings in degrees, and the file says who read them."""
    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert payload["measured_by"].strip() and payload["measured_on"]
    assert tuple(payload["joint_names"]) == tuple(load_limits(Path("config/synria_limits.yaml")).joint_names)
    for pose in payload["poses"].values():
        assert len(pose["read_angles_deg"]) == 6
        assert all(isinstance(value, float) for value in pose["read_angles_deg"])
        assert isinstance(pose["gripper_m"], float) and 0.0 <= pose["gripper_m"] <= 0.05


def test_reference_pose_offsets_are_the_reading_minus_the_model() -> None:
    """The stored difference is derived from the two angle lists, never typed separately."""
    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    pose = payload["poses"]["vendor_standby_reference"]
    expected = [round(read - model, 2) for read, model in
                zip(pose["read_angles_deg"], pose["model_angles_deg"], strict=True)]
    assert pose["read_minus_model_deg"] == expected
