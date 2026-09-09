"""SLAM and navigation helpers for the Yahboom ROSMASTER M3 Pro.

Public surface:
    map_store   — SLAM map metadata, storage, and validation
    nav_plan    — navigation waypoint planning and geometry
    calibration — odometry calibration evidence and pass/fail reporting
"""

from slam.calibration import (
    CalibrationReport,
    CalibrationTrial,
    demo_calibration_trials,
    run_calibration_session,
)
from slam.map_store import MapMetadata, MapRecord, MapStore
from slam.nav_plan import (
    NavigationPlan,
    NavPlanSegment,
    Waypoint,
    WaypointType,
    build_nav_plan,
    validate_nav_plan,
)

__all__ = [
    "CalibrationReport",
    "CalibrationTrial",
    "demo_calibration_trials",
    "run_calibration_session",
    "MapMetadata",
    "MapRecord",
    "MapStore",
    "NavigationPlan",
    "NavPlanSegment",
    "Waypoint",
    "WaypointType",
    "build_nav_plan",
    "validate_nav_plan",
]
