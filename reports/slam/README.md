# SLAM Evidence Reports

This directory holds odometry calibration, map capture, and navigation
evidence for the Yahboom ROSMASTER M3 Pro on Jetson Orin NX 8GB.

## Report Types

| File pattern | Contents |
|---|---|
| `calibration_demo.json` | CLI demo calibration run — deterministic sample data |
| `calibration_*.json` | Real hardware calibration sessions |
| `maps/map_index.json` | MapStore index — registered PGM/YAML map pairs |
| `nav_plans/` | Navigation plan JSON files for reproducible task runs |

## Calibration Schema

Each `calibration_*.json` is a `CalibrationReport`:

```json
{
  "robot_id": "yahboom-orin-01",
  "session_id": "2026-05-27T10:00:00+00:00",
  "status": "pass",
  "max_linear_error_pct": 7.0,
  "max_drift_m": 0.055,
  "max_angular_error_rad": 0.041,
  "trials": [
    {
      "name": "forward-1m",
      "axis": "x",
      "commanded_m": 1.0,
      "measured_m": 0.96,
      "drift_m": 0.028,
      "linear_error_pct": 4.0,
      "passes": true
    }
  ]
}
```

## Pass Criteria (Milestone B)

| Metric | Threshold |
|---|---|
| Linear distance error | ≤ 8% |
| Lateral drift per 1 m | ≤ 6 cm |
| Angular error per 90° | ≤ 0.08 rad (~4.6°) |

## Run CLI Demo

```bash
physical-ai-lab slam-calibration --robot-id yahboom-orin-01
physical-ai-lab mecanum-calibration
```

## Hardware Evidence Gate

Calibration runs in this directory are **planned evidence** until the Yahboom
robot is live and real motion logs are committed alongside them.
