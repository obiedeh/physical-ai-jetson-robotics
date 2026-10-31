# SLAM Evidence Reports

This directory holds odometry calibration, map capture, and navigation
evidence for the Yahboom ROSMASTER M3 Pro on Jetson Orin NX 8GB.

## Report Types

| File pattern | Contents |
|---|---|
| `calibration_demo.json` | CLI demo calibration run - deterministic sample data |
| `calibration_*.json` | Real hardware calibration sessions |
| `maps/map_index.json` | MapStore index - registered PGM/YAML map pairs |
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

## Session directories (`sessions/<date>_<run>/`)

Written on the Orin by `scripts/jetson/slam_session.py` and
`scripts/jetson/slam_pose_log.py`; protocol in
`docs/rosmaster_m3pro/SLAM_FIRST_RUN_PROTOCOL.md`. First one: `sessions/2026-09-16_stationary/`, robot stationary, resource cost only.

| File | Contents |
|---|---|
| `provenance.json` | device, L4T, git SHA, power mode, SLAM stack, topics, written before recording |
| `session.json` | still baseline and recording power per rail, thermal, RAM, per-process CPU, bag size and metadata hash |
| `samples.jsonl` | raw power, thermal and process samples behind `session.json` |
| `poses.jsonl` | map-frame pose at 5 Hz and every raw odometry message |
| `marks.jsonl` | operator marks: `lap_start`, `lap_end`, `kidnap_lift`, `kidnap_place` |
| `metrics.json` | loop-closure return error per lap and frame, re-localisation trials, `not_measured` list |
| `bag/metadata.yaml` | ros2 bag index; the data files are ignored |

## Offline replay of a recorded session

`scripts/jetson/slam_bag_replay.py` plays a session's bag (`/scan`, `/odom`,
`/tf`, `/tf_static`, bag clock) into a fresh SLAM Toolbox mapping node on the
rover's board, with the bag's own `map`-parented transforms filtered out, and
writes under `sessions/<session>/offline_replay_<date>/`:

| File | Contents |
|---|---|
| `metrics.json` | `slam-offline-replay-v1`: bag hashes and duration, the parameter file hash, playback exit code and wall time, transform filter counts, map update count, the replayed map's cell counts and known area, the session's own last map from the bag for comparison, node RSS peak and CPU time, board power from tegrastats |
| `map.pgm`, `map.yaml` | the replayed node's last map in map_server trinary form |
| `session_map.pgm`, `session_map.yaml` | the last `/map` the live session recorded, straight from the bag |
| `slam_toolbox.log`, `bag_play.log`, `tegrastats.log` | raw logs, home paths replaced |

The 2026-09-16 session is stationary, so its replay registers one scan node
and the map is one sweep's footprint. It exercises the mapping stack on
recorded data; loop closure and drift need a driven session.
