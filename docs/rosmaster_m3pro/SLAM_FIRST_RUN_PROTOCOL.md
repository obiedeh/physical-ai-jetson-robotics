# First measured SLAM run: protocol

Robot: Yahboom ROSMASTER M3 Pro, Jetson Orin NX 8 GB, dual ToF LiDAR
(`/scan0`, `/scan1`), micro-ROS base. SLAM stack: the vendor workspace on the
Orin (`M3Pro_ws`: slam_toolbox, cartographer, gmapping configs, dual-scan
merge), which is a dependency we launch, not work we present. Nothing in this
document is a result. Until a session directory is committed, every rover
SLAM row in the README stays `planned`.

Tools (this repository, offline-tested, never yet run on the Orin):

| Tool | Runs where | Does |
|---|---|---|
| `scripts/jetson/slam_session.py start` | Orin | provenance, still baseline, `ros2 bag record`, power, thermal, RAM, per-process CPU, `session.json` |
| `scripts/jetson/slam_pose_log.py` | Orin | `poses.jsonl`: map to base transform at 5 Hz plus every `/odom_raw` |
| `scripts/jetson/slam_session.py mark <label>` | Orin, second terminal | operator marks with timestamps |
| `scripts/jetson/slam_session.py summarize` | Orin or host | `metrics.json` from poses and marks |
| `slam/metrics.py` | anywhere | the arithmetic, recomputable from the committed files |

## 0. Safety gates, before anything moves

1. Board visible: `source scripts/jetson/orin_ros_env.sh && m3pro_preflight`.
2. Hardware e-stop tested with the robot on the stand: command motion via the
   vendor joystick, hit the e-stop, confirm the wheels stop. The base firmware
   has no `/cmd_vel` timeout (2026-08-20 finding); a dead teleop process
   leaves the last command running.
3. Teleop for mapping is the vendor joystick node held by the operator. No
   scripted `/cmd_vel` publisher runs during these sessions. `slam_session.py`
   and `slam_pose_log.py` publish nothing.
4. Room clear of people other than the operator. Doors closed.

## 1. Discovery on the Orin (record, do not assume)

```bash
source scripts/jetson/orin_ros_env.sh
ros2 pkg list | grep -i -E 'map|slam|laser|nav'
ls /home/jetson/M3Pro_ws/install/*/share/*/launch/ 2>/dev/null
ros2 topic list; ros2 topic hz /scan0; ros2 topic hz /odom_raw
df -h / ; free -m ; nvpmodel -q 2>/dev/null || cat /var/lib/nvpmodel/status
```

Write down: the launch file used, the SLAM package and version
(`ros2 pkg xml slam_toolbox | grep version`), the map frame and base frame
names (`ros2 run tf2_tools view_frames` after launch), free disk and RAM.
These go into `--stack` and `--note` and therefore into provenance.

## 2. Room preparation

- Tape a start square on the floor. Its centre is pose zero for laps.
- Tape three landmarks (wall corners or box edges) and measure their
  positions from the start square with a tape measure, in metres, to the
  centimetre. Record them in `landmarks.json` in the session directory.
- Plan a closed lap, roughly rectangular, 8 to 15 m long, that returns to the
  start square. Walk it once first.
- For kidnap trials, tape two target squares with measured `x`, `y` and
  heading relative to the start square. Record them in `targets.json` as
  `[{"x": .., "y": .., "yaw": ..}, ...]` in radians, one per planned trial in
  order.

## 3. Run A: mapping with resource sampling (the first artifact)

Terminal 1: launch the vendor SLAM stack. Terminal 2, from the repo root:

```bash
python3 scripts/jetson/slam_pose_log.py --out reports/slam/sessions/<date>_run_a \
    --map-frame <map frame> --base-frame <base frame>
```

Terminal 3:

```bash
python3 scripts/jetson/slam_session.py start --out reports/slam/sessions/<date>_run_a \
    --stack "<package launch, version>" --note "<room, route, operator>" --duration 300
```

Drive the lap at walking pace with the joystick. In terminal 4, mark:

```bash
python3 scripts/jetson/slam_session.py mark lap_start --out reports/slam/sessions/<date>_run_a
# ... drive the lap, stop centred on the start square, wait 3 s ...
python3 scripts/jetson/slam_session.py mark lap_end --out reports/slam/sessions/<date>_run_a
```

Three laps minimum. After the last lap, save the map next to the session:
`ros2 run nav2_map_server map_saver_cli -f reports/slam/sessions/<date>_run_a/map`.
Then `summarize`:

```bash
python3 scripts/jetson/slam_session.py summarize --out reports/slam/sessions/<date>_run_a
```

Memory: the default topic list excludes the camera. If `mem_available_mb.min`
in `session.json` drops under 500, the next run records fewer topics, not more.

## 4. Run B: kidnap and re-localisation

Requires a saved map and the stack in localisation mode (slam_toolbox
localization launch or Nav2 AMCL, whichever the vendor workspace provides;
record which). Same three terminals, a new session directory. Per trial:

1. Robot localised and still. `mark kidnap_lift`, lift it.
2. Set it down centred on target square `i`, heading as taped. `mark kidnap_place`.
3. Do not touch the joystick for 60 s. Then a short joystick nudge is allowed
   (note it) if the estimator needs motion to converge.

Five trials minimum. `summarize --targets reports/slam/sessions/<date>_run_b/targets.json`.

## 5. What gets committed, same session

Commit `session.json`, `provenance.json`, `marks.jsonl`, `poses.jsonl`,
`metrics.json`, `samples.jsonl`, `bag/metadata.yaml`, `map.yaml` and `map.pgm`
if under 2 MB, `landmarks.json` and `targets.json`. Bag data files stay out
(`.gitignore`). If a run is aborted, commit the partial directory with the
reason in `--note` or a `README.md` beside it; an aborted run is a record too.

## 6. What may be upgraded afterwards, and to what

| Layer | If the artifact exists | Label |
|---|---|---|
| Resource cost of the vendor SLAM stack while mapping | `session.json` with power, thermal, processes | measured, with date and Orin NX power mode |
| Loop-closure return error, map frame and raw odometry | `metrics.json` with `laps_scored >= 3` | measured |
| Re-localisation after kidnap | `metrics.json` with `trials_scored >= 5` | measured |
| Map quality against taped landmarks | not computed by this harness | stays planned |
| Sim map versus real map | needs the Isaac model out of measurement debt | stays planned |

Numbers are reported as index-based percentiles with `n`, never as a bare
mean. A run without `provenance.json` is not cited.
