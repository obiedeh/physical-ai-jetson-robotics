# Yahboom ROSMASTER M3 Pro — project strategy (2026-08-20)

Record update, 2026-09-07: these are chronological bring-up notes. Later entries
supersede earlier pending states, including keyboard recording verification in
section 27. Section 28's raw GR00T 1/20 is corrected to **0/20** by the
[corrected sidecar](../../reports/ludo_groot17_eval01/session_summary_corrected.json);
the cup was never grasped. Do not use that raw result as goal-conditioned success.

Decision memo. Supersedes the scope of `docs/YAHBOOM_ROBOT_CAR_PLAN.md`
(bring-up is done: `oedeh@jetsonorin`, inventory + day-one smoke committed).

## 1. What Yahboom already ships (do not rebuild)

The official course (yahboom.net/study/ROSMASTER-M3PRO, 24 modules) covers
the entire breadth a clone would: chassis + calibration, dual-ToF LiDAR
SLAM (gmapping/cartographer/slam_toolbox/RTAB-Map) and Nav2, depth camera
(Dabai DCW2) + YOLOv8/MediaPipe/AprilTag, 6DOF arm sorting/tracking with
MoveIt2, road-network planning, multi-vehicle formation, Docker, and an
LLM agent layer ("OpenClaw": Dify workflows, MCP tools, skills, voice,
vector memory, chassis/arm control, "intention estimation"). The on-device
workspace (`/home/jetson/yahboomcar_ws`: arm_driver/kin, OrbbecSDK_ROS2,
yahboomcar_mediapipe, yahboom_yolov8, MoveIt configs) is that curriculum.

Implication: every tutorial-shaped deliverable (SLAM map, color sorting,
gesture arm, voice command -> LLM -> action) is table stakes and reads as a
clone. Yahboom's drivers are DEPENDENCIES we launch, not work we present.

## 2. What the field rewards right now (research pull)

- Reusable eval pipelines, data engines, and sim loops "that travel across
  embodiments with less handcrafting" are the 2026 signal (ODSC roundup).
- Open physical-AI evaluation is being built in the open: Robocurve's
  Inspect Robots (MIT, YC) runs any LLM/VLA against real/sim benchmarks
  with immutable EvalLogs (config, git rev, transcripts, frames, Rerun).
  Embodiments today: YAM, Franka, AgiBot A2, Unitree G1, SO-ARM, WidowX,
  Isaac Lab, a generic ROS bridge. NO mobile manipulator, no sub-$1k
  mobile base, no Jetson-class on-device policy.
- Small VLAs that fit 8 GB exist (SmolVLA ~450M, NanoVLA on Orin Nano
  Super) and LeRobot is the common rail (SO-ARM, LeKiwi mobile base).
  Nobody has published an honest on-device VLA evidence pack for a
  Jetson Orin NX 8GB mobile manipulator with provenance and percentiles.
- Jetson AI Lab community winners are narrative + measured systems (VLA
  car on Thor, matcha bimanual, GPS-denied VSLAM drone, nvblox+Orbbec on
  AGX Orin) — not feature checklists.
- 8 GB is a real wall: nvblox + VSLAM + a camera wrapper already OOM an
  Orin Nano 8GB. A credible 8 GB project budgets memory explicitly.

## 3. What is uniquely ours to bring

1. Evaluation discipline that already exists in this repo: ledgers,
   pre-registered bars, frozen stats, the one correct percentile, honest
   false-positive handling (eval01 G5), provenance-first artifacts.
2. A three-machine lab: RTX 5090 trainer, AGX Thor (122 GB, TRT), Orin NX
   8GB on wheels — the exact "train anywhere, serve on the robot,
   evaluate in sim and real" loop, already proven once (ACT on Thor).
3. A LeRobot/GR00T/ACT data+training pipeline, Isaac Lab envs, and a
   vendor USD/URDF twin of the M3 Pro already in `isaac/usd/robots/`.
4. Safety observability (`physical-ai-safety-observability`, ops copilot
   node) — almost no hobby mobile-manipulation project has a runtime
   safety layer with structured events.
5. Telecom/edge background: latency budgets, offload decisions, fleet
   telemetry — the question "where should the policy run?" is native.

## 4. The flagship: "Audited Mobile Manipulation on 8 GB"

One narrative, four legs, each a standalone artifact with its own evidence.

### Leg A — Open eval embodiment (the differentiator)
Register the ROSMASTER M3 Pro as an Inspect Robots embodiment — the first
mobile manipulator there — plus its Isaac Lab twin as the sim embodiment
of the SAME action contract. Tasks: (1) tabletop pick-place (the Ludo
scorer, ported), (2) "mobile fetch": navigate to a marked station, pick,
return, place. Policies under test: our GR00T-1.7 and ACT checkpoints,
SmolVLA, and an LLM tool-calling agent (inspect-robots-agent) — the same
four scored on the same scenes, sim and real, with EvalLogs committed.
Upstream the adapter package (`inspect-robots-rosmaster`, MIT). Output:
a public leaderboard-style table nobody else has, and a merged OSS
contribution with our name on it.

### Leg B — On-device VLA evidence pack (the measurement)
LeRobot robot class for the M3 Pro arm+base (Rosmaster serial servos,
like LeKiwi's class) -> record 100-200 real episodes of the fetch task ->
fine-tune SmolVLA (and ACT as the floor) on the 5090 -> serve on the Orin
NX with TensorRT at fp16/int8 -> `day_one_smoke.py`-grade reporting:
n/p50/p95/p99/std latency, control-rate achieved vs required, VDD_IN
power, thermals, sustained memory, success rate with the Leg A scorer.
Same policy served from Thor over Wi-Fi as the comparison row (closes the
Thor-parity TODO). The deliverable is the table "what actually runs on
$500 of compute and what it costs in success rate".

### Leg C — Safety-observed autonomy (the operator story)
The fetch task runs with the safety platform in the loop: person-in-
workspace events from the depth camera (NanoOWL/YOLO open-vocab),
speed-envelope and stop-distance telemetry from the dual ToF LiDAR,
e-stop/arm-torque events, all as structured safety events to
`physical-ai-safety-observability`, with the ops copilot producing the
post-run triage. Measured: detection latency to motion-stop, false-stop
rate. This is the leg hiring managers in industrial robotics notice.

### Leg D — Sim-to-real gap ledger (the loop)
Isaac Lab env for the M3 Pro (USD exists; finish articulation + sensors),
domain-randomized nav+grasp, the same scorer in sim and real (Leg A), and
a standing "gap table": success sim vs real per task per policy, with
the causes ledgered (the way E31/eval01 are). PolaRiS-style real-to-sim
is the stretch reference.

### Stretch — Fleet/offload
Two embodiments (M3 Pro + Synria/Thor) sharing a task queue; policy
placement chosen by measured link latency; telemetry to the same safety
platform. Only after A-C have evidence.

## 5. What we explicitly will not do
- Re-present SLAM/Nav2/MoveIt/MediaPipe/YOLO/AprilTag lessons as projects.
- Fork YahboomTechnology/ROSMASTER-M3PRO; we depend on its drivers.
- Build another Dify/OpenClaw voice-agent; the agent we run is the
  open inspect-robots-agent, scored like any other policy.
- Isaac ROS nvblox/VSLAM on this 8 GB board: LiDAR SLAM (vendor) owns
  navigation; GPU memory is reserved for the policy and open-vocab
  perception. (Isaac ROS 3.x also targets Jazzy; this board is Humble.)

## 6. Constraints to design around
8 GB unified memory (budget per process, publish it); NVMe 89% full
(models on external/USB or prune vendor Docker images: Dify stack ~5 GB);
ROS 2 Humble / JetPack 6.2 / CUDA 12.6 / TRT 10.7; arm uses bus servos via
the STM32 board (no torque sensing — safety events use current/position
limits); Wi-Fi only.

## 7. Sequence (evidence at every step, nothing speculative in README)

DECIDED 2026-08-20: all four legs, in the order B -> A -> C -> D.
B first because it fills what the Alicia/Synria arm project lacks (real
hardware data, mobility, on-device inference, a second embodiment);
A next because it makes B comparable and public; C third because a
measured safety layer is absent from every robot project here and
from the field; D last because sim is already the portfolio's strength
and the M3 Pro twin is RTX-gated articulation work.

Week 1: LeRobot robot class for the M3 Pro (arm first), teleop + record
  10 episodes, `day_one_smoke`-style provenance on every recording.
  Inspect Robots: install, run CubePick mock + the ROS bridge against
  the M3 Pro arm, open the adapter repo skeleton.
Weeks 2-3: Leg A tabletop task on the real arm with ACT + LLM agent
  scored; first EvalLogs committed. Leg B: 100+ episodes, SmolVLA
  fine-tune on the 5090, TRT serve on the Orin, first latency/power rows.
Weeks 4-6: mobile fetch task (vendor Nav2 + our scorer), Leg C safety
  events wired, sim twin articulated for Leg D, gap table v1.
Then: upstream PR, write-up, video with the numbers on screen.

## 8. Sources
- yahboom.net/study/ROSMASTER-M3PRO (course catalog);
  github.com/YahboomTechnology/ROSMASTER-M3PRO; category.yahboom.net
- github.com/robocurve/inspect-robots; robocurve.org; inspectrobots.org
- huggingface.co/docs/lerobot/lekiwi; arxiv 2602.22818 (LeRobot)
- smolvla.net; arxiv 2510.25122 (NanoVLA); roboticscenter.ai VLA comparison
- jetson-ai-lab.com/community; developer.nvidia.com edge-AI on Jetson post
- nvidia-isaac-ros.github.io (nvblox), docs.nav2.org (Isaac Perceptor)
- opendatascience.com 2026 robotics projects; therobotreport.com

## 9. Leg B recon (2026-08-20, read-only, nothing moved)
- Arm path: `arm_driver` node -> `Rosmaster_Lib.Rosmaster` -> `/dev/myserial`
  (-> ttyUSB0, CP210x -> STM32 board). Calls: `set_uart_servo_angle(id,
  angle_deg, run_time_ms)`, `set_uart_servo_angle_array(joints, run_time)`,
  reads via `get_uart_servo_angle*`. Servo IDs 1-6, degrees; vendor home
  pose `[90, 150, 12, 20, 90, 0]` (servo 6 = gripper). Msgs: `ArmJoint`
  {id, run_time, angle, joints[]}, `ArmJoints` int16 x6 + time, `CurJoints`.
- CORRECTED (same day, robot powered on the stand): the M3 Pro is a
  micro-ROS design — `Rosmaster_Lib` is legacy and not needed. The STM32
  board runs a micro-ROS client over `/dev/myserial` @ 2,000,000 baud; the
  Orin runs `micro_ros_agent` natively (`~jetson/mircoROS_agent/install`,
  autostarted by `start_agent.sh`, ROS_DOMAIN_ID=30) plus the joystick node.
  Vendor workspaces ARE installed (`M3Pro_ws/install` 38 pkgs incl.
  ira_laser_tools + yahboom_laser_filter, `yahboomcar_ws/install`).
  Why nothing was visible: the agent/joy were started at boot while the
  clock read 2025-12-30; the NTP jump (+8 months) stalled FastDDS
  discovery (even the joy node vanished). Restarting the agent (the
  vendor's own `check_sensor.py` remedy) fixed it: session established,
  node `/YB_Node`. Restart recipe: kill `micro_ros_agent serial`, `ros2
  daemon stop`, relaunch the agent (log: ~jetson/agent_restart.log).
  LESSON: the Asia/Shanghai + wrong-clock factory state breaks DDS after
  NTP; now that TZ/clock are right this should not recur, but a DDS
  preflight (`ros2 node list --no-daemon --spin-time 6` must show
  /YB_Node) belongs in every launcher on this box.
- BOARD CONTRACT (read-only verified 2026-08-20, NO motion sent):
  publishes /scan0, /scan1 (LaserScan, 0..270deg, 0.54deg steps,
  0.05-12 m, frames laser0_frame/laser1_frame), /imu/data_raw (Imu, g on
  z = 9.85), /odom_raw (Odometry), /battery (Float32, 12.08 V);
  subscribes /arm6_joints (arm_msgs/ArmJoints: joint1..6 int16 deg +
  time ms), /arm_joint (ArmJoint: id, angle, run_time), /cmd_vel (Twist),
  /beep (UInt16), /rgb (ColorRGBA). Joint feedback is NOT published by
  the board; the vendor `arm_util/arm_status` node provides
  /arm_status/arm_joint (source for joint_state_publisher) — inspect
  how it reads angles before the LeRobot class relies on it.
- Direct serial config protocol (config_robot.py `MicroROS_Robot`,
  HEAD 0xFF, DEVICE 0xFC, 2 Mbaud): FIRMWARE_VERSION 0x51, REQUEST 0x50,
  ARM_TORQUE 0x22, ARM_OFFSET 0x24 (calibration), ARM_MID_VALUE 0x25,
  DOMAIN_ID 0x41 — config only; motion goes over micro-ROS topics. Do
  not open the serial port while the agent holds it.
- Vendor code drop cached on the 5090 scratchpad (1.3 GB
  `ROSMASTER M3 PRO-Code.zip` from the public Annex Drive folder
  13q6tB7duof3cQem1WR-J-hb1EEhl9ZX0: Board_Samples (STM32 + micro-ROS
  sources), orin/M3Pro_ws, orin/yahboomcar_ws, calibrate_arm.py,
  config_robot.py).
- `lerobot` not on the Orin; install into `oedeh`'s `.venv` (aarch64, torch
  2.5 system wheel) when the class is written. Serial is group `dialout`
  (oedeh is a member).

## 10. Added 2026-08-20: games on the M3 Pro (cross-cutting showcase)
Port the existing closed game loop (`ludo_engine` L4 + `game_core`
PickPlaceCommands + the turn executor's scorer; chess/checkers scenes
exist too) to the Yahboom arm. Stages:
1. M3 Pro arm plays Ludo turns on a physical board via the Leg B LeRobot
   class (scripted IK lane first — the 97.2% expert's discipline: scored
   turns, ledgered misses), same `turns.jsonl`/`ludo_stats.py` artifacts.
2. Same turns as Leg A tasks (Inspect Robots "ludo_turn" scorer already
   written) — policies (ACT/GR00T/SmolVLA) play real turns and get scored.
3. HEADLINE: two embodiments play each other — Synria arm (Thor) vs
   M3 Pro (Orin) — one game engine, two robots, each moving its own
   tokens; the mobile base can reposition to reach far squares (a real
   use of mobility). Telemetry + safety events (Leg C) from both.
4. Chess/checkers reuse the same loop once Ludo is stable.
Why it fits: real-hardware arm data for B, a public task for A, safety
in the loop for C, the sim twin plays the same engine for D — and it is
a demo people remember.

## 11. Preinstalled on the Orin (verified read-only 2026-08-20, factory image)
Runtime (autostart, ROS_DOMAIN_ID=30): `start_agent.sh` (micro_ros_agent
serial /dev/myserial 2 Mbaud), `joy_control/joy.sh` (yahboom_joy_M3Pro),
`check_sensor.py` (waits for /scan0 + /imu/data_raw, re-runs the agent),
`set.sh` (USB audio default sink/source). Services: jupyterlab,
yahboom_oled, ollama (installed, no models listed), docker, nvargus.
ROS 2 Humble apt: navigation2 + nav2-bringup, slam-toolbox,
cartographer-ros, rtabmap-ros, moveit, robot-localization, joy +
teleop-twist-joy, depthimage-to-laserscan, rosbridge-server, micro-ros-msgs.
Vendor workspaces (built): yahboomcar_ws = arm_interface, arm_kin (IK srv,
libarmkin.so), arm_msgs, M3Pro_config/demo/MoveIt_demo, orbbec_camera
(+msgs, description), yahboomcar_ctrl, yahboomcar_mediapipe,
yahboomcar_msgs, yahboom_M3Pro_DepthCam/description/laser, yahboom_yolov8.
M3Pro_ws = M3Pro core, M3Pro_navigation, slam_mapping/yahboom_mapping
(gmapping, cartographer, slam_toolbox configs), ekf_bringup, imu_tools,
ira_laser_tools (dual-scan merge), yahboom_laser_filter,
laserscan_to_point_publisher, calibration, patrol, M3Pro_KCF, largemodel +
largemodel_arm + text_chat + multi_brains (the LLM/OpenClaw layer: dashscope,
openai, dify-client), yahboom_app_save_map + web_savmap_interfaces,
updatecostmap, robot_pose_publisher_ros2. Assets: rtabmap.db,
yahboom_map.pbstream, rtab_nav*.rviz (a prior map of somewhere).
Python (system, aarch64): torch 2.5.0 nv24.8 + torchvision/torchaudio,
tensorrt 10.7 (+lean/dispatch), onnxruntime 1.22 + onnxruntime-gpu 1.20,
ultralytics 8.3.65, mediapipe 0.10.9, funasr 1.2.6 (ASR, MODELS/asr 897M),
piper-tts 1.3 (MODELS/tts 191M), openai 1.95, dashscope, dify-client,
pyserial, jetson-stats. Dify stack in Docker (~5.3 GB of images) +
weaviate. JEP 1.4G (Jetson examples), jetcam, jetson-gpio,
qr_code_recognition, uros_ws (micro-ROS client sources).
Disk: was 92% full (7.7 GB free); 2026-08-20 removed the unused Dify
stack (12 containers dead since the flash + 10 images, ~5.7 GB) at the
operator's request -> 14 GB free (85%). Kept: micro-ros-agent image, 17
Docker volumes (242 MB Dify/open-webui data), ~/software/dify sources
(372 MB), JEP 1.4G. The OpenClaw/Dify LLM layer is therefore not
runnable on this box anymore (by design — see section 5); `docker compose
pull` in ~/software/dify restores it.
DELETION RULE (operator, 2026-08-20): on the Orin remove only what we are
sure we will not use AND can easily reinstall. Data volumes, vendor
workspaces, ML/ASR/TTS stacks, and example trees stay unless both
conditions hold; ask before touching anything that is data.
Implication for Legs A-D: navigation, SLAM, MoveIt, IK, depth camera and
YOLO are all present as dependencies; our code adds the LeRobot robot
class (publishing /arm6_joints, /cmd_vel; consuming /scan*, /imu, /odom,
camera), the eval embodiment, the safety layer, and the provenance.

## 12. First motion tests (2026-08-20, operator present, robot on the stand)
Order and results (all via scripts/jetson/m3pro_pub.py, ROS_DOMAIN_ID=30,
as `oedeh` with the UDP-only FastDDS profile; `m3pro_preflight` first):
1. Zero-motion link: /beep 100->0 and /rgb blue->off — accepted (1
   subscriber = /YB_Node), operator to confirm audible/visible.
2. Base: /cmd_vel vx=0.10 m/s for 1.0 s then zero — /odom_raw went
   x 0 -> +57 mm (ramp-limited), y +20 mm, velocity ~0 after stop.
   Operator confirmed wheels spun and stopped.
3. Arm: /arm6_joints -> vendor home [90,150,12,20,90,0] over 3000 ms —
   operator confirmed "arm raised". Then gripper (servo 6) 0->40->0 and
   joint1 90->100->90 at 1500 ms each — accepted AND operator-confirmed
   (gripper opened/closed, joint 1 swung and returned); battery 12.08 ->
   11.99 V across the session. Operator also confirmed the wheels kept
   spinning after the crashed test until the manual stop (no watchdog).
Facts established for the LeRobot class:
- Command path: ArmJoints on /arm6_joints (int16 degrees x6 + run_time ms)
  — absolute servo angles, board interpolates over run_time; ArmJoint on
  /arm_joint for a single servo. Base: Twist on /cmd_vel (30 Hz-ish
  re-publish; the board stops when commands stop? — NOT yet verified,
  always send explicit zeros).
- State path: the board publishes NO joint angles; vendor demos publish
  their own `Curjoints` (CurJoints int16[]) from the last command
  (open-loop). observation.state will be commanded angles until a
  firmware joint-state publisher exists (micro-ROS client sources at
  ~jetson/uros_ws; STM32 flashing risk — later).
- Gotchas: `ros2 topic pub --once` hangs on shutdown against micro-ROS
  (two stuck procs) — use m3pro_pub.py / rclpy; Wi-Fi power save is ON
  on wlP1p1s0 (caused 30% loss / 300 ms RTT mid-session). FIXED by the
  operator: `iw ... set power_save off` + nmcli profile powersave=disable
  (persistent) -> 40/40 pings, 0% loss, 3.4 ms avg, -46 dBm; a second Linux user needs the UDP-only profile
  (scripts/jetson/config/fastdds_no_shm.xml) to see the agent.
SAFETY FINDING (m3pro_watchdog_test.py, 2026-08-20): the base firmware has
NO /cmd_vel timeout. After 1 s of vx=0.10 and then silence, /odom_raw
velocity stayed ~0.08 m/s for the full 4 s sample window — motion persists
until an explicit zero Twist. Consequences (non-negotiable for Legs B/C):
every base controller owns a deadman — zeros on exit, on exception (try/
finally), and on heartbeat loss (>0.5 s without a fresh policy/teleop
command => zero); a supervisor node republishes zeros when the controller
process dies; the hardware e-stop is tested before any untethered run.
Incident: an earlier quoting bug crashed a test script after its 1 s
command and before its zero — wheels ran on the stand ~20-30 s until a
manual stop. This is the first Leg C safety event, recorded as such.

## 13. Leg B started (2026-08-20): LeRobot plugin written
`robots/lerobot_robot_rosmaster_m3pro/` — pip-installable LeRobot 0.6.2
plugin (auto-discovered via the `lerobot_robot_*` distribution-name rule,
so `lerobot-record/teleoperate --robot.type=rosmaster_m3pro` work
unmodified). Robot: ArmJoints on /arm6_joints (absolute deg, run_time
150 ms), Twist on /cmd_vel, Odometry + battery subscribers, OpenCV
cameras; open-loop arm state (homes on connect — the only way to have a
valid state), joint limits + max_relative_target (10 deg/cmd) clamps,
base speed clamps, deadman thread (zeros after 0.5 s without an action),
zeros on disconnect and on any send_action exception, board preflight
(publisher must be matched by /YB_Node). Teleop:
`rosmaster_m3pro_keyboard` (pynput; q/a..y/h arm, i/k j/l u/o base,
space stop, 0 home). `m3pro-smoke` console script. Verified on the 5090:
types register, 13 obs / 9 action features. Orin: venv
`~/.venv-lerobot` (system-site: Jetson torch) + editable install of the
plugin; rclpy/arm_msgs come from the sourced ROS env; run from /tmp
(repo-local `lerobot/` dir shadows the library).

## 14. Leg B PIVOT (2026-08-20): host/client split (LeKiwi pattern)
BLOCKER: LeRobot 0.6.2 requires Python >=3.12; the Orin is Python 3.10 and
its CUDA torch is a cp310 Jetson wheel (no cp312 build), so a 3.12 venv
would lose GPU torch. Max lerobot on py3.10 is 0.4.4 (different API, no
`lerobot_robot_*` plugin discovery). Therefore the plugin cannot run ON
the Orin as first written.
RESOLUTION (also the better architecture, and what Leg B intended): split
like LeRobot's own LeKiwi (host on the Pi, client on the workstation).
  - Orin HOST (py3.10, deps rclpy+cv2+zmq+numpy, NO lerobot): owns the
    board via the measured contract, applies the safety floor (joint +
    relative clamps, base speed clamps, deadman/watchdog), serves
    observations over ZMQ PUSH and receives actions over ZMQ PULL
    (LeKiwi wire: cmd PULL+CONFLATE :5555, obs PUSH+SNDHWM2 :5556,
    watchdog 500 ms, JSON header + JPEG frames multipart).
  - 5090 CLIENT (lerobot 0.6.2, py3.12, venv lerobot17): a Robot subclass
    `rosmaster_m3pro_client` (ZMQ to the Orin) — records LeRobotDataset,
    fine-tunes, serves policies. This IS the train-on-5090 / act-on-robot
    loop, now honest about the Python/torch split.
REFACTOR: move the board I/O + safety out of robot.py into a lerobot-free
`hardware.py` (importable by the host); keep `RosmasterM3Pro(Robot)` as the
local/direct class for a future py3.12+ROS environment; add
`m3pro_host.py` (Orin) and `client.py` (`rosmaster_m3pro_client`, 5090).
ASIDE: the windowed Ludo demo (ludo_live_06) crashes with the Isaac
'--enable_cameras' camera error ONLY while the headless corpus recorder
(corpus02b) holds the Kit renderer/kvdb lock; same argv ran fine this
morning (ludo_live_02, 4/4) with the GPU free. Run windowed demos when no
headless Isaac job is active. Rendered MP4s of the expert game and the
GR00T eval were delivered to the operator.

## 15. Leg B on-robot bring-up (2026-08-20): software done, blocked on link
Software COMPLETE and committed: hardware.py (lerobot-free board I/O +
safety), m3pro_host.py (Orin ZMQ host — now connects the board BEFORE
cameras and treats camera-open as non-fatal with a bounded V4L2 timeout),
client.py (5090 LeRobot client), scripts/jetson/m3pro_host.sh launcher
(prepends the plugin to the ROS PYTHONPATH — a bare `PYTHONPATH=` drops
rclpy), orin_ros_env.sh. All types register on lerobot 0.6.2; host imports
without lerobot; system python3 on the Orin (ROS sourced) has rclpy+cv2+zmq.
Board verified live (11.1 V charging), motions validated earlier.
BLOCKED on the on-robot host<->client loop by the ENVIRONMENT, not code:
  1. Wi-Fi is intermittently dropping mid-command (0% loss one moment,
     total drop the next) — background launches and multi-second commands
     get cut off, so the host can't be reliably started/verified over SSH.
     FIX: wire the Orin to Ethernet (eno1) for data collection — removes the
     variable entirely. Recommended before the next on-robot session.
  2. cv2.VideoCapture(/dev/video0) hangs when a prior host was hard-killed
     and left the device locked (fixed in code: warn+skip+timeout; a clean
     robot reboot clears it).
  3. micro-ROS agent goes stale after an Orin reboot (STM32 keeps the old
     session) — recover by resetting the MCU (config_robot.reboot_device,
     no motion) + relaunching the agent (serial must be free; verify with
     fuser /dev/ttyUSB0, NOT pgrep which matches our own ssh cmd string).
NEXT SESSION (on Ethernet, robot rebooted clean): m3pro_host.sh --no-base
-> 5090 client smoke (RosmasterM3ProClient connect, read obs incl. camera,
send home) -> decide teleop scheme (vendor joystick observed vs keyboard)
-> record 10 episodes with provenance.
DECISION (operator): Orin left as-is (GUI + vendor stack) for now — 5.6 GB
free is enough for recording; revisit headless when on-device policy
serving needs the headroom.

## 16. Leg B host<->client loop VERIFIED on hardware (2026-08-20)
Full loop works end to end: Orin `m3pro_host` (arm+camera, board-first,
serving :5555/:5556) <- ZMQ -> `RosmasterM3ProClient` on the 5090
(lerobot 0.6.2). Client connected in 0.08 s; observations carried the 6
joint angles [90,150,12,20,90,0], battery ~10.9 V, and a live 480x640x3
Orbbec frame (saved, delivered to operator); a home action round-tripped.
ROOT CAUSE of the hours of "flaky launches": TWO self-inflicted bugs,
NOT Wi-Fi/camera as first thought:
  1. `pkill -f m3pro_host` inside an ssh command matched the ssh command's
     OWN cmdline (it contains "m3pro_host") and SIGKILLed the shell ->
     exit 255, empty output. FIX: anchor the pattern to the process, e.g.
     `pkill -9 -f "^/usr/bin/python3 -u -m lerobot"` (bash shell cmdline
     starts with "bash", so it is not matched).
  2. Opening a NEW ssh per command hammered the Orin sshd (connection
     storm -> intermittent refusals that looked like drops). FIX: ssh
     ControlMaster multiplexing (added to ~/.ssh/config for jetsonorin:
     ControlPath/ControlPersist 600).
Also real but secondary: `PYTHONPATH=pkg python` DROPS the ROS PYTHONPATH
(rclpy) -> must PREPEND `:$PYTHONPATH`; camera-open can hang -> host now
connects the board first and treats the camera as non-fatal.
Wired Ethernet (eno1 192.168.1.251) is stable (0% loss, ~0.9 ms) and is
now the alias/remote target; Wi-Fi (243) was a lesser factor than the two
bugs above.
BATTERY 10.9 V and dropping across the day (12.08 -> 10.88) — charge
before long recording; a low pack also risks servo brownouts.
NEXT: pick the teleop scheme (vendor joystick observed via a teleoperator
that subscribes to /arm6_joints + /cmd_vel, vs keyboard) and record the
first 10 episodes with provenance (host --no-base off for base motion,
robot on the floor with the deadman + operator e-stop).

## 17. Leg B joystick-observe teleop + recorder built (2026-08-20)
Chosen scheme: the operator drives with the familiar VENDOR JOYSTICK
(yahboom_joy_M3Pro already publishes /arm6_joints + /cmd_vel to the board);
we record passively. Implemented and verified (imports/registration on
lerobot 0.6.2; live recording pending the charged robot):
  - hardware.py `passive=True`: subscribes to /arm6_joints (ArmJoints) and
    /cmd_vel (Twist), captures the joystick command as the action, and does
    NOT publish/home/deadman (the joystick owns the board). read_action()
    returns the last joystick command; read_state() uses it as the arm-state
    proxy (board has no joint feedback -> the CAMERA is the real state).
  - m3pro_host --passive: embeds `action` (joystick cmd) in each obs header.
  - client passive: get_observation() surfaces the captured action via
    captured_action(); send_action() is a no-op (never fights the joystick).
  - record.py (5090, lerobot 0.6.2): LeRobotDataset.create(video) recorder;
    loops at fps reading client.get_observation()+captured_action(), writes
    observation.images.front / observation.state / action (+base if
    --use-base), task-in-frame; per-episode save + reset window.
RUN (charged robot, on the floor, operator + e-stop):
  Orin:  bash scripts/jetson/m3pro_host.sh --passive --cameras front:0:640:480:15
  5090:  cd /tmp && ~/.venv/lerobot17/bin/python -m \
         lerobot_robot_rosmaster_m3pro.record --repo-id oedeh/m3pro_teleop_v1 \
         --episodes 10 --episode-time-s 20 --remote-ip 192.168.1.251 \
         --task "pick up the cube and place it on the plate" [--use-base]
DATASET semantics: vision-based imitation. observation.state = joystick's
last arm command (proxy), action = joystick command this frame, the front
camera carries the real scene/arm state. Fine for BC/VLA; note state and
action are one-step-related, not independent (open-loop board).
OPS REMINDERS baked from today: ssh ControlMaster is in ~/.ssh/config
(avoid the connection storm); kill host with anchored pattern
`pkill -9 -f "^/usr/bin/python3 -u -m lerobot"` (bare -f m3pro_host
self-matches the ssh cmd); host launcher prepends the ROS PYTHONPATH.

RECORDER VALIDATED (2026-08-20, synthetic frames on the 5090): record.py
dataset path works end to end — LeRobotDataset.create(video) + add_frame
(task-in-frame) + save_episode wrote 2 eps / 40 frames: data parquet, h264
mp4 per camera, meta/info+stats+tasks. The charged recording session is
de-risked; only the live joystick drive remains.

## 18. Leg A STARTED — Inspect Robots embodiment conformant (2026-08-20)
Robocurve's Inspect Robots (real one, PyPI `inspect-robots` 0.57.1, homepage
github.com/robocurve/inspect-robots — light base deps are correct, VLA
backends are extras) installed on the 5090; CubePick eval validated end to
end (scripted policy 2/2, EvalLog written).
Adapter `robots/inspect-robots-rosmaster/` written and PASSES conformance:
  `inspect-robots doctor --embodiment rosmaster_m3pro` -> "conformant (no
  issues)". v1 is arm-only (`joint_pos`, 6 abs joint degrees), front camera,
  human reset (console poll), NO privileged success (real robot) — wraps our
  `RosmasterM3ProClient` (active mode). Registered via the
  `inspect_robots.embodiments` entry point. Installed alongside lerobot 0.6.2
  in the lerobot17 venv (numpy 2.2.6, no conflicts).
API learned: Embodiment = reset(scene,seed)->Observation + step(action)->
StepResult + close() + .info (EmbodimentInfo: Box action_space with
ActionSemantics control_mode in {joint_pos,eef_abs_pose,eef_delta_pos},
ObservationSpace with cameras + StateSpec(StateField(key,shape,unit))).
Conformance gotcha: absolute-target control REQUIRES one StateSpec field of
the action's shape as the proprio reference (state_keys alone fails).
NEXT (Leg A): a `min_distance`/Ludo-style scorer for tabletop pick-place;
run a real eval (host ACTIVE mode) with the `random`/`scripted`/`agent`
policy once the robot is charged; then a VLA policy. Mobile-fetch (base)
is the v2 embodiment.

## 19. Leg A scorer built (2026-08-20): m3pro_pickplace
Registered scorer `m3pro_pickplace` (inspect_robots.scorers entry point) for
the tabletop pick-place task. Real-robot HONEST (eval01 lesson): the board has
no object-pose sensor / no joint feedback, so there is NO privileged success
oracle. Design:
  - authoritative success = the operator/VLM grader (record.operator_judgement);
  - the scorer ADDS the measurable pick-place funnel from the recorded ARM
    trajectory (action/joint_pos 6-vectors): arm_active (joint travel >
    threshold), grasp_actuated (gripper joint6 moved > 15deg), grasp_release_
    cycle (closed then returned to ~open = pick+place signature), arm_travel_deg;
  - unattended fallback = a trajectory PROXY, explicitly labelled "NOT ground-
    truth success", so a log never fabricates success.
Scorer contract learned: `@scorer("name")` factory (from inspect_robots.registry,
NOT .scorer) returns a frozen class with `.name` + `__call__(record: TrialRecord,
target) -> Score(value, explanation, metadata)`; consumes the RECORDED
trajectory (record.steps[*].observation/action/result + operator_judgement) so
scoring is reproducible from the saved EvalLog. Unit-tested: grasp+release+
active->proxy True; operator judgement authoritative; idle->False; grasp-without-
release->False.
LEG A STATUS: embodiment conformant + scorer registered + CubePick validated.
A real eval is now one command once the robot is charged (host ACTIVE mode):
  inspect-robots run --embodiment rosmaster_m3pro --scorer m3pro_pickplace \
    --grader operator --instruction "pick up the cube and place it on the plate" \
    --policy scripted -E remote_ip=192.168.1.243

## 20. Leg C STARTED — safety-observability bridge (2026-08-20)
The M3 Pro now emits structured SafetyEvents into
physical-ai-safety-observability (makes that repo's "Jetson runtime metrics"
integration real). Built + verified END-TO-END (evaluated events POST to a
live API on :8077 and persist; GET returns them with correct severity/review):
  - hardware.py: subscribes /scan0 + /scan1 (LaserScan), read_state() carries
    min_obstacle_m (nearest valid return across both ToF lidars).
  - safety.py (pure, stdlib): evaluate(snapshot,cfg) -> SafetyEvent dicts,
    schema-matched to the platform (camera_id, rule_id, severity, confidence,
    human_review_required, evidence{telemetry_snapshot...}, summary). Rules:
    m3pro.battery (10.8/10.2/9.8 V -> med/high/crit), m3pro.stop_distance
    (only while base moving: 0.70/0.40/0.20 m -> med/high/crit),
    m3pro.deadman + m3pro.estop (high), m3pro.person_in_zone (conf-gated).
    Honest: rule-triggered from measured telemetry, not model guesses; every
    emitted dict constructs as the platform's SafetyEvent (validated).
  - safety_bridge.py: reads the host obs stream (battery, base speed,
    min_obstacle_m), evaluates, POSTs /events with a JSONL fallback and a
    per-(rule,severity) cooldown so a standing condition is not spammed.
RUN (recording/eval session, host in ACTIVE or PASSIVE mode both fine):
  cd /tmp && ~/.venv/lerobot17/bin/python -m \
    lerobot_robot_rosmaster_m3pro.safety_bridge --remote-ip 192.168.1.243 \
    --api http://<safety-host>:8000 --hz 5
LIVE-only remaining: person detection (NanoOWL/YOLO on the front camera ->
snapshot['person']) and detection-latency-to-stop timing; the wiring points
are in place (snapshot['person'], snapshot['estop'/'deadman_fired']).

## 21. Joystick teleop chain — gotchas (2026-08-20)
Dongle plugged AFTER boot, so the vendor joy_node (started at boot) never
bound /dev/input/js0. Restart needed. But headless restart hit walls:
  - the ROS `joy` driver is SDL-based and enumerates NO device headless on
    this Orin (raw `cat /dev/input/js0` DOES stream — hardware fine);
  - the vendor `yahboomcar_joy_launch.py` pulls a Qt/GUI component that fails
    headless ("no Qt platform plugin"); it works only in the desktop session;
  - `joy_linux` (evdev, no SDL) is NOT installed.
FIX shipped: `scripts/jetson/js0_joy.py` — reads /dev/input/js0 via the legacy
joystick API directly (no SDL/display/pkg) and publishes /joy; the vendor
`yahboom_joy_M3Pro` maps /joy -> /arm6_joints. Must run as a user that can
read js0 (jetson can; the removed jetson clone means copy it to /tmp) and
that shares the mapper's DDS (run both as `jetson`, ROS_DOMAIN_ID=30).
RELIABLE RUN (locally on the Orin — remote background-launch fought Wi-Fi):
  pkill -f js0_joy; pkill -f "joy/joy_node"
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=30 && source ~/yahboomcar_ws/install/setup.bash
  python3 /tmp/js0_joy.py --dev /dev/input/js0 &
  ros2 run yahboomcar_ctrl yahboom_joy_M3Pro &
Then joystick -> arm. (Alternative if this stays painful: install
ros-humble-joy-linux via apt, or use our keyboard teleop via the client in
ACTIVE mode.) NOTE: the recurring remote-launch failures were the ssh
connection storm (fixed for oedeh via ControlMaster; jetson@ has no mux) +
Wi-Fi drops — run joystick bring-up on the Orin console, not over ssh.

## 22. Joystick BLOCKED on kernel (2026-08-20) — pivot to keyboard
The bundled pad is an X-INPUT-only Xbox360 controller (USB 045e:028e, fixed —
no DirectInput/HID mode). The Jetson kernel (5.15.148-tegra) is stripped:
  - `# CONFIG_INPUT_UINPUT is not set` -> no /dev/uinput -> xboxdrv (userspace
    driver, installed, DETECTS the pad) cannot create a virtual joystick;
  - no `xpad` module either;
  - joydev IS present (CONFIG_INPUT_JOYDEV=m) so a NATIVE HID joystick would
    work, but this pad exposes only X-input which needs xpad/uinput.
=> Using THIS controller needs a kernel rebuild (CONFIG_INPUT_UINPUT=y +
optionally JOYSTICK_XPAD) — deferred, not worth it mid-session. Earlier "js0
with 184 bytes" was the MPI7003 HID display board, not the game pad.
PIVOT for first-dataset recording: our keyboard teleoperator
(`rosmaster_m3pro_keyboard`, already built) drives the arm via the client in
ACTIVE mode from the 5090 — no Orin kernel change. Alternatives for later: a
DirectInput/generic-HID gamepad (joydev handles it natively, zero kernel
work), or the kernel rebuild for the Xbox pad.

## 23. Joystick SOLVED (2026-08-20) — PC mode + systemd services
THE FIX (operator was right that it ships working): the bundled pad has two
modes. GREEN = X-BOX/X-input (USB 045e:028e) — needs uinput+xpad, which the
Jetson kernel LACKS (CONFIG_INPUT_UINPUT not set), so it's dead on Jetson.
RED = PC/PCS/DirectInput (USB 0079:181c DragonRise) — a generic HID pad that
the already-loaded joydev exposes as /dev/input/js0 with ZERO driver work.
Switch: hold MODE ~15 s until RED, press START. (Per Yahboom's own "Handle
control" doc: Jetson uses PC mode / red; RPi uses X-BOX / green.)
Then the chain, made durable as systemd services (remote background launches
kept dying to Wi-Fi + the pkill-self-match; services survive drops/reboots):
  - js0_joy.service: scripts/jetson/js0_joy.py reads /dev/input/js0 (legacy
    joystick API, no SDL/display) -> /joy. Verified: 8 axes, 15 buttons,
    stick full deflection max_axis=1.000, ~20 Hz.
  - yahboom-joy-mapper.service: ros2 run yahboomcar_ctrl yahboom_joy_M3Pro
    (runs headless; the vendor LAUNCH file fails on Qt, the bare node is fine)
    subscribes /joy -> publishes /arm6_joints.
Files staged in /home/jetson (persistent; /tmp clears on reboot):
  js0_joy.py, js0-joy.service, yahboom-joy-mapper.service, fastdds_udp.xml.
Install (operator sudo, one line each): cp unit to /etc/systemd/system,
daemon-reload, enable --now. WHY services and not the vendor autostart: the
autostart runs joy in the GUI session (SDL needs a display) and only binds
js0 if the dongle was present at boot — brittle; systemd + js0_joy is
headless and reconnect-proof.
INFRA that finally made remote ops reliable: ssh ControlMaster mux for BOTH
oedeh (jetsonorin) and jetson (orinjetson) aliases; wired Ethernet
(192.168.1.251) as the alias target (0% loss vs Wi-Fi drops). Kill the reader
with the anchored pattern, never `pkill -f m3pro_host`/`js0_joy` bare inside
an ssh cmd (self-match -> 255).

## 24. Joystick — REVERT to vendor stack (2026-08-20, operator's call)
Correction: don't reinvent it. Yahboom SHIPS working joystick control
(ros `joy` joy_node + yahboomcar_ctrl/yahboom_joy_M3Pro mapper), confirmed
in their launch (package='joy', executable='joy_node') and Handle-control
doc. Two real blockers, now understood:
  1. pad was in X-BOX mode (green) — needs PC mode (red) on Jetson (done);
  2. the vendor joy_node is SDL -> needs a DISPLAY, so it only runs in the
     DESKTOP session (their autostart uses gnome-terminal). Headless (ssh/
     systemd) SDL enumerates nothing — which is why my headless js0_joy/
     js0_to_arm detour happened. That was over-engineering.
PLAN: disable my custom services (js0-to-arm/js0-joy/joy-arm-teleop/
yahboom-joy-mapper) + reboot -> the desktop autostart runs the vendor
joystick with the pad in PC mode. Vendor control (from yahboom_joy_M3Pro):
press START (buttons[9]) to enable arm_activa, then A/B/X/Y + D-pad move
joints; commands go to /arm_joint (single ArmJoint), NOT /arm6_joints.
=> the passive RECORDER must capture /arm_joint (+ reconstruct the 6-vec)
and /cmd_vel, not just /arm6_joints. (js0_to_arm.py kept in scripts/jetson
as a headless fallback only.) Also fixed the account confusion: removed
the oedeh user on the Orin; ONE account (jetson) there, oedeh on the 5090;
jetsonorin ssh alias now -> jetson.

WIFI ROOT-CAUSE FIX (2026-08-20): the Orin had a NetworkManager `Hotspot`
connection with autoconnect=yes (+ stray Pi_Hot/Yahboom2/3/MIFI/DIRECT
nets). At boot it could flip wlP1p1s0 to AP mode and knock rocagen off —
the likely source of the session-long Wi-Fi drops. Fixed: autoconnect=no
on Hotspot + all strays, rocagen autoconnect=yes priority=100. Boots
straight to rocagen (managed). Wired eno1 192.168.1.251 preferred when
plugged. Also: STM32 board goes stale after an Orin reboot (agent restarts
but client does not re-register) -> power-cycle the control board, or the
vendor check_sensor autostart should re-run the agent.

## 25. Joystick WORKING via OEM stack (2026-08-20) + recorder ready
After the clean reboot (pad in PC/DirectInput mode, Wi-Fi fix): the VENDOR
joy_node publishes /joy WITH data (sticks to 1.0, buttons register) and the
vendor yahboom_joy_M3Pro mapper drives the arm — operator confirmed the OEM
buttons control it. My earlier "SDL broken" call was pre-reboot state; the
vendor stack works. Root issue was TWO teleop systems fighting (my
js0-to-arm publishing /arm6_joints vs the OEM mapper publishing /arm_joint)
-> disable js0-to-arm (+ js0-joy), use ONLY the OEM stack.
OEM control map (from yahboom_joy_M3Pro.py, verbatim): START(btn9)=enable
arm_activa; X=j1- B=j1+ A=j2- Y=j2+; D-pad L/R=j3 U/D=j4; L1/L2=j6(gripper)
or j5. Publishes single-joint /arm_joint (arm_msgs/ArmJoint id+angle), NOT
/arm6_joints; + /cmd_vel for base.
RECORDER ADAPTED (committed): hardware.py passive mode now subscribes
/arm_joint too, inits the running 6-vector to home and applies each
single-joint update -> record.py captures the reconstructed full pose from
the OEM teleop. Verified with synthetic ArmJoint updates.
RUN RECIPE (once js0-to-arm/js0-joy disabled; Ethernet preferred):
  Orin (jetson): bash scripts/jetson/m3pro_host.sh --passive --cameras front:0:640:480:15
  5090: cd /tmp && ~/.venv/lerobot17/bin/python -m \
    lerobot_robot_rosmaster_m3pro.record --repo-id oedeh/m3pro_teleop_v1 \
    --episodes 10 --episode-time-s 20 --remote-ip <orin-ip> \
    --task "pick up the cube and place it on the plate" --use-base
NOTE: for robustness on this flaky-Wi-Fi robot, an on-Orin RAW recorder
(rclpy+cv2, no lerobot, write states/actions/mp4 -> convert on the 5090)
would avoid the live ZMQ dependency — build if Wi-Fi recording proves
unreliable and Ethernet isn't available.

## 26. Bundled controller is FAULTY/limited (2026-08-20) — arm teleop blocked
After exhaustive debugging (PC mode, analog mode, service cleanup, calibration):
the bundled DragonRise DirectInput pad only produces working analog on axes
4/5 (ONE stick — drives the BASE via the vendor mapper, which works well).
Axes 0-3 (second stick) and 6/7 (D-pad) are FLAT even when moved; buttons
register unreliably (mostly index 0, an intermittent stuck one). => no working
input remains for the ARM. Vendor arm control needs buttons[9]=START + A/B/X/Y
which don't work on this pad. NOT a software issue — board/wifi/recorder/base
all work. Time sink >> value here.
DECISION PENDING (operator): (a) standard USB gamepad (2 sticks+buttons) ->
works immediately; (b) keyboard teleop (rosmaster_m3pro_keyboard, from the
5090) for the arm now; (c) scripted arm demos (Isaac-expert style) for clean
non-teleop data. Base teleop is fine with the current pad regardless.
EVERYTHING ELSE READY: board up, wifi hotspot-autoconnect disabled (stability),
recorder captures /arm_joint + reconstructs pose, ZMQ host/client + record.py
validated, Leg A embodiment+scorer conformant, Leg C safety bridge verified.

---

## 27. Leg B — recording pipeline LIVE (end-to-end verified 2026-08-20 ~23:15)

**Status: joystick abandoned (faulty DragonRise pad); keyboard-from-5090 path
is fully built, verified end-to-end, and push-button.** The only remaining step
is a human driving the arm to record episodes — deferred (operator wrapped for
the day).

What is live and verified:
- **Orin host = systemd service** `m3pro-host.service` (User=jetson, UDP FastDDS
  profile, `--no-home` to avoid crash-loop homing, active mode). `is-active`,
  both ZMQ ports listening (`:5555` cmd, `:5556` obs), camera `front` streaming.
  Auto-starts on boot. This replaces held-SSH background tasks (those die 255).
- **5090 client smoke PASSED**: connect 0.14s, camera (480,640,3) uint8,
  battery 11.68V, home command landed `[90,150,12,20,90,90]` — gripper fix (90,
  not 0) confirmed on hardware, no stall/twitch.
- **`record_kbd.py`** (committed f156bfa): keyboard active-mode recorder. 5090
  drives arm+base via client→ZMQ→host→board and records
  observation.images.front / observation.state / action to a LeRobotDataset.
  Skips NaN-state frames until first key press; per-episode save + reset window.

### Push-button drive+record (next session, once robot is powered on):
```
# Orin host auto-starts on boot; verify from 5090:
ssh jetsonorin 'systemctl is-active m3pro-host'      # -> active
# then drive+record from the 5090 (operator at the keyboard, scene set up):
cd /tmp && DISPLAY=:1 ~/.venv/lerobot17/bin/python -m \
    lerobot_robot_rosmaster_m3pro.record_kbd --repo-id oedeh/m3pro_kbd_v1 \
    --remote-ip 192.168.1.251 --episodes 8 --episode-time-s 25 --use-base
```
Keys (focus the terminal running it): `q/a`=j1 `w/s`=j2 `e/d`=j3 `r/f`=j4
`t/g`=j5 `y/h`=gripper(j6); base `i/k`=vx `j/l`=vy `u/o`=wz; `space`=stop base
`0`=home. First key press establishes the arm state (frames record after that).
2°/tick. If Wi-Fi/ZMQ proves flaky, fall back to the on-Orin RAW recorder
(rclpy+cv2, no lerobot) and convert on the 5090.

Battery was 11.68V at wrap. Servos hold at home while energized; power-cycle the
robot to relax them. js0-to-arm / vendor joy nodes killed; DragonRise unplugged.

---

## 28. 5090 + Thor autonomous session status (2026-08-20 late)

**5090 (flagship GR00T-Ludo):**
- corpus batch-2 top-up **complete** — `reports/ludo_corpus02b`, 127 eps / 183
  attempts (raw; constant-instruction, target-blind like corpus01).
- GR00T-1.7 closed-loop **eval01 root-caused**: 1/20 (5%) = chance placement
  (1/46 squares); the place target was unobservable (constant instruction).
  Full analysis: `docs/notes/groot17_eval01_findings_2026-08-20.md`.
- **Fix committed** (575a2a9): corpus mode now renders target-observable
  `{pick}/{place}` instructions; verified end-to-end (ep0 manifest:
  `target_observable=True, place=track29, instr='...place it on track 29'`).
- **Overnight experiment RUNNING**: `reports/ludo_corpus03_tobs` — 150 eps,
  seed 202, `--instruction "...place it on track {place}"`. A target-observable
  twin of corpus01 (only the instruction differs) for a controlled retrain +
  re-eval. ~12h; notifies on completion.
  - NEXT (after it finishes): convert to LeRobot → retrain the same LoRA →
    re-eval closed-loop → compare placement err vs eval01's p50 193mm.

**Thor (Synria box):** synced to origin/main (0 behind, 575a2a9). No training
running. Two **stale idle policy servers** (0 clients): `serve_act_policy.py`
(synria_act_v*) up ~9.7d and `run_gr00t_server.py`
(gr00t_synria_thor_v5_vis40k) up ~6.6d, ~5GB RAM total. Safe to reap when
Synria eval isn't needed: `pkill -f serve_act_policy; pkill -f run_gr00t_server`.
Disk 94% used (62G free) — worth watching. `reports/training/synria_e46_lerobot_v21/`
is an untracked local LeRobot dataset (correctly out of git).
