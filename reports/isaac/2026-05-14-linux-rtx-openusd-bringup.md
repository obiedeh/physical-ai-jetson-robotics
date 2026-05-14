# OpenUSD / Isaac Digital Twin Report

Date: 2026-05-14

Operator: oedeh

Machine: aimlstation, ASUSTeK COMPUTER INC. ROG G700TF, Ubuntu 24.04.4 LTS, Linux 6.17.0-23-generic, x86_64

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB VRAM
- Driver: 595.58.03; NVIDIA-SMI 580.126.09; reported CUDA runtime 13.2
- Isaac Sim version: 5.1.0 (`/home/oedeh/isaacsim/VERSION`: `5.1.0-rc.19+release.26219.9c81211b.gl`)
- Isaac Lab version: not found under `/home/oedeh`; not installed in this repo
- Python: 3.12.3
- Repo commit: d8f09ca62851523ba9a35e0b4c00cee8c91ffb0a

## Inventory Evidence

Inventory JSON:

```text
reports/inventory/linux_rtx.json
```

Commands run:

```bash
physical-ai-lab collect-inventory --target linux-rtx --output reports/inventory/linux_rtx.json
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 --version
hostnamectl
git rev-parse HEAD
bash scripts/linux_rtx/render_factory_cell.sh
```

## Scene

- USD scene path: `isaac/usd/factory_cell_v0.usda`
- Robot assets: placeholder `YahboomMobileBaseProxy` and `SynriaArmProxy` geometry
- Sensors: `OverviewCamera` and `RobotCamera` camera viewpoints
- Workcell objects: floor, inspection table, table pedestal, pick/place bins, inspection parts, safety zone, and arm reach envelope marker
- Lighting: distant key light and dome ambient light
- Physics enabled: `PhysicsScene` with Earth gravity metadata

## Validation

- Scene opens: USD stage validated with Isaac Sim bundled OpenUSD libraries; default prim `/World`, 32 traversable prims
- Camera renders: Isaac Sim render attempted with `bash scripts/linux_rtx/render_factory_cell.sh`; Isaac Sim crashes during RTX renderer startup before capture. Blender fallback render succeeded with `bash scripts/linux_rtx/render_factory_cell_blender.sh`.
- Robot spawns: placeholder mobile base and arm prims are present in the USD stage
- Collision geometry: safety zone and arm reach envelope marker are present; rigid-body collision physics not authored yet
- Screenshot path: Isaac target `isaac/outputs/factory_cell_v0.png` not captured because Isaac Sim exited with segmentation fault; Blender fallback captured `isaac/outputs/factory_cell_v0_blender.png`
- Known issues: Isaac Sim is installed outside PATH at `/home/oedeh/isaacsim`; Isaac Lab is not present; CUDA toolkit `nvcc` is not on PATH; ROS 2 is not on PATH; Isaac Sim RTX startup reports duplicate NVIDIA Vulkan ICD entries for the RTX 5090 and crashes in `librtx.scenedb.plugin.so` even with `active_gpu=0`, `multi_gpu=False`, and `max_gpu_count=1`
- Render logs: `/home/oedeh/isaacsim/kit/logs/Kit/Isaac-Sim Python/5.1/kit_20260514_022050.log`, `/home/oedeh/isaacsim/kit/logs/Kit/Isaac-Sim Python/5.1/kit_20260514_022131.log`
- Alternative stack: Blender 4.0.2, ROS 2 Jazzy desktop, ROS-Gazebo integration, MoveIt 2, ROS 2 control, ROS 2 controllers, joint state publisher GUI, and colcon are installed. ROS commands require `source /opt/ros/jazzy/setup.bash` in new shells.

## Synthetic Data

- Camera viewpoints: not configured yet
- Object classes: not configured yet
- Sample count: 0
- Output path: not created
- Dataset size: 0

## Robot Training

- Isaac Lab task: not created yet
- Observation space: not defined yet
- Action space: not defined yet
- Reward: not defined yet
- Training steps: 0
- Evaluation result: not run

## Business Relevance

The current Linux RTX workstation is suitable for the OpenUSD track: the RTX 5090 and Isaac Sim 5.1.0 install are present, and the repo now contains a first OpenUSD factory-cell scene. This v0 digital twin starts reducing real robot risk by making the workcell layout, camera viewpoints, safety zone, mobile base proxy, and arm proxy explicit before real Yahboom/Synria hardware motion. Isaac Sim rendering is parked as a platform issue so scene and robot asset work can continue through OpenUSD-first and ROS/Gazebo-compatible paths.

## Parked Platform Issue

- Isaac Sim 5.1.0 reaches startup on this workstation, but crashes in `librtx.scenedb.plugin.so` during RTX renderer initialization on driver `595.58.03`.
- The crash reproduces after `./isaac-sim.sh --reset-user`, so it is not being treated as a user-config problem.
- Driver rollback to the 580 branch is intentionally not in scope for this workstream.
- Keep the smoke script and logs as evidence, but do not block scene modeling, asset conversion, or ROS simulation planning on Isaac Sim screenshot capture.

## Alternative Path

Use the repo-local OpenUSD scene as the source of truth and validate it outside Isaac Sim first:

- inspect the USD stage with bundled or standalone OpenUSD tooling
- render portfolio screenshots through Blender USD import if Isaac RTX remains blocked
- build robot description and control simulation in ROS 2 + Gazebo Harmonic for mobile-base, sensors, and Nav2 workflows
- evaluate MuJoCo only for focused arm/control experiments where USD import is sufficient and ROS integration is not the main requirement

Detailed comparison: `docs/SIMULATION_ALTERNATIVES.md`

## Next Actions

- [x] Add a repo-local OpenUSD factory-cell scene under `isaac/usd/`
- [x] Add an Isaac Sim smoke script that opens the scene and attempts to render one camera frame
- [x] Add an alternate screenshot path using Blender USD import
- [x] Install or point `BLENDER` to a Blender executable, then capture `isaac/outputs/factory_cell_v0_blender.png`
- [x] Add a Gazebo Harmonic track for ROS 2 mobile-base and sensor simulation
- [x] Install/source ROS 2 and Gazebo Harmonic, then run `ros2 launch physical_ai_gazebo factory_cell.launch.py`
- [x] Add approximate Synria 6DOF arm ROS 2 description, ROS 2 control, and MoveIt 2 scaffolds
- [x] Validate Synria mock control and MoveIt planning launch smoke tests
- [ ] Replace placeholder Yahboom robo car and Synria arm geometry with asset references or converted USD assets
