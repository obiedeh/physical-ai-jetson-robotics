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

## Isaac Sim Crash Root Cause (Diagnosed 2026-05-14)

Two separate issues were found and investigated:

**Issue 1 — Duplicate Vulkan GPU enumeration (fixed)**

`mesa-vulkan-drivers` installs `gfxstream_vk_icd.json` alongside `nvidia_icd.json`. The
gfxstream ICD is an Android-emulation passthrough layer that re-presents the physical RTX 5090
as a second virtual Vulkan device with the same UUID and bus ID. Isaac Sim's crash report
showed three Vulkan devices (RTX 5090 × 2 + Intel ARL iGPU) instead of one.

Fix: set `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` before launching Isaac Sim.
This is now baked into `scripts/linux_rtx/render_factory_cell.sh`. After the fix, the crash
report correctly shows one GPU.

**Issue 2 — sm_120 (Blackwell) not supported by Isaac Sim 5.1.0 (root crash)**

Isaac Sim 5.1.0 ships Kit 107.3.3 with CUDA 12.0.107. The RTX 5090 uses Blackwell architecture
(GB202, CUDA compute capability `sm_120`). Blackwell support requires CUDA 12.5+. Isaac Sim's
bundled `librtx.scenedb.plugin.so` crashes in `carbOnPluginStartup` at the same vector
initialization site regardless of the Vulkan ICD count, because the RTX shader database has no
compiled kernels for sm_120. This is a closed-source binary; there is no configuration or
environment variable workaround.

**Fix: upgrade to Isaac Sim 6.0.0**, which ships Blackwell support:

```bash
# Install Isaac Sim 6.0 into a Python 3.10 venv
python3.10 -m venv ~/.venv/isaacsim6
source ~/.venv/isaacsim6/bin/activate
pip install isaacsim==6.0.0.0 \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://pypi.ngc.nvidia.com
```

After install, update `ISAAC_PYTHON` in the render script to point to the new venv's Python,
or set it in the shell before calling the script:

```bash
ISAAC_PYTHON=~/.venv/isaacsim6/bin/python \
  bash scripts/linux_rtx/render_factory_cell.sh
```

Both Vulkan ICD fixes apply to the 6.0 install as well and should be kept.

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
- [x] Diagnose Isaac Sim 5.1.0 crash: identified two root causes (gfxstream Vulkan duplicate + sm_120 not in 5.1 RTX stack)
- [x] Fix duplicate Vulkan ICD: VK_ICD_FILENAMES pin added to render_factory_cell.sh
- [ ] Install Isaac Sim 6.0.0 (Blackwell support) and re-run render smoke test
- [ ] Replace placeholder Yahboom robo car and Synria arm geometry with asset references or converted USD assets
