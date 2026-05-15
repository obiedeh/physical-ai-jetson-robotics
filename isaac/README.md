# Isaac Sim, Isaac Lab, and OpenUSD

This track contains simulation and digital twin assets.

## Factory Cell v0

The first repo-local OpenUSD scene is:

```text
isaac/usd/factory_cell_v0.usda
```

It is a lightweight factory-cell proxy for the Physical AI Operations Stack for Mobile Manipulation. The scene includes:

- a floor, safety zone, work table, bins, and inspection parts
- Yahboom mobile-base and Synria arm proxy geometry
- collision/reach-zone markers
- overview and robot camera viewpoints
- basic lighting and physics metadata

Render a smoke-test screenshot on the Linux RTX workstation:

```bash
bash scripts/linux_rtx/render_factory_cell.sh
```

The default screenshot output is:

```text
isaac/outputs/factory_cell_v0.png
```

Override the Isaac Sim Python path if needed:

```bash
ISAAC_PYTHON=/path/to/isaacsim/python.sh bash scripts/linux_rtx/render_factory_cell.sh
```

Render the same scene through Blender when Isaac Sim RTX rendering is blocked:

```bash
bash scripts/linux_rtx/render_factory_cell_blender.sh
```

The default Blender output is:

```text
isaac/outputs/factory_cell_v0_blender.png
```

This path has been validated on the Linux RTX workstation. The installed Ubuntu Blender package does not expose native USD import, so the script falls back to rebuilding the v0 factory-cell layout directly in Blender for screenshot capture.

Override the Blender executable if needed:

```bash
BLENDER=/path/to/blender bash scripts/linux_rtx/render_factory_cell_blender.sh
```

## Isaac Sim 6.0 Setup (Linux RTX Workstation)

Isaac Sim 6.0.0 is installed via pip into a Python 3.12 virtual environment.

```bash
python3 -m venv ~/.venv/isaacsim6
source ~/.venv/isaacsim6/bin/activate
pip install \
  "isaacsim[extscache]==6.0.0.0" \
  "isaacsim-core==6.0.0.0" \
  "isaacsim-asset==6.0.0.0" \
  "isaacsim-cortex==6.0.0.0" \
  "isaacsim-gui==6.0.0.0" \
  "isaacsim-replicator==6.0.0.0" \
  "isaacsim-robot==6.0.0.0" \
  "isaacsim-robot-motion==6.0.0.0" \
  "isaacsim-sensor==6.0.0.0" \
  "isaacsim-storage==6.0.0.0" \
  "isaacsim-utils==6.0.0.0" \
  "isaacsim-test==6.0.0.0" \
  "isaacsim-example==6.0.0.0" \
  "isaacsim-rl==6.0.0.0" \
  "omniverse-kit==110.0.0.276876" \
  --extra-index-url https://pypi.nvidia.com
pip install "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
  --extra-index-url https://pypi.nvidia.com
```

Run the render script:

```bash
ISAAC_PYTHON=~/.venv/isaacsim6/bin/python bash scripts/linux_rtx/render_factory_cell.sh
```

### Platform Notes

Isaac Sim 5.1.0 crashes on RTX 5090 (Blackwell, sm_120) — its CUDA 12.0 stack has no compiled
shaders for that architecture. Isaac Sim 6.0.0 (Kit 110.0.0, CUDA 12.8/13.x driver) resolves this.

The gfxstream Vulkan ICD presents a duplicate GPU and is excluded via `VK_ICD_FILENAMES` in the
render script. The render script pins Vulkan to the NVIDIA ICD directly.

Full diagnosis report:

```text
reports/isaac/2026-05-14-linux-rtx-openusd-bringup.md
```

## Planned Work

- replace proxy robot geometry with Yahboom robocar and Synria arm USD assets
- add a Gazebo Harmonic track for ROS 2 mobile-base and sensor simulation
- Isaac Lab training environments
- synthetic data generation workflows
