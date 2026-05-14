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

## Current Platform Note

Isaac Sim 5.1.0 currently crashes during RTX renderer startup on the Linux RTX workstation with NVIDIA driver `595.58.03`. The crash is documented in:

```text
reports/isaac/2026-05-14-linux-rtx-openusd-bringup.md
```

The issue is parked for now. Alternate paths for screenshots and robot simulation are tracked in:

```text
docs/SIMULATION_ALTERNATIVES.md
```

## Planned Work

- replace proxy robot geometry with robo car and robotic arm USD assets
- add a Blender or OpenUSD-based screenshot path while Isaac RTX rendering is blocked
- add a Gazebo Harmonic track for ROS 2 mobile-base and sensor simulation
- Isaac Lab training environments
- synthetic data generation workflows
