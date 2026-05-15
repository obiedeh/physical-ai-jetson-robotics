# Isaac Robot Assets

This directory holds Isaac Sim robot assets and import inputs.

Each active robot folder follows the same structure:

- `<robot>/<robot>.urdf`: generated working URDF for Isaac import.
- `<robot>/source_urdf/`: xacro source copy used to regenerate the URDF.
- `<robot>/*.usd` or `<robot>/*.usda`: Isaac-generated USD outputs when available.

Active robot folders:

- `synria_6dof_arm/`
- `yahboom-rosmaster-m3-pro/`

Top-level `*.urdf` and `*.usda` files are convenience entrypoints for existing import and scene workflows. Canonical editable ROS 2 package sources stay under `ros2_ws/src/*_description/urdf/`.

Other folders:

- `meshes/`: shared mesh assets used by robot descriptions.
