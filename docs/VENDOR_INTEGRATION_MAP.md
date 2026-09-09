# Vendor Integration Map

Updated: 2026-09-07.

## Removal Record

**Vendor robot files are no longer redistributed in the current tree.** On
2026-09-07, Task A removed both Yahboom ROS packages, Synria vendor URDF copies
and meshes, vendor-derived USD conversions, Yahboom CAD conversions and flattened
Robotiq geometry. See the [complete per-file provenance list](VENDOR_ASSET_REMOVAL.md).

Reason: the recorded Synria MIT email / GPL-3 package-classifier discrepancy
remains unresolved, and Yahboom redistribution terms are absent. The earlier
email-based MIT and non-commercial/portfolio redistribution posture is withdrawn.
First-party conversion scripts, ROS wrappers, approximate models, reference-only
patch/scene layers, reports, measurements and ledgers remain.

**This removes files from the current tree only. Git history still contains
vendor material from 2026-05-26 onward. Making the repository public exposes it.**
No history rewrite was performed. [NOTICE](../NOTICE) preserves attribution and
the prior licensing record.

## Sources and Local Destinations

| Asset | Vendor source / acquisition | Local destination |
| --- | --- | --- |
| Synria Alicia-D v5.6 50 mm gripper URDF and meshes | [Synria-Robot-Descriptions](https://github.com/Synria-Robotics/Synria-Robot-Descriptions), `synriard/urdf/Alicia_D_v5_6/` and `synriard/meshes/Alicia_D_v5_6/follower_standard/` | Ignored Isaac URDF/mesh paths and `ros2_ws/src/synria_arm_description/urdf/` inputs |
| Yahboom description package | [ROSMASTER-M3PRO](https://github.com/YahboomTechnology/ROSMASTER-M3PRO), [study page](https://www.yahboom.net/study/ROSMASTER-M3PRO) Download -> Code-Firmware, vendor support or onboard filesystem; archive path `yahboomcar_ws/src/yahboom_M3Pro_description/` | Ignored `ros2_ws/src/yahboom_M3Pro_description/` |
| Yahboom MoveIt package | Same archive, `yahboomcar_ws/src/M3Pro_config/` | Ignored `ros2_ws/src/M3Pro_config/` |
| Yahboom Parasolid CAD | Request `ROSMASTER-M3Pro.x_t` from Yahboom product support; no direct URL/version recorded | Ignored CAD input and CAD-derived USDs under `isaac/usd/robots/yahboom-rosmaster-m3-pro/` |
| Robotiq 2F-85 | Configured NVIDIA Isaac asset root, `Isaac/Robots/Robotiq/2F-85/configuration/Robotiq_2F_85_config.usd`; no pinned standalone URL/revision recorded | Local-only flattened geometry; first-party assembly scripts remain |
| Game-table scene inputs | External Alicia-D-ROS2 checkout, `scripts/isaac_sim/scenes/*table_norobot.usda`; request files/revision from Synria if absent | External checkout; retained mirrored layers reference these files |

Historical distribution corrections remain relevant: the 2026-05-27 Yahboom
email said the M3 Pro was not on GitHub and supplied only URDF. The integration
audit found the public repository and a fuller ROS workspace in Code-Firmware,
including calibration, driver scaffolds, demos, navigation, voice and third-party
packages. Those packages were not imported here. A public repository link does
not establish redistribution terms.

## Setup

The Python setup script stages user-obtained sources; it does not download or
execute vendor code implicitly. Review the vendor's terms and choose a source
revision/archive before running it. Original upstream commit/archive checksums
were not recorded, so a new acquisition is not claimed to reproduce old results.

```bash
python3 scripts/fetch_vendor_assets.py --help

git clone https://github.com/Synria-Robotics/Synria-Robot-Descriptions /tmp/synria
# Select a vendor revision, then use its actual identifier below.
python3 scripts/fetch_vendor_assets.py synria \
  --source /tmp/synria/synriard --source-id <chosen-commit>

# Obtain and extract Code-Firmware yourself, or use your onboard workspace.
python3 scripts/fetch_vendor_assets.py yahboom \
  --source <extracted-yahboomcar_ws>/src --source-id <archive-or-release-id>
```

Replace angle-bracket placeholders before running. Missing source files, incompatible
model names, missing meshes and existing destinations fail before installation.
`--check` validates without writing. Input and installed hashes are recorded
under ignored `.vendor-assets/`; source identifiers are explicitly user-supplied.

The Synria path transformation and existing first-party joint dynamics are applied
to locally obtained XML. Yahboom's two mesh package-name typos are fixed, preserving
the vendor's `arm4_Joiint` joint name. No vendor geometry is embedded in the script.

For the current Synria v2 simulation configuration, use the retained Isaac Sim 5.1
importer in a local staging directory, then copy its generated configuration:

```bash
"$ISAAC_PYTHON" isaac/scripts/import_synria_urdf_51.py \
  --out .vendor-assets/synria-import/synria_6dof_arm.usd
cp -R .vendor-assets/synria-import/configuration isaac/usd/robots/synria_6dof_arm_v2/
```

The first-party v2 root and patch layers remain. Recheck their composition and
contact behavior after a new import; historical USD byte/physics reproduction
has not been established. CAD, Robotiq and game-table acquisition instructions
are available through the script's `cad`, `robotiq` and `scenes` commands.
Those commands deliberately exit with failure because acquisition is manual.

## Ownership and Runtime Mapping

| Area | First-party material retained | External dependency / status |
| --- | --- | --- |
| Yahboom ROS | `rosmaster_m3pro_*` wrappers, launch and mock-control configuration | Description and MoveIt vendor packages required for corresponding launches |
| Synria ROS | `synria_arm_description` xacro wrapper, camera macro, launch; `synria_arm_gazebo` and `synria_arm_moveit_config` | Current wrapper includes the vendor URDF; it is no longer an approximate-only model |
| Isaac | Import/conversion/diagnostic scripts, first-party patch layers and scenes | Local vendor geometry and converted configuration required |
| Hardware | `robots/lerobot_robot_rosmaster_m3pro/`, local ROS/ZMQ host/client and recording code | Onboard vendor micro-ROS runtime and message packages are external |
| Evidence | `reports/` and experiment ledgers | Retained records, not renewed validation after asset removal |

The old `rosmaster_m3pro` primitive conversions and Synria
`source_urdf/` approximation remain first-party. Some old Synria
`payloads/Physics/` layers describe that approximate model, not Alicia-D;
they are not a working substitute for vendor physics. See the manifest's separate
ambiguity and retention notes before using those assets.

## Limits and Deployment Gates

Recorded Synria vendor guidance says URDF soft limits sit below firmware limits;
check live SDK/firmware constraints before physical commands. The vendor describes
URDF inertia as CAD-derived; that provenance does not establish real-arm dynamics.
Yahboom limits and control behavior require device validation. The documented base
command timeout failure and first-party deadman are recorded in
[the Yahboom bring-up notes](notes/yahboom_strategy_2026-08-20.md).

Synria hardware drivers remain outside the default launch path. ROS mock control,
successful conversion and simulation scores do not establish safe real-arm
operation. Current hardware measurements are under `reports/thor_trt_benchmark/`
and `reports/jetson/yahboom_day_one/`; sustained validation remains incomplete.
