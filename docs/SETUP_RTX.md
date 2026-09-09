# RTX 5090 Environment

Updated 2026-09-07.

**Correction:** the previous guide labelled its environment Isaac Sim 5.1 but
installed `isaacsim==6.0.0`, specified Python 3.10, and cloned Isaac Lab v2.0.0.
That recipe did not describe the recorded simulation environment and is withdrawn.

## Recorded Environment

The repository records Isaac Sim 5.1 with Isaac Lab 2.3.2 (commit `be9877d`).
The [Ludo run provenance](../reports/ludo_soak_s10/provenance.json) records
Python 3.11.13, RTX 5090 and driver 580.173.02. Its CUDA field is `nvcc`
output, not a measurement of every installed library's CUDA runtime.

Use a separate Isaac environment, installed from NVIDIA's distribution with its
matching dependencies. This repository does not contain a validated fresh-machine
lockfile. Obtain required checkpoints and the external Isaac-GR00T checkout
separately. The [July environment notes](handoff-2026-07-27-1937-CDT.md)
record dependency incompatibilities encountered during the experiments.

## Verify Your Installation

Set `ISAAC_PYTHON` to your Isaac environment's Python. Existing wrappers often
default to `~/.venv/isaacsim5/bin/python`; some also require `ISAACSIM_ROOT`.

```bash
"$ISAAC_PYTHON" --version
"$ISAAC_PYTHON" -m pip show isaacsim isaaclab torch
nvidia-smi
```

These inspect the environment; they do not validate robot behavior.

## Vendor Inputs and USD Conversion

First follow [vendor setup](VENDOR_INTEGRATION_MAP.md#setup). Use the explicit
`isaac/scripts/import_synria_urdf_51.py` path there, stage the generated
configuration locally and retain the first-party v2 patch layers.

The legacy `scripts/linux_rtx/import_synria_urdf.sh` calls the older
`import_synria_urdf.py` converter, not the 5.1-specific importer. The legacy
`train_isaaclab_synria.sh` also prechecks the old USDA path. These wrappers
are not claimed as the current v2 asset regeneration path. Do not treat a
successful import as contact validation; recheck composition and contact behavior.

**Do not commit vendor inputs or derived robot USDs.** They remain ignored.

## Simulation Runs

Use the exact command and checkpoint recorded in the relevant run's
`provenance.json` after obtaining its dependencies. The Ludo executor's
default path uses the recorded PPO checkpoint; `--policy-port` selects a
separate served GR00T evaluation. Mirrored game-table scenes require an external
Alicia-D-ROS2 checkout. Historical checkpoint paths are not downloadable assets.

The retained scene builders and diagnostic scripts remain available, but a
fresh fetch/import has not been validated as reproducing every historical
scene, patch or score. [NOT_CLAIMED](../reports/NOT_CLAIMED.md) applies.
