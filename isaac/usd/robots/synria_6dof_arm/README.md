# Synria Robot Assets

Vendor URDF, mesh and importer payloads were removed from this folder on
2026-09-07. [Fetch and conversion setup](../../../../docs/VENDOR_INTEGRATION_MAP.md).

Retained `source_urdf/` files describe the older first-party approximation;
`payloads/Physics/` and `payloads/robot.usda` likewise describe that older
model. They are not Alicia-D contact validation.

Correction to retained historical `metadata.json`: its
`convert_cad_to_usd.sh` command is not a Synria converter in this directory.
The current conversion tool is `isaac/scripts/import_synria_urdf_51.py`.
Vendor-derived outputs remain ignored; first-party v2 patch layers live in
the sibling `synria_6dof_arm_v2/` directory.
