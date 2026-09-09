# Yahboom ROSMASTER M3 Pro Assets

The vendor CAD-derived USDs were removed on 2026-09-07. Conversion tooling,
configuration, metadata and first-party approximate models remain.
[Acquisition and provenance](../../../../docs/VENDOR_INTEGRATION_MAP.md).

Correction to retained historical `metadata.json`: this directory's
`rosmaster_m3pro.urdf` and `rosmaster_m3pro/` USD layers are conversions of
the first-party primitive xacro, not the vendor M3Pro SolidWorks model.
The source xacros are retained under `source_urdf/`.

The current ROS wrapper in `ros2_ws/src/rosmaster_m3pro_description/`
instead includes the separately obtained vendor URDF. It is not the source of
the older primitive geometry retained here. No current vendor-URDF USD
reconversion is claimed. The CAD script requires a separately obtained
`ROSMASTER-M3Pro.x_t` and local Isaac paths.
