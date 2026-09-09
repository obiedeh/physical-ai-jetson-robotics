"""MuJoCo cross-check (G31): can the vendor finger meshes force-close a cup?

Minimal rig: the two Alicia-D 50 mm finger STLs mounted on prismatic
slides under a vertical lift rail; a 40 mm dia / 40 mm tall / 50 g cup
between them on the floor. Sequence: close fingers -> lift rail 10 cm ->
measure cup height. Cup rises with the rig => the ASSET can force-close
(PhysX-side problem). Cup stays => the mesh pads themselves cannot
generate closure on this cylinder.

Also runs a control with plain BOX pads (known-good geometry).
"""

import mujoco
import numpy as np

MESH_DIR = (
    "/tmp/claude-1000/-home-oedeh/759ac4a3-1de7-4848-9fee-3f53d87dcea0/"
    "scratchpad/vendor_descriptions/synriard/meshes/Alicia_D_v5_6/follower_standard"
)

XML_TMPL = """
<mujoco model="grip_test">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <mesh name="lf" file="{mesh_dir}/left_gripper_50mm.STL" />
    <mesh name="rf" file="{mesh_dir}/right_gripper_50mm.STL" />
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" friction="1.0 0.05 0.001"/>
    <body name="rail" pos="0 0 {rail_z}">
      <joint name="lift" type="slide" axis="0 0 1" range="-0.05 0.3"/>
      <geom name="palm" type="box" size="0.03 0.05 0.008" rgba="0.3 0.3 0.3 1"/>
      <body name="lfinger" pos="0 0.05 -0.045">
        <joint name="ljoint" type="slide" axis="0 -1 0" range="0 0.05"/>
        {lgeom}
      </body>
      <body name="rfinger" pos="0 -0.05 -0.045">
        <joint name="rjoint" type="slide" axis="0 1 0" range="0 0.05"/>
        {rgeom}
      </body>
    </body>
    <body name="cup" pos="0 0 0.021">
      <freejoint/>
      <geom name="cupg" type="cylinder" size="0.02 0.02" mass="0.05"
            friction="1.2 0.05 0.001" rgba="0.9 0.2 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="alift" joint="lift" kp="2000" kv="200" forcerange="-50 50"/>
    <position name="al" joint="ljoint" kp="400" kv="40" forcerange="-20 20"/>
    <position name="ar" joint="rjoint" kp="400" kv="40" forcerange="-20 20"/>
  </actuator>
</mujoco>
"""

MESH_GEOMS = (
    '<geom name="lg" type="mesh" mesh="lf" mass="0.026" friction="1.2 0.05 0.001" rgba="0.1 0.1 0.1 1"/>',
    '<geom name="rg" type="mesh" mesh="rf" mass="0.026" friction="1.2 0.05 0.001" rgba="0.1 0.1 0.1 1"/>',
)
BOX_GEOMS = (
    '<geom name="lg" type="box" size="0.01 0.005 0.03" friction="1.2 0.05 0.001" rgba="0.1 0.1 0.5 1"/>',
    '<geom name="rg" type="box" size="0.01 0.005 0.03" friction="1.2 0.05 0.001" rgba="0.1 0.1 0.5 1"/>',
)


def run(tag: str, lgeom: str, rgeom: str, rail_z: float = 0.075) -> None:
    xml = XML_TMPL.format(mesh_dir=MESH_DIR, lgeom=lgeom, rgeom=rgeom, rail_z=rail_z)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cup_qpos_z = model.jnt_qposadr[model.body("cup").jntadr[0]] + 2

    def step_to(lift, close, steps):
        data.ctrl[0] = lift
        data.ctrl[1] = close
        data.ctrl[2] = close
        for _ in range(steps):
            mujoco.mj_step(model, data)

    step_to(0.0, 0.0, 500)      # settle open
    z0 = float(data.qpos[cup_qpos_z])
    step_to(0.0, 0.045, 800)    # close: each finger travels toward the cup
    ncon_grip = sum(
        1 for i in range(data.ncon)
        if {model.geom(data.contact[i].geom1).name,
            model.geom(data.contact[i].geom2).name} & {"lg", "rg"}
        and "cupg" in {model.geom(data.contact[i].geom1).name,
                       model.geom(data.contact[i].geom2).name}
    )
    step_to(0.10, 0.045, 1500)  # lift 10 cm
    z1 = float(data.qpos[cup_qpos_z])
    print(f"[{tag}] finger-cup contacts at close: {ncon_grip}  "
          f"cup z: {z0:.3f} -> {z1:.3f}  "
          f"{'LIFTED' if z1 - z0 > 0.05 else 'NOT lifted'}")


run("BOX pads (control)", *BOX_GEOMS)
run("VENDOR mesh fingers", *MESH_GEOMS)
