
## Synria arm visual style
The Synria/Alicia-D robot arm must ALWAYS be black (near-black 0.03 RGB) in
every scene — training, GUI, viewer, demos. The color is authored in the
PROTOTYPE layer isaac/usd/robots/synria_6dof_arm_v2/configuration/
synria_6dof_arm_base.usd (displayColor + shader diffuse on all meshes).
NEVER de-instance prims to recolor (exploding 4096 env instances hung all
large-scale boots once already) — always paint the shared prototype layer.
