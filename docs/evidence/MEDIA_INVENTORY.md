# Existing Media Assessment

Generated path inventory: [media_inventory.csv](media_inventory.csv).
The catalog enumerates 1,723 tracked, present files from reports, scene/prop
layers and diagrams using [inventory_evidence_media.py](../../scripts/inventory_evidence_media.py).
It does not claim every episode was watched. Ignored/untracked local media are
not part of this reproducible public-tree inventory.

| Collection | What it shows / origin | Presentable as-is? | Needed |
| --- | --- | --- | --- |
| [G3 demo](../../reports/g3_session_01/g3_demo.mp4) | First-party composed simulated wrist view, ER-2 fix age/latency and ACT phase; vendor geometry depicted | A sampled frame was inspected. Useful diagnostic, visually ambiguous as a standalone demonstration | Caption simulation, camera framing, age and sidecar; no physical-success claim |
| [Gate audit](../../reports/gate_audit/) | First-party gate input captures; vendor geometry depicted; filenames include step/episode/verdict | Two representative images inspected; source images are small, useful with explanation | Pair verdict with visible input and ledger; no object-success inference from an image alone |
| [Gripper clearance](../../reports/synria_gripper_clearance.png) | First-party analytical schematic using model dimensions | Inspected and legible; not performance evidence | Retain units and modeling assumptions; link amended grasp audit |
| Training/evaluation episode videos and imagery | First-party report artifacts; robot geometry often vendor-derived, provenance must be checked per run | Not individually reviewed or selected wholesale | Review episode and manifest, confirm simulation/physical scope and scoring rule, caption failures as well as successes |
| [Isaac scene layers](../../isaac/usd/scenes/) | First-party scene/reference definitions; some reference external Alicia-D-ROS2 game tables and removed vendor robot payloads | Source/layer structure inspected; not renderable from a clean clone without external inputs | Obtain dependencies locally; resolve source paths; label as geometry/context, not performance evidence |
| [Mermaid views](../diagrams/) | First-party system, deployment, runtime and data flow | Source reviewed and corrected in Task C; architecture context only | Mermaid-capable renderer; keep evidence/pending-work distinctions |

Scene/robot provenance and uncertain upstream revisions are listed in
[the removal manifest](../VENDOR_ASSET_REMOVAL.md). A first-party recording can
depict vendor geometry; that does not make the mesh first-party or establish
redistribution terms for the source geometry. Source media remain unchanged.
