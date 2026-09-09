# Stack watchlist — tracked, not yet adopted

Additions require a measured gap they close; each entry names its
adoption trigger.

| item | what | trigger to adopt |
|---|---|---|
| [Newton](https://github.com/newton-physics/newton) | GPU physics engine (Disney/DeepMind/NVIDIA, Warp + MuJoCo-Warp, differentiable, OpenUSD; Linux Foundation) — announced future Isaac Lab backend | When Isaac Lab ships Newton as stable default: migrate with the framework, re-validate contact-sensitive numbers (grasp/release) before trusting old baselines. Differentiable-sim also = alternative lever for dock-skill refinement (see WAM future-work triggers). |
| Isaac Lab Mimic (MimicGen) | demo multiplication from few demonstrations | Next corpus-scaling need on Ludo-token task (P3) |
| Isaac ROS + FoundationPose | local 6-DoF object pose on Thor at camera rate | P4 real-hardware bring-up (board/token state) |
| cuMotion (Isaac Manipulator) | collision-aware GPU motion planning | P4: physical arm over physical tokens (safety) |
| Cosmos Transfer | world-model visual augmentation for sim-to-real | P4: before first real-camera policy runs |
| ER-2 streaming endpoint | persistent Live-API session for continuous watching | If G3+ sidecar cadence outgrows request/response |

## GR00T 1.7 migration (DECIDED 2026-08-19, in progress)
Chosen: migrate to GR00T-N1.7-3B via LeRobot-native fine-tuning for
the Ludo policy. Alternatives: stay on N1.5 (rejected — deprecated,
unsupported since July 2026); wait for N2 (rejected — no date).
Deciding constraint: training a NEW policy on a deprecated model
wastes the corpus investment. Would reverse if: 1.7 fine-tune on our
LeRobot v2.1 corpora underperforms the E48-class N1.5 baselines at
matched steps. Consequences: Thor TRT pipeline rebuild required
(Cosmos-Reason2/Qwen3-VL backbone); fresh venv (~/.venv/lerobot17,
real HF LeRobot from git — PyPI 'lerobot' is a squatter package);
weights at ~/models/gr00t_n17_base.
STATUS 2026-08-19: migration groundwork DONE — weights at
~/models/gr00t_n17_base (6.5G); ~/.venv/lerobot17 has real LeRobot
0.6.2 with GrootPolicy importable. FOOTGUN: our repo's local
lerobot/ package (Synria teleop data contracts) SHADOWS the library
when cwd is the repo — run 1.7 training/eval from outside the repo
or set PYTHONPATH order explicitly.

BLOCKER 2026-08-19: GR00T 1.7 fine-tune reaches the training loop but
the backbone processor loads from GATED nvidia/Cosmos-Reason2-2B —
requires the operator's HF account to accept the license (token
verified 401). Working command (post-access): lerobot-train
--policy.type=groot --policy.pretrained_path=~/models/gr00t_n17_base
--peft.method_type=LORA --peft.r=16 --peft.target_modules=all-linear
--policy.push_to_hub=false + v3.0 dataset at
reports/training/ludo_corpus01_lerobot (run OUTSIDE the repo).
Venv repair ledger: pip 'lerobot[dataset]' pulls the PyPI SQUATTER
(0.1.0) and clobbers the git install — pin extras to the git URL;
dependency churn left a stale libnccl (missing symbol despite correct
pip metadata) — force-reinstall fixed; old diffusers needed upgrade;
v2.1 corpora need gen_episodes_stats_v21.py then the v21->v30
migrator (local conversion succeeds; ignore its hub-push failure).
RESOLVED 2026-08-19: the "gated repo" blocker was MISDIAGNOSED. Root
cause: ~/.bashrc:124 exported HF_TOKEN=your_huggingface_token_here —
a PLACEHOLDER (from a recent profile edit) that OVERRODE the valid
token file (~/.cache/huggingface/token, account oedeh). With the env
var removed, the account already had Cosmos-Reason2-2B access (the
401 became a benign 404 probe). No license action was ever needed.
Chain to first training step also required: nvidia-cuda-nvrtc
force-reinstall (stale libnvrtc-builtins.so.13.0 after dependency
churn) + LD_LIBRARY_PATH to the venv's cu13 lib dir + dataset
--tolerance_s=0.001 (30fps float rounding in our mp4 timestamps
exceeded the 1e-4 default by exactly 1e-4). GR00T-1.7 LoRA fine-tune
(r=16, all-linear) now training: 20k steps, batch 8, ~2.3 steps/s,
24.6GB VRAM, loss 1.101->1.082 by step 700.
LESSON (goes in the E1 post family): an invalid credential in the
ENVIRONMENT silently outranks a valid one on disk — same failure
shape as the silent CPU fallback; assert the identity you think you
are using (whoami) before diagnosing permissions.
2026-08-20 status: ludo_groot17_v1 trained (20k steps, LoRA r=16,
loss 1.10->converged) and SERVED via serve_groot17.py (LoRA merge +
checkpoint processor pipelines; 40-step chunks at 76-146ms warm on the
5090). Closed-loop eval lane added: ludo_turn_executor.py
--policy-port 5591 (same harness + scoring rule as the 97.2% scripted
expert; writes the same turns.jsonl/session_summary.json; funnel
grasped/lifted/released/ok). Chain: isaac/scripts/run_groot17_eval.sh
(serve -> eval -> stats -> corpus batch-2 top-up into ludo_corpus02b;
ludo_corpus02 stopped at 23/150 when the GPU went to training).
BLOCKER 2026-08-20 10:24 (host, not stack): a Software-Updater
(aptdaemon) run upgraded the NVIDIA stack 580.173.02 -> 580.178.04 and
FAILED mid-way: dpkg refused libnvidia-compute-580 (file conflict
/usr/share/nvidia/files.d/sandboxutils-filelist.json with the OLD
libnvidia-common-580). nvidia-driver-580-open + libnvidia-gl-580 left
unpacked/unconfigured (iU), libcuda.so.1 still 580.173.02. The reboot
at ~11:02 loaded the NEW 580.178.04 kernel module (dkms) under the OLD
userland -> CUDA error 803 "unsupported display driver / cuda driver
combination"; nvidia-smi missing; Isaac Sim "No device could be
created". Fix (needs sudo; all .debs already in /var/cache/apt/archives,
the conflicting file is gone now that libnvidia-common-580 is 580.178.04,
no reboot needed): sudo dpkg --configure -a && sudo apt-get install -f
LESSON: mixed repos (NVIDIA cuda repo pin 600 over Ubuntu restricted
500) + GUI auto-upgrades = a driver stack that can half-apply under a
running campaign. Hold the driver packages (apt-mark hold
nvidia-driver-580-open libnvidia-compute-580 ...) for the campaign's
duration; check /proc/driver/nvidia/version == libcuda.so.1 version
in the preflight of every launcher (run_groot17_eval.sh does the
torch.cuda probe).
CORPUS DESIGN GAP (found while wiring the eval): the place square is
trial-manager state (mgr.zone_xy) — nothing renders it — and corpus-01
used a CONSTANT instruction. The policy had no observable placement
goal; expect grasp/lift/release to be the meaningful funnel stages of
eval01 and placement ~chance. Batch 2+ must encode the target
(--instruction "... place it on track square {place}" and/or a
rendered marker) before the placement number means anything.
ORIN ONBOARDED 2026-08-20 (Thor playbook, repeated): Yahboom ROSMASTER
M3 Pro / Jetson Orin NX 8GB Super, hostname `jetsonorin` (renamed from
vendor `yahboom` 2026-08-20 12:45 CDT; ssh alias `jetsonorin` ->
jetson@192.168.1.243 in ~/.ssh/config on the 5090, `jetsonthor` likewise). L4T
R36.4.4 / JetPack 6.2, Ubuntu 22.04, Python 3.10, CUDA 12.6, TensorRT
10.7, torch 2.5.0 (nv24.08, CUDA ok), ROS 2 Humble, Docker (a Dify
stack is resident), NVMe 89% full (11 GB free — keep clones sparse).
Repo: pushed over the LAN (5090 sshd is inactive, so 5090 -> Orin push,
receive.denyCurrentBranch=updateInstead), sparse checkout excluding
reports/* except reports/inventory + reports/jetson; origin =
git@github.com SSH (Orin's key `yahboom-orin-nx` awaiting addition at
github.com/settings/keys; until then Orin commits flow Orin -> 5090
(`git remote orin`) -> GitHub). Vendor leftovers fixed/flagged: git
identity was jetson-nx@example.com (set); timezone was Asia/Shanghai
(set to America/Chicago 2026-08-20; the day-one artifacts predate it —
their timestamp_utc fields are correct, only the `timezone` label reads CST+0800).
GitHub SSH key `yahboom-orin-nx` added; main tracks origin/main directly.
ACCOUNT 2026-08-20 12:50 CDT: work moved to a new sudo user `oedeh`
(matches Thor) — NOT a rename of `jetson`: 6 vendor service/autostart
files hardcode /home/jetson and the ROS workspaces live there, so
`jetson` stays untouched for the Yahboom stack. `oedeh` carries the
same hardware groups (video, i2c, gpio, dialout, docker, jtop, ...),
the same SSH keys, repo at /home/oedeh/github/physical-ai-jetson-robotics
(sparse, tracks origin/main, .venv), and the 5090's `jetsonorin` alias
+ `orin` git remote point at oedeh@. The interim jetson-owned clone was
removed (disk is 90% full: ~10 GB free — keep checkouts sparse).
Artifacts: reports/inventory/yahboom_orin_nx.json (Orbbec DaBai DCW2,
2x V4L2), reports/jetson/yahboom_day_one/ (day_one_smoke.py: MAXN_SUPER,
fp16 2048 matmul cold 173ms / warm p50 1.77ms p95 2.76 n=200 ~9.7
TFLOPS, /dev/video0 640x480 60.0ms/frame = 16.7fps fixed, VDD_IN 5.4W
idle -> 11.6W peak, 62C, no throttling).
BUG FOUND on day one: scripts/jetson/collect_inventory.sh ran `python
-m physical_ai_lab.cli`, but cli.py had no __main__ guard — it exited 0
silently and wrote nothing (fixed 890b24a). Any prior inventory
"collected" through that script never existed; Thor's should be
re-collected with the fixed CLI.
