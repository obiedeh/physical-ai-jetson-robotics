# First-party evidence pull — 2026-08-19

Superseded observations, 2026-09-07: this is a historical cross-repository audit.
Thor benchmark artifacts were recovered on 2026-08-20 under
[reports/thor_trt_benchmark](../../reports/thor_trt_benchmark/); 128 ms / 7.8 Hz
was a misattributed baseline, not TensorRT. Numeric parity remains unrecorded.
The root README is now written. Earlier missing-artifact/empty-README statements
below describe the audit date, not current status. Other repositories were not
re-audited in this task.
Generated on the Linux workstation (aimlstation) by searching all local repos, git history, artifacts, and ledgers.
Intended destination: `C:\Users\obinn\OneDrive\Linkedin Posts\01_System\first_party_evidence_pull.md` (no OneDrive mount on this machine — copy this file there).

Conventions: exact values quoted, never rounded. Every number carries conditions and data kind. ⚠ = NOT PUBLISHABLE, with reason.

## Repo visibility (verified via `gh repo view --json visibility`)

| Repo | Visibility |
|---|---|
| ai-ran-kpi-forecasting | **PUBLIC** |
| ai-phy-neural-receiver-benchmark | **PUBLIC** (history rewritten 2026-07-30; pre-rename history absent — README.md:250) |
| wireless-link-intelligence-system | **PUBLIC** |
| private-5g-edge-telemetry | **PUBLIC** (squashed evidence pack, 2 commits — README.md:229) |
| telecom-commissioning-copilot | PRIVATE |
| telecom-retention-intelligence-system | PRIVATE |
| jetson-edge-ai-security | PRIVATE |
| urban-edge-vision-analytics | PRIVATE |
| edge-traffic-sensor | PRIVATE |
| physical-ai-safety-observability | PRIVATE |
| physical-ai-jetson-robotics | PRIVATE |
| physical-ai-jetson-robotics-gemini-er-2 | PRIVATE |
| Alicia-D-Isaac-Teleop | PRIVATE (Alicia-D-ROS2 is a byte-identical second working copy of it; no separate GitHub repo) |
| obiedeh (hub), obiedeh.github.io | PUBLIC — **neither contains a single numeric performance claim** |

Readers can only open the four public wireless-cluster repos and the hub. Everything robotics is private.

---

# E1. Silent CPU fallback check

**The honest headline: the assert you want mostly does not exist. What exists is a line-cited catalog of how the check fails silently — which is the better post.**

The only hard device assert in ~16 repos:
`physical-ai-jetson-robotics/physical_ai_lab/rtx_training.py:28-30` (PRIVATE):
```python
if requested == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
```
Every Isaac script hard-codes `device="cuda"` with no assertion (eval_synria_sequence.py:129, eval_gr00t_system.py:154, ludo_turn_executor.py:158, +18 sites).

The only provider *resolution* recorded to an artifact:
`wireless-link-intelligence-system/edge/jetson_benchmark_template.py:65-75` (PUBLIC) — `_select_providers()` filters [TensorRT, CUDA, CPU] by `ort.get_available_providers()`, writes `"providers_used"` into the JSON (:132) and prints it (:174). It **records but never asserts** — a CPU-only run still writes a benchmark file, exit 0.

Documented silent-fallback traps (all PRIVATE, jetson-edge-ai-security unless noted):
- `deploy/thor/run_benchmark.py:101-116` — `CPUExecutionProvider` appended unconditionally; resolved provider never read back.
- `run_benchmark.py:297` — `"source_badge": "validated-thor-benchmark" if soc else "measured-cpu"` where `soc` = `os.getenv("JETSON_SOC")` or devicetree. **A CPU-only run on Jetson hardware is stamped `validated-thor-benchmark`; `JETSON_SOC=tegra-234` on an x86 box forges the badge.**
- `tests/test_thor_smoke.py:64,78,99` — `ort.InferenceSession(str(path))` with **no providers argument**: the "Thor smoke tests" run on CPU EP and pass. :128-132: the TRT test skips itself if engines were never built.
- `src/jetson_edge_ai_security/models/train.py:208-225` — a CLI subcommand named `cuda` that prints a stub notice and `raise typer.Exit(code=0)`. Wrapping scripts get green having run zero inference.
- `run_benchmark.py:77-94` — power capture runs `tegrastats` via `subprocess.run(..., timeout=0.5)`; tegrastats streams forever → `TimeoutExpired` → bare `except: pass` → power is `null` on every real run. **No power number can ever be produced by this script as written.**
- `ai-phy/scripts/export_onnx.py:124` (PUBLIC) — `InferenceSession(...)` with no providers; :149-157 — if onnxruntime is missing, writes `"parity_pass": None, "note": "onnxruntime not installed — parity not verified"` and **exits 0**.
- `physical-ai-jetson-robotics/edge_ai/onnx_runner.py:70-88` — silently falls back to a **mock** when onnxruntime absent (`is_mock` property exists; nothing asserts on it).
- Hard-pinned CPU, disclosed-in-code-only: `wireless/ai_link_estimation/snr_torch.py:178` `providers=["CPUExecutionProvider"]` — this produces the published INT8 latency numbers (see A6).

Tests that fail if the accelerator is missing: **NO EVIDENCE FOUND anywhere.** Nearest is the opposite — `wireless/tests/test_snr_torch.py:67-68` `pytest.importorskip("onnxruntime")`.

Genuine runtime device guard (capacity, not EP): `urban-edge-vision-analytics/api/routes/local_inference.py:510-580` (PRIVATE) — `nvidia-smi` free-VRAM check raising HTTP 409 with a self-labeling message ("Catalog estimate is {required_gb:.1f} GiB; currently free is {free_gb:.1f} GiB."); degrades open when nvidia-smi is absent.

---

# E2. Reproducibility as an argument-settling device

## make verify targets (what each regenerates)

| Repo | verify chain | Regenerates |
|---|---|---|
| ai-ran (PUBLIC) | `lint test run-sample model-comparison r1-dataflow-demo scenario-demo scenario-backhaul scenario-outage portal publish` (Makefile:104-112) | forecast pack, 3-model comparison, R1 dataflow bundle, 3 scenario packs, portal |
| ai-phy (PUBLIC) | `lint test bler-classical compare export dashboard` (Makefile:41-44) | ⚠ requires `models/neural_rx_best.pt`; **models/ does not exist in the repo** — verify cannot run from a fresh clone (disclosed README:191) |
| wireless (PUBLIC) | `lint test run-sim run-sim-ofdm run-sim-tdl channel-estimation generate-evidence train-evidence snr-torch dashboard validate-evidence` (Makefile:71) | see the reproducibility gap below |
| private-5g (PUBLIC) | `lint typecheck test scenarios business-cases portal dashboard benchmarks` + 17 `test -f` checks (Makefile:55-72) | everything incl. benchmarks |
| retention (PRIVATE) | `lint test report validate-artifacts` (Makefile:51) | **`bench-eval` is NOT in verify or CI** → `reports/evaluation_metrics.json` is stale-by-construction |
| jetson-edge-ai-security (PRIVATE) | `lint typecheck test demo-report static-reports validate-artifacts web-build` (Makefile:39) | demo report + 4 HTML reports; `validate-artifacts` is seven `test -s` non-emptiness checks only |
| urban (PRIVATE) | `lint typecheck test demo-report validate-artifacts` (Makefile:27) | mock inference report |
| safety-observability (PRIVATE) | `lint test` (Makefile:18) | **nothing** |
| edge-traffic-sensor (PRIVATE) | `lint type test` (Makefile:27) | nothing |
| physical-ai-jetson-robotics (PRIVATE) | **no Makefile.** One-command regeneration exists only for the Ludo stats: `python3 isaac/scripts/ludo_stats.py` |
| copilot (PRIVATE) | no Makefile; CI = lint+pytest only, **no evaluation run** |

## Bit-reproducible artifacts
- **private-5g `reports/benchmarks.json` `determinism` block — the strongest reproducibility artifact in the portfolio**: two independent re-runs at seed 42, SHA-256 byte-equality: manufacturing `1b0610d91c3f780908c7eb217a433b6cecc69492f7ad80afc42e4fd988fdedb8` (run A == run B), pharma `93cd5fd156b91e6e50213c2276c8694030fbd141275af1fad14c66610f46f953` (matches). PUBLIC.
- jetson-edge-ai-security `cli.py:344-348`: wall-clock deliberately pinned (`fixed_start = datetime(2026,1,1,...)`) + `sort_keys=True` (reporting.py:37,42) → byte-identical `make demo-report`. ⚠ Cost: `replay_report.md:13` prints `Duration seconds: 4.000000` — two hardcoded constants subtracted, presented in a Runtime Metrics section. NOT a measured duration.
- urban `examples/generate_mock_report.py:24,28,62-72` — seed 7, fixed timestamps, sort_keys. Deterministic.
- Robotics: the byte-identity property was used as an *instrument* twice — see topic 3.

## Seeds
Fixed: ai-phy `--seed 42` all scripts + `torch.manual_seed`; wireless `seed=7` sims / `seed=42` training / `random_state=11` estimators; private-5g seed 42 everywhere; retention `--n 500 --seed 42`; robotics argparse `--seed 123` (evals) / `--seed 11` (executor), landed in provenance.json.
NOT fixed / misleading:
- `ai-ran/src/ai_ran_kpi_forecasting/models.py:150-156` — public API accepts `random_state` that is **silently ignored** ("kept for API compat; seeding is model-internal", hard-coded 42). PUBLIC.
- ai-phy `sionna_link.py:110-111` — Sionna seed deliberately restored to `None`: pilots deterministic, **bits/channel non-reproducible run-to-run** → BER/BLER CSVs are not bit-reproducible.
- physical-ai-safety-observability and edge-traffic-sensor: zero seed occurrences in the entire repos.
- No `PYTHONHASHSEED` anywhere in any repo. No CUDA determinism flags anywhere.

## Test counts (raw `def test_` declarations)
robotics 684 (37 files) + game_core 7 + ludo_engine 9 · copilot 389 · jetson-edge 250 · urban 186 · private-5g 134 · wireless 48 (+11 parametrize) · retention 46 (matches README claim) · ai-ran 43 · ai-phy 37 · safety-obs 23 · edge-traffic 2.
Claim mismatches: ai-phy README:37 says "31/31 passing" vs 37 defined, 0 parametrize, 0 skips. wireless README self-contradicts: ":34 77 tests" vs ":222 15 pytest tests".

## CI matrices
Only edge-traffic-sensor tests on ARM64 (`ubuntu-latest + ubuntu-24.04-arm`, py 3.11/3.12) — and it has 2 tests, 1 commit. The repos that claim Jetson targets test on x86 only. No repo has GPU CI. ai-ran and private-5g run mypy; ai-ran CI regenerates all 9 artifact bundles + 30 `test -f` checks (the strongest CI gate).
⚠ edge-traffic-sensor's `contract-drift` CI job is a permanently-green no-op: `UPSTREAM_PIN:4` = `commit=REPLACE_DURING_PHASE_2` → both tests `pytest.skip` (test_contract_drift.py:60-62); the pin also points at `events.py`, a file that does not exist upstream (schema lives at `schemas.py`).
⚠ The public hub's CI runs **vendored snapshots**: `obiedeh/projects/jetson-edge-ai-security/tests/` has 8 test files vs 23 in the real repo — missing every ONNX-parity and Thor-smoke test. The hub's green badge does not validate the code it advertises.

## Regeneration catching a real discrepancy
The wireless case (documented in commit `bf5e860`): the SNR-convention fix was gated on regenerating and matching the committed curves — "AWGN smoke (2000 bits, seed=7): 2.5e-3 @ 0 dB … matches existing reports/ber_smoke_awgn.csv ✓ / AWGN full (1e6 bits, seed=7): 2.42e-3 / 1.83e-4 / 5.0e-6 / 0 → matches existing reports/ber_full_awgn.csv ✓". PUBLIC.
Robotics used byte-identity twice as a diagnostic instrument (both PRIVATE, SYNRIA_EXPERIMENT_LEDGER.md):
1. demo_06/demo_07 crashed **byte-identically at the same tick** → "identical numbers from a supposedly-changed dynamical system means the changed code never ran" (:2506-2507) → found the hidden episode-reset bug (topic 3).
2. Geometry refactor verified by "Phase 5 re-ran 12/12 with a byte-identical results file" (:877-878).
⚠ wireless `make verify` **overwrites committed artifacts with different numbers**: committed `link_estimation_metrics.json` has `"samples": 500`; Makefile:46 runs `--samples 120`; CI runs `--samples 60`. The committed artifact carries a Windows path (`"model_path": "models\\snr_estimator.joblib"`) — produced on a Windows box, outside both recipes. Same pattern in ai-ran (`dataflow_summary.md` contains `data\ran_kpi_sample.csv`).

---

# Topics 3 and 4. Evaluation-harness bugs, and pre-registration
All PRIVATE (physical-ai-jetson-robotics; ledger = docs/SYNRIA_EXPERIMENT_LEDGER.md). All simulated (Isaac Sim) unless marked.

## Harness-bug catalog (bugs in EVAL/BENCH code, not policy code)

1. **Hidden episode-clock reset corrupting every multi-command eval.** Symptom: sessions collapsed mid-command; demo_06 and demo_07 T2 crashed byte-identically at the same tick ("grasp at t~500, reset at global step ~5400 … cup materialized at (0.066,-0.129) on its side, later plowed 450mm", ledger:2495-2508). Root cause: env time_out at episode_length_s×6 = 5400 steps; session clock never resets between commands; the env reset silently, bypassing every executor-side guard. Fix: `env_cfg.episode_length_s = 36000.0` (ludo_turn_executor.py:164-170, commit cdac03a). Diagnosis spanned 4 debug sessions (demo_04→07) within one day.
2. **Screen-selection bias, measured on itself.** n=64 screen: grasp 36 events = **0.562/episode** (reports/logs/2026-08-06/e45_eval_ckpt36000.log). n=192 confirm: **0.406** (e45_conf_n192_ckpt36000.log). n=151 confirm: **0.304** (e45_n150_ck36_scratch.log). Same bias on the other paradigm: ACT 0.156→0.107, GR00T 0.484→0.351 (ledger:2250-2251). Ledger: "the 0.562 was a favorable n=64 draw; the honest headline number is 0.40-0.48."
3. **Gemini scorer temperature-0.0 hang, cost of 3 dead runs** (gemini-er-2 repo, GEMINI_LEDGER.md:86-90): "at temperature 0.0, a query whose target is undetectable can hang the ER-2 server in an unbounded reasoning loop (frame 13 reproducibly; same query at temperature 0.7 returns '[]' in 6.6 s)". Production rule: never temp 0.0, bounded timeouts, treat non-answers as data. Fix commits 6c28601, 903adff (an SSL read timeout separately killed a run at frame ~12).
4. **RESULT block silently discarded**: "the eval's RESULT block printed without flush=True … the first E23 eval exited 0 having silently thrown its numbers away. The progress lines flushed, so the run looked healthy right up to the missing table" (ledger:655-659).
5. **All-NaN summary under a bare except**: "the summary read runner.logger, an attribute RSL-RL does not have, under a bare `except: pass`" — real numbers recovered from TensorBoard (ledger:879-885).
6. **EIGHT bugs in one eval script (E32)** — ledger:1435-1442 — including "an INVERTED release (0.025 is closed, not open — every 'release' clamped the jaws shut). A reported 6/6 was false and was caught only by retracting the hand before judging." This is the harness-made-results-look-BETTER case.
7. **E27 takeover-controller: 3 of 4 low results were harness bugs** (exact-centre target vs 12mm tolerance against PLACEMENT_RADIUS_M=0.15; unlatched open oscillating below the 42mm detach threshold; position-only Jacobian rotating the wrist) — harness-made-results-look-WORSE case (ledger:908-913).
8. **The metric was the bug**: `align` z-tolerance 0.05 on a 50mm cup "cannot distinguish a body grasp from a rim grasp" → tightened to 0.02 → reads **0/267 = 0.000** vs legacy 32/267 = 0.120 (ledger:1059-1062, 1124-1131). OR-latched per-episode flags inflated a conditional 0.698 → recomputed exactly = 0.389 (ledger:1026-1038).
9. **E49 supervisor smoke bug**: drop-latch armed by the bare hand's ready pose — "16 false triggers at step 0" (e49_smoke.log: retries=16 at step 0, 89 in 500 steps) → fixed to latch only-while-holding + episode-boundary reset (e49_smoke2.log: retries=0 at step 0, 1 total). Commit ea67207.
10. **Language gate**: "with a paraphrased instruction the policy scores EXACTLY ZERO (64 ep); with the verbatim corpus instruction it performs. Eval clients must send the training string" (ledger:1895-1897).
11. **Protocol dominates reporting**: "ACT's harness-native 0.852 grasp becomes 0.188 under the funnel" (ledger:1941-1943).
12. Corpus/format tooling (2026-08-19, reports/logs/2026-08-19/): LeRobot v2.1→v3.0 migrator crashed on missing `meta/episodes_stats.jsonl` (ludo_v30_convert.log:32-49); the fix chain is recorded in STACK_WATCHLIST.md:41-46 (also: PyPI `lerobot` squatter clobbering the git install; a stale libnccl with a missing symbol despite correct pip metadata).
13. Earlier era (docs/SYNRIA_GR00T_LEDGER.md:985-1021): descend overshoot burying 60mm fingers 35mm underground; pose-pinning defeating contact impulses; a hover "held" criterion counting a fall as success — "two of this session's own harness bugs … produced two confident but WRONG 'the asset is broken' conclusions."

## Pre-registration instances (written BEFORE the run)
14 instances in the ledger; the ones with teeth:

| ID | Pre-committed bar | Outcome |
|---|---|---|
| E26 (:768-833) | "the decision rule is written here first so the conclusion cannot be bent to fit the data afterwards" — 3-run mean grasp > control by >spread AND >10% floor | PASS at **1.1006×** — "a 0.0004 smaller mean would have flipped it" |
| E29 (:1166-1315) | mean grasp_upright ≥ 1.25× control AND worst seed > control mean | **FAIL at 0.25×** (0.157→0.039). `place` rose 1.80× and was explicitly NOT claimed: "Reading a secondary metric as success after the primary fails is the error this ledger exists to prevent" |
| E39/E39b (:1703-1805) | retry place ≥1.5× no-retry; compound grasp prediction 0.69 if independent | E39 showed 1.86× on grasp (n=25/arm) → **E39b (n=40/arm) failed to replicate** (retry 0.350 vs no-retry 0.400) → "E39's 'mechanism CONFIRMED, 1.86x' was substantially sampling noise and is hereby **retracted**". Measured eval noise: ±0.15 on grasp at n=25-40 |
| E37 (:1661-1701) | final-40 grasp ≥0.45, place ≥0.10 | FAIL — the overfit cliff (see topic 7) |
| E43/E44/E45 (:2014-2167) | escalating "screen best > X" bars | PASS ×3 (0.188, 0.219, 0.562-screen/0.304-honest) — the scaling story |
| E49 (:2213-2266) | pre-declared metric set incl. **cycles per hour** ("retries consume wall-clock") + pre-logged calibration doubt: "before trusting S2 negative rates, visually audit a few wrist-camera gate frames" | The doubt cashed out exactly (see A1/A2: gate audit) |
| Ludo 95% bar (:2621-2682) | "The 95% bar needs the inner-edge place fix to hold" | Post-fix seeds 10-12: **35/36 = 97.2%** |
| E31 (:1351-1394) | tipped_rate ≤0.75× AND place not below control | ⚠ **No RESULT entry exists** — the one pre-registration without a recorded outcome |

What pre-registration prevented, concretely: E29's secondary-metric rescue (blocked by the written rule) and E39's false retry narrative (caught by the pre-planned replication, then retracted in writing).

---

# Topics 2 and 6. Retry supervisor, and the reliability split
All PRIVATE, simulated.

## Where it lives, triggers, bounds
- **S1 supervisor** `isaac/scripts/eval_gr00t_system.py`: triggers `pinched_empty = grasped & (width < 0.033) & (phase < 2) & ~lifted` sustained >12 steps (`PINCH_FAIL_STEPS = 12`, :49) OR `dropped = was_lifted & (phase == 0)` (:274-291). Recovery: lerp shoulder/elbow home at 0.06/step, gripper forced open, exit at >140 steps or (>40 steps & high) (:292-311). **No per-episode retry cap in this script** (an earlier E39-era version documented "Budget 3/episode").
- **Ludo executor** `isaac/scripts/ludo_turn_executor.py`: exactly **1 retry per command** (:1039-1043); approach fallback at 2500 idle steps (:528); carry timeout at 2000 attached-idle steps → polar drag (:621-629, "demo_24 T7: 636mm" fling documented inline); grasp-chain stuck-recycle at 900 (:653-656); RELEASE stuck-escape at 150; hard budget `CMD_TIMEOUT_STEPS = 9000` (:38).

## Before/after with trial counts
E49 rows (n=151 episode-equivalents, staged funnel, E48 ckpt-35k; logs reports/logs/2026-08-06/e48_n150_staged_S0.log, e49_S1.log, e49_S2.log):

| row | grasp | carry5s | retries | carry/hour | wall |
|---|---|---|---|---|---|
| S0 bare policy | 0.271 | 0.086 (13 events) | — | (not printed in S0 log) | — |
| S1 +supervisor | 0.318 | **0.192** (29 events) | 43 | **85.1** | 0.34 h |
| S2 +Gemini gates | 0.357 | 0.060 | 79 | 21.7 | 0.41 h |

⚠ The carry doubling rests on 13 vs 29 events at n=151 — suggestive, not paper-grade alone. ⚠ The ledger's "S1 85 carries/hour vs S0 ~39" — the S0 /hour figure is **derived, no artifact backs it** (S0 runner prints no wall clock). Flag if published.
Ludo (machine artifacts): 6-seed block first-try **59/73 = 80.8% ±4.6%** vs turn-level (with retry) **68/72 = 94.4%**; post-fix block first-try **32/36 = 88.9% ±5.2%** vs **35/36 = 97.2%** (reports/ludo_stats_summary.json + ledger:2644-2682).

## THE COST SIDE (computed from raw reports/ludo_soak_s{4..12}/turns.jsonl — 127 attempts; never published before)
- Retry attempts cost more: attempt-2 mean **5665.2 steps** vs attempt-1 mean **4685.8** (pooled) — **~21% more steps per retry**.
- Retries consumed **101,973 sim steps = 16.6% of all sim steps** across the 9 instrumented sessions (s4-s9 block alone: 20.2%).
- 5 of 18 retries failed anyway. Worst case: ludo_soak_s12 turn 5 — both attempts burned the full 8999-step budget: **17,998 steps for zero yield** (both classified `never-grasped (approach failed)`, reports/ludo_stats.csv).
- Wall-clock anchor: s10 session_summary.json `wall_clock_s 1659.5` for `sim_steps_total 51358` (~31 steps/s) → retry overhead ≈ 0.9 h of wall-clock across 9 sessions (derived).
- **Retrying made things worse — two documented cases:** (a) S2's grasp-confirm gate "rejected 72% of grasps" — `[sys] gates: {'queries': 53, 'negatives': 38, 'gate_retries': 38, 'no_verdict': 0}` (e49_S2.log) — carry5s 0.192→0.060, carries/hour 85.1→21.7. (b) E38+R: 40 episodes, **117 interventions**, grasp identical to no-retry (0.300), place 0.000 — "the retry gain is a MEASURE of failure correlation: v2's failures were substantially stochastic (retries compound), v3's are systematic (retries re-run the same mistake)" (ledger:1759-1769).
- The retracted retry win: E39 1.86× (n=25/arm) → E39b replication at n=40/arm reversed it; pooled effect "+0.06, within noise" (ledger:1787-1805). ⚠ NOT PUBLISHABLE as a retry effect.

## Policy-effort vs recovery-effort
Paradigm training budgets are quantified: RL 83.4 h / ACT ~86.9 h / GR00T 62.8 h GPU-hours (ledger:2325-2330). System-engineering (supervision/recovery) hours: "reported separately … they sit above all paradigms" (:2227-2229) — **but never actually quantified. NO EVIDENCE FOUND for a numeric policy-vs-recovery effort ratio.** The qualitative ledger verdict: "Every road in this campaign — RL scaling, imitation scaling, retries, the composite — terminates at that one number [upright-grasp ~0.33]" vs E49's "L2 supervision is a measured multiplier" (+123% carry).

---

# Topic 1. What doomed rollouts cost
PRIVATE, simulated. Computed from reports/ludo_soak_s{4..12}/turns.jsonl (127 attempts).

| | n | mean steps | median steps | ran to timeout |
|---|---|---|---|---|
| OK attempts | 104 | 4393.5 | 4161.5 | 12 (11.5%) |
| MISS attempts | 23 | **6773.9** | **8999** | **13 (56.5%)** |

**The post is real: the median failing attempt burns the entire 9000-step budget — ~2.2× the median success.** 56.5% of failures ran to the full timeout. (12 successes also carry the timeout flag — they placed the cup and then idled out the clock, a separate small inefficiency.)
Existing early-termination logic (with line numbers, ludo_turn_executor.py): approach fallback at 2500 idle steps (:528), S_YAW hand-back at 3000 (:563), carry timeout at 2000 (:624), grasp-chain stuck-recycle at 900 (:653-655), RELEASE escape at 150 (:689-691), GO_HOME break with 300-step escape. **There is no abort-on-hopelessness** — phase-level stalls are caught, but a doomed command still runs to 9000.
Counters that exist: `timed_out` flag per attempt (:938), `steps` per attempt, `wall_clock_s`/`sim_steps_total` per session. The share-of-time-after-outcome-decided is computable today from these (the numbers above ARE that computation at attempt granularity).
Smallest change for finer resolution: log the step index at which the last state-changing event occurred (attach, detach, hold-armed) per attempt — one line in the summary dict — giving "decided at step X, ran to step Y" per rollout.

---

# Topic 7. When to stop tuning and change the task
PRIVATE, simulated.

## Loss improving while closed-loop degrades (exact numbers)
- **E37**: "loss 0.394 → 0.275" while closed-loop collapsed to grasp 0.100 / place 0.000 from the 100k-step checkpoint's 0.325/0.100 — "task performance peaks at ~3-6 epochs over the corpus and collapses past ~10, while training loss improves monotonically throughout. Steps were the wrong axis" (ledger:1685-1701). Detected only by closed-loop eval of checkpoints, not by any training curve.
- **E34**: "Training converged cleanly (l1 0.75 → 0.24, no divergence)" while closed-loop grasp = 0.150 vs the RL baseline 0.777 (ledger:1521-1540).
- **E23**: "the reward terms said parity; the eval said 2-4x on every stage … Live monitoring on those terms produced two wrong calls in this run" (ledger:643-653) → the standing rule "do not judge runs on Episode_Reward terms."

## The fix was NOT the model (each with what it recovered)
- Feasible-spawn restriction (E6): near-bin grasp **0.837** vs far-bin **0.014** (n=123/141) — "half of all trials are geometrically unwinnable" → after restricting: grasp 0.782, "metrics now honest" (ledger:228-259).
- Kinematic attach-on-grasp (E8): removed accidental success channels — transport 0.413→0.077, place 0.148→0.000: "Old transport was largely involuntary piece-jostling … The task is now honest" (:318-327).
- Placement target added to the observation (E14): policy could not see the destination; place was 0.000 by construction (:400-410).
- Board repositioned into the reach annulus (commit 9523acd): near squares were inside minimum reach — unreachable squares had been scored as failures.
- origin_xy authority (commit 35ad319): "policy was homing on the stale task origin, not the teleported cup" — retired the OOD-freeze failure mode.
- Decoy in-distribution zone (commits 74d780a/a718444): the observation, not the policy, was out of distribution.
- Side-pick rule from the operator (commit c982ab2): grasp geometry from real-hardware experience; "policy detour retired."
- LEVEL-GLUE (commit d5fed7d): the pad squeeze pivoted the cup ~84° before the glue latched → upright-project at latch; demo_23 T7 carried at max 9° tilt.
- Cup-arrive gate (commit deaadf4): "the pad arriving within 20mm still left the offset cup 38mm off … the cup is what scores" → 12/12 session.
- Scene design over model blame (gemini-er-2, GEMINI_LEDGER.md:72-99): zone-pointing "failure" median **484.73 mm** was an invisible marker; with a visible magenta disc: 56/60 answered, median **2.8 mm**, p90 5.4 mm. "The v1 zone 'failure' was entirely the invisible marker."

---

# A1 and A2. Tiered stack and staleness

## Model and hardware names, exactly as written
- Cloud VLM (robotics): `gemini-robotics-er-2-preview` (URL string in eval_gr00t_system.py:70-71 and g2_report.json `model` field).
- Copilot tiers (telecom-commissioning-copilot/packages/shared/config.py:34-53, PRIVATE): local `ollama` `qwen3:1.7b` at `http://127.0.0.1:11434` (timeout 300 s); cloud `deepseek-v4-flash` (api.deepseek.com), `kimi-k2.6` (api.moonshot.ai/v1), `gemini-2.5-flash` (web-search tier). Local-tier hardware: NO EVIDENCE FOUND (never named).
- Robotics hardware: `"NVIDIA GeForce RTX 5090, 580.173.02"`, host `"aimlstation"`, `"Cuda compilation tools, release 13.2, V13.2.78"` (ludo_soak_s10/provenance.json); "Jetson AGX Thor 128 GB (train+serve)" (docs/paper/OUTLINE.md:23).

## Measured tier rates
- Copilot: **NO EVIDENCE FOUND** — no artifact records local-vs-cloud routing frequencies.
- Robotics eval tiers ARE measured (E49, n=151): L1 policy alone carry 0.086; +L2 supervisor 0.192 (43 retries); +L3 cloud gates 0.060 (53 queries, 38 negative). PRIVATE, simulated task / real cloud calls.
- safety-observability: the tier is a boolean `require_review` (severity HIGH/CRITICAL or confidence < 0.85 or runtime not nominal — rules/severity.py:4-7). ⚠ The one counter that would quantify autonomy (`human_review_required` rate) is **not instrumented** (telemetry/metrics.py counts events_total and critical_events_total only).

## Round-trip latency distribution to the cloud model (not just median)
- **G2 bench (REAL cloud measurements): p50 2.61 s, p90 5.01 s, n = 132 calls over 40 frames** — reports/perception_bench_g2/g2_report.json (`latency_p50_s`, `latency_p90_s`). Ledger verdict: "Static-phase serviceable; confirms L3 advisory role, physically incompatible with control-rate."
- G3 live sidecar (n=17, computed from reports/g3_session_01/sidecar.jsonl): p50 3.74 s, mean 3.91, min 1.76, max 7.31. ⚠ n=17 — NOT PUBLISHABLE as a distribution.
- **p95/p99: NO EVIDENCE FOUND anywhere.** No artifact records either percentile for any cloud call.
- ⚠ Copilot's only latency numbers (`mean_ms 0.4671954545454546, p95_ms 0.5987, max_ms 5.3351`, n=66) measure a **local in-process retrieval-planning span with no LLM call inside it** (packages/evaluation/runner.py:209-213). Do not present as model or network latency.

## Staleness: stamped, consumed, and the guard
- Robotics G3: every answer carries `t_ask`, `t_answer`, `lat`; the composer renders honest age per frame: `age = m["t"] - a["t_answer"]` → "age {age:+.1f}s" (g3_compose.py:59-62). E49 gates: `GATE_SETTLE = 15` steps before judging; at most one gate served per step; timeout/failure → `no_verdict` → **gate passes open** (eval_gr00t_system.py:22-23, 334-348). Measured `no_verdict: 0` of 53.
- Copilot has **three incompatible staleness definitions coexisting**: (1) citation-level `stale_doc_warn_months: 18` implemented as `18*31 = 558 days` (routes.py:1574-1578) — missing `effective_date` ⇒ silently not stale; (2) status-based `not in {"active","current"}` (provenance.py:160-161); (3) research coverage `stale_after_days = 3650` — **10 years, effectively never fires** (coverage.py:43,70-75; both committed artifacts show `stale_source_count: 0`).
- What the consumer does when too old (copilot): stale+current retrieval → `EvidenceConflict(conflict_type="revision_conflict", severity="medium", resolution="Prefer the newest applicable approved revision…")` → any conflict triggers the refusal composer: "The retrieved evidence conflicts, so I cannot give a field conclusion from it." confidence=0.3 (response_composer.py:955-977). Severity is always "medium" whether or not any current evidence exists.
- safety-observability: `Evidence.captured_at` is stamped but **nothing ever compares it to now** — no staleness guard exists (exhaustive grep). NO EVIDENCE FOUND.

## Measured stale-input-caused-wrong-action
NO EVIDENCE FOUND in any repo. The closest measured instance is spatial, not temporal: **the wrist-camera gate audit** (robotics ledger:2337-2351, 20 PNGs in reports/gate_audit/): "Every negative-verdict frame shows gripper + table with NO cup in the visible frustum … ER-2 answered the visible evidence correctly in every audited frame; the S2 carry collapse was caused by OUR viewport choice, not model judgment." The guard outcome: the S2 row stands as-run, labeled; a corrected S2' is future work. (This was the pre-registered calibration doubt from topic 4, cashed out.)

---

# A3 and A4. Contracts on boundaries, and decision provenance

## Schema inventory
| File | Governs | Enforced |
|---|---|---|
| ai-ran `schemas/kpm_input_v1.json` (PUBLIC) | KPM telemetry input | **NO** — jsonschema is in no requirements file, imported nowhere |
| ai-ran `schemas/a1_policy_v1.json` (PUBLIC) | A1 policy output | **Test-only, partial**: tests/test_a1_policy.py:111-118 checks the `required` key list only — the enums, pattern, `additionalProperties: false`, and min/max are never checked by anything |
| private-5g `private5g_pipeline/schema.py` (PUBLIC) | telemetry columns/ranges/enums | **YES, runtime, fail-soft** — `quarantine_invalid_rows` returns (valid, rejected+reasons), never raises |
| copilot pydantic models (workflows/schema.py, shared/models.py, rag/manifest.py) (PRIVATE) | API + corpus + workflow | YES — pydantic on every request |
| safety-observability `events/schemas.py` (PRIVATE) | events + evidence | YES (pydantic) |
| ai-phy, wireless | — | NO SCHEMA FILES |

## Schema catching a real malformed input
One documented case, with a caveat that matters: ai-ran commit `28f6933` — "data.py: fix 'technology' value '5G NSA' → '5G_NR' to match kpm_input_v1.json enum". The generator violated the schema enum — but **since no runtime validator exists, a code review caught it, not the schema**. The fix added a hand-written enum test. State it that way or a reader will find the gap.
private-5g's runtime path does catch malformed rows continuously: committed benchmark shows `rejected_rows: 432` of `total_rows: 8640` at 5% injected corruption, reasons strings like `"invalid timestamp"`, `"{col}: out of range [lower, upper]"` (benchmarks.json `quarantine` block). Synthetic corruption, real enforcement.

## forecast_basis — full field list (producer a1_policy.py:123-130; schema a1_policy_v1.json:50-80)
`model_name`, `target_kpi`, `predicted_peak`, `predicted_peak_timestamp`, `threshold_pct`, `model_metrics_ref` — required: the first three only. Committed instance (reports/r1_dataflow_demo/a1_policy_candidate.json): model_name `ridge_linear`, predicted_peak `56.306060143747175`, timestamp `2024-01-03T04:00:00Z`, threshold 80.0, ref `reports/forecast_examples/latest/metrics.json`.
⚠ `model_metrics_ref` is a **hard-coded string literal**, not derived from the run — the committed policy in `r1_dataflow_demo/` points at `forecast_examples/latest/`. No model version, no data hash, no run id, no training timestamp in the block.
Robotics equivalents (richer): provenance.json (15 fields incl. git SHA + `git_dirty`) and the per-attempt records (14 fields incl. `lane`, `hold_armed`, `closest_approach_mm`) — see topic 9/11. private-5g's `operational_decision_summary.json` carries `decision_basis: "seeded simulation + generated plots + budget sensitivity + multi-seed stability"` and `bottleneck_attribution.json` records the **rejected** hypothesis ("Radio / private 5G layer … Do not assume radio-capacity spend is the first fix").

## The no-action path — exact emissions
- ai-ran (PUBLIC), a1_policy.py:8-11 + :93-108: below threshold → `action: "no_action"`, rationale "Forecast prb_dl_util on CELL_001 peaks at 56.31 (model=ridge_linear, threshold=80.0). No policy action required." — "the candidate is still emitted so the policy plane has a **paper trail of 'we looked, nothing to do'**". Schema: "no_action = forecast did not cross thresholds (policy candidate emitted for audit only)." Both branches tested (test_a1_policy.py:76-88).
- copilot (PRIVATE), three refusal payloads (response_composer.py:907-1012): live-action refusal (confidence 0.65/0.35), evidence-conflict refusal (0.3/0.1), no-acceptance-criteria refusal (0.45/0.2) — each with structured sections and `missing_evidence`. ⚠ Confidence values are hand-assigned constants, not calibrated.
- safety-observability (PRIVATE): the healthy-negative channel — `"No Person Detected"` posted **every frame regardless of outcome** (edge/worker.py:105-110, 189-198), making silence and healthy-negative distinguishable. Tested. No measured rate of the three states.

---

# A5. Digital twin synchronisation

**NO EVIDENCE FOUND — write no A5 post from current repos.**
Exhaustive search across all repos: the only "twin" hit is matplotlib's `ax1.twinx()` (private-5g pharma.py:374). The robotics Isaac Sim environment is a training/eval simulator, not a synchronized twin — nothing syncs from physical state (the teleop bridge repo exists but its last commit is 2026-07-30, writes default off, and no sync-rate or divergence artifact exists). The nearest honest artifact is private-5g's explicit non-twin boundary: `"boundary": "Planning estimate from a simulated GPU+latency profile. Not a measurement of any specific factory or vendor edge platform."` (scenario_metrics.json:2).

---

# A6. Boundary verification

## ONNX parity — implemented, with artifacts
**ai-phy (PUBLIC)** — reports/onnx_parity_test.json, complete: `max_abs_diff 1.3828277587890625e-05`, `mean_abs_diff 2.4984924493764993e-06`, `parity_pass true`, `tolerance 0.0001`, opset 17, input [4,1,1,14,76]. Margin 7.2×. Conditions: PyTorch fp32 CPU vs ORT default provider; **input = one batch of 4 `torch.randn` frames (synthetic noise, unlabeled as such in the artifact)**; 7,296 LLR values. Notable engineering detail worth quoting: ONNX has no complex64, so the export wraps the model to take two real channels (`export_onnx.py:87-97`) — the exported graph is a wrapper, and parity is measured through it. ⚠ Two provenance defects: absolute local path embedded; references `models/neural_rx.onnx` which is not in the repo.
**jetson-edge-ai-security (PRIVATE)** — ONNX↔sklearn parity, multi-seed, asymmetric tolerances: detector `< 1e-4` (test_reference_detector.py:190,204), forecaster `atol=1e-3` seeds [0,7,42] (test_reference_forecaster.py:205-226), plus mock round-trips at 1e-4/1e-5. No derivation for the 10× looser forecaster bound; **the tests assert bounds but never record the observed delta** — no artifact answers "how far apart were they actually?"

## Quantization
**wireless INT8 (PUBLIC)** — reports/snr_quantization_comparison.json: sklearn MAE **0.07463915863831914 dB** / torch FP32 **0.28685245766989126** / ONNX FP32 **0.28685242727152244** / ONNX INT8 **0.27979221937243720**; sizes 12445→6487 bytes; latency 46.109899994917214 → 13.767600059509277 µs/sample (n_warmup=10, n_timed=1000, batch 1, **CPUExecutionProvider hard-pinned**, hardware unnamed, single run).
Derived deltas worth publishing precisely: export parity = **3.0398e-8 dB MAE delta** (torch→ONNX, unlabeled in repo); INT8 delta = **−0.0070602 dB (favorable = noise at n=100 holdout)**; size ratio **1.9184×** (⚠ the artifact's own `interpretation` string claims "~3-4× smaller" — contradicts its own numbers); latency ratio **3.3492×** (README's "~3.3×" ✓).
**The unstated headline: the sklearn baseline is 3.84× more accurate than the MLP the whole ONNX/INT8 path deploys** (0.0746 vs 0.2869 dB). Disclosed as intent ("deployment-path proof, not a DL flex") but never numerically.
INT8 elsewhere: **no INT8 path exists in any other repo** (no calibration, no quantization parity).

## TensorRT
- **Robotics/Thor**: ledger prose (:2352-2361, commit ec660af): "all 7 engines built and accuracy-verified vs PyTorch … Benchmark: backbone 49 ms + action head 71 ms = E2E 128 ms (7.8 Hz) on Thor. With 16-step chunks at 30 Hz control (533 ms per chunk), inference replans 4x faster than consumption." ⚠ **Tolerances and measured deltas are NOT RECORDED anywhere; no verification log or benchmark artifact exists on this machine** (engines live on Thor at ~/github/Isaac-GR00T/gr00t_trt_deployment/engines/). `TORCH_COMPILE_DISABLE` workaround: not in any repo file; the ledger records only "inductor-triton PTX bug in the benchmark baseline — engines unaffected." Before publishing the 49/71/128 ms numbers, pull the artifact off Thor or re-run — currently they are prose-only.
- **jetson-edge-ai-security `validate_engine`** (deploy/thor/build_tensorrt_engines.py:88-120): measures latency only, never compares outputs; FP16 flag set with no numerical check; and the call is **broken** — `execute_v2([int(d_input)])` passes one binding (no output buffer) and any failure is mislabeled "Validation skipped (pycuda not available)" while the engine is still counted a success (:168-171, :203).
- **wireless**: three-way Python↔ONNX↔TensorRT parity explicitly deferred and disclosed (README:297, :282).

## Parity check failed + root cause
**NO EVIDENCE FOUND in any repo** — every committed parity result is a pass; no commit/issue records a failure being diagnosed.

---

# Topic 5. Sim-to-real

**No matched sim+real pair exists anywhere. Verified.**
- reports/NOT_CLAIMED.md (robotics): "NOT real-hardware performance — all numbers are Isaac Sim with two sim-authority patches active (kinematic attach glue; release-hold), both compensating one documented broken contact model."
- Every robotics artifact self-stamps: `"data_kind": "simulated (Isaac Sim; … NOT real-hardware data)"`.
- What DOES exist on the sim side of a future pair: the **carry-orientation audit** (ledger:2369-2395, commit 442b8ab) — FK over 182 carry segments: `pitch_ref_deg 35.62550807957623`, p5-p95 `[25.529501488848485, 39.674796653321415]`, `carry_max_dev_p95_deg 47.299286599453616` (reports/carry_orientation_ref.json) — proving the sim corpus lacked level-carry behavior *before* deployment. Consequence recorded: "No hardware carry until addressed."
- The real-side assets: a real-rig teleop video (referenced in the ledger; **no file/hash committed anywhere — NO EVIDENCE FOUND for the artifact itself**) and the Alicia-D teleop bridge repo (machinery present; last commit 2026-07-30; hardware writes default OFF).
- To produce the first matched pair: (1) operator accepts real-arm session, (2) record with the LeRobot teleop recorder (data path built, corpus pending — docs/PIPELINE.md:40-42), (3) run the same square-to-square commands through the executor in sim and on hardware, same scoring rule, publish both numbers with the glue caveat on the sim side.

---

# Topics 9 and 11. Reporting discipline and evidence artifacts

## Where n / SD / CI / per-stage appear next to results (own reporting standard, shown)
- robotics ludo_stats_summary.json: `first_try {n 36, ok 32, rate 0.889, binomial_sd 0.052}` + the verbatim note: **"first_try = attempt 1 only; turn_level counts the built-in single retry; success != reward != normalized score (no reward exists in this pipeline)"** — the exact score/reward/success separation, machine-written.
- robotics session_summary.json: first-try vs with-retry kept distinct, p50/p95 with n, lane counts, per-attempt failure records embedded, wall-clock, sim steps.
- jetson training_run.json: `"n_runs": 200` on latency; CV method strings incl. "Leave-one-out CV on 25 lag samples"; pre-registered gates with criteria ("delta_auc >= 0.05" → PASS); dataset hash.
- wireless CSVs: per-row `num_bits` / `n_realizations` columns (1e6 bits AWGN; 200×10k Rayleigh; 50 chan-est).
- retention real-churn metrics: n=7043, train 5282, holdout 1761, **real labels, labeled as such** — the portfolio's only real-data ML result.
- private-5g: `n_cells: 3, samples_per_fleet_size_per_cell: 12`; full per-stage wall-clock table (benchmarks.md:24-33).
- copilot eval bundle: totals, per-vendor/category/question-set groups, latency block with budget.
- Standard-deviation reporting: robotics `binomial_sd` and the wireless jetson template's `std_us` (never yet run) are the only two. **Confidence intervals: NO EVIDENCE FOUND in any repo.**

## Full field-by-field contents
- **provenance.json** (robotics, 15 fields): timestamp_utc, git_sha, git_dirty, host, platform, python, gpu, cuda_runtime, seed, checkpoint, argv, control_hz, data_kind, scoring_rule. Append-only variant naming.
- **session_summary.json** (14 fields) + per-attempt record (15 fields: ok, err_mm, final_tilt_deg, carry_tilt_max_deg, steps, timed_out, lane, far_pick, far_place, hold_armed, grasped, pick_xy, place_xy, closest_approach_mm, attempt).
- **copilot cross-OEM bundle** (.runtime/evaluation/das-cross-oem-20260814-193701.json): 18 metrics enumerated in the agent pull; total 66, passed 66. ⚠ Saturated (all 1.0/0.0) — a fixture regression suite, not a benchmark; also gitignored and absent from CI.
- **private-5g operator decision bundle** (4 artifacts): operational_decision_summary (8 fields incl. `first_tested_unsafe_expansion_point_agvs: 120`), bottleneck_attribution (5 candidates × 6 fields incl. an explicit rejected hypothesis), operator_action_plan (5 × 5, every entry carrying a `boundary` string), operator_console_readiness (10 gates: 5 PASS / 3 NOT IMPLEMENTED / 2 NOT CLAIMED). The best decision-justification structure in the portfolio.
- **urban mock_inference_report.json**: includes `"unsupported_claims": ["real camera accuracy", "Jetson hardware latency", "automated enforcement readiness"]` as structured data, enforced by `test_tooltip_text_verbatim` — a test that fails if the honesty disclaimer drifts.

## What the artifacts CANNOT answer (the incident-gap post)
Robotics: no thermal/power data anywhere; no per-step timing (policy-server inference latency never logged in the executor path); no media checksums (the demo MP4s are unreferenced by any artifact); binomial_sd assumes turn independence that the ledger itself disputes ("the deterministic rails are per-session-history, not per-turn"); sessions s1-s3 predate instrumentation (no provenance at all); `git_dirty: true` on s10 — the SHA does not fully describe the code that ran; ludo_stats_summary.json is an overwritten derived view (the 6-seed numbers survive only in ledger prose).
jetson training_run.json: date-only timestamp (cannot order same-day runs); no library versions (an AUC of 0.9796 is unreproducible without the sklearn version); no ONNX file hash; no per-fold CV values; single f1 for a 15-class problem with no averaging mode stated.
copilot: the per-request stage latencies are emitted in responses and **never aggregated into any committed artifact**; the eval bundles are gitignored.
If an incident happened tomorrow, the questions you could NOT answer from current capture: "was the box thermally throttled?", "what did the model server latency look like at the time?", "which exact video corresponds to this failure?" (no hash), "was the code clean at that SHA?" (dirty flag says no).

---

# Topic 13. A benchmark that changed nothing

1. **ai-ran three-model comparison (PUBLIC)** — ridge_linear RMSE 0.8368114959180818 vs gradient_boosting 2.875489035353364 vs mlp 22.59257620531283 (comparison_metrics.csv); DEFAULT_MODEL stayed `ridge_linear`. Criteria in-code: "surface 'which family wins on this KPI on this dataset' honestly rather than cherry-picking." What would change the decision: the Telecom Italia MI benchmark (loader + make target committed, dataset deliberately not — "No benchmark metric claimed yet"). ⚠ n = 8 holdout points (48-row single-cell CSV) — NOT PUBLISHABLE as a model-selection result; publishable as a decision-process exhibit with n stated.
2. **ai-phy neural receiver (PUBLIC)** — evaluated, kept as a benchmark, not adopted as "better": at 15.0 dB neural BER 0.008045675712719299 vs classical 0.004710029468201754 (1.71× worse); at 20.0 dB 5.89× worse BER, 1.55× worse BLER (bler_comparison.csv, 11,673,600 bits/point, identical channel realizations both receivers). Also: `best_val_bler = 1.0` at all 50 validation points — the intended selection metric was inert; checkpoint fell to val BER (82000 vs 100000 differ by 1.77% relative on 1,280 frames = noise). ⚠ Do not present checkpoint selection as early-stopping evidence.
3. **wireless neural channel estimator (PUBLIC)** — wins MSE_h at 0 dB (0.0915 vs MMSE 0.1767) but loses 35.4× at 30 dB with an error floor at ~0.008; BLER 0.26 vs MMSE 0.10 at 30 dB. Not adopted (no downstream artifact). ⚠ n=50 realizations/point → directional only.
4. **Robotics STACK_WATCHLIST (PRIVATE)** — the standing not-adopted register, each entry with a named adoption trigger: Newton physics ("migrate with the framework, re-validate contact-sensitive numbers"), Isaac Lab Mimic ("next corpus-scaling need"), FoundationPose + cuMotion ("P4 real-hardware"), Cosmos Transfer ("before first real-camera policy runs"), ER-2 streaming ("if sidecar cadence outgrows request/response"). Plus the executed adoption decision with a written reversal condition: N1.5→GR00T-1.7 — "Would reverse if: 1.7 fine-tune on our LeRobot v2.1 corpora underperforms the E48-class N1.5 baselines at matched steps."
5. **edge-traffic-sensor ADR-001/002 (PRIVATE)** — four alternatives rejected with stated criteria (Zeek on ARM64 install surface + flow-finalization latency; tshark on per-packet rate; custom extractor on scope prohibition) + §Re-evaluation triggers. urban AD-4: vLLM over Ollama (⚠ its "~30% on Cosmos-2B in NVIDIA's own benchmarks" is an uncited third-party number — do not publish without the citation).

---

# JETSON HARDWARE NUMBERS — separate section

**Every Jetson measurement that exists anywhere: NONE. Zero. In all ~16 repos.**
- jetson-edge-ai-security `reports/thor_benchmark.json`: the only Jetson-shaped artifact — **every measurement field is null** ("run_id": "pending-thor-run"; all six tier rows p50/p95/p99 null; all four gates "pending"). Even when filled, its schema omits: power mode, batch size, precision label, warm/cold, throttling, sustained-vs-peak memory, std/CI.
- wireless `reports/jetson_inference_benchmark.json`: **does not exist**; the committed quantization artifact hard-codes a sentence pointing at it anyway (snr_quantization_comparison.json:24).
- The READMEs are honest about it — five separate "TO MEASURE"/pending declarations in wireless, README boundary lists in jetson-edge/urban/safety-obs/edge-traffic/hub ("Not claiming Jetson benchmark numbers until artifacts are committed"). The dashboards gate on artifact existence, not prose (wireless build_dashboard.py:347-354, 480, 693; jetson dashboard renders `pending-thor-run`).
- The nearest real numbers, correctly labeled non-Jetson: jetson-edge `training_run.json` CPU latencies (detector p50 0.014 ms / p95 0.015 / p99 0.016; forecaster 0.007/0.007/0.008; "RTX 5090 dev box (CPU-only)", badge "measured-cpu", n_runs 200, batch 1). ⚠ single run, no variance — publishable only as "sub-millisecond on CPU, n=200, single run."
- Thor numbers that exist in prose only (robotics ledger, PRIVATE): TRT E2E 128 ms / 7.8 Hz (see A6 caveat — no artifact on this machine) and "first Thor training job: 5.73 steps/s, 2.5x the 5090 on this decode-bound workload" (ledger:1687-1688; log lives on Thor).
- Device-naming inconsistencies to fix before any Jetson post: jetson-edge says AGX Thor + JetPack 6.x while its PORTFOLIO_DELIVERABLES says "Orin-class"; urban says "Jetson Orin"; edge-traffic says Thor + JetPack 7.x; wireless names Thor but its guide describes "Cortex-A78AE" (Orin-generation) and TECH_BRIEF says "drop onto a Jetson Orin/Nano"; wireless dashboard says the Thor "is in hand" while its README says "until hardware arrives."

---

# FINALLY

## 1. The three strongest pieces of evidence found

1. **The pre-registration ledger with a written retraction** (robotics, topics 3/4). Fourteen instances of bars written before runs, including: a pass by 0.0004 (E26), a 4×-worse failure where the tempting secondary metric was explicitly refused (E29), and a claimed 1.86× retry effect that the pre-planned replication reversed and the ledger **retracted in writing** (E39→E39b). Strong because it is dated, sequential, self-incriminating, and quotes its own decision rules — nobody fabricates a retraction.
2. **The doomed-rollout and retry-economics numbers, computed from machine-emitted artifacts** (robotics, topics 1/2/6). Median failing attempt = the full 8999-step budget (56.5% of failures run to timeout); retries cost ~21% more steps than first attempts and consumed 16.6% of all sim steps; one turn burned 17,998 steps for zero yield; and the case where adding a cloud verification gate *reduced* carry success 0.192→0.060 while raising cost (38/53 gates negative, later proven to be a camera-frustum problem, not a model problem). Strong because every number regenerates from `turns.jsonl` with one command, and the cost side has never been published by anyone.
3. **The evidence-surface-that-renders-its-own-absence family** (cross-portfolio). private-5g's SHA-256 byte-equality determinism check (two committed matching hashes); wireless/jetson dashboards that gate claims on artifact existence (`pending-thor-run` badge, "TO MEASURE" rows); urban's `unsupported_claims` as a structured JSON field enforced by a verbatim-match test. Strong because it is a *system*, demonstrated in four repos, and two of them are public.

## 2. Genuinely surprising, no post planned yet
- **The wireless SNR stack ships the worse model**: sklearn baseline MAE 0.0746 dB vs the deployed MLP's 0.2869 dB — 3.84× regression, disclosed as intent but never numerically. "The deployment path mattered more than the model — here's the number I never printed" is a strong post.
- **A capture executed on the first-ever live run of the multi-command path** (robotics, seed 5 turn 13): the game engine emitted a two-command plan (return blue's token, then move red's) and both commands succeeded — first time the path ever ran.
- **The saturated benchmark problem in your own portfolio**: copilot's 66/66 with every metric exactly 1.0/0.0 — a fixture suite reading as a benchmark. Pairs with ai-phy's val-BLER stuck at 1.0 for the whole run (selection metric inert). "Two ways a metric can be dead and still green."
- **The public hub's green CI validates stale vendored snapshots** missing 15 of 23 test files — including all parity and Jetson tests. The badge is not evidence about the repos it advertises.
- **The percentile-implementation audit**: three different off-by-one percentile bugs across the cluster (index 50 of 100 labeled p50; p95=max at n=8) with exactly one correct implementation (safety-obs `_percentile`). A tidy micro-post.
- **The 5-of-6 failed attempts clustered in one board quadrant** and were eliminated by one geometric fix (far-place polar drag): failure taxonomies pay rent.

## 3. Claims your public READMEs/docs make that the artifacts do NOT support
(Public repos first — these are the exposure.)
- wireless README:107 "125-sample **stratified** holdout" — the split has no `stratify=` argument (models.py:69). The headline 0.472 classifier accuracy comes from an unstratified n=125 split.
- wireless README:110-113 repro recipe cannot reproduce the committed numbers (recipe: 120 samples; CI: 60; committed artifact: 500, generated on Windows outside both).
- wireless README:34 "77 tests" vs README:222 "15 pytest tests" — self-contradiction in one file.
- wireless snr_quantization_comparison.json's own `interpretation` claims "~3-4× smaller" against its own 1.9184× measurement.
- ai-phy README:37 "31/31 passing" vs 37 defined tests, no parametrize/skips.
- ai-phy README:118 "10.4 min / 100k steps on RTX 5090" — in prose only; training_log.json has no wall-clock and no GPU name.
- ai-phy README:35 "2-3 dB effective BER gain" — sourced to an SVG; the code path that would compute the shift is dead (`snr_db - snr_db`).
- ai-phy README:184 implies curves from a 100k-step model; comparison_summary.json says trained_steps 82000.
- ai-ran headline model table omits n=8 everywhere; "make verify regenerates committed evidence" excludes the 8 edge-ai artifacts.
- private-5g README "Pipeline benchmarks (measured)" — n=1, hardware unnamed, and CI overwrites the numbers on different hardware; the 100-AGV ceiling rests on a p95 from 36 samples and **two committed artifacts disagree on it** (18.39 vs 19.962 ms at fleet 100, and 19.962 sits 0.038 ms under the budget).
- private-5g README:218 links a renamed repo.
- (Private but will leak into posts if unchecked): robotics docs/PIPELINE.md "7 engines, verified vs PyTorch, 7.8 Hz E2E" has no artifact on this machine — ledger prose only; "25 unit tests" for L4 is unverifiable (16 by definition count, collection errors on missing `chess` dep); "8-35mm precision" — 35 is the scoring threshold, not a measured max (say p50 19 / p95 29-30). jetson-edge README's three evidence links point at directories containing only `.gitkeep`. The robotics repo's own top-level README.md is **0 bytes**.
