# Instrumentation pass — executed 2026-08-20 (task dated 2026-08-19)

First-boot evidence update, 2026-09-07: the device-reconciliation section's
then-pending Orin identity is resolved by
[2026-08-20 provenance](../../reports/jetson/yahboom_day_one/provenance.json):
Orin NX, L4T R36.4.4. The retained Thor and Orin thermal figures describe short
benchmark/probe runs, not sustained validation.
Follow-up to docs/notes/first_party_evidence_pull.md. Exact values, never
rounded. Data kind stated per number.

## TASK A — recoveries

### A1. Thor TensorRT numbers — RECOVERED, and the record is corrected
Finding first: the prose claim was **misattributed**. The Thor pipeline log
(`~/github/Isaac-GR00T/gr00t_trt_deployment/pipeline.log`, 2026-08-16) shows
the "E2E 128 ms (7.8 Hz)" in the ledger/PIPELINE.md was the **torch.compile
baseline**; the TensorRT benchmark step **FAILED** that day on
`ModuleNotFoundError: No module named 'matplotlib'`
(`standalone_inference_script.py:38`). Fixed (matplotlib installed into the
Thor gr00t venv) and re-run 2026-08-20 with n=100 iterations, warmup 10,
seed 42, batch 1, bf16, on the existing 7 engines.

Recovered numbers — artifact `reports/thor_trt_benchmark/thor_trt_benchmark.json`
(+ raw bench.log, thermal_samples.json, nvpmodel.txt, l4t.txt, tegrastats on
Thor). Data kind: REAL hardware measurement (Jetson AGX Thor, 120W mode);
inputs from the simulated e46 dataset. Model: E48 ckpt-35000 (n1d7 family).

| mode | E2E median | mean ± std | min / max | Hz | backbone | action head |
|---|---|---|---|---|---|---|
| PyTorch Eager | 126.6 ms | 126.9 ± 2.5 ms | 122.2 / 139.0 | 7.9 | 48.19 ms | 70.44 ms |
| **TensorRT n17_full_pipeline** | **101.6 ms** | **101.8 ± 2.0 ms** | 97.8 / 107.8 | **9.8** | 31.69 ms | 61.73 ms |

Speedup 1.25× E2E. Thermal: 31 sysfs samples, GPU peak 42.0 C, tegrastats
1 Hz log shows tj peak ~40.5 C, VDD_GPU peak 10676 mW, VIN peak 39937 mW —
**no throttling** (clocks pinned nominal throughout).
**Accuracy-verification deltas vs PyTorch: NEVER RECORDED** — the 2026-08-16
log contains zero parity lines and the benchmark harness computes none. The
artifact states this in an explicit `accuracy_verification.note` field. A
tolerance-checked parity run remains TODO before any accuracy-equivalence
claim. `docs/PIPELINE.md:21` corrected accordingly.
Bonus finding: `engines/export_metadata.json` shows the engines are the
**gr00t n1d7 (1.7-family) pipeline** (bf16, batch 1) — the "Thor TRT rebuild
for 1.7" consequence may be substantially prepaid.

### A2. Precision percentiles replace "8-35mm" — DONE
Computed from per-attempt records across instrumented sessions s4-s12
(simulated), using the safety-observability `_percentile`
(round((p/100)*(n-1))) now ported into `isaac/scripts/ludo_stats.py` —
which itself previously carried the `int(0.95*n)` off-by-one; do-not-write-
a-fourth honored by fixing the third.
Pooled s4-s12: first-try OK err_mm **p50 19.3 / p95 30.1 (n=91)**;
first-try rate 91/109 = 0.835 (binomial_sd 0.036, caveat field attached);
turn-level 103/108 = 0.954. `docs/PIPELINE.md:66-72` rewritten to these
values; the 8-35mm phrasing is retired in-place with the reason (35 was the
scoring threshold).

### A3. Six-seed numbers frozen — DONE
`ludo_stats.py` now also writes an append-only campaign-stamped copy to
`reports/ludo_stats_frozen/stats_<utc>_<glob>.json` (never overwritten;
existing summary path unchanged). Committed frozen artifacts:
- `..._reports_ludo_soak_s[456789].json` — 6-seed block: first-try 59/73 =
  0.808, turn-level 68/72 = 0.944, err p50 19.4 / p95 30.1 (n=59).
- `..._reports_ludo_soak_s1[012].json` — post-fix block: first-try 32/36 =
  0.889, turn-level 35/36 = 0.972.
- `..._reports_ludo_soak_sSTAR.json` — pooled (numbers in A2).

## TASK B — forward capture

- **B1 per-inference timing** — `ludo_turn_executor.py`: policy call timed
  (`_t.perf_counter()` around `policy(obs)`), per-attempt fields
  `infer_ms_p50`, `infer_ms_p95`, `infer_n`; session summary gains
  `infer_ms_session` (p50-of-attempt-p50s, worst attempt p95, total calls).
- **B2 decided_at_step** — set on attach latch, on observed detach, and on
  hold-arm; emitted per attempt. Every future rollout reads "decided at X,
  ran to Y".
- **B3 thermal/power** — `_gpu_health_sample()` (nvidia-smi temp/power/
  throttle-reasons) sampled at each command boundary; session summary gains
  `gpu_health` {samples, temp peak/mean, power peak, throttle_events,
  throttled, note}. The broken tegrastats-with-timeout pattern is NOT
  copied; for Jetson the correct pattern ships in Task C's `ThermalWatch`
  (background thread + sysfs) and was already used live in the A1 recovery
  driver on Thor.
- **B4 media checksums** — corpus manifest rows gain
  `media_sha256{wrist,overhead}`; `ludo_compose.py` writes an append-only
  `media_manifest.json` (file, sha256, generated_utc, frames, turns) per
  composed demo clip.
- **B5 dirty-diff capture** — `_write_provenance()` writes
  `uncommitted.patch` (append-only naming) and records
  `uncommitted_patch_sha256` whenever `git_dirty` is true.
- **B6 device asserts + provider provenance** — provenance now raises the
  rtx_training-pattern RuntimeError if CUDA is unavailable, and records
  `torch_cuda_device`, `torch_version`,
  `onnxruntime_available_providers`. `edge_ai/onnx_runner.py:69-83`:
  implicit mock fallback now raises unless `force_mock=True` or
  `PHYSICAL_AI_ALLOW_MOCK=1` (existing tests pass — they already used
  force_mock; 11/11 green).
- **B7a E31** — searched. NO contemporaneous RESULT exists. Artifacts
  (`reports/eval/e31_upright_s{42,43,44}.json`, probes) cannot resolve the
  bar: the pre-registered primary `tipped_rate` was never written to any
  artifact and no control-arm main-eval exists. Ledger appended with an
  explicit UNRESOLVED-BY-CONSTRUCTION note carrying the exact retrospective
  readings (upright-arm place 0.019305019305019305 / 0.12177121771217712 /
  0.027131782945736434 at n=259/271/258).
- **B7b binomial_sd note** — `binomial_sd_note` field added beside the
  statistic: "assumes independent Bernoulli trials; the ledger disputes this
  (the deterministic rails are per-session-history, not per-turn), so treat
  the SD as a lower bound on uncertainty." Statistic itself unchanged.

## TASK C — Orin starts instrumented

`physical_ai_lab/jetson_provenance.py` created and contract-tested:
- `write_provenance()` — append-only, refuses vague `data_kind` (must state
  real/simulated/synthetic/fixture); machine-reads device-tree model, L4T,
  JetPack apt, nvpmodel, jetson_clocks, CUDA, TensorRT; captures dirty
  diffs; requires precision + batch_size as arguments.
- `fix_seeds()` — python/numpy/torch + PYTHONHASHSEED + CUDA determinism
  flags (`use_deterministic_algorithms(warn_only)`, cudnn.deterministic),
  all values recorded into provenance.
- `latency_stats()` — refuses to emit a latency without n>=2, percentiles
  (correct implementation), std, and a warm/cold label. Bare means are
  unconstructible.
- `ThermalWatch` — sysfs thermal + MemAvailable sampling thread with
  peak-vs-sustained summary and a throttling flag; explicitly NOT the
  tegrastats subprocess.run(timeout) anti-pattern.
- data_kind convention: hardware runs must use the positive form
  ("REAL hardware measurement (...)"); enforced by the vagueness check.

### Device reconciliation
Physically on the desk (per operator, 2026-08-19): a **Yahboom Jetson Orin
system**, just unboxed. Exact Orin module (Nano/NX/AGX), JetPack and L4T:
**NO EVIDENCE FOUND until first boot** — `jetson_facts()` will machine-read
all of it on day one; do not publish a device name before that runs.
Repos that currently disagree with "Orin on the desk":
- jetson-edge-ai-security: README + thor_benchmark.json say **AGX Thor,
  JetPack 6.x**; its own PORTFOLIO_DELIVERABLES.md:33 says "Orin-class".
- edge-traffic-sensor: ARCHITECTURE.md:264-268 says **Thor, JetPack 7.x**.
- wireless-link-intelligence-system: names **Thor** but
  JETSON_BENCHMARK_GUIDE.md:104 describes a Cortex-A78AE (Orin-generation
  CPU) and TECH_BRIEF.md:75 says "Jetson Orin/Nano"; dashboard says the
  Thor "is in hand" while README says "until hardware arrives".
- urban-edge-vision-analytics: README says **Jetson Orin** (consistent with
  the desk, inconsistent with its sibling repos).
Note: the AGX Thor named in this repo's artifacts is a REAL separate
machine (192.168.1.170) — the reconciliation applies to the four repos
above telling one edge story with three device names, not to this repo.

## Not done / out of scope
- TRT-vs-PyTorch accuracy parity run: TODO (harness computes none; needs a
  small script — flagged in the A1 artifact).
- The ~21 `device="cuda"` sites: the shared assert now lives in the
  provenance path every session runs through; per-site edits judged
  refactor-for-its-own-sake under the non-goals.
- Telecom repos: untouched per non-goals.
