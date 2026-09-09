# Paper outline — working title:
# "Equal-Budget Benchmarking of Learning Paradigms for Manipulation:
#  RL, Imitation, and VLA Fine-Tuning on Edge Hardware, with a Layered
#  Cloud-Perception System"

Target: research-level writeup from the Aug 2026 Synria campaign.
All numbers trace to docs/SYNRIA_EXPERIMENT_LEDGER.md entries and
reports/logs artifacts (git-hashed provenance).

## 1. Introduction
- Question: given equal development GPU-time on one's own hardware,
  what does each learning paradigm deliver on a real manipulation task?
- Contributions: (a) equal-budget protocol + measured budgets;
  (b) dose-response curves per paradigm; (c) protocol-dominance and
  language-gate measurement findings; (d) cloud-perception validation
  methodology (pre-registered bars, failure-mode mapping);
  (e) layered-system rows quantifying architecture as a multiplier.

## 2. Task & platform
- Synria/Alicia-D 6-DOF, Isaac Sim pick-carry-place funnel
  (grasp / carry5s / dock / cycle), scene, action space history
  (8-dim corpus convention vs 7-dim post-E30 env action space).
- Hardware: RTX 5090 (sim factory), Jetson AGX Thor 128 GB (train+serve).
- Fixed-harness protocol: seeds, n=64 screens, n>=150 confirmations,
  staged vs scratch start distributions.

## 3. Method: the equal-budget campaign
- Budget rule (83.4 h measured RL spend; audit method: gap-filtered
  artifact timestamps; naive lifespan overcounts 6x).
- Accounting: corpus charged to imitation family; screens excluded;
  pre-registration discipline; retraction policy (E39 case).

## 4. Paradigm arms
- RL/PPO: 21 runs, reward-design iterations; ceiling analysis
  (upright-grasp 0.33 wall = objective-design-bound).
- ACT: corpus lineage (e33 discipline, E38 style-dilution finding,
  overfit cliff / 5-epoch rule), E47 expanded-corpus retrain.
- GR00T N1.7: frozen -> full -> visual-only ablation (E41/E42);
  warm-restart confound (E43); schedule-length dose-response
  (E44/E45: 10k->40k, superlinear band 30-40k); E48 merged-corpus.

## 5. Results: bare policies
- Final closed-budget table (n>=150).
- Dose-response figures: per-paradigm lever curves.
- Variance analysis: cosine-tail checkpoint variance; screen-selection
  bias (0.562 @ n=64 -> 0.304 @ n=151 case study).
- The dock wall and its partial breach by schedule length alone.

## 6. Measurement findings (standalone value)
- Protocol dominates reporting: same policy 0.852 vs 0.156.
- The language gate: paraphrase -> 0.000 (VLA instruction as API).
- Instrument audits: zone-marker visibility case; affine-truth failure.

## 7. Cloud perception layer (Gemini Robotics-ER 2)
- Pre-registered validation: cup 2.6 mm / zone 2.8 mm / latency 2.6 s.
- Failure modes: temp-0 unbounded reasoning hang; consistent
  misidentification (~5%) defeating consensus guards; guessing vs
  declining on invisible targets.
- Latency-derived role assignment (30 Hz / 1 Hz / 0.4 Hz arithmetic).
- Live sidecar demonstration (staleness-honest overlay).

## 8. System rows (E49)
- S0 bare / S1 +retry supervisor / S2 +Gemini phase gates.
- Metrics: funnel + cycles-per-hour (retries consume time).
- Architecture-as-multiplier claim, measured.

## 9. Discussion
- What each paradigm's ceiling was actually made of (objective design /
  data / compute schedule).
- Edge-hardware enablement: what 128 GB unified memory changed.
- Threats to validity: sim-only; single task/embodiment; n limits;
  paradigm recipes are best-known-practice not exhaustive search.

## 10. Reproducibility appendix
- Repos (2), commit-hashed scripts per experiment, seeds, budget audit
  script, all eval logs/JSONs, corpus manifests, model/checkpoint
  inventory (what is kept vs pruned and why).

## Figures list (draft)
F1 funnel schematic; F2 budget accounting; F3 four-paradigm final
table; F4 dose-response curves; F5 protocol-effect bar pair;
F6 checkpoint-variance strip (E45 tail); F7 perception precision +
latency distributions; F8 sidecar frame with staleness overlay;
F9 system rows S0/S1/S2.
