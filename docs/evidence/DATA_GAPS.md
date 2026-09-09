# Data Gaps in This Presentation

No new measurements were generated. The following requested views cannot be
rendered literally from the retained files:

| Requested view | Available record | Presentation |
| --- | --- | --- |
| Thor 100-iteration latency distribution | [JSON](../../reports/thor_trt_benchmark/thor_trt_benchmark.json) and [log](../../reports/thor_trt_benchmark/bench.log) retain n, min/max, mean/SD and median, not the individual iterations | Range/median chart with mean/SD labels; no histogram, synthetic samples or fitted distribution |
| Thor power trace beside temperature | [thermal_samples.json](../../reports/thor_trt_benchmark/thermal_samples.json) contains temperature/time fields only; [benchmark JSON](../../reports/thor_trt_benchmark/thor_trt_benchmark.json) records rail peaks | Raw temperature points plus separately labelled rail peaks; no power timeline |
| Orin camera distribution and thermal/rail traces | [day_one_smoke.json](../../reports/jetson/yahboom_day_one/day_one_smoke.json) retains percentiles, ranges and summaries, not raw sequences | Summary panels; no interpolated sequence or reconstructed distribution |
| One Ludo distribution with pooled p50 19.3 / p95 30.1 | [pooled frozen record](../../reports/ludo_stats_frozen/stats_2026-08-20T050701Z_reports_ludo_soak_sSTAR.json) spans pre/post-fix configurations | Separate panels from [CSV](../../reports/ludo_stats.csv); historical pooled values explained in prose only |
| Thor junction-temperature peak | The [benchmark JSON](../../reports/thor_trt_benchmark/thor_trt_benchmark.json) throttling note states a tj peak near 40.5 C; the raw [thermal_samples.json](../../reports/thor_trt_benchmark/thermal_samples.json) peaks at tj 42.125 C and GPU 42.0 C over 31 samples | Charts and prose use the raw GPU sample peak of 42.0 C; the JSON prose value is a recorded inconsistency in an unaltered source, not a corrected figure |
| Physical ground truth for the false positive | [corrected sidecar](../../reports/ludo_groot17_eval01/session_summary_corrected.json) contains simulator events and cup displacement | Explicitly simulator evidence, never physical object-success ground truth |

The plotting script never reads the sentinel-bearing clearance field.
Figures show source paths and sample counts. Exact values and hashes remain in
[source records](figures/sources.json). Requested sources have not been altered.
