# Jetson Evidence Reports

This folder is for future Jetson runtime validation artifacts. Do not add claims here unless the matching files are committed.

Expected artifacts include:

- vLLM startup logs
- memory snapshots
- p95/p99 latency summaries
- sustained run notes
- model config used
- hardware/software versions
- failure/recovery notes

Use `sample_metrics_schema.json` as the expected shape for benchmark summaries. A filled metrics file should identify the device, JetPack/CUDA versions, model configuration, run duration, latency, throughput, memory pressure, and any observed failure mode.

