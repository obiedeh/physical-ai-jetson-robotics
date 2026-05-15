# Edge AI

This track will contain Jetson Thor/Orin inference and optimization work.

Current CLI entry point:

```bash
physical-ai-lab edge-plan --compute jetson-orin
physical-ai-lab edge-plan --compute jetson-thor --json
```

The initial benchmark plan defines measurable workload targets for FPS, latency,
power, and memory. It is intentionally hardware-evidence oriented: the next step
is to run the plan on Jetson Orin/Thor and write results under `reports/edge_ai/`.

Planned work:

- camera inference demos
- ONNX and TensorRT deployment paths
- latency, FPS, memory, power, and thermal benchmarks
- edge VLM/LLM experiments where practical
