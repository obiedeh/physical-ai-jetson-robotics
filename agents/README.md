# Robot Operations Copilot

This track will contain RAG and agentic workflows for robot operations.

Current CLI entry point:

```bash
physical-ai-lab ops-triage --robot-id robo-car-01
physical-ai-lab ops-triage --robot-id synria-arm-01 --json
```

The first implementation is deterministic rule-based triage over robot telemetry.
That gives the future RAG/copilot layer a stable safety-bounded baseline before
we connect manuals, ROS logs, and operator notes.

Planned inputs:

- robot telemetry
- ROS logs
- hardware manuals
- calibration notes
- maintenance procedures
- network/edge latency data

Planned outputs:

- root-cause summaries
- operator recommendations
- maintenance triage
- safety-bounded action plans
