# Agentic Ops Copilot

A thin enrichment layer that sits on top of the deterministic, safety-bounded `physical-ai-lab ops-triage` baseline. Closes the agentic pillar of the four-pillar portfolio frame (Physical AI / Robotics / Edge AI / Agentic Systems).

## What this does

- Consumes a `TriageReport` produced by `physical_ai_lab.ops_copilot.triage_telemetry` — the safety-bounded rule-based triage baseline.
- Asks a pluggable `Classifier` to enrich the triage with an operator-facing root-cause hypothesis, a list of recommended actions, a confidence label, and an explicit `escalate_to_human` flag.
- Emits a structured `OpsCopilotReport` that combines the deterministic triage and the classifier enrichment.

The deterministic triage is the safety-bounded baseline. The enrichment is decision-support context for qualified human operators, not autonomous decision output.

## Classifiers

The `Classifier` protocol decouples enrichment from any specific implementation.

### `DeterministicClassifier` (CI default)

Rule-based enrichment — no API call, no external dependency. Runs hermetically in CI and on machines without an API key configured. Always available as the safe fallback.

### `AnthropicClassifier` (LLM-backed, implemented)

Uses the Anthropic Messages API (`claude-haiku-4-5-20251001` by default) to generate a richer root-cause hypothesis and operator recommendations. Requires:

```bash
pip install "physical-ai-jetson-robotics-lab[llm]"   # adds anthropic>=0.34
export ANTHROPIC_API_KEY=sk-...
```

Falls back silently to `DeterministicClassifier` on any API failure or missing key, so the copilot always emits a valid report.

The classifier name is recorded in every report for observability:

```json
"classifier_name": "anthropic/claude-haiku-4-5-20251001"
```

## Run the prototype

**Deterministic (no API key required — hermetic CI default):**

```bash
python -m agents.ops_copilot.copilot \
  --robot-id robo-car-01 \
  --samples 12 \
  --output reports/agents/ops_copilot/latest.json
```

**LLM-backed (`AnthropicClassifier`):**

```bash
ANTHROPIC_API_KEY=sk-... python -m agents.ops_copilot.copilot \
  --robot-id robo-car-01 \
  --samples 12 \
  --llm \
  --output reports/agents/ops_copilot/latest.json
```

**Via the top-level CLI:**

```bash
physical-ai-lab ops-copilot                  # deterministic
physical-ai-lab ops-copilot --llm            # AnthropicClassifier (if key is set)
```

Inspect the result:

```bash
cat reports/agents/ops_copilot/latest.json
```

## Output shape

```json
{
  "robot_id": "robo-car-01",
  "status": "watch",
  "classifier_name": "deterministic",
  "confidence": "medium",
  "escalate_to_human": false,
  "root_cause_hypothesis": "Operational margins narrowing on latency, thermal. Continue with caution and capture telemetry evidence.",
  "operator_recommendation": [
    "Reduce speed limits and inspect load/friction before repeat runs.",
    "Prefer local inference and defer cloud-dependent decisions."
  ],
  "triage": {
    "robot_id": "robo-car-01",
    "status": "watch",
    "findings": [
      {
        "severity": "watch",
        "signal": "thermal",
        "summary": "Motor temperature is 67.4 C.",
        "recommended_action": "Reduce speed limits and inspect load/friction before repeat runs."
      }
    ]
  }
}
```

The `triage` block is the deterministic baseline (unchanged). The top-level fields are the enrichment layer.

## Swap path: simulated telemetry → real ROS 2

The current prototype consumes the deterministic `generate_demo_telemetry` batch generator. The `enrich()` function works against any `TriageReport`, so the swap path is:

1. Replace `generate_demo_telemetry(...)` with an `rclpy` subscriber that buffers a rolling window of samples from a real ROS 2 topic.
2. Loop: pop a window, run `triage_telemetry(window)`, run `enrich(triage, classifier)`, ship the report.
3. Nothing in `enrich()` or the classifier changes.

This swap is implemented in `ros2_ws/src/physical_ai_ops_copilot/` — a ready-to-deploy ROS 2 ament_python package with `OpsCopilotNode` that subscribes to `/robot_telemetry` and publishes to `/ops_copilot/report`.

## Safety boundary

This is decision-support tooling for qualified human operators. It is not autonomous control logic. The `escalate_to_human` flag on every report is load-bearing — a downstream control system must respect it before resuming autonomous operation.

Any future change to `physical_ai_lab/ops_copilot.py` or `physical_ai_lab/telemetry.py` must go through the `physical-ai-safety-reviewer` skill per `AGENTS.md`. The agentic layer here only consumes the safety-bounded baseline — it does not modify it.
