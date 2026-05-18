# Agentic Ops Copilot

A thin enrichment layer that sits on top of the deterministic, safety-bounded `physical-ai-lab ops-triage` baseline. Closes the agentic pillar of the four-pillar portfolio frame (Physical AI / Robotics / Edge AI / Agentic Systems).

## What this does

- Consumes a `TriageReport` produced by `physical_ai_lab.ops_copilot.triage_telemetry` — the safety-bounded rule-based triage baseline.
- Asks a pluggable `Classifier` to enrich the triage with an operator-facing root-cause hypothesis, a list of recommended actions, a confidence label, and an explicit `escalate_to_human` flag.
- Emits a structured `OpsCopilotReport` that combines the deterministic triage and the classifier enrichment.

The deterministic triage is the safety-bounded baseline. The enrichment is decision-support context for qualified human operators, not autonomous decision output.

## Why a pluggable classifier

The `Classifier` protocol decouples enrichment from any specific LLM SDK.

Today's default is `DeterministicClassifier` — a rule-based stub that produces structured enrichment from the existing severities and signals without any API call. It runs hermetically in CI and on machines without an LLM configured.

An LLM-backed classifier (for example, an Anthropic SDK adapter) becomes a focused follow-up PR. The plug point is the protocol; no existing code changes when the LLM adapter is added.

## Run the prototype

From the repo root, with the lab installed (`pip install -e .`):

```bash
python -m agents.ops_copilot.copilot \
  --robot-id robo-car-01 \
  --samples 12 \
  --output reports/agents/ops_copilot/latest.json
```

This generates synthetic telemetry, runs the deterministic triage, runs the deterministic enrichment, and writes a structured JSON report.

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

This swap is the bridge from this prototype to the ROS 2 ops copilot once Jetson + Synria hardware is online.

## Swap path: rule-based → LLM

The current prototype uses `DeterministicClassifier` (no API call). The swap path is:

1. Add an LLM dependency under `[project.optional-dependencies]`, for example `llm = ["anthropic>=0.34"]`.
2. Add a class implementing the `Classifier` protocol — same interface, LLM call inside `enrich(triage) -> Enrichment`.
3. Pass the new classifier instead of `DeterministicClassifier()` to `enrich()`.

Nothing else changes. The deterministic classifier stays as the hermetic CI / no-API-key fallback.

## Safety boundary

This is decision-support tooling for qualified human operators. It is not autonomous control logic. The `escalate_to_human` flag on every report is load-bearing — a downstream control system must respect it before resuming autonomous operation.

Any future change to `physical_ai_lab/ops_copilot.py` or `physical_ai_lab/telemetry.py` must go through the `physical-ai-safety-reviewer` skill per `AGENTS.md`. The agentic layer here only consumes the safety-bounded baseline — it does not modify it.
