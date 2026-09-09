"""Leg C: map ROSMASTER M3 Pro telemetry to structured SafetyEvents for the
physical-ai-safety-observability platform (POST /events).

Pure and stdlib-only (no ROS, no lerobot) so the rules are unit-testable off
the robot and reusable by the bridge. Each rule returns event dicts whose keys
match the platform's SafetyEvent schema (required: camera_id, rule_id, severity,
confidence, human_review_required, evidence, summary); evidence carries the
telemetry snapshot so a reviewer sees exactly what fired the rule.

Design (real-robot honesty): these are RULE-triggered events from measured
telemetry (LiDAR nearest-return, battery, deadman, base speed, optional person
detection) — not model guesses. Severity escalates with proximity/voltage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class SafetyConfig:
    camera_id: str = "m3pro_front"
    source_uri: str = "zmq://m3pro_host/front"
    rule_version: str = "m3pro-safety-v1"
    # battery (V): the pack is nominal ~12.6 full, browns out low
    batt_medium: float = 10.8
    batt_high: float = 10.2
    batt_critical: float = 9.8
    # LiDAR nearest obstacle (m) while the base is moving
    base_moving_mps: float = 0.02
    stop_medium_m: float = 0.70
    stop_high_m: float = 0.40
    stop_critical_m: float = 0.20
    # person-in-workspace detection confidence gate
    person_conf_min: float = 0.5


def _evidence(cfg: SafetyConfig, snapshot: dict, detections: list | None = None) -> dict:
    return {
        "frame_hash": snapshot.get("frame_hash", "n/a"),
        "source_uri": cfg.source_uri,
        "adapter_name": "m3pro_safety_bridge",
        "model_version": snapshot.get("model_version", "telemetry-rule"),
        "rule_version": cfg.rule_version,
        "captured_at": snapshot.get("captured_at")
        or datetime.now(timezone.utc).isoformat(),
        "telemetry_snapshot": snapshot,
        "detections": detections or [],
    }


def _event(cfg, rule_id, severity, confidence, summary, snapshot,
           human_review, detections=None) -> dict:
    return {
        "camera_id": cfg.camera_id,
        "rule_id": rule_id,
        "severity": severity,
        "confidence": round(float(confidence), 3),
        "human_review_required": bool(human_review),
        "summary": summary,
        "evidence": _evidence(cfg, snapshot, detections),
    }


def evaluate(snapshot: dict, cfg: SafetyConfig | None = None) -> list[dict]:
    """Telemetry snapshot -> list of SafetyEvent dicts (possibly empty).

    Recognised snapshot keys (all optional): battery_v, min_obstacle_m,
    base_speed_mps, deadman_fired (bool), person (dict {in_zone, confidence,
    box}), estop (bool).
    """
    cfg = cfg or SafetyConfig()
    out: list[dict] = []

    # --- battery ---
    bv = snapshot.get("battery_v")
    if bv is not None and bv == bv:  # not NaN
        if bv < cfg.batt_critical:
            out.append(_event(cfg, "m3pro.battery", "critical", 1.0,
                              f"battery critically low: {bv:.2f} V (servo brownout risk)",
                              snapshot, human_review=True))
        elif bv < cfg.batt_high:
            out.append(_event(cfg, "m3pro.battery", "high", 1.0,
                              f"battery low: {bv:.2f} V", snapshot, human_review=True))
        elif bv < cfg.batt_medium:
            out.append(_event(cfg, "m3pro.battery", "medium", 1.0,
                              f"battery getting low: {bv:.2f} V — charge soon",
                              snapshot, human_review=False))

    # --- stop distance (only while the base is moving) ---
    d = snapshot.get("min_obstacle_m")
    v = abs(float(snapshot.get("base_speed_mps", 0.0)))
    if d is not None and v > cfg.base_moving_mps:
        sev = conf = None
        if d < cfg.stop_critical_m:
            sev, conf = "critical", 1.0
        elif d < cfg.stop_high_m:
            sev, conf = "high", 0.9
        elif d < cfg.stop_medium_m:
            sev, conf = "medium", 0.7
        if sev:
            out.append(_event(cfg, "m3pro.stop_distance", sev, conf,
                              f"obstacle {d:.2f} m ahead while moving {v:.2f} m/s",
                              snapshot, human_review=(sev in ("high", "critical"))))

    # --- deadman / e-stop ---
    if snapshot.get("estop"):
        out.append(_event(cfg, "m3pro.estop", "high", 1.0,
                          "operator e-stop asserted — base commanded to zero",
                          snapshot, human_review=True))
    elif snapshot.get("deadman_fired"):
        out.append(_event(cfg, "m3pro.deadman", "high", 1.0,
                          "deadman: no fresh command within timeout — base zeroed",
                          snapshot, human_review=True))

    # --- person in workspace ---
    person = snapshot.get("person")
    if person and person.get("in_zone") and \
            float(person.get("confidence", 0.0)) >= cfg.person_conf_min:
        c = float(person["confidence"])
        out.append(_event(cfg, "m3pro.person_in_zone", "high", c,
                          f"person detected in the arm workspace (conf {c:.2f})",
                          snapshot, human_review=True,
                          detections=[{"label": "person", **person}]))
    return out
