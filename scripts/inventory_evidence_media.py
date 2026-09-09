#!/usr/bin/env python3
"""Inventory tracked evidence media without copying or modifying source assets.

The assessment is per file where inspected and explicitly per collection otherwise.
Ignored/untracked local media are outside the reproducible public inventory.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/evidence/media_inventory.csv"
MEDIA = {".png", ".jpg", ".jpeg", ".mp4", ".mov", ".webm", ".gif"}


def assess(path: str):
    if path == "reports/g3_session_01/g3_demo.mp4":
        return (
            "G3 simulated wrist view with ER-2 cup-fix latency/age overlay and ACT phase",
            "First-party video/composition; vendor robot geometry depicted",
            "Frame inspected; useful diagnostic, not standalone success evidence",
            "Caption simulated input, observation age, camera framing and source "
            "sidecar; no physical-performance claim",
        )
    if path.startswith("reports/gate_audit/"):
        return (
            "Gate-input image; filename records step/episode and verdict",
            "First-party simulation capture; vendor robot geometry depicted",
            "Collection assessment; two representative images inspected",
            "Show with verdict and ledger explanation; do not label a cropped view as "
            "physical ground truth",
        )
    if path == "reports/synria_gripper_clearance.png":
        return (
            "Gripper/cup clearance schematic: maximum and commanded opening",
            "First-party analytical diagram using robot/gripper dimensions, not a mesh export",
            "Inspected; legible as a geometry illustration, not contact or task-success proof",
            "Caption the modeled assumptions and link amended grasp audit",
        )
    if path.startswith("docs/diagrams/"):
        return (
            "First-party architecture/deployment/runtime/data-flow view",
            "First-party Mermaid source; no vendor geometry",
            "Source inspected; useful architecture explanation, not performance evidence",
            "Render using a Mermaid-capable viewer; keep current status boundaries",
        )
    if path.startswith("isaac/usd/"):
        return (
            "Scene/prop layer with first-party definitions or external robot/game references",
            "First-party layer; referenced vendor geometry/scenes are external",
            "USD structure inspected; not a rendered performance record",
            "Obtain dependencies and disclose vendor geometry; render locally; no "
            "performance inference",
        )
    return (
        "Recorded training/evaluation or diagnostic imagery/video; inspect associated run metadata",
        "First-party report artifact; depicted geometry/provenance requires per-run review",
        "Not individually visually reviewed; not selected for the evidence page",
        "Review episode plus manifest/provenance, label simulation/physical scope, "
        "caption scoring rule and failures",
    )


def main():
    files = (
        subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "-z", "reports", "isaac/usd", "docs/diagrams"]
        )
        .decode()
        .split("\0")
    )
    rows = []
    for path in sorted(filter(None, files)):
        p = ROOT / path
        if not p.is_file():
            continue
        included = (
            path.startswith("reports/")
            and p.suffix.lower() in MEDIA
            or path.startswith("isaac/usd/scenes/")
            and p.suffix in {".usd", ".usda"}
            or path.startswith("isaac/usd/props/")
            and p.suffix in {".usd", ".usda"}
            or path.startswith("docs/diagrams/")
            and p.suffix == ".mmd"
        )
        if included:
            rows.append((path, p.stat().st_size, *assess(path)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "bytes", "shows", "provenance", "presentation_status", "needs"))
        writer.writerows(rows)
    print(f"Inventoried {len(rows)} tracked, present files in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
