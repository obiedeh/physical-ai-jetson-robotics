from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from synria_lerobot.perception import (  # noqa: E402
    detect_tokens,
    locate_board,
    measure_accuracy,
    read_die,
    require_accuracy,
)

CONFIG = Path("config/synria_perception.json")


def render_board(colour: str = "red", missing: bool = False) -> np.ndarray:
    image = np.full((600, 600, 3), 230, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for label, (x, y) in enumerate([(10, 10), (530, 10), (530, 530), (10, 530)]):
        if missing and label == 0:
            continue
        marker = cv2.aruco.generateImageMarker(dictionary, label, 60)
        image[y : y + 60, x : x + 60] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    colours = {
        "red": (0, 0, 220),
        "blue": (220, 0, 0),
        "green": (0, 200, 0),
        "yellow": (0, 220, 220),
    }
    cv2.circle(image, (180, 180), 25, colours[colour], -1)
    return image


def render_die(value: int, offset: int = 0) -> np.ndarray:
    image = np.full((240, 240, 3), 60, np.uint8)
    cv2.rectangle(image, (30 + offset, 30), (210 + offset, 210), (230, 230, 230), -1)
    points = {
        1: [(120, 120)],
        2: [(70, 70), (170, 170)],
        3: [(70, 70), (120, 120), (170, 170)],
        4: [(70, 70), (170, 70), (70, 170), (170, 170)],
        5: [(70, 70), (170, 70), (70, 170), (170, 170), (120, 120)],
        6: [(70, y) for y in (70, 120, 170)] + [(170, y) for y in (70, 120, 170)],
    }
    for x, y in points.get(value, []):
        cv2.circle(image, (x + offset, y), 12, (20, 20, 20), -1)
    return image


def synthetic_corpus(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list] = {"board": [], "tokens": [], "die": []}
    for index in range(24):
        colour = ["red", "blue", "green", "yellow"][index % 4]
        missing = index % 6 == 5
        name = f"board-{index}.png"
        image = render_board(colour, missing)
        # Vary brightness without changing geometry or truth labels.
        image = np.clip(image.astype(float) * (0.9 + (index % 3) * 0.05), 0, 255).astype(np.uint8)
        cv2.imwrite(str(root / name), image)
        truth = None if missing else [[39.5, 39.5], [559.5, 39.5], [559.5, 559.5], [39.5, 559.5]]
        conditions = f"synthetic; brightness group {index % 3}; missing marker={missing}"
        rows["board"].append((name, json.dumps(truth), conditions))
        rows["tokens"].append(
            (
                name,
                json.dumps(
                    {
                        "A": "unlocated" if missing else colour,
                        "B": "unlocated" if missing else "empty",
                    }
                ),
                conditions,
            )
        )
        die = index % 6 + 1
        die_name = f"die-{index}.png"
        cv2.imwrite(str(root / die_name), render_die(die, index % 3 - 1))
        rows["die"].append((die_name, json.dumps(die), f"synthetic; pad offset {index % 3 - 1}"))
    paths = {}
    for component, entries in rows.items():
        path = root / f"{component}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "truth", "conditions"])
            writer.writerows(entries)
        paths[component] = path
    return paths


def test_synthetic_accuracy_and_missing_markers(tmp_path: Path) -> None:
    corpus = synthetic_corpus(tmp_path)
    for component, truth in corpus.items():
        report = measure_accuracy(
            component,
            truth,
            CONFIG,
            tmp_path / f"{component}.json",
            data_kind="synthetic",
            operator="synthetic truth",
        )
        assert report["image_count"] == 24 and report["accuracy"] == 1
    config = json.loads(CONFIG.read_text())
    assert locate_board(render_board(missing=True), config) is None
    assert read_die(render_die(0), config) is None
    assert detect_tokens(render_board("blue"), config) == {"A": "blue", "B": "empty"}


def test_accuracy_gate_requires_committed_matching_physical_report(tmp_path: Path) -> None:
    corpus = synthetic_corpus(tmp_path / "images")
    report_path = tmp_path / "accuracy.json"
    measure_accuracy(
        "die", corpus["die"], CONFIG, report_path, data_kind="synthetic", operator="synthetic truth"
    )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(ValueError, match="committed"):
        require_accuracy("die", report_path, CONFIG, tmp_path, synthetic=True)
    subprocess.run(["git", "add", "accuracy.json"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test Operator",
            "-c",
            "user.email=operator@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Record accuracy",
        ],
        cwd=tmp_path,
        check=True,
    )
    require_accuracy("die", report_path, CONFIG, tmp_path, synthetic=True)
    with pytest.raises(ValueError, match="gate"):
        require_accuracy("die", report_path, CONFIG, tmp_path)
    changed = tmp_path / "config.json"
    changed.write_text(CONFIG.read_text().replace('"accuracy": 0.95', '"accuracy": 1.0'))
    with pytest.raises(ValueError, match="gate"):
        require_accuracy("die", report_path, changed, tmp_path, synthetic=True)


def test_wrong_truth_is_counted_as_confusion(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "die.png"), render_die(2))
    truth = tmp_path / "truth.csv"
    truth.write_text("image,truth,conditions\ndie.png,6,synthetic deliberate wrong label\n")
    report = measure_accuracy(
        "die", truth, CONFIG, tmp_path / "report.json", data_kind="synthetic", operator="test"
    )
    assert report["accuracy"] == 0 and report["confusion_counts"] == {'["6", "2"]': 1}
