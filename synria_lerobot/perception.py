"""OpenCV board, colour-token and fixed-pad die perception with evidence gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import committed_bytes


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_board(image: Any, config: dict[str, Any]) -> tuple[Any, Any] | None:
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    if ids is None:
        return None
    labels = ids.flatten().tolist()
    expected = config["marker_ids"]
    if len(expected) != 4 or len(set(expected)) != 4 or any(labels.count(x) != 1 for x in expected):
        return None
    centers = np.array(
        [corners[labels.index(label)][0].mean(axis=0) for label in expected], dtype=np.float32
    )
    if not cv2.isContourConvex(centers) or cv2.contourArea(centers) < 100:
        return None
    target = np.asarray(config["corner_pixels"], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(centers, target)
    side = int(config["board_pixels"])
    warped = cv2.warpPerspective(image, transform, (side, side))
    return centers, warped


def detect_tokens(image: Any, config: dict[str, Any]) -> dict[str, str]:
    import cv2

    board = locate_board(image, config)
    if board is None:
        return {name: "unlocated" for name in config["squares"]}
    hsv = cv2.cvtColor(board[1], cv2.COLOR_BGR2HSV)
    result = {}
    for name, (x, y, width, height) in config["squares"].items():
        roi = hsv[y : y + height, x : x + width]
        if roi.shape[:2] != (height, width) or min(width, height) <= 0:
            raise ValueError("square outside calibrated board")
        found = []
        for colour, ranges in config["colours"].items():
            mask = np.zeros(roi.shape[:2], np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(roi, np.array(lower, np.uint8), np.array(upper, np.uint8))
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            area = max(stats[1:, cv2.CC_STAT_AREA], default=0) if count > 1 else 0
            if area >= width * height * config["token_min_fraction"]:
                found.append(colour)
        result[name] = found[0] if len(found) == 1 else "ambiguous" if found else "empty"
    return result


def read_die(image: Any, config: dict[str, Any]) -> int | None:
    import cv2

    x, y, width, height = config["die_roi"]
    roi = image[y : y + height, x : x + width]
    if roi.shape[:2] != (height, width):
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = [
        c for c in contours if width * height * 0.15 < cv2.contourArea(c) < width * height * 0.9
    ]
    if len(faces) != 1:
        return None
    bx, by, bw, bh = cv2.boundingRect(faces[0])
    if not 0.8 <= bw / bh <= 1.25 or bx <= 0 or by <= 0 or bx + bw >= width or by + bh >= height:
        return None
    face = gray[by + 4 : by + bh - 4, bx + 4 : bx + bw - 4]
    _, black = cv2.threshold(face, 90, 255, cv2.THRESH_BINARY_INV)
    pips, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted = []
    for pip in pips:
        area, perimeter = cv2.contourArea(pip), cv2.arcLength(pip, True)
        if perimeter and 0.002 < area / face.size < 0.06 and 4 * np.pi * area / perimeter**2 > 0.6:
            accepted.append(pip)
        elif area > face.size * 0.002:
            return None
    return len(accepted) if 1 <= len(accepted) <= 6 else None


def measure_accuracy(
    component: str,
    truth_csv: Path,
    config_path: Path,
    output: Path,
    *,
    data_kind: str,
    operator: str,
) -> dict[str, Any]:
    import cv2

    if component not in {"board", "tokens", "die"} or data_kind not in {"synthetic", "recorded"}:
        raise ValueError("invalid component or data kind")
    if not operator.strip():
        raise ValueError("truth-label operator required")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with truth_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["image"] for r in rows}) != len(rows):
        raise ValueError("nonempty unique image rows required")
    confusion: Counter[str] = Counter()
    inputs = []
    for row in rows:
        if not row["conditions"].strip():
            raise ValueError("conditions required for each image")
        path = truth_csv.parent / row["image"]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"unreadable image: {path}")
        truth = json.loads(row["truth"])
        pairs = []
        if component == "board":
            board = locate_board(image, config)
            expected = "located" if truth is not None else "missing"
            predicted = "located" if board is not None else "missing"
            if board is not None and truth is not None:
                error = np.linalg.norm(board[0] - np.asarray(truth), axis=1).max()
                if error > config["max_corner_error_px"]:
                    predicted = "mislocalized"
            pairs.append((expected, predicted))
        elif component == "tokens":
            if set(truth) != set(config["squares"]):
                raise ValueError("token truth must cover every configured square")
            predicted_tokens = detect_tokens(image, config)
            pairs.extend((truth[square], predicted_tokens[square]) for square in sorted(truth))
        else:
            pairs.append((str(truth), str(read_die(image, config))))
        for expected, predicted in pairs:
            confusion[json.dumps([expected, predicted])] += 1
        inputs.append({"image": row["image"], "sha256": file_hash(path)})
    total = sum(confusion.values())
    correct = sum(n for key, n in confusion.items() if len(set(json.loads(key))) == 1)
    report = {
        "component": component,
        "data_kind": data_kind,
        "operator": operator,
        "image_count": len(rows),
        "sample_count": total,
        "correct": correct,
        "accuracy": correct / total,
        "confusion_counts": dict(sorted(confusion.items())),
        "conditions": sorted({r["conditions"] for r in rows}),
        "inputs": inputs,
        "config_sha256": file_hash(config_path),
        "truth_sha256": file_hash(truth_csv),
        "status": "implemented, unmeasured",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def require_accuracy(
    component: str,
    report_path: Path,
    config_path: Path,
    repository: Path,
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    committed_bytes(report_path, repository)
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    threshold = config["thresholds"][component]
    confusion = report["confusion_counts"]
    total = sum(confusion.values())
    correct = sum(n for key, n in confusion.items() if len(set(json.loads(key))) == 1)
    if (
        report["component"] != component
        or report["config_sha256"] != file_hash(config_path)
        or report["data_kind"] != ("synthetic" if synthetic else "recorded")
        or not report["operator"]
        or not report["conditions"]
        or total <= 0
        or report["sample_count"] != total
        or report["correct"] != correct
        or report["accuracy"] != correct / total
        or report["image_count"] != len(report["inputs"])
        or report["image_count"] < threshold["min_images"]
        or correct / total < threshold["accuracy"]
    ):
        raise ValueError(f"{component} accuracy evidence does not meet configured gate")
    return report


class GatedPerception:
    def __init__(
        self,
        config_path: Path,
        reports: dict[str, Path],
        repository: Path,
        *,
        synthetic: bool = False,
    ) -> None:
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.synthetic = synthetic
        self.enabled = set(reports)
        self.report_hashes = {}
        for component, path in reports.items():
            require_accuracy(component, path, config_path, repository, synthetic=synthetic)
            self.report_hashes[component] = file_hash(path)

    def board(self, image: Any) -> tuple[Any, Any] | None:
        if "board" not in self.enabled:
            raise ValueError("board accuracy report required")
        return locate_board(image, self.config)

    def tokens(self, image: Any) -> dict[str, str]:
        if not {"board", "tokens"} <= self.enabled:
            raise ValueError("board and token accuracy reports required")
        return detect_tokens(image, self.config)

    def die(self, image: Any) -> int | None:
        if "die" not in self.enabled:
            raise ValueError("die accuracy report required")
        return read_die(image, self.config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["board", "tokens", "die"])
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-kind", choices=["synthetic", "recorded"], required=True)
    parser.add_argument("--operator", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            measure_accuracy(
                args.component,
                args.truth,
                args.config,
                args.output,
                data_kind=args.data_kind,
                operator=args.operator,
            )
        )
    )


if __name__ == "__main__":
    main()
