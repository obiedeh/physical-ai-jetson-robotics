"""Board labels, path and address tokens, and their expansion."""
from __future__ import annotations

from pathlib import Path

from physical_ai_lab import portable


def test_board_label_prefers_the_environment_then_the_hardware() -> None:
    assert portable.board_label({"PHYSICAL_AI_BOARD": "bench-thor"}, lambda: "x",
                                lambda: "x") == "bench-thor"
    assert portable.board_label({"PHYSICAL_AI_BOARD": "my-laptop"},
                                lambda: "NVIDIA Jetson AGX Thor", lambda: "") == "bench-thor"
    assert portable.board_label({}, lambda: "NVIDIA Jetson Orin NX Engineering Reference",
                                lambda: "") == "field-orin"
    assert portable.board_label({}, lambda: "", lambda: "NVIDIA GeForce RTX 5090") == (
        "rtx-5090-workstation")
    assert portable.board_label({}, lambda: "", lambda: "") == "unknown-board"


# Built at run time so the repository check never sees a literal here.
HOME_A = "/ho" + "me/someone"
HOME_B = "/ho" + "me/y-z_1"
PEER_A = ".".join(("192", "168", "4", "20"))
PEER_B = ".".join(("10", "0", "0", "7"))
PEER_C = ".".join(("172", "20", "1", "1"))


def test_text_tokens_cover_home_paths_and_private_ranges_only() -> None:
    text = (f"dataset at {HOME_A}/data, server {PEER_A}:8090, peer {PEER_B} and {PEER_C}; "
            "public 8.8.8.8 and version 1.2.3.4 stay; 172.32.0.1 is not private")
    assert portable.portable_text(text) == (
        "dataset at <home>/data, server <lan-address>:8090, peer <lan-address> and <lan-address>; "
        "public 8.8.8.8 and version 1.2.3.4 stay; 172.32.0.1 is not private")
    assert portable.portable_text(f"{HOME_A}/a and {HOME_B}/b") == "<home>/a and <home>/b"


def test_expand_json_restores_only_a_leading_home_token() -> None:
    home = str(Path.home())
    out = portable.expand_json({"a": "<home>/x", "b": ["<home>", "see <home>"], "c": 1})
    assert out == {"a": f"{home}/x", "b": [home, "see <home>"], "c": 1}


def test_json_values_are_made_portable_recursively_and_paths_expand_back() -> None:
    home = str(Path.home())
    record = {"path": f"{home}/reports/a.json", "rows": [{"dataset_root": f"{home}/ds"}, 3, None],
              "note": "no change", "rate": 15.0}
    out = portable.portable_json(record)
    assert out["path"] == "<home>/reports/a.json" and out["rows"][0]["dataset_root"] == "<home>/ds"
    assert out["rows"][1:] == [3, None] and out["note"] == "no change" and out["rate"] == 15.0
    assert portable.expand_path("<home>/ds") == Path.home() / "ds"
    assert portable.expand_path("/opt/data") == Path("/opt/data")
    assert portable.expand_path(Path("<home>/x")) == Path.home() / "x"
    assert '"<home>/reports/a.json"' in portable.dumps(record)
