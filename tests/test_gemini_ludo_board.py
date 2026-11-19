from physical_ai_lab.gemini_robotics.board_bench import parse_tokens, summarize, to_cells
from physical_ai_lab.gemini_robotics.ludo_board import (
    OverheadCamera,
    cell_of,
    occupied,
    sample_states,
    score_state,
)


def test_states_are_reproducible_and_complete():
    a, b = sample_states(5, 7), sample_states(5, 7)
    assert a == b
    assert all(len(s["tokens"]) == 16 for s in a)
    assert cell_of("red", ("track", 0)) == (1, 6)
    assert cell_of("green", ("done", 2)) == (8, 7)


def test_camera_round_trip_covers_every_cell():
    cam = OverheadCamera((672, 672))
    for col in range(15):
        for row in range(15):
            assert cam.px_to_cell(*cam.cell_to_px(col, row)) == (col, row)
    assert cam.px_to_cell(2.0, 2.0) is None


def test_points_map_to_cells_and_score():
    cam = OverheadCamera((672, 672))
    x, y = cam.cell_to_px(1, 6)
    point = f"[{y / 672 * 1000:f}, {x / 672 * 1000:f}]"
    text = (
        '```json\n[{"point": ' + point + ', "label": "Red token"}, '
        '{"point": [1, 1], "label": "blue"}]\n```'
    )
    cells, off = to_cells(parse_tokens(text), cam)
    assert cells == {("red", 1, 6)} and off == 1
    truth = occupied([{"color": "red", "cell": [1, 6]}, {"color": "blue", "cell": [8, 2]}])
    s = score_state(truth, cells | {("green", 8, 2)})
    assert (s["n_hit"], s["wrong_color"], s["exact"]) == (1, 1, False)
    shared = {("red", 1, 6), ("blue", 1, 6)}
    assert score_state(shared, shared)["wrong_color"] == 0
    assert parse_tokens("no json") is None


def test_summary_counts_only_answered_states():
    rows = [
        {
            "ok": True,
            "seconds": 2.0,
            "n_truth": 16,
            "n_pred": 15,
            "n_hit": 15,
            "exact": False,
            "wrong_color": 0,
            "off_board": 0,
        },
        {"ok": False, "seconds": 0.0},
    ]
    out = summarize("m", rows, 2)
    assert (out["n_answered"], out["states_exact"], out["recall"], out["precision"]) == (
        1,
        0,
        0.938,
        1.0,
    )
