from physical_ai_lab.gemini_robotics.point_bench import error_mm, parse_point, summarize, to_pixels


def test_parse_point_json_and_fenced():
    assert parse_point('[{"point": [280, 283], "label": "red cup"}]') == (280.0, 283.0)
    assert parse_point('```json\n{"point": [10.5, 20]}\n```') == (10.5, 20.0)
    assert parse_point("no point here") is None


def test_pixels_and_error():
    px = to_pixels((500.0, 250.0), (224, 224))
    assert px == (56.0, 112.0)
    assert error_mm((56.0, 112.0), (53.0, 108.0), 4.0) == 20.0


def test_summarize_counts_unanswered():
    rows = [
        {"ok": True, "err_mm": 2.0, "seconds": 1.0},
        {"ok": True, "err_mm": 30.0, "seconds": 3.0},
        {"ok": False, "seconds": 5.0},
    ]
    s = summarize("m", rows, 3)
    assert s["n_answered"] == 2 and s["within_20mm"] == 1 and s["median_mm"] == 16.0
