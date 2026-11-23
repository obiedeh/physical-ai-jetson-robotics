"""Tests for joining simulation datasets' per-episode logs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "merge_sim_datasets",
    Path(__file__).resolve().parents[1] / "scripts" / "merge_sim_datasets.py")
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


def test_extra_rows_are_renumbered_and_stills_repointed(tmp_path):
    base = [{"episode_index": 0, "final_still": "/a/final_stills/episode_000000_1.jpg"}]
    extra = [{"episode_index": 0, "final_still": "/b/final_stills/episode_000000_9.jpg"},
             {"episode_index": 1}]
    rows = merge.merge_rows(base, extra, 50, tmp_path)
    assert [row["episode_index"] for row in rows] == [0, 50, 51]
    assert rows[0]["final_still"] == str(tmp_path / "episode_000000_1.jpg")
    assert rows[1]["final_still"] == str(tmp_path / "episode_000050_9.jpg")
    assert extra[0]["episode_index"] == 0  # the sources are not changed


def test_exclusions_keep_their_paths_untouched():
    rows = merge.merge_rows([{"episode_index": 5, "note": "x"}],
                            [{"episode_index": 3, "note": "y"}], 50, None)
    assert [row["episode_index"] for row in rows] == [5, 53]


def test_rows_without_an_integer_index_are_refused():
    with pytest.raises(ValueError):
        merge.merge_rows([{"episode_index": "0"}], [], 1, None)


def test_still_names_must_carry_an_episode_number():
    assert merge.renumber_still("episode_000007_abc.jpg", 3) == "episode_000010_abc.jpg"
    with pytest.raises(ValueError):
        merge.renumber_still("still.jpg", 1)
