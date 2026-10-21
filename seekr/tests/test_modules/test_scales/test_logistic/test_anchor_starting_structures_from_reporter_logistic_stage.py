from __future__ import annotations
"""
Tests for merge-then-validate assign from Anchor_reporter restart files.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from seekr.modules.scales.logistic import (
    anchor_starting_structures_from_reporter_logistic_stage as assign_stage,
)


class DummyTraj:
    """Minimal traj: only length is used by the merge helper."""

    def __init__(self, n_frames: int, label: str = ""):
        self._n = n_frames
        self.label = label
        self.xyz = np.zeros((n_frames, 1, 3))

    def __len__(self) -> int:
        return self._n


def _filled_json(*indices: int) -> dict:
    return {
        f"anchor_{i}": {"index": i, "filled": True}
        for i in indices
    }


def test_merge_anchor_reporter_frames_last_frame_wins():
    traj1 = DummyTraj(4, "file1")
    traj2 = DummyTraj(4, "file2")
    json1 = _filled_json(0, 1, 2, 3)
    json2 = _filled_json(0, 1, 2, 3)
    merged = assign_stage._merge_anchor_reporter_frames(
        [(traj1, json1), (traj2, json2)], mode="save_last_frame")
    assert set(merged) == {0, 1, 2, 3}
    for anchor_idx in range(4):
        traj, fn = merged[anchor_idx]
        assert traj is traj2
        assert fn == anchor_idx


def test_merge_anchor_reporter_frames_first_frame_wins():
    traj1 = DummyTraj(2, "file1")
    traj2 = DummyTraj(4, "file2")
    json1 = _filled_json(0, 1)
    json2 = _filled_json(0, 1, 2, 3)
    merged = assign_stage._merge_anchor_reporter_frames(
        [(traj1, json1), (traj2, json2)], mode="save_first_frame")
    assert merged[0][0] is traj1
    assert merged[1][0] is traj1
    assert merged[2][0] is traj2
    assert merged[3][0] is traj2


def test_process_anchor_reporter_validates_winners_only(tmp_path):
    """A stale first-file frame that would fail Voronoi is ignored if last-frame-wins."""
    traj1 = DummyTraj(4, "stale")
    traj2 = DummyTraj(4, "complete")
    json1 = _filled_json(0, 1, 2, 3)
    json2 = _filled_json(0, 1, 2, 3)

    validated_frames: list[tuple] = []

    def fake_compute(model, traj, fn):
        validated_frames.append((traj.label, fn))
        # File 1's claimed anchor 3 would map to the wrong cell.
        if traj is traj1 and fn == 3:
            return [0.145]
        return [float(fn)]

    def fake_find(model, cv_values):
        if cv_values == [0.145]:
            return 2
        return int(cv_values[0])

    anchors = []
    for i in range(4):
        a = MagicMock()
        a.index = i
        a.directory = f"anchor_{i}"
        a.scale_types = ["molecular_dynamics"]
        a.bulk_anchor = False
        anchors.append(a)

    model = MagicMock()
    model.using_toy.return_value = True
    model.directory = str(tmp_path)
    model.anchors = anchors

    stage = MagicMock()
    stage.get_directory_name.return_value = "stage_9_assign"
    stage.get_output_set_name.return_value = "main"
    stage.output_positions_basename_noext = "output_positions"

    reporter = MagicMock()
    reporter.mode = "save_last_frame"

    with patch.object(
            assign_stage, "_compute_cv_values_from_frame", fake_compute), \
         patch.object(
            assign_stage, "_find_anchor_index_from_cv_values", fake_find):
        assign_stage._process_anchor_reporter(
            model, stage, [(traj1, json1), (traj2, json2)], reporter)

    assert ("stale", 3) not in validated_frames
    assert ("complete", 3) in validated_frames
    for i in range(4):
        out = (tmp_path / "anchor_sets" / "main" / f"anchor_{i}"
               / "stage_9_assign" / "output_positions.swarm_0.txt")
        assert out.is_file()


def test_process_anchor_reporter_raises_on_winner_mismatch():
    """The winning frame still hard-fails if it is not in the claimed cell."""
    traj = DummyTraj(1, "bad")
    json_data = _filled_json(3)

    def fake_compute(model, traj, fn):
        return [0.145]

    def fake_find(model, cv_values):
        return 2

    anchor = MagicMock()
    anchor.index = 3
    anchor.directory = "anchor_3"
    anchor.scale_types = ["molecular_dynamics"]
    anchor.bulk_anchor = False

    model = MagicMock()
    model.using_toy.return_value = True
    model.directory = "/tmp"
    model.anchors = [anchor]

    stage = MagicMock()
    stage.get_directory_name.return_value = "stage_9_assign"
    stage.get_output_set_name.return_value = "main"
    stage.output_positions_basename_noext = "output_positions"

    reporter = MagicMock()
    reporter.mode = "save_last_frame"

    with patch.object(
            assign_stage, "_compute_cv_values_from_frame", fake_compute), \
         patch.object(
            assign_stage, "_find_anchor_index_from_cv_values", fake_find):
        with pytest.raises(Exception, match="labeled as anchor 3"):
            assign_stage._process_anchor_reporter(
                model, stage, [(traj, json_data)], reporter)
