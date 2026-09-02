from __future__ import annotations
import json
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

import seekr.prepare as prepare
import seekr.run as run
import seekr.tests.create_seekr_input as create_seekr_input
import seekr.modules.reporters.structures as rs
from seekr.modules.reporters.openmm_interface import (
    CV_value_reporter_openmm_interface,
    Anchor_reporter_openmm_interface,
    Anchor_reporter,
    _openmm_box_to_array,
)

CV_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def test_position_reporter_pdb(tmp_path):
    """Changing position reporter extension to 'pdb' produces .pdb output files on disk."""
    seekr_input = create_seekr_input.create_toy_input(str(tmp_path))
    for r in seekr_input.stages[0].reporters:
        if hasattr(r, "extension"):
            r.extension = "pdb"
    model, model_xml_path = prepare.prepare(seekr_input)
    model.directory = os.path.abspath(os.path.dirname(model_xml_path))
    run.run(model, "equilibration")
    assert len(list(tmp_path.rglob("*position_reporter*.pdb"))) > 0


def test_cv_value_reporter_get_progress_from_disk(tmp_path):
    """get_progress() reads fraction of filled bins from the JSON on disk."""
    json_path = tmp_path / "cv_reporter0.json"
    data = {f"cv_value_{i}": {"index": i, "frame_index": (0 if i == 1 else -1)}
            for i in range(len(CV_VALUES) - 1)}
    json_path.write_text(json.dumps(data))

    iface = CV_value_reporter_openmm_interface(
        rs.CV_value_reporter(cv_values=CV_VALUES, interval=10), cv_index=0, swarm_id=0)
    iface._json_filename = str(json_path)
    assert iface.get_progress() == pytest.approx(1 / (len(CV_VALUES) - 1))


def test_anchor_reporter_get_progress_from_disk(tmp_path):
    """get_progress() reads fraction of visited anchors from disk JSON (interrupted run)."""
    json_path = tmp_path / "anchor_reporter0.json"
    json_path.write_text(json.dumps(
        {"anchor_0": {"index": 0, "filled": True},
         "anchor_1": {"index": 1, "filled": False}}))

    iface = Anchor_reporter_openmm_interface(
        rs.Anchor_reporter(name="a", interval=10), swarm_id=0)
    iface._json_filename = str(json_path)
    assert iface.get_progress() == pytest.approx(0.5)


def test_openmm_box_to_array_from_rows():
    box = np.eye(3) * 5.0
    arr = _openmm_box_to_array(box)
    assert arr is not None
    assert arr.shape == (3, 3)
    assert np.allclose(arr, box)
    assert _openmm_box_to_array(None) is None


def test_anchor_reporter_classify_uses_minimum_image():
    """Wrapped COM–COM is classified with min-image, not the raw Euclidean vector."""
    smi = MagicMock()
    smi._model = MagicMock()
    smi.find_current_anchor_index.side_effect = (
        lambda cv: 0 if cv[0] < 2.0 else 1)

    reporter = object.__new__(Anchor_reporter)
    reporter._sampling_method_interface = smi
    reporter._distance_masses = np.array([12.0, 12.0])
    reporter._distance_group1 = [0]
    reporter._distance_group2 = [1]

    pos_nm = np.array([[0.1, 0.0, 0.0], [9.1, 0.0, 0.0]])
    box = np.eye(3) * 10.0
    idx_raw, dist_raw = reporter._classify_positions_nm(pos_nm, None)
    idx_pbc, dist_pbc = reporter._classify_positions_nm(pos_nm, box)
    assert dist_raw == pytest.approx(9.0)
    assert dist_pbc == pytest.approx(1.0)
    assert idx_raw == 1
    assert idx_pbc == 0


def _minimal_anchor_reporter(n_anchors: int = 3) -> Anchor_reporter:
    """Bare reporter for finalize() tests; skips disk writes (no topology)."""
    from openmm import unit

    smi = MagicMock()
    smi._model = MagicMock()
    reporter = object.__new__(Anchor_reporter)
    reporter._sampling_method_interface = smi
    reporter._md_anchor_indices = list(range(n_anchors))
    reporter._buffered_positions = [None] * n_anchors
    reporter._buffered_box_vectors = [None] * n_anchors
    reporter._topology = None
    reporter._filepath = "unused.dcd"
    reporter._json_filename = "unused.json"
    reporter._extension = "dcd"
    reporter._dt = 1.0
    pos = unit.Quantity(np.array([[0.0, 0.0, 0.0]]), unit.nanometer)
    reporter._buffered_positions[1] = pos
    return reporter


def test_finalize_keeps_frames_when_distance_classify_unavailable():
    """Toy/external CVs never set distance masses; do not treat None as a mismatch."""
    reporter = _minimal_anchor_reporter()
    reporter.finalize()
    assert reporter._buffered_positions[1] is not None


def test_finalize_discards_distance_cv_slot_mismatch():
    """When a distance CV classifies the frame to another cell, drop the slot."""
    reporter = _minimal_anchor_reporter()
    reporter._distance_masses = np.array([12.0])
    reporter._distance_group1 = [0]
    reporter._distance_group2 = [0]
    reporter._sampling_method_interface.find_current_anchor_index.return_value = 0
    reporter.finalize()
    assert reporter._buffered_positions[1] is None

