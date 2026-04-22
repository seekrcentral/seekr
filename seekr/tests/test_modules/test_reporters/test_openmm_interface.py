import json
import os
import pytest

import seekr.prepare as prepare
import seekr.run as run
import seekr.tests.create_seekr_input as create_seekr_input
import seekr.modules.reporters.structures as rs
from seekr.modules.reporters.openmm_interface import (
    CV_value_reporter_openmm_interface,
    Anchor_reporter_openmm_interface,
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
