"""
test_distance_cv.py

Unit tests for seekr/modules/collective_variables/distance_cv.py.
"""

import numpy as np
import mdtraj
import pytest
from unittest.mock import MagicMock

import seekr.modules.collective_variables.distance_cv as distance_cv


def test_get_atom_group_names():
    """get_atom_group_names returns both group names in order."""
    cv = distance_cv.Distance_CV(index=0, group1_name="host", group2_name="guest")
    assert cv.get_atom_group_names() == ["host", "guest"]


def test_get_mdtraj_cv_value():
    """get_mdtraj_cv_value returns the correct COM-COM distance."""
    cv = distance_cv.Distance_CV(index=0, group1_name="lig", group2_name="rec")

    # Two-atom trajectory: atom 0 at origin, atom 1 at (1, 0, 0) nm.
    # Single-atom groups bypass mdtraj mass-weighting.
    top = mdtraj.Topology()
    chain = top.add_chain()
    res = top.add_residue("DUM", chain)
    top.add_atom("C", mdtraj.element.carbon, res)
    top.add_atom("C", mdtraj.element.carbon, res)
    xyz = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    traj = mdtraj.Trajectory(xyz, top)

    mock_engine = MagicMock()
    mock_engine.get_atom_indices_by_selection_name.side_effect = (
        lambda name: [0] if name == "lig" else [1]
    )
    mock_model = MagicMock()
    mock_model.get_engine_by_scale_type.return_value = mock_engine

    result = cv.get_mdtraj_cv_value(mock_model, traj, frame_index=0)
    assert result == pytest.approx(1.0, abs=1e-5)


def test_get_mdtraj_cv_value_zero_distance():
    """get_mdtraj_cv_value returns 0 when both groups are at the same position."""
    cv = distance_cv.Distance_CV(index=0, group1_name="a", group2_name="b")

    top = mdtraj.Topology()
    chain = top.add_chain()
    res = top.add_residue("DUM", chain)
    top.add_atom("C", mdtraj.element.carbon, res)
    top.add_atom("C", mdtraj.element.carbon, res)
    xyz = np.array([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]], dtype=np.float32)
    traj = mdtraj.Trajectory(xyz, top)

    mock_engine = MagicMock()
    mock_engine.get_atom_indices_by_selection_name.side_effect = (
        lambda name: [0] if name == "a" else [1]
    )
    mock_model = MagicMock()
    mock_model.get_engine_by_scale_type.return_value = mock_engine

    result = cv.get_mdtraj_cv_value(mock_model, traj, frame_index=0)
    assert result == pytest.approx(0.0, abs=1e-5)
