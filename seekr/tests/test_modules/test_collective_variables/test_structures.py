from __future__ import annotations
"""
test_structures.py

Unit tests for seekr/modules/collective_variables/structures.py.
"""

import numpy as np
import mdtraj
import pytest
from unittest.mock import MagicMock

import seekr.modules.collective_variables.distance_cv as distance_cv
import seekr.modules.collective_variables.structures as cv_structures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_atom_traj(pos0, pos1):
    """1-frame, 2-atom trajectory."""
    top = mdtraj.Topology()
    chain = top.add_chain()
    res = top.add_residue("DUM", chain)
    top.add_atom("C", mdtraj.element.carbon, res)
    top.add_atom("C", mdtraj.element.carbon, res)
    xyz = np.array([[pos0, pos1]], dtype=np.float32)
    return mdtraj.Trajectory(xyz, top)


def _mock_model_with_distance_cv():
    """Mock model with one Distance_CV; atom 0 = group1, atom 1 = group2."""
    cv = distance_cv.Distance_CV(index=0, group1_name="lig", group2_name="rec")
    mock_engine = MagicMock()
    mock_engine.get_atom_indices_by_selection_name.side_effect = (
        lambda name: [0] if name == "lig" else [1]
    )
    mock_model = MagicMock()
    mock_model.collective_variables = [cv]
    mock_model.get_engine_by_scale_type.return_value = mock_engine
    return mock_model


# ---------------------------------------------------------------------------
# get_mdtraj_cv_value
# ---------------------------------------------------------------------------

def test_get_mdtraj_cv_value_single_cv():
    """Returns a list with one entry equal to the distance between the two atoms."""
    model = _mock_model_with_distance_cv()
    traj = _make_two_atom_traj([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    values = cv_structures.get_mdtraj_cv_value(model, traj, frame_index=0)
    assert len(values) == 1
    assert values[0] == pytest.approx(1.0, abs=1e-5)


def test_get_mdtraj_cv_value_zero_distance():
    """Distance CV returns 0 when both groups coincide."""
    model = _mock_model_with_distance_cv()
    traj = _make_two_atom_traj([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    values = cv_structures.get_mdtraj_cv_value(model, traj, frame_index=0)
    assert values[0] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# check_value_within_boundary
# Logic: me_total = (me - cv)^2, neighbor_total = (neighbor - cv)^2
#        True  if me_total <= neighbor_total  (cv closer to me milestone)
#        False if me_total >  neighbor_total  (cv closer to neighbor)
# ---------------------------------------------------------------------------

def test_check_value_within_boundary_inside():
    """cv=1, me=2, neighbor=4 → me_d=1 < nb_d=9 → True."""
    milestone_vars = {"me_0": 2.0, "neighbor_0": 4.0}
    assert cv_structures.check_value_within_boundary([1.0], milestone_vars) is True


def test_check_value_within_boundary_outside():
    """cv=3.5, me=2, neighbor=4 → me_d=2.25 > nb_d=0.25 → False."""
    milestone_vars = {"me_0": 2.0, "neighbor_0": 4.0}
    assert cv_structures.check_value_within_boundary([3.5], milestone_vars) is False


def test_check_value_within_boundary_on_boundary():
    """cv=3, me=2, neighbor=4 → me_d=1, nb_d=1, distance=0 → True (not strictly >0)."""
    milestone_vars = {"me_0": 2.0, "neighbor_0": 4.0}
    assert cv_structures.check_value_within_boundary([3.0], milestone_vars) is True


def test_check_value_within_boundary_tolerance():
    """cv=3.5, distance=2.0; tolerance=3.0 → (2.0-3.0)=-1.0, not >0 → True."""
    milestone_vars = {"me_0": 2.0, "neighbor_0": 4.0}
    assert cv_structures.check_value_within_boundary(
        [3.5], milestone_vars, tolerance=3.0) is True


# ---------------------------------------------------------------------------
# check_mdtraj_within_boundary
# ---------------------------------------------------------------------------

def test_check_mdtraj_within_boundary_inside():
    """Atom 1 nm away, me=2, neighbor=4 → inside → True."""
    model = _mock_model_with_distance_cv()
    traj = _make_two_atom_traj([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    milestone_vars = {"me_0": 2.0, "neighbor_0": 4.0}
    assert cv_structures.check_mdtraj_within_boundary(
        model, traj, milestone_vars, frame_index=0) is True


def test_check_mdtraj_within_boundary_outside():
    """Atom 3.5 nm away, me=2, neighbor=4 → outside → False."""
    model = _mock_model_with_distance_cv()
    traj = _make_two_atom_traj([0.0, 0.0, 0.0], [3.5, 0.0, 0.0])
    milestone_vars = {"me_0": 2.0, "neighbor_0": 4.0}
    assert cv_structures.check_mdtraj_within_boundary(
        model, traj, milestone_vars, frame_index=0) is False
