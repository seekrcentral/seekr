"""
test_external_cv.py

Unit tests for seekr/modules/collective_variables/external_cv.py.
"""

import numpy as np
import mdtraj
import pytest

import seekr.modules.collective_variables.external_cv as external_cv


def _make_cv(expression, group_names=None):
    if group_names is None:
        group_names = ["particle0"]
    return external_cv.External_CV(
        index=0, group_names=group_names, expression=expression)


# ---------------------------------------------------------------------------
# get_atom_group_names
# ---------------------------------------------------------------------------

def test_get_atom_group_names():
    cv = _make_cv("x1", group_names=["a", "b", "c"])
    assert cv.get_atom_group_names() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# get_cv_value
# ---------------------------------------------------------------------------

def test_get_cv_value_single_coord():
    """Expression 'y1' returns the y-coordinate of particle 0 (real toy scenario)."""
    cv = _make_cv("y1")
    result = cv.get_cv_value([[0.0, 3.5, 0.0]])
    assert result == pytest.approx(3.5, abs=1e-7)


def test_get_cv_value_power_operator():
    """Expression 'x1^2' exercises the ^ -> ** conversion."""
    cv = _make_cv("x1^2")
    result = cv.get_cv_value([[3.0, 0.0, 0.0]])
    assert result == pytest.approx(9.0, abs=1e-7)


def test_get_cv_value_two_groups():
    """Expression 'x2 - x1' exercises multi-group coordinate binding."""
    cv = _make_cv("x2 - x1", group_names=["g1", "g2"])
    result = cv.get_cv_value([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    assert result == pytest.approx(5.0, abs=1e-7)


# ---------------------------------------------------------------------------
# get_mdtraj_cv_value
# ---------------------------------------------------------------------------

def test_get_mdtraj_cv_value():
    """model is unused; positions are taken directly from traj.xyz."""
    cv = _make_cv("y1")

    top = mdtraj.Topology()
    chain = top.add_chain()
    res = top.add_residue("DUM", chain)
    top.add_atom("C", mdtraj.element.carbon, res)
    xyz = np.array([[[0.0, 2.0, 0.0]]], dtype=np.float32)
    traj = mdtraj.Trajectory(xyz, top)

    result = cv.get_mdtraj_cv_value(None, traj, frame_index=0)
    assert result == pytest.approx(2.0, abs=1e-5)
