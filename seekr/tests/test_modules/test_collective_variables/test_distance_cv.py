from __future__ import annotations
"""
test_distance_cv.py

Unit tests for seekr/modules/collective_variables/distance_cv.py.
"""

import numpy as np
import mdtraj
import pytest
from unittest.mock import MagicMock, patch

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
    mock_engine.hydrogen_mass = None
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
    mock_engine.hydrogen_mass = None
    mock_engine.get_atom_indices_by_selection_name.side_effect = (
        lambda name: [0] if name == "a" else [1]
    )
    mock_model = MagicMock()
    mock_model.get_engine_by_scale_type.return_value = mock_engine

    result = cv.get_mdtraj_cv_value(mock_model, traj, frame_index=0)
    assert result == pytest.approx(0.0, abs=1e-5)


def _ch_plus_dummy_topology():
    """C–H bonded pair plus a dummy carbon; C at origin, H at 0.1 nm on x."""
    top = mdtraj.Topology()
    chain = top.add_chain()
    res = top.add_residue("DUM", chain)
    atom_c = top.add_atom("C", mdtraj.element.carbon, res)
    atom_h = top.add_atom("H", mdtraj.element.hydrogen, res)
    atom_d = top.add_atom("C", mdtraj.element.carbon, res)
    top.add_bond(atom_c, atom_h)
    xyz = np.array(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=np.float32)
    traj = mdtraj.Trajectory(xyz, top)
    return traj


def test_get_mdtraj_cv_value_hmr_shifts_com():
    """HMR (hydrogen_mass=3.0) moves the C–H COM toward H vs element masses."""
    cv = distance_cv.Distance_CV(index=0, group1_name="ch", group2_name="dummy")
    traj = _ch_plus_dummy_topology()

    mC = mdtraj.element.carbon.mass
    mH = mdtraj.element.hydrogen.mass
    com_elem = (mC * 0.0 + mH * 0.1) / (mC + mH)
    dist_elem = 1.0 - com_elem
    mH_hmr = 3.0
    mC_hmr = mC - (mH_hmr - mH)
    com_hmr = (mC_hmr * 0.0 + mH_hmr * 0.1) / (mC_hmr + mH_hmr)
    dist_hmr = 1.0 - com_hmr
    assert dist_hmr != pytest.approx(dist_elem, abs=1e-6)

    def _engine(hydrogen_mass):
        mock_engine = MagicMock()
        mock_engine.hydrogen_mass = hydrogen_mass
        mock_engine.get_atom_indices_by_selection_name.side_effect = (
            lambda name: [0, 1] if name == "ch" else [2]
        )
        mock_model = MagicMock()
        mock_model.get_engine_by_scale_type.return_value = mock_engine
        return mock_model

    result_elem = cv.get_mdtraj_cv_value(_engine(None), traj, frame_index=0)
    result_hmr = cv.get_mdtraj_cv_value(_engine(3.0), traj, frame_index=0)
    assert result_elem == pytest.approx(dist_elem, abs=1e-5)
    assert result_hmr == pytest.approx(dist_hmr, abs=1e-5)


def test_get_mdtraj_cv_value_minimum_image():
    """Wrapped coordinates use min-image COM–COM distance, not the raw vector."""
    cv = distance_cv.Distance_CV(index=0, group1_name="a", group2_name="b")

    top = mdtraj.Topology()
    chain = top.add_chain()
    res = top.add_residue("DUM", chain)
    top.add_atom("C", mdtraj.element.carbon, res)
    top.add_atom("C", mdtraj.element.carbon, res)
    xyz = np.array([[[0.1, 0.0, 0.0], [9.1, 0.0, 0.0]]], dtype=np.float32)
    traj = mdtraj.Trajectory(
        xyz, top, unitcell_lengths=np.array([[10.0, 10.0, 10.0]]),
        unitcell_angles=np.array([[90.0, 90.0, 90.0]]))

    mock_engine = MagicMock()
    mock_engine.hydrogen_mass = None
    mock_engine.get_atom_indices_by_selection_name.side_effect = (
        lambda name: [0] if name == "a" else [1]
    )
    mock_model = MagicMock()
    mock_model.get_engine_by_scale_type.return_value = mock_engine

    result = cv.get_mdtraj_cv_value(mock_model, traj, frame_index=0)
    raw = float(np.linalg.norm(xyz[0, 1] - xyz[0, 0]))
    assert raw == pytest.approx(9.0, abs=1e-5)
    assert result == pytest.approx(1.0, abs=1e-5)
    assert result < raw


def test_get_mdtraj_cv_value_hmr_warns_without_bonds():
    """HMR requested on a topology with no H–heavy bonds logs a warning."""
    cv = distance_cv.Distance_CV(index=0, group1_name="a", group2_name="b")
    top = mdtraj.Topology()
    chain = top.add_chain()
    res = top.add_residue("DUM", chain)
    top.add_atom("C", mdtraj.element.carbon, res)
    top.add_atom("H", mdtraj.element.hydrogen, res)
    xyz = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    traj = mdtraj.Trajectory(xyz, top)

    mock_engine = MagicMock()
    mock_engine.hydrogen_mass = 3.0
    mock_engine.get_atom_indices_by_selection_name.side_effect = (
        lambda name: [0] if name == "a" else [1]
    )
    mock_model = MagicMock()
    mock_model.get_engine_by_scale_type.return_value = mock_engine
    mock_model.using_toy.return_value = True

    with patch.object(distance_cv.logger, "warning") as warn:
        result = cv.get_mdtraj_cv_value(mock_model, traj, frame_index=0)
    assert result == pytest.approx(1.0, abs=1e-5)
    warn.assert_called()
    assert "no H–heavy bonds" in warn.call_args[0][0]


def test_get_mdtraj_cv_value_provided_masses_without_bonds():
    """Caller-supplied OpenMM masses shift COM even with no H–heavy bonds."""
    cv = distance_cv.Distance_CV(index=0, group1_name="lig", group2_name="site")
    top = mdtraj.Topology()
    chain = top.add_chain()
    res = top.add_residue("INH", chain)
    top.add_atom("C", mdtraj.element.carbon, res)
    top.add_atom("O", mdtraj.element.oxygen, res)
    top.add_atom("C", mdtraj.element.carbon, res)
    xyz = np.array(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=np.float32)
    traj = mdtraj.Trajectory(xyz, top)

    mock_engine = MagicMock()
    mock_engine.hydrogen_mass = 3.0
    mock_engine.get_atom_indices_by_selection_name.side_effect = (
        lambda name: [0, 1] if name == "lig" else [2]
    )
    mock_model = MagicMock()
    mock_model.get_engine_by_scale_type.return_value = mock_engine
    mock_model.using_toy.return_value = True

    mC = mdtraj.element.carbon.mass
    mO = mdtraj.element.oxygen.mass
    dist_elem = 1.0 - (mC * 0.0 + mO * 0.1) / (mC + mO)
    fallback = cv.get_mdtraj_cv_value(mock_model, traj, frame_index=0)
    assert fallback == pytest.approx(dist_elem, abs=1e-5)

    omm_masses = np.array([6.034, 16.0, 12.01])
    dist_omm = 1.0 - (6.034 * 0.0 + 16.0 * 0.1) / (6.034 + 16.0)
    result = cv.get_mdtraj_cv_value(
        mock_model, traj, frame_index=0, masses=omm_masses)
    assert result == pytest.approx(dist_omm, abs=1e-5)
    assert result != pytest.approx(fallback, abs=1e-6)

