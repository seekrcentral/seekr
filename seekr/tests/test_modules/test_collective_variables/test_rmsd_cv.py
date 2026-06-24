from __future__ import annotations
"""
test_rmsd_cv.py

Unit tests for seekr/modules/collective_variables/rmsd_cv.py.
"""

import numpy as np
import mdtraj
import pytest
from unittest.mock import MagicMock

import seekr.modules.collective_variables.rmsd_cv as rmsd_cv


def _make_cv(group_rmsd="lig", group_align="rec"):
    return rmsd_cv.RMSD_CV(
        index=0,
        directory="collective_variable_0",
        group_rmsd_name=group_rmsd,
        group_align_name=group_align,
        ref_structure="ref.pdb",
    )


def _make_two_atom_traj(pos0, pos1):
    top = mdtraj.Topology()
    chain = top.add_chain()
    res = top.add_residue("DUM", chain)
    top.add_atom("CA", mdtraj.element.carbon, res)
    top.add_atom("CB", mdtraj.element.carbon, res)
    xyz = np.array([[pos0, pos1]], dtype=np.float32)
    return mdtraj.Trajectory(xyz, top)


# ---------------------------------------------------------------------------
# Simple attribute methods
# ---------------------------------------------------------------------------

def test_get_atom_group_names():
    cv = _make_cv(group_rmsd="solute", group_align="backbone")
    assert cv.get_atom_group_names() == ["solute", "backbone"]


def test_is_periodic_default():
    assert _make_cv().is_periodic() is False


def test_get_min_max_value():
    cv = _make_cv()
    assert cv.get_min_value() == 0.0
    assert cv.get_max_value() == 3.0


# ---------------------------------------------------------------------------
# get_mdtraj_cv_value
# ---------------------------------------------------------------------------

def _mock_model(tmp_path, ref_traj):
    """Write ref.pdb to tmp_path and return a mock model pointing there."""
    pdb_path = tmp_path / "collective_variable_sets" / "main" \
        / "collective_variable_0" / "ref.pdb"
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    ref_traj.save_pdb(str(pdb_path))
    mock_engine = MagicMock()
    # atom 0 = align group, atom 1 = RMSD group
    mock_engine.get_atom_indices_by_selection_name.side_effect = (
        lambda name: [1] if name == "lig" else [0]
    )
    mock_model = MagicMock()
    mock_model.directory = str(tmp_path)
    mock_model.md_engine = mock_engine
    return mock_model


def test_get_mdtraj_cv_value_identical(tmp_path):
    """RMSD is zero when query positions match the reference exactly."""
    ref_traj = _make_two_atom_traj([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    model = _mock_model(tmp_path, ref_traj)
    query_traj = _make_two_atom_traj([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    result = _make_cv().get_mdtraj_cv_value(model, query_traj, frame_index=0)
    assert result == pytest.approx(0.0, abs=1e-5)


def test_get_mdtraj_cv_value_displaced(tmp_path):
    """RMSD equals displacement of RMSD atom after alignment on atom 0."""
    # ref: atom 1 at (1,0,0); traj: atom 1 at (2,0,0); align atom 0 at origin in both
    # superpose on atom 0 makes no movement -> RMSD = sqrt((2-1)^2) = 1.0 nm
    ref_traj = _make_two_atom_traj([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    model = _mock_model(tmp_path, ref_traj)
    query_traj = _make_two_atom_traj([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
    result = _make_cv().get_mdtraj_cv_value(model, query_traj, frame_index=0)
    assert result == pytest.approx(1.0, abs=1e-5)
