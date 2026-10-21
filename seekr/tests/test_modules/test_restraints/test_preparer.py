from __future__ import annotations
"""
Tests for modules/restraints/preparer.py (Z and torsion restraints).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from seekr.modules.restraints.preparer import (
    Torsion_restraint_input,
    Z_restraint_input,
)


def test_z_restraint_input_make_restraint():
    unpartitioned = MagicMock()
    unpartitioned.get_atom_indices_by_selection_name.return_value = [1, 3, 5]
    restraint_input = Z_restraint_input(
        selection_name="guest", force_constant=250.0)
    restraint = restraint_input.make_restraint(unpartitioned, root_directory="/tmp")
    assert restraint.type == "z"
    assert restraint.atom_indices == [1, 3, 5]
    assert restraint.force_constant == 250.0
    assert restraint.coordinates_filename is None
    unpartitioned.get_atom_indices_by_selection_name.assert_called_once_with(
        "guest")


def test_torsion_restraint_input_make_restraint():
    restraint_input = Torsion_restraint_input(
        torsion_indices=[(0, 1, 2, 3)],
        torsion_values=[0.5],
        force_constant=100.0,
    )
    restraint = restraint_input.make_restraint(MagicMock(), root_directory="/tmp")
    assert restraint.type == "torsion"
    assert restraint.torsion_indices == [(0, 1, 2, 3)]
    assert restraint.torsion_values == [0.5]
    assert restraint.force_constant == 100.0


def test_torsion_restraint_input_mismatched_lengths_raises():
    restraint_input = Torsion_restraint_input(
        torsion_indices=[(0, 1, 2, 3)],
        torsion_values=[0.1, 0.2],
    )
    with pytest.raises(AssertionError, match="does not match"):
        restraint_input.make_restraint(MagicMock(), root_directory="/tmp")


def test_torsion_restraint_input_out_of_range_value_raises():
    restraint_input = Torsion_restraint_input(
        torsion_indices=[(0, 1, 2, 3)],
        torsion_values=[float(np.pi) + 0.1],
    )
    with pytest.raises(AssertionError, match="not within the range"):
        restraint_input.make_restraint(MagicMock(), root_directory="/tmp")
