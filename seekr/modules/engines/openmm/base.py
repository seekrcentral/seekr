"""
modules/engines/openmm/base.py

Base functions, types, and constants for the OpenMM engine.

Imports only allowed from modules/base.py, modules/engines/base.py 
and external libraries.
Imported by anything in the modules/engines/openmm subdirectory.
"""

import re
import typing

import numpy as np

if typing.TYPE_CHECKING:
    import openmm.unit as unit

def get_box_vectors_from_pdb(pdb_filename: str) -> "unit.Quantity":
    """
    Extract the box_vectors from the CRYST line in a pdb file.
    """
    import openmm.unit as unit
    import openmm.app as openmm_app
    pdb_obj = openmm_app.PDBFile(pdb_filename)
    box_vectors = pdb_obj.topology.getPeriodicBoxVectors()
    assert box_vectors is not None, "No box vectors "\
    "found in {}. ".format(pdb_filename) \
    + "Box vectors for an anchor must be defined with a CRYST "\
    "line within the PDB file, or explicitly set in the model "\
    "input XML file."
    length1 = np.linalg.norm(box_vectors[0].value_in_unit(unit.nanometers))
    length2 = np.linalg.norm(box_vectors[1].value_in_unit(unit.nanometers))
    length3 = np.linalg.norm(box_vectors[2].value_in_unit(unit.nanometers))
    assert (length1 > 0.0) and (length2 > 0.0) and (length3 > 0.0), \
        "Box vector length(s) of zero detected. Please check box vectors in " \
        "input PDB."
    return box_vectors

def convert_openmm_to_python_expr(
        old_function_str: str) -> str:
    """
    Convert the expression from a form used by OpenMM
    to a form used by Python.
    """
    new_function_str = re.sub(r"\^", "**", old_function_str)
    return new_function_str