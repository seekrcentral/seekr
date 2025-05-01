"""
test_base.py

Test the modules/base.py module.
"""

import os
import pytest
import random

import numpy as np

import seekr.modules.base as base

TEST_DIRECTORY = os.path.dirname(__file__)

def test_order_files_numerically():
    string_list = ["/path/to/anchor0/output0_0", "/path/to/anchor0/output0_1",
                   "/path/to/anchor0/output0_2", "/path/to/anchor0/output1_0",
                   "/path/to/anchor0/output1_1", "/path/to/anchor0/output1_2",
                   "/path/to/anchor1/output0_0", "/path/to/anchor1/output0_1",
                   "/path/to/anchor1/output2_0", "/path/to/anchor1/output10_0"]
    desired_list = string_list[:]
    random.shuffle(string_list)
    ordered_list = base.order_files_numerically(string_list)
    
    for item1, item2 in zip(ordered_list, desired_list):
        assert item1==item2
    
    scrambled_list = ["me2.345.pdb", "me1.234.pdb", "me-3.456.pdb"]
    correct_list = ["me-3.456.pdb", "me1.234.pdb", "me2.345.pdb"]
    ordered_list = base.order_files_numerically(scrambled_list, func=float)
    for item1, item2 in zip(ordered_list, correct_list):
        assert item1==item2
    
    string_list = ["mmvt.restart1.out", "mmvt.restart2.out"]
    desired_list = string_list[:]
    random.shuffle(string_list)
    ordered_list = base.order_files_numerically(string_list)
    
    for item1, item2 in zip(ordered_list, desired_list):
        assert item1==item2
    
    string_list = ["/path/to/anchor_0.1/output0_0", 
                   "/path/to/anchor_0.2/output0_1"]
    
    desired_list = string_list[:]
    random.shuffle(string_list)
    ordered_list = base.order_files_numerically(string_list)
    
    return

# TODO: this needs to be changed to use updated methods
@pytest.mark.needs_openmm
def test_box_vectors():
    from openmm import unit
    box_vector_q = unit.Quantity(
        [[64.9127105, 0.0, 0.0], 
         [-21.6375684, 61.2002909, 0.0], 
         [-21.6375684, -30.6001417, 53.0010088]], 
                      unit=unit.angstrom)
    box_vector = base.Box_vectors()
    box_vector.from_quantity(box_vector_q, unit.nanometers)
    assert np.isclose(box_vector.ax, 6.49127105)
    assert np.isclose(box_vector.ay, 0.0)
    assert np.isclose(box_vector.az, 0.0)
    assert np.isclose(box_vector.bx, -2.16375684)
    assert np.isclose(box_vector.by, 6.12002909)
    assert np.isclose(box_vector.bz, 0.0)
    assert np.isclose(box_vector.cx, -2.16375684)
    assert np.isclose(box_vector.cy, -3.06001417)
    assert np.isclose(box_vector.cz, 5.30010088)
    box_vector_q2 = box_vector.to_quantity(unit.Quantity, unit.nanometers)
    assert np.isclose(box_vector_q2.value_in_unit(unit.nanometers)[0][0], 
                      6.49127105)
    assert np.isclose(box_vector_q2.value_in_unit(unit.nanometers)[0][1], 0.0)
    assert np.isclose(box_vector_q2.value_in_unit(unit.nanometers)[0][2], 0.0)
    assert np.isclose(box_vector_q2.value_in_unit(unit.nanometers)[1][0], 
                      -2.16375684)
    assert np.isclose(box_vector_q2.value_in_unit(unit.nanometers)[1][1], 
                      6.12002909)
    assert np.isclose(box_vector_q2.value_in_unit(unit.nanometers)[1][2], 0.0)
    assert np.isclose(box_vector_q2.value_in_unit(unit.nanometers)[2][0], 
                      -2.16375684)
    assert np.isclose(box_vector_q2.value_in_unit(unit.nanometers)[2][1], 
                      -3.06001417)
    assert np.isclose(box_vector_q2.value_in_unit(unit.nanometers)[2][2], 
                      5.30010088)
    box_6_vector = [61.239410, 61.239410, 61.239410, 109.471, 109.471, 109.471]
    box_vector2 = base.Box_vectors()
    box_vector2.from_6_vector(box_6_vector)
    assert np.isclose(box_vector2.ax, 6.1239410, atol=0.001)
    assert np.isclose(box_vector2.ay, 0.0)
    assert np.isclose(box_vector2.az, 0.0)
    assert np.isclose(box_vector2.bx, -2.041313667, atol=0.001)
    assert np.isclose(box_vector2.by, 5.773707003, atol=0.001)
    assert np.isclose(box_vector2.bz, 0.0)
    assert np.isclose(box_vector2.cx, -2.041313667, atol=0.001)
    assert np.isclose(box_vector2.cy, -2.886853152, atol=0.001)
    assert np.isclose(box_vector2.cz, 5.00017714, atol=0.001)
    expected_box_6_vector = [6.1239410, 6.1239410, 6.1239410, 109.47122, 
                             109.47122, 109.47122]
    resulting_box_6_vector = box_vector2.to_6_vector()
    assert np.isclose(np.array(expected_box_6_vector), 
                      np.array(resulting_box_6_vector)).all()
    return

