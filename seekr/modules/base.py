"""
base.py

Contain common functions and utilities used throughout SEEKR.
"""

import os
import re
from typing import Callable

import numpy as np

def order_files_numerically(
        file_list: list[str], 
        func: Callable[[str], float]=float,
        use_basename: bool=False
        ) -> list[str]:
    """
    If there is a list of files, order them numerically, not
    alphabetically and return the sorted list of files. Note that
    only the base name is sorted, not any part of the directory.
    
    Parameters
    ----------
    file_list : list
        A list of strings that contain one or more numerical values. 
        The strings will be sorted by only the numerical values within
        them.

    func : function, Default float
        A function that converts a string to a number. By default, 
        this is the float function, but it could be int or any other
        function that takes a string and returns a number.
    
    use_basename : bool, Default False
        Whether to just use the base name (without the extenion) to
        sort the files. By default, this is False.
        
    Returns
    -------
    sorted_file_list : list
        A new list of strings sorted numerically
    """
    
    sorted_file_list = []
    numerical_dict = {}
    for i, file_name in enumerate(file_list):
        file_name2 = os.path.splitext(file_name)[0]
        if use_basename:
            file_name2 = os.path.basename(file_name2)
            
        numbers = re.findall(r"(-?\d+(?:\.\d+)?)", file_name2)
        numbers = tuple([func(j) for j in numbers])
        numerical_dict[numbers] = i
        
    numerical_list = sorted(numerical_dict.keys())
    sorted_file_list = [file_list[numerical_dict[number]] for number in numerical_list]
    return sorted_file_list

def convert_openmm_to_python_expr(
        old_function_str: str) -> str:
    """
    Convert the expression from a form used by OpenMM
    to a form used by Python.
    """
    new_function_str = re.sub(r"\^", "**", old_function_str)
    return new_function_str

class Box_vectors():
    """
    A box vector object that contains the a, b, and c vectors in units
    of nanometers.
    """
    
    def __init__(self) -> None:
        self.ax = 1.0
        self.ay = 0.0
        self.az = 0.0
        self.bx = 0.0
        self.by = 1.0
        self.bz = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.cz = 1.0
        
    def to_quantity(
            self, 
            quantity_obj: Callable, 
            my_unit):
        """
        Convert this object to a Quantity object that could be used
        in OpenMM or parmed.
        """
        box_vector = quantity_obj(
            [[self.ax, self.ay, self.az], 
             [self.bx, self.by, self.bz], 
             [self.cx, self.cy, self.cz]], unit=my_unit)
        return box_vector
    
    def from_quantity(self, quantity):
        """
        Fill out the entries within this object from a Quantity
        object representing box vectors.
        """
        
        values = quantity.value_in_unit(unit.nanometer)
        self.ax = float(values[0][0])
        self.ay = float(values[0][1])
        self.az = float(values[0][2])
        self.bx = float(values[1][0])
        self.by = float(values[1][1])
        self.bz = float(values[1][2])
        self.cx = float(values[2][0])
        self.cy = float(values[2][1])
        self.cz = float(values[2][2])
        return
    
    def from_6_vector(self, box_6_vector):
        """
        Sometimes box vectors are represented using 6 numbers, of which
        the first three are vector lengths, and the next three are the
        angles between them. Compute the full 3x3 box_vectors and assign
        to this object.
        
        box_6_vector is assumed to be in Angstroms and degrees.
        """
        
        # convert box_6_vector from Angstroms to nm
        box_6_vector[0] *= 0.1
        box_6_vector[1] *= 0.1
        box_6_vector[2] *= 0.1
        
        self.ax = float(box_6_vector[0])
        self.ay = 0.0
        self.az = 0.0
        self.bx = float(box_6_vector[1] \
                        * np.cos(box_6_vector[5]*np.pi/180.0))
        self.by = float(box_6_vector[1] \
                        * np.sin(box_6_vector[5]*np.pi/180.0))
        self.bz = 0.0
        self.cx = (box_6_vector[2] * np.cos(box_6_vector[4]*np.pi/180.0))
        if box_6_vector[5] == 90.0:
            self.cy = 0.0
        else:
            self.cy = float(box_6_vector[2] * (
                np.cos(box_6_vector[3]*np.pi/180.0) \
                - np.cos(box_6_vector[4]*np.pi/180.0) \
                * np.cos(box_6_vector[5]*np.pi/180.0)) \
                / np.sin(box_6_vector[5]*np.pi/180.0))
        self.cz = float(np.sqrt(box_6_vector[2]**2 - self.cx**2 \
            - self.cy**2))
        return
    
    def to_6_vector(self):
        """
        Sometimes box vectors are represented using 6 numbers, of which
        the first three are vector lengths, and the next three are the
        angles between them. Use this object's full 3x3 box_vectors and 
        return the equivalent 6-vector.
        """
        a = np.array([self.ax, self.ay, self.az])
        b = np.array([self.bx, self.by, self.bz])
        c = np.array([self.cx, self.cy, self.cz])
        a_length = np.linalg.norm(a)
        b_length = np.linalg.norm(b)
        c_length = np.linalg.norm(c)
        alpha = (180.0/np.pi) * np.acos(np.dot(b, c) / (b_length * c_length))
        beta = (180.0/np.pi) * np.acos(np.dot(c, a) / (c_length * a_length))
        gamma = (180.0/np.pi) * np.acos(np.dot(a, b) / (a_length * b_length))
        six_vector = [a_length, b_length, c_length, alpha, beta, gamma]
        return six_vector
    
    def get_volume(self):
        """
        Compute the volume of the box defined by these box vectors.
        Units in nm^3
        """
        
        A = np.array([self.ax, self.ay, self.az])
        B = np.array([self.bx, self.by, self.bz])
        C = np.array([self.cx, self.cy, self.cz])
        volume = abs(np.dot(A, np.cross(B, C)))
        return volume
    
    def get_min_length(self):
        """
        Get the dimension of largest magnitude.
        """
        lengths = []
        lengths.append(np.linalg.norm(np.array([self.ax, self.ay, self.az]))) 
        lengths.append(np.linalg.norm(np.array([self.bx, self.by, self.bz])))
        lengths.append(np.linalg.norm(np.array([self.cx, self.cy, self.cz])))
        return min(lengths)