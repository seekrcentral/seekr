"""
base.py

Contain common functions and utilities used throughout SEEKR.
"""

import os
import re
import json
import typing
import logging

import numpy as np
from attrs import define, field, validators, Factory

XML_ROOT = "seekr_model"
INITIAL_DIRECTORY = "stage_0_initial"

# To prevent circular imports due to type checking
if typing.TYPE_CHECKING:
    import openmm

@define
class Selection:
    """
    A selection of atoms to help with the definition of CVs.
    """
    name: str = field(default="",
                        validator=validators.instance_of(str))
    # TODO: support additional types of selections like resname, noh, etc.?
    #  probably not - the automated site finder and anchor assignment should
    #  take care of that.
    atom_indices: list[int] = field(default=Factory(list),
                                    validator=validators.deep_iterable(
                                        member_validator=[validators.instance_of(int),
                                                    validators.ge(0)],
                                        iterable_validator=validators.instance_of(list),
                                    ))

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
            quantity_obj: "openmm.unit.Quantity", 
            my_unit: "openmm.unit.Unit") -> "openmm.unit.Quantity":
        """
        Convert this object to a Quantity object that could be used
        in OpenMM or parmed.
        """
        box_vector = quantity_obj(
            [[self.ax, self.ay, self.az], 
             [self.bx, self.by, self.bz], 
             [self.cx, self.cy, self.cz]], unit=my_unit)
        return box_vector
    
    def from_quantity(
            self, 
            quantity: "openmm.unit.Quantity", 
            my_unit: "openmm.unit.Unit") -> None:
        """
        Fill out the entries within this object from a Quantity
        object representing box vectors.
        """
        
        values = quantity.value_in_unit(my_unit)
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
    
    def from_6_vector(
            self, 
            box_6_vector: list[float]) -> None:
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
        alpha = (180.0/np.pi) * np.arccos(np.dot(b, c) / (b_length * c_length))
        beta = (180.0/np.pi) * np.arccos(np.dot(c, a) / (c_length * a_length))
        gamma = (180.0/np.pi) * np.arccos(np.dot(a, b) / (a_length * b_length))
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
    
#Define a "NoIndent" wrapper class and a custom JSON encoder
class NoIndent:
    """
    Wrap a list of integers in this class to signal that it should *not* be
    split over multiple lines when JSON-encoded with `indent=4`.
    """
    def __init__(self, value):
        self.value = value

class NoIndentListEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that encodes NoIndent-wrapped objects as a single
    line, while still pretty-printing everything else with indent=4.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store a map from "placeholder" -> JSON fragment
        self._replacement_map = {}
    
    def default(self, obj):
        # Whenever we see a NoIndent wrapper, encode its contents as a normal
        # JSON string, but store that string in a replacement map instead of
        # letting the default pretty-printer insert newlines.
        if isinstance(obj, NoIndent):
            # Encode the *contents* of obj normally
            encoded = super().encode(obj.value)
            encoded = encoded.replace("\n", "")
            encoded = encoded.replace(" ", "")
            # Create a unique placeholder string
            placeholder = f"__NoIndent_{id(obj)}__"
            # Store the final JSON string for that placeholder
            self._replacement_map[placeholder] = encoded
            # Return just the placeholder as if it were a string
            return placeholder
        
        # Otherwise, handle it as normal
        return super().default(obj)
    
    def encode(self, obj):
        # Let the default encoder run first (it will produce multi-line JSON)
        result = super().encode(obj)
        
        # Now replace each placeholder with its single-line representation
        for placeholder, encoded in self._replacement_map.items():
            # The default encoder will have put quotes around our placeholder.
            # We need to remove those quotes and replace with the raw JSON.
            quoted = json.dumps(placeholder)  # e.g. "\"__NoIndent_12345__\""
            result = result.replace(quoted, encoded)
        
        return result

@define
class Ion:
    name: str | None = field(default=None,
                             validator=validators.optional(
                                 validators.instance_of(str)))
    radius: float = field(default=0.9,
                            validator=[validators.instance_of(float),
                                         validators.gt(0)])
    charge: float = field(default=1.0,
                            validator=[validators.instance_of(float)])
    concentration: float = field(default=0.0,
                                    validator=[validators.instance_of(float),
                                                validators.ge(0)])


def configure_logger(
        logname: str, 
        overwrite: bool=False) -> None:
    """
    Set the logger for the SEEKR program.
    """
    if overwrite:
        filemode = 'w'
    else:
        filemode = 'a'
    logging.basicConfig(
        filename=logname, filemode=filemode,
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    return

def order_files_numerically(
        file_list: list[str], 
        func: typing.Callable[[str], float]=float,
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
    
    numerical_dict: dict[tuple,int] = {}
    for i, file_name in enumerate(file_list):
        file_name2: str = os.path.splitext(file_name)[0]
        if use_basename:
            file_name2 = os.path.basename(file_name2)
            
        numbers: list[str] = re.findall(r"(-?\d+(?:\.\d+)?)", file_name2)
        numbers_tuple: tuple = tuple([func(j) for j in numbers])
        numerical_dict[numbers_tuple] = i
        
    numerical_list: list[tuple] = sorted(numerical_dict.keys())
    sorted_file_list: list[str] = [file_list[numerical_dict[number]] for number in numerical_list]
    return sorted_file_list

def convert_openmm_to_python_expr(
        old_function_str: str) -> str:
    """
    Convert the expression from a form used by OpenMM
    to a form used by Python.
    """
    new_function_str = re.sub(r"\^", "**", old_function_str)
    return new_function_str

# Recursively wrap "lists of integers" in NoIndent
def wrap_noindent_integers(obj):
    """
    Recursively go through a data structure (dict/list/whatever),
    and whenever we find a list *all of whose elements are ints*,
    wrap it in NoIndent so it won't get multi-line indentation.
    
    This is mainly for JSON file writing.
    """
    if isinstance(obj, list):
        if len(obj) > 0 and all(isinstance(x, int) for x in obj):
            return NoIndent(obj)
        else:
            # Recursively check each element
            return [wrap_noindent_integers(x) for x in obj]
    elif isinstance(obj, dict):
        # Recursively check each value in the dict
        return {k: wrap_noindent_integers(v) for k, v in obj.items()}
    else:
        # For anything else (strings, floats, etc.), do nothing
        return obj
    
def make_tree(
        directory: str, 
        tree: dict[str, dict]) -> None:
    """
    Construct the file tree given a directory, populating
    it with all subsequent branches and leaves in the file tree.
    
    Parameters:
    -----------
    directory : str
        a string indicating the path in which to place the root of
        the file tree 
    
    tree: optional nested dictionary structure
        of the tree. (see Filetree.__doc__)
    
    Returns:
    --------
    None
    """
    assert os.path.isdir(directory), "'directory' argument must be a "\
        "real directory"
    for subbranch in list(tree.keys()):
        # first create each subbranch
        subbranchpath = os.path.join(directory,subbranch)
        if not os.path.exists(subbranchpath):
            os.mkdir(subbranchpath)
        if not tree[subbranch]: 
            # if its an empty list, then we have a leaf
            continue
        else: 
            # then we can descend further, recursively
            make_tree(subbranchpath, tree=tree[subbranch])
    return
