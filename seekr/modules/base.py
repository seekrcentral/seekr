"""
modules/base.py

Contain common functions and utilities used throughout seekr.

Imports from external libraries only.
Imported by anything else in the seekr codebase.
"""

import os
import sys
import re
import json
import typing
import logging

import numpy as np
import numpy.typing as nptyping
import mdtraj

XML_ROOT = "seekr_model"
INITIAL_DIRECTORY = "stage_0_initial"
MAX_SEEKR_SIMULATION_STEPS = 100000000000 # ~100 us
SYSTEMWIDE_DIRECTORY = "systemwide"
STEP_TOLERANCE = 5

Positions_type = typing.Union[typing.List, str, None]
Float_or_list_type = typing.Union[typing.List, float, None]
Int_or_list_type = typing.Union[typing.List, int, None]

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
 
    # NOTE: this function was replaced by to_array. Simply use to_array()
    #  and multiply by the unit.nanometers
    #def to_quantity(
    #        self, 
    #        quantity_obj: "Quantity", 
    #        my_unit: "Unit") -> "Quantity":
    def to_array(self) -> nptyping.ArrayLike:
        """
        Convert this object to a np.Array object that could be used
        in OpenMM or parmed.
        """
        box_vector = np.array(
            [[self.ax, self.ay, self.az], 
             [self.bx, self.by, self.bz], 
             [self.cx, self.cy, self.cz]])
        return box_vector
    
    # NOTE: this function was replaced by to_array. Simply use to_array()
    #  and multiply by the unit.nanometers
    #def from_quantity(
    #        self, 
    #        quantity: "Quantity", 
    #        my_unit: "Unit") -> None:
    def from_array(
            self, 
            array: nptyping.ArrayLike
            ) -> None:
        """
        Fill out the entries within this object from a Numpy Array
        object representing box vectors.
        """
        self.ax = float(array[0][0])
        self.ay = float(array[0][1])
        self.az = float(array[0][2])
        self.bx = float(array[1][0])
        self.by = float(array[1][1])
        self.bz = float(array[1][2])
        self.cx = float(array[2][0])
        self.cy = float(array[2][1])
        self.cz = float(array[2][2])
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
    MAX_LINE_WIDTH = 50
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
            new_str_list = []
            linewidth = 0
            inserted_newline_at_beginning = False
            for char in encoded:
                if char in ["\n", " "]:
                    continue
                if char == ",":
                    if linewidth > self.MAX_LINE_WIDTH:
                        new_str_list.append(",\n")
                        linewidth = 0
                        if not inserted_newline_at_beginning:
                            new_str_list.insert(0, "\n")
                            inserted_newline_at_beginning = True
                    else:
                        new_str_list.append(",")
                        linewidth += 1
                else:
                    new_str_list.append(char)
                    linewidth += 1
            
            encoded = "".join(new_str_list)
            #encoded = encoded.replace("\n", "")
            #encoded = encoded.replace(" ", "")
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

def configure_file_only_logger(
        logname: str, 
        overwrite: bool=False) -> None:
    """
    Set the file-only logger for the seekr program.
    """
    logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    if overwrite:
        filemode = 'w'
    else:
        filemode = 'a'
    logging.basicConfig(
        filename=logname, filemode=filemode,
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    return

def configure_file_and_stdout_logger(
        logname: str,
        overwrite: bool=False
        ) -> None:
    """
    Set the file and stdout logger for the seekr program.
    """
    configure_file_only_logger(logname, overwrite)
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    return

def configure_stdout_only_logger() -> None:
    """
    Set the file and stdout logger for the seekr program.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    console = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)
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

# TODO: move to openmm/base.py
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

# Otherwise Mypy has a problem with the str.lower method.
def string_lower(string: str | None) -> str | None:
    if isinstance(string, str):
        return string.lower()
    elif string is None:
        return None
    else:
        raise TypeError("string_lower only accepts strings or None.")
    
def get_pdb_num_frames(full_pdb_filename: str) -> int:
    """
    Using MDtraj, get the number of frames in a PDB file.
    """
    mdtraj_obj = mdtraj.load(full_pdb_filename)
    num_frames = mdtraj_obj.n_frames
    return num_frames