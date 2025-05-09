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
import glob
import typing
import logging

import numpy.typing as nptyping
import mdtraj

XML_ROOT = "seekr_model"
INITIAL_DIRECTORY = "stage_0_initial"
MAX_SEEKR_SIMULATION_STEPS = 100000000000 # ~100 us
SYSTEMWIDE_DIRECTORY = "systemwide"
STEP_TOLERANCE = 5

# NOTE: Positions_type needed for toy system inputs. Do not delete.
Positions_type = typing.Union[typing.List, str, None]
Float_or_list_type = typing.Union[typing.List, float, None]
Int_or_list_type = typing.Union[typing.List, int, None]

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

# Recursively wrap "lists of integers" (or floats) in NoIndent
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
        elif len(obj) > 0 and all(isinstance(x, float) for x in obj):
            return NoIndent(obj)
        else:
            # Recursively check each element
            return [wrap_noindent_integers(x) for x in obj]
    elif isinstance(obj, dict):
        # Recursively check each value in the dict
        return {k: wrap_noindent_integers(v) for k, v in obj.items()}
    else:
        # For anything else (strings, etc.), do nothing
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

def glob_or(
        patterns: typing.List[str],
        directory: str = ".") -> list[str]:
    """
    Find files matching any of the patterns in the list.
    """
    result = []
    for pattern in patterns:
        result.extend(glob.glob(os.path.join(directory, pattern)))
    return result

def mdtraj_center_of_mass(
        traj: mdtraj.Trajectory
        ) -> typing.Union[nptyping.ArrayLike]:
    """
    Returns a center of mass array by frames for a traj. Avoids NaN
    errors that come when using mdtraj's .
    """
    if traj.xyz.shape[1] == 1:
        com_array = traj.xyz[:,0,:]
    else:
        com_array = mdtraj.compute_center_of_mass(traj)
    return com_array
