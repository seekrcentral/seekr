from __future__ import annotations
from typing import Optional
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
import shlex
import socket
import getpass
import logging
import importlib

from attrs import define, field, validators, Factory
import numpy as np
import numpy.typing as nptyping
import mdtraj

logger = logging.getLogger(__name__)

XML_ROOT = "seekr_model"
INITIAL_DIRECTORY = "stage_0_initial"
MAX_SEEKR_SIMULATION_STEPS = 100000000000 # ~100 us
UNPARTITIONED_DIRECTORY = "unpartitioned_scope"
STEP_TOLERANCE = 5
ANCHOR_SETS_DIR = "anchor_sets"
CVS_DIR = "collective_variables"
SELECTIONS_DIR = "selections"
PREPARE_SET_NAME = "main"

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

SEEKR_LOGGER_NAME = "seekr"

def configure_logging(
        file: Optional[str] = None,
        *,
        console: bool = True,
        overwrite: bool = False,
        level: int = logging.INFO,
        ) -> None:
    """
    Configure the seekr package logger ("seekr"). Any handlers from a
    prior call are removed and replaced, so re-invoking with a different
    file path swaps the destination cleanly.

    The seekr package logger is configured (not the root logger), so
    third-party libraries are not affected and do not write into seekr
    log files.
    """
    logger = logging.getLogger(SEEKR_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])
    if file is not None:
        file_handler = logging.FileHandler(
            file, mode="w" if overwrite else "a")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
    if console:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)
    return

def log_invocation_metadata(seekr_version: str) -> None:
    """
    Log SEEKR version, Python version, hostname, username, and the
    command line. Intended to be called once per entry-point invocation
    (prepare/run/analyze) to capture reproducibility context.
    """
    logger.info(
        f"SEEKR version: {seekr_version}; "
        f"Python: {sys.version.split()[0]}; "
        f"hostname: {socket.gethostname()}; "
        f"user: {getpass.getuser()}; "
        f"command: {shlex.join(sys.argv)}")
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
def string_lower(string: Optional[str]) -> Optional[str]:
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

def particle_masses_from_mdtraj_topology(
        topology: mdtraj.Topology
        ) -> np.ndarray:
    """
    Element masses (amu) for every atom in an mdtraj Topology.
    """
    return np.array(
        [atom.element.mass if atom.element is not None else 0.0
         for atom in topology.atoms],
        dtype=float)

def apply_hydrogen_mass_repartitioning(
        masses: np.ndarray,
        topology: mdtraj.Topology,
        hydrogen_mass: float
        ) -> tuple[np.ndarray, int]:
    """
    Apply OpenMM createSystem(hydrogenMass=...) mass transfer.

    For each H–heavy bond, set the hydrogen mass to hydrogen_mass and
    subtract (hydrogen_mass - original_H_mass) from the heavy atom.
    Returns (new_masses, number of H–heavy bonds updated).
    """
    masses = np.array(masses, dtype=float, copy=True)
    n_transfers = 0
    for atom1, atom2 in topology.bonds:
        e1 = atom1.element
        e2 = atom2.element
        if e1 is None or e2 is None:
            continue
        if e1.atomic_number == 1:
            atom1, atom2 = atom2, atom1
            e1, e2 = e2, e1
        if e2.atomic_number == 1 and e1.atomic_number != 1:
            orig_h = masses[atom2.index]
            transfer = hydrogen_mass - orig_h
            masses[atom2.index] = hydrogen_mass
            masses[atom1.index] -= transfer
            n_transfers += 1
    return masses, n_transfers

def mass_weighted_com(
        xyz: np.ndarray,
        indices: typing.Sequence[int],
        masses: np.ndarray
        ) -> np.ndarray:
    """
    Mass-weighted center of mass of atoms `indices` in a single-frame xyz.
    """
    inds = np.asarray(indices, dtype=int)
    m = masses[inds][:, None]
    return (xyz[inds] * m).sum(axis=0) / m.sum()

def minimum_image_delta(
        delta: np.ndarray,
        box_vectors: np.ndarray
        ) -> np.ndarray:
    """
    Minimum-image vector for a triclinic box whose rows are the box vectors.
    """
    inv_box = np.linalg.inv(box_vectors)
    frac = np.dot(delta, inv_box)
    frac -= np.round(frac)
    return np.dot(frac, box_vectors)

def group_com_distance(
        xyz: np.ndarray,
        group1_indices: typing.Sequence[int],
        group2_indices: typing.Sequence[int],
        masses: np.ndarray,
        box_vectors: np.ndarray | None = None
        ) -> float:
    """
    Distance between mass-weighted COMs of two groups.

    If box_vectors is a (3, 3) array (rows = box vectors) with a
    non-singular finite cell, the COM–COM vector uses the minimum-image
    convention (OpenMM CustomCentroidBondForce with periodic boundary
    conditions). Group atoms are not unwrapped. Toy / missing / degenerate
    boxes stay as the raw Euclidean distance.
    """
    com1 = mass_weighted_com(xyz, group1_indices, masses)
    com2 = mass_weighted_com(xyz, group2_indices, masses)
    delta = com2 - com1
    if box_vectors is not None:
        box = np.asarray(box_vectors, dtype=float)
        if (box.shape == (3, 3)
                and np.all(np.isfinite(box))
                and abs(float(np.linalg.det(box))) > 1e-12):
            delta = minimum_image_delta(delta, box)
    return float(np.linalg.norm(delta))

def get_plugin_str(
        plugin_name: str,
        submodule_name: str,
        require_structures: bool = True,
        require_preparer: bool = False,
        require_runner: bool = False,
        require_openmm_interface: bool = True,
        skip_silently: bool = True
        ) -> Optional[str]:
    """
    Import a sub-module of a seekr plugin and check the necessary sub-modules.
    """
    if submodule_name != "":
        dot_submodule_name = "."+submodule_name
    else:
        dot_submodule_name = ""

    plugin_spec = importlib.util.find_spec(plugin_name)
    if plugin_spec is None:
        raise ImportError(f"Plugin '{plugin_name}' not found. Please install it or check the spelling.")
    
    # see if plugin.modules exists, and if not, raise an error
    modules_spec = importlib.util.find_spec(plugin_name+".modules")
    if modules_spec is None:
        raise ImportError(f"Directory 'modules' not found for plugin '{plugin_name}'.")
    
    # Then if plugin.modules.{submodule} doesn't exist, skip quietly
    submodule_spec = importlib.util.find_spec(plugin_name+f".modules{dot_submodule_name}")
    if submodule_spec is None:
        if skip_silently:
            logger.warning(f"Plugin '{plugin_name}' does not have a '{submodule_name}' module. Skipping.")
            return None
        else:
            raise ImportError(f"Directory '{submodule_name}' not found for plugin '{plugin_name}'.")
        
    if require_structures:
        # Then if plugin.modules.{submodule}.structures doesn't exist, raise an error
        structures_spec = importlib.util.find_spec(plugin_name+f".modules{dot_submodule_name}.structures")
        if structures_spec is None:
            raise ImportError(f"Module 'structures.py' not found for plugin '{plugin_name}' "\
                            f"submodule '{submodule_name}'.")
        
    if require_preparer:
        # Then if plugin.modules.{submodule}.preparer doesn't exist, raise an error
        preparer_spec = importlib.util.find_spec(plugin_name+f".modules{dot_submodule_name}.preparer")
        if preparer_spec is None:
            raise ImportError(f"Module 'preparer.py' not found for plugin '{plugin_name}' "\
                            f"submodule '{submodule_name}'.")
        
    if require_runner:
        # Then if plugin.modules.{submodule}.runner doesn't exist, raise an error
        runner_spec = importlib.util.find_spec(plugin_name+f".modules{dot_submodule_name}.runner")
        if runner_spec is None:
            raise ImportError(f"Module 'runner.py' not found for plugin '{plugin_name}' "\
                            f"submodule '{submodule_name}'.")
        
    if require_openmm_interface:
        # Then if plugin.modules.{submodule}.openmm_interface doesn't exist, raise an error
        openmm_spec = importlib.util.find_spec(plugin_name+f".modules{dot_submodule_name}.openmm_interface")
        if openmm_spec is None:
            raise ImportError(f"Module 'openmm_interface.py' not found for plugin '{plugin_name}' "\
                            f"submodule '{submodule_name}'.")
    module_str = plugin_name+f".modules{dot_submodule_name}"
    return module_str

# NOTE: Even though these properly belong in modules/structures.py, because
#  they are structurable, they are imported by modules that depend on structures.py
#  so we want to avoid circular imports. They are leaves on the structure tree, so
#  they will not need to import anything.
@define
class NVT_ensemble:
    """
    Use the constant volume/temperature ensemble.
    """
    type: typing.Literal["NVT"] = "NVT"

@define
class NPT_ensemble:
    """
    Use the constant pressure/temperature ensemble.
    """
    type: typing.Literal["NPT"] = "NPT"
    pressure: float = field(default=1.0,
                            validator=[validators.instance_of(float),
                                       validators.gt(0)])

@define
class NPT_membrane_ensemble:
    """
    Use the constant pressure/temperature ensemble with box 
    adjustments useful for membrane sims.
    """
    type: typing.Literal["NPT_membrane"] = "NPT_membrane"
    pressure: float = field(default=1.0,
                            validator=[validators.instance_of(float),
                                       validators.gt(0)])

@define
class NVE_ensemble:
    """
    Use the constant volume/energy ensemble.
    """
    type: typing.Literal["NVE"] = "NVE"

@define
class State_point:
    """
    An object that can be put into the input file to indicate the location
    of a state within a CV or Combo. The name of the anchor containing this
    point will be changed, and end_state will be set to True.
    """
    name: str = field(default="",
                        validator=validators.instance_of(str))
    value: list[Optional[float]] = field(default=Factory(list),)
                                #validator=validators.deep_iterable(
                                #    member_validator=validators.instance_of(float),
                                #    iterable_validator=validators.instance_of(list),
                                #))
    # whether to set the end_anchor attribute to True in an Anchor
    end_state: bool = field(default=True,
                            validator=validators.instance_of(bool))
    bulk_state: bool = field(default=False,
                            validator=validators.instance_of(bool))

Ensemble = typing.Union[NVT_ensemble, NPT_ensemble, NPT_membrane_ensemble, 
                        NVE_ensemble]
