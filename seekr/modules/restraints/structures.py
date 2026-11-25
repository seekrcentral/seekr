from __future__ import annotations
"""
modules/restraints/structures.py

Structures for any restraints that are applied to the system.
"""

import typing

from attrs import define, field, validators, Factory
import numpy as np


class Base_restraint:
    """
    Base class for all restraints.
    """
    pass

def load_positional_coordinate_blocks(
        filename: str, 
        n_atoms: int
        ) -> np.ndarray:
    """
    Load a set of coordinates from a file into a list of blocks, where each
    block contains the coordinates for a subset of atoms.
    """
    positions_all_frames = np.loadtxt(filename, ndmin=2)
    assert positions_all_frames.shape[1] == 3, \
        f"Coordinates file {filename} does not have 3 columns."
    assert positions_all_frames.shape[0] % n_atoms == 0, \
        f"Number of coordinate sets ({positions_all_frames.shape[0]}) "\
        f"in {filename} does not divide evenly into number of atom indices ({n_atoms})"
    return positions_all_frames.reshape(-1, n_atoms, 3)

@define
class Positional_restraint(Base_restraint):
    """
    Restraint that keeps a subset of atoms close to a set of coordinates,
    if provided. If not provided, the atoms are constrained to starting
    coordinates. If swarms are being used, then the coordinates may define,
    for each swarm member, a different set of coordinates to restrain the
    individual swarm member to.
    """
    type: typing.Literal["positional"] = "positional"
    atom_indices: list[int] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(int),
                              validators.ge(0)],
            iterable_validator=validators.instance_of(list),))
    force_constant: float = field(
        default=400.0, validator=[
            validators.instance_of(float),
            validators.ge(0)])
    coordinates_filename: typing.Optional[str] = field(
        default=None, 
        validator=validators.optional(validators.instance_of(str)))

@define
class Z_restraint(Base_restraint):
    """
    Restraint that keeps a subset of atoms close to a set of coordinates,
    if provided, or their starting Z-coordinates if not provided.
    """
    type: typing.Literal["z"] = "z"
    atom_indices: list[int] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(int),
                        validators.ge(0)],
            iterable_validator=validators.instance_of(list),))
    force_constant: float = field(
        default=400.0, validator=[
            validators.instance_of(float),
            validators.ge(0)])
    coordinates_filename: typing.Optional[str] = field(
        default=None, 
        validator=validators.optional(validators.instance_of(str)))

@define
class Torsion_restraint(Base_restraint):
    """
    Restraint that keeps a set of atomic proper or improper torsions
    close to a defined set of values.
    """
    type: typing.Literal["torsion"] = "torsion"
    torsion_indices: list[tuple[int, int, int, int]] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(tuple),
                              validators.min_len(4),
                              validators.max_len(4)],
            iterable_validator=validators.instance_of(list),))
    force_constant: float = field(
        default=400.0,
        validator=[validators.instance_of(float),
                   validators.ge(0)])    
    torsion_values: list[float] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(float)],
            iterable_validator=validators.instance_of(list),))
                   

@define
class Pairwise_restraint(Base_restraint):
    """
    Restraint that keeps pairs of atoms close to their starting
    distances from each other.
    """
    type: typing.Literal["pairwise"] = "pairwise"
    atom_indices: list[int] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(int),
                              validators.ge(0)],
            iterable_validator=validators.instance_of(list),))
    force_constant: float = field(
        default=400.0,
        validator=[validators.instance_of(float),
                   validators.ge(0)])

@define
class Proximity_restraint(Base_restraint):
    """
    Restraint that keeps two subsets of atoms within a certain 
    distance of each other.
    """
    type: typing.Literal["proximity"] = "proximity"
    atom_indices1: list[int] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(int),
                              validators.ge(0)],
            iterable_validator=validators.instance_of(list),))
    atom_indices2: list[int] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(int),
                              validators.ge(0)],
            iterable_validator=validators.instance_of(list),))
    radius: float = field(
        default=2.0,
        validator=[validators.instance_of(float),
                   validators.gt(0)])
    force_constant: float = field(
        default=10000.0,
        validator=[validators.instance_of(float),
                   validators.ge(0)])

@define
class Cylinder_restraint(Base_restraint):
    """
    Restrain a group of atoms (atom_indices1) to remain 
    within a cylinder whose axis is defined by two other 
    groups of atoms (atom_indices2 and atom_indices3).
    """
    type: typing.Literal["cylinder"] = "cylinder"
    atom_indices1: list[int] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(int),
                              validators.ge(0)],
            iterable_validator=validators.instance_of(list),))
    atom_indices2: list[int] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(int),
                              validators.ge(0)],
            iterable_validator=validators.instance_of(list),))
    atom_indices3: list[int] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(int),
                              validators.ge(0)],
            iterable_validator=validators.instance_of(list),))
    radius: float = field(
        default=2.0,
        validator=[validators.instance_of(float),
                   validators.gt(0)])
    force_constant: float = field(
        default=10000.0,
        validator=[validators.instance_of(float),
                   validators.ge(0)])

@define
class Center_of_mass_restraint(Base_restraint):
    """
    Restraint that keeps a list of subsets of atoms close to a set of 
    coordinates, if provided. If not provided, the atoms are constrained 
    to starting center of mass coordinates. If swarms are being used, 
    then the coordinates may define, for each swarm member, a different 
    set of coordinates to restrain the individual swarm member to.
    """
    type: typing.Literal["center_of_mass"] = "center_of_mass"
    atom_indices_list: list[list[int]] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.deep_iterable(
                member_validator=[validators.instance_of(int),
                                  validators.ge(0)],
                iterable_validator=validators.instance_of(list),)],
            iterable_validator=validators.instance_of(list),))
    force_constant: float = field(
        default=400.0, validator=[
            validators.instance_of(float),
            validators.ge(0)])
    coordinates_filename: typing.Optional[str] = field(
        default=None, 
        validator=validators.optional(validators.instance_of(str)))