from __future__ import annotations
"""
modules/restraints/structures.py

Structures for any restraints that are applied to the system.
"""

import typing
from attrs import define, field, validators, Factory

class Base_restraint:
    """
    Base class for all restraints.
    """
    pass

@define
class Positional_restraint(Base_restraint):
    """
    Restraint that keeps a subset of atoms close to their starting
    Cartesian coordinates.
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
    