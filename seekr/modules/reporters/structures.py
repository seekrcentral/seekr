"""
modules/reporters/structures.py

Structures for reporters in seekr simulations.
"""

import typing

from attrs import define, field, validators

class Base_reporter(typing.Protocol):
    """
    Base class for all reporters.
    """
    type: typing.Literal["base"] = "base"

    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        raise NotImplementedError("Subclasses must implement this method.")

@define
class DCD_reporter(Base_reporter):
    """
    A DCD reporter for SEEKR.
    """
    type: typing.Literal["dcd"] = "dcd"
    interval: int = field(
        default=1000,
        validator=[validators.instance_of(int),
                   validators.gt(0)])
    enforce_periodic_box: bool = field(
        default=True,
        validator=validators.instance_of(bool))
    atom_subset: typing.Optional[typing.List[int]] = field(
        default=None,
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(int),
                iterable_validator=validators.instance_of(list))))
    make_topology: bool = field(
        default=True, 
        validator=validators.instance_of(bool))
    
    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        return "dcd"
    
@define
class PDB_reporter(Base_reporter):
    """
    A PDB reporter for SEEKR.
    """
    type: typing.Literal["pdb"] = "pdb"
    interval: int = field(
        default=1000,
        validator=[validators.instance_of(int),
                   validators.gt(0)])
    enforce_periodic_box: bool = field(
        default=True,
        validator=validators.instance_of(bool))
    atom_subset: typing.Optional[typing.List[int]] = field(
        default=None,
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(int),
                iterable_validator=validators.instance_of(list))))
    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        return "pdb"
    
@define
class State_reporter(Base_reporter):
    """
    A state reporter for SEEKR.
    """
    type: typing.Literal["state"] = "state"
    to_stdout: bool = field(default=True,
                          validator=validators.instance_of(bool))
    interval: int = field(default=1000,
                          validator=[validators.instance_of(int),
                                     validators.gt(0)])
    step: bool = field(default=True,
                      validator=validators.instance_of(bool))
    time: bool = field(default=False,
                      validator=validators.instance_of(bool))
    potential_energy: bool = field(default=True,
                      validator=validators.instance_of(bool))
    kinetic_energy: bool = field(default=False,
                      validator=validators.instance_of(bool))
    total_energy: bool = field(default=False,
                      validator=validators.instance_of(bool))
    temperature: bool = field(default=True,
                      validator=validators.instance_of(bool))
    volume: bool = field(default=True,
                        validator=validators.instance_of(bool))
    density: bool = field(default=False,
                        validator=validators.instance_of(bool))
    speed: bool = field(default=False,
                        validator=validators.instance_of(bool))
    elapsed_time: bool = field(default=False,
                        validator=validators.instance_of(bool))
    
    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        return "dat"
    