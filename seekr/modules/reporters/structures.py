from __future__ import annotations
"""
modules/reporters/structures.py

Structures for reporters in seekr simulations.
"""

import typing

from attrs import define, field, validators, Factory

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

# TODO: replace with positions interval reporter? Just assign the OpenMM
#  reporter based on .extension attribute?

@define
class Position_reporter(Base_reporter):
    """
    A position reporter for SEEKR - either a PDB or DCD file.
    """
    type: typing.Literal["position"] = "position"
    name: str | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)))
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
    extension: str = field(
        default="dcd",
        validator=validators.instance_of(str))
    
    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        return self.extension
    
@define
class State_reporter(Base_reporter):
    """
    A state reporter for SEEKR - energy, temperature, volume, etc. info.
    """
    type: typing.Literal["state"] = "state"
    name: str | None = field(default=None,
                      validator=validators.optional(validators.instance_of(str)))
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
    
@define
class CV_value_reporter(Base_reporter):
    """
    A CV value reporter for SEEKR can write sampled positions of a simulation
    that correspond to a region of CV space.
    """
    type: typing.Literal["cv_value"] = "cv_value"
    name: str | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)))
    cv_name: str = field(
        default="main",
        validator=validators.instance_of(str))
    cv_values: list[float] = field(
        default=Factory(list),
        validator=validators.instance_of(list))
    interval: int = field(
        default=1000,
        validator=[validators.instance_of(int),
                   validators.gt(0)])
    enforce_periodic_box: bool = field(
        default=True,
        validator=validators.instance_of(bool))
    extension: str = field(
        default="dcd",
        validator=validators.instance_of(str))
    save_interval: int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)))
    mode: str = field(
        default="save_last_frame",
        validator=validators.in_(["save_last_frame", "save_first_frame"]))
    
    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        return self.extension
    
@define
class Anchor_reporter(Base_reporter):
    """
    An anchor reporter writes the positions of a simulation that exist within
    an anchor.
    """
    type: typing.Literal["anchor"] = "anchor"
    name: str | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)))
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
    extension: str = field(
        default="dcd",
        validator=validators.instance_of(str))
    save_interval: int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)))
    mode: str = field(
        default="save_last_frame",
        validator=validators.in_(["save_last_frame", "save_first_frame"]))
    
    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        return self.extension