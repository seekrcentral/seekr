"""
modules/reporters/preparer.py

Handle inputs for simulation reporters into SEEKR.
"""

import typing

from attrs import define, field, validators, Factory

import seekr.modules.reporters.structures as reporters_structures
import seekr.modules.engines.preparer as engines_preparer

class Base_reporter_input(typing.Protocol):
    """
    Base class for reporter inputs.
    """
    type: typing.Literal["base"] = "base"

    def make_reporter(
            self,
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
        ) -> reporters_structures.Base_reporter:
        """
        Make the reporter object for the model file.
        """
        raise NotImplementedError("Subclasses must implement this method.")

@define
class DCD_reporter_input(Base_reporter_input):
    """
    Input for DCD reporter.
    """
    type: typing.Literal["dcd"] = "dcd"
    interval: int = field(default=1000,
                          validator=[validators.instance_of(int),
                                     validators.gt(0)])
    enforce_periodic_box: bool = field(default=True,
                                       validator=validators.instance_of(bool))
    atom_subset: str | None = field(default=None,
                                     validator=validators.optional(
                                         validators.instance_of(str)))

    def make_reporter(
            self,
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
        ) -> reporters_structures.DCD_reporter:
        """
        Make the reporter object for the model file.
        """
        reporter = reporters_structures.DCD_reporter()
        if self.atom_subset is None:
            atom_indices = None
            reporter.make_topology = False
        else:
            atom_indices = systemwide_settings.get_atom_indices_by_selection_name(
                self.atom_subset)
            reporter.make_topology = True

        reporter.interval = self.interval
        reporter.enforce_periodic_box = self.enforce_periodic_box
        reporter.atom_subset = atom_indices
        return reporter

@define
class PDB_reporter_input(Base_reporter_input):
    """
    Input for PDB reporter.
    """
    type: typing.Literal["pdb"] = "pdb"
    interval: int = field(default=1000,
                          validator=[validators.instance_of(int),
                                     validators.gt(0)])
    enforce_periodic_box: bool = field(default=True,
                                       validator=validators.instance_of(bool))
    atom_subset: str | None = field(default=None,
                                     validator=validators.optional(
                                         validators.instance_of(str)))
    def make_reporter(
            self,
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
        ) -> reporters_structures.PDB_reporter:
        """
        Make the reporter object for the model file.
        """
        if self.atom_subset is None:
            atom_indices = None
        else:
            atom_indices = systemwide_settings.get_atom_indices_by_selection_name(
                self.atom_subset)
        reporter = reporters_structures.PDB_reporter()
        reporter.interval = self.interval
        reporter.enforce_periodic_box = self.enforce_periodic_box
        reporter.atom_subset = atom_indices
        return reporter
    
@define
class State_reporter_input(Base_reporter_input):
    """
    Input for state reporter
    """
    type: typing.Literal["state"] = "state"
    to_stdout: bool = field(default=True,
                          validator=validators.instance_of(bool))
    interval: int = field(default=1000,
                          validator=[validators.instance_of(int),
                                     validators.gt(0)])
    
    def make_reporter(
            self,
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
        ) -> reporters_structures.State_reporter:
        """
        Make the reporter object for the model file.
        """
        reporter = reporters_structures.State_reporter()
        reporter.to_stdout = self.to_stdout
        reporter.interval = self.interval
        return reporter
    
@define
class CV_value_reporter_input(Base_reporter_input):
    """
    Input for CV value reporter - write structures that correspond to specific CV values.
    """
    type: typing.Literal["cv_value"] = "cv_value"
    cv_name: str = field(default="main",
                         validator=validators.instance_of(str))
    cv_values: list[float] = field(default=Factory(list),
                                  validator=validators.instance_of(list))
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
    extension: str = field(
        default="dcd",
        validator=validators.instance_of(str))
    
    def make_reporter(
            self,
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
        ) -> reporters_structures.CV_value_reporter:
        """
        Make the reporter object for the model file.
        """
        reporter = reporters_structures.CV_value_reporter()
        reporter.cv_name = self.cv_name
        reporter.cv_values = self.cv_values
        reporter.interval = self.interval
        reporter.enforce_periodic_box = self.enforce_periodic_box
        reporter.atom_subset = self.atom_subset
        if self.extension == "dcd":
            reporter.make_topology = True
        elif self.extension == "pdb":
            reporter.make_topology = False
        else:
            raise ValueError(f"Unsupported extension {self.extension} for CV value reporter.")
        reporter.extension = self.extension
        return reporter

@define
class Anchor_reporter_input(Base_reporter_input):
    """
    Input for anchor reporter - write structures that exist within certain
    anchors.
    """
    type: typing.Literal["anchor"] = "anchor"
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
    extension: str = field(
        default="dcd",
        validator=validators.instance_of(str))
    
    def make_reporter(
            self,
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
        ) -> reporters_structures.CV_value_reporter:
        """
        Make the reporter object for the model file.
        """
        reporter = reporters_structures.CV_value_reporter()
        reporter.cv_name = self.cv_name
        reporter.cv_values = self.cv_values
        reporter.interval = self.interval
        reporter.enforce_periodic_box = self.enforce_periodic_box
        reporter.atom_subset = self.atom_subset
        if self.extension == "dcd":
            reporter.make_topology = True
        elif self.extension == "pdb":
            reporter.make_topology = False
        else:
            raise ValueError(f"Unsupported extension {self.extension} for CV value reporter.")
        reporter.extension = self.extension
        return reporter
