"""
modules/reporters/preparer.py

Handle inputs for simulation reporters into SEEKR.
"""

import typing

from attrs import define, field, validators

import seekr.modules.reporters.structures as reporters_structures
import seekr.modules.engines.preparer as engines_preparer

@define
class DCD_reporter_input:
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
class PDB_reporter_input:
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
class State_reporter_input:
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
    
Reporter_input = typing.Union[
    DCD_reporter_input,
    PDB_reporter_input,
    State_reporter_input
]
