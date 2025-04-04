"""
modules/engines/preparer.py

Import input classes and functions from the different engines
for easy access here.
"""

import os
from shutil import copyfile

from attrs import define, field, validators, Factory

import seekr.modules.base as base
import seekr.modules.engines.structures as engines_structures

@define
class MD_systemwide_settings_input:
    # To be used for a common set of parameters for all anchors
    # within MD simulations
    directory: str = field(default=base.SYSTEMWIDE_DIRECTORY, 
                           validator=validators.instance_of(str))
    md_parameters_topology: engines_structures.Parameters_topology | None = None
    # This can be a position assignment to be automatically placed into an anchor
    positions: base.Positions_type = None
    selections: list[engines_structures.Selection] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(engines_structures.Selection),
            iterable_validator=validators.instance_of(list),
        ))
    
    def make_systemwide_settings_for_model(
            self
            ) -> engines_structures.MD_systemwide_settings:
        new_md_systemwide_settings = engines_structures.MD_systemwide_settings()
        new_md_systemwide_settings.directory = self.directory
        new_md_systemwide_settings.selections = self.selections
        if self.positions is not None and self.positions != "":
            stage_parameters = \
                [{"index": 0, "directory": base.INITIAL_DIRECTORY, 
                  "positions": self.positions}]
        else:
            stage_parameters = [None]
        new_md_systemwide_settings.stage_parameters = stage_parameters
        new_md_systemwide_settings.md_parameters_topology = self.md_parameters_topology
        return new_md_systemwide_settings

    def copy_initial_files(
            self, 
            root_directory: str,
            new_systemwide_settings: engines_structures.MD_systemwide_settings,
            toy_mode: bool
            ) -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        systemwide_initial_directory = os.path.join(
            root_directory, self.directory, base.INITIAL_DIRECTORY)
        if toy_mode:
            pass

        else:
            if self.md_parameters_topology is not None \
                    or (self.positions is not None \
                        and self.positions != ""):
                new_systemwide_settings.generate_filetree(root_directory)

            if self.md_parameters_topology is not None:
                new_systemwide_settings.md_parameters_topology = self.md_parameters_topology
                self.md_parameters_topology.copy_files(
                    systemwide_initial_directory,
                    new_systemwide_settings.md_parameters_topology)

            if (self.positions is not None) and (self.positions != ""):
                assert isinstance(self.positions, str), \
                    "MD_systemwide_settings.positions must be a string."
                pdb_basename = os.path.basename(self.positions)
                anchor_pdb_full_path = os.path.join(systemwide_initial_directory,
                                                    pdb_basename)
                copyfile(os.path.expanduser(self.positions), anchor_pdb_full_path)
                new_systemwide_settings.stage_parameters[0]["positions"] = pdb_basename