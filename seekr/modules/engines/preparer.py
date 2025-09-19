"""
modules/engines/preparer.py

Import input classes and functions from the different engines
for easy access here.
"""

import os
import typing
from shutil import copyfile

import numpy as np
from attrs import define, field, validators, Factory
import mdtraj

import seekr.modules.base as base
import seekr.modules.engines.structures as engines_structures

POSITIONS_BASENAME = "positions"

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
        stage_parameters: list[dict | None]
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
        if new_systemwide_settings.stage_parameters[0] is not None:
            systemwide_initial_directory = os.path.join(
                root_directory, self.directory, base.INITIAL_DIRECTORY)
            if self.md_parameters_topology is not None \
                    or (self.positions is not None \
                        and self.positions != ""):
                new_systemwide_settings.generate_filetree(root_directory)
            new_systemwide_settings.stage_parameters[0]["positions"] = POSITIONS_BASENAME
            if toy_mode:
                # MYPY: an error happens here because toy input positions take an array,
                # but non-toy inputs take str (representing the PDB file) - not sure 
                # how best to resolve it.
                for frame_id, frame_positions in enumerate(self.positions):
                    full_positions_path = os.path.join(systemwide_initial_directory,
                                                    f"{POSITIONS_BASENAME}.swarm_{frame_id}.txt")
                    np.savetxt(full_positions_path, np.array(frame_positions))

            else:
                if self.md_parameters_topology is not None:
                    new_systemwide_settings.md_parameters_topology = self.md_parameters_topology
                    self.md_parameters_topology.copy_files(
                        systemwide_initial_directory,
                        new_systemwide_settings.md_parameters_topology)

                if (self.positions is not None) and (self.positions != ""):
                    assert isinstance(self.positions, str), \
                        "MD_systemwide_settings.positions must be a string."
                    mdtraj_obj = mdtraj.load(self.positions)
                    num_frames = mdtraj_obj.n_frames
                    for frame_id in range(num_frames):
                        output_pdb_full_path = os.path.join(
                            systemwide_initial_directory,
                            f"{POSITIONS_BASENAME}.swarm_{frame_id}.pdb")
                        mdtraj_obj[frame_id].save(output_pdb_full_path)

    def get_atom_indices_by_selection_name(
            self,
            selection_name: str
            ) -> list[int]:
        """
        Get the atom indices for a selection by its name.
        """
        for selection in self.selections:
            if selection.name == selection_name:
                return selection.atom_indices
        raise ValueError(f"Selection with name {selection_name} not found.")
                        