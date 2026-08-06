from __future__ import annotations
"""
modules/engines/preparer.py

Import input classes and functions from the different engines
for easy access here.
"""

import os
import logging

import numpy as np
from attrs import define, field, validators, Factory
import mdtraj

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.engines.structures as engines_structures

logger = logging.getLogger(__name__)

POSITIONS_BASENAME = "positions"

@define
class MD_unpartitioned_settings_input:
    # To be used for a common set of parameters for all anchors
    # within MD simulations
    directory: str = field(default=base.UNPARTITIONED_DIRECTORY, 
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
    
    def make_unpartitioned_settings_for_model(
            self
            ) -> engines_structures.MD_unpartitioned_settings:
        new_md_unpartitioned_settings = engines_structures.MD_unpartitioned_settings()
        new_md_unpartitioned_settings.directory = self.directory
        new_md_unpartitioned_settings.selections = self.selections
        stage_parameters: dict[str, dict | None]
        if self.positions is not None and self.positions != "":
            stage_parameters = \
                {"initial":{"index": 0, "directory": base.INITIAL_DIRECTORY, 
                  "positions": self.positions}}
        else:
            stage_parameters = {"initial": None}
        new_md_unpartitioned_settings.stage_parameters = stage_parameters
        new_md_unpartitioned_settings.md_parameters_topology = self.md_parameters_topology
        return new_md_unpartitioned_settings

    def copy_initial_files(
            self, 
            root_directory: str,
            new_unpartitioned_settings: engines_structures.MD_unpartitioned_settings,
            toy_mode: bool
            ) -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        if new_unpartitioned_settings.stage_parameters["initial"] is not None:
            unpartitioned_initial_directory = os.path.join(
                root_directory, self.directory, base.INITIAL_DIRECTORY)
            if self.md_parameters_topology is not None \
                    or (self.positions is not None \
                        and self.positions != ""):
                new_unpartitioned_settings.generate_filetree(root_directory)
            new_unpartitioned_settings.stage_parameters["initial"]["positions"] = POSITIONS_BASENAME
            if toy_mode:
                # MYPY: an error happens here because toy input positions take an array,
                # but non-toy inputs take str (representing the PDB file) - not sure 
                # how best to resolve it.
                for frame_id, frame_positions in enumerate(self.positions):
                    full_positions_path = os.path.join(unpartitioned_initial_directory,
                                                    f"{POSITIONS_BASENAME}.swarm_{frame_id}.txt")
                    logger.info(f"Writing positions file: {full_positions_path} with "
                                 "positions taken directly from input file.")
                    np.savetxt(full_positions_path, np.array(frame_positions))

            else:
                if self.md_parameters_topology is not None:
                    new_unpartitioned_settings.md_parameters_topology = self.md_parameters_topology
                    positions_pdb = self.positions if isinstance(
                        self.positions, str) else None
                    self.md_parameters_topology.copy_files(
                        unpartitioned_initial_directory,
                        new_unpartitioned_settings.md_parameters_topology,
                        positions_pdb=positions_pdb)

                if (self.positions is not None) and (self.positions != ""):
                    assert isinstance(self.positions, str), \
                        "MD_unpartitioned_settings.positions must be a string."
                    mdtraj_obj = mdtraj.load(self.positions)
                    num_frames = mdtraj_obj.n_frames
                    for frame_id in range(num_frames):
                        output_pdb_full_path = os.path.join(
                            unpartitioned_initial_directory,
                            f"{POSITIONS_BASENAME}.swarm_{frame_id}.pdb")
                        logger.info(f"Writing positions file: {output_pdb_full_path} "
                                     f"by extracting frame from input PDB named: {self.positions}.")
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

def get_spherical_milestones_and_cv_index_in_neighbor_anchor(
        model: structures.Seekr_model,
        neighbor_anchor: structures.Anchor,
        ) -> tuple[list[structures.Milestone], int]:
    cvs = model.collective_variables
    spherical_milestones_in_neighbor_anchor = []
    this_anchors_bd_cv_index = None
    for milestone in neighbor_anchor.milestones:
        found_variable_in_this_milestone = False
        for variable in milestone.variables:
            if variable.startswith("me"):
                cv_index = milestone.get_cv_index_from_variable(variable)
                if cvs[cv_index].type == "distance":
                    spherical_milestones_in_neighbor_anchor.append(
                        milestone)
                    if this_anchors_bd_cv_index is None:
                        this_anchors_bd_cv_index = cv_index
                    else:
                        assert this_anchors_bd_cv_index == cv_index, \
                            "BD anchor is attempting to use multiple cvs to "\
                            "define reaction criteria - not implemented."
                    if not found_variable_in_this_milestone:
                        found_variable_in_this_milestone = True
                    else:
                        raise AssertionError("Milestone has multiple distance CVs"\
                                            " - not implemented")

    assert len(spherical_milestones_in_neighbor_anchor) == 2, \
        "BD anchor must have exactly 2 spherical milestones - "\
        "other arrangements not yet implemented"
    # Sort by the average between the positions of the neighboring anchors
    spherical_milestones_in_neighbor_anchor.sort(
        key=lambda x: x.variables["avg_value"], reverse=True)
    return spherical_milestones_in_neighbor_anchor, this_anchors_bd_cv_index
