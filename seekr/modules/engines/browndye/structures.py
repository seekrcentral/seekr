"""
modules/engines/browndye/structures.py

Structures for the Browndye engine - anything that might appear in the model.json.
"""

import os
import typing
import glob
import logging

from attrs import define, field, validators, Factory
from attrs.validators import instance_of, optional, gt, ge

import seekr.modules.base as base
import seekr.modules.engines.browndye.base as browndye_base
import seekr.modules.engines.structures as engines_structures

@define
class Browndye_molecule:
    name: str = field(default="", 
                        validator=instance_of(str))
    pqr_filename: str = field(default="",
                            validator=instance_of(str))
    pqrxml_filename: str | None = field(default=None,
                            validator=optional(instance_of(str)))
    original_indices: typing.Dict[str, typing.List[int]] = field(
        default=Factory(dict),)
    ghost_site_indices: typing.Dict[str, int] | None = field(
        default=None, validator=optional(instance_of(dict)))
    
    def generate_pqrxml_file(
            self, 
            directory: str,
            browndye_bin: str,
            output_xml_filename: str | None = None,
            ) -> None:
        """
        Generate the pqrxml file for this molecule
        """
        input_pqr_filename = os.path.join(directory, self.pqr_filename)
        if output_xml_filename is None:
            output_xml_filename = ".".join(os.path.splitext(input_pqr_filename)[:-1]) + ".xml"
        pqr2xml_binary = os.path.join(browndye_bin, "pqr2xml")
        pqr2xml_command = pqr2xml_binary + " < " + input_pqr_filename + " > " \
            + output_xml_filename
        # NOTE: to filter out CRYST and other problematic lines
        #pqr2xml_command = "grep -e ATOM -e HETATM " + input_pqr_filename + " | " \
        #    + pqr2xml_binary + " > " + output_xml_filename
        logging.info("running command: " + pqr2xml_command)
        os.system(pqr2xml_command)
        assert os.path.exists(output_xml_filename), "Problem generating XML from "\
            f"PQR file: file not created: {output_xml_filename}"
        self.pqrxml_filename = os.path.basename(output_xml_filename)
        return

@define
class Browndye_unpartitioned_settings:
    # To be used for a common set of parameters for all anchors
    # including bulk anchor, within BD simulations
    molecules: list[Browndye_molecule] = field(default=Factory(list),
                    validator=validators.deep_iterable(
                        member_validator=instance_of(
                            Browndye_molecule),
                        iterable_validator=instance_of(list),
                    ))

@define
class Browndye_bd_engine:
    type: typing.Literal["browndye"] = "browndye"
    scale_type: typing.Literal["brownian_dynamics"] = "brownian_dynamics"
    binary_directory: str = field(default="",
                                    validator=instance_of(str))
    apbs_grid_spacing: float = field(default=0.5,
                                    validator=[instance_of(float),
                                               gt(0)])
    
    ions: list[engines_structures.Ion] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=instance_of(
                                    engines_structures.Ion),
                                iterable_validator=instance_of(list),
                            ))
    debye_length: float = field(default=9e9,
                                validator=[instance_of(float), gt(0)])
    random_seed: int = field(default=0,
                             validator=[instance_of(int), ge(0)])
    b_surface_directory: str = field(default="b_surface",
                                    validator=instance_of(str))
    
    results_glob: str = field(default="results*.xml",
                            validator=instance_of(str))
    
    unpartitioned_settings: Browndye_unpartitioned_settings | None = None

    def get_bd_transition_counts(
            self, 
            root_directory: str,
            set_name: str
            ) -> dict:
        """
        Get the transition counts from the BD results files.
        """
        output_file_glob = os.path.join(
            root_directory, "anchor_sets", set_name, 
            self.b_surface_directory, self.results_glob)
        output_file_list = glob.glob(output_file_glob)
        output_file_list = base.order_files_numerically(output_file_list)
        bd_transition_counts = {}
        if len(output_file_list) > 0:
            transition_counts = browndye_base.compute_transition_counts(output_file_list)
            bd_transition_counts["b_surface"] = transition_counts
        return bd_transition_counts
    
    def modify_bd_transition_counts(
            self, 
            bd_transition_counts: dict, 
            lower_milestone_index: int,
            upper_milestone_index: int,
            ) -> None:
        """
        Dont' need to modify anything for Browndye.
        """
        
        return bd_transition_counts
    
    def get_random_seed(
            self
            ) -> int:
        """
        Get the random seed for the Browndye engine.
        """
        return self.random_seed

    def set_random_seed(
            self, 
            seed: int
            ) -> None:
        """
        Set the random seed for the Browndye engine.
        """
        self.random_seed = seed
        return
    
    def can_handle_toy(
            self
            ) -> bool:
        """
        Can this engine handle toy systems?
        """
        return False
    
    def save_selections(
            self, 
            directory: str, 
            ) -> None:
        """
        Save the selections to files.
        """
        # Does nothing for now - there are no selections to save for Browndye
        return
    
    def load_selections(
            self,
            directory: str,
            ) -> None:
        """
        Load the selections from a directory.
        """
        # Does nothing for now - there are no selections to load for Browndye
        return
    
    def get_selection_names(
            self,
            ) -> list[str]:
        """
        Get the names of all selections.
        """
        selection_names = []
        # Does nothing for now
        return selection_names