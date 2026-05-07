"""
modules/scales/brownian_dynamics/structures.py

Structures for scales within seekr: molecular_dynamics,
brownian_dynamics, ... - anything that might appear within the 
model.json file.
"""

import os
import glob
import typing

from attrs import define, field, validators, Factory
import cattrs

import seekr.modules.base as base
import seekr.modules.scales.base as scales_base
import seekr.modules.sampling_methods.structures as sampling_methods_structures
import seekr.modules.completion_criteria.structures as compl_crit_struct

sampling_methods_bd \
        = {"NAM": sampling_methods_structures.NAM_bd_sampling_method,
           "WE": sampling_methods_structures.WE_bd_sampling_method,}

@define
class Scale_bd(scales_base.Base_scale):
    """
    A Brownian dynamics scale.
    """
    type: typing.Literal["brownian_dynamics"] = "brownian_dynamics"
    engine: str = field(default="browndye",
                          validator=validators.instance_of(str))
    
    # TODO: remove? Not used anywhere else.
    def has_files_anchor(
            self, 
            anchor: "seekr.modules.structures.Anchor",
            root_directory: str,
            ) -> bool:
        """
        Check if the anchor has the necessary files for this scale.
        """
        # If anchor.bd is True, while anchor.md is False, then this
        # must be the b-surface directory
        b_surface_directory = os.path.join(
            root_directory, anchor.directory)
        assert os.path.exists(b_surface_directory)
        glob_str = "*"
        output_files_glob = os.path.join(b_surface_directory, glob_str)
        if len(glob.glob(output_files_glob)) > 0:
            return True
        else:
            return False
        
    def generate_filetree_anchor(
            self, 
            anchor: "seekr.modules.structures.Anchor",
            root_directory: str,
            anchor_set_name: str = base.PREPARE_SET_NAME
            ) -> None:
        # If anchor.bd is True, while anchor.md is False, then this
        # must be the b-surface directory
        b_surface_dict: dict[str, dict] = {anchor.directory: {}}
        anchor_set_dict: dict[str, dict] = {anchor_set_name: b_surface_dict}
        anchor_sets_dict: dict[str, dict] = {base.ANCHOR_SETS_DIR: anchor_set_dict}
        base.make_tree(root_directory, anchor_sets_dict)

@define
class Stage_bd(scales_base.Base_stage):
    type: typing.Literal["bd_stage"] = "bd_stage"
    scale_type: typing.Literal["brownian_dynamics"] = "brownian_dynamics"
    index: int = field(default=0,
                        validator=[validators.instance_of(int),
                                   validators.ge(0)])
    name: str = field(default="encounter", 
                      validator=validators.instance_of(str))
    n_trajectories_per_output: int = field(default=1000,
                                            validator=[validators.instance_of(int),
                                                         validators.gt(0)])
    n_steps_per_output: int = field(default=100000,
                                    validator=[validators.instance_of(int),
                                               validators.gt(0)])
    sampling_method: sampling_methods_structures.Base_bd_sampling_method = field(
        default=Factory(sampling_methods_structures.NAM_bd_sampling_method))
    completion_criteria: compl_crit_struct.Number_of_trajectories_completion_criteria\
         = field(default=Factory(
             compl_crit_struct.Number_of_trajectories_completion_criteria))
    # Lists that are the bases of filenames that will be used to populate the cvs, selections,
    # and anchor_sets for this stage.
    set_name: str | None = field(
        default=base.PREPARE_SET_NAME, 
        validator=validators.optional(validators.instance_of(str)))
    # TODO: implement restraints in BD?
    #restraints: list[restraint_structures.Restraint] = field(default=Factory(list),)
    input_stage_index: int = field(default=0,
                                    validator=[validators.instance_of(int),
                                               validators.ge(0)])
    scope: str = field(default="unpartitioned",
                       validator=validators.in_(
                           ["unpartitioned", "partitioned"]))
    
    def has_files(
            self,
            model: "Seekr_model",
            anchor_index: int | None = None,
            ) -> bool:
        """
        Returns True if the b_surface directory contains BD results files.

        BD output is written to a single b_surface directory rather than
        per-anchor subdirectories, so anchor_index is accepted for API
        consistency but is not used.
        """
        bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
        if bd_engine is None:
            return False
        b_surface_dir = os.path.join(
            model.directory, "anchor_sets", self.get_output_set_name(),
            bd_engine.b_surface_directory)
        if not os.path.exists(b_surface_dir):
            return False
        output_files_glob = os.path.join(b_surface_dir, bd_engine.results_glob)
        return len(glob.glob(output_files_glob)) > 0

    def load_sets_into_model(
            self, 
            model: "Seekr_model"
            ) -> None:
        """
        Load the CVs, selections, and anchor set for this stage into the model.
        """
        model.load_cvs(self.set_name)
        model.load_selections(self.set_name)
        model.load_anchor_set(self.set_name)
        return
    
    def get_output_set_name(self) -> str:
        """
        Get the name of the set that this stage will output to.
        """
        if self.set_name is None:
            raise ValueError(f"Stage '{self.name}' must have a set name to get output set name.")
        return self.set_name
    
    def is_logistic_stage(self) -> bool:
        """
        Returns false since this is a brownian dynamics stage.
        """
        return False
    
    def add_stage_parameters_to_anchors(
            self, 
            model: "Seekr_model",
            ) -> None:
        """
        Add the stage directory to all anchors in the model.
        """
        for anchor in model.anchors:
            #directory_name: str = self.get_directory_name()
            stage_parameter: dict[str, int | str] = {}
            stage_parameter["index"] = self.index
            #stage_parameter["directory"] = directory_name
            #stage_parameter["positions"] \
            #    = self.sampling_method.get_output_positions_basename_noext()
            anchor.stage_parameters[self.name] = stage_parameter
        return
    
    def add_stage_parameters_to_unpartitioned_settings(
            self, 
            model: "Seekr_model"
            ) -> None:
        """
        Add the stage parameters to the unpartitioned settings.
        """
        bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
        assert bd_engine is not None, "BD stage provided, but "\
            "BD engine settings not provided."
        assert bd_engine.unpartitioned_settings is not None, \
            "BD engine unpartitioned settings not provided, yet BD stage "\
            "is provided."
        #directory_name: str = self.get_directory_name()
        stage_parameter: dict[str, int | str] = {}
        stage_parameter["index"] = self.index
        #stage_parameter["directory"] = directory_name
        #stage_parameter["positions"] \
        #    = self.sampling_method.get_output_positions_basename_noext()
        bd_engine.unpartitioned_settings.stage_parameters[self.name] \
            = stage_parameter
        return
    
def struct_bd_sampling_method(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[sampling_methods_structures.Base_bd_sampling_method]
        ) -> sampling_methods_structures.Base_bd_sampling_method:
    typ = obj.get("type")
    return cattrs.structure(obj, sampling_methods_bd[typ])

def unstruct_bd_sampling_method(
        obj: sampling_methods_structures.Base_bd_sampling_method
        ) -> typing.Dict[str, typing.Any]:
    """
    Unstructure a completion criteria object to a dictionary.
    """
    d = cattrs.unstructure(obj)
    d["type"] = obj.type
    return d

def register_structure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the structure hooks for all scales and their sampling methods.
    """
    converter.register_structure_hook(sampling_methods_structures.Base_bd_sampling_method,
                                        struct_bd_sampling_method)
    return

def register_unstructure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the unstructure hooks for all scales and their sampling methods.
    """
    converter.register_unstructure_hook(sampling_methods_structures.Base_bd_sampling_method,
                                        unstruct_bd_sampling_method)
    return