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
import seekr.modules.sampling_methods.structures as sampling_methods_structures
import seekr.modules.completion_criteria.structures as compl_crit_struct

sampling_methods_bd \
        = {"NAM": sampling_methods_structures.NAM_bd_sampling_method,
           "WE": sampling_methods_structures.WE_bd_sampling_method,}

@define
class Scale_bd:
    """
    A Brownian dynamics scale.
    """
    type: typing.Literal["brownian_dynamics"] = "brownian_dynamics"
    engine: str = field(default="browndye",
                          validator=validators.instance_of(str))
    
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
            root_directory: str
            ) -> None:
        # If anchor.bd is True, while anchor.md is False, then this
        # must be the b-surface directory
        b_surface_dict: dict[str, dict] = {anchor.directory: {}}
        base.make_tree(root_directory, b_surface_dict)

@define
class Stage_bd:
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
    # TODO: implement restraints in BD?
    #restraints: list[restraint_structures.Restraint] = field(default=Factory(list),)
    input_stage_index: int = field(default=0,
                                    validator=[validators.instance_of(int),
                                               validators.ge(0)])
    input_from: None = None
    output_to: None = None
    
def struct_bd_sampling_method(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[sampling_methods_structures.Base_bd_sampling_method]
        ) -> sampling_methods_structures.Base_bd_sampling_method:
    typ = obj.pop("type")
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