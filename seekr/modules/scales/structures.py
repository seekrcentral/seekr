"""

"""

import os
import glob
import typing
from shutil import rmtree

from attrs import define, field, validators, Factory

import seekr.modules.structures as structures
import seekr.modules.scales.molecular_dynamics.structures as md_structures
import seekr.modules.scales.brownian_dynamics.structures as bd_structures

Scale = typing.Union[md_structures.Scale_md, bd_structures.Scale_bd]
Stage = typing.Union[md_structures.Stage_md, bd_structures.Stage_bd]

@define
class Stage_placeholder:
    """
    A placeholder stage allows an external program, such as QMrebind
    or a string method, for example, to take the stage's input structures
    and provide new output structures for later stages.
    """
    type: typing.Literal["placeholder_stage"] = "placeholder_stage"
    index: int = field(default=0,
                        validator=[validators.instance_of(int),
                                      validators.ge(0)])
    name: str = field(default="placeholder", 
                      validator=validators.instance_of(str))
    output_positions_basename_noext: str = field(
        default="output_positions", 
        validator=validators.instance_of(str))
    output_positions_glob_list: list[str] = field(
        default=Factory(lambda: ["*.pdb", "*.txt"]), 
        validator=validators.instance_of(list))
    input_stage_index: int = field(default=0,
                                      validator=[validators.instance_of(int),
                                                 validators.ge(0)])
    input_from: str = field(default="systemwide", 
                            validator=validators.instance_of(str))
    output_to: str = field(default="anchor", 
                           validator=validators.instance_of(str))
    
    def get_directory_name(self) -> str:
        """
        Get the directory of this stage.
        """
        directory_name: str = f"stage_{self.index}_{self.name}"
        return directory_name
    
    def add_stage_parameters_to_anchors(
            self, 
            model: "Seekr_model",
            do_none: bool = False
            ) -> None:
        """
        Add the stage directory to all anchors in the model.
        """
        for anchor in model.anchors:
            if do_none:
                anchor.stage_parameters.append(None)
                continue
            directory_name: str = self.get_directory_name()
            stage_parameter: dict[str, int | str] = {}
            stage_parameter["index"] = self.index
            stage_parameter["directory"] = directory_name
            stage_parameter["positions"] = self.output_positions_basename_noext
            anchor.stage_parameters.append(stage_parameter)
        return
    
    def add_stage_parameters_to_systemwide_settings(
            self, 
            systemwide_settings: typing.Any,
            do_none: bool = False
            ) -> None:
        """
        Add the stage parameters to the systemwide settings.
        """
        if do_none:
            systemwide_settings.stage_parameters.append(None)
            return
        directory_name: str = self.get_directory_name()
        stage_parameter: dict[str, int | str] = {}
        stage_parameter["index"] = self.index
        stage_parameter["directory"] = directory_name
        stage_parameter["positions"] = self.output_positions_basename_noext
        systemwide_settings.stage_parameters.append(stage_parameter)
        return
    
    def has_files(
            self, 
            model: "Seekr_model"
            ) -> bool:
        """
        Returns true if any of the stage directories contain
        outputs from any of the stages.
        """
        stage_output_globs: list[str] = self.output_positions_glob_list
        my_directory = self.get_directory_name()
        for anchor in model.anchors:
            for stage_parameter in anchor.stage_parameters:
                if stage_parameter is None:
                    continue
                stage_directory = stage_parameter["directory"]
                if stage_directory == my_directory:
                    for stage_output_glob in stage_output_globs:
                        output_files_glob = os.path.join(
                            model.directory, anchor.directory, stage_directory, 
                            stage_output_glob)
                        if len(glob.glob(output_files_glob)) > 0:
                            return True
        return False
    
    def remove_directory(
            self, 
            model: "Seekr_model"
            ) -> None:
        """
        Remove entire directory in all anchors and systemwide settings for this stage.
        """
        my_directory = self.get_directory_name()
        for anchor in model.anchors:
            full_directory = os.path.join(model.directory, anchor.directory, my_directory)
            if os.path.exists(full_directory):
                rmtree(full_directory)
        return

def register_structure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the structure hooks for all scales and their sampling methods.
    """
    md_structures.register_structure_hooks(converter)
    bd_structures.register_structure_hooks(converter)
    return

def register_unstructure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the unstructure hooks for all scales and their sampling methods.
    """
    md_structures.register_unstructure_hooks(converter)
    bd_structures.register_unstructure_hooks(converter)
    return

def get_sampling_methods() -> typing.Dict[str, typing.Dict[str, typing.Any]]:
    return {
        "molecular_dynamics": md_structures.sampling_methods_md,
        "brownian_dynamics": bd_structures.sampling_methods_bd,
    }

def update_sampling_methods(
        new_sampling_methods: typing.Dict[str, typing.Dict[str, typing.Any]]
        ) -> None:
    """
    Update the sampling methods input dictionaries for all scales.
    """
    md_structures.sampling_methods_md.update(
        new_sampling_methods.get("molecular_dynamics", {}))
    bd_structures.sampling_methods_bd.update(
        new_sampling_methods.get("brownian_dynamics", {}))