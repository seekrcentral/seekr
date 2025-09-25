"""
modules/scales/preparer.py

Routines for preparing the various scales.
"""

import typing

from attrs import define, field, validators, Factory

import seekr.modules.structures as structures
import seekr.modules.scales.structures as scale_struct
import seekr.modules.scales.molecular_dynamics.preparer as md_preparer
import seekr.modules.scales.brownian_dynamics.preparer as bd_preparer
import seekr.modules.engines.openmm.preparer as openmm_preparer
import seekr.modules.engines.browndye.preparer as browndye_preparer
import seekr.modules.engines.sda.preparer as sda_preparer

Input_stage = typing.Union[md_preparer.Input_stage_md, bd_preparer.Input_stage_bd]
Engine_settings_input = typing.Union[openmm_preparer.Openmm_settings_input, 
                                     browndye_preparer.Browndye_settings_input,
                                     sda_preparer.SDA_settings_input]

@define
class Input_stage_placeholder:
    """
    A placeholder stage allows an external program, such as QMrebind
    or a string method, for example, to take the stage's input structures
    and provide new output structures for later stages.
    """
    type: typing.Literal["placeholder_stage"] = "placeholder_stage"
    name: str = field(default="placeholder", 
                      validator=validators.instance_of(str))
    output_positions_basename_noext: str = field(
        default="output_positions", 
        validator=validators.instance_of(str))
    output_positions_glob_list: list[str] = field(
        default=Factory(lambda: ["*.pdb", "*.txt"]), 
        validator=validators.instance_of(list))

    def get_scale_type(self) -> str:
        return "unknown"

    def make_model_stage(
            self, 
            index: int,
            systemwide_stage_input: bool,
            input_from_systemwide_settings_output_to_anchors: bool,
            input_from_anchors_output_to_systemwide_settings: bool,
            ) -> typing.Tuple[scale_struct.Stage_placeholder, bool]:
        """
        Given this input stage object, create the model's Stage object.
        """
        stage = scale_struct.Stage_placeholder()
        stage.index = index
        stage.name = self.name
        if systemwide_stage_input: # If coming from systemwide settings
            stage.input_from = "systemwide"
            if input_from_systemwide_settings_output_to_anchors:
                new_systemwide_stage_input = False
                stage.output_to = "anchors"
                # Handle different situations?
            else:
                new_systemwide_stage_input = True
                stage.output_to = "systemwide"
        else:
            stage.input_from = "anchors"
            if input_from_anchors_output_to_systemwide_settings:
                new_systemwide_stage_input = True
                stage.output_to = "systemwide"
            else:
                new_systemwide_stage_input = False
                stage.output_to = "anchors"
        return stage, new_systemwide_stage_input

def generate_files(
        model: structures.Seekr_model, 
        directory: str,
        scale_type: str
        ) -> None:
    """
    Determine the scale engines and prepare the necessary files for them.
    """
    if scale_type == "brownian_dynamics":
        bd_preparer.generate_bd_files(model, directory)
    elif scale_type == "molecular_dynamics":
        pass
    else:
        raise Exception(f"Scale type {scale_type} not recognized.")
    
def register_structure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the structure hooks for all scales and their sampling methods.
    """
    md_preparer.register_structure_hooks(converter)
    bd_preparer.register_structure_hooks(converter)
    
def get_sampling_methods_input() -> typing.Dict[str, typing.Dict[str, typing.Any]]:
    return {
        "molecular_dynamics": md_preparer.sampling_methods_input_md,
        "brownian_dynamics": bd_preparer.sampling_methods_input_bd,
    }

def update_sampling_methods_input(
        new_sampling_methods_input: typing.Dict[str, typing.Dict[str, typing.Any]]
        ) -> None:
    """
    Update the sampling methods input dictionaries for all scales.
    """
    md_preparer.sampling_methods_input_md.update(
        new_sampling_methods_input.get("molecular_dynamics", {}))
    bd_preparer.sampling_methods_input_bd.update(
        new_sampling_methods_input.get("brownian_dynamics", {}))