from __future__ import annotations
"""
modules/scales/brownian_dynamics/preparer.py

Routines for preparing the Brownian Dynamics scale.
"""

import typing

from attrs import define, field, validators, Factory
import cattrs

import seekr.modules.scales.base as scales_base
import seekr.modules.structures as structures
import seekr.modules.completion_criteria.structures as compl_crit_struct
import seekr.modules.scales.brownian_dynamics.structures as scale_bd
import seekr.modules.engines.browndye.preparer as browndye_preparer
import seekr.modules.engines.sda.preparer as sda_preparer
import seekr.modules.sampling_methods.preparer as sampling_methods_preparer

BD_settings_input = typing.Union[browndye_preparer.Browndye_settings_input,
                                sda_preparer.SDA_settings_input]

sampling_methods_input_bd: typing.Dict[str, typing.Any] \
        = {"NAM": sampling_methods_preparer.NAM_bd_input,
           "WE": sampling_methods_preparer.WE_bd_input,}

@define
class Input_stage_bd(scales_base.Base_input_stage):
    """
    A BD stage of the SEEKR calculation.
    """
    type: typing.Literal["bd_stage"] = "bd_stage"
    name: str = field(default="BD",
                        validator=validators.instance_of(str))
    input_stage_name: str | None = field(default=None,
                                 validator=validators.optional(
                                     validators.instance_of(str)))
    n_trajectories_per_output: int = field(default=1000,
                                            validator=[validators.instance_of(int),
                                                         validators.gt(0)])
    n_steps_per_output: int = field(default=100000,
                                    validator=[validators.instance_of(int),
                                               validators.gt(0)])
    sampling_method: sampling_methods_preparer.Base_bd_sampling_method_input = field(
        default=Factory(sampling_methods_preparer.NAM_bd_input))
    completion_criteria: compl_crit_struct.Base_completion_criteria | None = None
    
    def get_scale_type(self) -> str:
        return "brownian_dynamics"

    def make_model_stage(
            self, 
            index: int,
            scope: str,
            bd_settings: BD_settings_input,
            dummy_running_dir: str = "",
            ) -> scale_bd.Stage_bd:
        """
        Given this input stage object, create the model's Stage object.
        """
        stage = scale_bd.Stage_bd()
        stage.index = index
        stage.name = self.name
        stage.n_trajectories_per_output = self.n_trajectories_per_output
        stage.n_steps_per_output = self.n_steps_per_output
        assert self.completion_criteria is not None
        stage.completion_criteria = self.completion_criteria
        # At present, BD stages will always be unpartitioned
        #stage.input_from = None
        #stage.output_to = None
        # TODO: implement restraints in BD - that will use bd_settings argument
        return stage

def generate_bd_files(
        model: structures.Seekr_model, 
        directory: str,
        ) -> None:
    """
    Determine the BD engine and prepare the necessary files for it.
    """
    found_function = False
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    assert bd_engine is not None
    if bd_engine.type == "browndye":
        browndye_preparer.generate_bd_files(model, directory)
        found_function = True
    if bd_engine.type == "sda":
        sda_preparer.generate_bd_files(model, directory)
        found_function = True
    else:
        # implement check of installed plugins for other BD engines
        # Loop over modules:
        # found_function = function_in_module(...)
        pass

    if not found_function:
        raise Exception(f"BD engine type {bd_engine.type} not recognized.")
    
def struct_bd_sampling_method_inputs(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[sampling_methods_preparer.Base_bd_sampling_method_input]
        ) -> sampling_methods_preparer.Base_bd_sampling_method_input:
    typ = obj.get("type")
    return cattrs.structure(obj, sampling_methods_input_bd[typ])

def register_structure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the structure hooks for all scales and their sampling methods.
    """
    converter.register_structure_hook(sampling_methods_preparer.Base_bd_sampling_method_input,
                                        struct_bd_sampling_method_inputs)