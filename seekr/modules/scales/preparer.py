"""
modules/scales/preparer.py

Routines for preparing the various scales.
"""

import os
import typing

import seekr.modules.base as base
import seekr.modules.scales.logistic.base as logistic_scale_base
import seekr.modules.structures as structures
import seekr.modules.scales.molecular_dynamics.preparer as md_preparer
import seekr.modules.scales.brownian_dynamics.preparer as bd_preparer
import seekr.modules.scales.logistic.reassign_anchors_logistic_stage \
    as reassign_anchors_logistic_stage
import seekr.modules.scales.logistic\
    .anchor_starting_structures_from_reporter_logistic_stage \
    as anchor_starting_structures_from_reporter_logistic_stage
import seekr.modules.engines.openmm.preparer as openmm_preparer
import seekr.modules.engines.browndye.preparer as browndye_preparer
import seekr.modules.engines.sda.preparer as sda_preparer

Input_stage = typing.Union[md_preparer.Input_stage_md, bd_preparer.Input_stage_bd,
                           logistic_scale_base.Input_stage_logistic,
                           reassign_anchors_logistic_stage.Input_stage_reassign_anchors,
                           anchor_starting_structures_from_reporter_logistic_stage\
                                .Input_stage_anchor_starting_structures_from_reporter,
                           ]

Engine_settings_input = typing.Union[openmm_preparer.Openmm_settings_input, 
                                     browndye_preparer.Browndye_settings_input,
                                     sda_preparer.SDA_settings_input]

def generate_files(
        model: structures.Seekr_model, 
        directory: str,
        anchor_set_name: str,
        scale_type: str
        ) -> None:
    """
    Determine the scale engines and prepare the necessary files for them.
    """
    if scale_type == "brownian_dynamics":
        anchor_set_full_dir = os.path.join(directory, base.ANCHOR_SETS_DIR, anchor_set_name)
        bd_preparer.generate_bd_files(model, anchor_set_full_dir)
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