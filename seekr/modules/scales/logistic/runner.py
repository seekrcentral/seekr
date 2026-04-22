"""
modules/scales/logistic/runner.py

Perform runner functions specific to logistic stages.
"""

import cattrs
import logging

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.scales.logistic.base as logistic_base
import seekr.modules.scales.logistic.reassign_anchors_logistic_stage \
    as reassign_anchors_logistic_stage
import seekr.modules.scales.logistic\
        .anchor_starting_structures_from_reporter_logistic_stage \
    as anchor_starting_structures_from_reporter_logistic_stage
import seekr.modules.anchor_schemes.preparer as anchor_schemes_preparer

anchor_scheme_types = {
    "voronoi_cells": anchor_schemes_preparer.Anchors_as_voronoi_cells_anchor_scheme,
}

def is_finished_logistic_with_message(
        model: structures.Seekr_model,
        stage: logistic_base.Stage_logistic,
        use_json: bool = True
    ) -> tuple[bool, str | dict]:
    """

    """
    if stage.type == "reassign_anchors_stage":
        return reassign_anchors_logistic_stage\
            .is_finished_reassign_anchors_logistic_with_message(
                model, stage, use_json=use_json)
    elif stage.type == "anchor_starting_structures_from_reporter_stage":
        return anchor_starting_structures_from_reporter_logistic_stage\
            .is_finished_anchor_starting_structures_from_reporter_logistic_with_message(
                model, stage, use_json=use_json)
    else:
        raise Exception(f"Logistic stage type not recognized: {stage.type}.")

def choose_next_simulation_logistic(
        model: structures.Seekr_model, 
        stage: logistic_base.Stage_logistic,
        force_overwrite: bool,
        use_json: bool = True,
        silent: bool = False
        ) -> list:
    if stage.type == "reassign_anchors_stage":
        instructions = reassign_anchors_logistic_stage\
            .choose_next_simulation_reassign_anchors_logistic(
                model, stage, force_overwrite, use_json, silent=silent)
    elif stage.type == "anchor_starting_structures_from_reporter_stage":
        instructions = anchor_starting_structures_from_reporter_logistic_stage\
            .choose_next_simulation_anchor_starting_structures_from_reporter_logistic(
                model, stage, force_overwrite, use_json, silent=silent)
    else:
        raise Exception(f"Logistic stage type not recognized: {stage.type}.")
    
    if not silent:
        stage.write_to_log(model, instructions)
    return instructions
    
def do_run_instruction_logistic(
        model: structures.Seekr_model,
        force_overwrite: bool,
        stage_index: int,
        instruction: list
        ) -> None:
    stage = model.stages[stage_index]
    if stage.type == "reassign_anchors_stage":
        anchor_scheme = None
        if stage.new_anchor_scheme_dict is not None:
            d = stage.new_anchor_scheme_dict
            cattrs.register_structure_hook(base.Positions_type, 
                                           structures.structure_positions_type_union)
            anchor_scheme = cattrs.structure(d, anchor_scheme_types[d["type"]])
        reassign_anchors_logistic_stage\
            .do_run_instruction_reassign_anchors_logistic(
                model, force_overwrite, stage_index=stage_index,
                instruction=instruction, anchor_scheme=anchor_scheme)
    elif stage.type == "anchor_starting_structures_from_reporter_stage":
        anchor_starting_structures_from_reporter_logistic_stage\
            .do_run_instruction_anchor_starting_structures_from_reporter_logistic(
                model, force_overwrite, stage_index=stage_index,
                instruction=instruction)
    else:
        raise Exception(f"Logistic stage type not recognized: {stage.type}.")
    return