"""
modules/completion_criteria/runner.py

Procedures involving completion criteria at runtime.
"""

import os
import glob

import seekr.modules.structures as structures
import seekr.modules.completion_criteria.structures as completion_criteria_structures

def filled_and_unfilled_anchors(
        model: structures.Seekr_model,
        stage: structures.Stage_md
    ) -> tuple[list[int], list[int]]:
    """
    Return two lists: one of the indices of anchors that have been filled
    and one of the indices of anchors that have not been filled.
    """
    filled_anchors = []
    unfilled_anchors = []
    stage_index = stage.index
    stage_output_globs: list[str] = stage.sampling_method.get_output_globs()
    for anchor in model.anchors:
        if not anchor.md:
            continue
        found_outputs_for_anchor = False
        stage_parameter = anchor.stage_parameters[stage_index]
        assert stage_parameter is not None, \
            f"Stage parameter for stage {stage_index} is None."
        assert "directory" in stage_parameter, \
            f"Stage parameter for stage {stage_index} does not contain "\
            +"a directory."
        stage_directory = stage_parameter["directory"]
        for stage_output_glob in stage_output_globs:
            output_files_glob = os.path.join(
                model.directory, anchor.directory, stage_directory, 
                stage_output_glob)
            if len(glob.glob(output_files_glob)) > 0:
                found_outputs_for_anchor = True
        
        if found_outputs_for_anchor:
            filled_anchors.append(anchor.index)
        else:
            unfilled_anchors.append(anchor.index)
    
    return filled_anchors, unfilled_anchors

def is_finished_md(
        model: structures.Seekr_model,
        stage: structures.Stage_md,
        current_step: int,
        return_message: bool = False
    ) -> bool:
    """
    Check if the simulation is finished.
    """
    completion_criteria: completion_criteria_structures.MD_completion_criteria = \
        stage.completion_criteria
    if completion_criteria.type == "number_of_steps":
        message = f"steps {current_step} of {completion_criteria.number_of_steps} completed."
        if current_step >= completion_criteria.number_of_steps:
            if return_message:
                return True, message
            else:
                return True
        else:
            if return_message:
                return False, message
            else:
                return False
    elif completion_criteria.type == "all_anchors_filled":
        filled_anchor_indices, unfilled_anchor_indices = filled_and_unfilled_anchors(
            model, stage)
        if (len(unfilled_anchor_indices) == 0):
            message = "all anchors filled."
            if return_message:
                return True, message
            else:
                return True
        elif current_step >= completion_criteria.get_max_sim_len():
            message = f"Maximum steps exceeded: {current_step} >= {completion_criteria.get_max_sim_len()}."
            if return_message:
                return True, message
            else:
                return True
        else:
            total_number_of_anchors = len(filled_anchor_indices) + \
                len(unfilled_anchor_indices)
            message = f"{len(filled_anchor_indices)} of {total_number_of_anchors} anchors filled."
            if return_message:
                return False, message
            else:
                return False