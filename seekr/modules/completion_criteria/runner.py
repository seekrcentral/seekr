"""
modules/completion_criteria/runner.py

Procedures involving completion criteria at runtime.
"""

import os
import glob
import importlib

import seekr.modules.structures as structures
import seekr.modules.completion_criteria.structures as completion_criteria_structures
import seekr.modules.engines.structures as engines_structures

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
    stage_output_globs: list[str] = stage.sampling_method.get_output_positions_glob_list()
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
        anchor_or_systemwide: engines_structures.MD_systemwide_settings | structures.Anchor,
        current_step: int,
        swarm_id: int = 0,
    ) -> bool:
    """
    Check if the simulation is finished.
    """
    ending, _ = is_finished_md_with_message(model, stage, anchor_or_systemwide, current_step)
    return ending

def is_finished_md_with_message(
        model: structures.Seekr_model,
        stage: structures.Stage_md,
        anchor_or_systemwide: engines_structures.MD_systemwide_settings | structures.Anchor,
        current_step: int,
        swarm_id: int = 0,
    ) -> tuple[bool, str]:
    """
    Check if the simulation is finished and return with message.
    """
    completion_criteria: completion_criteria_structures.MD_completion_criteria = \
        stage.completion_criteria
    message = None
    if completion_criteria.type == "number_of_steps":
        message = f"steps {current_step} of {completion_criteria.number_of_steps} completed."
        if current_step >= completion_criteria.number_of_steps:
            return True, message
        else:
            return False, message
    elif completion_criteria.type == "all_anchors_filled":
        filled_anchor_indices, unfilled_anchor_indices = filled_and_unfilled_anchors(
            model, stage)
        if (len(unfilled_anchor_indices) == 0):
            message = "all anchors filled."
            return True, message
            
        elif current_step >= completion_criteria.get_max_sim_len():
            message = f"Maximum steps exceeded: {current_step} >= {completion_criteria.get_max_sim_len()}."
            return True, message
        else:
            total_number_of_anchors = len(filled_anchor_indices) + \
                len(unfilled_anchor_indices)
            message = f"{len(filled_anchor_indices)} of {total_number_of_anchors} anchors filled."
            return False, message
    
    else:
        # Run through plugins
        for plugin_name in model.plugins:
            spec = importlib.util.find_spec(plugin_name)
            if spec is None:
                raise ImportError(f"Plugin '{plugin_name}' not found. Please install it or check the spelling.")
            # TODO: see if plugin.modules exists, and if not, raise an error
            #  Then if plugin.modules.completion_criteria doesn't exist, skip quietly
            #  Then if plugin.modules.completion_criteria.runner doesn't exist, raise an error
            plugin_module_comp_crit_runner = importlib.import_module(plugin_name+".modules.completion_criteria.runner")
            found, completed, message = plugin_module_comp_crit_runner.is_finished_md_with_message(
                model, stage, anchor_or_systemwide, current_step, swarm_id)
            if found:
                return completed, message

    assert message is not None, \
        "Completion criteria not found: " + completion_criteria.type