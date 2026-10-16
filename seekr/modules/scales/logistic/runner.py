from __future__ import annotations
"""
modules/scales/logistic/runner.py

Perform runner functions specific to logistic stages.
"""

import cattrs
import logging
import importlib
import importlib.util

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
        anchor_or_unpartitioned: structures.Anchor | None,
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
                model, stage, anchor_or_unpartitioned, use_json=use_json)
    else:
        # Run through plugins
        for plugin_name in model.plugins:
            spec = importlib.util.find_spec(plugin_name)
            if spec is None:
                raise ImportError(f"Plugin '{plugin_name}' not found. Please install "\
                                  "it or check the spelling.")
            runner_module_name = plugin_name+".modules.scales.logistic.runner"
            if importlib.util.find_spec(runner_module_name) is None:
                continue
            plugin_module_scales_logistic_runner = importlib.import_module(
                runner_module_name)
            found, completed, message = plugin_module_scales_logistic_runner\
                .is_finished_logistic_with_message(
                    model, stage, anchor_or_unpartitioned, use_json=use_json)
            if found:
                return completed, message
        
    raise Exception(f"Logistic stage type not recognized: {stage.type}.")

def choose_next_simulation_logistic(
        model: structures.Seekr_model, 
        stage: logistic_base.Stage_logistic,
        force_overwrite: bool,
        use_json: bool = True,
        silent: bool = False
        ) -> list:
    instructions = None
    if stage.type == "reassign_anchors_stage":
        instructions = reassign_anchors_logistic_stage\
            .choose_next_simulation_reassign_anchors_logistic(
                model, stage, force_overwrite, use_json, silent=silent)
    elif stage.type == "anchor_starting_structures_from_reporter_stage":
        instructions = anchor_starting_structures_from_reporter_logistic_stage\
            .choose_next_simulation_anchor_starting_structures_from_reporter_logistic(
                model, stage, force_overwrite, use_json, silent=silent)
    else:
        # Run through plugins
        for plugin_name in model.plugins:
            spec = importlib.util.find_spec(plugin_name)
            if spec is None:
                raise ImportError(f"Plugin '{plugin_name}' not found. Please install "\
                                  "it or check the spelling.")
            runner_module_name = plugin_name+".modules.scales.logistic.runner"
            if importlib.util.find_spec(runner_module_name) is None:
                continue
            plugin_module_scales_logistic_runner = importlib.import_module(
                runner_module_name)
            found, instructions = plugin_module_scales_logistic_runner\
                .choose_next_simulation_logistic(
                    model, stage, force_overwrite, use_json=use_json, silent=silent)
            if found:
                break

    if instructions is None:
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
        # Run through plugins
        for plugin_name in model.plugins:
            spec = importlib.util.find_spec(plugin_name)
            if spec is None:
                raise ImportError(f"Plugin '{plugin_name}' not found. Please install "\
                                  "it or check the spelling.")
            runner_module_name = plugin_name+".modules.scales.logistic.runner"
            if importlib.util.find_spec(runner_module_name) is None:
                continue
            plugin_module_scales_logistic_runner = importlib.import_module(
                runner_module_name)
            found = plugin_module_scales_logistic_runner\
                .do_run_instruction_logistic(
                    model, force_overwrite, stage_index, instruction)
            if found:
                break
    return