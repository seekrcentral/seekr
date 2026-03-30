"""
modules/scales/runner.py

Differentiate between different scales and call the appropriate
runner functions for each scale.
"""

import seekr.modules.structures as structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.scales.structures as scales_structures
import seekr.modules.scales.molecular_dynamics.runner as md_runner
import seekr.modules.scales.brownian_dynamics.runner as bd_runner
import seekr.modules.scales.logistic.runner as logistic_runner

def is_finished_with_message(
        model: structures.Seekr_model,
        stage: scales_structures.Stage,
        anchor_index: str | None = None,
        use_json: bool = True
    ) -> tuple[bool, list[str]]:
    """
    Check if the given stage is finished, and return a tuple of (is_finished,
    message).
    """
    if use_json:
        final_message = {}
    else:
        final_message = ""
    if stage.scale_type == "molecular_dynamics":
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        anchor_or_unpartitioned_index_list = md_runner\
            .construct_anchor_or_unpartitioned_index_list(model, stage, anchor_index)
        anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings \
            | structures.Anchor
        for anchor_or_unpartitioned_index in anchor_or_unpartitioned_index_list:
            if anchor_or_unpartitioned_index is None:
                assert md_engine.unpartitioned_settings is not None
                anchor_or_unpartitioned = md_engine.unpartitioned_settings
            else:
                anchor_or_unpartitioned \
                    = model.anchors[anchor_or_unpartitioned_index]
            current_step = md_runner.get_current_md_step(
                anchor_or_unpartitioned, stage.index, 0, model.directory
            )
            finished, message = md_runner.is_finished_md_with_message(
                model, stage, anchor_or_unpartitioned, current_step, use_json=use_json)
            if use_json:
                final_message[anchor_or_unpartitioned_index] = message
            else:
                if anchor_or_unpartitioned_index is None:
                    final_message += f"    Unpartitioned: {message}\n"
                else:
                    final_message += f"    Anchor {anchor_or_unpartitioned_index}: {message}\n"

    elif stage.scale_type == "brownian_dynamics":
        finished, message = bd_runner.is_finished_bd_with_message(
            model, stage, use_json=use_json)
        if use_json:
            final_message[None] = message
        else:
            final_message += f"    Unpartitioned: {message}\n"

    else:
        raise Exception(f"Stage type not recognized: {stage.type}.")
    return finished, final_message

def construct_anchor_or_unpartitioned_index_list(
        model: structures.Seekr_model,
        stage: scales_structures.Stage,
        anchor_index: str
        ) -> list[int | None]:
    """
    Construct a list of anchor or unpartitioned indices based on the stage and
    anchor index.
    """
    if stage.scale_type == "molecular_dynamics":
        return md_runner.construct_anchor_or_unpartitioned_index_list(
            model, stage, anchor_index)
    elif stage.scale_type == "brownian_dynamics":
        raise NotImplementedError("BD stages do not support anchor-specific "\
                                  "operations.")
    else:
        raise Exception(f"Stage type not recognized: {stage.type}.")

def choose_next_simulation(
        model: structures.Seekr_model,
        stage: scales_structures.Stage,
        choose_params: dict,
        force_overwrite: bool,
        swarm_id: int | None = None,
        use_json: bool = True,
        silent: bool = False
        ) -> list:
    """
    Choose the next simulation to run based on the stage scale, and return
    a list of instructions to run in this round.
    """
    if stage.scale_type == "molecular_dynamics":
        anchor_index_instruction = choose_params["anchor_index_instruction"]
        device_index = choose_params["device_index"]
        run_instructions = md_runner.choose_next_simulation_md(
            model, stage, anchor_index_instruction, force_overwrite, 
            device_index, swarm_id, use_json, silent=silent)
    elif stage.scale_type == "brownian_dynamics":
        assert swarm_id is None, "Swarm ID is not used in BD simulations."
        run_instructions = bd_runner.choose_next_simulation_bd(
            model, stage, force_overwrite, use_json, silent=silent)
    elif stage.scale_type == "logistic":
        run_instructions = logistic_runner.choose_next_simulation_logistic(
            model, stage, force_overwrite, use_json, silent=silent)
    else:
        raise Exception(f"Stage type not recognized: {stage.type}.")
    return run_instructions
    
def get_calculation_directory(
        engine: structures.Engine,
        root_directory: str):
    """
    Return the calculation directory for a given BD stage index.
    """
    if engine.scale_type == "brownian_dynamics":
        full_directory = bd_runner.get_calculation_directory_bd(
            engine, root_directory)
    elif engine.scale_type == "molecular_dynamics":
        full_directory = md_runner.get_calculation_directory_md(
            engine, root_directory)
    return full_directory
    
def do_run_instruction(
        model: structures.Seekr_model,
        force_overwrite: bool,
        stage_index: int,
        instruction: list,
        device_index: str | None
    ) -> None:
    """
    Execute the run instructions for a given scale type.
    """
    stage = model.stages[stage_index]
    if stage.scale_type == "molecular_dynamics":
        md_runner.do_run_instruction_md(
            model, force_overwrite, stage_index=stage_index, 
            instruction=instruction, device_index=device_index)
    elif stage.scale_type == "brownian_dynamics":
        bd_runner.do_run_instruction_bd(
            model, force_overwrite, stage_index=stage_index, 
            instruction=instruction)
    elif stage.scale_type == "logistic":
        logistic_runner.do_run_instruction_logistic(
            model, force_overwrite, stage_index=stage_index, 
            instruction=instruction)
    return