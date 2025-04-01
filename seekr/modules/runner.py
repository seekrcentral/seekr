"""
run_base.py

Base objects and functions for running SEEKR calculations.
"""

import os
import json
import logging

import cattrs
from attrs import define, field, validators

import seekr.modules.base as base
import seekr.modules.engines.structures as engines_structures
import seekr.modules.structures as structures
import seekr.modules.engines.engines_runner as engines_runner

INFORMATION_CHECKPOINT_FILENAME_NOEXT = "checkpoint_information"
INFORMATION_CHECKPOINT_FILENAME_EXT = ".json"
RESTART_CHECKPOINT_FILENAME_NOEXT = "backup"
RESTART_CHECKPOINT_FILENAME_EXT = ".checkpoint"
STEP_TOLERANCE = 5
MAX_RESTART_ATTEMPTS = 3

@define
class Checkpoint_information:
    """
    A class to store information about the current state of a SEEKR
    calculation.
    """
    current_step: int = field(default=0, 
                              validator=[validators.instance_of(int),
                                               validators.ge(0)])
    
    def save(
            self, 
            filename: str
            ) -> None:
        """
        Write the model object to a file.
        """
        checkpoint_information_dict: dict = cattrs.unstructure(self)
        checkpoint_information_dict_noindent: dict \
            = base.wrap_noindent_integers(checkpoint_information_dict)
        json_dump: str = json.dumps(checkpoint_information_dict_noindent,
                                    indent=4, cls=base.NoIndentListEncoder)
        with open(filename, "w") as file:
            file.write(json_dump)

        return
    
def load_checkpoint_information(
        filename: str
        ) -> Checkpoint_information:
    """
    Load a model object from a file.
    """
    with open(filename, "r") as file:
        json_string: str = json.load(file)
    
    # Use a cattrs Converter to define how to structure the dictionary input
    converter: cattrs.Converter = cattrs.Converter()
    checkpoint_information: Checkpoint_information \
        = converter.structure(json_string, Checkpoint_information)
    return checkpoint_information

def get_checkpoint_information_filename(
        swarm_index: int | None
        ) -> str:
    if swarm_index is None:
        checkpoint_information_filename \
            = INFORMATION_CHECKPOINT_FILENAME_NOEXT \
                + INFORMATION_CHECKPOINT_FILENAME_EXT
    else:
        checkpoint_information_filename \
            = INFORMATION_CHECKPOINT_FILENAME_NOEXT \
            + f".swarm_{swarm_index}" \
            + INFORMATION_CHECKPOINT_FILENAME_EXT
    return checkpoint_information_filename

def get_current_md_step(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings \
            | structures.Anchor_input, 
        stage_index: int, 
        swarm_index: int | None,
        root_directory: str):
    """
    Return the current step for a given stage index.
    """
    assert len(anchor_or_systemwide.stage_parameters) > stage_index, "Stage index "\
        "provided is out of range."
    assert anchor_or_systemwide.stage_parameters[stage_index] is not None
    if isinstance(anchor_or_systemwide, structures.Anchor):
        if not anchor_or_systemwide.md:
            raise NotImplementedError("BD current step not yet implemented.")
    stage_directory = anchor_or_systemwide.stage_parameters[stage_index]["directory"]
    full_directory = os.path.join(root_directory, anchor_or_systemwide.directory, stage_directory)
    checkpoint_info_basename = get_checkpoint_information_filename(swarm_index)
    checkpoint_information_filename = os.path.join(
        full_directory, checkpoint_info_basename)
    if os.path.exists(checkpoint_information_filename):
        checkpoint_information = load_checkpoint_information(
            checkpoint_information_filename)
        current_step = checkpoint_information.current_step
    else: 
        current_step = 0
    return current_step

def get_restart_checkpoint_basename(
        swarm_index: int | None
        ) -> str:
    if swarm_index is None:
        restart_checkpoint_filename = RESTART_CHECKPOINT_FILENAME_NOEXT\
            + RESTART_CHECKPOINT_FILENAME_EXT
    else:
        restart_checkpoint_filename = RESTART_CHECKPOINT_FILENAME_NOEXT\
            + str(swarm_index) + RESTART_CHECKPOINT_FILENAME_EXT
    return restart_checkpoint_filename

def get_restart_checkpoint_filename(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings \
            | structures.Anchor_input, 
        stage_index: int, 
        swarm_index: int | None,
        root_directory: str):
    """
    Return the restart checkpoint filename for a given stage index.
    """
    stage_directory = anchor_or_systemwide.stage_parameters[stage_index]["directory"]
    full_directory = os.path.join(root_directory, anchor_or_systemwide.directory, stage_directory)
    restart_checkpoint_basename = get_restart_checkpoint_basename(swarm_index)
    restart_checkpoint_filename = os.path.join(
        full_directory, restart_checkpoint_basename)
    return restart_checkpoint_filename
     
def choose_next_simulation_md(
        stage_md: structures.MD_stage,
        model: structures.Seekr_model, 
        anchor_index: str, 
        force_overwrite: bool,
        device_index: str | None = None
        ) -> list:
    assert model.md_engine is not None, "MD stage provided, but "\
        "MD engine settings not provided."
    if device_index is not None:
        assert model.md_engine.platform_settings is not None
        model.md_engine.platform_settings.device_index = \
            device_index
    
    min_total_sim_len = stage_md.completion_criteria.get_min_sim_len()
    max_total_sim_len = stage_md.completion_criteria.get_max_sim_len()
    
    anchor_info_to_run_unsorted: list = []
    if stage_md.input_from_systemwide_settings:
        systemwide = model.md_engine.systemwide_settings
        num_swarm_frames = systemwide.get_starting_structure_num_frames(
            model, stage_md.input_stage_index)
        for swarm_frame in range(num_swarm_frames):
            if num_swarm_frames <= 1:
                swarm_frame = None
            # TODO
            current_step = get_current_md_step(systemwide,
                stage_md.index, swarm_frame, model.directory)
            restart_checkpoint_filename \
                = get_restart_checkpoint_filename(
                    systemwide, stage_md.index, swarm_frame, model.directory)
            if (not os.path.exists(restart_checkpoint_filename)) \
                    or force_overwrite: # or umbrella_restart_mode:
                    # TODO: extract from sampling method whether restarts will be done
                restart = False
                current_step = 0
            else:
                restart = True
            if current_step > max_total_sim_len:
                # this simulation has run long enough
                continue
            steps_to_go_to_min = min_total_sim_len - current_step
            if steps_to_go_to_min > STEP_TOLERANCE:
                logging.info(
                    "Systemwide has not run the minimum number of steps",
                    current_step, "of", min_total_sim_len, 
                    "in swarm index", swarm_frame)
                total_simulation_length = min_total_sim_len
                num_transitions = 0
                alpha = None
                anchor_info = [steps_to_go_to_min, num_transitions, 
                            alpha, restart, total_simulation_length,
                            swarm_frame]
                anchor_info_to_run_unsorted.append(anchor_info)

    else:
        for alpha, anchor in enumerate(model.anchors):
            if not anchor.md:
                continue
            if anchor_index not in ["any"]:
                try:
                    anchor_index_int = int(anchor_index)
                except ValueError:
                    return []
                if alpha != anchor_index_int:
                    continue
            
            num_swarm_frames = anchor.get_starting_structure_num_frames(
                model, stage_md.input_stage_index)
            
            for swarm_frame in range(num_swarm_frames):
                if num_swarm_frames <= 1:
                    swarm_frame = None
                current_step = get_current_md_step(anchor,
                    stage_md.index, swarm_frame, model.directory)
                restart_checkpoint_filename = get_restart_checkpoint_filename(
                        anchor, stage_md.index, swarm_frame, model.directory)
                if (not os.path.exists(restart_checkpoint_filename)) \
                        or force_overwrite: # or umbrella_restart_mode:
                        # TODO: extract from sampling method whether restarts will be done
                    restart = False
                    current_step = 0
                else:
                    restart = True
                if current_step > max_total_sim_len:
                    # this simulation has run long enough
                    continue
                steps_to_go_to_min = min_total_sim_len - current_step
                steps_to_go_to_min = min_total_sim_len - current_step
                if steps_to_go_to_min > STEP_TOLERANCE:
                    logging.info(
                        "Systemwide has not run the minimum number of steps",
                        current_step, "of", min_total_sim_len, 
                        "in swarm index", swarm_frame)
                    total_simulation_length = min_total_sim_len
                    num_transitions = 0
                    anchor_info = [steps_to_go_to_min, num_transitions, 
                                alpha, restart, total_simulation_length,
                                swarm_frame]
                    anchor_info_to_run_unsorted.append(anchor_info)

    # sort anchors first by how many steps still to run, then by how many
    # transitions have been completed.    
    anchor_info_to_run = sorted(
        anchor_info_to_run_unsorted, 
        key=lambda item: (min_total_sim_len-item[0], item[1]))
    
    return anchor_info_to_run
    
def choose_next_simulation_bd(
            stage_bd: structures.BD_stage, 
            model: structures.Seekr_model, 
            force_overwrite: bool,
            ) -> list:
        assert model.bd_engine is not None, "BD stage provided, but "\
            "BD engine settings not provided."
        # Device_index is currently irrelevant because no BD engine
        #  has been implemented that uses a GPU.
        bulk_index = model.get_bulk_index()
        if stage_bd.completion_criteria.type == "number_of_trajectories":
            num_trajs_needed = stage_bd.completion_criteria.number_of_trajectories
            min_b_surface_encounters = None
            bd_sim_chunk_size = 1
        else:
            raise NotImplementedError("Completion criteria not yet implemented: "\
                                      f"{stage_bd.completion_criteria.type}.")
            # NOTE: min_b_surface_encounters and bd_sim_chunk_size will need
            #  to be set based on the completion criteria.

        if force_overwrite:
            bulk_anchor = model.anchors[bulk_index]
            engines_runner.cleanse_outputs_bd(bulk_anchor, model)

        info_to_run_unsorted: list = []
        bd_transition_counts = model.bd_engine.get_bd_transition_counts(model.directory)
        if "b_surface" not in bd_transition_counts:
            info_to_run_unsorted.append(
                [num_trajs_needed, 0, bulk_index, False, num_trajs_needed])
        else:
            # see how many simulations need to be run in b_surface
            b_surface_sims_so_far = bd_transition_counts["b_surface"]["total"]
            milestone_b_surface_counts = {}
            for key in bd_transition_counts["b_surface"]:
                if isinstance(key, int):
                    milestone_b_surface_counts[key] \
                        = bd_transition_counts["b_surface"][key]
            
            total_b_surface_encounters = min(
                milestone_b_surface_counts.values())
            steps_to_go_minimum = num_trajs_needed - b_surface_sims_so_far
            
            if steps_to_go_minimum > 0:
                info_to_run_unsorted.append(
                    [steps_to_go_minimum, total_b_surface_encounters, \
                     "b_surface", True, b_surface_sims_so_far])
            
            elif min_b_surface_encounters is not None:
                if total_b_surface_encounters < min_b_surface_encounters:
                    total_bd_simulation_length \
                        = (b_surface_sims_so_far // bd_sim_chunk_size + 1) \
                        * bd_sim_chunk_size
                    info_to_run_unsorted.append(
                        [total_bd_simulation_length-b_surface_sims_so_far, \
                         total_b_surface_encounters, "b_surface", True, \
                         b_surface_sims_so_far])
                    
        return info_to_run_unsorted