from __future__ import annotations
"""
prepare.py

Accept an input file and other arguments and settings for creating a 
SEEKR calculation. Generated files including the model file as well as 
the entire filetree.
"""

import os
import argparse
import logging
import typing

import numpy as np

import seekr._version as seekr_version
import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.preparer as preparer
import seekr.modules.scales.preparer as scale_preparer
import seekr.modules.restraints.structures as restraint_structures
import seekr.modules.check as check

logger = logging.getLogger(__name__)

__version__ = seekr_version.__version__

def check_starting_structures(
        model: structures.Seekr_model,
        root_directory: str
        ) -> None:
    """
    Loop through all stages that have input_stage_index of 0, and 
    a Positional restraint with coordinates_filename defined. For each,
    reload the coordinates file, if frames == 1, accept. Otherwise collect 
    swarm ids with get_starting_structure_swarm_ids on the initial stage 
    (input_stage_index 0, set name main): scope == "unpartitioned": 
    the engine's unpartitioned settings. scope == "partitioned": each anchor 
    that actually has initial swarm files. Anchors with no initial positions 
    are skipped. If no swarm files were written, skip the count check. Those 
    swarms are not defined by an input PDB. If any counts were found, they 
    must all be equal, and n_blocks must equal that count. One file cannot 
    cover anchors whose PDBs have different frame counts. Stages whose input 
    is a later stage, including anchor_starting_structures_from_reporter_stage, 
    are not checked here. Their swarm count is not known until that earlier 
    stage runs.
    """
    for stage in model.stages:
        if stage.input_stage_index != 0 or stage.type != "md_stage":
            continue
        for restraint in stage.restraints:
            if restraint.type != "positional" \
                    or restraint.coordinates_filename is None:
                continue
            coordinates_filename = os.path.join(
                root_directory, restraint.coordinates_filename)
            coordinates_blocks = restraint_structures\
                .load_positional_coordinate_blocks(
                    coordinates_filename, len(restraint.atom_indices))
            n_blocks = coordinates_blocks.shape[0]
            if len(coordinates_blocks) == 1:
                continue
            if stage.scope == "unpartitioned":
                md_engine = model.get_engine_by_scale_type("molecular_dynamics")
                swarm_ids = md_engine.unpartitioned_settings\
                    .get_starting_structure_swarm_ids(root_directory, 0, "dummy")
                counts = [len(swarm_ids)] if swarm_ids else []
            elif stage.scope == "partitioned":
                counts = []
                for anchor in model.anchors:
                    swarm_ids = anchor.get_starting_structure_swarm_ids(
                        root_directory, 0, stage.set_name)
                    if swarm_ids:
                        counts.append(len(swarm_ids))
            if not counts:
                continue
            assert all(count == counts[0] for count in counts), (
                f"Stage '{stage.name}' anchors do not all have "
                f"{counts[0]} initial swarms.")
            assert n_blocks == counts[0], (
                f"Stage '{stage.name}' positional coordinates have {n_blocks} "
                f"blocks, but the initial positions have {counts[0]} swarms.")
            
def prepare(
        seekr_input: preparer.Seekr_input, 
        force_overwrite: bool=False
        ) -> typing.Tuple[structures.Seekr_model, str]:
    """
    Using the Input from the user, prepare the Model
    object and the filetree. Then prepare all building files
    for each anchor and structure the Model.
    """
    curdir = os.getcwd()
    model: structures.Seekr_model = seekr_input.create_model()
    directory = os.path.abspath(
        os.path.expanduser(seekr_input.directory))
    seekr_input.directory = directory
    preparer.generate_directory(directory)
    log_directory = model.get_log_directory(directory)
    preparer.generate_directory(log_directory)
    synopsis_filename = model.get_synopsis_filename(directory)
    base.configure_logging(synopsis_filename, console=False)
    logger.info("Preparing new model.")
    base.log_invocation_metadata(__version__)
    file_path = os.path.join(directory, "model.json")
    preparer.prepare_model_cvs_and_anchors(
        model, seekr_input, force_overwrite)
    base.configure_logging(synopsis_filename)
    for engine_settings in seekr_input.engine_settings_list:
        assert model.get_engine_by_scale_type(engine_settings.scale_type) is not None, \
            f"Engine for scale type {engine_settings.scale_type} is not defined."
        engine_settings.copy_initial_files(
            model, directory, anchor_set_name=base.PREPARE_SET_NAME, 
            using_toy=model.using_toy())
        scale_preparer.generate_files(model, directory, base.PREPARE_SET_NAME, 
                                      engine_settings.scale_type)
    check_starting_structures(model, directory)
    model.save(file_path)
    # Add function to save any cvs, selections, or anchor sets that are defined
    #  in the prepare stage
    model.save_cvs(directory, base.PREPARE_SET_NAME)
    model.save_selections(directory, base.PREPARE_SET_NAME)
    model.save_anchor_set(directory, base.PREPARE_SET_NAME)
    os.chdir(curdir)
    return model, file_path

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "input_file", metavar="INPUT_FILE", type=str, 
        help="The name of input file for a SEEKR calculation.")
    argparser.add_argument(
        "-f", "--force_overwrite", dest="force_overwrite", default=False,
        help="Toggle whether to overwrite existing simulation output files "\
        "within any anchor that might have existed in an old model that would "\
        "be overwritten by generating this new model. If not toggled, this "\
        "program will throw an exception instead of performing any such "\
        "overwrite.", action="store_true")
    argparser.add_argument(
        "-s", "--skip_checks", dest="skip_checks", default=False, 
        help="By default, pre-simulation checks will be run after the "\
        "preparation is complete, and if the checks fail, the SEEKR "\
        "model will not be saved. This argument bypasses those "\
        "checks and allows the model to be generated anyways.", 
        action="store_true")    
    argspace = argparser.parse_args() # parse the args into a dictionary
    args = vars(argspace)
    model_input_filename = args["input_file"]
    force_overwrite = args["force_overwrite"]
    skip_checks = args["skip_checks"]
    # Read the input JSON file into the input object
    seekr_input = preparer.read_input(model_input_filename)
    model, file_path = prepare(seekr_input, force_overwrite)
    if model.directory == ".":
        model_dir = os.path.dirname(file_path)
        model.directory = os.path.abspath(model_dir)
    if not skip_checks:
        # NOTE: model needs to be reloaded to ensure that the plugins are 
        #   properly loaded
        reloaded_model = structures.load_model(file_path)
        check.check_pre_simulation_all(reloaded_model)