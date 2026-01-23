"""
prepare.py

Accept an input file and other arguments and settings for creating a 
SEEKR calculation. Generated files including the model file as well as 
the entire filetree.
"""

import os
import sys
import argparse
import logging
import typing
import getpass
import socket

import seekr._version as seekr_version
import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.preparer as preparer
import seekr.modules.scales.preparer as scale_preparer
import seekr.modules.check as check

__version__ = seekr_version.__version__

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
    base.configure_file_only_logger(synopsis_filename, overwrite=False)
    full_command = " ".join(sys.argv)
    logging.info(f"Preparing new model with command: {full_command}")
    logging.info(f"hostname: {socket.gethostname()}. username: {getpass.getuser()}. "\
                 f"SEEKR version: {__version__}")
    file_path = os.path.join(directory, "model.json")
    if len(seekr_input.partitions) > 0:
        preparer.prepare_model_cvs_and_anchors(
            model, seekr_input, force_overwrite)
    else:
        # TODO: decide whether to create an empty partition if not are passed,
        # or make do with no partitions
        raise Exception("Partition must be defined in input file, "\
                        "empty partitions are not yet implemented.")
    
    base.configure_file_and_stdout_logger(synopsis_filename, overwrite=False)
    for settings in seekr_input.settings_list:
        assert model.get_engine_by_scale_type(settings.scale_type) is not None, \
            f"Engine for scale type {settings.scale_type} is not defined."
        settings.copy_initial_files(model, directory, using_toy=model.using_toy())
        scale_preparer.generate_files(model, directory, base.PREPARE_SET_NAME, 
                                      settings.scale_type)
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
        check.check_pre_simulation_all(model)