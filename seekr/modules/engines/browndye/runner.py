"""
modules/engines/browndye/runner.py

Run simulations for Browndye engine.
"""

import os
import re
import glob
import logging

import seekr.modules.engines.browndye.base as browndye_base
import seekr.modules.structures as structures

def cleanse_bd_outputs(directory, results_glob, check_mode=True):
    """
    Check for simulation outputs in an existing directory, optionally
    delete them to make way for new outputs. If 'check_mode' is
    False, then files won't be deleted, but their existence will be
    reported.
    """
    files_will_be_removed = False
    output_files_glob = os.path.join(directory, results_glob)
    output_files_list = glob.glob(output_files_glob)
    for output_file in output_files_list:
        files_will_be_removed = True
        if not check_mode:
            os.remove(output_file)
    
    simulation_file_glob = os.path.join(directory, "*_simulation.xml")
    for simulation_file in glob.glob(simulation_file_glob):
        files_will_be_removed = True
        if not check_mode:
            os.remove(simulation_file)
    
    traj_file_glob = os.path.join(directory, "traj*.xml")
    for traj_file in glob.glob(traj_file_glob):
        files_will_be_removed = True
        if not check_mode:
            os.remove(traj_file)
    return files_will_be_removed

def run_bd_top(
        model: structures.Seekr_model, 
        set_name: str,
        restart: bool = False,
        force_overwrite: bool = False):
    """
    Prepare Browndye input files for simulation.
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    assert bd_engine.type == "browndye"
    bd_directory = os.path.join(model.directory, "anchor_sets", set_name, 
                                bd_engine.b_surface_directory)
    browndye_bin_dir = bd_engine.binary_directory
    curdir = os.getcwd()
    logging.info("moving to directory: " + bd_directory)
    os.chdir(bd_directory)

    simulation_glob = "*_simulation.xml"
    simulation_files = glob.glob(simulation_glob)
    #assert len(simulation_files) == 1, \
    #    "There should be exactly one simulation file in the "\
    #    "browndye directory. Found %d files." % len(simulation_files)
    if len(simulation_files) > 0:
        simulation_filename = simulation_files[0]
        if os.path.exists(simulation_filename) and not restart:
            if force_overwrite:
                logging.warning("force_overwrite set to True: existing files will be "\
                    "overwritten.")
                os.remove(simulation_filename)
            else:
                logging.critical("This anchor already has existing output files and the "\
                    "entered command would overwrite them. If you desire to "\
                    "overwrite the existing files, then use the "\
                    "--force_overwrite (-f) option, and all outputs will be "\
                    "deleted and replaced by a new run.")
                raise Exception("Cannot overwrite existing Browndye outputs.")
            
    bd_command = os.path.join(browndye_bin_dir, "bd_top")
    command = bd_command + " " + browndye_base.BROWNDYE_INPUT_FILENAME
    assert os.path.exists(browndye_base.BROWNDYE_INPUT_FILENAME), \
        "Necessary file doesn't exist: %s" % \
        browndye_base.BROWNDYE_INPUT_FILENAME
    logging.info("running command: " + command)
    os.system(command)
    simulation_glob = "*_simulation.xml"
    simulation_files = glob.glob(simulation_glob)
    assert len(simulation_files) == 1, \
        "There should be exactly one simulation file in the "\
        "browndye directory. Found %d files." % len(simulation_files)
    simulation_filename = simulation_files[0]
    assert os.path.exists(simulation_filename), "Problem occurred running "\
        f"bd_top: simulation file {simulation_filename} was not generated."
    os.chdir(curdir)
    return

def modify_variables(
        model: structures.Seekr_model, 
        n_trajectories: int, 
        set_name: str,
        n_threads: int | None = None, 
        seed: int | None = None, 
        output_file: str | None = None, 
        restart: bool = False, 
        n_trajectories_per_output: int | None = None):
    """
    Modify several variables within the *_simulation.xml file at 
    runtime based on user inputs.
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    #bd_directory = os.path.join(model.directory, bd_engine.b_surface_directory)
    bd_directory = os.path.join(model.directory, "anchor_sets", set_name, 
                                bd_engine.b_surface_directory)
    bd_output_glob = bd_engine.results_glob
    simulation_filename_glob_base = "*_simulation.xml"
    simulation_filename_base = os.path.join(bd_directory, simulation_filename_glob_base)
    simulation_files = glob.glob(simulation_filename_base)
    assert len(simulation_files) == 1, \
        "There should be exactly one simulation file in the "\
        "browndye directory. Found %d files." % len(simulation_files)
    simulation_filename = simulation_files[0]
    #simulation_filename_base = browndye_base.BROWNDYE_RECEPTOR + "_" \
    #    + browndye_base.BROWNDYE_LIGAND + "_simulation.xml"
    #simulation_filename = os.path.join(bd_directory, 
    #                                   simulation_filename_base)
    sim_file_old_lines = []
    sim_file_new_lines = []
    with open(simulation_filename, 'r') as f:
        for line in f.readlines():
            sim_file_old_lines.append(line)
            
    results_file_glob = os.path.join(bd_directory, bd_output_glob)
    results_file_list = glob.glob(results_file_glob)
    if restart:
        assert len(results_file_list) > 0, \
            "No simulation has yet been run: cannot use restart mode."
        restart_index = len(results_file_list) + 1
        if seed is None:
            seed = restart_index
        else:
            seed += restart_index
    else:
        restart_index = 1
        cleanse_bd_outputs(bd_directory, bd_output_glob, check_mode=False)
        
    trajectory_file = f"traj{restart_index}_"
    
    if output_file is None:
        output_file = f"results{restart_index}.xml"
    
    for line in sim_file_old_lines:
        new_line = line
        if n_trajectories is not None:
            new_line = re.sub(r"(?is)<n_trajectories>.+</n_trajectories>", 
                   f"<n_trajectories> {n_trajectories} </n_trajectories>", 
                   new_line)
        
            if n_trajectories_per_output is None:
                n_trajectories_per_output = n_trajectories
            
            new_line = re.sub(r"(?is)<n_trajectories_per_output>.+"\
                              "</n_trajectories_per_output>", 
                   f"<n_trajectories_per_output> {n_trajectories_per_output} "\
                   "</n_trajectories_per_output>", 
                   new_line)
        
        if n_threads is not None:
            new_line = re.sub(r"(?is)<n_threads>.+</n_threads>", 
                   f"<n_threads> {n_threads} </n_threads>", new_line)
            
        if seed is not None:
            new_line = re.sub(r"(?is)<seed>.+</seed>", 
                   f"<seed> {seed} </seed>", new_line)
            
        if output_file is not None:
            new_line = re.sub(r"(?is)<output>.+</output>", 
                   f"<output> {output_file} </output>", new_line)
            
        new_line = re.sub(r"(?is)<trajectory_file>.+</trajectory_file>", 
            f"<trajectory_file> {trajectory_file} </trajectory_file>", 
            new_line)
            
        sim_file_new_lines.append(new_line)
    
    with open(simulation_filename, 'w') as f:
        for line in sim_file_new_lines:
            f.write(line)
            
    return

def run_nam_simulation(
        model: structures.Seekr_model,
        set_name: str
    ) -> None:
    """
    Run the Browndye2 simulation.
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    #bd_directory = os.path.join(model.directory, bd_engine.b_surface_directory)
    bd_directory = os.path.join(model.directory, "anchor_sets", set_name, 
                                bd_engine.b_surface_directory)
    browndye_bin_dir = bd_engine.binary_directory
    bd_output_glob = bd_engine.results_glob
    curdir = os.getcwd()
    logging.info("moving to directory: " + bd_directory)
    os.chdir(bd_directory)
    bd_command = os.path.join(browndye_bin_dir, "nam_simulation")
    simulation_glob = "*_simulation.xml"
    simulation_files = glob.glob(simulation_glob)
    assert len(simulation_files) == 1, \
        "There should be exactly one simulation file in the "\
        "browndye directory. Found %d files." % len(simulation_files)
    simulation_filename = simulation_files[0]
    #simulation_filename = browndye_base.BROWNDYE_RECEPTOR + "_" \
    #    + browndye_base.BROWNDYE_LIGAND + "_simulation.xml"
    command = bd_command + " " + simulation_filename
    logging.info("running command: " + command)
    os.system(command)
    results_glob = glob.glob(bd_output_glob)
    assert len(results_glob) > 0, "Problem occurred running "\
        "nam_simulation: results file was not generated."
    os.chdir(curdir)
    return

def run_simulation(
        model: structures.Seekr_model, 
        n_trajectories: int, 
        restart: bool, 
        set_name: str,
        force_overwrite: bool = False
    ) -> None:
    """
    Determine which bd engine to run and run it
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    assert bd_engine.type == "browndye"
    run_bd_top(
        model, set_name, restart, force_overwrite)
    n_trajectories_per_output = browndye_base.DEFAULT_N_TRAJ_PER_OUT
    if n_trajectories is not None:
        if n_trajectories_per_output > n_trajectories:
            n_trajectories_per_output = n_trajectories
    print("n_trajectories:", n_trajectories)
    modify_variables(
        model, n_trajectories, set_name, restart=restart, 
        n_trajectories_per_output=n_trajectories_per_output)
    run_nam_simulation(model, set_name)
    return