"""
modules/engines/sda/runner.py

Run simulations for SDA engine.
"""

import os
import re
import glob
import logging

import seekr.modules.engines.sda.base as sda_base
import seekr.modules.structures as structures
import seekr.modules.engines.sda.structures as sda_structures

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
    
    restart_input_glob = os.path.join(directory, "sda_*.in")
    for restart_input_file in glob.glob(restart_input_glob):
        files_will_be_removed = True
        if not check_mode:
            os.remove(restart_input_file)

    """ #TODO: add SDA output files here
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
    """
    return files_will_be_removed

def modify_variables(bd_milestone_directory, bd_output_glob, 
                     n_trajectories, seed, restart=False):
    """
    Modify several variables within the sda*.out file at 
    runtime based on user inputs.
    
    Parameters
    ----------
    bd_milestone_directory : str
        The path to the directory holding the simulation XML.
        
    bd_output_glob : str
        A glob used to detect the presence of existing BD results
        files. If they are not detected, restart mode may not be used.
        
    n_trajectories : None or int
        The parameter <n_trajectories> represents the total number of
        individual BD trajectories that will be run per invocation of
        the 'nam_simulation' program. If left at None, then the 
        existing value of <n_trajectories> will be left alone in the 
        simulation file. Otherwise, if changed to an integer, the 
        value for <n_trajectories> will be modified in the simulation
        file.
        
    seed : None or int
        The parameter <seed> represents the random number seed which 
        will be used in BD. If left at None, then the 
        existing value of <seed> will be left alone in the 
        simulation file. Otherwise, if changed to an integer, the 
        value for <seed> will be modified in the simulation
        file.
        
    restart : bool, Default False
        Whether or not this file is being modified for a restart 
        simulation. If set to True, the 'seed' will be automatically
        incremented to generate randomness.
    """
    simulation_basename = sda_base.SDA_INPUT_BASENAME
    simulation_filename = os.path.join(bd_milestone_directory, 
                                       simulation_basename)
    sim_file_old_lines = []
    sim_file_new_lines = []
    with open(simulation_filename, 'r') as f:
        for line in f.readlines():
            sim_file_old_lines.append(line)
            
    results_file_glob = os.path.join(bd_milestone_directory, bd_output_glob)
    results_file_list = glob.glob(results_file_glob)
    if restart:
        assert len(results_file_list) > 0, \
            "No simulation has yet been run: cannot use restart mode."
        restart_index = len(results_file_list) + 1
        seed += restart_index
    else:
        restart_index = 1
        cleanse_bd_outputs(bd_milestone_directory, bd_output_glob, check_mode=False)
        
    trajectory_file = "traj_%d" % restart_index
    complexes_file = "complexes_%d" % restart_index
    
    for line in sim_file_old_lines:
        new_line = line
        if n_trajectories is not None:
            new_line = re.sub(r"(?is)nrun.+", 
                   "nrun =  %d\n" % n_trajectories, 
                   new_line)
            
        if seed is not None:
            new_line = re.sub(r"(?is)dseed.+", 
                   "dseed = %d\n" % seed, new_line)
            
        new_line = re.sub(r"(?is)ftrajectories.+", 
            "ftrajectories = %s\n" % trajectory_file, 
            new_line)
        
        new_line = re.sub(r"(?is)fcomplexes.+", 
            "fcomplexes = %s\n" % complexes_file, 
            new_line)
            
        sim_file_new_lines.append(new_line)
    
    if os.path.exists(simulation_filename):
        os.rename(simulation_filename, 
                  os.path.join(bd_milestone_directory, 
                               f"sda_{restart_index-1}.in"))
        output_full_path = os.path.join(bd_milestone_directory, 
                              f"sda.out")
        if os.path.exists(output_full_path):
            os.rename(output_full_path,
                      os.path.join(bd_milestone_directory, 
                                   f"sda_{restart_index-1}.out"))

    with open(simulation_filename, 'w') as f:
        for line in sim_file_new_lines:
            f.write(line)
            
    return

def run_nam_simulation(sda_bin_dir, bd_directory, sda_output_glob, 
                       force_overwrite=False):
    """
    Run the SDA simulation.
    
    Parameters
    ----------
    sda_bin_dir : str
        The location of all the SDA binaries (programs). If the
        binaries have been added to the system PATH environmental
        variable, then this can be an empty string: "".
        
    bd_directory : str
        The path to the directory where SDA simulations will be run.
        
    sda_output_glob : str
        A glob to detect the presence of results files to ensure that
        simulations ran properly.
    """

    curdir = os.getcwd()
    print("moving to directory:", bd_directory)
    os.chdir(bd_directory)
    bd_command = os.path.join(sda_bin_dir, "sda_flex")
    simulation_filename = sda_base.SDA_INPUT_BASENAME
    command = bd_command + " " + simulation_filename + \
    " > sda.out"

    if force_overwrite:
        print("running command:", command)
        os.system(command)
    else:
        if not os.path.exists("sda.out"):
            print("running command:", command)
            os.system(command)
        else:
            pattern = "INFO All your SDA runs/trajectories finished successfully."
            ran = False
            with open('your_file.txt', 'r') as file:
                for line in file:
                    if re.search(pattern, line):  # Check if pattern is in the line
                        ran = True
                        break
            if not ran:
                print("running command:", command)
                os.system(command)
            else:
                print("SDA simulations already ran.")
                print("If you want to run them again, set force_overwrite.")

    results_glob = glob.glob(sda_output_glob)
    assert len(results_glob) > 0, "Problem occurred running "\
        "nam_simulation: results file was not generated."
    os.chdir(curdir)
    return

def run_simulation(
        model: structures.Seekr_model, 
        n_trajectories: int, 
        restart: bool, 
        force_overwrite: bool = False
    ) -> None:
    """
    Determine which bd engine to run and run it
    """
    assert model.bd_engine is not None
    assert model.bd_engine.type == "sda"
    bd_directory = os.path.join(model.directory, model.bd_engine.b_surface_directory)
    sda_output_glob = sda_base.SDA_RESULTS_GLOB
    bd_seed = model.bd_engine.random_seed
    modify_variables(
        bd_directory, sda_output_glob, n_trajectories, bd_seed, restart=restart)
    sda_bin_dir = os.path.expanduser(model.bd_engine.binary_directory)
    if not os.path.exists(sda_bin_dir):
        raise FileNotFoundError(f"SDA binary directory does not exist: {sda_bin_dir}")
    # Run the SDA simulation
    run_nam_simulation(sda_bin_dir, bd_directory, sda_output_glob, 
                       force_overwrite=force_overwrite)
    return