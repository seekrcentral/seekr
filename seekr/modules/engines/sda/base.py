"""
modules/engines/sda/base.py

Base functions, types, and constants for the SDA engine.

Imports only allowed from modules/base.py, modules/engines/base.py 
and external libraries.
Imported by anything in the modules/engines/sda subdirectory.
"""

from collections import defaultdict

SDA_TRAJ_NAME = "traj"
SDA_COMPLEXES_NAME = "complexes"
SDA_INPUT_BASENAME = "sda.in"
APBS_INPUT_BASENAME = "apbs_input.xml"
REACTION_BASENAME = "p2.rxna"
SDA_RECEPTOR = "p1"
SDA_LIGAND = "p2"
SDA_RESULTS_GLOB = "sda*.out"

def compute_transition_counts(results_filename_list):
    """
    Compute the transition counts from the BD results files.
    """
    transition_counts = defaultdict(int)
    transition_counts_all = defaultdict(int)
    total_n_trajectories = 0
    # first, obtain counts directly from the results file
    for results_filename in results_filename_list:
        with open(results_filename) as f:
            output_lines = f.readlines()
            read_kon_table = False

            for line in output_lines:
                if "Nrun" in line:
                    nruns = int(line.split()[1])
                    total_n_trajectories += nruns
                elif line.startswith("b_rate"):
                    kb = float(line.split()[1])
                elif line.startswith("c_rate"):
                    kc = float(line.split()[1])
                elif "rxn       M^-1s^-1      beta      time" in line:
                    read_kon_table = True
                elif read_kon_table and line[0] != "#" and \
                    len(line.split()) > 0:
                    win_dist, beta = \
                        float(line.split()[0]), float(line.split()[2])
                    transition_counts_all[win_dist] += int(beta*nruns)            
                elif read_kon_table and len(line.split()) == 0:
                    break
    

    for win_dist_angstroms, count in transition_counts_all.items():
        win_dist_nm = win_dist_angstroms / 10.0
        transition_counts[win_dist_nm] += count
        
    #transition_counts["escaped"] = \
    #    total_n_trajectories - transition_counts[inner_milestone_index]
    transition_counts["total"] = total_n_trajectories
    return transition_counts