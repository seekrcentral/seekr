from __future__ import annotations
"""
modules/engines/browndye/base.py

Base functions, types, and constants for the Browndye engine.

Imports from modules/base.py, modules/engines/base.py and external 
libraries only.
Imported by any Browndye-specific modules.
"""

import logging
import subprocess
from collections import defaultdict
import xml.etree.ElementTree as ET

import numpy as np

BROWNDYE_TRAJ_PREFIX = "traj"
BROWNDYE_INPUT_FILENAME = "input.xml"
BROWNDYE_APBS_INPUT_FILENAME = "apbs_input.xml"
BROWNDYE_RECEPTOR = "receptor"
BROWNDYE_LIGAND = "ligand"
REACTION_FILENAME = "rxns.xml"
DEFAULT_N_TRAJ_PER_OUT = 1000

def compute_transition_counts(results_filename_list):
    """
    Compute the transition counts from the BD results files.
    """
    transition_counts = defaultdict(int)
    # first, obtain counts directly from the results file
    for results_filename in results_filename_list:
        results_root = ET.parse(results_filename)
        reactions = results_root.find("reactions")
        n_trajectories = int(
            reactions.find("n_trajectories").text.strip())
        stuck = int(reactions.find("stuck").text.strip())
        escaped = int(reactions.find("escaped").text.strip())
        for completed in reactions.iter("completed"):
            name = completed.find("name").text.strip().split("_")
            dest = int(name[1])
            n = int(completed.find("n").text.strip())
            transition_counts[dest] += n
        transition_counts["escaped"] += escaped
        transition_counts["stuck"] += stuck
        transition_counts["total"] += n_trajectories
    return transition_counts

def run_compute_rate_constant(compute_rate_constant_program,
                              results_filename_list, 
                              sample_error_from_normal=False):
    """
    run the BrownDye program compute_rate_constant to find k_ons
    and the value k(b).
    
    Parameters:
    -----------
    compute_rate_constant_program : str
        The exact command to use to run the Browndye 
        compute_rate_constant program.
    
    results_filename_list : list
        A list of string paths to the results XML file, which is an 
        output from running the BrownDye program.
    
    sample_error_from_normal : bool, default False
        Add a fluctuation to the k-on value(s) sampled from a
        normal distribution with a standard deviation equal to the
        k-on's error. This is used by the error estimators to 
        incorporate the error in the k_b value obtained from BrownDye.
    
    Results:
    --------
    k_ons : defaultdict
        dictionary whose keys are various milestoning states and
        whose values are the k-ons to those states.
    
    k_on_errors : defaultdict
        dictionary whose keys are various milestoning states and
        whose values are the errors in the k-ons to those states.
    
    reaction_probabilities : defaultdict
        dictionary whose keys are various milestoning states and
        whose values are the probabilities of reaching those states
        from the b-surface.
    
    reaction_probability_errors : defaultdict
        dictionary whose keys are various milestoning states and
        whose values are the errors in the probabilities of 
        reaching those states from the b-surface.
        
    transition_counts : defaultdict
        dictionary whose keys are the various milestone states and 
        also the "escaped" and "stuck" states. The values are the 
        counts of crossings of these various states.
    """
    
    k_ons = defaultdict(float)
    k_on_errors = defaultdict(float)
    reaction_probabilities = defaultdict(float)
    reaction_probability_errors = defaultdict(float)
    transition_counts = defaultdict(int)
    k_b = 0.0
    transition_counts = compute_transition_counts(results_filename_list)
    logger = logging.getLogger(__name__)
    for results_filename in results_filename_list:
        cmd = "%s < %s" % (compute_rate_constant_program, 
                           results_filename)
        logger.info("running command: " + cmd)
        output_string = subprocess.check_output(cmd, shell=True)
        # TODO: validate output string somehow to catch errors
        # Sep 26, 2025: This is not an error so far, so I will wait
        #  until I see the problem to fix.
        root = ET.fromstring(output_string)
        # now obtain probabilities and rates
        for rate in root.iter("rate"):
            rate_constant = rate.find("rate_constant")
            k_on_tag = rate_constant.find("mean")
            k_on_tag_high = rate_constant.find("high")
            assert k_on_tag is not None
            assert k_on_tag_high is not None
            k_on = float(k_on_tag.text)
            k_on_error = 0.5*(float(k_on_tag_high.text) - k_on)
            reaction_probability = rate.find("reaction_probability")
            beta_tag = reaction_probability.find("mean")
            assert beta_tag is not None
            beta = float(beta_tag.text)
            if beta != 0.0:
                k_b = k_on / beta
            
    for key in transition_counts:
        reaction_probabilities[key] \
            = transition_counts[key] / transition_counts["total"]
        k_ons[key] = k_b * reaction_probabilities[key]
        if transition_counts[key] > 1:
            reaction_probability_errors[key] = reaction_probabilities[key] \
                / np.sqrt(transition_counts[key]-1)
            k_on_errors[key] = k_b * reaction_probability_errors[key]
        else:
            reaction_probability_errors[key] = 1e99
            k_on_errors[key] = 1e99
        
        if sample_error_from_normal:
            k_ons[key] = np.random.normal(loc=k_ons[key], 
                                          scale=k_on_errors[key])
    
    assert k_ons.keys() == transition_counts.keys()
    return k_ons, k_on_errors, reaction_probabilities, \
        reaction_probability_errors, transition_counts