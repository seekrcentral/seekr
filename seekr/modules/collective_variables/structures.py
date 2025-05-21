"""
modules/collective_variables/structures.py

Structures for collective variables within seekr - anything that might
be found within the model.json file.

This module mainly defines types common to collective variables.
"""

import typing

import mdtraj

import seekr.modules.base as base
import seekr.modules.collective_variables.distance_cv as distance_cv
import seekr.modules.collective_variables.external_cv as external_cv
import seekr.modules.collective_variables.rmsd_cv as rmsd_cv

if typing.TYPE_CHECKING:
    import seekr.modules.structures as structures

Collective_variable = typing.Union[distance_cv.Distance_CV,
                                   external_cv.External_CV,
                                   rmsd_cv.RMSD_CV]

def get_mdtraj_cv_value(
          model: "structures.Seekr_model",
          traj: mdtraj.Trajectory,
          frame_index: int = 0
          ) -> list[float]:
        """
        Determine the current CV value for an mdtraj object.
        """
        values = []
        for i, cv in enumerate(model.collective_variables):
            cv_value = cv.get_mdtraj_cv_value(model, traj, frame_index)
            values.append(cv_value)
        
        return values

def check_value_within_boundary(
        values: float,
        milestone_variables: dict[str, float],
        tolerance: float = 0.0
        ) -> bool:
    """
    Check if a value is within the boundary defined by the
    milestone variables.
    """
    me_total_value = 0.0
    neighbor_total_value = 0.0
    cv_indices = []
    for key in milestone_variables.keys():
        if key.startswith("me_"):
            cv_indices.append(int(key.split("_")[1]))
        
    for i, cv_index in enumerate(cv_indices):
        cv_value = values[i]
        me_i_value = milestone_variables[f"me_{cv_index}"]
        neighbor_i_value = milestone_variables[f"neighbor_{cv_index}"]
        me_this_value = (me_i_value - cv_value)**2
        neighbor_this_value = (neighbor_i_value - cv_value)**2
        me_total_value += me_this_value
        neighbor_total_value += neighbor_this_value
            
    distance = me_total_value - neighbor_total_value
    if (distance-tolerance) > 0.0: 
        return False
    
    return True

def check_mdtraj_within_boundary(
        model: "structures.Seekr_model", 
        traj: mdtraj.Trajectory, 
        milestone_variables: dict[str, float],
        frame_index: int = 0,
        tolerance: float = 0.0,
        ) -> bool:
    """
    Check if an mdtraj Trajectory describes a system that remains
    within the expected anchor. Return True if passed, return
    False if failed.
    """
    #for frame_index in range(traj.n_frames):
    values = get_mdtraj_cv_value(model, traj, frame_index)
    result = check_value_within_boundary(
        values, milestone_variables, tolerance=tolerance)
    if not result:
        return False
        
    return True