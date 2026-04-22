"""
modules/sampling_methods/runner.py

Functions that may be called by a sampling method during the
run stage.
"""

import numpy as np

# TODO: one day, this module will import sampling
#  methods added to this directory, but for now, they must be
#  defined here

import seekr.modules.structures as structures

def find_anchor_value(
        model: structures.Seekr_model, 
        anchor_index: int
        ) -> list[float]:
    """
    
    """
    value = []
    anchor = model.anchors[anchor_index]
    for variable_key, variable_value in anchor.variables.items():
        if variable_key.startswith("value_"):
            value.append(variable_value)
    assert len(value) > 0, "No value variables found for anchor."
    return value

def find_edge_distance(
        model: structures.Seekr_model, 
        src_anchor_index: int, 
        dest_anchor_index: int
        ) -> float:
    """
    Find the distance of the variables commonly shared between two
    anchors. The distance is assumed to be Euclidean.
    
    Parameters:
    -----------
    model : Seekr_model
        The model object containing the relevant anchors.
    src_anchor_index : int
        The index of the anchor within the model that will be the
        'source' - where the steered simulation will start from.
    dest_anchor_index : int
        The index of the anchor within the model that will be the
        'destination' - where the steered simulation will go to.
        
    Returns:
    --------
    distance : float
        The distance (in nm) between the source and destination anchors.
    """
    # NOTE: How to handle the Jacobian of CV space? I guess just assume it's 
    # approximately constant over the space.
    src_anchor = model.anchors[src_anchor_index]
    dest_anchor = model.anchors[dest_anchor_index]
    variable_distance_sq = 0.0
    for variable in src_anchor.variables:
        if variable.startswith("value_"):
            if variable in dest_anchor.variables:
                distance = src_anchor.variables[variable] - dest_anchor.variables[variable]
                variable_distance_sq += distance**2
            
    return np.sqrt(variable_distance_sq)

def find_next_anchor_index(
        model: structures.Seekr_model, 
        visited_anchor_dict: dict[int, float],
        present_cv_values: list[float]
        ) -> tuple[int, int, float]:
    """
    Given a set of explored anchors, find a candidate for the next step
    in the pathfinding based on which will be the closest anchor (node)
    to reach.
    
    Parameters:
    -----------
    model : Seekr_model
        The model object containing the relevant anchors.
    visited_anchor_dict : dict
        A dictionary whose keys are anchor indices and whose values are
        a distance to a starting anchor.
        
    Returns:
    --------
    prev_anchor_index : int
        For the next step, starting at this anchor will be the step that
        will explore the closest available node (anchor).
    next_anchor_index : int
        For the next step, this anchor will be the step that explores 
        the closest available node (anchor).
    next_anchor_distance : float
        The total distance from one of the starting nodes (anchors) to
        the destination of this step.
    """
    FACTOR = 0.1
    prev_anchor_index = None
    next_anchor_index = None
    next_anchor_distance = 9e9
    if len(visited_anchor_dict) == 0:
        # If no anchors have been visited, find the closest one to the current cv_values
        for anchor_index, anchor in enumerate(model.anchors):
            if "molecular_dynamics" not in anchor.scale_types:
                continue
            anchor_value = find_anchor_value(model, anchor_index)
            assert len(anchor_value) == len(present_cv_values), \
                "Anchor and present CV values have different dimensions."
            total_distance_sq = 0.0
            for dim_val_anchor, dim_val_present in zip(anchor_value, present_cv_values):
                total_distance_sq += (dim_val_anchor - dim_val_present)**2
            total_distance = np.sqrt(total_distance_sq)
            if total_distance < next_anchor_distance:
                next_anchor_distance = total_distance
                next_anchor_index = anchor_index
                prev_anchor_index = anchor_index

    else:
        for visited_anchor_index in visited_anchor_dict:
            for milestone in model.anchors[visited_anchor_index].milestones:
                neighbor_index = milestone.neighbor_anchor_index
                if "molecular_dynamics" not in model.anchors[neighbor_index].scale_types:
                    continue
                if neighbor_index not in visited_anchor_dict:
                    edge_distance = find_edge_distance(model, visited_anchor_index, 
                                                neighbor_index)
                    total_distance = visited_anchor_dict[visited_anchor_index] \
                        * FACTOR + edge_distance
                    if total_distance < next_anchor_distance:
                        next_anchor_distance = total_distance
                        prev_anchor_index = visited_anchor_index
                        next_anchor_index = neighbor_index

    assert next_anchor_index is not None
    assert prev_anchor_index is not None
    return prev_anchor_index, next_anchor_index, next_anchor_distance

def get_smd_procedure(
            model: structures.Seekr_model,
            source_anchor_indices: list[int],
            destination_anchor_indices: list[int],
            present_cv_values: list[float]
            ) -> list:
        """
        The procedure is a list of anchor pairs that should be sampled
        in sequence.
        """
        procedure = []
        visited_anchor_dict = {}
        tmp_destination_list = destination_anchor_indices[:]
        for source_anchor_index in source_anchor_indices:
            visited_anchor_dict[source_anchor_index] = 0.0
        
        for i in range(len(model.anchors)):
            prev_anchor_index, next_anchor_index, next_anchor_distance \
                = find_next_anchor_index(model, visited_anchor_dict, present_cv_values)
            if next_anchor_index is None:
                # No more paths found
                break
            visited_anchor_dict[next_anchor_index] = next_anchor_distance
            if next_anchor_index in destination_anchor_indices:
                procedure.append((prev_anchor_index, next_anchor_index))
            if next_anchor_index in tmp_destination_list:
                tmp_destination_list.remove(next_anchor_index)
            if len(tmp_destination_list) == 0:
                break
            
        return procedure