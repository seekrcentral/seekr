"""
modules/completion_criteria/structures.py

Structures for any completion criteria that can be used in SEEKR. 
That is - any imaginable way to determine when a simulation is complete.
"""

import typing

from attrs import define, field, validators

import seekr.modules.base as base

@define
class Number_of_steps_completion_criteria:
    """
    A completion criteria based on the number of steps.
    """
    type: typing.Literal["number_of_steps"] = "number_of_steps"
    number_of_steps: int = field(default=100000,
                                validator=[validators.instance_of(int),
                                           validators.gt(0)])
    
    def get_min_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        return self.number_of_steps
    
    def get_max_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        return self.number_of_steps
    
    def get_evaluation_interval(self) -> int:
        return 0

# TODO: generalize this to a subset of anchors filled
@define
class All_anchors_filled_completion_criteria:
    """
    A completion criteria based on all anchors being filled.
    """
    type: typing.Literal["all_anchors_filled"] = "all_anchors_filled"
    eval_interval: int = field(default=250,
                                validator=[validators.instance_of(int),
                                           validators.gt(0)])

    def get_min_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        return 0
    
    def get_max_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        return base.MAX_SEEKR_SIMULATION_LENGTH
    
    def get_evaluation_interval(self) -> int:
        return self.eval_interval

# TODO: Maybe implement someday?
#@define
#class Surface_crossed_completion_criteria:
#    """
#    A completion criteria based on the number of surfaces crossed:
#    used by the NAM algorithm, for instance.
#    """
#    type: typing.Literal["surface_crossed"] = "surface_crossed"
#    n_surfaces_crossed: int = field(default=1,
#                                    validator=[validators.instance_of(int),
#                                               validators.gt(0)])
#    max_number_of_trajs: int = field(default=1000000,
#                                    validator=[validators.instance_of(int),
#                                               validators.gt(0)])
#    max_number_of_steps: int = field(default=100000000,
#                                    validator=[validators.instance_of(int),
#                                               validators.gt(0)])

@ define
class Number_of_trajectories_completion_criteria:
    """
    A completion criteria based on the number of trajectories in a 
    BD simulation.
    """
    type: typing.Literal["number_of_trajectories"] = "number_of_trajectories"
    number_of_trajectories: int = field(default=10000,
                                        validator=[validators.instance_of(int),
                                                   validators.gt(0)])
    max_number_of_steps: int = field(default=100000000,
                                    validator=[validators.instance_of(int),
                                               validators.gt(0)])

MD_completion_criteria = typing.Union[Number_of_steps_completion_criteria,
                                      All_anchors_filled_completion_criteria,
                                      None]

BD_completion_criteria = typing.Union[Number_of_trajectories_completion_criteria,
                                      None]
