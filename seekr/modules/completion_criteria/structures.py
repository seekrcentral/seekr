"""
modules/completion_criteria/structures.py

Structures for any completion criteria that can be used in SEEKR. 
That is - any imaginable way to determine when a simulation is complete.
"""

import typing

from attrs import define, field, validators

MAX_ANCHORS_FILLED_STEPS = 2000000 # ~4 ns
MAX_BD_STEPS = 1e12 # a large number - essential infinite

class Base_completion_criteria(typing.Protocol):
    """
    Base class for all completion criteria.
    """
    type: typing.Literal["base"] = "base"

    def get_min_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_max_sim_len(self) -> int:
        """
        Return the maximum simulation length for this completion criteria.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_evaluation_interval(self) -> int:
        """
        Return the evaluation interval for this completion criteria.
        """
        raise NotImplementedError("Subclasses must implement this method.")

@define
class Number_of_steps_completion_criteria(Base_completion_criteria):
    """
    A completion criteria based on the number of steps.
    """
    type: typing.Literal["number_of_steps"] = "number_of_steps"
    number_of_steps: int = field(
        default=100000, validator=[validators.instance_of(int), 
                                   validators.gt(0)])
    
    # NOTE: these functions are needed for chunking in runner modules
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

@define
class Anchors_filled_completion_criteria(Base_completion_criteria):
    """
    A completion criteria based on all anchors being filled.
    """
    type: typing.Literal["anchors_filled"] = "anchors_filled"
    exclude: list[int] = field(
        factory=list, validator=validators.deep_iterable(
            member_validator=validators.instance_of(int),
            iterable_validator=validators.instance_of(list),
        ))
    eval_interval: int = field(
        default=250, validator=[validators.instance_of(int),
                                validators.gt(0)])

    # NOTE: these functions are needed for chunking in runner modules
    def get_min_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        return 0
    
    def get_max_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        return MAX_ANCHORS_FILLED_STEPS #base.MAX_SEEKR_SIMULATION_STEPS
    
    def get_evaluation_interval(self) -> int:
        return self.eval_interval
    
#  Sep 26, 2025: I don't foresee needing this anytime soon, but
#   I want to keep these ideas here for possible future implementation.
#    - lvotapka
# TODO: Maybe implement someday?
#@define
#class CV_value_attained_completion_criteria:

# TODO: Maybe implement someday?
#@define
#class Surface_crossed_completion_criteria:

@define
class Number_of_trajectories_completion_criteria(Base_completion_criteria):
    """
    A completion criteria based on the number of trajectories in a 
    BD simulation.
    """
    type: typing.Literal["number_of_trajectories"] = "number_of_trajectories"
    number_of_trajectories: int = field(
        default=10000, validator=[validators.instance_of(int),
                                  validators.gt(0)])
    max_number_of_steps: int = field(
        default=100000000, validator=[validators.instance_of(int),
                                      validators.gt(0)])
    
    # NOTE: these functions are needed for chunking in runner modules
    def get_min_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        return 0
    
    def get_max_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        return MAX_BD_STEPS
    
    def get_evaluation_interval(self) -> int:
        return 0
    
@define
class Number_of_encounters_completion_criteria(Base_completion_criteria):
    """
    A completion criteria based on the number of encounters in a 
    BD simulation.
    """
    type: typing.Literal["number_of_encounters"] = "number_of_encounters"
    number_of_encounters: int = field(
        default=1000, validator=[validators.instance_of(int),
                                 validators.gt(0)])
    evaluation_interval: int = field(
        default=1000, validator=[validators.instance_of(int),
                                 validators.gt(0)])
    max_number_of_trajectories: int = field(
        default=10000000, validator=[validators.instance_of(int),
                                      validators.gt(0)])
    max_number_of_steps: int = field(
        default=100000000, validator=[validators.instance_of(int),
                                       validators.gt(0)])
    
    # NOTE: these functions are needed for chunking in runner modules
    def get_min_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        return 0
    
    def get_max_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        return MAX_BD_STEPS
    
    def get_evaluation_interval(self) -> int:
        return self.evaluation_interval
