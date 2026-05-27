from __future__ import annotations
"""
modules/completion_criteria/structures.py

Structures for any completion criteria that can be used in SEEKR. 
That is - any imaginable way to determine when a simulation is complete.
"""

import typing

from attrs import define, field, validators

MAX_BD_STEPS = 1e12 # a large number - essential infinite
BENCHMARK_MD_STEPS = 10000 # a small number for benchmarking purposes
BENCHMARK_BD_TRAJS = 100 # a small number for benchmarking purposes

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
    
    def enable_benchmark_mode(self) -> None:
        """
        Enable benchmark mode for this completion criteria, which may adjust
        the behavior of the completion criteria to be more suitable for 
        benchmarking (e.g. by reducing the number of steps required).
        """
        pass

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
        Return the maximum simulation length for this completion criteria.
        """
        return self.number_of_steps
    
    def get_evaluation_interval(self) -> int:
        return 0
    
    def enable_benchmark_mode(self) -> None:
        """
        Enable benchmark mode for this completion criteria, which may adjust
        the behavior of the completion criteria to be more suitable for 
        benchmarking (e.g. by reducing the number of steps required).
        """
        self.number_of_steps = BENCHMARK_MD_STEPS

@define
class Reporter_progress_completion_criteria(Base_completion_criteria):
    """
    A completion criteria based on reporter progress.
    """
    type: typing.Literal["reporter_progress"] = "reporter_progress"
    reporter_name: str = field(
        default="anchor_reporter", validator=validators.instance_of(str))
    reporter_progress_threshold: float = field(
        default=1.0, validator=[validators.instance_of(float), 
                                 validators.gt(0), validators.le(1)])
    interval: int = field(
        default=250, validator=[validators.instance_of(int),
                                validators.gt(0)])
    max_number_of_steps: int = field(
        default=2000000, validator=[validators.instance_of(int),
                                       validators.gt(0)]) # ~4 ns
    # NOTE: these functions are needed for chunking in runner modules
    def get_min_sim_len(self) -> int:
        """
        Return the minimum simulation length for this completion criteria.
        """
        return 0
    
    def get_max_sim_len(self) -> int:
        """
        Return the maximum simulation length for this completion criteria.
        """
        return self.max_number_of_steps
    
    def get_evaluation_interval(self) -> int:
        return self.interval
    
    def enable_benchmark_mode(self) -> None:
        """
        Enable benchmark mode for this completion criteria, which may adjust
        the behavior of the completion criteria to be more suitable for 
        benchmarking (e.g. by reducing the number of steps required).
        """
        self.max_number_of_steps = BENCHMARK_MD_STEPS

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
        Return the maximum simulation length for this completion criteria.
        """
        return self.max_number_of_steps

    def get_evaluation_interval(self) -> int:
        return 0
    
    def enable_benchmark_mode(self) -> None:
        """
        Enable benchmark mode for this completion criteria, which may adjust
        the behavior of the completion criteria to be more suitable for 
        benchmarking (e.g. by reducing the number of steps required).
        """
        self.number_of_trajectories = BENCHMARK_BD_TRAJS
    
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
        Return the maximum simulation length for this completion criteria.
        """
        return MAX_BD_STEPS
    
    def get_evaluation_interval(self) -> int:
        return self.evaluation_interval
    
    def enable_benchmark_mode(self) -> None:
        """
        Enable benchmark mode for this completion criteria, which may adjust
        the behavior of the completion criteria to be more suitable for 
        benchmarking (e.g. by reducing the number of steps required).
        """
        self.max_number_of_trajectories = BENCHMARK_BD_TRAJS
        if self.evaluation_interval > BENCHMARK_BD_TRAJS:
            self.evaluation_interval = BENCHMARK_BD_TRAJS
