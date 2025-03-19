"""
run_base.py

Base objects and functions for running SEEKR calculations.
"""

import json

import cattrs
from attrs import define, field, validators, Factory

import seekr.modules.base as base

INFORMATION_CHECKPOINT_FILENAME_NOEXT = "checkpoint_information"
INFORMATION_CHECKPOINT_FILENAME_EXT = ".json"
RESTART_CHECKPOINT_FILENAME_NOEXT = "backup"
RESTART_CHECKPOINT_FILENAME_EXT = ".checkpoint"
STEP_TOLERANCE = 5
MAX_RESTART_ATTEMPTS = 3

@define
class Checkpoint_information:
    """
    A class to store information about the current state of a SEEKR
    calculation.
    """
    current_step: int = field(default=0, 
                              validator=[validators.instance_of(int),
                                               validators.ge(0)])
    
    def save(
            self, 
            filename: str
            ) -> None:
        """
        Write the model object to a file.
        """
        checkpoint_information_dict: dict = cattrs.unstructure(self)
        checkpoint_information_dict_noindent: dict \
            = base.wrap_noindent_integers(checkpoint_information_dict)
        json_dump: str = json.dumps(checkpoint_information_dict_noindent,
                                    indent=4, cls=base.NoIndentListEncoder)
        with open(filename, "w") as file:
            file.write(json_dump)

        return
    
def load_checkpoint_information(
        filename: str
        ) -> Checkpoint_information:
    """
    Load a model object from a file.
    """
    with open(filename, "r") as file:
        json_string: str = json.load(file)
    
    # Use a cattrs Converter to define how to structure the dictionary input
    converter: cattrs.Converter = cattrs.Converter()
    checkpoint_information: Checkpoint_information \
        = converter.structure(json_string, Checkpoint_information)
    return checkpoint_information

def get_checkpoint_information_filename(
        swarm_index: int | None
        ) -> str:
    if swarm_index is None:
        checkpoint_information_filename \
            = INFORMATION_CHECKPOINT_FILENAME_NOEXT \
                + INFORMATION_CHECKPOINT_FILENAME_EXT
    else:
        checkpoint_information_filename \
            = INFORMATION_CHECKPOINT_FILENAME_NOEXT \
            + f".swarm_{swarm_index}" \
            + INFORMATION_CHECKPOINT_FILENAME_EXT
    return checkpoint_information_filename

def get_restart_checkpoint_filename(
        swarm_index: int | None
        ) -> str:
    if swarm_index is None:
        restart_checkpoint_filename = RESTART_CHECKPOINT_FILENAME_NOEXT\
            + RESTART_CHECKPOINT_FILENAME_EXT
    else:
        restart_checkpoint_filename = RESTART_CHECKPOINT_FILENAME_NOEXT\
            + str(swarm_index) + RESTART_CHECKPOINT_FILENAME_EXT
    return restart_checkpoint_filename