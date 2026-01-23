"""
modules/scales/logistic/structures.py

Structures for the logistic scale - not affiliated with any
space or time scale, but only exists to add/remove/modify files
and settings in the middle of a seekr calculation.
"""

import os
import glob
import typing
from shutil import rmtree

from attrs import define, field, validators, Factory

@define
class Stage_logistic:
    """
    A logistic stage exists within SEEKR to allow files and settings
    to change between simulation stages. For instance, a metadynamics
    stage can be analyzed, and then the results can be used to define
    an initial anchor placement for a subsequent MMVT calculation.
    """
    type: typing.Literal["logistic_stage"] = "logistic_stage"
    scale_type: typing.Literal["logistic"] = "logistic"
    index: int = field(default=0,
                        validator=[validators.instance_of(int),
                                      validators.ge(0)])
    name: str = field(default="logistic", 
                      validator=validators.instance_of(str))
    comment: str = field(default="This is a logistic stage.",
                         validator=validators.instance_of(str))
    output_positions_basename_noext: str = field(
        default="output_positions", 
        validator=validators.instance_of(str))
    output_positions_glob_list: list[str] = field(
        default=Factory(lambda: ["*.pdb", "*.txt"]), 
        validator=validators.instance_of(list))
    input_stage_index: int = field(default=0,
                                      validator=[validators.instance_of(int),
                                                 validators.ge(0)])
    input_from: str = field(default="systemwide", 
                            validator=validators.instance_of(str))
    output_to: str = field(default="anchor", 
                           validator=validators.instance_of(str))
    input_set_name: typing.Optional[str] = field(
        default=None, 
        validator=validators.optional(validators.instance_of(str)))
    output_set_name: typing.Optional[str] = field(
        default=None, 
        validator=validators.optional(validators.instance_of(str)))
    
    def get_directory_name(self) -> str:
        """
        Get the directory of this stage.
        """
        directory_name: str = f"stage_{self.index}_{self.name}"
        return directory_name
    
    def has_files(
            self, 
            model: "Seekr_model"
            ) -> bool:
        """
        Returns true if any of the stage directories contain
        outputs from any of the stages.
        """
        stage_output_globs: list[str] = self.output_positions_glob_list
        my_directory = self.get_directory_name()
        for anchor in model.anchors:
            for stage_parameter in anchor.stage_parameters:
                if stage_parameter is None:
                    continue
                stage_directory = stage_parameter["directory"]
                if stage_directory == my_directory:
                    for stage_output_glob in stage_output_globs:
                        output_files_glob = os.path.join(
                            model.directory, anchor.directory, stage_directory, 
                            stage_output_glob)
                        if len(glob.glob(output_files_glob)) > 0:
                            return True
        return False
    
    def remove_directory(
            self, 
            model: "Seekr_model"
            ) -> None:
        """
        Remove entire directory in all anchors and systemwide settings for this stage.
        """
        my_directory = self.get_directory_name()
        for anchor in model.anchors:
            full_directory = os.path.join(model.directory, anchor.directory, my_directory)
            if os.path.exists(full_directory):
                rmtree(full_directory)
        return
    
    def is_logistic_stage(self) -> bool:
        """
        Returns true if this is a logistic stage.
        """
        return True
    