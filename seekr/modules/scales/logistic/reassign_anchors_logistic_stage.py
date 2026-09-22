from __future__ import annotations
"""
This module defines a stage (along with its structures, preparer, and 
runner details) that reassigns the anchors of a model according to a 
predefined anchor scheme.

This can be effectively accomplished using an anchor scheme from the 
prepare stage, but this stage is mainly to function as a test for 
logistic stages.
"""

import os
import glob
import typing
import logging
from shutil import rmtree
from collections import defaultdict

import cattrs
from attrs import define, field, validators, Factory

import seekr.modules.base as base
import seekr.modules.anchor_schemes.base as anchor_schemes_base
import seekr.modules.scales.logistic.base as logistic_scale_base

logger = logging.getLogger(__name__)

# TODO: need to implement a way to modify a set of anchors, like with "main"

@define
class Stage_reassign_anchors(logistic_scale_base.Stage_logistic):
    """
    This stage reassigns the anchors of the model according to a predefined
    anchor scheme.
    """
    type: typing.Literal["reassign_anchors_stage"] = "reassign_anchors_stage"
    scale_type: typing.Literal["logistic"] = "logistic"
    index: int = field(default=0,
                        validator=[validators.instance_of(int),
                                      validators.ge(0)])
    name: str = field(default="reassign_anchors", 
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
    input_set_name: typing.Optional[str] = field(
        default="main", 
        validator=validators.optional(validators.instance_of(str)))
    output_set_name: typing.Optional[str] = field(
        default=None, 
        validator=validators.optional(validators.instance_of(str)))
    new_anchor_scheme_dict: dict | None \
        = field(default=None)
    new_state_points: list[base.State_point] = field(default=Factory(list),)
    scope: str = field(default="partitioned",
                       validator=validators.in_(
                           ["unpartitioned", "partitioned"]))
    
    def get_directory_name(self) -> str:
        """
        Get the directory of this stage.
        """
        directory_name: str = f"stage_{self.index}_{self.name}"
        return directory_name

    def add_stage_parameters_to_anchors(
            self, 
            model: "Seekr_model",
            ) -> None:
        """
        Add the stage directory to all anchors in the model.
        """
        
        for anchor in model.anchors:
            directory_name: str = self.get_directory_name()
            stage_parameter: dict[str, int | str] = {}
            stage_parameter["index"] = self.index
            stage_parameter["directory"] = directory_name
            stage_parameter["positions"] \
                = anchor.stage_parameters["initial"]["positions"] \
                    if "initial" in anchor.stage_parameters else None
            anchor.stage_parameters[self.name] = stage_parameter
            if "initial" in anchor.stage_parameters:
                del anchor.stage_parameters["initial"]
        return
    
    def has_files(
            self, 
            model: "Seekr_model",
            anchor_index: int | None = None,
            ) -> bool:
        """
        Returns true if any of the stage directories contain
        outputs from any of the stages.
        """
        stage_output_globs: list[str] = self.output_positions_glob_list
        my_directory = self.get_directory_name()
        anchor_set_directory = os.path.join(
            model.directory, "anchor_sets", my_directory)
        for anchor in model.anchors:
            if anchor_index is not None and anchor.index != anchor_index:
                continue
            for stage_parameter in anchor.stage_parameters.values():
                stage_directory = stage_parameter["directory"]
                if stage_directory == my_directory:
                    for stage_output_glob in stage_output_globs:
                        output_files_glob = os.path.join(
                            anchor_set_directory, anchor.directory, 
                            stage_directory, stage_output_glob)
                        if len(glob.glob(output_files_glob)) > 0:
                            return True
        return False
    
    def remove_directory(
            self, 
            model: "Seekr_model"
            ) -> None:
        """
        Remove entire directory in all anchors and unpartitioned settings for this stage.
        """
        my_directory = self.get_directory_name()
        for anchor in model.anchors:
            full_directory = os.path.join(
                model.directory, "anchor_sets", self.get_output_set_name(), 
                anchor.directory, my_directory)
            if os.path.exists(full_directory):
                rmtree(full_directory)
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine.unpartitioned_settings is not None
        full_directory = os.path.join(
            model.directory, md_engine.unpartitioned_settings.directory, my_directory)
        if os.path.exists(full_directory):
            rmtree(full_directory)
        return
    
    def is_logistic_stage(self) -> bool:
        """
        Returns true since this is a logistic stage.
        """
        return True
    
    def load_sets_into_model(
            self, 
            model: "Seekr_model"
            ) -> None:
        """
        Load only the CVs and selections into the model, since the anchors
        will be computed anew.
        """
        model.load_cvs(self.input_set_name)
        model.load_selections(self.input_set_name)
        return
    
    def get_output_set_name(self) -> str | None:
        """
        Get the output set name for this stage.
        """
        return self.output_set_name
    
    def write_to_log(
            self,
            model: "Seekr_model",
            instructions: list[logistic_scale_base.Run_instruction_logistic]
        ) -> None:
        """
        Write the given instructions to every anchor's log file within this 
        stage's directory.
        """
        for instruction in instructions:
            log_filename = model.get_synopsis_filename()
            base.configure_logging(log_filename)
            logger.info(instruction.message.strip())
    
@define
class Input_stage_reassign_anchors(logistic_scale_base.Input_stage_logistic):
    """
    This stage reassigns the anchors of the model according to a predefined
    anchor scheme Partition.
    """
    type: typing.Literal["reassign_anchors_stage"] = "reassign_anchors_stage"
    name: str = field(default="reassign_anchors", 
                      validator=validators.instance_of(str))
    comment: str = field(default="Reassign anchors according to a Partition.",
                         validator=validators.instance_of(str))
    input_stage_name: str | None = field(default=None,
                                 validator=validators.optional(
                                     validators.instance_of(str)))
    input_set_name: typing.Optional[str] = field(
        default="main", 
        validator=validators.optional(validators.instance_of(str)))
    output_set_name: typing.Optional[str] = field(
        default="main", 
        validator=validators.optional(validators.instance_of(str)))
    new_anchor_scheme: anchor_schemes_base.Base_anchor_scheme | None = field(
        default=None)
    new_state_points: list[base.State_point] = field(default=Factory(list),)
    
    def get_scale_type(self) -> str:
        return "logistic"

    def make_model_stage(
            self, 
            index: int,
            dummy_scope: str,
            dummy_settings: typing.Any = None,
            dummy_running_dir: str = "",
            ) -> typing.Tuple[Stage_reassign_anchors, str]:
        """
        Given this input stage object, create the model's Stage object.
        """
        stage = Stage_reassign_anchors()
        stage.index = index
        stage.name = self.name
        stage.comment = self.comment
        stage.input_set_name = self.input_set_name
        stage.output_set_name = self.output_set_name
        if self.new_anchor_scheme is not None:
            stage.new_anchor_scheme_dict = cattrs.unstructure(self.new_anchor_scheme)
        else:
            stage.new_anchor_scheme_dict = None
        stage.new_state_points = self.new_state_points
        new_scope = "partitioned"
        return stage, new_scope

def is_finished_reassign_anchors_logistic_with_message(
        model: "Seekr_model",
        stage: Stage_reassign_anchors,
        use_json: bool = True
    ) -> tuple[bool, str | dict]:
    """
    A reassign anchors logistic stage is finished if the new anchor set has 
    been created.
    """
    model.load_anchor_set(stage.output_set_name)
    if use_json:
        final_message: dict[str, dict] = {
            "finished": False,
            "num_swarms": 1,
            "swarms": {
                "0": {
                    "finished": False,
                    "progress": 0.0,
                }
            }
        }
        if len(model.anchors) == 0:
            return False, final_message
        final_message["finished"] = True
        final_message["swarms"]["0"]["finished"] = True
        final_message["swarms"]["0"]["progress"] = 1.0
        return True, final_message
    else:
        if len(model.anchors) == 0:
            return False, "New anchor set has not been created."
        return True, "New anchor set has been created."


def choose_next_simulation_reassign_anchors_logistic(
        model: "Seekr_model", 
        stage: Stage_reassign_anchors,
        force_overwrite: bool,
        use_json: bool = True,
        silent: bool = False
        ) -> list:
    """
    Choose the next simulation to run for a reassign anchors logistic stage.
    """
    # Check to see if the new anchor set has been created. If not, add the necessary
    # instruction.
    run_instructions: list[typing.NamedTuple] = []
    completed, message = is_finished_reassign_anchors_logistic_with_message(
        model, stage, use_json)
    if not completed or force_overwrite:
        run_instruction = logistic_scale_base.Run_instruction_logistic(
            anchor_index=None,
            swarm_id=0,
            message=message
        )
        run_instructions.append(run_instruction)
    return run_instructions
    

def do_run_instruction_reassign_anchors_logistic(
        model: "Seekr_model",
        force_overwrite: bool,
        stage_index: int,
        instruction: list,
        anchor_scheme: anchor_schemes_base.Base_anchor_scheme
        ) -> None:
    """
    Perform the reassignment of anchors according to the new anchor scheme.
    """
    import seekr.modules.preparer as preparer
    stage: Stage_reassign_anchors = model.stages[stage_index]
    set_name = stage.output_set_name
    assert anchor_scheme is not None, \
        "New anchor scheme must be provided."
    starting_anchor_index = 0
    starting_milestone_index = 0
    connection_flag_dict: dict[int|str, list["Anchor"]] \
        = defaultdict(list)
    associated_input_anchor: dict[int, typing.Any] = {}
    anchors, anchor_connections, milestone_index\
        = anchor_scheme.make_anchors(
            model.collective_variables, starting_anchor_index, 
            starting_milestone_index, connection_flag_dict, 
            associated_input_anchor)
    
    model.anchors = anchors
    model.num_anchors = len(anchors)
    preparer.assign_all_state_points(stage.new_state_points, model)
    dummy_anchors, index_reference = preparer.resolve_connections(
        connection_flag_dict, model, associated_input_anchor, 
        anchor_connections)

    anchor_sets_directory = os.path.join(model.directory, "anchor_sets")
    if not os.path.exists(anchor_sets_directory):
        os.makedirs(anchor_sets_directory)
    this_anchor_set_directory = os.path.join(
        anchor_sets_directory, set_name)
    if not os.path.exists(this_anchor_set_directory):
        os.makedirs(this_anchor_set_directory)
    for anchor in anchors:
        this_anchor_directory = os.path.join(
            this_anchor_set_directory, anchor.directory)
        if not os.path.exists(this_anchor_directory):
            os.makedirs(this_anchor_directory)

    for iter_stage in model.stages:
        if iter_stage.scale_type in ["molecular_dynamics", "logistic"]:
            if iter_stage.scope == "partitioned":
                # NOTE: mypy error is handled with "md_stage" type check above
                iter_stage.add_stage_parameters_to_anchors(model)
                
            elif iter_stage.scope == "unpartitioned":
                iter_stage.add_stage_parameters_to_unpartitioned_settings(model)

            else:
                raise ValueError(f"Scope {stage.scope} not recognized.")
                
        else:
            pass

    for old_index in index_reference:
        new_index = index_reference[old_index]
        if new_index < len(anchors):
            input_anchor = associated_input_anchor[old_index]
            anchors[new_index].generate_filetree(
                model, model.directory, set_name)
            input_anchor.copy_initial_files(
                anchors[new_index], model.directory, set_name, 
                model.using_toy(), stage_name=stage.name)

    
    for anchor in anchors:
        anchor.save(this_anchor_set_directory)
    
    # NOTE: need to save a new CV and selection set, even tho they are
    # unchanged from main, since the stage output set_name will be 
    # loaded by later stages.
    model.clear_cv_set(model.directory, set_name)
    model.save_cvs(model.directory, set_name)
    model.save_selections(model.directory, set_name)
    return
