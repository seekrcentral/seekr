"""
This logistic stage will take the outputs to a reporter and 
will assign them as the starting structures for anchors within a
future stage.

This involves changing the scope of a future stage to be a 
'partitioned' scope from a 'unpartitioned' scope.
"""

import typing

from attrs import define, field, validators, Factory

import seekr.modules.scales.logistic.base as logistic_scale_base

@define
class Stage_anchor_starting_structures_from_reporter(
        logistic_scale_base.Stage_logistic):
    """
    This stage takes the output of a reporter within an input stage and
    assigns starting structures to the anchors of a later stage, changing
    the scope of the scale to 'partitioned'.

    If there's more than one reporter in the stage, assume the first one.
    """
    type: typing.Literal["anchor_starting_structures_from_reporter_stage"] \
        = "anchor_starting_structures_from_reporter_stage"
    scale_type: typing.Literal["logistic"] = "logistic"
    index: int = field(default=0,
                        validator=[validators.instance_of(int),
                                      validators.ge(0)])
    name: str = field(default="anchor_starting_structures_from_reporter", 
                      validator=validators.instance_of(str))
    comment: str = field(default="This is a logistic stage.",
                         validator=validators.instance_of(str))
    output_positions_basename_noext: str = field(
        default="output_positions", 
        validator=validators.instance_of(str))
    output_positions_glob_list: list[str] = field(
        default=Factory(lambda: ["*.pdb", "*.txt"]), 
        validator=validators.instance_of(list))
    input_stage_name: str | None = field(default=None,
                                 validator=validators.optional(
                                     validators.instance_of(str)))
    output_stage_name: str | None = field(default=None,
                                 validator=validators.optional(
                                     validators.instance_of(str)))
    
@define
class Input_stage_anchor_starting_structures_from_reporter(logistic_scale_base.Input_stage_logistic):
    """
    This stage assigns the starting structures of the anchors of the model
    according to the output of a reporter stage.
    """
    type: typing.Literal["anchor_starting_structures_from_reporter_stage"] \
        = "anchor_starting_structures_from_reporter_stage"
    name: str = field(default="anchor_starting_structures_from_reporter", 
                      validator=validators.instance_of(str))
    comment: str = field(default="Assign anchor starting structures from a reporter.",
                         validator=validators.instance_of(str))
    # If these two are None, then assume previous/next stage.
    input_stage_name: str | None = field(default=None,
                                 validator=validators.optional(
                                     validators.instance_of(str)))
    output_stage_name: str | None = field(default=None,
                                 validator=validators.optional(
                                     validators.instance_of(str)))
    
    def get_scale_type(self) -> str:
        return "logistic"

    def make_model_stage(
            self, 
            index: int,
            dummy_scope: str,
            dummy_settings: typing.Any = None
            ) -> typing.Tuple[Stage_anchor_starting_structures_from_reporter, str]:
        """
        Given this input stage object, create the model's Stage object.
        """
        stage = Stage_anchor_starting_structures_from_reporter()
        stage.index = index
        stage.name = self.name
        stage.comment = self.comment
        stage.input_stage_name = self.input_stage_name
        stage.output_stage_name = self.output_stage_name
        new_scope = "partitioned"
        return stage, new_scope

def choose_next_simulation_anchor_starts_from_reporter_logistic(
        model: "Seekr_model", 
        stage: Stage_anchor_starting_structures_from_reporter,
        force_overwrite: bool,
        use_json: bool = True,
        silent: bool = False
        ) -> list:
    """
    Choose the next simulation to run for a reassign anchors logistic stage.
    """
    # Check to see if the new anchor set has been created. If not, add the necessary
    # instruction.
    model.load_anchor_set(stage.output_set_name)
    if (len(model.anchors) == 0) or force_overwrite:
        # TODO: does this ever need to be different?
        #if use_json:
        instruction = "Assign anchor starting structures from reporter output."
        return [instruction]
    return []

def do_run_instruction_anchor_starts_from_reporter_logistic(
        model: "Seekr_model",
        force_overwrite: bool,
        stage_index: int,
        instruction: list
        ) -> None:
    """
    Perform the reassignment of anchors according to the new anchor scheme.
    """
    stage: Stage_anchor_starting_structures_from_reporter = model.stages[stage_index]
    # TODO: extract starting structures from reporter and assign to anchor set.
    return
