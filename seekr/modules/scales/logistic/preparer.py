"""
modules/scales/logistic/preparer.py

Logistic scale input objects and procedures.
"""

import typing

from attrs import define, field, validators, Factory

import seekr.modules.scales.logistic.structures as logistic_scale_struct

@define
class Input_stage_logistic:
    """
    A logistic stage allows one to modify position files, as well as
    CVs, anchors, selections, or other settings in the middle of a SEEKR
    calculation between simulation stages.
    """
    type: typing.Literal["logistic_stage"] = "logistic_stage"
    name: str = field(default="logistic", 
                      validator=validators.instance_of(str))
    comment: str = field(default="This is a logistic stage.",
                         validator=validators.instance_of(str))
    input_set_name: typing.Optional[str] = field(
        default=None, 
        validator=validators.optional(validators.instance_of(str)))
    output_set_name: typing.Optional[str] = field(
        default=None, 
        validator=validators.optional(validators.instance_of(str)))

    def get_scale_type(self) -> str:
        return "logistic"

    def make_model_stage(
            self, 
            index: int,
            systemwide_stage_input: bool,
            dummy_settings: typing.Any = None
           ) -> typing.Tuple[logistic_scale_struct.Stage_logistic, bool]:
        raise NotImplementedError("This method should be defined in a subclass.")
        