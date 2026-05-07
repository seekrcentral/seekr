"""
modules/scales/logistic/structures.py

Structures for the logistic scale - not affiliated with any
space or time scale, but only exists to add/remove/modify files
and settings in the middle of a seekr calculation.
"""

import typing

import cattrs

import seekr.modules.scales.logistic.base as logistic_base
import seekr.modules.scales.logistic.reassign_anchors_logistic_stage \
    as reassign_anchors_logistic_stage
import seekr.modules.scales.logistic\
        .anchor_starting_structures_from_reporter_logistic_stage \
    as anchor_starting_structures_from_reporter_logistic_stage

stage_types_logistic \
        = {"reassign_anchors_stage": 
                reassign_anchors_logistic_stage.Stage_reassign_anchors,
            "anchor_starting_structures_from_reporter_stage":
                anchor_starting_structures_from_reporter_logistic_stage\
                    .Stage_anchor_starting_structures_from_reporter
           }

def struct_md_sampling_method(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[logistic_base.Stage_logistic]
        ) -> logistic_base.Stage_logistic:
    typ = obj.get("type")
    return cattrs.structure(obj, stage_types_logistic[typ])

def unstruct_md_sampling_method(
        obj: logistic_base.Stage_logistic
        ) -> typing.Dict[str, typing.Any]:
    """
    Unstructure a completion criteria object to a dictionary.
    """
    d = cattrs.unstructure(obj)
    d["type"] = obj.type
    return d

def register_structure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the structure hooks for all scales and their sampling methods.
    """
    converter.register_structure_hook(logistic_base.Stage_logistic,
                                        struct_md_sampling_method)
    return
    
def register_unstructure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the unstructure hooks for all scales and their sampling methods.
    """
    converter.register_unstructure_hook(logistic_base.Stage_logistic,
                                        unstruct_md_sampling_method)
    return