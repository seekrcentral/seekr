"""

"""

import os
import glob
import typing
from shutil import rmtree

import seekr.modules.scales.molecular_dynamics.structures as md_structures
import seekr.modules.scales.brownian_dynamics.structures as bd_structures
import seekr.modules.scales.logistic.base as logistic_base
import seekr.modules.scales.logistic.reassign_anchors_logistic_stage \
    as reassign_anchors_logistic_stage
import seekr.modules.scales.logistic.anchor_starting_structures_from_reporter_logistic_stage \
    as anchor_starting_structures_from_reporter_logistic_stage
import seekr.modules.scales.logistic.structures as logistic_struct


Scale = typing.Union[md_structures.Scale_md, bd_structures.Scale_bd]
Stage = typing.Union[md_structures.Stage_md, bd_structures.Stage_bd,
                     logistic_base.Stage_logistic,
                     reassign_anchors_logistic_stage.Stage_reassign_anchors,
                     anchor_starting_structures_from_reporter_logistic_stage\
                        .Stage_anchor_starting_structures_from_reporter,
                     ]

def register_structure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the structure hooks for all scales and their sampling methods.
    """
    md_structures.register_structure_hooks(converter)
    bd_structures.register_structure_hooks(converter)
    logistic_struct.register_structure_hooks(converter)
    return

def register_unstructure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the unstructure hooks for all scales and their sampling methods.
    """
    md_structures.register_unstructure_hooks(converter)
    bd_structures.register_unstructure_hooks(converter)
    logistic_struct.register_unstructure_hooks(converter)

def get_sampling_methods() -> typing.Dict[str, typing.Dict[str, typing.Any]]:
    return {
        "molecular_dynamics": md_structures.sampling_methods_md,
        "brownian_dynamics": bd_structures.sampling_methods_bd,
    }

def update_sampling_methods(
        new_sampling_methods: typing.Dict[str, typing.Dict[str, typing.Any]]
        ) -> None:
    """
    Update the sampling methods input dictionaries for all scales.
    """
    md_structures.sampling_methods_md.update(
        new_sampling_methods.get("molecular_dynamics", {}))
    bd_structures.sampling_methods_bd.update(
        new_sampling_methods.get("brownian_dynamics", {}))