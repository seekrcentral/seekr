from __future__ import annotations
"""
modules/scales/base.py

Base structure for all scales and stages.
"""

import typing

import seekr.modules.structures as structures


class Base_scale(typing.Protocol):
    """
    Base class for all scales.
    """
    type: typing.Literal["base"] = "base"

class Base_stage(typing.Protocol):
    """
    Base class for all stages.
    """
    type: typing.Literal["base"] = "base"

class Base_input_stage(typing.Protocol):
    """
    Base class for all input stages.
    """
    type: typing.Literal["base"] = "base"

def construct_anchor_or_unpartitioned_index_list(
        model: structures.Seekr_model,
        stage: Base_stage,
        anchor_index: str
        ) -> list[int | None]:
    """
    Construct, for a given stage, either a list of one "None" for 
    unpartitioned scopes or a list of anchor indices for partitioned scopes.
    """
    anchor_or_unpartitioned_index_list: list[int | None] = []
    if stage.scope == "unpartitioned":
        anchor_or_unpartitioned_index_list.append(None)
    elif stage.scope == "partitioned":
        for alpha, anchor in enumerate(model.anchors):
            if "molecular_dynamics" not in anchor.scale_types:
                continue
            if anchor_index not in ["any"]:
                try:
                    anchor_index_int = int(anchor_index)
                except ValueError:
                    return []
                if alpha != anchor_index_int:
                    continue
            anchor_or_unpartitioned_index_list.append(alpha)
    return anchor_or_unpartitioned_index_list
