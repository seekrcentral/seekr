"""
modules/scales/base.py

Base structure for all scales and stages.
"""

import typing

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