"""
modules/collective_variables/base.py

This module contains the base classes for collective variables and
any other common structures or functions related to CVs.
"""

import typing

class Base_CV(typing.Protocol):
    """
    Base class for all collective variables.
    """
    pass

class Base_CV_input(typing.Protocol):
    """
    Base class for all collective variable inputs.
    """
    pass