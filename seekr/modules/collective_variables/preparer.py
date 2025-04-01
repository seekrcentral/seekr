"""
collective_variables/input.py

Collective variable, state, and anchor objects for the prepare
stage of SEEKR.
"""

import typing

import seekr.modules.collective_variables.distance_cv as distance_cv

CV_input = typing.Union[distance_cv.Distance_cv_input]
Collective_variable = typing.Union[distance_cv.Distance_CV]