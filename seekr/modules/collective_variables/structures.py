"""
modules/collective_variables/structures.py

Structures for collective variables within seekr - anything that might
be found within the model.json file.

This module mainly defines types common to collective variables.
"""

import typing

import seekr.modules.collective_variables.distance_cv as distance_cv
import seekr.modules.collective_variables.external_cv as external_cv

Collective_variable = typing.Union[distance_cv.Distance_CV,
                                   external_cv.External_CV]