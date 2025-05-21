"""
collective_variables/input.py

Collective variable, state, and anchor objects for the prepare
stage of SEEKR.
"""

import typing

import seekr.modules.collective_variables.distance_cv as distance_cv
import seekr.modules.collective_variables.external_cv as external_cv
import seekr.modules.collective_variables.rmsd_cv as rmsd_cv

CV_input = typing.Union[distance_cv.Distance_cv_input,
                        external_cv.External_cv_input,
                        rmsd_cv.RMSD_cv_input]
Collective_variable = typing.Union[distance_cv.Distance_CV,
                                   external_cv.External_CV,
                                   rmsd_cv.RMSD_CV]