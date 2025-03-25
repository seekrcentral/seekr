"""
Import input classes and functions from the different engines
for easy access here.
"""

import os
import typing
from shutil import copyfile

from attrs import define, field, validators, Factory

import seekr.modules.base as base
import seekr.modules.engines.base as engines_base
import seekr.modules.engines.openmm.input as openmm_input
import seekr.modules.engines.browndye.input as browndye_input

SYSTEMWIDE_DIRECTORY = "systemwide"

# Create unions so that the classes may be easily structure, 
#  unstructured, and validated
MD_settings_input = typing.Union[openmm_input.Openmm_settings_input]
BD_settings_input = typing.Union[browndye_input.Browndye_settings_input]

Positions_type = typing.Union[typing.List, str, None]
