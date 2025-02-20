"""
Import input classes and functions from the different engines
for easy access here.
"""

import typing

import seekr.modules.data_structures as data_structures
import seekr.modules.engines.openmm.input as openmm_input
import seekr.modules.engines.browndye.input as browndye_input

# Create unions so that the classes may be easily structure, 
#  unstructured, and validated
MD_settings_input = typing.Union[openmm_input.Openmm_settings_input]
BD_settings_input = typing.Union[browndye_input.Browndye_settings_input]