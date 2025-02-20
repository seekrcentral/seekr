"""
Import base objects from all implemented engines.
"""

import typing

import seekr.modules.engines.openmm.base as openmm_base
import seekr.modules.engines.browndye.base as browndye_base

MD_engine = typing.Union[openmm_base.Openmm_md_engine]
BD_engine = typing.Union[browndye_base.Browndye_bd_engine]

from seekr.modules.engines.base import MD_systemwide_settings
BD_systemwide_settings = typing.Union[browndye_base.Browndye_systemwide_settings]