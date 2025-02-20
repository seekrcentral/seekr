"""
base functions and classes for sampling methods within SEEKR.
"""

import typing

from attrs import define, field, validators

# TODO: one day, this module will import sampling
#  methods added to this directory, but for now, they must be
#  defined here

@define
class Conventional_md_sampling_method:
    """
    A conventional MD sampling method.
    """
    type: typing.Literal["conventional"] = "conventional"

    def input_from_systemwide_settings_output_to_anchors(self):
        return False
    
    def input_from_anchors_output_to_systemwide_settings(self):
        return False
    
    def needs_to_be_in_anchor(self):
        return False

@define
class Metadynamics_md_sampling_method:
    """
    A conventional MD sampling method.
    """
    type: typing.Literal["metadynamics"] = "metadynamics"
    number_of_points: int = field(default=101,
                                    validator=[validators.instance_of(int),
                                                 validators.gt(0)])
    bias_factor: float = field(default=10.0,
                                validator=[validators.instance_of(float),
                                           validators.ge(0)])
    gaussian_height: float = field(default=0.5,
                                    validator=[validators.instance_of(float),
                                               validators.gt(0)])
    gaussian_width: float = field(default=0.1,
                                    validator=[validators.instance_of(float),
                                               validators.gt(0)])
    steps_per_update: int = field(default=250,
                                    validator=[validators.instance_of(int),
                                               validators.gt(0)])
    
    def input_from_systemwide_settings_output_to_anchors(self):
        return True
    
    def input_from_anchors_output_to_systemwide_settings(self):
        return False
    
    def needs_to_be_in_anchor(self):
        return False


@define
class Milestoning_md_sampling_method:
    """
    A generic milestoning MD sampling method.
    """
    type: typing.Literal["milestoning"] = "milestoning"
    milestoning_method: str = field(default="MMVT",
                                    validator=[validators.instance_of(str),
                                               validators.in_(["MMVT", "elber"])])

    def input_from_systemwide_settings_output_to_anchors(self):
        return False
    
    def input_from_anchors_output_to_systemwide_settings(self):
        return False
    
    def needs_to_be_in_anchor(self):
        return True

@define
class NAM_bd_sampling_method:
    """
    Northrup-Allison-McCammon (NAM) BD sampling method.
    """
    type: typing.Literal["NAM"] = "NAM"

@define
class WE_bd_sampling_method:
    """
    Weighted-ensemble (WE) BD sampling method.
    """
    type: typing.Literal["WE"] = "WE"

MD_sampling_method = typing.Union[Conventional_md_sampling_method,
                                    Metadynamics_md_sampling_method,
                                    Milestoning_md_sampling_method]

BD_sampling_method = typing.Union[NAM_bd_sampling_method,
                                  WE_bd_sampling_method]