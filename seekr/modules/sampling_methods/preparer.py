"""
Inputs for different available sampling methods
"""

import typing

from attrs import define, field, validators

import seekr.modules.sampling_methods.structures as sampling_method_structures

# TODO: one day, this module will import sampling
#  methods added to this directory, but for now, they must be
#  defined here

@define
class Conventional_md_input:
    """
    A conventional MD sampling method input.
    """
    type: typing.Literal["conventional"] = "conventional"

    def make_model_sampling_method(self):
        """
        Make the model sampling method object for this input.
        """
        return sampling_method_structures.Conventional_md_sampling_method()

@define
class Metadynamics_md_input:
    """
    A conventional MD sampling method input.
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
    
    def make_model_sampling_method(self):
        """
        Make the model sampling method object for this input.
        """
        metaD = sampling_method_structures.Metadynamics_md_sampling_method()
        metaD.number_of_points = self.number_of_points
        metaD.bias_factor = self.bias_factor
        metaD.gaussian_height = self.gaussian_height
        metaD.gaussian_width = self.gaussian_width
        metaD.steps_per_update = self.steps_per_update
        return metaD

@define
class Milestoning_md_input:
    """
    A generic milestoning MD sampling method input.
    """
    type: typing.Literal["milestoning"] = "milestoning"
    milestoning_method: str = field(default="MMVT",
                                    validator=[validators.instance_of(str),
                                               validators.in_(["MMVT", "elber"])])
    
    def make_model_sampling_method(self):
        """
        Make the model sampling method object for this input.
        """
        milestoning = sampling_method_structures.Milestoning_md_sampling_method()
        milestoning.milestoning_method = self.milestoning_method
        return milestoning

@define
class NAM_bd_input:
    """
    A Northrup-Allison-McCammon (NAM) BD sampling method input.
    """
    type: typing.Literal["NAM"] = "NAM"

    def make_model_sampling_method(self):
        """
        Make the model sampling method object for this input.
        """
        return sampling_method_structures.NAM_bd_sampling_method()

@define
class WE_bd_input:
    """
    A weighted-ensemble (WE) BD sampling method input.
    """
    type: typing.Literal["WE"] = "WE"

    def make_model_sampling_method(self):
        """
        Make the model sampling method object for this input.
        """
        return sampling_method_structures.WE_bd_sampling_method()

MD_sampling_method_input = typing.Union[Conventional_md_input,
                                    Metadynamics_md_input,
                                    Milestoning_md_input]

BD_sampling_method_input = typing.Union[NAM_bd_input,
                                        WE_bd_input]