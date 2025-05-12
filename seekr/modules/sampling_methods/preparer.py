"""
Inputs for different available sampling methods
"""

import typing

from attrs import define, field, validators, Factory

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
    A metadynamics MD sampling method input.
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
    cv_names: typing.List[str] = field(default=Factory(list),
                                        validator=validators.instance_of(list))
    
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
        metaD.cv_names = self.cv_names
        return metaD

@define
class Steered_md_input:
    """
    A steered MD sampling method input.
    """
    type: typing.Literal["steered"] = "steered"
    force_constant: float = field(default=10000.0,
                                    validator=[validators.instance_of(float),
                                               validators.ge(0)])
    velocity: float = field(default=1.0,
                            validator=[validators.instance_of(float),
                                       validators.ge(0)])
    cv_names: typing.List[str] = field(default=Factory(list),
                                        validator=validators.instance_of(list))
    
    def make_model_sampling_method(self):
        """
        Make the model sampling method object for this input.
        """
        smd = sampling_method_structures.Steered_md_sampling_method()
        smd.force_constant = self.force_constant
        smd.velocity = self.velocity
        smd.cv_names = self.cv_names
        return smd

@define
class MMVT_md_input:
    """
    A Markovian Milestoning with Voronoi Tesselation (MMVT) MD sampling method input.
    """
    type: typing.Literal["mmvt"] = "mmvt"
    
    def make_model_sampling_method(self):
        """
        Make the model sampling method object for this input.
        """
        milestoning = sampling_method_structures.MMVT_md_sampling_method()
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
                                    Steered_md_input,
                                    MMVT_md_input]

BD_sampling_method_input = typing.Union[NAM_bd_input,
                                        WE_bd_input]