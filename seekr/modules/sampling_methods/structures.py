"""
modules/sampling_methods/structures.py

Structures for sampling methods within seekr - anything that might
appear within the model.json file.
"""

import typing

from attrs import define, field, validators, Factory

# TODO: one day, this module will import sampling
#  methods added to this directory, but for now, they must be
#  defined here

import seekr.modules.base as base

@define
class Conventional_md_sampling_method:
    """
    A conventional MD sampling method.
    """
    type: typing.Literal["conventional"] = "conventional"

    def input_from_systemwide_settings_output_to_anchors(self) -> bool:
        return False
    
    def input_from_anchors_output_to_systemwide_settings(self) -> bool:
        return False
    
    def needs_to_be_in_anchor(self) -> bool:
        return False
    
    def get_output_positions_basename_noext(self) -> str:
        return "conventional"

    def get_output_positions_glob_list(self) -> list[str]:
        return ["conventional*.pdb", "conventional*.txt"]
    
    def get_final_pdb_output_basename(
            self, 
            toy_mode: bool,
            frame_number: int | None = None,
            ) -> str:
        if toy_mode:
            extension = "txt"
        else:
            extension = "pdb"
        if frame_number is None:
            return f"conventional.{extension}"
        else:
            return f"conventional.swarm_{frame_number}.{extension}"
    
    def get_output_md_parameters_topology(self) -> None:
        return None

@define
class Metadynamics_md_sampling_method:
    """
    A Metadynamics MD sampling method.
    """
    type: typing.Literal["metadynamics"] = "metadynamics"
    cv_names: typing.List[str] = field(default=Factory(list),
                                            validator=validators.instance_of(list))
    number_of_points: base.Int_or_list_type = field(default=101)
    bias_factor: float = field(default=10.0,
                                validator=[validators.instance_of(float),
                                           validators.ge(0)])
    gaussian_height: float = field(default=0.5,
                                    validator=[validators.instance_of(float),
                                               validators.gt(0)])
    gaussian_width: base.Float_or_list_type = field(default=0.1)
    steps_per_update: int = field(default=250,
                                    validator=[validators.instance_of(int),
                                               validators.gt(0)])
    write_biases: bool = field(default=False,
                                validator=validators.instance_of(bool))
    
    def input_from_systemwide_settings_output_to_anchors(self) -> bool:
        return True
    
    def input_from_anchors_output_to_systemwide_settings(self) -> bool:
        return False
    
    def needs_to_be_in_anchor(self) -> bool:
        return False
    
    def get_output_positions_pdb_glob(self) -> str:
        return "metadynamics*.pdb"
    
    def get_output_positions_txt_glob(self) -> str:
        return "metadynamics*.txt"
    
    def get_output_basename(self) -> str:
        return "metadynamics"
    
    def get_output_extension(self) -> str:
        return "pdb"
    
    def get_output_glob(self, swarm_index) -> str:
        basename = self.get_output_basename()
        extension = self.get_output_extension()
        if swarm_index is None:
            swarm_string = ""
        else:
            swarm_string = ".swarm_{}".format(swarm_index)
        return f"{basename}{swarm_string}*.{extension}"
    
    def get_final_pdb_output_glob(self) -> str:
        basename = self.get_output_basename()
        return f"{basename}*.pdb"
    
    def get_final_pdb_output_basename(self) -> str:
        basename = self.get_output_basename()
        return f"{basename}.pdb"

    # TODO: update
    def get_output_globs(self) -> typing.List[str]:
        """
        Used for detecting any output files for an stage directory.
        """
        return ["metadynamics*.pdb"]
    
    def get_output_md_parameters_topology(self) -> None:
        return None
    
@define
class Milestoning_md_sampling_method:
    """
    A generic milestoning MD sampling method.
    """
    type: typing.Literal["milestoning"] = "milestoning"
    milestoning_method: str = field(default="MMVT",
                                    validator=[validators.instance_of(str),
                                               validators.in_(["MMVT", "elber"])])

    def input_from_systemwide_settings_output_to_anchors(self) -> bool:
        return False
    
    def input_from_anchors_output_to_systemwide_settings(self) -> bool:
        return False
    
    def needs_to_be_in_anchor(self) -> bool:
        return True
    
    def get_output_positions_pdb_glob(self) -> str:
        return "milestoning*.pdb"
    
    def get_output_positions_txt_glob(self) -> str:
        return "milestoning*.txt"
    
    def get_output_basename(self) -> str:
        return "milestoning"
    
    def get_output_extension(self) -> str:
        return "out"
    
    def get_output_glob(self, swarm_index) -> str:
        basename = self.get_output_basename()
        extension = self.get_output_extension()
        if swarm_index is None:
            swarm_string = ""
        else:
            swarm_string = ".swarm_{}".format(swarm_index)
        return f"{basename}{swarm_string}*.{extension}"
    
    def get_final_pdb_output_glob(self) -> str:
        basename = self.get_output_basename()
        return f"{basename}*.pdb"
    
    def get_final_pdb_output_basename(self) -> str:
        basename = self.get_output_basename()
        return f"{basename}.pdb"

    # TODO: update
    def get_output_globs(self) -> typing.List[str]:
        """
        Used for detecting any output files for an stage directory.
        """
        return ["milestoning.*"]
    
    def get_output_md_parameters_topology(self) -> None:
        return None

# TODO: insert QMrebind as a stage? that returns new md_parameters_topology
#  Alternatively, it can be run separately between stages, but modifies the
#  model JSON file.

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