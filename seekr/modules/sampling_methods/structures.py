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

class Base_md_sampling_method(typing.Protocol):
    """
    Base class for all MD sampling methods.
    """
    type: typing.Literal["base"] = "base"

    def input_from_systemwide_settings_output_to_anchors(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def input_from_anchors_output_to_systemwide_settings(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def needs_to_be_in_anchor(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_output_positions_basename_noext(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_output_positions_glob_list(self) -> list[str]:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_final_pdb_output_basename(
            self, 
            toy_mode: bool,
            frame_number: int | None = None,
            ) -> str:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_output_md_parameters_topology(self) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

@define
class Conventional_md_sampling_method(Base_md_sampling_method):
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
class Metadynamics_md_sampling_method(Base_md_sampling_method):
    """
    A Metadynamics MD sampling method.
    """
    type: typing.Literal["metadynamics"] = "metadynamics"
    cv_names: typing.List[str] = field(default=Factory(list),
                                            validator=validators.instance_of(list))
    #number_of_points: base.Int_or_list_type = field(default=101)
    number_of_points: list[int] = field(default=Factory(list),)
    bias_factor: float = field(default=10.0,
                                validator=[validators.instance_of(float),
                                           validators.ge(0)])
    gaussian_height: float = field(default=0.5,
                                    validator=[validators.instance_of(float),
                                               validators.gt(0)])
    #gaussian_width: base.Float_or_list_type = field(default=0.1)
    gaussian_width: list[float] = field(default=Factory(list),)
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
    
    def get_output_positions_basename_noext(self) -> str:
        return "metadynamics"
    
    def get_output_positions_glob_list(self) -> list[str]:
        return ["metadynamics*.pdb", "metadynamics*.txt"]
    
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
            return f"metadynamics.{extension}"
        else:
            return f"metadynamics.swarm_{frame_number}.{extension}"
        
    def get_output_md_parameters_topology(self) -> None:
        return None

@define
class Steered_md_sampling_method(Base_md_sampling_method):
    """
    A Steered MD sampling method.
    """
    type: typing.Literal["steered"] = "steered"
    cv_names: typing.List[str] = field(default=Factory(list),
                                            validator=validators.instance_of(list))
    force_constant: float = field(default=10000.0,
                                    validator=[validators.instance_of(float),
                                               validators.ge(0)])
    velocity: float = field(default=1.0,
                            validator=[validators.instance_of(float),
                                       validators.ge(0)])
    
    def input_from_systemwide_settings_output_to_anchors(self) -> bool:
        return True
    
    def input_from_anchors_output_to_systemwide_settings(self) -> bool:
        return False
    
    def needs_to_be_in_anchor(self) -> bool:
        return False
    
    def get_output_positions_basename_noext(self) -> str:
        return "steered"
    
    def get_output_positions_glob_list(self) -> list[str]:
        return ["steered*.pdb", "steered*.txt"]
    
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
            return f"steered.{extension}"
        else:
            return f"steered.swarm_{frame_number}.{extension}"
        
    def get_output_md_parameters_topology(self) -> None:
        return None
    

@define
class MMVT_md_sampling_method(Base_md_sampling_method):
    """
    A generic milestoning MD sampling method.
    """
    type: typing.Literal["mmvt"] = "mmvt"

    def input_from_systemwide_settings_output_to_anchors(self) -> bool:
        return False
    
    def input_from_anchors_output_to_systemwide_settings(self) -> bool:
        return False
    
    def needs_to_be_in_anchor(self) -> bool:
        return True
    
    def get_output_positions_basename_noext(self) -> str:
        return "mmvt"
    
    def get_output_positions_glob_list(self) -> list[str]:
        return ["mmvt*.pdb", "mmvt*.txt"]
    
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
            return f"mmvt.{extension}"
        else:
            return f"mmvt.swarm_{frame_number}.{extension}"

    def get_output_md_parameters_topology(self) -> None:
        return None
    
    def write_output_file(
            self,
            filename: str,
            step_count: int,
            time_count: float,
            first_encounter_time: int,
            last_milestone_alias_touched: int,
            last_milestone_time_touched: int,
            milestone_count_dict: dict[str, int],
            milestone_time_dict: dict[str, int]
            ) -> None:
        block_str = \
              f"step_count: {step_count}\n" \
            + f"time_count: {time_count}\n" \
            + f"first_encounter_time: {first_encounter_time}\n" \
            + f"last_milestone_alias_touched: {last_milestone_alias_touched}\n" \
            + f"last_milestone_time_touched: {last_milestone_time_touched}\n"
        for key, value in milestone_count_dict.items():
            block_str += f"{key}: {value}\n"
        for key, value in milestone_time_dict.items():
            block_str += f"{key}: {value}\n"
        block_str += 20 * "#" + "\n"
        with open(filename, "a") as f:
            f.write(block_str)

# TODO: insert QMrebind as a stage? that returns new md_parameters_topology
#  Alternatively, it can be run separately between stages, but modifies the
#  model JSON file.

class Base_bd_sampling_method(typing.Protocol):
    """
    Base class for all BD sampling methods.
    """
    pass

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

#MD_sampling_method = typing.Union[Conventional_md_sampling_method,
#                                    Metadynamics_md_sampling_method,
#                                    Steered_md_sampling_method,
#                                    MMVT_md_sampling_method]

#BD_sampling_method = typing.Union[NAM_bd_sampling_method,
#                                  WE_bd_sampling_method]