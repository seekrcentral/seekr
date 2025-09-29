"""
modules/collective_variables/distance_cv.py

Model and input classes for the distance collective variable.
This CV measures the distance between the centers of mass of two 
groups of atoms.
"""

import typing

from attrs import define, field, validators
import numpy as np
import mdtraj

import seekr.modules.base as base
import seekr.modules.collective_variables.base as cv_base

if typing.TYPE_CHECKING:
    import seekr.modules.structures as structures

@define
class Distance_CV(cv_base.Base_CV):
    """
    A distance collective variable represents the distance between the
    centers of mass of two different groups of atoms.

    Collective variables represent the function of system positions
    and velocities such that the cells and boundaries can be defined.
    """
    index: int = field(validator=[validators.instance_of(int),
                                  validators.ge(0)])
    group1_name: str = field(validator=validators.instance_of(str))
    group2_name: str = field(validator=validators.instance_of(str))
    name: str = field(default="",
                        validator=validators.instance_of(str))
    unit: str = field(default="nm",
                      validator=validators.instance_of(str))
    num_groups: typing.Literal[2] = 2
    variable_name = "r"
    periodic: bool = False
    min_value: float = 0.0
    max_value: float = 3.0
    type: typing.Literal["distance"] = "distance"
    
    def is_periodic(self) -> bool:
        """
        Return whether this CV is periodic.
        """
        return self.periodic

    def get_min_value(self) -> float:
        """
        Return the minimum value of this CV.
        """
        return self.min_value
    
    def get_max_value(self) -> float:
        """
        Return the maximum value of this CV.
        """
        return self.max_value
    
    def get_mdtraj_cv_value(
            self, 
            model: "structures.Seekr_model", 
            traj: mdtraj.Trajectory, 
            frame_index: int = 0
            ) -> float:
        """
        Determine the current CV value for an mdtraj object.
        """
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        group1_atom_indices = md_engine.get_atom_indices_by_selection_name(
            self.group1_name)
        group2_atom_indices = md_engine.get_atom_indices_by_selection_name(
            self.group2_name)
        traj1 = traj.atom_slice(group1_atom_indices)
        traj2 = traj.atom_slice(group2_atom_indices)
        com1_array = base.mdtraj_center_of_mass(traj1)
        com2_array = base.mdtraj_center_of_mass(traj2)
        com1 = com1_array[frame_index,:]
        com2 = com2_array[frame_index,:]
        radius = np.linalg.norm(com2-com1)
        return radius
    
    def get_atom_group_names(self) -> typing.List[str]:
        """
        Get the atom group names for this CV.
        """
        return [self.group1_name, self.group2_name]

@define
class Distance_cv_input(cv_base.Base_CV_input):
    """
    Distance CV input resulting in concentric spherical anchors
    and milestones. This CV input is used as an object for the 
    input JSON or API within the prepare stage of a seekr calculation.
    """
    name: str = field(default="",
                        validator=validators.instance_of(str))
    group1_name: str = field(default="",
                             validator=validators.instance_of(str))
    group2_name: str = field(default="",
                             validator=validators.instance_of(str))
    type: typing.Literal["distance"] = "distance"
    _index: int = field(default=0,
                          validator=validators.instance_of(int))
    min_value: float = field(default=0.0,
                             validator=validators.instance_of(float))
    max_value: float = field(default=3.0,
                                validator=validators.instance_of(float))
    
    def make_cv_object(
            self, 
            index: int,
            root_directory: str) -> Distance_CV:
        """
        Create a Distance_CV object from the Input to be placed into
        the Model.
        """
        self._index = index
        cv = Distance_CV(index, self.group1_name, self.group2_name)
        if self.name == "":
            cv.name = f"distance_{self.group1_name}_{self.group2_name}"
        else:
            cv.name = self.name
        cv.min_value = self.min_value
        cv.max_value = self.max_value
        return cv
    