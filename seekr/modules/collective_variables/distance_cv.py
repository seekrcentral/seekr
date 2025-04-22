"""
modules/collective_variables/distance_cv.py

Model and input classes for the distance collective variable.
This CV measures the distance between the center of mass of two 
groups of atoms.
"""

import typing

from attrs import define, field, validators

@define
class Distance_CV:
    """
    A distance collective variable represents the distance between two
    different groups of atoms.

    Collective variables represent the function of system positions
    and velocities along with the anchor cells and boundaries can be
    defined.
    """
    index: int = field(validator=[validators.instance_of(int),
                                  validators.ge(0)])
    group1_name: str = field(validator=validators.instance_of(str))
    group2_name: str = field(validator=validators.instance_of(str))
    name: str = field(default="",
                        validator=validators.instance_of(str))
    #cv_expression: typing.Literal["distance(g1, g2)"] = "distance(g1, g2)"
    num_groups: typing.Literal[2] = 2
    #per_dof_variables: typing.List[str] = ["k", "value"]
    #global_variables: typing.List[str] = []
    variable_name = "r"
    periodic: bool = False
    min_value: float = 0.0
    max_value: float = 3.0
    type: typing.Literal["distance"] = "distance"
    
    def fill_input_values_from_model(
            self, # TODO: use kwargs?
            root_directory: str) -> None:
        """
        Fill extra values from model immediately after creation of this object.

        Example: if the CV requires knowledge of the root directory, for
        instance to have a place for a reference structure, this function
        can be used to fill in the value.
        """
        return
    
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

    # TODO: break down these functions into parts that can depend on
    #  the Selection object, then employ the mathematical function of
    #  this CV to determine the value.
    #  Should these be moved to unattached functions?
    """
    def get_mdtraj_cv_value(self, traj, frame_index):
        ""
        Determine the current CV value for an mdtraj object.
        ""
        traj1 = traj.atom_slice(self.group1)
        traj2 = traj.atom_slice(self.group2)
        com1_array = mmvt_cv_base.traj_center_of_mass(traj1)
        com2_array = mmvt_cv_base.traj_center_of_mass(traj2)
        
        #if traj1.xyz.shape[1] == 1:
        #    com1_array = traj1.xyz
        #else:
        #    com1_array = mdtraj.compute_center_of_mass(traj1)
        
        #if traj2.xyz.shape[1] == 1:
        #    com2_array = traj2.xyz
        #else:
        #    com2_array = mdtraj.compute_center_of_mass(traj2)
            
        com1 = com1_array[frame_index,:]
        com2 = com2_array[frame_index,:]
        radius = np.linalg.norm(com2-com1)
        return radius
    
        
    def check_mdtraj_within_boundary(self, traj, milestone_variables, 
                                     verbose=False, TOL=0.001):
        ""
        Check if an mdtraj Trajectory describes a system that remains
        within the expected anchor. Return True if passed, return
        False if failed.
        ""
        for frame_index in range(traj.n_frames):
            radius = self.get_mdtraj_cv_value(traj, frame_index)
            result = self.check_value_within_boundary(
                radius, milestone_variables, verbose, tolerance=TOL)
            if not result:
                return False
            
        return True
    """

    
    """
    # This function can be used for both mdtraj and OpenMM
    def check_value_within_boundary(self, radius, milestone_variables, 
                                    verbose=False, tolerance=0.0):
        ""
        Check if the given CV value is within the milestone's boundaries.
        ""
        milestone_k = milestone_variables["k"]
        milestone_radius = milestone_variables["radius"]
        if milestone_k*(radius - milestone_radius) > tolerance:
            if verbose:
                warnstr = ""The center of masses of atom group1 and atom
group2 were found to be {:.4f} nm apart.
This distance falls outside of the milestone
boundary at {:.4f} nm."".format(radius, milestone_radius)
                print(warnstr)
            return False
        return True
    
    
    # This function can also make use of MDtraj implementations within the
    #  Selections
    def check_mdtraj_close_to_boundary(self, traj, milestone_variables, 
                                     verbose=False, max_avg=0.03, max_std=0.05):
        ""
        Given an mdtraj Trajectory, check if the system lies close
        to the MMVT boundary. Return True if passed, return False if 
        failed.
        ""
        traj1 = traj.atom_slice(self.group1)
        traj2 = traj.atom_slice(self.group2)
        
        com1_array = mmvt_cv_base.traj_center_of_mass(traj1)
        com2_array = mmvt_cv_base.traj_center_of_mass(traj2)
        #if traj1.xyz.shape[1] == 1:
        #    com1_array = traj1.xyz
        #else:
        #    com1_array = mdtraj.compute_center_of_mass(traj1)
        
        #if traj2.xyz.shape[1] == 1:
        #    com2_array = traj2.xyz
        #else:
        #    com2_array = mdtraj.compute_center_of_mass(traj2)
        
        distances = []
        for frame_index in range(traj.n_frames):
            com1 = com1_array[frame_index,:]
            com2 = com2_array[frame_index,:]
            radius = np.linalg.norm(com2-com1)
            milestone_radius = milestone_variables["radius"]
            distances.append(radius - milestone_radius)
            
        assert np.isfinite(distances).all(), "Non-finite numbers detected in \
            'distances'."
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        if abs(avg_distance) > max_avg or std_distance > max_std:
            if verbose:
                warnstr = ""The distance between the system and central 
    milestone were found on average to be {:.4f} nm apart.
    The standard deviation was {:.4f} nm."".format(avg_distance, std_distance)
                print(warnstr)
            return False
            
        return True
    """
    
    # TODO: employ the Model object to obtain the groups needed
    #  by this CV
    """
    def get_atom_groups(self):
        ""
        Return a 2-list of this CV's atomic groups.
        ""
        return [self.group1, self.group2]
            
    # TODO: remove?
    def get_variable_values(self):
        ""
        This type of CV has no extra variables, so an empty list is 
        returned.
        ""
        return []
    """

@define
class Distance_cv_input:
    """
    Distance CV input resulting in concentric spherical anchors
    and milestones.
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
            index: int) -> Distance_CV:
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
    
    # TODO: move this sort of check to another object? An external function?
    # To the partition???
    def check(self) -> None:
        """
        Check that this input CV is valid.
        """
        # TODO: implement the checks from SEEKR2 for distance CVs
        pass