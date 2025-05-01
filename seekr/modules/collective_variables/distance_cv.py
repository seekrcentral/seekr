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
    num_groups: typing.Literal[2] = 2
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