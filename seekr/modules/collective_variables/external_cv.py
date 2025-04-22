"""
modules/collective_variables/external_cv.py

Model and input classes for the external collective variable.
Mainly intended for toy systems, this CV takes a general function 
of atomic positions and bases a CV from it.
"""

import typing

from attrs import define, field, validators, Factory

@define
class External_CV:
    """
    An external collective variable is a general arbitrary function of
    atomic positions. This CV is generally intended for toy systems.

    Collective variables represent the function of system positions
    and velocities along with the anchor cells and boundaries can be
    defined.
    """
    index: int = field(validator=[validators.instance_of(int),
                                  validators.ge(0)])
    group_names: list[str] = field()
    name: str = field(default="",
                        validator=validators.instance_of(str))
    expression: str = field(default="",
                            validator=validators.instance_of(str))
    num_groups: int = field(default=0, validator=validators.instance_of(int))
    variable_name = "v"
    periodic: bool = field(default=False,
                            validator=validators.instance_of(bool))
    min_value: float = field(default=-9e9, validator=validators.instance_of(float))
    max_value: float = field(default=9e9, validator=validators.instance_of(float))
    type: typing.Literal["external"] = "external"
    
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
class External_cv_input:
    """
    External CV input resulting in a general function of atomic
    positions.
    """
    name: str = field(default="",
                        validator=validators.instance_of(str))
    group_names: list[str] = field(default=Factory(list),)
    expression: str = field(default="",
                            validator=validators.instance_of(str))
    type: typing.Literal["external"] = "external"
    _index: int = field(default=0,
                          validator=validators.instance_of(int))
    min_value: float = field(default=-9e9, validator=validators.instance_of(float))
    max_value: float = field(default=9e9, validator=validators.instance_of(float))
    periodic: bool = field(default=False,
                            validator=validators.instance_of(bool))
    
    def make_cv_object(
            self, 
            index: int) -> External_CV:
        """
        Create a Distance_CV object from the Input to be placed into
        the Model.
        """
        self._index = index
        cv = External_CV(index, self.group_names)
        cv.num_groups = len(self.group_names)
        if self.name == "":
            cv.name = f"external_{'_'.join(self.group_names)}"
        else:
            cv.name = self.name
        cv.expression = self.expression
        cv.min_value = self.min_value
        cv.max_value = self.max_value
        cv.periodic = self.periodic
        return cv
    
    # TODO: move this sort of check to another object? An external function?
    # To the partition???
    def check(self) -> None:
        """
        Check that this input CV is valid.
        """
        # TODO: implement the checks from SEEKR2 for distance CVs
        pass