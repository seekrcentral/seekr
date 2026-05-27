from __future__ import annotations
"""
modules/collective_variables/external_cv.py

Model and input classes for the external collective variable.
Mainly intended for toy systems, this CV takes a general function 
of atomic positions and bases a CV from it.
"""

import math
import typing

from attrs import define, field, validators, Factory
import numpy as np
import mdtraj

import seekr.modules.engines.openmm.base as engines_openmm_base
import seekr.modules.collective_variables.base as cv_base

if typing.TYPE_CHECKING:
    import seekr.modules.structures as structures

@define
class External_CV(cv_base.Base_CV):
    """
    An external collective variable is a general arbitrary function of
    atomic positions. This CV is generally intended for toy systems.

    Collective variables represent the function of system positions
    and velocities along with the anchor cells and boundaries can be
    defined.
    """
    index: int = field(default=0,
                        validator=[validators.instance_of(int),
                                  validators.ge(0)])
    group_names: list[str] = field(default=Factory(list))
    # NOTE: these are down here cause the ones above are assigned at init
    type: typing.Literal["external"] = "external"
    name: str = field(default="",
                        validator=validators.instance_of(str))
    unit: str = field(default="nm",
                      validator=validators.instance_of(str))
    expression: str = field(default="",
                            validator=validators.instance_of(str))
    num_groups: int = field(default=0, validator=validators.instance_of(int))
    variable_name = "v"
    periodic: bool = field(default=False,
                            validator=validators.instance_of(bool))
    min_value: float = field(default=-9e9, validator=validators.instance_of(float))
    max_value: float = field(default=9e9, validator=validators.instance_of(float))
    
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
    
    def get_cv_value(self, positions):
        """
        Get the value of the cv for the set of positions.
        """
        sqrt = math.sqrt
        exp = math.exp
        log = math.log
        sin = math.sin
        cos = math.cos
        tan = math.tan
        asin = math.asin
        acos = math.acos
        atan = math.atan
        sinh = math.sinh
        cosh = math.cosh
        tanh = math.tanh
        erf = math.erf
        erfc = math.erfc
        floor = math.floor
        ceil = math.ceil
        step = lambda x : 0 if x < 0 else 1
        delta = lambda x : 1 if x == 0 else 0
        select = lambda x, y, z : z if x == 0 else y
        expr = ""
        for i, position in enumerate(positions):
            expr_x = f"x{i+1} = {position[0]};"
            expr_y = f"y{i+1} = {position[1]};"
            expr_z = f"z{i+1} = {position[2]};"
            expr += expr_x + expr_y + expr_z
            
        expr += engines_openmm_base.convert_openmm_to_python_expr(
            "result="+self.expression)
        mylocals = locals()
        exec(expr, globals(), mylocals)
        result = mylocals["result"]
        return result

    def get_mdtraj_cv_value(
            self, 
            model: "structures.Seekr_model", 
            traj: mdtraj.Trajectory, 
            frame_index: int = 0
            ) -> float:
        """
        Determine the current CV value for an mdtraj object.
        """
        positions = traj.xyz[frame_index, :,:]
        value = self.get_cv_value(positions)
        return value
    
    def get_atom_group_names(self) -> typing.List[str]:
        """
        Get the atom group names for this CV.
        """
        return self.group_names

@define
class External_cv_input(cv_base.Base_CV_input):
    """
    External CV input resulting in a general function of atomic
    positions.
    """
    type: typing.Literal["external"] = "external"
    name: str = field(default="",
                        validator=validators.instance_of(str))
    group_names: list[str] = field(default=Factory(list),)
    expression: str = field(default="",
                            validator=validators.instance_of(str))
    _index: int = field(default=0,
                          validator=validators.instance_of(int))
    min_value: float = field(default=-9e9, validator=validators.instance_of(float))
    max_value: float = field(default=9e9, validator=validators.instance_of(float))
    periodic: bool = field(default=False,
                            validator=validators.instance_of(bool))
    
    def make_cv_object(
            self, 
            index: int,
            root_directory: str) -> External_CV:
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
    