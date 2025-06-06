"""
modules/collective_variables/rmsd_cv.py

Model and input classes for the RMSD collective variable.
This CV measures the RMSD between the same group of atoms,
one structure is the current moment in the simulation, and 
the other is a reference structure. The atoms are aligned
by another (or the same) set of atoms.
"""

import os
import typing
import shutil

from attrs import define, field, validators
import numpy as np
#import numpy.typing as nptyping
import mdtraj

import seekr.modules.base as base
import seekr.modules.collective_variables.base as cv_base

if typing.TYPE_CHECKING:
    import seekr.modules.structures as structures

@define
class RMSD_CV(cv_base.Base_CV):
    """
    A RMSD collective variable represents the RMSD of a group of
    atoms compared to a reference structure.

    Collective variables represent the function of system positions
    and velocities along with the anchor cells and boundaries can be
    defined.
    """
    index: int = field(validator=[validators.instance_of(int),
                                  validators.ge(0)])
    directory: str = field(validator=validators.instance_of(str))
    group_rmsd_name: str = field(validator=validators.instance_of(str))
    group_align_name: str = field(validator=validators.instance_of(str))
    ref_structure: str = field(validator=validators.instance_of(str))
    name: str = field(default="",
                        validator=validators.instance_of(str))
    num_groups: typing.Literal[2] = 2
    variable_name = "v"
    periodic: bool = False
    min_value: float = 0.0
    max_value: float = 3.0
    type: typing.Literal["rmsd"] = "rmsd"
    
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
            frame_index: int = 0,
            #ref_positions: nptyping.array_like | None = None # NOTE: implement someday?
            ) -> float:
        """
        Determine the current CV value for an mdtraj object.
        """
        group_rmsd_atom_indices = model.md_engine.get_atom_indices_by_selection_name(
            self.group_rmsd_name)
        group_align_atom_indices = model.md_engine.get_atom_indices_by_selection_name(
            self.group_align_name)
        #if ref_positions is None:
        ref_traj_filename = os.path.join(
            model.directory, self.directory, self.ref_structure)
        ref_traj = mdtraj.load(ref_traj_filename)
        traj.superpose(ref_traj, atom_indices=group_align_atom_indices)
        ref_xyz = ref_traj.xyz[0]
        traj_xyz = traj.xyz[frame_index]
        value = 0.0
        for index in group_rmsd_atom_indices:
            increment = (traj_xyz[index,0] - ref_xyz[index,0])**2 \
                + (traj_xyz[index,1] - ref_xyz[index,1])**2 \
                + (traj_xyz[index,2] - ref_xyz[index,2])**2
            value += increment
        
        rmsd = np.sqrt(value/len(group_rmsd_atom_indices))
        return rmsd
    
    def get_atom_group_names(self) -> typing.List[str]:
        """
        Get the atom group names for this CV.
        """
        return [self.group_rmsd_name, self.group_align_name]

@define
class RMSD_cv_input(cv_base.Base_CV_input):
    """
    RMSD CV input resulting in concentric spherical anchors
    and milestones.
    """
    name: str = field(default="",
                        validator=validators.instance_of(str))
    group_rmsd_name: str = field(default="",
                             validator=validators.instance_of(str))
    group_align_name: str = field(default=None,
                             validator=validators.optional(validators.instance_of(str)))
    ref_structure: str = field(default="",
                             validator=validators.instance_of(str))
    type: typing.Literal["rmsd"] = "rmsd"
    _index: int = field(default=0,
                          validator=validators.instance_of(int))
    min_value: float = field(default=0.0,
                             validator=validators.instance_of(float))
    max_value: float = field(default=3.0,
                                validator=validators.instance_of(float))
    
    def make_cv_object(
            self, 
            index: int,
            root_directory: str
            ) -> RMSD_CV:
        """
        Create a RMSD_CV object from the Input to be placed into
        the Model.
        """
        self._index = index
        directory = f"collective_variable_{index}"
        if self.name == "":
            if self.group_align_name is None:
                name = f"rmsd_{self.group_rmsd_name}"
            else:
                name = f"rmsd_{self.group_rmsd_name}_{self.group_align_name}"
        else:
            name = self.name
        if self.group_align_name is None:
            self.group_align_name = self.group_rmsd_name
        cv = RMSD_CV(index, directory, self.group_rmsd_name, self.group_align_name,
                      self.ref_structure)
        cv.name = name
        cv.min_value = self.min_value
        cv.max_value = self.max_value
        cv_tree = {cv.directory: {}}
        base.make_tree(root_directory, cv_tree)
        rmsd_ref_pdb = f"rmsd_reference_cv_{index}.pdb"
        absolute_RMSD_ref_pdb = os.path.join(root_directory, cv.directory, rmsd_ref_pdb)
        assert os.path.exists(self.ref_structure), \
            f"RMSD CV Reference structure file {self.ref_structure} does not exist."
        shutil.copyfile(self.ref_structure, absolute_RMSD_ref_pdb)
        return cv
    
    # TODO: move this sort of check to another object? An external function?
    # To the partition???
    def check(self) -> None:
        """
        Check that this input CV is valid.
        """
        # TODO: implement the checks from SEEKR2 for distance CVs
        pass
