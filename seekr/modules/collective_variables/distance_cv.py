from __future__ import annotations
"""
modules/collective_variables/distance_cv.py

Model and input classes for the distance collective variable.
This CV measures the distance between the centers of mass of two 
groups of atoms.
"""

import typing
import logging

from attrs import define, field, validators
import numpy as np
import mdtraj

import seekr.modules.base as base
import seekr.modules.collective_variables.base as cv_base

logger = logging.getLogger(__name__)

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
    index: int = field(default=0,
                        validator=[validators.instance_of(int),
                                  validators.ge(0)])
    group1_name: str = field(default="",
                            validator=validators.instance_of(str))
    group2_name: str = field(default="",
                            validator=validators.instance_of(str))
    # NOTE: these are down here cause the ones above are assigned at init
    type: typing.Literal["distance"] = "distance"
    name: str = field(default="",
                        validator=validators.instance_of(str))
    unit: str = field(default="nm",
                      validator=validators.instance_of(str))
    num_groups: typing.Literal[2] = 2
    variable_name = "r"
    periodic: bool = False
    min_value: float = 0.0
    max_value: float = 3.0
    
    
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
            masses: np.ndarray | None = None,
            ) -> float:
        """
        Determine the current CV value for an mdtraj object.

        Uses the same COM definition as OpenMM CustomCentroidBondForce:
        mass-weighted group COM and minimum-image PBC on the group–group
        vector when unit cell vectors are present.

        Masses (amu) prefer OpenMM System particles (HMR included) when
        available. Pass `masses` to override. If neither is available,
        element masses plus HMR from the mdtraj topology bonds are used.
        """
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        group1_atom_indices = md_engine.get_atom_indices_by_selection_name(
            self.group1_name)
        group2_atom_indices = md_engine.get_atom_indices_by_selection_name(
            self.group2_name)
        if masses is None:
            from seekr.modules.engines.openmm import sim as openmm_sim
            omm_masses = openmm_sim.particle_masses_from_openmm_parameters(
                model)
            if omm_masses is not None and len(omm_masses) == traj.n_atoms:
                masses = omm_masses
            elif omm_masses is not None:
                logger.warning(
                    "OpenMM particle masses length %s does not match "
                    "trajectory n_atoms %s; falling back to mdtraj masses.",
                    len(omm_masses), traj.n_atoms)
        if masses is None:
            masses = base.particle_masses_from_mdtraj_topology(traj.topology)
            hydrogen_mass = getattr(md_engine, "hydrogen_mass", None)
            if isinstance(hydrogen_mass, (int, float)):
                masses, n_transfers = base.apply_hydrogen_mass_repartitioning(
                    masses, traj.topology, float(hydrogen_mass))
                if n_transfers == 0:
                    logger.warning(
                        "hydrogen_mass=%s is set but the topology has no "
                        "H–heavy bonds; element masses will be used. "
                        "A PDB without CONECT/connectivity cannot apply HMR.",
                        hydrogen_mass)
        else:
            masses = np.asarray(masses, dtype=float)
            if len(masses) != traj.n_atoms:
                raise ValueError(
                    f"masses length {len(masses)} does not match "
                    f"trajectory n_atoms {traj.n_atoms}")
        box = None
        if traj.unitcell_vectors is not None:
            box = np.asarray(traj.unitcell_vectors[frame_index], dtype=float)
        xyz = np.asarray(traj.xyz[frame_index], dtype=float)
        return base.group_com_distance(
            xyz, group1_atom_indices, group2_atom_indices, masses, box)
    
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
    type: typing.Literal["distance"] = "distance"
    name: str = field(default="",
                        validator=validators.instance_of(str))
    group1_name: str = field(default="",
                             validator=validators.instance_of(str))
    group2_name: str = field(default="",
                             validator=validators.instance_of(str))
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
    