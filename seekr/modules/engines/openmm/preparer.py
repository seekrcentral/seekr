from __future__ import annotations
"""
modules/engines/openmm/preparer.py

Objects and routines for the prepare stage of a scope that
uses OpenMM.
"""

import os
import typing
from attrs import define, field, validators

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.engines.openmm.structures as openmm_structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.preparer as engines_preparer
import seekr.modules.scales.molecular_dynamics.structures as md_scale

@define
class Openmm_settings_input(engines_structures.Base_engine_settings_input):
    type: typing.Literal["openmm"] = "openmm"
    scale_type: typing.Literal["molecular_dynamics"] = "molecular_dynamics"
    nonbonded_method: str = field(default="pme",
                                  converter=base.string_lower,
                                  validator=[validators.instance_of(str),
                                             validators.in_(["nocutoff", 
                                                             "cutoffperiodic",
                                                             "cutoffnonperiodic",
                                                             "pme",
                                                             "ljpme",
                                                             "ewald"])])
    # MYPY: errors below with validators. But tests indicate no problems.
    nonbonded_cutoff: float | None = field(default=0.9,
                                    validator=validators.optional(
                                        [validators.instance_of(float),
                                         validators.gt(0)]))
    constraints: str = field(default="hbonds",
                             converter=base.string_lower,
                             validator=[validators.instance_of(str),
                                        validators.in_(["none", 
                                                        "hbonds", 
                                                        "allbonds",
                                                        "hangles"])])
    # MYPY: errors below with validators. But tests indicate no problems.
    hydrogen_mass: float | None = field(default=None,
                                        validator=validators.optional(
                                            [validators.instance_of(float),
                                             validators.gt(0)]))
    timestep: float = field(default=0.002,
                            validator=[validators.instance_of(float),
                                       validators.gt(0)])
    friction_coefficient: float | None = field(
        default=1.0, validator=validators.optional(validators.instance_of(float)))
    integrator_type: str = field(default="langevin",
                                 converter=base.string_lower,
                                 validator=[validators.instance_of(str),
                                            validators.in_(["langevin", 
                                                            "langevinmiddle",
                                                            "brownian",
                                                            "verlet",
                                                            "nose_hoover"])])
    platform_type: str = field(default="cuda",
                              converter=base.string_lower,
                              validator=[validators.instance_of(str),
                                         validators.in_(
                                             ["cuda", "opencl", "cpu", "reference"])])
    unpartitioned_settings: engines_preparer.MD_unpartitioned_settings_input | None = None

    def make_model_scale(
            self
            ) -> md_scale.Scale_md:
        scale = md_scale.Scale_md()
        scale.engine = "openmm"
        return scale
    
    def make_model_engine(
            self, 
            model_temperature: float
            ) -> openmm_structures.Openmm_md_engine:
        """
        Make the model engine object for this input.
        """
        openmm_engine = openmm_structures.Openmm_md_engine()
        openmm_engine.nonbonded_method = self.nonbonded_method
        openmm_engine.nonbonded_cutoff = self.nonbonded_cutoff
        openmm_engine.constraints = self.constraints
        openmm_engine.hydrogen_mass = self.hydrogen_mass
        openmm_engine.initial_temperature = model_temperature
        if self.integrator_type == "langevin":
            openmm_engine.integrator_settings = \
                engines_structures.Langevin_integrator_settings()
            openmm_engine.integrator_settings.timestep = self.timestep
            openmm_engine.integrator_settings.target_temperature = model_temperature
            openmm_engine.integrator_settings.friction = self.friction_coefficient
        elif self.integrator_type == "langevinmiddle":
            openmm_engine.integrator_settings = \
                engines_structures.Langevin_middle_integrator_settings()
            openmm_engine.integrator_settings.timestep = self.timestep
            openmm_engine.integrator_settings.target_temperature = model_temperature
            openmm_engine.integrator_settings.friction = self.friction_coefficient
        elif self.integrator_type == "brownian":
            openmm_engine.integrator_settings = \
                engines_structures.Brownian_integrator_settings()
            openmm_engine.integrator_settings.timestep = self.timestep
            openmm_engine.integrator_settings.target_temperature = model_temperature
            openmm_engine.integrator_settings.friction = self.friction_coefficient
        elif self.integrator_type == "verlet":
            openmm_engine.integrator_settings = \
                engines_structures.Verlet_integrator_settings()
            openmm_engine.integrator_settings.timestep = self.timestep
        elif self.integrator_type == "nose_hoover":
            openmm_engine.integrator_settings = \
                engines_structures.Nose_hoover_integrator_settings()
            openmm_engine.integrator_settings.timestep = self.timestep
            openmm_engine.integrator_settings.target_temperature = model_temperature
            openmm_engine.integrator_settings.collision_frequency = self.friction_coefficient
        
        if self.platform_type == "cuda":
            openmm_engine.platform_settings = engines_structures.Cuda_platform_settings()
        # TODO: implement these platforms - shouldn't be too hard.
        elif self.platform_type == "opencl":
            raise Exception("OpenCL platform not yet implemented.")
        elif self.platform_type == "cpu":
            raise Exception("CPU platform not yet implemented.")
        elif self.platform_type == "reference":
            openmm_engine.platform_settings = None
        assert self.unpartitioned_settings is not None
        openmm_engine.unpartitioned_settings \
            = self.unpartitioned_settings.make_unpartitioned_settings_for_model()
        return openmm_engine
    
    def start_from_unpartitioned(
            self,
            using_toy: bool
        ) -> bool:
        """
        Whether this engine uses unpartitioned settings to start from.
        This depends on whether this settings object has unpartitioned_settings
        defined.
        """
        if self.unpartitioned_settings is not None:
            if self.unpartitioned_settings.positions is None:
                # Then the first stage is started from the anchors
                unpartitioned_stage_input = False
            else:
                if not using_toy:
                    assert self.unpartitioned_settings\
                        .md_parameters_topology is not None, \
                        "If starting from an unpartitioned position, the topology file "\
                        "must be specified unpartitioned unless using toy mode."
                unpartitioned_stage_input = True
        else:
            # This is actually possible, if everything is defined in the anchors
            unpartitioned_stage_input = False
        return unpartitioned_stage_input
    
    def copy_initial_files(
            self,
            model: structures.Seekr_model,
            directory: str,
            anchor_set_name: str,
            using_toy: bool
        ) -> None:
        """
        Copy the MD files into the unpartitioned directory.
        """
        # NOTE: anchor_set_name not needed here, but arg needed for other
        #  engines, so keeping for consistency. Might be necessary someday
        #  for copies to anchor sets.
        engine: openmm_structures.Openmm_md_engine \
            = model.get_engine_by_scale_type("molecular_dynamics")
        assert engine.type == "openmm", \
            f"Engine for molecular_dynamics must be openmm, not {engine.type}."
        self.unpartitioned_settings.copy_initial_files(
            directory, engine.unpartitioned_settings, using_toy)
    
