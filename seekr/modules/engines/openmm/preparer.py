"""
engines/openmm/input.py

Objects and routines for the input stage of an anchor that
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
class Openmm_settings_input:
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
    
    integrator_type: str = field(default="langevin",
                                 converter=base.string_lower,
                                 validator=[validators.instance_of(str),
                                            validators.in_(["langevin", 
                                                            "langevinMiddle",
                                                            "brownian",
                                                            "verlet",])])
    platform_type: str = field(default="cuda",
                              converter=base.string_lower,
                              validator=[validators.instance_of(str),
                                         validators.in_(
                                             ["cuda", "opencl", "cpu", "reference"])])
    systemwide_settings: engines_preparer.MD_systemwide_settings_input | None = None

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
        elif self.integrator_type == "langevinMiddle":
            openmm_engine.integrator_settings = \
                engines_structures.Langevin_middle_integrator_settings()
            openmm_engine.integrator_settings.timestep = self.timestep
            openmm_engine.integrator_settings.target_temperature = model_temperature
        elif self.integrator_type == "brownian":
            openmm_engine.integrator_settings = \
                engines_structures.Brownian_integrator_settings()
            openmm_engine.integrator_settings.timestep = self.timestep
            openmm_engine.integrator_settings.target_temperature = model_temperature
        elif self.integrator_type == "verlet":
            openmm_engine.integrator_settings = \
                engines_structures.Verlet_integrator_settings()
            openmm_engine.integrator_settings.timestep = self.timestep
        
        if self.platform_type == "cuda":
            openmm_engine.platform_settings = engines_structures.Cuda_platform_settings()
        # TODO: implement these platforms - shouldn't be too hard.
        elif self.platform_type == "opencl":
            raise Exception("OpenCL platform not yet implemented.")
        elif self.platform_type == "cpu":
            raise Exception("CPU platform not yet implemented.")
        elif self.platform_type == "reference":
            openmm_engine.platform_settings = None
        assert self.systemwide_settings is not None
        openmm_engine.systemwide_settings \
            = self.systemwide_settings.make_systemwide_settings_for_model()
        return openmm_engine
    
    def start_from_systemwide(
            self,
            using_toy: bool
        ) -> bool:
        """
        Whether this engine uses systemwide settings to start from.
        This depends on whether this settings object has systemwide_settings
        defined.
        """
        if self.systemwide_settings is not None:
            if self.systemwide_settings.positions is None:
                # Then the first stage is started from the anchors
                systemwide_stage_input = False
            else:
                if not using_toy:
                    assert self.systemwide_settings\
                        .md_parameters_topology is not None, \
                        "If starting from a systemwide position, the topology file "\
                        "must be specified systemwide unless using toy mode."
                systemwide_stage_input = True
        else:
            # This is actually possible, if everything is defined in the anchors
            systemwide_stage_input = False
        return systemwide_stage_input
    
    def copy_initial_files(
            self,
            model: structures.Seekr_model,
            directory: str,
            using_toy: bool
        ) -> None:
        """
        Copy the MD files into the systemwide directory.
        """
        engine: openmm_structures.Openmm_md_engine \
            = model.get_engine_by_scale_type("molecular_dynamics")
        assert engine.type == "openmm", \
            f"Engine for molecular_dynamics must be openmm, not {engine.type}."
        self.systemwide_settings.copy_initial_files(
            directory, engine.systemwide_settings, using_toy)
    