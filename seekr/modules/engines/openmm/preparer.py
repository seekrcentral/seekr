"""
engines/openmm/input.py

Objects and routines for the input stage of an anchor that
uses OpenMM.
"""

import typing
from attrs import define, field, validators

import seekr.modules.engines.openmm.structures as openmm_structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.preparer as engines_preparer

@define
class Openmm_settings_input:
    type: typing.Literal["openmm"] = "openmm"
    nonbonded_method: str = field(default="pme",
                                  converter=str.lower,
                                  validator=[validators.instance_of(str),
                                             validators.in_(["nocutoff", 
                                                             "cutoffperiodic",
                                                             "cutoffnonperiodic",
                                                             "pme",
                                                             "ljpme",
                                                             "ewald"])])
    nonbonded_cutoff: float | None = field(default=0.9,
                                    validator=validators.optional(
                                        [validators.instance_of(float),
                                         validators.gt(0)]))
    constraints: str = field(default="hbonds",
                             converter=str.lower,
                             validator=[validators.instance_of(str),
                                        validators.in_(["none", 
                                                        "hbonds", 
                                                        "allbonds",
                                                        "hangles"])])
    hydrogen_mass: float | None = field(default=None,
                                        validator=validators.optional(
                                            [validators.instance_of(float),
                                             validators.gt(0)]))
    timestep: float = field(default=0.002,
                            validator=[validators.instance_of(float),
                                       validators.gt(0)])
    
    integrator_type: str = field(default="langevin",
                                 converter=str.lower,
                                 validator=[validators.instance_of(str),
                                            validators.in_(["langevin", ])])
    platform_type: str = field(default="cuda",
                              converter=str.lower,
                              validator=[validators.instance_of(str),
                                         validators.in_(
                                             ["cuda", "opencl", "cpu", "reference"])])
    systemwide_settings: engines_preparer.MD_systemwide_settings_input | None = None

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
        
        if self.platform_type == "cuda":
            openmm_engine.platform_settings = engines_structures.Cuda_platform_settings()
        elif self.platform_type == "opencl":
            raise Exception("OpenCL platform not yet implemented.")
        elif self.platform_type == "cpu":
            raise Exception("CPU platform not yet implemented.")
        elif self.platform_type == "reference":
            openmm_engine.platform_settings = None

        #openmm_engine.systemwide_settings = self.systemwide_settings
        openmm_engine.systemwide_settings \
            = self.systemwide_settings.make_systemwide_settings_for_model()
        return openmm_engine
    
    def assign_toy_settings(
            self, 
            seekr_model_toy_settings : engines_structures.Toy_settings) -> None:
        """
        Assign the toy settings to the model engine.
        """
        seekr_model_toy_settings.toy_settings = engines_structures.Toy_settings()
        seekr_model_toy_settings.toy_settings.potential_energy_expression = self.potential_energy_expression
        seekr_model_toy_settings.toy_settings.num_particles = self.num_particles
        seekr_model_toy_settings.toy_settings.masses = self.masses
        return