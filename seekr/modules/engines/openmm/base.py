"""

"""

import typing
from attrs import define, field, validators, Factory

import seekr.modules.engines.base as engine_base

@define
class Openmm_md_engine:
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
    nonbonded_cutoff: float = field(default=0.9,
                                    validator=[validators.instance_of(float),
                                               validators.gt(0)])
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
    initial_temperature: float = field(default=298.15,
                                       validator=[validators.instance_of(float),
                                                  validators.gt(0)])
    integrator_settings: engine_base.Integrator_settings | None = None
    # TODO: remove from this class and put in a Stage class?
    # This setting will be stage-dependent
    #barostat_settings: Monte_carlo_barostat_settings | None = None
    # If cuda_platform_settings is None, then use Reference platform
    platform_settings: engine_base.Platform_settings | None \
        = engine_base.Cuda_platform_settings
    toy_settings: engine_base.Toy_settings | None = None
    systemwide_settings: engine_base.MD_systemwide_settings | None = None

    