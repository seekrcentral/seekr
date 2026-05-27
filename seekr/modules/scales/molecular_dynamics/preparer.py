from __future__ import annotations
"""
modules/scales/molecular_dynamics/preparer.py

Molecular dynamics input objects and procedures.
"""

import typing

from attrs import define, field, validators, Factory
import cattrs

import seekr.modules.base as base
import seekr.modules.scales.base as scales_base
import seekr.modules.completion_criteria.structures as compl_crit_struct
import seekr.modules.scales.molecular_dynamics.structures as scale_md
import seekr.modules.engines.openmm.preparer as openmm_preparer
import seekr.modules.engines.preparer as engines_preparer
import seekr.modules.restraints.preparer as restraints_preparer
import seekr.modules.reporters.preparer as reporters_preparer
import seekr.modules.sampling_methods.preparer as sampling_methods_preparer

MD_settings_input = typing.Union[openmm_preparer.Openmm_settings_input]

sampling_methods_input_md: typing.Dict[str, typing.Any] \
        = {"conventional": sampling_methods_preparer.Conventional_md_input,
           "metadynamics": sampling_methods_preparer.Metadynamics_md_input,
           "steered": sampling_methods_preparer.Steered_md_input,
           "mmvt": sampling_methods_preparer.MMVT_md_input,}

@define
class Input_stage_md(scales_base.Base_input_stage):
    """
    A MD stage of the SEEKR calculation, whether that is an initial
    equilibration, a sampling method to obtain starting structures,
    a settling stage, or a production stage for obtaining kinetics.
    """
    type: typing.Literal["md_stage"] = "md_stage"
    name: str = field(default="equilibration", 
                      validator=validators.instance_of(str))
    input_stage_name: str | None = field(default=None,
                                 validator=validators.optional(
                                     validators.instance_of(str)))
    run_minimization: bool = field(default=False,
                                      validator=validators.instance_of(bool))
    checkpoint_interval: int = field(default=1000,
                                  validator=[validators.instance_of(int),
                                             validators.gt(0)])
    ensemble: base.Ensemble = field(default=Factory(
        base.NVT_ensemble))
    sampling_method: sampling_methods_preparer.Base_md_sampling_method_input = field(
        default=Factory(sampling_methods_preparer.Conventional_md_input))
    completion_criteria: compl_crit_struct.Base_completion_criteria | None = None
    restraints: list[restraints_preparer.Base_restraint_input] = field(default=Factory(list))
    reporters: list[reporters_preparer.Base_reporter_input] = field(default=Factory(list))
    
    def get_scale_type(self) -> str:
        return "molecular_dynamics"

    def make_model_stage(
            self, 
            index: int,
            scope: str,
            md_settings: MD_settings_input
            ) -> scale_md.Stage_md:
        """
        Given this input stage object, create the model's Stage object.
        """
        unpartitioned_settings_input: engines_preparer.MD_unpartitioned_settings_input \
            = md_settings.unpartitioned_settings
        stage = scale_md.Stage_md()
        stage.index = index
        stage.name = self.name
        stage.run_minimization = self.run_minimization
        stage.md_restart_checkpoint_interval = self.checkpoint_interval
        stage.ensemble = self.ensemble
        stage.sampling_method = self.sampling_method.make_model_sampling_method()
        assert self.completion_criteria is not None
        stage.completion_criteria = self.completion_criteria
        if scope == "unpartitioned": # If coming from unpartitioned settings
            assert not stage.sampling_method.needs_to_be_in_anchor(), \
                f"Stage '{stage.name}' with sampling method '{stage.sampling_method.type}' "\
                "must be run within one or more anchors. A stage that populates anchors "\
                "must precede this stage."
            
        stage.scope = scope
        for restraint_input in self.restraints:
            restraint = restraint_input.make_restraint(unpartitioned_settings_input)
            stage.restraints.append(restraint)
        for reporter_input in self.reporters:
            reporter = reporter_input.make_reporter(unpartitioned_settings_input)
            stage.reporters.append(reporter)
        type_counters: dict[str, int] = {}
        for reporter in stage.reporters:
            if reporter.name is None:
                base = f"{reporter.type}_reporter"
                count = type_counters.get(base, 0) + 1
                type_counters[base] = count
                reporter.name = f"{base}_{count}"
        # Now check all reporter names are unique
        reporter_names = [reporter.name for reporter in stage.reporters]
        assert len(reporter_names) == len(set(reporter_names)), \
            f"Reporter names must be unique within a stage."
        return stage
    
def struct_md_sampling_method_inputs(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[sampling_methods_preparer.Base_md_sampling_method_input]
        ) -> sampling_methods_preparer.Base_md_sampling_method_input:
    typ = obj.get("type")
    return cattrs.structure(obj, sampling_methods_input_md[typ])

def register_structure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the structure hooks for all scales and their sampling methods.
    """
    converter.register_structure_hook(
        sampling_methods_preparer.Base_md_sampling_method_input,
        struct_md_sampling_method_inputs
    )