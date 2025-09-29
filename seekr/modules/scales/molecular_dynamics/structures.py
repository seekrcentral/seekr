"""
modules/scales/molecular_dynamics/structures.py

Structures for scales within seekr: molecular_dynamics,
brownian_dynamics, ... - anything that might appear within the 
model.json file.
"""

import os
import glob
import typing
from shutil import rmtree

from attrs import define, field, validators, Factory
import cattrs

import seekr.modules.base as base
import seekr.modules.sampling_methods.structures as sampling_methods_structures
import seekr.modules.completion_criteria.structures as compl_crit_struct
import seekr.modules.restraints.structures as restraints_structures
import seekr.modules.reporters.structures as reporters_structures

sampling_methods_md \
        = {"conventional": sampling_methods_structures.Conventional_md_sampling_method,
           "metadynamics": sampling_methods_structures.Metadynamics_md_sampling_method,
           "steered": sampling_methods_structures.Steered_md_sampling_method,
           "mmvt": sampling_methods_structures.MMVT_md_sampling_method,}

@define
class Scale_md:
    """
    A molecular dynamics scale.
    """
    type: typing.Literal["molecular_dynamics"] = "molecular_dynamics"
    engine: str = field(default="openmm",
                          validator=validators.instance_of(str))
    
    def has_files_anchor(
            self, 
            anchor: "Anchor",
            root_directory: str,
            ) -> bool:
        """
        Check if the anchor has the necessary files for this scale.
        """
        for stage_parameter in anchor.stage_parameters:
            if stage_parameter is None:
                continue
            stage_directory = stage_parameter["directory"]
            stage_directory = os.path.join(
                root_directory, self.directory, stage_directory)
            if os.path.exists(stage_directory):
                # TODO: is this too general? Should the glob specify a more
                # specific pattern? Perhaps from the stage object itself?
                glob_str = "*"
                output_files_glob = os.path.join(stage_directory, glob_str)
                if len(glob.glob(output_files_glob)) > 0:
                    return True
                else:
                    continue
            else:
                continue

    def generate_filetree_anchor(
            self, 
            anchor: "Anchor",
            root_directory: str
            ) -> None:
        anchor_dict: dict[str, dict] = {anchor.directory: {}}
        for stage_parameter in anchor.stage_parameters:
            if stage_parameter is None:
                continue
            stage_directory = stage_parameter["directory"]
            anchor_dict[anchor.directory][stage_directory] = {}

        base.make_tree(root_directory, anchor_dict)

@define
class Stage_md:
    """
    A md stage of the SEEKR calculation, whether that is an initial
    equilibration, a sampling method to obtain starting structures,
    a settling stage, or a production stage for obtaining kinetics.
    """
    type: typing.Literal["md_stage"] = "md_stage"
    scale_type: typing.Literal["molecular_dynamics"] = "molecular_dynamics"
    index: int = field(default=0,
                        validator=[validators.instance_of(int),
                                      validators.ge(0)])
    name: str = field(default="equilibration", 
                      validator=validators.instance_of(str))
    run_minimization: bool = field(default=False,
                                   validator=validators.instance_of(bool))
    md_restart_checkpoint_interval: int = field(default=0,
                                                validator=[validators.instance_of(int),
                                                           validators.ge(0)])
    ensemble: base.Ensemble = field(default=Factory(base.NVT_ensemble))
    sampling_method: sampling_methods_structures.Base_md_sampling_method = field(
        default=Factory(sampling_methods_structures.Conventional_md_sampling_method))
    completion_criteria: compl_crit_struct.Base_completion_criteria = field(
        default=Factory(compl_crit_struct.Number_of_steps_completion_criteria))
    restraints: list[restraints_structures.Base_restraint] = field(default=Factory(list),)
    reporters: list[reporters_structures.Base_reporter] = field(default=Factory(list),)
    # The index of a previous stage (initial stage = 0) which provides input to
    # this stage.
    input_stage_index: int = field(default=0,
                                      validator=[validators.instance_of(int),
                                                 validators.ge(0)])
    input_from: str = field(default="systemwide", 
                            validator=validators.instance_of(str))
    output_to: str = field(default="anchor", 
                           validator=validators.instance_of(str))
    
    def get_directory_name(self) -> str:
        """
        Get the directory of this stage.
        """
        directory_name: str = f"stage_{self.index}_{self.name}"
        return directory_name

    def add_stage_parameters_to_anchors(
            self, 
            model: "Seekr_model",
            do_none: bool = False
            ) -> None:
        """
        Add the stage directory to all anchors in the model.
        """
        for anchor in model.anchors:
            if do_none:
                anchor.stage_parameters.append(None)
                continue
            directory_name: str = self.get_directory_name()
            stage_parameter: dict[str, int | str] = {}
            stage_parameter["index"] = self.index
            stage_parameter["directory"] = directory_name
            stage_parameter["positions"] = self.sampling_method.get_output_positions_basename_noext()
            anchor.stage_parameters.append(stage_parameter)
        return
    
    def add_stage_parameters_to_systemwide_settings(
            self, 
            model: "Seekr_model",
            do_none: bool = False
            ) -> None:
        """
        Add the stage parameters to the systemwide settings.
        """
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine is not None, "MD stage provided, but "\
            "MD engine settings not provided."
        assert md_engine.systemwide_settings is not None, "MD engine systemwide "\
            "settings not provided, yet MD stage is provided."
        if do_none:
            md_engine.systemwide_settings.stage_parameters.append(None)
            return
        directory_name: str = self.get_directory_name()
        stage_parameter: dict[str, int | str] = {}
        stage_parameter["index"] = self.index
        stage_parameter["directory"] = directory_name
        stage_parameter["positions"] = self.sampling_method.get_output_positions_basename_noext()
        md_engine.systemwide_settings.stage_parameters.append(stage_parameter)
        return
    
    def has_files(
            self, 
            model: "Seekr_model"
            ) -> bool:
        """
        Returns true if any of the stage directories contain
        outputs from any of the stages.
        """
        stage_output_globs: list[str] = self.sampling_method.get_output_positions_glob_list()
        my_directory = self.get_directory_name()
        for anchor in model.anchors:
            for stage_parameter in anchor.stage_parameters:
                if stage_parameter is None:
                    continue
                stage_directory = stage_parameter["directory"]
                if stage_directory == my_directory:
                    for stage_output_glob in stage_output_globs:
                        output_files_glob = os.path.join(
                            model.directory, anchor.directory, stage_directory, 
                            stage_output_glob)
                        if len(glob.glob(output_files_glob)) > 0:
                            return True
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine.systemwide_settings is not None
        for stage_parameter in md_engine.systemwide_settings.stage_parameters:
            if stage_parameter is None:
                continue
            stage_directory = stage_parameter["directory"]
            if stage_directory == my_directory:
                for stage_output_glob in stage_output_globs:
                    output_files_glob = os.path.join(
                        model.directory, stage_directory, stage_output_glob)
                    if len(glob.glob(output_files_glob)) > 0:
                        return True
        return False
    
    def remove_directory(
            self, 
            model: "Seekr_model"
            ) -> None:
        """
        Remove entire directory in all anchors and systemwide settings for this stage.
        """
        my_directory = self.get_directory_name()
        for anchor in model.anchors:
            full_directory = os.path.join(model.directory, anchor.directory, my_directory)
            if os.path.exists(full_directory):
                rmtree(full_directory)
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine.systemwide_settings is not None
        full_directory = os.path.join(
            model.directory, md_engine.systemwide_settings.directory, my_directory)
        if os.path.exists(full_directory):
            rmtree(full_directory)
        return

def struct_md_sampling_method(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[sampling_methods_structures.Base_md_sampling_method]
        ) -> sampling_methods_structures.Base_md_sampling_method:
    typ = obj.pop("type")
    return cattrs.structure(obj, sampling_methods_md[typ])

def unstruct_md_sampling_method(
        obj: sampling_methods_structures.Base_md_sampling_method
        ) -> typing.Dict[str, typing.Any]:
    """
    Unstructure a completion criteria object to a dictionary.
    """
    d = cattrs.unstructure(obj)
    d["type"] = obj.type
    return d

def register_structure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the structure hooks for all scales and their sampling methods.
    """
    converter.register_structure_hook(sampling_methods_structures.Base_md_sampling_method,
                                        struct_md_sampling_method)
    return
    
def register_unstructure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the unstructure hooks for all scales and their sampling methods.
    """
    converter.register_unstructure_hook(sampling_methods_structures.Base_md_sampling_method,
                                        unstruct_md_sampling_method)
    return