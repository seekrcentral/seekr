from __future__ import annotations
"""
modules/preparer.py

Routines for preparing the model file given input that contains:
 - physical settings (temperature, pressure, etc.)
 - simulation engine specific settings (OpenMM, NAMD, Browndye, SDA, etc.)
 - SEEKR calculation type (MMVT, other milestoning, MetaD, steered MD, RAMD, 
      scaled MD, etc.)
 - CV and Anchor information)

"""

import os
import glob
import json
from collections import defaultdict
import typing
import shutil
import random
import logging
import importlib

from attrs import define, field, validators, Factory
import cattrs

import seekr.modules.base as base
import seekr.modules.anchor_schemes.base as anchor_schemes_base
import seekr.modules.collective_variables.base as cv_base
import seekr.modules.engines.base as engines_base
import seekr.modules.scales.base as scales_base
import seekr.modules.structures as structures
import seekr.modules.completion_criteria.structures as compl_crit_struct
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.browndye.preparer as browndye_preparer
import seekr.modules.engines.sda.preparer as sda_preparer
import seekr.modules.engines.openmm.preparer as openmm_preparer
import seekr.modules.restraints.preparer as restraints_preparer
import seekr.modules.reporters.preparer as reporters_preparer
import seekr.modules.scales.preparer as scales_preparer
import seekr.modules.anchor_schemes.preparer as anchor_schemes_preparer
import seekr.modules.scales.molecular_dynamics.preparer as md_preparer
import seekr.modules.scales.brownian_dynamics.preparer as bd_preparer
import seekr.modules.scales.logistic\
    .anchor_starting_structures_from_reporter_logistic_stage \
        as anch_start_struct_rpt_stage
import seekr.modules.scales.logistic.reassign_anchors_logistic_stage \
    as reassign_anchors_logistic_stage

engine_inputs: typing.Dict[str, typing.Type[engines_structures.Base_engine]] \
        = {"browndye": browndye_preparer.Browndye_settings_input,
           "sda": sda_preparer.SDA_settings_input,
           "openmm": openmm_preparer.Openmm_settings_input}
stage_inputs: typing.Dict[str, typing.Type[scales_base.Base_input_stage]] \
        = {"md_stage": md_preparer.Input_stage_md,
           "bd_stage": bd_preparer.Input_stage_bd,
           "anchor_starting_structures_from_reporter_stage": \
                anch_start_struct_rpt_stage\
                    .Input_stage_anchor_starting_structures_from_reporter,
            "reassign_anchors_stage": reassign_anchors_logistic_stage\
                .Input_stage_reassign_anchors}
restraint_inputs: typing.Dict[str, typing.Type[restraints_preparer.Base_restraint_input]] \
        = {"positional": restraints_preparer.Positional_restraint_input,
           "pairwise": restraints_preparer.Pairwise_restraint_input,
           "proximity": restraints_preparer.Proximity_restraint_input,
           "cylinder": restraints_preparer.Cylinder_restraint_input,}
reporter_inputs: typing.Dict[str, typing.Type[reporters_preparer.Base_reporter_input]] \
        = {"position": reporters_preparer.Position_reporter_input,
           "state": reporters_preparer.State_reporter_input,
           "cv_value": reporters_preparer.CV_value_reporter_input,
           "anchor": reporters_preparer.Anchor_reporter_input
           }
compl_crit_inputs: typing.Dict[str, typing.Type[compl_crit_struct.Base_completion_criteria]] \
        = structures.completion_criteria
anchor_schemes: typing.Dict[str, typing.Type[anchor_schemes_base.Base_anchor_scheme]] \
         = {"voronoi_cells": anchor_schemes_preparer.Anchors_as_voronoi_cells_anchor_scheme}
from seekr.modules.collective_variables.preparer import cv_inputs

@define
class Toy_settings_input:
    potential_energy_expression: str = field(default="",
                                    validator=validators.instance_of(str))
    num_particles: int = field(default=1,
                                validator=[validators.instance_of(int),
                                           validators.gt(0)])
    masses: list[float] = field(default=Factory(list),
                                validator=validators.deep_iterable(
                                    member_validator=validators.instance_of(float),
                                    iterable_validator=validators.instance_of(list),
                                ))
    
    def assign_toy_settings(
            self, 
            seekr_model_engine : engines_structures.Base_engine
            ) -> None:
        """
        Assign the toy settings to the model engine.
        """
        seekr_model_engine.toy_settings = engines_structures.Toy_settings()
        seekr_model_engine.toy_settings.potential_energy_expression = self.potential_energy_expression
        seekr_model_engine.toy_settings.num_particles = self.num_particles
        seekr_model_engine.toy_settings.masses = self.masses
        return

@define
class Partition_input:
    """
    Partition information concerns how the system is divided into
    partitions - that is, involving functions of the N-D positions.
    This means CVs, anchors, state_points, etc.
    """
    cvs: list[cv_base.Base_CV_input] = field(default=Factory(list),)
                            #validator=validators.deep_iterable(
                            #    member_validator=validators.instance_of(
                            #        cv_preparer.CV_input),
                            #    iterable_validator=validators.instance_of(list),
                            #))
    state_points: list[base.State_point] = field(default=Factory(list),)
                            #validator=validators.deep_iterable(
                            #    member_validator=validators.instance_of(
                            #        State_point),
                            #    iterable_validator=validators.instance_of(list),
                            #))
    # TODO: How are we going to define Voronoi points independently from
    #  anchor information such as structures, etc.?
    #anchors: list[Anchor_input] = field(default=Factory(list),
    #                        validator=validators.deep_iterable(
    #                            member_validator=validators.instance_of(
    #                                Anchor_input),
    #                            iterable_validator=validators.instance_of(list),
    #                        ))
    anchor_scheme: anchor_schemes_base.Base_anchor_scheme | None = field(
        default=None)
    
    def make_anchors(
            self,
            starting_anchor_index: int,
            starting_milestone_index: int,
            connection_flag_dict: dict[int|str, list[structures.Anchor]],
            associated_input_anchor: dict[int, typing.Any]
            ) -> typing.Tuple[list[structures.Anchor], list[tuple[int, int]], int]:
        """

        """
        return self.anchor_scheme.make_anchors(
            self.cvs, starting_anchor_index, starting_milestone_index,
            connection_flag_dict, associated_input_anchor
        )

@define
class Seekr_input:
    """
    Parameters that are input by the user.
    """
    directory: str = field(default="",
                            validator=validators.instance_of(str))
    plugins: list[str] = field(default=Factory(list),)
    temperature: float = field(default=298.15,
                               validator=[validators.instance_of(float),
                                          validators.gt(0)])
    random_seed: int | None = field(default=None,
                                    validator=validators.optional(
                                        validators.instance_of(int)))
    toy_settings: Toy_settings_input | None = None
    engine_settings_list: list[engines_structures.Base_engine_settings_input] = field(
        default=Factory(list),)
    initial_scopes: dict[str, str] | None = None
    stages: list[scales_base.Base_input_stage] = field(default=Factory(list),)
                            #validator=validators.deep_iterable(
                            #    member_validator=validators.instance_of(
                            #        Input_stage),
                            #    iterable_validator=validators.instance_of(list),
                            #))
    # A single partition can contain multiple CVs, state_points, and anchors
    # and multiple partitions can be connected using the connection flags.
    # Multiple partitions would be used to represent, for instance, multiple
    # binding sites.
    partitions: list[Partition_input] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=validators.instance_of(
                                    Partition_input),
                                iterable_validator=validators.instance_of(list),
                            ))
    
    def create_model(
            self, 
            use_absolute_directory: bool = False
            ) -> structures.Seekr_model:
        """
        Given the Seekr_input object, which contains the settings, 
        create the Model object which contains more detailed 
        information.
        """
        model = structures.Seekr_model()
        if use_absolute_directory:
            model.directory = self.directory
        else:
            model.directory = "."
        # Check the input plugins
        for plugin in self.plugins:
            spec = importlib.util.find_spec(plugin)
            assert spec is not None, \
                f"Plugin '{plugin}' not found. Please install it or check the spelling."
        model.plugins = self.plugins
        model.temperature = self.temperature
        for settings in self.engine_settings_list:
            scale = settings.make_model_scale()
            model.scales.append(scale)
            engine = settings.make_model_engine(self.temperature)
            model.engines.append(engine)

        if self.toy_settings is None:
            model.is_toy_system = False
        else:
            model.is_toy_system = True
            assigned_toy_params = False
            for scale in model.scales:
                engine = model.get_engine_by_scale_type(scale.type)
                if engine.can_handle_toy():
                    assigned_toy_params = True
                    self.toy_settings.assign_toy_settings(engine)

            assert assigned_toy_params, \
                "Toy settings were provided, but no capable engine was found to "\
                    "assign them to."
        
        # Assign stage information
        model.stages = []
        stage_names = set()
        settings_by_scale: dict[str, typing.Any] = {}
        for stage_input in self.stages:
            if stage_input.get_scale_type() not in settings_by_scale:
                settings_by_scale[stage_input.get_scale_type()] = None
        for settings in self.engine_settings_list:
            settings_by_scale[settings.scale_type] = settings
        
        last_stage_index_by_scale: dict[str, int] = defaultdict(int)
        last_any_stage_index: int = 0
        stage_name_to_index: dict[str, int] = {}
        current_scopes = {}
        current_scope = "unpartitioned"
        if self.initial_scopes is None:
            self.initial_scopes = {}
            for scale in settings_by_scale.keys():
                self.initial_scopes[scale] = "unpartitioned"
        
        for scale in settings_by_scale.keys():
            assert scale in self.initial_scopes, \
                f"Initial scope for scale '{scale}' not found."
            assert self.initial_scopes[scale] in ["unpartitioned", "partitioned"], \
                f"Invalid initial scope '{self.initial_scopes[scale]}' for scale '{scale}'."
            current_scopes[scale] = self.initial_scopes[scale]
        
        # Logistic stages may assign the scope of future stages, keep track
        #  of this here.
        scope_future = {} # key: stage_name, value: scope name

        for i, stage_input in enumerate(self.stages):
            scale_type = stage_input.get_scale_type()
            settings = settings_by_scale[scale_type]
            if scale_type == "logistic":
                stage, new_scope = stage_input.make_model_stage(
                    i+1, current_scope, settings)
                if new_scope is not None:
                    assert i + 1 < len(self.stages), \
                        "A scope-changing logistic stage cannot be the last stage."
                    scope_future[self.stages[i+1].name] = new_scope
                stage.input_stage_index = last_any_stage_index
                if stage_input.input_stage_name is not None:
                    stage.input_stage_index = stage_name_to_index[stage_input.input_stage_name]
                # A logistic stage advances the last-seen index for all simulation scale types
                for k in settings_by_scale:
                    if k != "logistic":
                        last_stage_index_by_scale[k] = stage.index
            else:
                if stage_input.name in scope_future:
                    current_scopes[scale_type] = scope_future[stage_input.name]
                current_scope = current_scopes[scale_type]
                stage = stage_input.make_model_stage(i+1, current_scope, settings)
                stage.input_stage_index = last_stage_index_by_scale[scale_type]
                if stage_input.input_stage_name is not None:
                    stage.input_stage_index = stage_name_to_index[stage_input.input_stage_name]
                last_stage_index_by_scale[scale_type] = stage.index
            
            model.stages.append(stage)
            stage_name_to_index[stage.name] = stage.index
            last_any_stage_index = stage.index
            assert stage.name != "initial", \
                "Stage name 'initial' is reserved."
            assert stage.name not in stage_names, \
                f"Duplicate stage name '{stage.name}' detected."
            stage_names.add(stage.name)

        if self.random_seed is not None:
            model.set_random_seeds(self.random_seed)
        else:
            model.set_random_seeds(random.randint(0, 1000000))
        
        return model

def generate_directory(
        directory: str, 
        empty_dir: bool = False
        ) -> None:
    """
    Create (or recreate) a directory, optionally overwrite.
    
    Parameters:
    -----------
    directory : str
        A string path to the new directory.
    
    empty_dir : bool, default False
        Whether to delete an existing directory, if it exists.
    
    """
    if empty_dir and os.path.isdir(directory):
        logging.info("Deleting all subdirectories/files in directory:", 
                     directory)
        shutil.rmtree(directory)
        
    if not os.path.isdir(directory):
        os.mkdir(directory)
    
    return
    
def read_input(input_filename: str) -> Seekr_input:
    """
    Read the input JSON file into the input object.
    """
    global engine_inputs
    global stage_inputs
    global restraint_inputs
    global reporter_inputs
    global compl_crit_inputs
    global anchor_schemes
    # Use a cattrs Converter to define how to structure the dictionary input
    converter = cattrs.Converter()

    def struct_restraint_inputs(
            obj: typing.Dict[str, typing.Any],
            _: typing.Type[restraints_preparer.Base_restraint_input]
            ) -> restraints_preparer.Base_restraint_input:
        typ = obj.get("type")
        return converter.structure(obj, restraint_inputs[typ])

    def struct_reporter_inputs(
            obj: typing.Dict[str, typing.Any],
            _: typing.Type[reporters_preparer.Base_reporter_input]
            ) -> reporters_preparer.Base_reporter_input:
        typ = obj.get("type")
        return converter.structure(obj, reporter_inputs[typ])

    def struct_completion_criteria(
            obj: typing.Dict[str, typing.Any],
            _: typing.Type[compl_crit_struct.Base_completion_criteria]
            ) -> compl_crit_struct.Base_completion_criteria:
        typ = obj.get("type")
        return converter.structure(obj, compl_crit_inputs[typ])

    def struct_anchor_schemes(
            obj: typing.Dict[str, typing.Any],
            _: typing.Type[anchor_schemes_base.Base_anchor_scheme]
            ) -> anchor_schemes_base.Base_anchor_scheme:
        typ = obj.get("type")
        return converter.structure(obj, anchor_schemes[typ])

    def struct_cv_inputs(
            obj: typing.Dict[str, typing.Any],
            _: typing.Type[cv_base.Base_CV_input]
            ) -> cv_base.Base_CV_input:
        typ = obj.get("type")
        return converter.structure(obj, cv_inputs[typ])

    def struct_engine_inputs(
            obj: typing.Dict[str, typing.Any],
            _: typing.Type[engines_structures.Base_engine_settings_input]
            ) -> engines_structures.Base_engine_settings_input:
        typ = obj.get("type")
        return converter.structure(obj, engine_inputs[typ])

    def struct_stage_inputs(
            obj: typing.Dict[str, typing.Any],
            _: typing.Type[scales_base.Base_input_stage]
            ) -> scales_base.Base_input_stage:
        typ = obj.get("type")
        return converter.structure(obj, stage_inputs[typ])

    with open(input_filename, "r") as file:
        json_string = json.load(file)
    
    for plugin_name in json_string['plugins']:
        engines_module_str = base.get_plugin_str(
            plugin_name, submodule_name="engines", require_preparer=True,
            require_openmm_interface=False) # require_runner=True
        if engines_module_str is not None:
            engines_module_prep = importlib.import_module(engines_module_str+".preparer")
            engine_inputs.update(engines_module_prep.engine_inputs) #!!!

        restraints_module_str = base.get_plugin_str(
            plugin_name, submodule_name="restraints", require_preparer=True)
        if restraints_module_str is not None:
            restraints_module_prep = importlib.import_module(restraints_module_str+".preparer")
            restraint_inputs.update(restraints_module_prep.restraint_inputs)

        reporters_module_str = base.get_plugin_str(
            plugin_name, submodule_name="reporters", require_preparer=True)
        if reporters_module_str is not None:
            reporters_module_prep = importlib.import_module(reporters_module_str+".preparer")
            reporter_inputs.update(reporters_module_prep.reporter_inputs)

        compl_crit_module_str = base.get_plugin_str(
            plugin_name, submodule_name="completion_criteria")
        if compl_crit_module_str is not None:
            compl_crit_module_struct = importlib.import_module(compl_crit_module_str+".structures")
            compl_crit_inputs.update(compl_crit_module_struct.completion_criteria)

        cv_module_str = base.get_plugin_str(
            plugin_name, submodule_name="collective_variables", require_preparer=True)
        if cv_module_str is not None:
            cv_module_prep = importlib.import_module(cv_module_str+".preparer")
            cv_inputs.update(cv_module_prep.cv_inputs)

        anchor_schemes_module_str = base.get_plugin_str(
            plugin_name, submodule_name="anchor_schemes", require_preparer=True,
            require_openmm_interface=False, require_structures=False)
        if anchor_schemes_module_str is not None:
            anchor_schemes_module_prep = importlib.import_module(anchor_schemes_module_str+".preparer")
            anchor_schemes.update(anchor_schemes_module_prep.anchor_schemes)

        sampl_meth_module_str = base.get_plugin_str(
            plugin_name, submodule_name="sampling_methods", require_preparer=True)
        if sampl_meth_module_str is not None:
            sampl_meth_module_prep = importlib.import_module(sampl_meth_module_str+".preparer")
            #sampl_meth_module_struct = importlib.import_module(sampl_meth_module_str+".structures")
            for scale_type in scales_preparer.get_sampling_methods_input():
                if scale_type in sampl_meth_module_prep.sampling_methods:
                    scales_preparer.get_sampling_methods_input()[scale_type].update(
                        sampl_meth_module_prep.sampling_methods[scale_type]
                    )

        scales_module_str = base.get_plugin_str(
            plugin_name, submodule_name="scales", require_preparer=True,
            require_openmm_interface=False) # require_runner=True
        if scales_module_str is not None:
            scales_module_prep = importlib.import_module(scales_module_str+".preparer")
            stage_inputs.update(scales_module_prep.stage_inputs)
            scales_module_prep.register_structure_hooks(converter)
           
    cattrs.register_structure_hook(base.Positions_type, 
                                      structures.structure_positions_type_union)
    converter.register_structure_hook(base.Positions_type, 
                                      structures.structure_positions_type_union)
    converter.register_structure_hook(base.Float_or_list_type, 
                                      structures.structure_float_or_list_type_union)
    converter.register_structure_hook(base.Int_or_list_type, 
                                      structures.structure_int_or_list_type_union)
    converter.register_structure_hook_func(
        lambda t: t is restraints_preparer.Base_restraint_input,
        struct_restraint_inputs)
    converter.register_structure_hook_func(
        lambda t: t is reporters_preparer.Base_reporter_input,
        struct_reporter_inputs)
    converter.register_structure_hook_func(
        lambda t: t is compl_crit_struct.Base_completion_criteria,
        struct_completion_criteria)
    
    converter.register_structure_hook_func(
        lambda t: t is cv_base.Base_CV_input,
        struct_cv_inputs)
    converter.register_structure_hook_func(
        lambda t: t is anchor_schemes_base.Base_anchor_scheme,
        struct_anchor_schemes)
    converter.register_structure_hook_func(
        lambda t: t is engines_structures.Base_engine_settings_input,
        struct_engine_inputs)
    converter.register_structure_hook_func(
        lambda t: t is scales_base.Base_input_stage,
        struct_stage_inputs)
    scales_preparer.register_structure_hooks(converter)
    seekr_input = converter.structure(json_string, Seekr_input)

    return seekr_input

def create_cvs(
        seekr_input: Seekr_input, 
        root_directory: str
        ) -> list[cv_base.Base_CV]:
    """
    Create the collective variable objects for the model.
    """
    cvs = []
    cv_counter = 0
    for partition in seekr_input.partitions:
        assert len(partition.cvs) > 0, \
            "Collective variables must be defined if a partition exists."
        for cv_input in partition.cvs:
            # TODO: remove - or put into partition for checking
            #cv_input.check()
            cv: cv_base.Base_CV \
                = cv_input.make_cv_object(index=cv_counter, 
                                          root_directory=root_directory)
            cvs.append(cv)
            cv_counter += 1

    return cvs

def make_state_points(
        seekr_input: Seekr_input,
        num_model_cvs: int
        ) -> list[base.State_point]:
    """
    Create the State_point objects for the Model from the input partitions.
    This is necessary because partitions will, in general, have a different
    number of CVs than the model. This means that, for the model, the "value"
    attribute needs to be filled out with None values for the CVs that are
    in other partitions.
    """
    state_points: list[base.State_point] = []
    for partition in seekr_input.partitions:
        start_cv_index = partition.cvs[0]._index
        for input_state_point in partition.state_points:
            input_value_entries = input_state_point.value
            new_value: list[float | None] = []
            counter: int = 0
            # Need to fill out model state points to have the right size of
            #  value lists in each object.
            for _ in range(start_cv_index):
                new_value.append(None)
                counter += 1
            for input_value_entry in input_value_entries:
                new_value.append(input_value_entry)
                counter += 1
            for _ in range(counter, num_model_cvs):
                new_value.append(None)
            state_point = base.State_point(
                name=input_state_point.name,
                value=new_value,
                end_state=input_state_point.end_state)
            state_points.append(state_point)
    
    return state_points

def create_anchors(
        seekr_input: Seekr_input
        ) -> tuple[list[structures.Anchor], dict, dict, list, int]:
    """
    Create the Anchor objects for the Model from the input partitions.
    
    Parameters:
    -----------
    model : Model()
        The model object representing the entire calculation.
    
    """
    anchor_index: int = 0
    anchors: list[structures.Anchor] = []
    connection_flag_dict: dict[int|str, list[structures.Anchor]] = defaultdict(list)
    associated_input_anchor: dict[int, typing.Any] = {}
    anchor_connections: list[tuple[int, int]] = []
    anchor_index = 0
    milestone_index = 0
    for partition in seekr_input.partitions:
        if partition.anchor_scheme is not None:
            partition_anchors, partition_anchor_connections, milestone_index\
                = partition.make_anchors(
                    anchor_index, milestone_index, connection_flag_dict, 
                    associated_input_anchor)
            anchors += partition_anchors
            anchor_index += len(partition_anchors)
            anchor_connections += partition_anchor_connections

    return anchors, connection_flag_dict, associated_input_anchor, \
        anchor_connections, milestone_index

def assign_all_state_points(
        state_points: list[base.State_point], 
        model: structures.Seekr_model
        ) -> None:
    """
    Given a State_point object, find which anchor it belongs in, and
    assign that anchor's name and end_state accordingly.
    """
    for state_point in state_points:
        already_found_anchor = False
        for anchor in model.anchors:
            if len(anchor.milestones) == 0:
                continue
            
            if anchor.is_point_within(state_point.value):
                assert not already_found_anchor, \
                    f"State point named {state_point.name} found in multiple anchors."
                
                anchor.name = state_point.name
                if state_point.end_state:
                    anchor.end_anchor = True
                already_found_anchor = True

        if not already_found_anchor:
            raise Exception(
                f"Cannot assign state point named {state_point.name}: no suitable "\
                "anchors found. The point might be located on a boundary.")
    return

def prepare_model_cvs_and_anchors(
        model: structures.Seekr_model, 
        seekr_input: Seekr_input, 
        force_overwrite: bool
        ) -> None:
    """
    Fill out the CV and anchor items within the model based on the 
    input objects.
    """
    cvs = create_cvs(seekr_input, seekr_input.directory)
    model.collective_variables = cvs
    anchors, connection_flag_dict, associated_input_anchor, \
        anchor_connections, milestone_index = create_anchors(seekr_input)
    
    for anchor in anchors:
        model.anchors.append(anchor)
    
    # TODO: more thoroughly test make_state_points, because once
    #  it incorrectly assigned the name of a Voronoi anchor in a 2D
    #  tesselation.
    state_points = make_state_points(seekr_input, len(cvs))
    # NOTE: this must be done BEFORE resolve_connections() because
    #  it calls modify_model() which requires the BD anchors to be
    #  assigned.
    
    # assign connections between anchors
    anchors, index_reference = resolve_connections(
        connection_flag_dict, model, associated_input_anchor, 
        anchor_connections)
    resolve_modifications_and_filetree(
        model, anchors, index_reference, associated_input_anchor, 
        seekr_input.directory, force_overwrite)
    
    # Check for repeat alias indices among the milestones in the anchor.
    for anchor in anchors:
        alias_indices = set()
        for milestone in anchor.milestones:
            assert milestone.alias_index not in alias_indices, \
                "Repeated alias indices detected in anchor {}".format(
                    anchor.index)
            alias_indices.add(milestone.alias_index)
            
    model.num_anchors = len(anchors)
    model.num_milestones = milestone_index
    assign_all_state_points(state_points, model)
    return

def cleanse_outputs_all(
        model: "structures.Seekr_model",
        anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings \
            | structures.Anchor,
        swarm_id: int = 0
        ) -> None:
    """
    Delete all simulation outputs from unpartitioned_settings or anchor.
    """
    for stage in model.stages:
        if stage.type not in ("md_stage", "bd_stage"):
            continue
        cleanse_outputs_stage(model, anchor_or_unpartitioned, stage.index-1, swarm_id)
    return

def cleanse_outputs_stage(
        model: "structures.Seekr_model",
        anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings \
            | structures.Anchor, 
        stage_index: int,
        swarm_id: int
        ) -> None:
    """
    Delete all simulation outputs from unpartitioned_settings or anchor.
    """
    stage = model.stages[stage_index]
    swarm_string = f"*swarm_{swarm_id}"
    # Replace with methods in the scales
    if stage.type == "md_stage":
        if stage.scope == "unpartitioned":
            full_directory = os.path.join(
                model.directory, anchor_or_unpartitioned.directory, 
                stage.get_directory_name())
        elif stage.scope == "partitioned":
            full_directory = os.path.join(
                model.directory, "anchor_sets", stage.set_name, 
                anchor_or_unpartitioned.directory, stage.get_directory_name())
        else:
            raise Exception(f"Unknown stage scope: {stage.scope}")
        
        if os.path.exists(full_directory):
            file_globs = stage.sampling_method.get_output_positions_glob_list(swarm_id)
            for file_glob in file_globs:
                full_glob = os.path.join(full_directory, file_glob)
                for file in glob.glob(full_glob):
                    #print("removing file:", file)
                    os.remove(file)
                    
            dcd_glob = os.path.join(full_directory, f"{swarm_string}*.dcd")
            for dcd_file in glob.glob(dcd_glob):
                #print("removing file:", dcd_file)
                os.remove(dcd_file)
            pdb_glob = os.path.join(full_directory, f"{swarm_string}*.pdb")
            for pdb_file in glob.glob(pdb_glob):
                #print("removing file:", pdb_file)
                os.remove(pdb_file)
            state_glob = os.path.join(full_directory, f"{swarm_string}*.state")
            for state_file in glob.glob(state_glob):
                #print("removing file:", state_file)
                os.remove(state_file)
            json_glob = os.path.join(full_directory, f"{swarm_string}*.json")
            for json_file in glob.glob(json_glob):
                #print("removing file:", json_file)
                os.remove(json_file)
            out_glob = os.path.join(full_directory, f"{swarm_string}.out")
            for out_file in glob.glob(out_glob):
                #print("removing file:", out_file)
                os.remove(out_file)
            #log_glob = os.path.join(full_directory, "*.log")
            #for log_file in glob.glob(log_glob):
            #    os.remove(log_file)
            backup_info_glob = os.path.join(full_directory, 
                engines_base.INFORMATION_CHECKPOINT_FILENAME_NOEXT \
                + f"{swarm_id}" + engines_base.INFORMATION_CHECKPOINT_FILENAME_EXT)
            for backup_info_file in glob.glob(backup_info_glob):
                #print("removing file:", backup_info_file)
                os.remove(backup_info_file)
            backup_glob = os.path.join(full_directory, 
                engines_base.RESTART_CHECKPOINT_FILENAME_NOEXT \
                + f"{swarm_id}" + engines_base.RESTART_CHECKPOINT_FILENAME_EXT)
            for backup_file in glob.glob(backup_glob):
                #print("removing file:", backup_file)
                os.remove(backup_file)
            lock_filename = os.path.join(full_directory, "LOCK")
            if os.path.exists(lock_filename):
                #print("removing file:", lock_filename)
                os.remove(lock_filename)
    elif stage.type == "bd_stage":
        bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
        assert bd_engine is not None
        if anchor_or_unpartitioned.directory == bd_engine.b_surface_directory:
            full_directory = os.path.join(
                model.directory, bd_engine.b_surface_directory)
            if os.path.exists(full_directory):
                output_files_glob = os.path.join(
                    full_directory, bd_engine.results_glob)
                output_files_list = glob.glob(output_files_glob)
                for output_file in output_files_list:
                    os.remove(output_file)
                
                simulation_file_glob = os.path.join(
                    full_directory, "*_simulation.xml")
                for simulation_file in glob.glob(simulation_file_glob):
                    os.remove(simulation_file)
                
                traj_file_glob = os.path.join(full_directory, "traj*.xml")
                for traj_file in glob.glob(traj_file_glob):
                    os.remove(traj_file)
    else:
        raise Exception(f"Unknown stage type: {stage.type}")
    
    for plugin in model.plugins:
        plugin_module_str = base.get_plugin_str(
            plugin, submodule_name="", require_structures=False, require_preparer=True,
            require_runner=False, require_openmm_interface=False)
        if plugin_module_str is not None:
            plugin_module_prep = importlib.import_module(plugin_module_str+".preparer")
            plugin_module_prep.cleanse_outputs_stage(model, anchor_or_unpartitioned, stage_index)
    return

def modify_model(
        old_model: structures.Seekr_model, 
        new_model: structures.Seekr_model, 
        root_directory: str, 
        force_overwrite: bool = False) -> None: #-> list[int]:
    """
    If someone runs the prepare stage on a model with existing
    directories and simulation results, examine the changes to delete
    and rename directories properly, and only overwrite simulation files
    if forced to do so.
    """
    old_model.directory = root_directory
    
    #######################################
    # 1. check Stages
    #######################################
    # Look through the new and old stages and pair them up by name
    old_stages_to_delete: list[int] = []
    new_stages_to_create: list[int] = []
    stage_pairs: list[tuple[int, int]] = []
    stage_i: structures.Stage
    stage_j: structures.Stage
    for i, stage_i in enumerate(new_model.stages):
        i_paired: bool = False
        for j, stage_j in enumerate(old_model.stages):
            if stage_i.name == stage_j.name:
                pair = (i, j)
                stage_pairs.append(pair)
                i_paired = True
                break
        if not i_paired:
            new_stages_to_create.append(i)
        
    for j, stage_j in enumerate(old_model.stages):
        j_paired = False
        for pair in stage_pairs:
            if j == pair[1]:
                j_paired = True
                break
        if not j_paired:
            old_stages_to_delete.append(j)
    
    pair_indices_where_stages_differed: set[tuple] = set()

    for i, j in stage_pairs:
        new_stage = new_model.stages[i]
        old_stage = old_model.stages[j]
        stages_the_same = True
        
        
        # NOTE: not including completion_criteria, restraints, or reporters 
        # When determining whether stages are the same, because in these cases
        # later already-run stages might not be significantly affected. We will
        # only mark a stage as stages_the_same=False if there's no chance that the
        # output files could be retained meaningfully, such that any child stages
        # would also need to be redone.
        

        if old_stage.type != new_stage.type:
            stages_the_same = False
        if old_stage.scale_type != new_stage.scale_type:
            stages_the_same = False
        if old_stage.input_stage_index != new_stage.input_stage_index:
            stages_the_same = False
        if old_stage.scope != new_stage.scope:
            stages_the_same = False
        if new_stage.type in ("md_stage", "bd_stage"):
            if old_stage.sampling_method.type != new_stage.sampling_method.type:
                stages_the_same = False
            if old_stage.set_name != new_stage.set_name:
                stages_the_same = False
        elif new_stage.is_logistic_stage():
            if old_stage.get_output_set_name() != new_stage.get_output_set_name():
                stages_the_same = False

        if not stages_the_same:
            pair_indices_where_stages_differed.add((i,j))
            old_stages_to_delete.append(j)
            new_stages_to_create.append(i)

        # All child stages of a stage that has changed will also be considered 
        # to have changed.
        input_stage_pair = (new_stage.input_stage_index, 
                            old_stage.input_stage_index)
        if input_stage_pair in pair_indices_where_stages_differed:
            pair_indices_where_stages_differed.add((i,j))
            old_stages_to_delete.append(j)
            new_stages_to_create.append(i)

        # Now check if 

    old_stages_to_delete.sort()
    new_stages_to_create.sort()

    stages_to_delete_indices_with_files: list[int] = []
    stages_to_delete_names_with_files: list[str] = []
    for i in old_stages_to_delete:
        old_model_stage = old_model.stages[i]
        if old_model_stage.has_files(old_model):
            stages_to_delete_indices_with_files.append(old_model_stage.index)
            stages_to_delete_names_with_files.append(old_model_stage.name)
    
    #######################################
    # 2. check Anchors
    #######################################
    # NOTE: assumed that anchor set "main" is loaded.
    # Look through the new and old anchors and pair them up by the value
    #  of their variables
    old_anchors_to_delete: list[int] = []
    new_anchors_to_create: list[int] = []
    anchor_pairs: list[tuple[int, int]] = []
    anchor_alpha: structures.Anchor
    anchor_beta: structures.Anchor
    for alpha, anchor_alpha in enumerate(new_model.anchors):
        alpha_paired: bool = False
        for beta, anchor_beta in enumerate(old_model.anchors):
            if anchor_alpha.variables == anchor_beta.variables:
                pair = (alpha, beta)
                anchor_pairs.append(pair)
                alpha_paired = True
                break
        if not alpha_paired:
            new_anchors_to_create.append(alpha)
        
    for beta, anchor_beta in enumerate(old_model.anchors):
        beta_paired = False
        for pair in anchor_pairs:
            if beta == pair[1]:
                beta_paired = True
                break
        if not beta_paired:
            old_anchors_to_delete.append(beta)
        
    # Now check all the paired anchors to see if anyone's milestones
    # have changed
    old_anchors_with_changed_milestones: list[int] = []
    for pair in anchor_pairs:
        (alpha, beta) = pair
        anchor_alpha = new_model.anchors[alpha]
        anchor_beta = old_model.anchors[beta]
        milestones_changed = False
        milestone_pairs = []
        for i, milestone_alpha in enumerate(anchor_alpha.milestones):
            milestone_alpha_paired = False
            for j, milestone_beta in enumerate(anchor_beta.milestones):
                if milestone_alpha.variables == milestone_beta.variables:
                    milestone_pairs.append((i,j))
                    milestone_alpha_paired = True
                    break
            if not milestone_alpha_paired:
                milestones_changed = True
        for j, milestone_beta in enumerate(anchor_beta.milestones):
            milestone_beta_paired = False
            for milestone_pair in milestone_pairs:
                if j == milestone_pair[1]:
                    milestone_beta_paired = True
                    break
            if not milestone_beta_paired:
                milestones_changed = True
        if milestones_changed:
            old_anchors_with_changed_milestones.append(beta)
    
    #######################################
    # 3. display warnings
    #######################################
    # NOTE: no need to worry about unpartitioned_settings because there are no 
    # values or milestones.
    warning_msg = ""

    # check whether deleting anchor directories contain simulation files
    deleting_anchor_directories_with_files = []
    for del_anchor_index in old_anchors_to_delete:
        del_anchor: structures.Anchor = old_model.anchors[del_anchor_index]
        if del_anchor.has_files(old_model, root_directory):
            deleting_anchor_directories_with_files.append(del_anchor.directory)
    
    if len(deleting_anchor_directories_with_files) > 0 :
        warning_msg+=f"The anchor(s) {deleting_anchor_directories_with_files} "\
            +"already have existing output files and the entered command "\
            "would overwrite them. If you desire to overwrite the existing "\
            "files, then use the --force_overwrite (-f) option, and these "\
            f"anchors will be deleted in directory: {root_directory}\n\n"
    
    # Check whether directories with changed milestones already contain
    # simulation files
    cleansing_anchors: list[structures.Anchor] = []
    cleansing_directories: list[str] = []
    for cleanse_anchor_index in old_anchors_with_changed_milestones:
        cleanse_anchor: structures.Anchor \
            = old_model.anchors[cleanse_anchor_index]
        if cleanse_anchor.has_files(old_model, root_directory):
            cleansing_anchors.append(cleanse_anchor)
            cleansing_directories.append(cleanse_anchor.directory)
    
    if len(cleansing_directories) > 0 :
        warning_msg+=f"The anchor(s) {cleansing_directories} "\
            +"already have existing output files yet the milestone locations "\
            "will be changed if this command proceeds. If you desire "\
            "to overwrite the existing files, then use the "\
            "--force_overwrite (-f) option, and these outputs will be "\
            f"deleted in directory: {root_directory}\n\n"
    
    if len(stages_to_delete_indices_with_files) > 0:
        warning_msg+=f"The stage(s) with indices {stages_to_delete_indices_with_files} "\
            +f"(and names {stages_to_delete_names_with_files}) "\
            +"already have existing output files and the entered command "\
            "would overwrite them. If you desire to overwrite the existing "\
            "files, then use the --force_overwrite (-f) option, and these "\
            f"anchors will be deleted in directory: {root_directory}\n\n"

    if warning_msg and not force_overwrite:
        print(warning_msg)
        raise Exception("Cannot overwrite existing files.")
    
    #############################################
    # 4. File/directory modifications/deletions
    #############################################

    for del_anchor_index in old_anchors_to_delete:
        del_anchor = old_model.anchors[del_anchor_index]
        full_path = os.path.join(
            root_directory, "anchor_sets", "main", del_anchor.directory)
        print("removing directory: "+del_anchor.directory)
        shutil.rmtree(full_path, ignore_errors=True)
    
    for cleansing_anchor in cleansing_anchors:
        print("removing output files from anchor:", cleansing_anchor.name)
        cleanse_outputs_all(old_model, cleansing_anchor)
    
    for stage_index in stages_to_delete_indices_with_files:
        stage = old_model.stages[stage_index]
        print("removing output files from stage:", stage.name)
        # TODO: method doesn't exist: fix this
        stage.remove_directory(old_model)
        if stage.is_logistic_stage():
            output_set_name = stage.get_output_set_name()
            if output_set_name != "main":
                # We want "main" excluded from deletion because the
                # new model might redefine it.
                stage.delete_output_set(output_set_name)

    # Now that the equivalent anchors are known, we know which ones will
    # be deleted, and which ones are being changed, with or without
    # simulation data, 
    
    for pair in anchor_pairs:
        (alpha, beta) = pair
        anchor1 = old_model.anchors[beta]
        anchor2 = new_model.anchors[alpha]
        full_path1 = os.path.join(
            root_directory, "anchor_sets", "main", anchor1.directory)
        full_path2 = os.path.join(
            root_directory, "anchor_sets", "main", anchor2.directory)
        if full_path1 != full_path2:
            if os.path.exists(full_path2):
                # then we cannot rename to this without giving this one
                # a temporary directory
                for problem_anchor in old_model.anchors:
                    if problem_anchor.directory == "":
                        continue
                    if problem_anchor.directory == anchor2.directory:
                        temporary_name = problem_anchor.directory+"temp"
                        temp_full_path1 = os.path.join(
                            root_directory, "anchor_sets", "main", problem_anchor.directory)
                        temp_full_path2 = os.path.join(
                            root_directory, "anchor_sets", "main", temporary_name)
                        print("moving directory {} to {}".format(
                            problem_anchor.directory, temporary_name))
                        os.rename(temp_full_path1, temp_full_path2)
                        problem_anchor.directory = temporary_name
                        break
            
            print("moving directory {} to {}".format(anchor1.directory, 
                                                 anchor2.directory))
            # If it gets here, it's because a new directory was made that
            # has no data in it. It can be safely deleted
            if os.path.exists(full_path2):
                shutil.rmtree(full_path2)
                
            os.rename(full_path1, full_path2)
        
        # Do not do this, because this information will be calculated anew,
        # even if existing structures are kept.
        #anchor2.stage_parameters = anchor1.stage_parameters
        
    
    # TODO: ensure that the new model has the same plugins as the old
    # as well as the same engines, scales, etc.
    # TODO: return something else here - there are no more new anchors
    # with starting PDBs to keep.
    #return new_anchors_with_starting_pdbs_to_keep
    return

def resolve_modifications_and_filetree(
        model: structures.Seekr_model, 
        anchors: list[structures.Anchor], 
        index_reference: dict[int, int],
        associated_input_anchor: dict[int, typing.Any],
        root_directory: str, 
        force_overwrite: bool
        ) -> None:
    """
    Resolve modifications to the model and the filetree.
    """
    json_path = os.path.join(root_directory, "model.json")
    if os.path.exists(json_path):
        # then a model file already exists at this location: update
        # the anchor directories.
        old_model = structures.load_model(json_path)
        old_model.load_cvs("main")
        old_model.load_selections("main")
        old_model.load_anchor_set("main")
        logging.info("Model file already existed at this location. "\
                     "Modified the existing model.")
        model.set_random_seeds(old_model.get_random_seed())
        modify_model(old_model, model, root_directory, force_overwrite)
    
    # NOTE: This is the name of the anchor set when a anchor_stage is activated. 
    #  This value can be changed by a logistic stage
    on_set_name = "main"
    for stage in model.stages:
        # TODO: transitions between unpartitioned and partitioned stages
        #  via a logistic stage.
        #if stage.scale_type in ["molecular_dynamics", "logistic"]:
        # BEGIN INDENT (remove)
        if stage.scope == "partitioned":
            # NOTE: mypy error is handled with "md_stage" type check above
            stage.add_stage_parameters_to_anchors(model)
            
        elif stage.scope == "unpartitioned":
            stage.add_stage_parameters_to_unpartitioned_settings(model)
        
        # END INDENT (remove)
                
        else:
            pass
            
        if stage.is_logistic_stage():
            on_set_name = stage.get_output_set_name()
        else:
            stage.set_name = on_set_name
        
    for old_index in index_reference:
        new_index = index_reference[old_index]
        if new_index < len(anchors):
            input_anchor = associated_input_anchor[old_index]
            anchors[new_index].generate_filetree(model, root_directory, "main")
            input_anchor.copy_initial_files(anchors[new_index], root_directory, 
                                                "main", model.using_toy())
    return

def resolve_connections(
        connection_flag_dict: dict, 
        model: structures.Seekr_model, 
        associated_input_anchor: dict[int, typing.Any],
        anchor_connections: list[tuple[int, int]]
        ) -> tuple[list[structures.Anchor], dict[int, int]]:
    """
    Given connection information between anchors, merge anchors
    that are connected.
    """
    anchors: list[structures.Anchor] = model.anchors
    anchor_indices_to_remove = []
    index_reference = {}
    bulk_anchors: list[structures.Anchor] = []
    for anchor in anchors:
        index_reference[anchor.index] = anchor.index
    
    for key in connection_flag_dict:
        anchor_connection_list: list[structures.Anchor] \
            = connection_flag_dict[key]
        #bulk_discarded_indices = []
        if key == "bulk":
            bulk_anchors.append(anchor_connection_list[0])
            #for anchor_discarded in anchor_connection_list:
                #anchor_indices_to_remove.append(anchor_discarded.index)
                #bulk_discarded_indices.append(anchor_discarded.index)
                
        else:
            assert len(anchor_connection_list) > 1, \
                f"Connection flag {key} needs to have endpoints "\
                +"in at least two anchors."
        
        anchor_kept = anchor_connection_list[0]
        for anchor_discarded in anchor_connection_list[1:]:
            input_anchor_kept = associated_input_anchor[anchor_kept.index]
            input_anchor_discarded = associated_input_anchor[anchor_discarded.index]
            assert anchor_kept.end_anchor == anchor_discarded.end_anchor, \
                "The anchors connected by connection flag {} ".format(key) \
                +"must have the same value for 'end_anchor'."
            if anchor_kept.name == "bulk":
                assert anchor_kept.name == anchor_discarded.name, \
                    "The anchors connected by connection flag {} ".format(key) \
                    +"must have the same name 'bulk'."   
            # Ensure that both anchors don't have starting structures 
            # defined
            if (input_anchor_kept.positions is None) \
                    or (input_anchor_kept.positions == ""):
                input_anchor_kept.positions \
                    = input_anchor_discarded.positions
            
            if (input_anchor_discarded.positions is None) \
                    or (input_anchor_discarded.positions == ""):
                input_anchor_discarded.positions \
                    = input_anchor_kept.positions
            assert input_anchor_kept.positions \
                    == input_anchor_discarded.positions, \
                "The anchors connected by connection flag {} ".format(key) \
                +"have inconsistent positions defined."
            assert input_anchor_kept.same_scale_parameters(
                input_anchor_discarded), \
                "The anchors connected by connection flag {} ".format(key) \
                +"have inconsistent scale parameters defined."
            anchor_indices_to_remove.append(anchor_discarded.index)
            anchor_kept.milestones += anchor_discarded.milestones
            anchor_kept.variables.update(anchor_discarded.variables)
            index_reference[anchor_discarded.index] = anchor_kept.index
            
    # sort and remove all redundant anchors
    anchor_indices_to_remove = sorted(set(anchor_indices_to_remove), 
                                      reverse=True)
    for anchor_index_to_remove in anchor_indices_to_remove:
        if len(anchors) > anchor_index_to_remove+1:
            for lower_anchor in anchors[anchor_index_to_remove+1:]:
                index_reference[lower_anchor.index] -= 1
                
    for anchor_index_to_remove in anchor_indices_to_remove:
        anchors.pop(anchor_index_to_remove)

    visited_new_indices = []
    
    for old_index in index_reference:
        new_index = index_reference[old_index]
        if new_index < len(anchors):
            anchors[new_index].index = new_index
            anchors[new_index].directory = f"anchor_{new_index:04d}"
            if anchors[new_index].name.startswith("anchor_"):
                anchors[new_index].name = anchors[new_index].directory
            
            for pair in anchor_connections:
                pair_tuple = set(pair)
                if old_index in pair:
                    pair_tuple.remove(old_index)
                    old_other_index = pair_tuple.pop()
                    new_other_index = index_reference[old_other_index]
                    assert new_other_index != new_index, "An anchor cannot "\
                        "be a neighbor to itself."
                    anchors[new_index].neighbor_anchor_indices.append(new_other_index)

        visited_new_indices.append(new_index)
    
    # TODO: check the scale to ensure that a lack of bulk anchors is alright
    if len(bulk_anchors) == 0:
        assert not model.using_scale("brownian_dynamics"), \
            "A bulk anchor must be defined if using BD. This can be done by "\
            "naming an input_anchor with the name 'bulk', or by setting the "\
            "'bulk_anchor' flag to True."
    else:
        for bulk_anchor in bulk_anchors:
            bulk_anchor = anchors[bulk_anchor.index]
            if model.using_scale("brownian_dynamics"):
                # Will throw error if BD engine not existent
                bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
                bulk_anchor.scale_types.append("brownian_dynamics")
                bulk_anchor.directory = bd_engine.b_surface_directory
                for neighbor_index in bulk_anchor.neighbor_anchor_indices:
                    neighbor_anchor = anchors[neighbor_index]
                    neighbor_anchor.scale_types.append("brownian_dynamics")

    return anchors, index_reference