"""
prepare_base.py

Routines for preparing the model file given input that contains:
 - physical settings (temperature, pressure, etc.)
 - simulation engine specific settings (OpenMM, NAMD, Browndye, SDA, etc.)
 - SEEKR calculation type (MMVT, other milestoning, etc.)
 - HIDR calculation type (MetaD, steered MD, RAMD, scaled MD, etc.)
 - CV and Anchor information)

"""

import os
import glob
import json
from collections import defaultdict
import typing
import shutil
import logging

from attrs import define, field, validators, Factory
from cattrs import Converter
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import QhullError

import seekr.modules.base as base
import seekr.modules.data_structures as data_structures
import seekr.modules.collective_variables.data_structures as cv_data_struct
import seekr.modules.collective_variables.input as cv_input
# TODO: these don't need to be imported here, and in the future,
#  they will be imported from a generic input.py file in the 
#  engines directory.
import seekr.modules.engines.base as engines_base
import seekr.modules.engines.browndye.base as browndye_base
import seekr.modules.engines.data_structures as engines_data_struct
import seekr.modules.engines.input as engines_input
#import seekr.modules.engines.openmm.input as openmm_input
#import seekr.modules.engines.browndye.input as browndye_input
import seekr.modules.sampling_methods.input as sampling_methods_input

KEEP_ANCHOR_RE = "hidr*|string*"

# TODO: modify this function to include multiple stages
# TODO: move to CV module?
def anchor_has_files(model, anchor):
    """
    Returns True if seekr simulations have already been run in this anchor.
    """
    
    output_directory = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.production_directory)
    output_files_glob = os.path.join(output_directory, anchor.md_output_glob)
    output_restarts_list = glob.glob(output_files_glob)
    if len(output_restarts_list) > 0:
        return True
    else:
        return False

# TODO: Combine with function above to handle multiple stages
def anchor_has_hidr_files(model, anchor):
    """
    Returns True if HIDR has already been run in this anchor.
    """
    
    building_directory = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.building_directory)
    input_files_glob = os.path.join(building_directory, "hidr*")
    input_files_list = glob.glob(input_files_glob)
    if len(input_files_list) > 0:
        return True
    else:
        return False

# TODO: modify this function to include multiple stages
"""
def cleanse_anchor_outputs(model, anchor):
    ""
    Delete all simulation outputs for an existing anchor to make way
    for new outputs.
    ""
    
    if model.openmm_settings is not None:
        import seekr2.modules.runner_openmm as runner_openmm
        runner_openmm.cleanse_anchor_outputs(model, anchor)
    elif model.namd_settings is not None:
        import seekr2.modules.runner_namd as runner_namd
        runner_namd.cleanse_anchor_outputs(model, anchor)
    return
"""

@define
class Input_stage_md:
    """
    A MD stage of the SEEKR calculation, whether that is an initial
    equilibration, a sampling method to obtain starting structures,
    a settling stage, or a production stage for obtaining kinetics.
    """
    type: typing.Literal["md_stage"] = "md_stage"
    name: str = field(default="equilibration", 
                      validator=validators.instance_of(str))
    run_minimization: bool = field(default=False,
                                      validator=validators.instance_of(bool))
    output_interval: int = field(default=1000,
                                  validator=[validators.instance_of(int),
                                             validators.gt(0)])
    ensemble: data_structures.Ensemble = field(default=Factory(
        data_structures.NVT_ensemble))
    sampling_method: sampling_methods_input.MD_sampling_method_input = field(
        default=Factory(sampling_methods_input.Conventional_md_input))
    completion_criteria: data_structures.Completion_criteria | None = None
    
    def make_model_stage(
            self, 
            index: int,
            systemwide_stage_input: bool
            ) -> typing.Tuple[data_structures.Stage_md, bool]:
        """
        Given this input stage object, create the model's Stage object.
        """
        stage = data_structures.Stage_md()
        stage.index = index
        stage.name = self.name
        stage.run_minimization = self.run_minimization
        stage.md_state_output_interval = self.output_interval
        stage.md_trajectory_output_interval = self.output_interval
        stage.md_restart_checkpoint_interval = self.output_interval
        stage.ensemble = self.ensemble
        stage.sampling_method = self.sampling_method.make_model_sampling_method()
        stage.completion_criteria = self.completion_criteria
        if systemwide_stage_input: # If coming from systemwide settings
            assert not stage.sampling_method.needs_to_be_in_anchor(), \
                f"Stage '{stage.name}' with sampling method '{stage.sampling_method.type}' "\
                "must be run within one or more anchors. A stage that populates anchors "\
                "must precede this stage."
            stage.input_from_systemwide_settings = True
            if stage.sampling_method.input_from_systemwide_settings_output_to_anchors():
                new_systemwide_stage_input = False
                stage.output_to_systemwide_settings = False
            else:
                new_systemwide_stage_input = True
                stage.output_to_systemwide_settings = True
        else:
            stage.input_from_systemwide_settings = False
            if stage.sampling_method.input_from_anchors_output_to_systemwide_settings():
                new_systemwide_stage_input = True
                stage.output_to_systemwide_settings = True
            else:
                new_systemwide_stage_input = False
                stage.output_to_systemwide_settings = False
        return stage, new_systemwide_stage_input
    

@define
class Input_stage_bd:
    """
    A BD stage of the SEEKR calculation.
    """
    type: typing.Literal["bd_stage"] = "bd_stage"
    name: str = field(default="BD",
                        validator=validators.instance_of(str))
    n_trajectories_per_output: int = field(default=1000,
                                            validator=[validators.instance_of(int),
                                                         validators.gt(0)])
    n_steps_per_output: int = field(default=100000,
                                    validator=[validators.instance_of(int),
                                               validators.gt(0)])
    sampling_method: sampling_methods_input.BD_sampling_method_input = field(
        default=Factory(sampling_methods_input.NAM_bd_input))
    completion_criteria: data_structures.Surface_crossed_completion_criteria = field(
        default=Factory(data_structures.Surface_crossed_completion_criteria))
    
    def make_model_stage(
            self, 
            index: int,
            systemwide_stage_input: bool
            ) -> typing.Tuple[data_structures.Stage_bd, bool]:
        """
        Given this input stage object, create the model's Stage object.
        """
        stage = data_structures.Stage_bd()
        stage.index = index
        stage.name = self.name
        stage.n_trajectories_per_output = self.n_trajectories_per_output
        stage.n_steps_per_output = self.n_steps_per_output
        stage.completion_criteria = self.completion_criteria
        stage.input_from_systemwide_settings = True
        stage.output_to_systemwide_settings = True
        # At present, BD stages will always be systemwide
        return stage, systemwide_stage_input

Input_stage = typing.Union[Input_stage_md, Input_stage_bd]

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
            seekr_model_md_engine : engines_data_struct.MD_engine
            ) -> None:
        """
        Assign the toy settings to the model engine.
        """
        seekr_model_md_engine.toy_settings = engines_base.Toy_settings()
        seekr_model_md_engine.toy_settings.potential_energy_expression = self.potential_energy_expression
        seekr_model_md_engine.toy_settings.num_particles = self.num_particles
        seekr_model_md_engine.toy_settings.masses = self.masses
        return

@define
class State_point:
    """
    An object that can be put into the input file to indicate the location
    of a state within a CV or Combo. The name of the anchor containing this
    point will be changed, and end_state will be set to True.
    """
    name: str = field(default="",
                        validator=validators.instance_of(str))
    value: list[float | None] = field(default=Factory(list),)
                                #validator=validators.deep_iterable(
                                #    member_validator=validators.instance_of(float),
                                #    iterable_validator=validators.instance_of(list),
                                #))
    # whether to set the end_anchor attribute to True in an Anchor
    end_state: bool = field(default=True,
                            validator=validators.instance_of(bool))

def find_voronoi_anchor_neighbors(
        anchors: list[data_structures.Anchor]
        ) -> dict[int, set[int]]:
    """
    Given anchor positions, assign each anchor to a Voronoi cell
    and then return the neighbors of each anchor.

    This is not necessarily all anchors within a model. It could be a
    subset of anchors that are connected by a Voronoi tesselation
    """
    n_points = len(anchors)
    assert n_points > 0, "No anchors in CV"
    n_dimensions = len(anchors[0].variables)
    points = np.zeros((n_points, n_dimensions))
    neighbor_anchors = defaultdict(set)
    for alpha, anchor in enumerate(anchors):
        assert n_dimensions == len(anchor.variables), \
            "all anchors within a multidimensional subset must have the same dimensions."
        variables_sorted_by_cv_index = sorted(
            anchor.variables.keys(), key=lambda name: name.split("_")[1])
        for i, variable_name in enumerate(variables_sorted_by_cv_index):
            points[alpha, i] = anchor.variables[variable_name]
    
    try:
        triangulation = Delaunay(points)
        for simplex in triangulation.simplices:
            for anchor_index1 in simplex:
                for anchor_index2 in simplex:
                    if anchor_index1 == anchor_index2:
                        continue
                    neighbor_anchors[anchor_index1].add(anchor_index2)
                    
    except QhullError:
        # This indicates a Voronoi tesselation where all points are
        # on a single line. In this case, we will just connect
        # each anchor to its nearest neighbors.
        for alpha, anchor in enumerate(anchors):
            if alpha > 0:
                neighbor_anchors[alpha].add(alpha-1)
            if alpha < n_points-1:
                neighbor_anchors[alpha].add(alpha+1)
        
    return neighbor_anchors

def make_milestone_between_two_voronoi_anchors(
            anchor: data_structures.Anchor, 
            neighbor_anchor: data_structures.Anchor,
            milestone_index: int, 
            cv_indices: list[int]) -> int:
        neighbor_index = neighbor_anchor.index
        assert anchor.index != neighbor_anchor.index
        for milestone in anchor.milestones:
            if milestone.neighbor_anchor_index == neighbor_anchor.index:
                return milestone_index
        
        milestone1 = data_structures.Milestone()
        milestone1.index = milestone_index
        milestone1.neighbor_anchor_index = neighbor_index
        milestone1.alias_index = len(anchor.milestones)+1
        milestone1.variables = {"k": 1.0}
        milestone2 = data_structures.Milestone()
        milestone2.index = milestone_index
        milestone2.neighbor_anchor_index = anchor.index
        milestone2.alias_index = len(neighbor_anchor.milestones)+1
        milestone2.variables = {"k": 1.0}
        for i, cv_index in enumerate(cv_indices):
            me_key = f"me_{i}"
            neighbor_key = f"neighbor_{i}"
            variable_key = f"value_{cv_index}"
            me_value = anchor.variables[variable_key]
            neighbor_value = neighbor_anchor.variables[variable_key]
            milestone1.variables[me_key] = me_value
            milestone1.variables[neighbor_key] = neighbor_value
            milestone2.variables[me_key] = neighbor_value
            milestone2.variables[neighbor_key] = me_value
            if (len(cv_indices) == 1): # And is a spherical CV
                avg_value = (me_value + neighbor_value) / 2.0
                milestone1.variables["avg_value"] = avg_value
                milestone2.variables["avg_value"] = avg_value
            
        # don't add the milestone if it already exists here or there
        already_here = False
        for me_milestone in anchor.milestones:
            if me_milestone.neighbor_anchor_index == neighbor_index:
                already_here = True
                assert me_milestone.variables == milestone1.variables
        if not already_here:
            anchor.milestones.append(milestone1)
            for neighbor_milestone in neighbor_anchor.milestones:
                assert neighbor_milestone.neighbor_anchor_index != anchor.index, \
                    f"Neighbor anchor {neighbor_index} already " \
                    +f"has milestone, but anchor {anchor.index} doesn't"
                
            neighbor_anchor.milestones.append(milestone2)
            milestone_index += 1
        else:
            logging.error("found milestone already at anchor:", anchor.index, \
                          "with neighbor:", neighbor_index)
        
        return milestone_index

@define
class Partition_input:
    """
    Partition information concerns how the system is divided into
    partitions - that is, involving functions of the N-D positions.
    This means CVs, anchors, state_points, etc.
    """
    cvs: list[cv_input.CV_input] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=validators.instance_of(
                                    cv_input.CV_input),
                                iterable_validator=validators.instance_of(list),
                            ))
    state_points: list[State_point] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=validators.instance_of(
                                    State_point),
                                iterable_validator=validators.instance_of(list),
                            ))
    anchors: list[data_structures.Anchor_input] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=validators.instance_of(
                                    data_structures.Anchor_input),
                                iterable_validator=validators.instance_of(list),
                            ))

    def make_anchors(
            self, 
            starting_anchor_index: int,
            starting_milestone_index: int,
            connection_flag_dict: dict[int|str, list[data_structures.Anchor]], 
            associated_input_anchor: dict[int, data_structures.Anchor_input]
            ) -> typing.Tuple[typing.List[data_structures.Anchor], list[tuple[int, int]], int]:
        """
        For this partition, create the anchors that are defined
        herein.
        """
        anchors = []
        need_voronoi_connections = False
        if len(self.cvs) == 0:
            raise Exception("CVs must be defined within a Partition_input.")
        
        anchor_connections: list[tuple[int, int]] = []
        for alpha, input_anchor in enumerate(self.anchors):
            anchor: data_structures.Anchor = data_structures.Anchor()
            anchor.index = starting_anchor_index + alpha
            anchor.directory = f"anchor_{anchor.index}"
            if input_anchor.name == "":
                anchor.name = anchor.directory
            else:
                anchor.name = input_anchor.name

            if not input_anchor.bulk_anchor:
                anchor.md = True
                if input_anchor.end_anchor:
                    anchor.end_anchor = input_anchor.end_anchor
            else:
                anchor.md = False
                anchor.name = "bulk"
                anchor.end_anchor = True

            associated_input_anchor[anchor.index] = input_anchor
            cv_indices = []
            for j, cv_input in enumerate(self.cvs):
                #variable_name = f"{self.cvs[j]._variable_name}_{cv_input._index}"
                variable_name = f"value_{cv_input._index}"
                variable_value = input_anchor.get_variable_value(j)
                anchor.variables[variable_name] = variable_value
                cv_indices.append(cv_input._index)
                
            if len(self.cvs) == 1:
                for connection_flag in input_anchor.connection_flags:
                    connection_flag_dict[connection_flag].append(anchor)
                if input_anchor.bulk_anchor:
                    connection_flag_dict["bulk"].append(anchor)
                if alpha > 0:
                    pair = (alpha-1, alpha)
                    if (pair not in anchor_connections) \
                            and (pair[::-1] not in anchor_connections):
                        anchor_connections.append(pair)
                
            else:
                # TODO: connection flags might work with Voronoi tesselations now:
                #  test it out.
                #assert len(input_anchor.connection_flags) == 0, \
                #    "Connection flags within Voronoi CVs not currently supported."
                need_voronoi_connections = True

            # Temporarily place the absolute path of the topology file(s)
            #  and position file in the anchor. This will later be changed
            #  to relative paths.
            # NOTE: this approach was changed to using anchor.stage_parameters
            #anchor.md_parameters_topology = input_anchor.md_parameters_topology
            #anchor.positions = input_anchor.positions
            
            anchor.stage_parameters = []
            if input_anchor.positions is not None and input_anchor.positions != "":
                stage_parameter = {}
                stage_parameter["directory"] = base.INITIAL_DIRECTORY
                #stage_parameter["md_parameters_topology"] = input_anchor.md_parameters_topology
                stage_parameter["positions"] = input_anchor.positions
                anchor.stage_parameters = [stage_parameter]
            anchors.append(anchor)
        
        # TODO: implement voronoi connections - maybe different per
        # sampling method?
        if need_voronoi_connections:
            neighbor_anchors = find_voronoi_anchor_neighbors(anchors)
            for alpha, neighbor_set in neighbor_anchors.items():
                for beta in neighbor_set:
                    if beta > alpha:
                        pair = (alpha, beta)
                    else:
                        pair = (beta, alpha)
                    if (pair not in anchor_connections) \
                            and (pair[::-1] not in anchor_connections):
                        anchor_connections.append(pair)

        # TODO: make milestones
        milestone_index = starting_milestone_index
        for pair in anchor_connections:
            alpha, beta = pair
            anchor_alpha = anchors[alpha]
            anchor_beta = anchors[beta]
            milestone_index = make_milestone_between_two_voronoi_anchors(
                anchor_alpha, anchor_beta, milestone_index,
                cv_indices)
            
        return anchors, anchor_connections, milestone_index
    
@define
class Seekr_input:
    """
    Parameters that are input by the user.
    """
    directory: str = field(default="",
                            validator=validators.instance_of(str))
    temperature: float = field(default=298.15,
                               validator=[validators.instance_of(float),
                                          validators.gt(0)])
    random_seed: int | None = field(default=None,
                                    validator=validators.optional(
                                        validators.instance_of(int)))
    toy_settings: Toy_settings_input | None = None
    md_settings: engines_input.MD_settings_input | None = None
    bd_settings: engines_input.BD_settings_input | None = None
    stages: list[Input_stage] = field(default=Factory(list),)
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
            use_absolute_directory: bool=False
            ) -> data_structures.Seekr_model:
        """
        Given the Seekr_input object, which contains the settings, 
        create the Model object which contains more detailed 
        information.
        """
        
        model = data_structures.Seekr_model()
        if use_absolute_directory:
            model.directory = self.directory
        else:
            model.directory = "."
        model.temperature = self.temperature
        if self.md_settings is not None:
            model.md_engine = self.md_settings.make_model_engine(self.temperature)
        if self.toy_settings is None:
            model.is_toy_system = False
        else:
            model.is_toy_system = True
            if model.md_engine is not None:
                self.toy_settings.assign_toy_settings(model.md_engine)
        if self.bd_settings is not None:
            model.bd_engine = self.bd_settings.make_model_engine()
        # Assign stage information
        model.stages = []
        if self.md_settings is not None:
            if self.md_settings.systemwide_settings is not None:
                if self.md_settings.systemwide_settings.positions is None:
                    # Then the first stage is started from the anchors
                    systemwide_stage_input = False
                else:
                    if not model.using_toy():
                        assert self.md_settings.systemwide_settings\
                            .md_parameters_topology is not None, \
                            "If starting from a systemwide position, the topology file "\
                            "must be specified systemwide unless using toy mode."
                    systemwide_stage_input = True
        
        last_md_stage_index = 0
        for i, stage_input in enumerate(self.stages):
            stage, systemwide_stage_input = stage_input.make_model_stage(
                i+1, systemwide_stage_input)
            if stage.type == "md_stage":
                stage.input_stage_index = last_md_stage_index
                last_md_stage_index = stage.index
            model.stages.append(stage)
                
        return model

def generate_directory(directory, empty_dir=False):
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

def read_input(input_filename):
    """
    Read the input JSON file into the input object.
    """
    with open(input_filename, "r") as file:
        json_string = json.load(file)
    
    # Use a cattrs Converter to define how to structure the dictionary input
    converter = Converter()
    # Register a structure hook for the Positions_type
    converter.register_structure_hook(engines_base.Positions_type, 
                                      data_structures.structure_positions_type_union)
    #converter.register_structure_hook(data_structures.Completion_criteria, 
    #                                  data_structures.structure_completion_criteria_type_union)
    #converter.register_structure_hook(data_structures.Stage, 
    #                                  data_structures.structure_stage_type_union)
    seekr_input = converter.structure(json_string, Seekr_input)
    return seekr_input

def copy_bd_files(
        model: data_structures.Seekr_model, 
        seekr_input: Seekr_input,
        root_directory: str
        ) -> None:
    """
    Copy the BD files into the anchor directory.
    """
    assert seekr_input.bd_settings is not None, \
        "BD settings must be defined to copy BD files."
    assert model.bd_engine is not None, \
        "BD engine must be defined to copy BD files."
    assert model.bd_engine.systemwide_settings is not None, \
        "BD systemwide settings must be defined - anchor-by-anchor " \
        "not yet implemented."
    for anchor in model.anchors:
        if anchor.bd and not anchor.md:
            # If anchor.bd is True, while anchor.md is False, then this
            # must be the b-surface directory - "bulk anchor"
            b_surface_directory = os.path.join(
                root_directory, anchor.directory)
            for molecule in seekr_input.bd_settings.molecules:
                pqr_filename = os.path.basename(molecule.pqr_filename)
                new_bd_molecule = browndye_base.Browndye_molecule()
                new_bd_molecule.name = os.path.splitext(pqr_filename)[0]
                new_bd_molecule.pqr_filename = pqr_filename
                pqr_new_filename = os.path.join(b_surface_directory, pqr_filename)
                ghost_site_indices = molecule.add_ghost_atoms_to_pqr(pqr_new_filename)
                new_bd_molecule.ghost_site_indices = ghost_site_indices
                model.bd_engine.systemwide_settings.molecules.append(new_bd_molecule)
                
    return

def create_cvs(
        model: data_structures.Seekr_model, 
        seekr_input: Seekr_input, 
        root_directory: str
        ) -> list[cv_data_struct.Collective_variable]:
    """
    Create the collective variable objects for the model.
    
    Parameters:
    -----------
    model : Seekr_model()
        The model object representing the entire calculation.
    
    collective_variable_inputs : list
        A list of cv input objects used to construct the anchors for
        this model.
        
    root_directory : str
        A path to the root directory of this model. Used to copy over
        RMSD CV reference structure.
    
    Returns:
    --------
    cvs : list
        A list of collective variable objects that will be placed into
        the model.
    
    """
    cvs = []
    cv_counter = 0
    for partition in seekr_input.partitions:
        assert len(partition.cvs) > 0, \
            "Collective variables must be defined if a partition exists."
        for cv_input in partition.cvs:
            cv_input.check(partition.anchors)
            cv: cv_data_struct.Collective_variable \
                = cv_input.make_cv_object(index=cv_counter)
            cv.fill_input_values_from_model(model, root_directory)
            cvs.append(cv)
            cv_counter += 1

    return cvs

def make_state_points(
        seekr_input: Seekr_input,
        num_model_cvs: int
        ) -> list[State_point]:
    """
    Create the State_point objects for the Model from the input partitions.
    This is necessary because partitions will, in general, have a different
    number of CVs than the model. This means that, for the model, the "value"
    attribute needs to be filled out with None values for the CVs that are
    in other partitions.
    """
    state_points: list[State_point] = []
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
            state_point = State_point(
                name=input_state_point.name,
                value=new_value,
                end_state=input_state_point.end_state)
            state_points.append(state_point)
    
    return state_points

def create_anchors(
        seekr_input: Seekr_input
        ) -> tuple[list[data_structures.Anchor], dict, dict, list, int]:
    """
    Create the Anchor objects for the Model from the input partitions.
    
    Parameters:
    -----------
    model : Model()
        The model object representing the entire calculation.
    
    """
    anchor_index: int = 0
    anchors: list[data_structures.Anchor] = []
    connection_flag_dict: dict[int|str, list[data_structures.Anchor]] = defaultdict(list)
    associated_input_anchor: dict[int, data_structures.Anchor_input] = {}
    anchor_connections: list[tuple[int, int]] = []
    anchor_index = 0
    milestone_index = 0
    for partition in seekr_input.partitions:
        partition_anchors, partition_anchor_connections, milestone_index\
            = partition.make_anchors(
                anchor_index, milestone_index, connection_flag_dict, 
                associated_input_anchor)
        anchors += partition_anchors
        anchor_index += len(partition_anchors)
        anchor_connections += partition_anchor_connections

    return anchors, connection_flag_dict, associated_input_anchor, \
        anchor_connections, milestone_index

"""
def assign_bd_anchors(
        model: data_structures.Seekr_model,
        seekr_input: Seekr_input
        ) -> None:
    ""
    Detect anchors that will be significant for BD calculation.
    ""
    for anchor in model.anchors:
        if anchor.name == "bulk":
            anchor.bd = True
"""

def assign_all_state_points(
        state_points: list[State_point], 
        model: data_structures.Seekr_model
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
                "anchors found. The point might be located on a milestone.")
    return

def prepare_model_cvs_and_anchors(
        model: data_structures.Seekr_model, 
        seekr_input: Seekr_input, 
        force_overwrite: bool
        ) -> None:
    """
    Fill out the CV and anchor items within the model based on the 
    input objects.
    """
    cvs = create_cvs(model, seekr_input, seekr_input.directory)
    model.collective_variables = cvs
    anchors, connection_flag_dict, associated_input_anchor, \
        anchor_connections, milestone_index = create_anchors(seekr_input)
    for anchor in anchors:
        model.anchors.append(anchor)

    # TODO: more thoroughly test make_state_points, because once
    #  it incorrectly assigned the name of a Voronoi anchor in a 2D
    #  tesselation.
    state_points = make_state_points(seekr_input, len(cvs))
    # Note: this must be done BEFORE resolve_connections() because
    #  it calls modify_model() which requires the BD anchors to be
    #  assigned.
    
    # assign connections between anchors
    anchors, index_reference = cv_input.resolve_connections(
        connection_flag_dict, model, associated_input_anchor, 
        anchor_connections)
    cv_input.resolve_modifications_and_filetree(
        model, anchors, index_reference, associated_input_anchor, 
        seekr_input.directory, force_overwrite)
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
