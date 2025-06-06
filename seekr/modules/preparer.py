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
import types
import typing
import shutil
import random
import logging
import importlib

from attrs import define, field, validators, Factory
import cattrs
from cattrs import Converter
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import QhullError
import mdtraj

import seekr.modules.base as base
import seekr.modules.collective_variables.base as cv_base
import seekr.modules.engines.base as engines_base
import seekr.modules.structures as structures
import seekr.modules.collective_variables.structures as cv_structures
import seekr.modules.completion_criteria.structures as compl_crit_struct
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.preparer as engines_preparer
import seekr.modules.restraints.preparer as restraints_preparer
import seekr.modules.reporters.preparer as reporters_preparer
import seekr.modules.sampling_methods.preparer as sampling_methods_preparer
import seekr.modules.collective_variables.preparer as cv_preparer
import seekr.modules.engines.browndye.preparer as browndye_preparer
import seekr.modules.engines.sda.preparer as sda_preparer
import seekr.modules.engines.openmm.preparer as openmm_preparer

BD_settings_input = typing.Union[browndye_preparer.Browndye_settings_input,
                                sda_preparer.SDA_settings_input]
MD_settings_input = typing.Union[openmm_preparer.Openmm_settings_input]

#MD_sampling_method_input = typing.Union[
#    sampling_methods_preparer.MD_sampling_method_input]

restraint_inputs: typing.Dict[str, typing.Type[restraints_preparer.Base_restraint_input]] \
        = {"positional": restraints_preparer.Positional_restraint_input,
           "pairwise": restraints_preparer.Pairwise_restraint_input,
           "proximity": restraints_preparer.Proximity_restraint_input,}
reporter_inputs: typing.Dict[str, typing.Type[reporters_preparer.Base_reporter_input]] \
        = {"dcd": reporters_preparer.DCD_reporter_input,
           "pdb": reporters_preparer.PDB_reporter_input,
           "state": reporters_preparer.State_reporter_input,}
compl_crit_inputs: typing.Dict[str, typing.Type[compl_crit_struct.Base_completion_criteria]] \
        = structures.completion_criteria
#cv_inputs: typing.Dict[str, typing.Type[cv_base.Base_CV_input]] \
#        = cv_preparer.cv_inputs
from seekr.modules.collective_variables.preparer import cv_inputs
md_sampling_methods_input: typing.Dict[str, typing.Type[sampling_methods_preparer.Base_md_sampling_method_input]] \
        = {"conventional": sampling_methods_preparer.Conventional_md_input,
           "metadynamics": sampling_methods_preparer.Metadynamics_md_input,
           "steered": sampling_methods_preparer.Steered_md_input,
           "mmvt": sampling_methods_preparer.MMVT_md_input,}
bd_sampling_methods_input: typing.Dict[str, typing.Type[sampling_methods_preparer.Base_md_sampling_method_input]] \
        = {"NAM": sampling_methods_preparer.NAM_bd_input,
           "WE": sampling_methods_preparer.WE_bd_input,}

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
    checkpoint_interval: int = field(default=1000,
                                  validator=[validators.instance_of(int),
                                             validators.gt(0)])
    ensemble: structures.Ensemble = field(default=Factory(
        structures.NVT_ensemble))
    sampling_method: sampling_methods_preparer.Base_md_sampling_method_input = field(
        default=Factory(sampling_methods_preparer.Conventional_md_input))
    completion_criteria: compl_crit_struct.Base_completion_criteria | None = None
    restraints: list[restraints_preparer.Base_restraint_input] = field(default=Factory(list))
    reporters: list[reporters_preparer.Base_reporter_input] = field(default=Factory(list))
    
    def make_model_stage(
            self, 
            index: int,
            systemwide_stage_input: bool,
            systemwide_settings_input: engines_preparer.MD_systemwide_settings_input
            ) -> typing.Tuple[structures.Stage_md, bool]:
        """
        Given this input stage object, create the model's Stage object.
        """
        stage = structures.Stage_md()
        stage.index = index
        stage.name = self.name
        stage.run_minimization = self.run_minimization
        #stage.md_state_output_interval = self.output_interval
        #stage.md_trajectory_output_interval = self.output_interval
        stage.md_restart_checkpoint_interval = self.checkpoint_interval
        stage.ensemble = self.ensemble
        stage.sampling_method = self.sampling_method.make_model_sampling_method()
        assert self.completion_criteria is not None
        stage.completion_criteria = self.completion_criteria
        if systemwide_stage_input: # If coming from systemwide settings
            assert not stage.sampling_method.needs_to_be_in_anchor(), \
                f"Stage '{stage.name}' with sampling method '{stage.sampling_method.type}' "\
                "must be run within one or more anchors. A stage that populates anchors "\
                "must precede this stage."
            stage.input_from = "systemwide"
            if stage.sampling_method.input_from_systemwide_settings_output_to_anchors():
                new_systemwide_stage_input = False
                stage.output_to = "anchors"
                # Handle different situations?
            else:
                new_systemwide_stage_input = True
                stage.output_to = "systemwide"
        else:
            stage.input_from = "anchors"
            if stage.sampling_method.input_from_anchors_output_to_systemwide_settings():
                new_systemwide_stage_input = True
                stage.output_to = "systemwide"
            else:
                new_systemwide_stage_input = False
                stage.output_to = "anchors"
        for restraint_input in self.restraints:
            restraint = restraint_input.make_restraint(systemwide_settings_input)
            stage.restraints.append(restraint)
        for reporter_input in self.reporters:
            reporter = reporter_input.make_reporter(systemwide_settings_input)
            stage.reporters.append(reporter)
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
    sampling_method: sampling_methods_preparer.Base_bd_sampling_method_input = field(
        default=Factory(sampling_methods_preparer.NAM_bd_input))
    completion_criteria: compl_crit_struct.Base_completion_criteria = field(
        default=Factory(compl_crit_struct.Number_of_trajectories_completion_criteria))
    
    def make_model_stage(
            self, 
            index: int,
            systemwide_stage_input: bool,
            bd_settings: BD_settings_input
            ) -> typing.Tuple[structures.Stage_bd, bool]:
        """
        Given this input stage object, create the model's Stage object.
        """
        stage = structures.Stage_bd()
        stage.index = index
        stage.name = self.name
        stage.n_trajectories_per_output = self.n_trajectories_per_output
        stage.n_steps_per_output = self.n_steps_per_output
        stage.completion_criteria = self.completion_criteria
        # At present, BD stages will always be systemwide
        #stage.input_from = None
        #stage.output_to = None
        # TODO: implement restraints in BD - that will use bd_settings argument
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
            seekr_model_md_engine : structures.MD_engine
            ) -> None:
        """
        Assign the toy settings to the model engine.
        """
        seekr_model_md_engine.toy_settings = engines_structures.Toy_settings()
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

# TODO: change this function to accept a list of positions, not
#  Anchor objects, since anchors might not be the voronoi cells necessarily.
#  Part of the broader attempt to enable Voronoi edges as anchors.
def find_voronoi_anchor_neighbors(
        anchors: list[structures.Anchor]
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
            anchor: structures.Anchor, 
            neighbor_anchor: structures.Anchor,
            milestone_index: int, 
            cv_indices: list[int]) -> int:
        neighbor_index = neighbor_anchor.index
        assert anchor.index != neighbor_anchor.index
        for milestone in anchor.milestones:
            if milestone.neighbor_anchor_index == neighbor_anchor.index:
                return milestone_index
        
        milestone1 = structures.Milestone()
        milestone1.index = milestone_index
        milestone1.neighbor_anchor_index = neighbor_index
        milestone1.alias_index = len(anchor.milestones)+1
        milestone1.variables = {"k": 1.0}
        milestone2 = structures.Milestone()
        milestone2.index = milestone_index
        milestone2.neighbor_anchor_index = anchor.index
        milestone2.alias_index = len(neighbor_anchor.milestones)+1
        milestone2.variables = {"k": -1.0}
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
            if (len(cv_indices) == 1):
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

# TODO: rethink this naming - this is a set of Voronoi points, and the anchors might be
#  assigned differently
@define
class Anchor_input:
    """
    A base class for Anchor_inputs.

    In SEEKR, an anchor is a point in CV space. How this point
    is used within SEEKR depends on the enhanced sampling
    method. For instance, an anchor is used with MMVT to define
    a Voronoi cell, whose boundaries become the milestones. In
    Elber milestoning, two anchors are used to define two 
    Voronoi cells, and the milestone in between the two cells
    is the starting milestone, while the other boundaries define
    the ending milestones.
    """
    name: str = field(default="",
                           validator=validators.instance_of(str))
    value: list[float] = field(default=Factory(list),
                                validator=validators.deep_iterable(
                                    member_validator=[validators.instance_of(float)],
                                    iterable_validator=validators.instance_of(list),
                                ))
    md_parameters_topology: engines_structures.Parameters_topology | None = None
    positions: base.Positions_type = None
    selections: list[engines_structures.Selection] | None = field(default=None,
                                        validator=validators.optional(
                                            validators.deep_iterable(
                                                member_validator=validators.instance_of(
                                                    engines_structures.Selection),
                                                iterable_validator=validators.instance_of(list),
                                            )))
    end_anchor: bool = field(default=False,
                            validator=validators.instance_of(bool))
    bulk_anchor: bool = field(default=False,
                            validator=validators.instance_of(bool))
    connection_flags: list[int] = field(default=Factory(list),
                                        validator=validators.deep_iterable(
                                            member_validator=[validators.instance_of(int)],
                                            iterable_validator=validators.instance_of(list),
                                        ))
    
    def check(self) -> None:
        """
        Check the anchor input for errors.
        """
        if self.bulk_anchor:
            assert self.end_anchor, "Bulk anchor must be an end anchor."
        return
    
    def get_variable_value(
            self,
            index: int
            ) -> float:
        """
        Return the variable value for this anchor.
        """
        return self.value[index]
    
    def same_md_parameters(
            self,
            other: "Anchor_input"
            ) -> bool:
        """

        """
        if self.md_parameters_topology is None:
            if other.md_parameters_topology is None:
                return True
            else:
                return False
        else:
            if other.md_parameters_topology is None:
                return False
            else:
                if self.md_parameters_topology.same_parameters(other.md_parameters_topology):
                    return True
                else:
                    return False
                
    def copy_initial_files(
            self, 
            anchor: structures.Anchor, 
            root_directory: str,
            toy_mode: bool
            ) -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        if anchor.stage_parameters[0] is not None:
            anchor_initial_directory: str = os.path.join(
                root_directory, anchor.directory, anchor.stage_parameters[0]["directory"])
            anchor.stage_parameters[0]["positions"] = engines_preparer.POSITIONS_BASENAME
            if toy_mode:
                # MYPY: Similar to the systemwide case, mypy is not happy because
                # the toy positions are an array of positions, but non-toy systems
                # are a filename for a PDB. Not sure how to resolve this.
                for frame_id, frame_positions in enumerate(self.positions):
                    full_positions_path = os.path.join(
                        anchor_initial_directory,
                        f"{engines_preparer.POSITIONS_BASENAME}.swarm{frame_id}.txt")
                    np.savetxt(full_positions_path, np.array(frame_positions))
            else:
                if self.md_parameters_topology is not None:
                    anchor.md_parameters_topology = self.md_parameters_topology
                    self.md_parameters_topology.copy_files(
                        anchor_initial_directory, anchor.md_parameters_topology)
                if (self.positions is not None) and (self.positions != ""):
                    assert isinstance(self.positions, str), \
                        "anchor.positions must be a filename str if not in Toy mode."
                    pdb_basename: str = os.path.basename(self.positions)
                    anchor_pdb_full_path: str = os.path.join(anchor_initial_directory,
                                                            pdb_basename)
                    mdtraj_obj = mdtraj.load(anchor_pdb_full_path)
                    num_frames = mdtraj_obj.n_frames
                    for frame_id in range(num_frames):
                        output_pdb_full_path = os.path.join(
                            anchor_initial_directory,
                            f"{engines_preparer.POSITIONS_BASENAME}.swarm_{frame_id}.pdb")
                        mdtraj_obj[frame_id].save(output_pdb_full_path)

# TODO: implement different types of partitions for different
#  sampling methods: I.E.: Elber milestoning will need different ways
#  of defining anchors than MMVT.
# Is this going to require a new type of Partition_input class?
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
    state_points: list[State_point] = field(default=Factory(list),)
                            #validator=validators.deep_iterable(
                            #    member_validator=validators.instance_of(
                            #        State_point),
                            #    iterable_validator=validators.instance_of(list),
                            #))
    # TODO: How are we going to define Voronoi points independently from
    #  anchor information such as structures, etc.?
    anchors: list[Anchor_input] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=validators.instance_of(
                                    Anchor_input),
                                iterable_validator=validators.instance_of(list),
                            ))

    def make_anchors(
            self, 
            starting_anchor_index: int,
            starting_milestone_index: int,
            connection_flag_dict: dict[int|str, list[structures.Anchor]], 
            associated_input_anchor: dict[int, "Anchor_input"]
            ) -> typing.Tuple[typing.List[structures.Anchor], list[tuple[int, int]], int]:
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
            anchor: structures.Anchor = structures.Anchor()
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

            anchor.stage_parameters = []
            if input_anchor.positions is not None and input_anchor.positions != "":
                stage_parameter: dict[str, str | int | list] = {}
                stage_parameter["index"] = 0
                stage_parameter["directory"] = base.INITIAL_DIRECTORY
                #stage_parameter["md_parameters_topology"] = input_anchor.md_parameters_topology
                stage_parameter["positions"] = input_anchor.positions
                anchor.stage_parameters = [stage_parameter]
            else:
                anchor.stage_parameters = [None]
            anchors.append(anchor)
        
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
    plugins: list[str] = field(default=Factory(list),)
    temperature: float = field(default=298.15,
                               validator=[validators.instance_of(float),
                                          validators.gt(0)])
    random_seed: int | None = field(default=None,
                                    validator=validators.optional(
                                        validators.instance_of(int)))
    toy_settings: Toy_settings_input | None = None
    md_settings: typing.Union[MD_settings_input, None] = None
    bd_settings: BD_settings_input | None = None
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
            if stage_input.type == "md_stage":
                assert self.md_settings is not None
                assert self.md_settings.systemwide_settings is not None
                stage_md, systemwide_stage_input = stage_input.make_model_stage(
                    i+1, systemwide_stage_input, self.md_settings.systemwide_settings)
                stage_md.input_stage_index = last_md_stage_index
                last_md_stage_index = stage_md.index
                model.stages.append(stage_md) 
            elif stage_input.type == "bd_stage":
                assert self.bd_settings is not None
                stage_bd, systemwide_stage_input = stage_input.make_model_stage(
                    i+1, systemwide_stage_input, self.bd_settings)
                model.stages.append(stage_bd)  
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

def struct_restraint_inputs(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[restraints_preparer.Base_restraint_input]
        ) -> restraints_preparer.Base_restraint_input:
    typ = obj.pop("type")
    return cattrs.structure(obj, restraint_inputs[typ])

def struct_reporter_inputs(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[reporters_preparer.Base_reporter_input]
        ) -> reporters_preparer.Base_reporter_input:
    typ = obj.pop("type")
    return cattrs.structure(obj, reporter_inputs[typ])

def struct_completion_criteria(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[compl_crit_struct.Base_completion_criteria]
        ) -> compl_crit_struct.Base_completion_criteria:
    typ = obj.pop("type")
    return cattrs.structure(obj, compl_crit_inputs[typ])

def struct_cv_inputs(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[cv_base.Base_CV_input]
        ) -> cv_base.Base_CV_input:
    typ = obj.pop("type")
    return cattrs.structure(obj, cv_inputs[typ])

def struct_md_sampling_method_inputs(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[sampling_methods_preparer.Base_md_sampling_method_input]
        ) -> sampling_methods_preparer.Base_md_sampling_method_input:
    typ = obj.pop("type")
    return cattrs.structure(obj, md_sampling_methods_input[typ])

def struct_bd_sampling_method_inputs(
        obj: typing.Dict[str, typing.Any],
        _: typing.Type[sampling_methods_preparer.Base_bd_sampling_method_input]
        ) -> sampling_methods_preparer.Base_bd_sampling_method_input:
    typ = obj.pop("type")
    return cattrs.structure(obj, bd_sampling_methods_input[typ])

def read_input(input_filename: str) -> Seekr_input:
    """
    Read the input JSON file into the input object.
    """
    global restraint_inputs
    global reporter_inputs
    global compl_crit_inputs
    with open(input_filename, "r") as file:
        json_string = json.load(file)
    
    for plugin_name in json_string['plugins']:
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
            plugin_name, submodule_name="completion_criteria", require_runner=True)
        if compl_crit_module_str is not None:
            compl_crit_module_struct = importlib.import_module(compl_crit_module_str+".structures")
            compl_crit_inputs.update(compl_crit_module_struct.completion_criteria)

        cv_module_str = base.get_plugin_str(
            plugin_name, submodule_name="collective_variables", require_preparer=True)
        if cv_module_str is not None:
            cv_module_prep = importlib.import_module(cv_module_str+".preparer")
            cv_inputs.update(cv_module_prep.cv_inputs)

        sampl_meth_module_str = base.get_plugin_str(
            plugin_name, submodule_name="sampling_methods", require_preparer=True)
        if sampl_meth_module_str is not None:
            sampl_meth_module_prep = importlib.import_module(sampl_meth_module_str+".preparer")
            md_sampling_methods_input.update(sampl_meth_module_prep.md_sampling_methods_input)
            bd_sampling_methods_input.update(sampl_meth_module_prep.bd_sampling_methods_input)
        
    # Use a cattrs Converter to define how to structure the dictionary input
    converter = Converter()
    
    # Register a structure hook for the Positions_type
    converter.register_structure_hook(base.Positions_type, 
                                      structures.structure_positions_type_union)
    converter.register_structure_hook(base.Float_or_list_type, 
                                      structures.structure_float_or_list_type_union)
    converter.register_structure_hook(base.Int_or_list_type, 
                                      structures.structure_int_or_list_type_union)
    
    converter.register_structure_hook(restraints_preparer.Base_restraint_input,
                                        struct_restraint_inputs)
    converter.register_structure_hook(reporters_preparer.Base_reporter_input,
                                        struct_reporter_inputs)
    converter.register_structure_hook(compl_crit_struct.Base_completion_criteria,
                                        struct_completion_criteria)
    converter.register_structure_hook(cv_base.Base_CV_input,
                                        struct_cv_inputs)
    converter.register_structure_hook(sampling_methods_preparer.Base_md_sampling_method_input,
                                        struct_md_sampling_method_inputs)
    converter.register_structure_hook(sampling_methods_preparer.Base_bd_sampling_method_input,
                                        struct_bd_sampling_method_inputs)
    
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
    associated_input_anchor: dict[int, Anchor_input] = {}
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

def assign_all_state_points(
        state_points: list[State_point], 
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
        anchor_or_systemwide: engines_structures.MD_systemwide_settings \
            | structures.Anchor
        ) -> None:
    """
    Delete all simulation outputs from systemwide or anchor.
    """
    for stage in model.stages:
        cleanse_outputs_stage(model, anchor_or_systemwide, stage.index-1)
    return

def cleanse_outputs_stage(
        model: "structures.Seekr_model",
        anchor_or_systemwide: engines_structures.MD_systemwide_settings \
            | structures.Anchor, 
        stage_index: int
        ) -> None:
    """
    Delete all simulation outputs from systemwide or anchor.
    """
    stage = model.stages[stage_index]
    if stage.type == "md_stage":
        full_directory = os.path.join(model.directory, anchor_or_systemwide.directory, 
                                    stage.get_directory_name())
        if os.path.exists(full_directory):
            file_globs = stage.sampling_method.get_output_positions_glob_list()
            for file_glob in file_globs:
                full_glob = os.path.join(full_directory, file_glob)
                for file in glob.glob(full_glob):
                    os.remove(file)
            dcd_glob = os.path.join(full_directory, "*.dcd")
            for dcd_file in glob.glob(dcd_glob):
                os.remove(dcd_file)
            out_glob = os.path.join(full_directory, "*.out")
            for out_file in glob.glob(out_glob):
                os.remove(out_file)
            log_glob = os.path.join(full_directory, "*.log")
            for log_file in glob.glob(log_glob):
                os.remove(log_file)
            backup_info_glob = os.path.join(full_directory, 
                engines_base.INFORMATION_CHECKPOINT_FILENAME_NOEXT \
                + "*" + engines_base.INFORMATION_CHECKPOINT_FILENAME_EXT)
            for backup_info_file in glob.glob(backup_info_glob):
                os.remove(backup_info_file)
            backup_glob = os.path.join(full_directory, 
                engines_base.RESTART_CHECKPOINT_FILENAME_NOEXT \
                + "*" + engines_base.RESTART_CHECKPOINT_FILENAME_EXT)
            for backup_file in glob.glob(backup_glob):
                os.remove(backup_file)
            lock_filename = os.path.join(full_directory, "LOCK")
            if os.path.exists(lock_filename):
                os.remove(lock_filename)
    elif stage.type == "bd_stage":
        assert model.bd_engine is not None
        if anchor_or_systemwide.directory == model.bd_engine.b_surface_directory:
            full_directory = os.path.join(model.directory, model.bd_engine.b_surface_directory)
            if os.path.exists(full_directory):
                output_files_glob = os.path.join(full_directory, model.bd_engine.results_glob)
                output_files_list = glob.glob(output_files_glob)
                for output_file in output_files_list:
                    os.remove(output_file)
                
                simulation_file_glob = os.path.join(full_directory, "*_simulation.xml")
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
            plugin_module_prep.cleanse_outputs_stage(model, anchor_or_systemwide, stage_index)

    return

def modify_model(
        old_model: structures.Seekr_model, 
        new_model: structures.Seekr_model, 
        root_directory: str, 
        force_overwrite=False) -> list[int]:
    """
    If someone runs the prepare stage on a model with existing
    directories and simulation results, examine the changes to delete
    and rename directories properly, and only overwrite simulation files
    if forced to do so.
    """
    
    old_model.directory = root_directory
    anchor_pairs: list[tuple[int, int]] = []
    old_anchors_to_delete: list[int] = []
    new_anchors_to_create: list[int] = []
    
    #index_where_stages_begin_to_differ = None
    # NOTE: I think we don't want to do this - if the stage settings change
    # I think it is up to the user to choose whether to force overwrite subsequent
    # stages.
    #for i, (old_stage, new_stage) in enumerate(zip(old_model.stages, new_model.stages)):
    #    if old_stage != new_stage:
    #        index_where_stages_begin_to_differ = i
    
    #stages_to_delete_indices = []
    #if index_where_stages_begin_to_differ is not None:
    #    for i in range(index_where_stages_begin_to_differ, len(old_model.stages)):
    #        stages_to_delete_indices.append(i)
    
    stages_to_delete_indices_with_files: list[int] = []
    stages_to_delete_names_with_files: list[str] = []
    #for i in stages_to_delete_indices:
    #    old_model_stage = old_model.stages[i]
    #    if old_model_stage.has_files(old_model):
    #        stages_to_delete_indices_with_files.append(old_model_stage.index)
    #        stages_to_delete_names_with_files.append(old_model_stage.name)
            
    # Look through the new and old milestones and pair them up by the value
    #  of their variables
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
    old_anchors_with_starting_pdbs_to_keep: list[int] = []
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
            
    # NOTE: no need to worry about systemwide because there are no values or milestones.
    warning_msg = ""
    # check whether deleting directories contain simulation files
    deleting_directories_with_files = []
    for del_anchor_index in old_anchors_to_delete:
        del_anchor: structures.Anchor = old_model.anchors[del_anchor_index]
        if del_anchor.has_files(root_directory):
            deleting_directories_with_files.append(del_anchor.directory)
    
    if len(deleting_directories_with_files) > 0 :
        warning_msg+=f"The anchor(s) {deleting_directories_with_files} "\
            +"already have existing output files and the entered command "\
            "would overwrite them. If you desire to overwrite the existing "\
            "files, then use the --force_overwrite (-f) option, and these "\
            f"anchors will be deleted in directory: {root_directory}\n\n"
    
    # Check whether directories with changed milestones already contain
    # simulation files
    cleansing_anchors: list[structures.Anchor] = []
    cleansing_directories: list[str] = []
    for cleanse_anchor_index in old_anchors_with_changed_milestones:
        cleanse_anchor: structures.Anchor = old_model.anchors[cleanse_anchor_index]
        if cleanse_anchor.has_files(root_directory):
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
    
    for del_anchor_index in old_anchors_to_delete:
        del_anchor = old_model.anchors[del_anchor_index]
        full_path = os.path.join(root_directory, del_anchor.directory)
        print("removing directory: "+del_anchor.directory)
        shutil.rmtree(full_path)
    
    for cleansing_anchor in cleansing_anchors:
        print("removing output files from anchor:", cleansing_anchor.name)
        cleanse_outputs_all(old_model, cleansing_anchor)
    
    # NOTE: remove this block because we want to trust the users to re-run
    # the necessary stages if they want to overwrite the files.
    #for stage_index in stages_to_delete_indices:
    #    stage = old_model.stages[stage_index]
    #    print("removing output files from stage:", stage.name)
    #    # TODO: method doesn't exist: fix this
    #    stage.remove_directory(old_model)

    # Now that the equivalent anchors are known, we know which ones will
    # be deleted, and which ones are being changed, with or without
    # simulation data, 
    
    new_anchors_with_starting_pdbs_to_keep = []
    for pair in anchor_pairs:
        (alpha, beta) = pair
        anchor1 = old_model.anchors[beta]
        anchor2 = new_model.anchors[alpha]
        full_path1 = os.path.join(root_directory, anchor1.directory)
        full_path2 = os.path.join(root_directory, anchor2.directory)
        if full_path1 != full_path2:
            if os.path.exists(full_path2):
                # then we cannot rename to this without giving this one
                # a temporary directory
                for problem_anchor in old_model.anchors:
                    if problem_anchor.directory == "":
                        continue
                    if problem_anchor.directory == anchor2.directory:
                        temporary_name = problem_anchor.directory+"temp"
                        temp_full_path1 = os.path.join(root_directory, 
                                                  problem_anchor.directory)
                        temp_full_path2 = os.path.join(root_directory, 
                                                  temporary_name)
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
            
        if anchor1.index in old_anchors_with_starting_pdbs_to_keep:
            new_anchors_with_starting_pdbs_to_keep.append(anchor2.index)
    
    for new_anchor_index_to_create in new_anchors_to_create:
        print("creating new anchor {}".format(new_anchor_index_to_create))
    
    return new_anchors_with_starting_pdbs_to_keep

def resolve_modifications_and_filetree(
        model: structures.Seekr_model, 
        anchors: list[structures.Anchor], 
        index_reference: dict[int, int],
        associated_input_anchor: dict[int, Anchor_input],
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
        logging.info("Model file already existed at this location. "\
                     "Modified the existing model.")
        model.set_random_seeds(old_model.get_random_seed())
        new_anchors_with_starting_pdbs_to_keep = modify_model(
            old_model, model, root_directory, force_overwrite)
    else:
        new_anchors_with_starting_pdbs_to_keep = None
    # Add all stages to the model anchors or systemwide
    allowed_inputs = ["anchors", "systemwide", None]
    allowed_outputs = ["anchors", "systemwide", None]
    for stage in model.stages:
        assert stage.input_from in allowed_inputs, \
            f"The input_from may not be {stage.input_from}."
        assert stage.output_to in allowed_outputs, \
            f"The output_to may not be {stage.output_to}."
        
        did_anchor_stage = False
        did_systemwide_stage = False
        if (stage.input_from == "anchors") \
                and stage.type == "md_stage":
            # NOTE: mypy error is handled with "md_stage" type check above
            stage.add_stage_parameters_to_anchors(model)
            did_anchor_stage = True
        elif (stage.input_from == "systemwide") \
                and stage.type == "md_stage":
            stage.add_stage_parameters_to_systemwide_settings(model)
            did_systemwide_stage = True

        if (stage.output_to == "anchors") \
                and stage.type == "md_stage" and not did_anchor_stage:
            # NOTE: mypy error is handled with "md_stage" type check above
            stage.add_stage_parameters_to_anchors(model)
            did_anchor_stage = True
        elif (stage.output_to == "systemwide") \
                and stage.type == "md_stage" and not did_systemwide_stage:
            stage.add_stage_parameters_to_systemwide_settings(model)
            did_systemwide_stage = True
        
        # Add Nones wherever a stage wasn't added somewhere
        if not did_anchor_stage and stage.type == "md_stage":
            stage.add_stage_parameters_to_anchors(model, do_none=True)
        if not did_systemwide_stage and stage.type == "md_stage":
            stage.add_stage_parameters_to_systemwide_settings(model, do_none=True)
    
    for old_index in index_reference:
        new_index = index_reference[old_index]
        if new_index < len(anchors):
            input_anchor = associated_input_anchor[old_index]
            anchors[new_index].generate_filetree(root_directory)
            if (new_anchors_with_starting_pdbs_to_keep is not None) \
                    and (new_index in new_anchors_with_starting_pdbs_to_keep):
                logging.info(f"Keeping starting structure for anchor: {new_index}")
            else:
                input_anchor.copy_initial_files(anchors[new_index], root_directory, 
                                                model.using_toy())
    return

def generate_bd_files(
        model: structures.Seekr_model, 
        directory: str
        ) -> None:
    """
    Determine the BD engine and prepare the necessary files for it.
    """
    found_function = False
    assert model.bd_engine is not None
    if model.bd_engine.type == "browndye":
        browndye_preparer.generate_bd_files(model, directory)
        found_function = True
    if model.bd_engine.type == "sda":
        sda_preparer.generate_bd_files(model, directory)
        found_function = True
    else:
        # implement check of installed plugins for other BD engines
        # Loop over modules:
        # found_function = function_in_module(...)
        pass#

    if not found_function:
        raise Exception(f"BD engine type {model.bd_engine.type} not recognized.")
    
def resolve_connections(
        connection_flag_dict: dict, 
        model: structures.Seekr_model, 
        associated_input_anchor: dict[int, Anchor_input],
        anchor_connections: list[tuple[int, int]]
        ) -> tuple[list[structures.Anchor], dict[int, int]]:
    """
    Given connection information between anchors, merge anchors
    that are connected.
    """
    anchors: list[structures.Anchor] = model.anchors
    anchor_indices_to_remove = []
    index_reference = {}
    bulk_anchor: structures.Anchor | None = None
    for anchor in anchors:
        index_reference[anchor.index] = anchor.index
    
    for key in connection_flag_dict:
        anchor_connection_list: list[structures.Anchor] \
            = connection_flag_dict[key]
        bulk_discarded_indices = []
        if key == "bulk":
            bulk_anchor = anchor_connection_list[0]
            for anchor_discarded in anchor_connection_list:
                anchor_indices_to_remove.append(anchor_discarded.index)
                bulk_discarded_indices.append(anchor_discarded.index)
                
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
            assert input_anchor_kept.same_md_parameters(
                input_anchor_discarded), \
                "The anchors connected by connection flag {} ".format(key) \
                +"have inconsistent MD parameters defined."
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
    
    if bulk_anchor is not None:
        # make sure the bulk anchor is the last anchor always
        anchors.append(bulk_anchor)
        index_reference[bulk_anchor.index] = len(anchors) - 1
        for bulk_discarded in bulk_discarded_indices:
            index_reference[bulk_discarded] = len(anchors) - 1
    
    visited_new_indices = []
    
    for old_index in index_reference:
        new_index = index_reference[old_index]
        if new_index < len(anchors):
            anchors[new_index].index = new_index
            anchors[new_index].directory = f"anchor_{new_index}"
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
    
    if bulk_anchor is None:
        assert not model.using_bd(), "A bulk anchor must be defined "\
            "if using BD. This can be done by naming an input_anchor "\
            "with the name 'bulk', or by setting the 'bulk_anchor' flag "\
            "to True."
    else:
        bulk_anchor.md = False
        if model.using_bd():
            bulk_anchor.bd = True
            assert model.bd_engine is not None, "A BD engine must be "\
                "defined if using BD."
            bulk_anchor.directory = model.bd_engine.b_surface_directory

            for neighbor_index in bulk_anchor.neighbor_anchor_indices:
                neighbor_anchor = anchors[neighbor_index]
                neighbor_anchor.bd = True


    return anchors, index_reference