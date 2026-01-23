"""
modules/anchor_schemes/preparer.py

Input objects for native anchor schemes to seekr.
"""

import os
import typing
import logging
from collections import defaultdict

from attrs import define, field, validators, Factory
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import QhullError
import mdtraj

import seekr.modules.base as base
import seekr.modules.collective_variables.base as cv_base
import seekr.modules.structures as structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.preparer as engines_preparer

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

def make_milestone_between_two_voronoi_cell_anchors(
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

@define
class Voronoi_cell:
    """
    A set of Voronoi cells may be used to define a Voronoi
    tesselation. The 'value' attribute is the point at the
    center of this Voronoi cell.
    """
    name: str = field(
        default="",
        validator=validators.instance_of(str))
    value: list[float] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(float)],
            iterable_validator=validators.instance_of(list),
        ))
    # TODO: replace with string list of scales involved in this anchor
    md: bool = field(
        default=True,
        validator=validators.instance_of(bool))
    bd: bool = field(
        default=False,
        validator=validators.instance_of(bool))
    # TODO: somehow replace this parameter with something more general
    # that could contain parameter and topology information for any type of scale
    md_parameters_topology: engines_structures.Parameters_topology | None = None
    positions: base.Positions_type = None
    selections: list[engines_structures.Selection] | None = field(
        default=None,
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(
                    engines_structures.Selection),
                iterable_validator=validators.instance_of(list),
            )))
    end_anchor: bool = field(
        default=False,
        validator=validators.instance_of(bool))
    bulk_anchor: bool = field(
        default=False,
        validator=validators.instance_of(bool))
    connection_flags: list[int] = field(
        default=Factory(list),
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
            other: "Voronoi_cell"
            ) -> bool:
        ""

        ""
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
            anchor_set_name: str,
            toy_mode: bool
            ) -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        if anchor.stage_parameters[0] is not None:
            anchor_initial_directory: str = os.path.join(
                root_directory, base.ANCHOR_SETS_DIR, anchor_set_name, anchor.directory, 
                anchor.stage_parameters[0]["directory"])
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

class Base_anchor_scheme(typing.Protocol):
    """
    Base class for all anchor schemes.
    """
    pass

@define
class Anchors_as_voronoi_cells_anchor_scheme(Base_anchor_scheme):
    """
    A Voronoi cell-based anchor scheme.
    """
    type: typing.Literal["voronoi_cells"] = "voronoi_cells"
    voronoi_cells: list[Voronoi_cell] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(Voronoi_cell),
            iterable_validator=validators.instance_of(list),
        )
    )

    def make_anchors(
            self, 
            cvs: list[cv_base.Base_CV_input],
            starting_anchor_index: int,
            starting_milestone_index: int,
            connection_flag_dict: dict[int|str, list[structures.Anchor]], 
            associated_input_anchor: dict[int, Voronoi_cell]
            ) -> typing.Tuple[typing.List[structures.Anchor], list[tuple[int, int]], int]:
        """
        For this partition, create the anchors that are defined
        herein.
        """
        anchors = []
        need_voronoi_connections = False
        if len(cvs) == 0:
            raise Exception("CVs must be defined within a Partition.")
        
        anchor_connections: list[tuple[int, int]] = []
        for alpha, voronoi_cell in enumerate(self.voronoi_cells):
            anchor: structures.Anchor = structures.Anchor()
            anchor.index = starting_anchor_index + alpha
            anchor.directory = f"anchor_{anchor.index:04d}"
            if voronoi_cell.name == "":
                anchor.name = anchor.directory
            else:
                anchor.name = voronoi_cell.name

            if not voronoi_cell.bulk_anchor:
                # TODO: find a smarter way to assign scale types - they 
                #  might or might not be MD or BD.
                anchor.scale_types.append("molecular_dynamics")
                if voronoi_cell.end_anchor:
                    anchor.end_anchor = voronoi_cell.end_anchor
                anchor.bulk_anchor = False
            else:
                anchor.name = "bulk"
                anchor.end_anchor = True
                anchor.bulk_anchor = True

            associated_input_anchor[anchor.index] = voronoi_cell
            cv_indices = []
            for j, cv_input in enumerate(cvs):
                variable_name = f"value_{cv_input._index}"
                variable_value = voronoi_cell.get_variable_value(j)
                anchor.variables[variable_name] = variable_value
                cv_indices.append(cv_input._index)

            if len(cvs) == 1:
                for connection_flag in voronoi_cell.connection_flags:
                    connection_flag_dict[connection_flag].append(anchor)
                if voronoi_cell.bulk_anchor:
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
            if voronoi_cell.positions is not None and voronoi_cell.positions != "":
                stage_parameter: dict[str, str | int | list] = {}
                stage_parameter["index"] = 0
                stage_parameter["directory"] = base.INITIAL_DIRECTORY
                #stage_parameter["md_parameters_topology"] = voronoi_cell.md_parameters_topology
                stage_parameter["positions"] = voronoi_cell.positions
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
            milestone_index = make_milestone_between_two_voronoi_cell_anchors(
                anchor_alpha, anchor_beta, milestone_index,
                cv_indices)
            
        return anchors, anchor_connections, milestone_index


