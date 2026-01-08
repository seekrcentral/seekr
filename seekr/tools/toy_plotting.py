"""
tools/toy_plotting.py

Utilities to plot toy systems.
"""

import os
import re
import glob
import argparse
from math import *

import numpy as np
import numpy.typing as nptyping
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mdtraj

import seekr.modules.structures as structures
import seekr.modules.scales.molecular_dynamics.structures as md_structures
import seekr.run as run

def plot_voronoi_tesselation(
        model: structures.Seekr_model, 
        boundaries: nptyping.ArrayLike
        ) -> tuple[plt.Figure, plt.Axes, list[list[float]]]:
    points = []
    num_variables = len([varname for varname in model.anchors[0].variables \
                         if varname.startswith("value_")])
    for anchor in model.anchors:
        values = []
        for i in range(num_variables):
            var_name = "value_{}".format(i)
            values.append(anchor.variables[var_name])
        
        points.append(values)
    
    points.append([-100, -100])
    points.append([100, 100])
    
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor, line_width=2, show_vertices=False)
    ax = plt.gca()
    ax.axis(boundaries)
    ax.set_aspect('equal', adjustable='box')
    
    return fig, ax, points[:-2]

class Toy_plot():
    def __init__(
            self, 
            model: structures.Seekr_model, 
            stage: md_structures.Stage_md,
            title: str, 
            boundaries: nptyping.ArrayLike | None, 
            landscape_resolution: int = 100, 
            stride: int = 1, 
            voronoi: bool = False
            ) -> None:
        """
        Object for plotting toy SEEKR calculations.
        """
        self.num_steps_by_anchor = {}
        assert model.using_toy(), "Model must be a toy model."
        self.model = model
        self.title = title
        self.boundaries = boundaries
        if (self.boundaries == None).any():
            if stage.input_from == "anchors":
                anchor_indices = [i for i in range(model.num_anchors)]
            elif stage.input_from == "systemwide":
                anchor_indices = None
            _, auto_boundaries = self.load_trajs(stage, anchor_indices)
            self.boundaries = auto_boundaries

        self.landscape_resolution = landscape_resolution
        self.voronoi = voronoi
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        self.function_str = self.convert_function_str(
            md_engine.toy_settings.potential_energy_expression)
        fig, ax = self.plot_energy_landscape()
        self.fig = fig
        self.ax = ax
        self.stride = stride
        return
    
    def convert_function_str(
            self, 
            old_function_str: str
            ) -> str:
        """
        Convert the energy landscape from a form used by OpenMM
        to a form used by Python.
        """
        new_function_str = re.sub("\^", "**", old_function_str)
        return new_function_str

    def plot_energy_landscape(
            self, 
            max_Z: int = 20
            ) -> tuple[plt.Figure, plt.Axes]:
        """
        Create the plot and plot the energy landscape.
        """
        z1 = 0.0
        min_x = self.boundaries[0,0]
        max_x = self.boundaries[0,1]
        min_y = self.boundaries[1,0]
        max_y = self.boundaries[1,1]
        bounds_x = np.linspace(min_x, max_x, self.landscape_resolution)
        bounds_y = np.linspace(min_y, max_y, self.landscape_resolution)
        X,Y = np.meshgrid(bounds_x, bounds_y)
        Z = np.zeros((self.landscape_resolution, self.landscape_resolution))
        for i, x1 in enumerate(bounds_x):
            for j, y1 in enumerate(bounds_y):
                # fill out landscape here
                Z[j,i] = eval(self.function_str)
                if Z[j,i] > max_Z:
                    Z[j,i] = max_Z
        
        if self.voronoi:
            voronoi_boundaries = [min_x, max_x, min_y, max_y]
            fig, axs, points = plot_voronoi_tesselation(self.model, voronoi_boundaries)
        else:
            fig, axs = plt.subplots(nrows=1, ncols=1)
        p = axs.pcolor(X, Y, Z, cmap=plt.cm.jet, vmin=Z.min(), 
                          vmax=Z.max())
        axs.set_title(self.title)
        axs.set_xlabel("$x_{1}$ (nm)")
        axs.set_ylabel("$y_{1}$ (nm)")
        cbar = plt.colorbar(p)
        cbar.set_label("Energy (kcal/mol)")
        return fig, axs
    
    def load_trajs(
            self, 
            stage: md_structures.Stage_md, 
            anchor_indices: None | list[int],
            swarm_id_wanted: int | None = None
            ) -> list[list[nptyping.ArrayLike]]:
        """
        Load the trajectory files of the toy system and return arrays of 
        its positions over time.
        """
        rootdir = self.model.directory
        stage_id = stage.index
        positions_list = []
        minx = 9e9
        maxx = -9e9
        miny = 9e9
        maxy = -9e9
        md_engine = self.model.get_engine_by_scale_type("molecular_dynamics")
        systemwide_object = md_engine.systemwide_settings
        if anchor_indices is None:
            anchor_systemwide_objects = [md_engine.systemwide_settings]
            
        else:
            anchor_systemwide_objects = [self.model.anchors[i] for i in anchor_indices]

        for anchor_index, anchor_or_systemwide in enumerate(anchor_systemwide_objects):
            if anchor_or_systemwide.stage_parameters[stage_id] is None:
                pdb_path = os.path.join(
                    rootdir, systemwide_object.directory, 
                    systemwide_object.stage_parameters[stage_id]["directory"], "toy.pdb")
            else:
                pdb_path = os.path.join(
                    rootdir, anchor_or_systemwide.directory, 
                    anchor_or_systemwide.stage_parameters[stage_id]["directory"], "toy.pdb")
            swarm_dcds_glob = os.path.join(
                rootdir, anchor_or_systemwide.directory, 
                anchor_or_systemwide.stage_parameters[stage_id]["directory"], 
                "*1.swarm_*.reporter_*.dcd")
            swarm_dcds = glob.glob(swarm_dcds_glob)
            swarm_ids = []
            for swarm_dcd in swarm_dcds:
                match = re.search(r"swarm_(\d+)", swarm_dcd)
                if match:
                    swarm_id = int(match.group(1))
                    if swarm_id_wanted is not None and swarm_id != swarm_id_wanted:
                        continue
                    swarm_ids.append(swarm_id)
            swarm_ids.sort()
            positions_frame_list = []
            num_steps = 0
            for swarm_id in swarm_ids:
                dcd_file_glob = "*.swarm_{}.reporter_*.dcd".format(swarm_id)
                dcd_glob = os.path.join(
                    rootdir, anchor_or_systemwide.directory, 
                    anchor_or_systemwide.stage_parameters[stage_id]["directory"], 
                    dcd_file_glob)
                dcd_files = glob.glob(dcd_glob)
                if len(dcd_files) == 0:
                    continue
                # TODO: sort the dcd files
                traj = mdtraj.load(dcd_files, top=pdb_path)
                num_steps += traj.n_frames
                positions_frame_list.append(traj.xyz)
                if np.min(traj.xyz[:,:,0]) < minx:
                    minx = np.min(traj.xyz[:,:,0])
                if np.max(traj.xyz[:,:,0]) > maxx:
                    maxx = np.max(traj.xyz[:,:,0])
                if np.min(traj.xyz[:,:,1]) < miny:
                    miny = np.min(traj.xyz[:,:,1])
                if np.max(traj.xyz[:,:,1]) > maxy:
                    maxy = np.max(traj.xyz[:,:,1])
                
                
            self.num_steps_by_anchor[anchor_index] = num_steps
            positions_list.append(positions_frame_list)
        
        new_boundaries = np.array([[minx, maxx], [miny, maxy]])
        return positions_list, new_boundaries
    
    def traj_walk(
            self, 
            positions_list: list[list[nptyping.ArrayLike]]
            ) -> list[nptyping.ArrayLike]:
        """
        Create the 'walk' that will be plotted - a series of points on 
        the 2D plot that will be animated
        """
        walks = []
        for positions_anchor in positions_list:
            for positions_swarm in positions_anchor:
                walk = []
                for positions_frame in positions_swarm:
                    walk.append(positions_frame[0,0:2])
                    
                walks.append(np.array(walk))
        
        return walks
    
    def update_lines(
            self, 
            num: int, 
            walks: list[nptyping.ArrayLike], 
            lines: list[plt.Line2D]
            ):
        """
        This function is called every step of the animation to determine
        where to draw the animated points and lines.
        """
        
        index = self.stride * num
        for i, (walk, line) in enumerate(zip(walks,lines)):
            if len(walk) == 0:
                continue
            circle1 = self.circle1_list[i]
            circle2 = self.circle2_list[i]
            if index < len(walk):
                line.set_data(walk[:index+1, 0], walk[:index+1, 1])
                circle1.center=(walk[index, 0], walk[index, 1])
                circle2.center=(walk[index, 0], walk[index, 1])
            else:
                line.set_data(walk[:, 0], walk[:, 1])
                circle1.center=(walk[-1, 0], walk[-1, 1])
                circle2.center=(walk[-1, 0], walk[-1, 1])
        
        # NOTE: time string taken out, because the md_trajectory_output_interval
        #  is no longer included in the model. Instead, maybe find the DCD reporter?
        #time_str = "{:.2f} ps".format((self.stride + index) * self.model.get_md_timestep() * self.model.stages[0].md_trajectory_output_interval)
        #self.frame_text.set_text(time_str)
        
        return lines
    
    def make_graphical_objects(
            self, 
            num_walks: int
            ) -> list[plt.Line2D]:
        """
        Create the circles and lines - graphical objects shown in the
        animation.
        """
        self.circle1_list = []
        self.circle2_list = []
        font = {"weight":"bold"}
        self.frame_text = plt.text(-1.4, 1.9, "0 ns", fontdict=font)
        lines = []
        for i in range(num_walks):
            circle1 = plt.Circle((0,0), 0.05, color="r", zorder=2.5)
            circle2 = plt.Circle((0,0), 0.025, color="k", zorder=2.5)
            self.ax.add_patch(circle1)
            self.ax.add_patch(circle2)
            self.circle1_list.append(circle1)
            self.circle2_list.append(circle2)
            line, = self.ax.plot([], [], lw=1, color="w")
            lines.append(line)
        return lines
    
    def animate_trajs(
            self, 
            stage: md_structures.Stage_md,
            animating_anchor_indices: None | list[int],
            swarm_id: int | None = None
            ) -> animation.FuncAnimation:
        """
        Animate the trajectories to view how the system progresses along
        the landscape.
        """
        positions_list, _ = self.load_trajs(stage, animating_anchor_indices, swarm_id)
        walks = self.traj_walk(positions_list)
        lines = self.make_graphical_objects(len(walks))
        
        #num_frames = self.num_steps // self.stride
        assert len(self.num_steps_by_anchor.values()) > 0, "No trajectories found in this stage."
        num_frames = max(self.num_steps_by_anchor.values()) // self.stride
        ani = animation.FuncAnimation(self.fig, self.update_lines, fargs=(walks, lines), frames=num_frames, interval=60, repeat=False)
        return ani
    
def draw_linear_milestones(
        toy_plot: Toy_plot, 
        milestone_cv_functions: list[str], 
        anchors: list[structures.Anchor] | None = None
        ) -> None:
    """
    Create a series of milestones linear in shape.
    """
    model = toy_plot.model
    #x1 = toy_plot.boundaries[0,0] * 0.1
    #x2 = toy_plot.boundaries[0,1] * 0.1
    x1_ref = 0
    x2_ref = 0.1
    y1_ref = 0
    y2_ref = 0.1
    values = []
    cv_indices = []
    for anchor in model.anchors:
        if anchors is not None and anchor.index not in anchors:
            continue
        for milestone in anchor.milestones:
            value = milestone.variables["avg_value"]
            if len(values) > 0:
                if value == values[-1] and 0 == cv_indices[-1]:
                    continue
            values.append(value)
            #cv_indices.append(milestone.cv_index)
            cv_indices.append(0)
    
    for value, cv_index in zip(values, cv_indices):
        milestone_function = milestone_cv_functions[cv_index]
        if "x=" in milestone_function:
            milestone_function = milestone_function.strip("x=")
            y = y1_ref
            x1 = eval(milestone_function)
            y = y2_ref
            x2 = eval(milestone_function)
            plt.axline((x1, y1_ref), (x2, y2_ref), color="k", lw=2)
        else:
            x = x1_ref
            y1 = eval(milestone_function)
            x = x2_ref
            y2 = eval(milestone_function)
            plt.axline((x1_ref, y1), (x2_ref, y2), color="k", lw=2)
        
    return

def toy_plotting(
        model: structures.Seekr_model, 
        stage_arg: str, 
        title: str = "Toy System", 
        minx: float | None = None, 
        maxx: float | None = None, 
        miny: float | None = None, 
        maxy: float | None = None, 
        animating_anchor_indices: list[int] | None = None,
        traj_stride: int = 1,
        swarm_id: int | None = None,
        ) -> None:
    """
    Main function to run the toy plotting.
    """
    stage_index = run.extract_stage_index(model, stage_arg)
    stage = model.stages[stage_index]
    if animating_anchor_indices is None:
        if stage.input_from == "systemwide":
            animating_anchor_indices = None
        elif stage.input_from == "anchors":
            animating_anchor_indices = [i for i in range(model.num_anchors)]
        else:
            raise Exception(f"Invalid option for stage.input_from: {stage.input_from}.")
    boundaries = np.array([[minx, maxx], [miny, maxy]])
    if len(model.collective_variables) == 1:
        voronoi = False
    elif len(model.collective_variables) == 2:
        voronoi = True
    else:
        raise Exception("Toy plotting is only available for 1D and 2D systems.")
    toy_plot = Toy_plot(model, stage, title, boundaries, stride=traj_stride,
                        voronoi=voronoi)
    if not voronoi:
        if model.collective_variables[0].expression == "y1":
            milestone_function_list = ["value"]
        elif model.collective_variables[0].expression == "x1":
            milestone_function_list = ["x=value"]
        else:
            raise Exception("1D CV functions including both x1 and y1 not yet implemented.")
        draw_linear_milestones(toy_plot, ["value"])
    ani = toy_plot.animate_trajs(stage, animating_anchor_indices, swarm_id)
    plt.show()
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of the model file for SEEKR calculation. This would "\
        "be the JSON file generated in the prepare stage.")
    argparser.add_argument(
        "stage", metavar="STAGE", type=str,
        help="The name or index of the stage to run.")
    argparser.add_argument(
        "-S", "--stride", dest="stride", type=int, default=1,
        help="The stride to use for trajectories. Default: 1.")
    argparser.add_argument(
        "-t", "--title", dest="title", type=str, default="Toy System",
        help="The title of the plot. Default: 'Toy System'.")
    argparser.add_argument(
        "-x", "--minx", dest="minx", type=float, default=None,
        help="The minimum X value to plot. If minx, maxx, miny, or maxy is left "
        "at None, it will be chosen from the DCD trajectories within the "
        "simulation stage. Default: None.")
    argparser.add_argument(
        "-X", "--maxx", dest="maxx", type=float, default=None,
        help="The maximum X value to plot. If minx, maxx, miny, or maxy is left "
        "at None, it will be chosen from the DCD trajectories within the "
        "simulation stage. Default: None.")
    argparser.add_argument(
        "-y", "--miny", dest="miny", type=float, default=None,
        help="The minimum Y value to plot. If minx, maxx, miny, or maxy is left "
        "at None, it will be chosen from the DCD trajectories within the "
        "simulation stage. Default: None.")
    argparser.add_argument(
        "-Y", "--maxy", dest="maxy", type=float, default=None,
        help="The maximum Y value to plot. If minx, maxx, miny, or maxy is left "
        "at None, it will be chosen from the DCD trajectories within the "
        "simulation stage. Default: None.")
    argparser.add_argument(
        "-a", "--animating_anchor_indices", dest="animating_anchor_indices", type=str, 
        default="all", help="A comma-separated list of anchor indices whose "
        "trajectories will be plotted. If 'all' is entered, then all anchors with "
        "existing trajectories will be plotted. Default: 'all'.")
    argparser.add_argument(
        "-s", "--swarm_id", dest="swarm_id", type=int, default=None,
        help="The swarm ID to run. This is only relevant for stages that "\
            "use swarm sampling methods. If not specified, all swarm IDs "\
            "that need to be run will be run. Default: None.",)
    
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    stage_arg = args["stage"]
    stride = args["stride"]
    title = args["title"]
    minx = args["minx"]
    maxx = args["maxx"]
    miny = args["miny"]
    maxy = args["maxy"]
    animating_anchor_indices = args["animating_anchor_indices"]
    swarm_id = args["swarm_id"]

    if animating_anchor_indices == "all":
        animating_anchor_indices = None
    else:
        animating_anchor_indices = [int(i) for i in animating_anchor_indices.split(",")]
    model = structures.load_model(model_file)
    toy_plotting(model, stage_arg, title, minx, maxx, miny, maxy, animating_anchor_indices, 
                 traj_stride=stride, swarm_id=swarm_id)