from __future__ import annotations
"""
modules/analyzers/metadynamics_analyzer.py

Analyze the results of a SEEKR metadynamics calculation.
"""

import os
import logging

from attrs import define, field, validators
import numpy as np

import seekr.modules.structures as structures

METADYNAMICS_IMAGE_DIR = "metadynamics"

@define
class Metadynamics_analyzer:
    """
    Analyze the results of a SEEKR metadynamics calculation.
    """
    model: structures.Seekr_model = field(
        validator=validators.instance_of(structures.Seekr_model))
    stage_index: int = field(
        validator=validators.instance_of(int),)
    directory: str = field(
        validator=validators.instance_of(str), default="")

    def print_results(self):
        """
        Print results of metadynamics analysis - does nothing yet.
        """
        return

    def get_statistics_message(
            self,
            model: structures.Seekr_model,
            anchor_index_instruction: str | None = None,
            use_json: bool = False
            ) -> str | dict:
        """
        Statistics instruction output for status.py. Does nothing yet.
        """
        final_message = "" if not use_json else {}
        return final_message

    def make_image_directory(
            self, 
            image_directory: str
            ) -> None:
        """
        Create the directory and sub-directories for images to be written.
        """
        if image_directory != "" and not os.path.exists(image_directory):
            os.mkdir(image_directory)
        
        if not os.path.exists(os.path.join(
                image_directory, METADYNAMICS_IMAGE_DIR)):
            os.mkdir(os.path.join(image_directory, METADYNAMICS_IMAGE_DIR))
    
    def plot_deltaG_landscape_1d(
            self,
            cv_index: int = 0,
            image_directory: str | None = None,
            return_plot=True
            ) -> tuple | None:
        """
        Plot the pi_alpha values for each anchor in 2D.
        """
        import matplotlib.pyplot as plt
        cv_0 = self.model.collective_variables[cv_index]
        boundaries = [cv_0.min_value, cv_0.max_value]
        cv_0_name = cv_0.name
        cv_0_unit = cv_0.unit
        x_coordinate_title = f"{cv_0_name} ({cv_0_unit})"
        dG_filename = os.path.join(self.directory, "dG_metaD.txt")
        if not os.path.exists(dG_filename):
            logging.info("dG_metaD.txt file not found in metadynamics directory.")
            return None
        dG = np.loadtxt(dG_filename)
        x = np.linspace(boundaries[0], boundaries[1], 
                        dG.shape[0])
        pi_fig, ax = plt.subplots()
        plt.plot(x, dG)
        plt.title("Metadynamics Free Energy Landscape")
        plt.xlabel(x_coordinate_title)
        plt.tight_layout()
        if return_plot:
            return fig, ax
        else:
            filename = os.path.join(image_directory, f"dG_metaD_stage_{self.stage_index}.png")
            plt.savefig(filename)
            plt.close()
            return

    def plot_deltaG_landscape_2d(
            self,
            cv_indices: tuple[int, int] = (0, 1),
            image_directory: str | None = None,
            return_plot=True
            ) -> tuple | None:
        """
        Plot the pi_alpha values for each anchor in 2D.
        """
        import matplotlib.pyplot as plt
        cv_0 = self.model.collective_variables[cv_indices[0]]
        cv_1 = self.model.collective_variables[cv_indices[1]]
        boundaries = [cv_0.min_value, cv_0.max_value, cv_1.min_value, 
            cv_1.max_value]
        cv_0_name = cv_0.name
        cv_0_unit = cv_0.unit
        x_coordinate_title = f"{cv_0_name} ({cv_0_unit})"
        cv_1_name = cv_1.name
        cv_1_unit = cv_1.unit
        y_coordinate_title = f"{cv_1_name} ({cv_1_unit})"
        
        dG_filename = os.path.join(self.directory, "dG_metaD.txt")
        if not os.path.exists(dG_filename):
            logging.info("dG_metaD.txt file not found in metadynamics directory.")
            return None
        dG = np.loadtxt(dG_filename)
        p = plt.imshow(
            dG, extent=boundaries, vmax=np.max(dG), cmap=plt.cm.jet, origin="lower")
        plt.title("Metadynamics Free Energy Landscape")
        cbar = plt.colorbar(p)
        cbar.set_label("Energy (kcal/mol)")
        plt.xlabel(x_coordinate_title)
        plt.ylabel(y_coordinate_title)
        plt.tight_layout()
        if return_plot:
            return fig, ax
        else:
            filename = os.path.join(image_directory, f"dG_metaD_stage_{self.stage_index}.png")
            plt.savefig(filename)
            plt.close()
            return

    def save_plots(
            self, 
            image_directory: str
            ) -> None:
        """
        Save a potentially useful series of plots of some quantities
        obtained during the analysis.
        """
        #print("All plots being saved to:", analyzers_base.DEFAULT_IMAGE_DIR)
        stage = self.model.stages[self.stage_index-1]
        metad_image_dir = os.path.join(image_directory, METADYNAMICS_IMAGE_DIR)
        if len(stage.sampling_method.cv_names) == 1:
            self.plot_deltaG_landscape_1d(
                image_directory=metad_image_dir, return_plot=False)

        elif len(stage.sampling_method.cv_names) > 1:
            self.plot_deltaG_landscape_2d(
                image_directory=metad_image_dir, return_plot=False)

def analyze_metadynamics(
        model: structures.Seekr_model,
        stage_index: int,
        skip_checks: bool = False,
        min_time_ps: float = 0.0,
        max_time_ps: float | None = None,
        error_analysis_method: str = "bootstrap",
        only_extract_data: bool = False,
        num_error_samples: int | None = None
        ) -> Metadynamics_analyzer:
    """
    Analyze the results of a SEEKR MMVT calculation.
    
    Parameters
    ----------
    model : structures.Seekr_model
        The SEEKR model to analyze.
    
    Returns
    -------
    Metadynamics_analyzer
        An instance of Metadynamics_analyzer containing the analysis results.
    """
    stage = model.stages[stage_index-1]
    if stage.scope == "unpartitioned":
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        unpartitioned = md_engine.unpartitioned_settings
        unpartitioned_dir = unpartitioned.directory
        next_stage_params = unpartitioned\
            .get_stage_parameters_by_index(stage_index)
        stage_directory = next_stage_params["directory"]
    elif stage.scope == "partitioned":
        raise ValueError("Metadynamics does not support partitioned stages.")
    running_directory = os.path.join(model.directory, unpartitioned_dir, 
        stage_directory)
    analysis = Metadynamics_analyzer(model=model, stage_index=stage_index,
        directory=running_directory)
    return analysis