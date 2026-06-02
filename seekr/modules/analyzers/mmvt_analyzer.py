from __future__ import annotations
"""
modules/analyzers/mmvt_analyzer.py

Analyze the results of a SEEKR MMVT calculation.
"""

import os
import math

from attrs import define, field, validators
import numpy as np
import matplotlib.pyplot as plt

import seekr.modules.analyzers.base as analyzers_base
import seekr.modules.structures as structures
import seekr.modules.analyzers.bulk_anchor_statistics as bulk_anchor_statistics
import seekr.modules.scales.mmvt_analyzer as scales_mmvt_analyzer
import seekr.modules.analyzers.mmvt_anchor_statistics as mmvt_anchor_statistics
import seekr.modules.analyzers.mmvt_data_window as mmvt_data_window
import seekr.modules.analyzers.mmvt_error_analysis as mmvt_error_analysis
import seekr.modules.analyzers.mmvt_convergence as mmvt_convergence

@define
class Mmvt_analyzer:
    """
    Analyze the results of a SEEKR MMVT calculation.
    """
    model: structures.Seekr_model = field(
        validator=validators.instance_of(structures.Seekr_model))
    stage_index: int = field(
        validator=validators.instance_of(int),)
    anchor_stats_alpha_list: list = field(
        factory=list,
        validator=validators.instance_of(list),)
    bulk_stats: bulk_anchor_statistics.Bulk_anchor_statistics | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(
            bulk_anchor_statistics.Bulk_anchor_statistics)),)
    window_index_list: list[int] = field(
        factory=list,
        validator=validators.instance_of(list),)
    main_data_window: mmvt_data_window.Mmvt_data_window | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(
            mmvt_data_window.Mmvt_data_window)),)
    data_window_list: list[mmvt_data_window.Mmvt_data_window] = field(
        factory=list,
        validator=validators.instance_of(list),)
    
    def extract_data(
            self,
            min_time_ps: float,
            max_time_ps: float | None
            ) -> None:
        """
        Extract data from the SEEKR MMVT simulation output.
        """
        # If possible, avoid expensive I/O
        if len(self.anchor_stats_alpha_list) > 0:
            # Files have already been read, and anchor_stats_alpha_list is populated.
            return
        
        time_between_statistics = None
        set_name = self.model.stages[self.stage_index-1].set_name
        for anchor in self.model.anchors:
            anchor_stats_alpha, tmp_time_between_statistics \
                = scales_mmvt_analyzer.make_anchor_stats_alpha(
                    self.model, anchor, self.stage_index,
                    min_time_ps, max_time_ps,
                    time_between_statistics)
            if tmp_time_between_statistics is not None:
                time_between_statistics = tmp_time_between_statistics
            self.anchor_stats_alpha_list.append(anchor_stats_alpha)
            if anchor.bulk_anchor:
                bulk_stats = scales_mmvt_analyzer.make_bulk_anchor_stats(
                    self.model, anchor, set_name)
                if bulk_stats is not None:
                    assert self.bulk_stats is None, \
                        "Only one anchor may have bulk statistics (Like from BD)."
                self.bulk_stats = bulk_stats

        return time_between_statistics

    def fill_out_and_process_data_samples(
            self,
            time_between_statistics: float,
            max_convergence_windows=analyzers_base.DEFAULT_MAX_CONVERGENCE_WINDOWS
            ) -> None:
        """
        Create and fill out the data samples for MMVT analysis.
        """
        # TODO: remake the anchor_stats_alpha_list based
        #  on error sampling method.
        # Choose the length of the longest sublist within the anchor_stats_alpha_list
        max_windows_this_stats = max(len(stats) for stats in self.anchor_stats_alpha_list \
                                     if stats is not None)
        # choose a stride, such that the number of windows is evenly spaced, but
        # make sure that it is at least 1
        window_stride = max(1, max_windows_this_stats \
                            // max_convergence_windows)
        self.data_window_list = []
        self.window_index_list = []
        for window_index in range(
                window_stride-1, max_windows_this_stats, window_stride):
            #window_time = window_index * time_between_statistics
            data_window = mmvt_data_window.Mmvt_data_window()
            #data_window.time = window_time
            self.data_window_list.append(data_window)
            self.window_index_list.append(window_index)
        
        mmvt_anchor_statistics.pare_down_anchor_statistics(
            self.anchor_stats_alpha_list, self.window_index_list)
        mmvt_data_window.assign_data_windows_sample(
            self.model, self.anchor_stats_alpha_list, self.bulk_stats,
            self.window_index_list, self.data_window_list, assign_to_main=True)
        self.main_data_window = self.data_window_list[-1]

    def error_analysis(
            self,
            method: str = "bootstrap",
            num_error_samples: int | None = None,
            random_seed: int | None = None,
            ) -> None:
        """
        Perform error analysis on the extracted data.
        
        This method should be implemented to calculate errors for the various
        fields in the Mmvt_analyzer instance.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        if method == "bootstrap":
            piecewise_anchors_statistics = mmvt_anchor_statistics\
                .make_piecewise_anchor_statistics(
                    self.anchor_stats_alpha_list)
            num_anchors = len(piecewise_anchors_statistics)
            if num_error_samples is None:
                num_error_samples = analyzers_base.NUM_BOOTSTRAP_SAMPLES
            for _ in range(num_error_samples):
                bootstrap_anchors_statistics_piecewise = []
                for alpha in range(num_anchors):
                    if piecewise_anchors_statistics[alpha] is None:
                        bootstrap_anchors_statistics_piecewise.append(None)
                        continue
                    N = len(piecewise_anchors_statistics[alpha])
                    indices = np.random.choice(np.arange(N), size=N, replace=True)
                    bootstrap_anchor_statistics_piecewise \
                        = [piecewise_anchors_statistics[alpha][i] for i in indices]
                    bootstrap_anchors_statistics_piecewise.append(
                        bootstrap_anchor_statistics_piecewise)
                
                bootstrap_anchors_statistics_cumulative \
                    = mmvt_anchor_statistics.make_cumulative_anchor_statistics(
                        bootstrap_anchors_statistics_piecewise)
                
                if self.bulk_stats is not None:
                    resampled_bulk_stats = bulk_anchor_statistics\
                        .resample_bulk_anchor_statistics(self.bulk_stats)
                else:
                    resampled_bulk_stats = None
                mmvt_data_window.assign_data_windows_sample(
                    self.model, bootstrap_anchors_statistics_cumulative, 
                    resampled_bulk_stats, self.window_index_list, 
                    self.data_window_list, assign_to_main=False)
        elif method == "resample":
            if num_error_samples is None:
                num_error_samples = analyzers_base.NUM_RESAMPLES
            for _ in range(num_error_samples):
                resampled_anchors_statistics_piecewise = mmvt_error_analysis\
                    .resample_anchor_statistics(
                        self.anchor_stats_alpha_list)
                resampled_anchors_statistics_cumulative \
                    =   mmvt_anchor_statistics.make_cumulative_anchor_statistics(
                        resampled_anchors_statistics_piecewise)
                if self.bulk_stats is not None:
                    resampled_bulk_stats = bulk_anchor_statistics\
                        .resample_bulk_anchor_statistics(self.bulk_stats)
                else:
                    resampled_bulk_stats = None
                mmvt_data_window.assign_data_windows_sample(
                    self.model, resampled_anchors_statistics_cumulative, 
                    resampled_bulk_stats, self.window_index_list, 
                    self.data_window_list, assign_to_main=False)
        elif method == "montecarlo":
            if self.bulk_stats is not None:
                resampled_bulk_stats = bulk_anchor_statistics\
                        .resample_bulk_anchor_statistics(self.bulk_stats)
            else:
                resampled_bulk_stats = None
            if num_error_samples is None:
                num_error_samples = analyzers_base.NUM_MONTE_CARLO_SAMPLES
            mmvt_error_analysis.monte_carlo_data_windows(
                self.model, self.window_index_list, self.data_window_list,
                num_error_samples)
        elif method is None:
            pass
        else:
            raise Exception(f"Unsupported error analysis method: {method}")

        for data_window in self.data_window_list:
            data_window.fill_out_errors_from_data_samples()

    def print_results(self):
        """Print all results of the analysis calculation."""
        print("Printing results from MMVT SEEKR calculation")
        if self.main_data_window.k_off is not None:
            print("k_off (1/s):", analyzers_base.pretty_string_value_error(
                self.main_data_window.k_off, self.main_data_window.k_off_err))
        if len(self.main_data_window.k_ons) > 0:
            print("k_ons :")
        for key in self.main_data_window.k_ons:
            k_on = float(self.main_data_window.k_ons[key])
            if k_on > 0.0:
                diss_constant = self.main_data_window.k_off / k_on
            else:
                diss_constant = float("inf")
            delta_G = analyzers_base.GAS_CONSTANT*self.model.temperature\
                *math.log(diss_constant)
            if key in self.main_data_window.k_ons_err:
                k_on_err = float(self.main_data_window.k_ons_err[key])
                print("  k_on (1/s * 1/M) to state", key, ":", 
                      analyzers_base.pretty_string_value_error(k_on, k_on_err))
                if k_on > 0.0 and self.main_data_window.k_off > 0.0:
                    diss_constant_err = diss_constant \
                        * analyzers_base.quadriture(
                        k_on_err/k_on, 
                        self.main_data_window.k_off_err/self.main_data_window.k_off)
                    delta_G_err = diss_constant_err*analyzers_base.GAS_CONSTANT\
                    *self.model.temperature/diss_constant
                else:
                    diss_constant_err = None
                    delta_G_err = None
            else:
                print("  k_on (1/s * 1/M) to state", key, ":", 
                      analyzers_base.pretty_string_value_error(k_on, None))
                diss_constant_err = None
                delta_G_err = None
            
            print("  Dissociation constant (M) to state", key, ":", 
                  analyzers_base.pretty_string_value_error(
                      diss_constant, diss_constant_err))
            if key in self.main_data_window.k_ons_err:
                print("  \u0394G (kcal/mol) to state", key, ":", 
                      analyzers_base.pretty_string_value_error(
                          delta_G, delta_G_err))
        
        print("Mean first passage times (s):")
        for key in self.main_data_window.anchor_to_anchor_MFPTs:
            state1 = key[0]
            state2 = key[1]
            if key in self.main_data_window.anchor_to_anchor_MFPTs_err:
                print("  MFPT from", state1, "to", state2, ":",
                      analyzers_base.pretty_string_value_error(
                          float(self.main_data_window.\
                                anchor_to_anchor_MFPTs[key]*1.0e-12), 
                          float(self.main_data_window.\
                                anchor_to_anchor_MFPTs_err[key]*1.0e-12)))
            else:
                print("  MFPT from", state1, "to", state2, ":",
                      analyzers_base.pretty_string_value_error(
                          float(self.main_data_window.\
                                anchor_to_anchor_MFPTs[key]*1.0e-12), None))

        if len(self.main_data_window.anchor_to_anchor_energies) > 0:
            print("Free energy differences (kcal/mol):")
            for key in self.main_data_window.anchor_to_anchor_energies:
                state1 = key[0]
                state2 = key[1]
                if key in self.main_data_window.anchor_to_anchor_energies_err:
                    print("  Energy from", state1, "to", state2, ":",
                          analyzers_base.pretty_string_value_error(
                              float(self.main_data_window\
                                    .anchor_to_anchor_energies[key]), 
                              float(self.main_data_window\
                                    .anchor_to_anchor_energies_err[key])))
                else:
                    print("  Energy from", state1, "to", state2, ":",
                          analyzers_base.pretty_string_value_error(
                              float(self.main_data_window\
                                    .anchor_to_anchor_energies[key]), None))
        return
    
    def get_statistics_message(
            self,
            model: structures.Seekr_model,
            anchor_index_instruction: str | None = None,
            use_json: bool = False
            ) -> str | dict:
        """

        """
        assert anchor_index_instruction is not None, \
            "MMVT may not use 'unpartitioned' statistics."
        final_message = "" if not use_json else {}
        for alpha, anchor in enumerate(model.anchors):
            if anchor_index_instruction != "any" \
                    and alpha != anchor_index_instruction:
                continue
            anchor_stats_anchor = self.anchor_stats_alpha_list[alpha]
            if anchor_stats_anchor is None:
                continue
            if len(anchor_stats_anchor) == 0:
                anchor_stats = mmvt_anchor_statistics.Mmvt_anchor_statistics(
                    alpha)
            else:
                anchor_stats = anchor_stats_anchor[-1]
            message = anchor_stats.get_statistics_message(
                model, alpha=alpha, use_json=use_json)
            if use_json:
                final_message[alpha] = message
            else:
                final_message += f"  Anchor {alpha}:\n{message}"

        if self.bulk_stats is not None:
            message = self.bulk_stats.get_statistics_message(use_json=use_json)
            if use_json:
                final_message["bulk"] = message
            else:
                final_message += "  Bulk anchor statistics:\n" + message

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
                image_directory, analyzers_base.N_ALPHA_BETA_DIR)):
            os.mkdir(os.path.join(image_directory, analyzers_base.N_ALPHA_BETA_DIR))
            
        if not os.path.exists(os.path.join(
                image_directory, analyzers_base.T_ALPHA_DIR)):
            os.mkdir(os.path.join(image_directory, analyzers_base.T_ALPHA_DIR))

        if not os.path.exists(os.path.join(
                image_directory, analyzers_base.K_ALPHA_BETA_DIR)):
            os.mkdir(os.path.join(image_directory, analyzers_base.K_ALPHA_BETA_DIR))

        if not os.path.exists(os.path.join(
                image_directory, analyzers_base.N_IJ_ALPHA_DIR)):
            os.mkdir(os.path.join(image_directory, analyzers_base.N_IJ_ALPHA_DIR))

        if not os.path.exists(
                os.path.join(image_directory, analyzers_base.R_I_ALPHA_DIR)):
            os.mkdir(os.path.join(image_directory, analyzers_base.R_I_ALPHA_DIR))

        if not os.path.exists(
                os.path.join(image_directory, analyzers_base.N_IJ_DIR)):
            os.mkdir(os.path.join(image_directory, analyzers_base.N_IJ_DIR))

        if not os.path.exists(
                os.path.join(image_directory, analyzers_base.R_I_DIR)):
            os.mkdir(os.path.join(image_directory, analyzers_base.R_I_DIR))
        
        return

    def plot_pi_alpha_1d(
            self,
            image_directory: str | None = None,
            return_plot=True
            ) -> tuple | None:
        """
        Plot the pi_alpha values for each anchor.
        """
        anchor_indices = np.zeros(len(self.model.anchors), dtype=np.int8)
        anchor_values = np.zeros(len(self.model.anchors), dtype=np.float64)
        for i, anchor in enumerate(self.model.anchors):
            anchor_indices[i] = anchor.index
            anchor_values[i] = list(anchor.variables.values())[0]

        if self.main_data_window.pi_alpha is None:
            return
        pi_fig, ax = plt.subplots()
        plt.errorbar(
            anchor_values, 
            self.main_data_window.pi_alpha.flatten()[0:len(anchor_indices)],
            yerr=self.main_data_window.pi_alpha_err.flatten()[0:len(anchor_indices)], 
            ecolor="k", capsize=2)
        plt.title("$\\pi_{\\alpha}$")
        plt.xticks(anchor_values, anchor_values, rotation=90)
        plt.ylabel("$\u03C0_{\u03B1}$")
        plt.xlabel("anchor value")
        plt.yscale("log", nonpositive="mask")
        plt.tight_layout()
        if return_plot:
            return pi_fig, ax
        else:
            pi_fig.savefig(os.path.join(image_directory, "pi_alpha.png"))
            plt.close(pi_fig)
            return
        
    def plot_pi_alpha_2d(
            self,
            cv_indices: tuple[int, int] = (0, 1),
            image_directory: str | None = None,
            return_plot=True
            ) -> tuple | None:
        """
        Plot the pi_alpha values for each anchor in 2D.
        """
        anchor_values = analyzers_base.find_anchor_points_2d(self.model)
        boundaries = analyzers_base.make_boundaries(anchor_values)
        cv_0_name = self.model.collective_variables[cv_indices[0]].name
        cv_0_unit = self.model.collective_variables[cv_indices[0]].unit
        x_coordinate_title = f"{cv_0_name} ({cv_0_unit})"
        cv_1_name = self.model.collective_variables[cv_indices[1]].name
        cv_1_unit = self.model.collective_variables[cv_indices[1]].unit
        y_coordinate_title = f"{cv_1_name} ({cv_1_unit})"
        if self.main_data_window.pi_alpha is None:
            return
        anchor_pi_alpha = {}
        for alpha, fill_value in enumerate(self.main_data_window.pi_alpha):
            if not np.isfinite(fill_value):
                anchor_pi_alpha[alpha] = np.max(
                    self.main_data_window.pi_alpha[np.isfinite(
                        self.main_data_window.pi_alpha)])
            else:
                anchor_pi_alpha[alpha] = fill_value
        ret = analyzers_base.plot_anchors_2d(
            model=self.model, 
            anchor_values=anchor_values,
            boundaries=boundaries, 
            title="$\\pi_{\\alpha}$",
            x_coordinate_title=x_coordinate_title,
            y_coordinate_title=y_coordinate_title,
            colorbar_label="unitless",
            fill_values=anchor_pi_alpha,
            plot_basename="pi_alpha",
            plot_dir=image_directory, 
            return_plot=return_plot)
        return ret

    def plot_p_i_1d(
            self, 
            image_directory: str | None = None,
            return_plot=True
            ) -> tuple | None:
        """
        Plot the p_i values for each milestone.
        """
        if self.main_data_window.p_i is None:
            return
        
        milestone_indices = np.zeros(self.main_data_window.p_i.shape[0], 
                                     dtype=np.int8)
        milestone_values = np.zeros(self.main_data_window.p_i.shape[0], 
                                    dtype=np.float64)
        milestone_value_dict = analyzers_base.make_milestone_value_dict(
            self.model)
        for i in range(self.main_data_window.p_i.shape[0]):
            milestone_indices[i] = i
            milestone_values[i] = milestone_value_dict[i]
        
        # save p_i
        p_i_fig, ax = plt.subplots()
        plt.errorbar(
            np.round(milestone_values, 3), 
            self.main_data_window.p_i.flatten()[0:len(milestone_indices)], 
            yerr=self.main_data_window.p_i_err.flatten()[0:len(milestone_indices)], 
            ecolor="k", capsize=2)
        plt.title("$p_i$")
        plt.xticks(np.round(milestone_values, 3), np.round(milestone_values, 3),
                   rotation=90)
        plt.ylabel("$p_i$")
        plt.xlabel("milestone value")
        plt.yscale("log", nonpositive="mask")
        plt.tight_layout()
        if return_plot:
            return p_i_fig, ax
        else:
            p_i_fig.savefig(os.path.join(image_directory, "p_i.png"))
            plt.close(p_i_fig)
            return
        
    def plot_p_i_2d(
            self,
            cv_indices: tuple[int, int] = (0, 1),
            image_directory: str | None = None,
            return_plot=True
            ) -> tuple | None:
        """
        Plot the p_i values for each milestone in 2D.
        """
        anchor_values = analyzers_base.find_anchor_points_2d(self.model)
        boundaries = analyzers_base.make_boundaries(anchor_values)
        cv_0_name = self.model.collective_variables[cv_indices[0]].name
        cv_0_unit = self.model.collective_variables[cv_indices[0]].unit
        x_coordinate_title = f"{cv_0_name} ({cv_0_unit})"
        cv_1_name = self.model.collective_variables[cv_indices[1]].name
        cv_1_unit = self.model.collective_variables[cv_indices[1]].unit
        y_coordinate_title = f"{cv_1_name} ({cv_1_unit})"
        if self.main_data_window.p_i is None:
            return
        milestone_p_i = {}
        for i, fill_value in enumerate(self.main_data_window.p_i):
            if not np.isfinite(fill_value):
                milestone_p_i[i] = np.max(self.main_data_window.p_i[np.isfinite(
                    self.main_data_window.p_i)])
            else:
                milestone_p_i[i] = fill_value
        ret = analyzers_base.plot_milestones_2d(
            model=self.model, 
            boundaries=boundaries, 
            title="$p_i$",
            x_coordinate_title=x_coordinate_title,
            y_coordinate_title=y_coordinate_title,
            colorbar_label="unitless",
            fill_values=milestone_p_i,
            plot_basename="p_i",
            plot_dir=image_directory, 
            return_plot=return_plot)
        return ret
        
    def plot_free_energy_milestones_1d(
            self, 
            image_directory: str | None = None,
            return_plot=True
            ) -> tuple | None:
        """
        Plot the free energy profile by milestone.
        """
        if self.main_data_window.p_i is None:
            return
        
        milestone_indices = np.zeros(
            self.main_data_window.p_i.shape[0], dtype=np.int8)
        milestone_values = np.zeros(
            self.main_data_window.p_i.shape[0], dtype=np.float64)
        milestone_value_dict = analyzers_base.make_milestone_value_dict(self.model)
        for i in range(self.main_data_window.p_i.shape[0]):
            milestone_indices[i] = i
            milestone_values[i] = milestone_value_dict[i]
        
        # save free energy milestone profile
        free_energy_milestones_fig, ax = plt.subplots()
        plt.errorbar(
            np.round(milestone_values, 3), 
            self.main_data_window.free_energy_milestones, 
            yerr=self.main_data_window.free_energy_milestones_err, 
            ecolor="k", capsize=2)
        plt.title("\u0394G by milestone")
        plt.xticks(np.round(milestone_values, 3), np.round(milestone_values, 3),
                   rotation=90)
        plt.ylabel("\u0394G(milestone) (kcal/mol)")
        plt.xlabel("milestone value")
        plt.tight_layout()
        if return_plot:
            return free_energy_milestones_fig, ax
        else:
            free_energy_milestones_fig.savefig(os.path.join(
                image_directory, "free_energy_profile_milestones.png"))
            plt.close(free_energy_milestones_fig)
            return
        
    def plot_free_energy_milestones_2d(
            self,
            cv_indices: tuple[int, int] = (0, 1),
            image_directory: str | None = None,
            return_plot=True
            ) -> tuple | None:
        """
        Plot the free energy profile by milestone in 2D.
        """
        anchor_values = analyzers_base.find_anchor_points_2d(self.model)
        boundaries = analyzers_base.make_boundaries(anchor_values)
        cv_0_name = self.model.collective_variables[cv_indices[0]].name
        cv_0_unit = self.model.collective_variables[cv_indices[0]].unit
        x_coordinate_title = f"{cv_0_name} ({cv_0_unit})"
        cv_1_name = self.model.collective_variables[cv_indices[1]].name
        cv_1_unit = self.model.collective_variables[cv_indices[1]].unit
        y_coordinate_title = f"{cv_1_name} ({cv_1_unit})"
        if self.main_data_window.p_i is None:
            return
        milestone_free_energies = {}
        for i, fill_value in enumerate(
                self.main_data_window.free_energy_milestones):
            if not np.isfinite(fill_value):
                milestone_free_energies[i] = np.max(
                    self.main_data_window.free_energy_milestones[np.isfinite(
                        self.main_data_window.free_energy_milestones)])
            else:
                milestone_free_energies[i] = fill_value
        ret = analyzers_base.plot_milestones_2d(
            model=self.model, 
            boundaries=boundaries, 
            title="\u0394G by milestone",
            x_coordinate_title=x_coordinate_title,
            y_coordinate_title=y_coordinate_title,
            colorbar_label="kcal/mol",
            fill_values=milestone_free_energies,
            plot_basename="free_energy_profile_milestones",
            plot_dir=image_directory, 
            return_plot=return_plot)
        return ret
        
    def plot_free_energy_anchors_1d(
            self, 
            image_directory: str | None = None,
            return_plot=True
            ) -> tuple | None:
        """
        Plot the free energy profile by anchor.
        """
        anchor_indices = np.zeros(len(self.model.anchors), dtype=np.int8)
        anchor_values = np.zeros(len(self.model.anchors), dtype=np.float64)
        for i, anchor in enumerate(self.model.anchors):
            anchor_indices[i] = anchor.index
            anchor_values[i] = list(anchor.variables.values())[0]

        if self.main_data_window.free_energy_anchors is None:
            return
        # save free energy anchor profile
        free_energy_anchors_fig, ax = plt.subplots()
        plt.errorbar(
            anchor_values, 
            self.main_data_window.free_energy_anchors, 
            yerr=self.main_data_window.free_energy_anchors_err, 
            ecolor="k", capsize=2)
        plt.title("\u0394G by anchor")
        plt.xticks(anchor_values, anchor_values, rotation=90)
        plt.ylabel("\u0394G(anchor) (kcal/mol)")
        plt.xlabel("anchor value")
        plt.tight_layout()
        if return_plot:
            return free_energy_anchors_fig, ax
        else:
            free_energy_anchors_fig.savefig(os.path.join(
                image_directory, "free_energy_profile_anchor.png"))
            plt.close(free_energy_anchors_fig)
            return
        
    def plot_free_energy_anchors_2d(
            self,
            cv_indices: tuple[int, int] = (0, 1),
            image_directory: str | None = None,
            return_plot=True
            ) -> tuple | None:
        """
        Plot the free energy values for each anchor in 2D.
        """
        anchor_values = analyzers_base.find_anchor_points_2d(self.model)
        boundaries = analyzers_base.make_boundaries(anchor_values)
        cv_0_name = self.model.collective_variables[cv_indices[0]].name
        cv_0_unit = self.model.collective_variables[cv_indices[0]].unit
        x_coordinate_title = f"{cv_0_name} ({cv_0_unit})"
        cv_1_name = self.model.collective_variables[cv_indices[1]].name
        cv_1_unit = self.model.collective_variables[cv_indices[1]].unit
        y_coordinate_title = f"{cv_1_name} ({cv_1_unit})"
        if self.main_data_window.free_energy_anchors is None:
            return
        anchor_free_energies = {}
        for alpha, fill_value in enumerate(
                self.main_data_window.free_energy_anchors):
            if not np.isfinite(fill_value):
                anchor_free_energies[alpha] = np.max(
                    self.main_data_window.free_energy_anchors[np.isfinite(
                        self.main_data_window.free_energy_anchors)])
            else:
                anchor_free_energies[alpha] = fill_value
        ret = analyzers_base.plot_anchors_2d(
            model=self.model, 
            anchor_values=anchor_values,
            boundaries=boundaries, 
            title="\u0394G by anchor",
            x_coordinate_title=x_coordinate_title,
            y_coordinate_title=y_coordinate_title,
            colorbar_label="kcal/mol",
            fill_values=anchor_free_energies,
            plot_basename="free_energy_profile_anchors",
            plot_dir=image_directory, 
            return_plot=return_plot)
        return ret
    
    def save_plots(
            self, 
            image_directory: str
            ) -> None:
        """
        Save a potentially useful series of plots of some quantities
        obtained during the analysis.
        
        TODO: interact with model, because the way these plots are saved
        depends on the structure of the CVs.
        """
        # TODO: modify these plot functions to be individual for calling
        # from a jupyter notebook. But also pass an arg to save to file
        # and then close to save memory.
        print("All plots being saved to:", analyzers_base.DEFAULT_IMAGE_DIR)
        if len(self.model.anchors[0].variables) == 1:
            self.plot_pi_alpha_1d(
                image_directory, return_plot=False)
            self.plot_p_i_1d(
                image_directory, return_plot=False)
            self.plot_free_energy_milestones_1d(
                image_directory, return_plot=False)
            self.plot_free_energy_anchors_1d(
                image_directory, return_plot=False)

        elif len(self.model.anchors[0].variables) > 1:
            self.plot_pi_alpha_2d(
                image_directory=image_directory, return_plot=False)
            self.plot_p_i_2d(
                image_directory=image_directory, return_plot=False)
            self.plot_free_energy_milestones_2d(
                image_directory=image_directory, return_plot=False)
            self.plot_free_energy_anchors_2d(
                image_directory=image_directory, return_plot=False)

        mmvt_convergence.write_convergence_plots(
            image_directory, self.data_window_list)
        return
            
def analyze_mmvt(
        model: structures.Seekr_model,
        stage_index: int,
        skip_checks: bool = False,
        min_time_ps: float = 0.0,
        max_time_ps: float | None = None,
        error_analysis_method: str = "bootstrap",
        only_extract_data: bool = False,
        num_error_samples: int | None = None
        ) -> Mmvt_analyzer:
    """
    Analyze the results of a SEEKR MMVT calculation.
    
    Parameters
    ----------
    model : structures.Seekr_model
        The SEEKR model to analyze.
    
    Returns
    -------
    Mmvt_analyzer
        An instance of Mmvt_analyzer containing the analysis results.
    """
    analysis = Mmvt_analyzer(model=model, stage_index=stage_index)
    time_between_statistics = analysis.extract_data(
        min_time_ps=min_time_ps, max_time_ps=max_time_ps)
    if not skip_checks:
        mmvt_data_window.check_extraction(
            analysis.model, analysis.anchor_stats_alpha_list)
        if analysis.bulk_stats is not None:
            analysis.bulk_stats.check()
    if only_extract_data:
        return analysis
    analysis.fill_out_and_process_data_samples(time_between_statistics)
    analysis.error_analysis(error_analysis_method, num_error_samples,
                            random_seed=model.get_random_seed())
    return analysis