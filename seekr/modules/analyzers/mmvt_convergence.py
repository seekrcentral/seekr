from __future__ import annotations
"""
modules/analyzers/mmvt_convergence.py

This module contains the Mmvt_convergence class, which analyzes the convergence
of MMVT simulations by examining the stability of computed rates and other
properties over time.
"""

import os
import functools
from collections import defaultdict

from attrs import define
import numpy as np

import seekr.modules.analyzers.base as analyzers_base
import seekr.modules.analyzers.mmvt_data_window as mmvt_data_window

@define
class Mmvt_converger:
    num_windows: int | None = None
    data_window_times: np.ndarray | None = None
    k_off_conv: np.ndarray | None = None
    k_off_conv_err: np.ndarray | None = None
    k_on_conv: np.ndarray | None = None
    k_on_conv_err: np.ndarray | None = None
    N_alpha_beta_conv: defaultdict | None = None
    T_alpha_conv: defaultdict | None = None
    k_alpha_beta_conv: defaultdict | None = None
    k_alpha_beta_conv_err: defaultdict | None = None
    pi_alpha_conv: defaultdict | None = None
    pi_alpha_conv_err: defaultdict | None = None
    N_ij_alpha_conv: defaultdict | None = None
    R_i_alpha_conv: defaultdict | None = None
    N_ij_conv: defaultdict | None = None
    N_ij_conv_err: defaultdict | None = None
    R_i_conv: defaultdict | None = None
    R_i_conv_err: defaultdict | None = None
    p_i_conv: defaultdict | None = None
    p_i_conv_err: defaultdict | None = None
    free_energy_milestones_conv: defaultdict | None = None
    free_energy_milestones_conv_err: defaultdict | None = None
    free_energy_anchors_conv: defaultdict | None = None
    free_energy_anchors_conv_err: defaultdict | None = None
    MFPTs_anchor_conv: defaultdict | None = None
    MFPTs_anchor_conv_err: defaultdict | None = None
    MFPTs_milestone_conv: defaultdict | None = None
    MFPTs_milestone_conv_err: defaultdict | None = None

    def compute_milestone_convergence(
            self,
            data_window_list: list[mmvt_data_window.Mmvt_data_window],
            ) -> None:
        """
        
        """
        self.num_windows = len(data_window_list)
        num_windows = self.num_windows
        self.data_window_times = np.zeros(num_windows)
        self.k_off_conv = np.zeros(num_windows)
        self.k_off_conv_err = np.zeros(num_windows)
        self.k_on_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.k_on_conv_err = defaultdict(functools.partial(np.zeros, num_windows))
        # partial allows us to create a defaultdict with values that are
        # empty arrays of a certain size DEFAULT_NUM_POINTS
        self.N_alpha_beta_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.T_alpha_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.k_alpha_beta_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.k_alpha_beta_conv_err = defaultdict(functools.partial(np.zeros, num_windows))
        self.pi_alpha_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.pi_alpha_conv_err = defaultdict(functools.partial(np.zeros, num_windows))
        self.N_ij_alpha_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.R_i_alpha_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.N_ij_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.N_ij_conv_err = defaultdict(functools.partial(np.zeros, num_windows))
        self.R_i_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.R_i_conv_err = defaultdict(functools.partial(np.zeros, num_windows))
        self.p_i_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.p_i_conv_err = defaultdict(functools.partial(np.zeros, num_windows))
        self.free_energy_milestones_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.free_energy_milestones_conv_err = defaultdict(functools.partial(np.zeros, num_windows))
        self.free_energy_anchors_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.free_energy_anchors_conv_err = defaultdict(functools.partial(np.zeros, num_windows))
        self.MFPTs_anchor_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.MFPTs_anchor_conv_err = defaultdict(functools.partial(np.zeros, num_windows))
        self.MFPTs_milestone_conv = defaultdict(functools.partial(np.zeros, num_windows))
        self.MFPTs_milestone_conv_err = defaultdict(functools.partial(np.zeros, num_windows))

        for window_index in range(num_windows):
            data_window = data_window_list[window_index]
            self.data_window_times[window_index] = data_window.time \
                * analyzers_base.NS_PER_PS
            for k_on in data_window.k_ons:
                self.k_on_conv[k_on][window_index] \
                    = data_window.k_ons[k_on]
                self.k_on_conv_err[k_on][window_index] \
                    = data_window.k_ons_err[k_on]
            self.k_off_conv[window_index] = data_window.k_off
            self.k_off_conv_err[window_index] = data_window.k_off_err
            for N_alpha_beta_key in data_window.main_data_sample.N_alpha_beta:
                self.N_alpha_beta_conv[N_alpha_beta_key][window_index] \
                    = data_window.main_data_sample.N_alpha_beta[N_alpha_beta_key]
            for T_alpha_key in range(len(data_window.main_data_sample.T_alpha)):
                self.T_alpha_conv[T_alpha_key][window_index] \
                    = data_window.main_data_sample.T_alpha[T_alpha_key]
            for k_alpha_beta_key in data_window.main_data_sample.k_alpha_beta:
                self.k_alpha_beta_conv[k_alpha_beta_key][window_index] \
                    = data_window.main_data_sample.k_alpha_beta[k_alpha_beta_key]
                k_alpha_beta_value_list = []
                for data_sample in data_window.data_sample_list:
                    if k_alpha_beta_key in data_sample.k_alpha_beta:
                        k_alpha_beta_value_list.append(
                            data_sample.k_alpha_beta[k_alpha_beta_key])
                if len(k_alpha_beta_value_list) == 0:
                    self.k_alpha_beta_conv_err[k_alpha_beta_key][window_index] = None
                else:
                    self.k_alpha_beta_conv_err[k_alpha_beta_key][window_index] \
                        = np.std(k_alpha_beta_value_list)

            if data_window.pi_alpha is not None:
                for pi_alpha_key in range(len(data_window.pi_alpha)):
                    self.pi_alpha_conv[pi_alpha_key][window_index] \
                        = data_window.pi_alpha[pi_alpha_key, 0]
                    self.pi_alpha_conv_err[pi_alpha_key][window_index] \
                        = data_window.pi_alpha_err[pi_alpha_key, 0]
            for alpha, N_i_j_alpha_subdict in enumerate(data_window.main_data_sample.N_i_j_alpha):
                for N_ij_alpha_key in N_i_j_alpha_subdict:
                    new_key = (N_ij_alpha_key[0], N_ij_alpha_key[1], alpha)
                    self.N_ij_alpha_conv[new_key][window_index] \
                        = N_i_j_alpha_subdict[N_ij_alpha_key]
            for alpha, R_i_alpha_subdict in enumerate(data_window.main_data_sample.R_i_alpha):
                for R_i_alpha_key in R_i_alpha_subdict:
                    new_key = (R_i_alpha_key, alpha)
                    self.R_i_alpha_conv[new_key][window_index] \
                        = R_i_alpha_subdict[R_i_alpha_key]
            for N_ij_key in data_window.main_data_sample.N_ij:
                self.N_ij_conv[N_ij_key][window_index] = data_window.main_data_sample.N_ij[N_ij_key]
                N_ij_value_list = []
                for counter, data_sample in enumerate(data_window.data_sample_list):
                    if N_ij_key in data_sample.N_ij:
                        N_ij_value_list.append(
                            data_sample.N_ij[N_ij_key])
                if len(N_ij_value_list) == 0:
                    self.N_ij_conv_err[N_ij_key][window_index] = None
                else:
                    self.N_ij_conv_err[N_ij_key][window_index] \
                        = np.std(N_ij_value_list)

            for R_i_key in data_window.main_data_sample.R_i:
                self.R_i_conv[R_i_key][window_index] = data_window.main_data_sample.R_i[R_i_key]
                R_i_value_list = []
                for data_sample in data_window.data_sample_list:
                    if R_i_key in data_sample.R_i:
                        R_i_value_list.append(
                            data_sample.R_i[R_i_key])
                if len(R_i_value_list) == 0:
                    self.R_i_conv_err[R_i_key][window_index] = None
                else:
                    self.R_i_conv_err[R_i_key][window_index] \
                        = np.std(R_i_value_list)
                    
            if data_window.p_i is not None:
                for p_i_key in range(len(data_window.p_i)):
                    self.p_i_conv[p_i_key][window_index] \
                        = data_window.p_i[p_i_key]
                    self.p_i_conv_err[p_i_key][window_index] \
                        = data_window.p_i_err[p_i_key]
                    
            if data_window.free_energy_milestones is not None:
                for free_energy_key in range(len(data_window.free_energy_milestones)):
                    self.free_energy_milestones_conv[free_energy_key][window_index] \
                        = data_window.free_energy_milestones[free_energy_key]
                    self.free_energy_milestones_conv_err[free_energy_key][window_index] \
                        = data_window.free_energy_milestones_err[free_energy_key]
                    
            if data_window.free_energy_anchors is not None:
                for free_energy_key in range(len(data_window.free_energy_anchors)):
                    self.free_energy_anchors_conv[free_energy_key][window_index] \
                        = data_window.free_energy_anchors[free_energy_key]
                    self.free_energy_anchors_conv_err[free_energy_key][window_index] \
                        = data_window.free_energy_anchors_err[free_energy_key]
            
            for MFPT_key in data_window.anchor_to_anchor_MFPTs:
                self.MFPTs_anchor_conv[MFPT_key][window_index] \
                    = data_window.anchor_to_anchor_MFPTs[MFPT_key]
                if MFPT_key not in data_window.anchor_to_anchor_MFPTs_err:
                    print("Problem! MFPT_key not found in data_window.anchor_to_anchor_MFPTs_err")
                    print(f"MFPT_key: {MFPT_key}")
                    print(f"Available keys: {data_window.anchor_to_anchor_MFPTs_err.keys()}")
                self.MFPTs_anchor_conv_err[MFPT_key][window_index] \
                    = data_window.anchor_to_anchor_MFPTs_err[MFPT_key]

            for MFPT_key in data_window.milestone_to_milestone_MFPTs:
                self.MFPTs_milestone_conv[MFPT_key][window_index] \
                    = data_window.milestone_to_milestone_MFPTs[MFPT_key]
                if MFPT_key not in data_window.milestone_to_milestone_MFPTs_err:
                    print("Problem! MFPT_key not found in data_window.milestone_to_milestone_MFPTs_err")
                    print(f"MFPT_key: {MFPT_key}")
                    print(f"Available keys: {data_window.milestone_to_milestone_MFPTs_err.keys()}")
                self.MFPTs_milestone_conv_err[MFPT_key][window_index] \
                    = data_window.milestone_to_milestone_MFPTs_err[MFPT_key]
        return

    def plot_k_off(
            self,
            k_off_directory: str,
            return_plot: bool = True
            ) -> tuple | None:
        filename = os.path.join(k_off_directory, "k_off_convergence.png")
        ret = analyzers_base.plot_scalar_conv(
            self.k_off_conv, self.data_window_times, filename, 
            title="$k_{off}$ Convergence", label="k_{off} (s^{-1})", 
            conv_errors=self.k_off_conv_err, return_plot=return_plot,)
        return ret
    
    def plot_k_on(
            self,
            k_on_directory: str,
            return_plot: bool = False
            ) -> list | None:
        ret = analyzers_base.plot_dict_conv(
            self.k_on_conv, self.data_window_times,
            directory=k_on_directory,
            label_base="k_{on},{}", unit="ps", conv_errors=self.k_on_conv_err,
            y_axis_logarithmic=True, title_suffix=" Convergence", 
            name_base="k_on", draw_double=False, return_plot=return_plot)
        return ret

    def plot_N_alpha_beta(
            self,
            N_alpha_beta_directory: str,
            return_plot: bool = False
            ) -> list | None:
        ret = analyzers_base.plot_dict_conv(
            self.N_alpha_beta_conv, self.data_window_times,
            directory=N_alpha_beta_directory, label_base="N", unit="",
            y_axis_logarithmic=False, title_suffix="", draw_double=True, 
            return_plot=return_plot)
        return ret

    def plot_T_alpha(
            self,
            T_alpha_directory: str,
            return_plot: bool = False
            ) -> list | None:
        ret = analyzers_base.plot_dict_conv(
            self.T_alpha_conv, self.data_window_times, 
            directory=T_alpha_directory, label_base="T", unit="ps",
            y_axis_logarithmic=False, title_suffix="", draw_double=True,
            return_plot=return_plot)
        return ret

    def plot_k_alpha_beta(
            self, 
            k_alpha_beta_directory: str,
            return_plot: bool = False
            ) -> list | None:
        ret = analyzers_base.plot_dict_conv(
            self.k_alpha_beta_conv, self.data_window_times, 
            directory=k_alpha_beta_directory, label_base="k", unit="",
            y_axis_logarithmic=False, title_suffix="", draw_double=True,
            conv_errors=self.k_alpha_beta_conv_err, return_plot=return_plot)
        return ret

    # TODO: make 2D plot of pi_alpha convergence - but only makes sense with
    #  a single CV - two or more CVs should be plotted with the dict plotter
    #  so that needs to be implemented as a separate function, probably.
    def plot_pi_alpha_2d(
            self,
            pi_alpha_filename: str,
            return_plot: bool = False
            ) -> tuple | None:
        ret = analyzers_base.plot_2d_conv(
            self.pi_alpha_conv, self.data_window_times,
            filename=pi_alpha_filename, title="$\\pi_{\\alpha}$ Convergence",
            xlabel="Anchor index $\\alpha$", zlabel="$\\pi_{\\alpha}$",
            conv_errors=self.pi_alpha_conv_err, return_plot=return_plot)
        return ret

    def plot_N_ij_alpha(
            self,
            N_ij_alpha_directory: str,
            return_plot: bool = False
            ) -> list | None:
        ret = analyzers_base.plot_dict_conv(
            self.N_ij_alpha_conv, self.data_window_times,
            directory=N_ij_alpha_directory, label_base="N", unit="",
            y_axis_logarithmic=False, title_suffix="", draw_double=True,
            return_plot=return_plot)
        return ret
    
    def plot_R_i_alpha(
            self,
            R_i_alpha_directory: str,
            return_plot: bool = False
            ) -> list | None:
        ret = analyzers_base.plot_dict_conv(
            self.R_i_alpha_conv, self.data_window_times,
            directory=R_i_alpha_directory, label_base="R", unit="ps",
            title_suffix="", y_axis_logarithmic=False, draw_double=True,
            return_plot=return_plot)
        return ret
    
    def plot_N_ij(
            self,
            N_ij_directory: str,
            return_plot: bool = False
            ) -> list | None:
        ret = analyzers_base.plot_dict_conv(
            self.N_ij_conv, self.data_window_times, directory=N_ij_directory,
            label_base="N", unit="", y_axis_logarithmic=False,
            title_suffix="", draw_double=True, return_plot=return_plot)
        return ret
    
    def plot_R_i(
            self,
            R_i_directory: str,
            return_plot: bool = False
            ) -> list | None:
        ret = analyzers_base.plot_dict_conv(
            self.R_i_conv, self.data_window_times, directory=R_i_directory,
            label_base="R", unit="ps", y_axis_logarithmic=False,
            title_suffix="", draw_double=True, return_plot=return_plot)
        return ret

    def plot_p_i(
            self,
            p_i_filename: str,
            return_plot: bool = False
            ) -> tuple | None:
        ret = analyzers_base.plot_2d_conv(
            self.p_i_conv, self.data_window_times, filename=p_i_filename,
            title="$p_i$ Convergence", xlabel="Milestone index $i$",
            zlabel="$p_i$", conv_errors=self.p_i_conv_err, return_plot=return_plot)
        return ret

    def plot_free_energy_milestones(
            self,
            free_energy_milestones_filename: str,
            return_plot: bool = False
            ) -> tuple | None:
        ret = analyzers_base.plot_2d_conv(
            self.free_energy_milestones_conv, self.data_window_times,
            filename=free_energy_milestones_filename,
            title="Milestone Free Energy Convergence", xlabel="Milestone index $i$",
            zlabel="\u0394G (kcal/mol)", conv_errors=self.free_energy_milestones_conv_err,
            y_axis_logarithmic=False, return_plot=return_plot)
        return ret

    def plot_free_energy_anchors(
            self,
            free_energy_anchors_filename: str,
            return_plot: bool = False
            ) -> tuple | None:
        ret = analyzers_base.plot_2d_conv(
            self.free_energy_anchors_conv, self.data_window_times,
            filename=free_energy_anchors_filename,
            title="Anchor Free Energy Convergence", xlabel="Anchor index $\\alpha$",
            zlabel="\u0394G (kcal/mol)", conv_errors=self.free_energy_anchors_conv_err,
            y_axis_logarithmic=False, return_plot=return_plot)
        return ret

    def plot_MFPTs_anchors(
            self,
            MFPTs_anchors_directory: str,
            return_plot: bool = False
            ) -> list | None:
        ret = analyzers_base.plot_dict_conv(
            self.MFPTs_anchor_conv, self.data_window_times,
            directory=MFPTs_anchors_directory,
            label_base="MFPT", unit="ps", conv_errors=self.MFPTs_anchor_conv_err,
            y_axis_logarithmic=True, title_suffix=" Convergence", draw_double=False,
            return_plot=return_plot)
        return ret

    def plot_MFPTs_milestones(
            self,
            MFPTs_milestones_directory: str,
            return_plot: bool = False
            ) -> list | None:
        ret = analyzers_base.plot_dict_conv(
            self.MFPTs_milestone_conv, self.data_window_times,
            directory=MFPTs_milestones_directory,
            label_base="MFPT", unit="ps", conv_errors=self.MFPTs_milestone_conv_err,
            y_axis_logarithmic=True, title_suffix=" Convergence", draw_double=False,
            return_plot=return_plot)
        return ret

def write_convergence_plots(
        image_directory: str,
        data_window_list: list[mmvt_data_window.Mmvt_data_window],
        ) -> None:
    """
    
    """
    # TODO: assign converger to analysis object
    converger = Mmvt_converger()
    converger.compute_milestone_convergence(data_window_list)
    converger.plot_k_off(image_directory, return_plot=False)
    converger.plot_k_on(image_directory, return_plot=False)
    N_alpha_beta_directory = os.path.join(
        image_directory, analyzers_base.N_ALPHA_BETA_DIR)
    converger.plot_N_alpha_beta(N_alpha_beta_directory, return_plot=False)
    T_alpha_fig_directory = os.path.join(
        image_directory, analyzers_base.T_ALPHA_DIR)
    converger.plot_T_alpha(T_alpha_fig_directory, return_plot=False)
    k_alpha_fig_directory = os.path.join(
        image_directory, analyzers_base.K_ALPHA_BETA_DIR)
    converger.plot_k_alpha_beta(k_alpha_fig_directory, return_plot=False)
    pi_alpha_fig_filename = os.path.join(
        image_directory, "pi_alpha_convergence.png")
    converger.plot_pi_alpha_2d(pi_alpha_fig_filename, return_plot=False)
    N_ij_alpha_fig_directory = os.path.join(
        image_directory, analyzers_base.N_IJ_ALPHA_DIR)
    converger.plot_N_ij_alpha(N_ij_alpha_fig_directory, return_plot=False)
    R_i_alpha_fig_directory = os.path.join(
        image_directory, analyzers_base.R_I_ALPHA_DIR)
    converger.plot_R_i_alpha(R_i_alpha_fig_directory, return_plot=False)
    N_ij_fig_directory = os.path.join(
        image_directory, analyzers_base.N_IJ_DIR)
    converger.plot_N_ij(N_ij_fig_directory, return_plot=False)
    R_i_fig_directory = os.path.join(
        image_directory, analyzers_base.R_I_DIR)
    converger.plot_R_i(R_i_fig_directory, return_plot=False)
    p_i_filename = os.path.join(
        image_directory, "p_i_convergence.png")
    converger.plot_p_i(p_i_filename, return_plot=False)
    free_energy_milestones_filename = os.path.join(
        image_directory, "free_energy_milestones_convergence.png")
    converger.plot_free_energy_milestones(free_energy_milestones_filename, 
                                          return_plot=False)
    free_energy_anchors_filename = os.path.join(
        image_directory, "free_energy_anchors_convergence.png")
    converger.plot_free_energy_anchors(free_energy_anchors_filename, 
                                        return_plot=False)
    converger.plot_MFPTs_anchors(image_directory, return_plot=False)
    converger.plot_MFPTs_milestones(image_directory, return_plot=False)
    return