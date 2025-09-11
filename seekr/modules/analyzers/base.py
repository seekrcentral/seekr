"""
modules/analyzers/base.py

Base function and constants that might be common for analyzers.
"""

import copy
import typing

import numpy as np
from scipy import linalg as la
import PyGT

import seekr.modules.structures as structures

LOW_FLUX = 1e-19
FAST_FLUX = 1e6
LOW_N_IJ = 1e-19
HIGH_N_IJ = 1e3
DEFAULT_R_I = 1.0
DEFAULT_Q_IJ = 1.0
MATRIX_EXPONENTIAL = 9999999
GAS_CONSTANT = 0.0019872 # in kcal/mol*K
DEFAULT_MAX_CONVERGENCE_WINDOWS = 20
NUM_BOOTSTRAP_SAMPLES = 100
NUM_RESAMPLES = 10 #100 #TODO: set back to 100? Very slow.
NUM_MONTE_CARLO_SAMPLES = 100
DEFAULT_IMAGE_DIR = "images_and_plots/"

# MMVT Constants
NS_PER_PS = 1e-3
N_ALPHA_BETA_DIR = "N_alpha_beta/"
T_ALPHA_DIR = "T_alpha/"
K_ALPHA_BETA_DIR = "k_alpha_beta/"
PI_ALPHA_DIR = "pi_alpha/"
N_IJ_ALPHA_DIR = "N_ij_alpha/"
R_I_ALPHA_DIR = "R_i_alpha/"
N_IJ_DIR = "N_ij/"
R_I_DIR = "R_i/"

# The threshold beneath which to skip plotting the convergence
MIN_PLOT_NORM = 1e-18

class MissingStatisticsError(Exception):
    """
    Catch an error in analysis stage where statistics are missing for a 
    particular anchor or systemwide situation.
    """
    pass

def check_graph_connected(
        graph: dict[int, list[int]], 
        state_index1: int, 
        state_index2: int
        ) -> bool:
    """
    A graph is made of anchor indices that have recorded bounces against
    neighboring anchors. If pathways can be found, then sufficient statistics
    likely exist to compute kinetics from the resulting bounces.
    
    Parameters:
    -----------
    graph : dict
        A dictionary whose keys are node incides and whose values are
        lists of connected node indices
    state_index1 : int
        The index of the "source" node.
    state_index2 : int
        The index o the "sink" node.
        
    Returns:
    --------
    connected : bool
        If a valid path can be found between state_index1 and 
        state_index2, then return True. Otherwise return False.
    """
    max_iter = len(graph) + 5
    if state_index1 == state_index2:
        return True
    explored_nodes = [state_index1]
    edge_nodes: list[int] = copy.deepcopy(graph[state_index1])
    counter = 0
    while len(edge_nodes) > 0:
        edge_node = edge_nodes.pop()
        if state_index2 == edge_node:
            return True
        explored_nodes.append(edge_node)
        for possible_edge_node in graph[edge_node]:
            if possible_edge_node not in edge_nodes \
                    and possible_edge_node not in explored_nodes:
                edge_nodes.append(possible_edge_node)
        
        counter += 1
        assert counter < max_iter, "Maximum iterations exceeded."
        
    return False

def rate_matrix_to_transition_matrix(
        Q: np.ndarray
        ) -> np.ndarray:
    """
    Given a rate matrix Q, compute probability transition matrix K.
    """
    K = np.zeros(Q.shape, dtype=np.longdouble)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            if i == j:
                K[i,j] = 0.0
            else:
                K[i,j] = -Q[i,j] / Q[i,i]
    
    return K

def compute_rate_matrix(
        num_states: int,
        N_ij: typing.DefaultDict[tuple[int, int], int | float],
        R_i: typing.DefaultDict[int, float]
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Q and K from N_ij and R_i.
    """
    Q = np.zeros((num_states, num_states), dtype=np.longdouble)
    for i in range(num_states):
        for j in range(num_states):
            if R_i[i] == 0.0:
                Q[i,j] = 0.0
            else:
                Q[i,j] = N_ij[i,j] / R_i[i]
            assert Q[i,j] >= 0.0, "Q[i,j]: {}".format(Q[i,j])

    for i in range(num_states):
        row_sum = np.sum(Q[i], dtype=np.longdouble)
        if row_sum == 0:
            new_row_sum = np.longdouble(0.0)
            for j in range(num_states):
                Q[i,j] = Q[j,i]
                new_row_sum += Q[j,i]
            
            Q[i,i] = -new_row_sum
        else:
            Q[i,i] = -row_sum
            
    K = rate_matrix_to_transition_matrix(Q)
    return Q, K
        
def calculate_thermodynamics(
        num_states: int,
        Q: np.ndarray,
        K: np.ndarray,
        N_ij: typing.DefaultDict[tuple[int, int], int | float],
        R_i: typing.DefaultDict[int, float],
        temperature: float,
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Use this data sample's statistics to construct the 
    thermodynamic quantities.
    """
    eigvals, eigvecs = la.eig(K.T)
    closest_eigval = -1e9
    closest_eigvec = None
    for i, eigval, in enumerate(eigvals):
        if (eigval - 1.0)**2 < (closest_eigval - 1.0)**2:
            closest_eigval = eigval
            closest_eigvec = eigvecs[:,i]
    
    K = abs(K)
    q_i = np.real(closest_eigvec / np.sum(closest_eigvec))
    q_i = q_i @ np.linalg.matrix_power(K, MATRIX_EXPONENTIAL)
    q_i = q_i / np.sum(q_i)
    incubation_times = np.zeros(num_states)
    p_i = np.zeros(num_states)
    for i in range(num_states):
        N_i = 0
        for j in range(num_states):
            N_i += N_ij[i,j]
        
        if N_i == 0:
            incubation_times[i] = -1.0/Q[i,i]
        else:
            incubation_times[i] = R_i[i] / N_i

    for i, q_i_val in enumerate(q_i):
        p_i[i] = abs(q_i_val * incubation_times[i])
    
    sum_p_i = np.sum(p_i)
    for i in range(num_states):
        p_i[i] /= sum_p_i
    
    free_energy_profile = np.zeros(p_i.shape)
    highest_p_i = max(p_i)
    for i, p_i_val in enumerate(p_i):
        free_energy = -GAS_CONSTANT*temperature*np.log(
            p_i_val / highest_p_i)
        free_energy_profile[i] = free_energy
    
    return p_i, free_energy_profile

def calculate_kinetics(
        num_states: int,
        Q: np.ndarray,
        K: np.ndarray,
        end_states: typing.DefaultDict[str, list[int]]
):
    """
    Once the rate matrix Q is computed, determine the timescales 
    and probabilities of transfers between different states. Fill
    out all kinetics quantities.
    """
    assert len(end_states) > 0, "No end (bound or otherwise) states "\
        "defined in this model. Kinetics calculations will not work."
    if np.any(Q.sum(axis=1) > 1.E-10):
        problem_milestone = np.argmin(Q.T.sum(axis=1))
        error_msg = """The rate matrix Q has a numerically overflowed row 
        sum placed into the main diagonal of milestone {}. This can be
        caused by large discrepancies of rates between adjacent milestones.
        You can try additional sampling of whichever anchor(s) have this 
        milestone. If possible, you may consider adding more milestones in
        this area, or omitting calculations in this portion of the CV(s). 
        If this problem persists, please contact the developers.""".format(
            problem_milestone)
        raise Exception(error_msg)
    
    milestone_src_to_milestone_dest_MFPTs = {}
    committor_function_by_milestone = {}
    end_state_to_end_state_probabilities_by_milestone = {}
    for dest_state_name, dest_milestone_list in end_states.items():
        for src_state_name, src_milestone_list in end_states.items():
            if src_state_name == dest_state_name:
                continue
            
            # Rate matrix for MFPT
            B, tau = PyGT.tools.load_CTMC(Q.T)
            for i, dest_milestone_index in enumerate(dest_milestone_list):
                # Prob matrices for state-to-state probs and committors
                # This matrix is to see the probability of a system, starting on
                #  src_milestone_index will end up at each of any of the 
                #  dest_milestone_index milestones.
                K_hat_many_dests = copy.deepcopy(K)
                # This matrix is to see the committor probability from any
                # milestones to this particular milestone
                K_hat_one_dest = copy.deepcopy(K)
                K_hat_one_dest[dest_milestone_index, :] = 0.0
                K_hat_one_dest[dest_milestone_index, dest_milestone_index] = 1.0
                source_vec = np.zeros((num_states, 1))
                source_vec[dest_milestone_index] = 1.0
                for j, src_milestone_index in enumerate(src_milestone_list):
                    if dest_milestone_index == src_milestone_index:
                        continue
                    mfpt_ij, mfpt_ji = PyGT.mfpt.compute_MFPT(
                        src_milestone_index, dest_milestone_index, B, tau)
                    mfpt = mfpt_ji
                    milestone_src_to_milestone_dest_MFPTs[
                        (src_milestone_index, dest_milestone_index)] = mfpt
                    K_hat_many_dests[src_milestone_index, :] = 0.0
                    K_hat_many_dests[src_milestone_index, src_milestone_index] = 1.0
                    # Setting all states to be sink states
                    K_hat_one_dest[src_milestone_index, :] = 0.0

                K_hat_inf_one_dest = np.linalg.matrix_power(
                    K_hat_one_dest, MATRIX_EXPONENTIAL)
                K_hat_inf_many_dests = np.linalg.matrix_power(
                    K_hat_many_dests, MATRIX_EXPONENTIAL)
                state_to_state_probs = np.dot(K_hat_inf_many_dests.T, source_vec)
                
                for j, src_milestone_index in enumerate(src_milestone_list):
                    if dest_milestone_index == src_milestone_index:
                        continue
                    end_state_to_end_state_probabilities_by_milestone[
                        (src_milestone_index, dest_milestone_index)] \
                            = float(state_to_state_probs[src_milestone_index])
                    
                committor_function_by_milestone[dest_milestone_index] = \
                    K_hat_inf_one_dest[:,dest_milestone_index]

    return milestone_src_to_milestone_dest_MFPTs, committor_function_by_milestone,\
        end_state_to_end_state_probabilities_by_milestone

def pretty_string_value_error(
        value: float | None, 
        error: float | None, 
        error_digits: int = 2, 
        use_unicode: bool = True
        ) -> str:
    """
    Returns a value/error combination of numbers in a scientifically 
    'pretty' format.
    
    Scientific quantities often come as a *value* (the actual 
    quantity) and the *error* (the uncertainty in the value).
        
    Given two floats, value and error, return the two in a 
    'pretty' formatted string: where the value and error are truncated
    at the correct precision.
    
    Parameters
    ----------
    value : float
        The quantity in question
        
    error : float
        The uncertainty of the quantity
        
    error_digits : int, default 2
        How many significant figures the error has. Scientific 
        convention holds that errors have 1 or (at most) 2 significant
        figures. Two digits is chosen here by default.
        
    Returns
    -------
    new_string : str
        A new list of strings sorted numerically
    
    Examples
    --------
    
        >>> pretty_string_value_error(1.23456789e8, 4.5678e5, 
                                      error_digits=2)
        "1.2346 +/- 0.0046 * 10^+08"
    
        >>> pretty_string_value_error(5.6e-2, 2.0e-3, error_digits=1)
        "5.6 +/- 0.2 * 10^-02"
    """
    if value is None:
        return "None"
    if error is None or not np.isfinite(error):
        if use_unicode:
            new_string = "{:.6E} \u00B1 UNKNOWN ERROR MARGIN".format(value)
        else:
            new_string = "{:.6E} +/- UNKNOWN ERROR MARGIN".format(value)
    else:
        if not np.isfinite(value):
            return str(value)
        assert "e" in "{:e}".format(value), "Cannot convert into scientific "\
            "notation: {1}".format(value)
        value_mantissa_str, value_exponent_str = \
            "{:e}".format(value).strip().split('e')
        value_mantissa = float(value_mantissa_str)
        value_exponent = int(value_exponent_str)
        error_mantissa_str, error_exponent_str = \
            "{:e}".format(error).strip().split('e')
        error_mantissa = float(error_mantissa_str)
        error_exponent = int(error_exponent_str)
        padding = value_exponent - error_exponent + error_digits - 1
        if padding < 1: padding = 1
        exp_diff = error_exponent - value_exponent
        string_for_formatting = "{:.%df}" % padding
        new_value_mantissa = string_for_formatting.format(value_mantissa)
        new_error_mantissa = string_for_formatting.format(
            error_mantissa*10**exp_diff)
        if use_unicode:
            new_string = "(%s \u00B1 %s) * 10^%s" % (
                new_value_mantissa, new_error_mantissa, value_exponent_str)
        else:
            new_string = "(%s +/- %s) * 10^%s" % (
                new_value_mantissa, new_error_mantissa, value_exponent_str)
    return new_string

def quadriture(
        err1: float | None, 
        err2: float | None
    ) -> float | None:
    """Add two errors in quadriture."""
    if err1 is None or err2 is None:
        return None
    return float(np.sqrt(err1**2 + err2**2))

def make_milestone_value_dict(
        model: structures.Seekr_model
    ) -> dict[int, float]:
        """
        Make a dictionary with milestone indices as keys, and
        milestone values as dictionary values.
        """
        milestone_value_dict = {}
        for i, anchor in enumerate(model.anchors):
            for milestone in anchor.milestones:
                value = milestone.variables["avg_value"]
                if milestone.index not in milestone_value_dict:
                    milestone_value_dict[milestone.index] = value
                    
        return milestone_value_dict

def plot_scalar_conv(
        conv_values: np.ndarray, 
        conv_times: np.ndarray, 
        label: str, 
        title: str, 
        conv_errors: dict | None = None,
        y_axis_logarithmic: bool = True
        ) -> tuple["plt.Figure", "plt.Axes"]:
    """
    Plot convergence of off/on rate or other scalar values as a 
    function of simulation time.

    Parameters
    ----------
    conv_values : list
        list of calculated scalar values for each convergence interval

    conv_times : list
        list of convergence interval step numbers for which samples 
        are taken.
        
    label : str
        The label to give this plot.
    
    title : str
        The title of this plot.
    
    timestep_in_ns : float
        The length of the timestep in units of nanoseconds.
        
    y_axis_logarithmic : bool, default True
        Whether the y-axis will be plotted on a logarithmic scale.

    Returns
    -------
    fig : matplotlib figure
        matplotlib figure plotting N convergence for each milestone
    ax : object
        matplotlib Axes object
    """
    import matplotlib.pyplot as plt
    #if not np.any(np.isfinite(conv_values)) or np.all(conv_values == 0):
    #    return None, None
    for i, conv_value in enumerate(conv_values):
        if not np.isfinite(conv_value) or conv_value == 0:
            conv_values[i] = np.NAN
    
    if not np.any(np.isfinite(conv_values)):
        #print("skipping key:", key, "because values aren't finite")
        return None, None
    
    fig, ax = plt.subplots()
    ax.plot(conv_times, conv_values, 
            linestyle="-", marker="o", markersize=1)
    if conv_errors is not None:
        ax.fill_between(conv_times, conv_values - conv_errors, 
                            conv_values + conv_errors, alpha=0.2)
    x_lim = ax.get_xlim()
    upper_limit = x_lim[1]
    ax.set_xlim((0.0, upper_limit))
    plt.ylabel("$"+label+"$")
    # TODO: change this to plotting over time
    plt.xlabel("time per anchor (ns)")
    plt.title(title)
    if y_axis_logarithmic:
        plt.yscale("log", nonpositive="mask")
    plt.tight_layout()
    return fig, ax

def plot_2d_conv(
        conv_dict: dict, 
        conv_times: np.ndarray, 
        title: str, 
        xlabel: str,
        zlabel: str,
        conv_errors: dict | None = None,
        y_axis_logarithmic: bool = True,
        num_convergence_steps: int | None = 8
        ) -> tuple["plt.Figure", "plt.Axes"]:
    """
    
    """
    import matplotlib
    import matplotlib.pyplot as plt
    
    x_axis = sorted(conv_dict.keys())
    if num_convergence_steps is None:
        indices = np.arange(len(conv_times))
        y_axis = conv_times
    else:
        indices = np.linspace(0, len(conv_times) - 1, num_convergence_steps, dtype=int)
        y_axis = conv_times[indices]

    y_arr = np.asarray(y_axis, dtype=float)
    cmap = matplotlib.cm.berlin.reversed()   # or 'cool', 'winter', 'copper', etc.
    norm = matplotlib.colors.Normalize(vmin=np.nanmin(y_arr), vmax=np.nanmax(y_arr))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for j, y_val in zip(indices, y_axis):
        x_s = []
        y_s = []
        z_s = []
        z_err_lowers = []
        z_err_uppers = []
        for i, x_val in enumerate(x_axis):
            if y_axis_logarithmic:
                if conv_dict[i][j] > 0:
                    z_s.append(np.log10(conv_dict[i][j]))
                    if conv_errors is not None:
                        #z_errs.append(np.log10(conv_errors[i][j]))
                        lower = conv_dict[i][j] - conv_errors[i][j]
                        if lower <= 0.0:
                            lower = conv_dict[i][j]
                        upper = conv_dict[i][j] + conv_errors[i][j]
                        if upper <= 0.0:
                            upper = conv_dict[i][j]
                        z_err_lowers.append(np.log10(lower))
                        z_err_uppers.append(np.log10(upper))
                    y_s.append(y_val)
                    x_s.append(x_val)
            else:
                z_s.append(conv_dict[i][j])
                if conv_errors is not None:
                    #z_errs.append(conv_errors[i][j])
                    z_err_lowers.append(conv_dict[i][j] - conv_errors[i][j])
                    z_err_uppers.append(conv_dict[i][j] + conv_errors[i][j])
                y_s.append(y_val)
                x_s.append(x_val)

        x_s = np.array(x_s)
        y_s = np.array(y_s)
        z_s = np.array(z_s)
        z_err_lowers = np.array(z_err_lowers)
        z_err_uppers = np.array(z_err_uppers)

        if len(x_s) > 0 and np.all(np.isfinite(z_s)) \
                and np.all(np.isfinite(x_s)) and np.all(np.isfinite(y_s)):
            color = cmap(norm(float(y_val)))
            ax.plot(x_s, y_s, z_s, zdir="z", color=color, alpha=0.75)
            if conv_errors is not None:
                for xi, yi, e_lower, e_upper in zip(x_s, y_s, z_err_lowers, z_err_uppers):
                    ax.plot([xi, xi], [yi, yi], [e_lower, e_upper], 
                            color=color, alpha=0.75)
            if y_axis_logarithmic:
                zticks = ax.get_zticks()
                ztick_labels = [f"{10**tick:.3f}" for tick in zticks]
                ax.set_zticks(zticks) # suppresses warning
                ax.set_zticklabels(ztick_labels)

    upper_limit = ax.get_ylim()[1]
    ax.set_ylim((upper_limit, 0))
    ax.set_xlabel(xlabel, labelpad=12)
    ax.set_ylabel("time (ns)", labelpad=12)
    ax.set_zlabel(zlabel + (" (log₁₀ scale)" if y_axis_logarithmic else ""), labelpad=14)
    plt.title(title)
    return fig, ax

def plot_dict_conv(
        conv_dict: dict, 
        conv_times: np.ndarray, 
        label_base: str,
        unit: str, 
        conv_errors: dict | None = None,
        skip_null: bool = True,
        y_axis_logarithmic: bool = True,
        title_suffix: str = "", 
        name_base: str | None = None,
        draw_double: bool = True
        ) -> tuple["plt.Figure", "plt.Axes"]:
    """
    Plot convergence of N_ij or R_i or other dictionary-based value 
    as a function of simulation time.

    Parameters
    ----------
    conv_dict : dict
        dict of lists of calculated off rates for each convergence 
        interval.
        
    conv_interval : list
        list of convergence interval step numbers for which samples 
        are taken.
        
    label_base : str
        The base of the label to give this plot, though the dictionary
        keys will be appended to the label.
    
    timestep_in_ns : float
        The length of the timestep in units of nanoseconds
    
    skip_null : bool, Default True
        If true, than empty convergence lists will be omitted from
        any plots.
    
    y_axis_logarithmic : bool, True
        Whether the y-axis will be plotted on a logarithmic scale.
    
    Returns
    -------
    fig_list : list
        A list of matplotlib figures plotting convergence for each 
        milestone.
        
    ax_list : list
        A list of matplotlib Axes objects.
        
    title_list : list
        A list of the plots' titles.
    
    name_list : list
        A list of the plots' file names.
    """
    import matplotlib.pyplot as plt
    fig_list = []
    ax_list = []
    title_list = []
    name_list = []
    if name_base is None:
        name_base = label_base
        
    for key in conv_dict:
        conv_values = conv_dict[key]
        conv_error = None
        if conv_errors is not None:
            if key in conv_errors:
                conv_error = conv_errors[key]

        if not np.all(np.isfinite(conv_values)):
            continue
        if skip_null:
            if np.linalg.norm(conv_values) < MIN_PLOT_NORM:
                continue
        if unit == "":
            unitstr = ""
        else:
            unitstr = "(" + unit +")"
        if isinstance(key, tuple):
            # If right arrow is preferred
            #label = "$" + label_base + "_{" + "\\rightarrow".join(
            #    map(str, key)) + "}(" + unit +")$"
            #title = "$" + label_base + "_{" + "\\rightarrow".join(
            #    map(str, key)) + "}$" + title_suffix
            
            label = "$" + label_base + "_{" + ",".join(
                map(str, key)) + "}" + unitstr + "$"
            title = "$" + label_base + "_{" + ",".join(
                map(str, key)) + "}$" + title_suffix
            name = name_base + "_" + "_".join(map(str, key))
        elif isinstance(key, int):
            label = "$" + label_base + "_{" + str(key) + "}" + unitstr + "$"
            title = "$" + label_base + "_{" + str(key) + "}$" + title_suffix
            name = name_base + "_" + str(key) + ""
        else:
            raise Exception("key type not implemented: {}".format(type(key)))
        
        fig, ax1 = plt.subplots()
        x_data = conv_times
        ax1.plot(conv_times, conv_values, 
                linestyle='-', marker="o", color="r", markersize = 1)
        if conv_error is not None:
            ax1.fill_between(conv_times, conv_values - conv_error, 
                             conv_values + conv_error, color="r", alpha=0.2)
        ax1.set_ylabel(label, color="r")
        ax1.set_xlabel("time per anchor (ns)")
        if y_axis_logarithmic:
            plt.yscale("log", nonpositive="mask")
        ax_list.append(ax1)
        if draw_double:
            ax2 = ax1.twinx()
            ax2.set_ylabel(label + "/time(ps)", color="b")
            data_per_time = np.zeros(conv_times.shape)
            data_per_time_err_upper = np.zeros(conv_times.shape)
            data_per_time_err_lower = np.zeros(conv_times.shape)
            for i in range(len(conv_times)):
                if x_data[i] == 0:
                    divisor = 1
                else:
                    divisor = x_data[i]
                data_per_time[i] = conv_values[i] / divisor
                if conv_error is not None:
                    data_per_time_err_lower[i] = (conv_values[i] - conv_error[i]) / divisor
                    data_per_time_err_upper[i] = (conv_values[i] + conv_error[i]) / divisor
            ax2.plot(x_data, data_per_time, 
                linestyle='-', marker="o", color="b", markersize = 1)
            if conv_error is not None:
                ax2.fill_between(x_data, data_per_time_err_lower, 
                                 data_per_time_err_upper, color="b", alpha=0.2)
            if y_axis_logarithmic:
                plt.yscale("log", nonpositive="mask")
            ax_list.append(ax2)

        x_lim = ax1.get_xlim()
        upper_limit = x_lim[1]
        ax1.set_xlim((0.0, upper_limit))

        plt.title(title)
        fig_list.append(fig)
        title_list.append(title)
        name_list.append(name)
        plt.tight_layout()
        
    return fig_list, ax_list, title_list, name_list
